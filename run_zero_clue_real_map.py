#!/usr/bin/env python3
from __future__ import annotations

"""Role-free, answer-free real EBSD relationship discovery.

This script is intentionally outside the frozen backend.  It consumes the
sealed anonymous challenge and calls only frozen repository machinery for the
actual crystallographic inverse solution.

No material names, phase roles, OR, correspondence, expected operator class, or
literature relationship are accepted as inputs.

For every segmentation threshold and every ordered pair of anonymous phases:

    candidate parent phase A <- candidate product phase M

the script:

1. segments the full map without using any OR;
2. constructs product/product grain adjacency;
3. splits those boundaries into deterministic fit and holdout sets;
4. discovers the child/child-observable OR class on FIT ONLY;
5. evaluates exact operator residuals on the untouched HOLDOUT boundaries;
6. independently tests the discovered OR against observed cross-phase
   interfaces, which were not used in the OR fit;
7. performs a permutation null test on those cross-phase interfaces;
8. repeats the whole role test across all configured segmentation thresholds.

The script reports evidence.  It does not force a parent/product role.
"""

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from cualni_cryst.ebsd_analysis import build_neighbor_graph
from cualni_cryst.ebsd_blind_discovery import (
    BlindORSettings,
    discover_orientation_relationship_blind,
)
from cualni_cryst.ebsd_map import EBSDMap, EBSDPhase, audit_map
from cualni_cryst.ebsd_theory_bridge import build_theory_library
from cualni_cryst.ebsd_validation import PreparedBoundaryOperatorKernel
from cualni_cryst.lattice import Lattice


EXPECTED_CHALLENGE_SHA256 = (
    "c51345e32d09e5b4ca9d4a2db7ec0f3a172fc3e78f100ece16b50497c35b5b21"
)
EXPECTED_OBSERVATIONS_SHA256 = (
    "db172ea6c8bdf283bd7503fd0bd25bb7cced687aa71d0597ca8cabb81d749484"
)

FORBIDDEN_EXACT_ANSWER_KEYS = {
    "orientation_relationship",
    "expected_or",
    "true_or",
    "oracle",
    "answer",
    "parent_phase_id",
    "product_phase_id",
    "material_name",
    "chemical_identity",
}

EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "data_file",
    "orientation_convention",
    "phases",
    "grid_metadata",
    "analysis_contract",
    "settings",
}

EXPECTED_ANALYSIS_CONTRACT = {
    "chemical_identity_withheld": True,
    "correspondence_withheld": True,
    "expected_operator_families_withheld": True,
    "literature_labels_withheld": True,
    "orientation_relationship_withheld": True,
    "parent_product_roles_withheld": True,
    "phase_names_withheld": True,
    "solver_must_test_all_unordered_phase_pairs": True,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value



def validate_zero_clue_contract(challenge: Mapping[str, Any]) -> None:
    """Validate semantics without substring false positives.

    A key such as ``orientation_relationship_withheld`` is evidence that the
    answer is absent, not an answer leak.  Therefore clue detection is exact-
    key structural validation, never substring matching over serialized JSON.
    The byte-level SHA-256 check separately guarantees that the sealed challenge
    is exactly the one prepared before running the solver.
    """

    keys = set(challenge)
    if keys != EXPECTED_TOP_LEVEL_KEYS:
        missing = sorted(EXPECTED_TOP_LEVEL_KEYS - keys)
        extra = sorted(keys - EXPECTED_TOP_LEVEL_KEYS)
        raise RuntimeError(
            f"sealed challenge schema mismatch; missing={missing}, extra={extra}"
        )

    def walk(value: Any, path: str = "root") -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key)
                if key_text in FORBIDDEN_EXACT_ANSWER_KEYS:
                    raise RuntimeError(
                        f"answer-bearing key present at {path}.{key_text}"
                    )
                walk(item, f"{path}.{key_text}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(challenge)

    contract = challenge.get("analysis_contract")
    if contract != EXPECTED_ANALYSIS_CONTRACT:
        raise RuntimeError(
            "analysis_contract is not the exact zero-clue withholding contract"
        )

    if challenge.get("orientation_convention") != (
        "crystal_to_sample_cartesian_matrices"
    ):
        raise RuntimeError("unexpected orientation convention in sealed challenge")

    phases = challenge.get("phases")
    if not isinstance(phases, list) or len(phases) < 2:
        raise RuntimeError("sealed challenge must contain at least two anonymous phases")

    seen_ids: set[int] = set()
    for index, phase in enumerate(phases):
        if not isinstance(phase, Mapping):
            raise RuntimeError(f"phases[{index}] must be an object")
        allowed = {"id", "anonymous_label", "lattice", "point_group"}
        if set(phase) != allowed:
            raise RuntimeError(
                f"phases[{index}] contains non-anonymous metadata keys: "
                f"{sorted(set(phase) - allowed)}"
            )
        phase_id = int(phase["id"])
        if phase_id <= 0 or phase_id in seen_ids:
            raise RuntimeError("phase IDs must be unique positive integers")
        seen_ids.add(phase_id)
        if str(phase["anonymous_label"]) != f"phase_{phase_id}":
            raise RuntimeError(
                f"phase {phase_id} label is not strictly anonymous"
            )


def phase_from_spec(spec: Mapping[str, Any]) -> EBSDPhase:
    allowed = {"id", "anonymous_label", "lattice", "point_group"}
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise ValueError(f"phase spec has unknown keys: {unknown}")
    lattice = spec["lattice"]
    L = Lattice(
        float(lattice["a"]),
        float(lattice["b"]),
        float(lattice["c"]),
        float(lattice.get("alpha_deg", 90.0)),
        float(lattice.get("beta_deg", 90.0)),
        float(lattice.get("gamma_deg", 90.0)),
        label=str(spec["anonymous_label"]),
    )
    return EBSDPhase.from_point_group(
        int(spec["id"]),
        str(spec["anonymous_label"]),
        L,
        str(spec["point_group"]),
    )


def rotation_quaternions(matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrices, dtype=float)
    flat = values.reshape(-1, 3, 3)
    q = Rotation.from_matrix(flat).as_quat()
    q /= np.linalg.norm(q, axis=1)[:, None]
    return q.reshape(values.shape[:-2] + (4,))


def projective_quaternion_angle_deg(dots: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(dots, dtype=float), 0.0, 1.0)
    sine_half = np.sqrt(np.maximum(0.0, 1.0 - value * value))
    return np.degrees(2.0 * np.arctan2(sine_half, value))


def exact_same_phase_edge_angles(
    data: EBSDMap,
    phases: Mapping[int, EBSDPhase],
    edges: np.ndarray,
    *,
    batch_size: int = 20000,
) -> np.ndarray:
    """Exact full same-phase disorientation using group closure.

    The frozen scalar definition is

        min_{S1,S2 in G} angle(S1.T @ Delta @ S2).

    Because trace is cyclic and G is closed,

        trace(S1.T Delta S2) = trace(Delta K),  K=S2 S1.T in G,

    so a single group loop is exactly sufficient.  This is a performance
    reduction, not a weaker quotient.
    """

    edges = np.asarray(edges, dtype=int)
    result = np.full(len(edges), np.nan, dtype=float)

    for phase_id, phase in phases.items():
        mask = (
            data.indexed[edges[:, 0]]
            & data.indexed[edges[:, 1]]
            & (data.phase_id[edges[:, 0]] == phase_id)
            & (data.phase_id[edges[:, 1]] == phase_id)
        )
        positions = np.flatnonzero(mask)
        if len(positions) == 0:
            continue

        symmetry = np.asarray(
            phase.proper_symmetry_cartesian, dtype=float
        )

        for start in range(0, len(positions), batch_size):
            local_positions = positions[start : start + batch_size]
            a = edges[local_positions, 0]
            b = edges[local_positions, 1]
            raw = np.einsum(
                "nji,njk->nik",
                data.orientations[a],
                data.orientations[b],
                optimize=True,
            )
            # traces[n,s] = tr(raw[n] @ symmetry[s])
            product = np.einsum(
                "nij,sjk->nsik",
                raw,
                symmetry,
                optimize=True,
            )
            traces = np.trace(product, axis1=2, axis2=3)
            cosine = np.clip((traces - 1.0) * 0.5, -1.0, 1.0)
            angles = np.degrees(np.arccos(cosine))
            result[local_positions] = np.min(angles, axis=1)

    return result


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass(frozen=True)
class FastSegmentation:
    labels: np.ndarray
    n_grains: int
    threshold_deg: float


def segmentation_from_precomputed_edges(
    data: EBSDMap,
    edges: np.ndarray,
    edge_angles: np.ndarray,
    *,
    threshold_deg: float,
    minimum_points: int,
) -> FastSegmentation:
    uf = UnionFind(data.n_points)
    eligible = np.flatnonzero(
        np.isfinite(edge_angles) & (edge_angles <= threshold_deg)
    )
    for index in eligible:
        a, b = edges[int(index)]
        uf.union(int(a), int(b))

    components: dict[int, list[int]] = {}
    for point in np.flatnonzero(data.indexed):
        root = uf.find(int(point))
        components.setdefault(root, []).append(int(point))

    kept = [
        points
        for points in components.values()
        if len(points) >= minimum_points
    ]
    kept.sort(key=lambda points: min(points))
    labels = np.full(data.n_points, -1, dtype=np.int64)
    for grain_id, points in enumerate(kept):
        labels[np.asarray(points, dtype=int)] = grain_id
    return FastSegmentation(
        labels=labels,
        n_grains=len(kept),
        threshold_deg=float(threshold_deg),
    )


def mean_orientation_fast(
    orientations: np.ndarray,
    symmetry: Sequence[np.ndarray],
    *,
    maximum_iterations: int = 8,
) -> np.ndarray:
    """Symmetry-aligned quaternion mean.

    Only one crystal-symmetry action is required to choose the copy nearest a
    fixed mean representative.  The result is iterated until stable.
    """

    values = np.asarray(orientations, dtype=float)
    if len(values) == 1:
        return values[0].copy()

    sym = np.asarray(symmetry, dtype=float)
    mean = values[0].copy()

    for _ in range(maximum_iterations):
        candidates = np.einsum(
            "nij,sjk->nsik",
            values,
            sym,
            optimize=True,
        )
        q_candidates = rotation_quaternions(candidates)
        q_mean = rotation_quaternions(mean.reshape(1, 3, 3))[0]
        dots = np.abs(
            np.einsum("nsi,i->ns", q_candidates, q_mean, optimize=True)
        )
        best = np.argmax(dots, axis=1)
        chosen = q_candidates[np.arange(len(values)), best]

        signs = np.sign(chosen @ q_mean)
        signs[signs == 0.0] = 1.0
        chosen = chosen * signs[:, None]
        q = np.sum(chosen, axis=0)
        q /= np.linalg.norm(q)
        updated = Rotation.from_quat(q).as_matrix()

        delta_q = abs(
            float(
                rotation_quaternions(updated.reshape(1, 3, 3))[0]
                @ q_mean
            )
        )
        change = float(projective_quaternion_angle_deg(delta_q))
        mean = updated
        if change <= 1.0e-8:
            break

    return mean


@dataclass(frozen=True)
class GrainRecord:
    grain_id: int
    phase_id: int
    size: int
    centroid: np.ndarray
    mean_orientation: np.ndarray


def grain_records(
    data: EBSDMap,
    phases: Mapping[int, EBSDPhase],
    segmentation: FastSegmentation,
) -> tuple[GrainRecord, ...]:
    output: list[GrainRecord] = []
    for grain_id in range(segmentation.n_grains):
        points = np.flatnonzero(segmentation.labels == grain_id)
        phase_values = np.unique(data.phase_id[points])
        if len(phase_values) != 1:
            raise AssertionError("segmentation mixed phases")
        phase_id = int(phase_values[0])
        output.append(
            GrainRecord(
                grain_id=grain_id,
                phase_id=phase_id,
                size=len(points),
                centroid=np.mean(data.coordinates[points], axis=0),
                mean_orientation=mean_orientation_fast(
                    data.orientations[points],
                    phases[phase_id].proper_symmetry_cartesian,
                ),
            )
        )
    return tuple(output)


def grain_adjacencies(
    data: EBSDMap,
    labels: np.ndarray,
    edges: np.ndarray,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
]:
    same: set[tuple[int, int]] = set()
    cross: set[tuple[int, int]] = set()

    for a, b in edges:
        ga = int(labels[int(a)])
        gb = int(labels[int(b)])
        if ga < 0 or gb < 0 or ga == gb:
            continue
        pair = tuple(sorted((ga, gb)))
        pa = int(data.phase_id[int(a)])
        pb = int(data.phase_id[int(b)])
        if pa == pb:
            same.add(pair)
        else:
            cross.add(pair)
    return tuple(sorted(same)), tuple(sorted(cross))


def exact_same_phase_distance_deg(
    first: np.ndarray,
    second: np.ndarray,
    symmetry: Sequence[np.ndarray],
) -> float:
    raw = np.asarray(first, dtype=float).T @ np.asarray(second, dtype=float)
    sym = np.asarray(symmetry, dtype=float)
    products = np.einsum("ij,sjk->sik", raw, sym, optimize=True)
    traces = np.trace(products, axis1=1, axis2=2)
    cosine = np.clip((traces - 1.0) * 0.5, -1.0, 1.0)
    return float(np.min(np.degrees(np.arccos(cosine))))


def deterministic_fit_holdout(
    pairs: Sequence[tuple[int, int]],
    *,
    holdout_fraction: float = 0.25,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
]:
    pairs = tuple(sorted(set(pairs)))
    if len(pairs) < 5:
        return pairs, tuple()

    def key(pair: tuple[int, int]) -> bytes:
        return hashlib.sha256(f"{pair[0]}:{pair[1]}".encode()).digest()

    ordered = sorted(pairs, key=key)
    n_holdout = max(1, int(round(len(ordered) * holdout_fraction)))
    n_holdout = min(n_holdout, len(ordered) - 4)
    if n_holdout <= 0:
        return pairs, tuple()
    holdout = tuple(sorted(ordered[:n_holdout]))
    holdout_set = set(holdout)
    fit = tuple(pair for pair in pairs if pair not in holdout_set)
    return fit, holdout


def holdout_operator_residuals(
    grain_orientations: np.ndarray,
    holdout: Sequence[tuple[int, int]],
    product_phase: EBSDPhase,
    parent_phase: EBSDPhase,
    recovered_R: np.ndarray,
) -> np.ndarray:
    if not holdout:
        return np.empty(0, dtype=float)
    theory = build_theory_library(
        parent_phase,
        product_phase,
        recovered_R,
        crosscheck_topology=True,
    )
    kernel = PreparedBoundaryOperatorKernel.prepare(
        grain_orientations,
        holdout,
        product_phase.proper_symmetry_cartesian,
    )
    return kernel.best_class_residuals_deg(theory.boundary_operators)


def interphase_validation(
    grains: Sequence[GrainRecord],
    cross_pairs: Sequence[tuple[int, int]],
    *,
    candidate_parent_phase_id: int,
    candidate_product_phase_id: int,
    parent_phase: EBSDPhase,
    product_phase: EBSDPhase,
    recovered_R: np.ndarray,
    null_permutations: int,
    seed: int,
) -> dict[str, Any]:
    theory = build_theory_library(
        parent_phase,
        product_phase,
        recovered_R,
        crosscheck_topology=True,
    )
    variants = theory.variant_set.variants

    contacts: list[tuple[int, int]] = []
    for ga, gb in cross_pairs:
        pa = grains[ga].phase_id
        pb = grains[gb].phase_id
        if pa == candidate_parent_phase_id and pb == candidate_product_phase_id:
            contacts.append((ga, gb))
        elif pb == candidate_parent_phase_id and pa == candidate_product_phase_id:
            contacts.append((gb, ga))

    if not contacts:
        return {
            "n_cross_phase_contacts": 0,
            "median_residual_deg": None,
            "p90_residual_deg": None,
            "null_permutation_count": 0,
            "empirical_p_value": None,
        }

    parent_indices = [
        index for index, grain in enumerate(grains)
        if grain.phase_id == candidate_parent_phase_id
    ]
    if not parent_indices:
        raise AssertionError("cross-phase contacts exist but no parent grains exist")

    def contact_residual(parent_index: int, product_index: int) -> float:
        gA = grains[parent_index].mean_orientation
        gM = grains[product_index].mean_orientation
        best = 180.0
        for variant in variants:
            candidate_parent = (
                gM @ variant.R_parent_from_product.T
            )
            residual = exact_same_phase_distance_deg(
                gA,
                candidate_parent,
                parent_phase.proper_symmetry_cartesian,
            )
            if residual < best:
                best = residual
        return best

    observed = np.asarray(
        [contact_residual(a, m) for a, m in contacts],
        dtype=float,
    )
    observed_median = float(np.median(observed))

    rng = np.random.default_rng(seed)
    null_medians = np.empty(null_permutations, dtype=float)
    for permutation in range(null_permutations):
        sampled_parent = rng.choice(
            parent_indices,
            size=len(contacts),
            replace=True,
        )
        values = [
            contact_residual(
                int(sampled_parent[i]),
                contacts[i][1],
            )
            for i in range(len(contacts))
        ]
        null_medians[permutation] = float(np.median(values))

    p_value = (
        1.0
        + float(np.count_nonzero(null_medians <= observed_median))
    ) / (1.0 + float(null_permutations))

    return {
        "n_cross_phase_contacts": len(contacts),
        "median_residual_deg": observed_median,
        "p90_residual_deg": float(np.quantile(observed, 0.90)),
        "minimum_residual_deg": float(np.min(observed)),
        "maximum_residual_deg": float(np.max(observed)),
        "null_permutation_count": int(null_permutations),
        "null_median_mean_deg": float(np.mean(null_medians)),
        "null_median_p05_deg": float(np.quantile(null_medians, 0.05)),
        "empirical_p_value": float(p_value),
    }


def blind_settings_for_map(seed: int) -> BlindORSettings:
    # No material-specific values.  These are generic numerical search settings.
    return BlindORSettings(
        global_samples=768,
        coarse_boundary_limit=32,
        coarse_keep=32,
        exact_keep=10,
        local_radius_deg=30.0,
        local_max_iterations=70,
        trim_fraction=0.70,
        huber_delta_deg=2.0,
        support_threshold_deg=3.0,
        minimum_support_fraction=0.60,
        minimum_boundaries=4,
        minimum_distinct_operator_classes=2,
        candidate_dedup_deg=0.50,
        ambiguity_median_gap_deg=0.25,
        acceptance_median_deg=2.0,
        sobol_seed=int(seed),
    )


def role_run(
    grains: Sequence[GrainRecord],
    same_pairs: Sequence[tuple[int, int]],
    cross_pairs: Sequence[tuple[int, int]],
    *,
    candidate_parent_phase_id: int,
    candidate_product_phase_id: int,
    phases: Mapping[int, EBSDPhase],
    null_permutations: int,
    seed: int,
) -> dict[str, Any]:
    parent_global = [
        i for i, grain in enumerate(grains)
        if grain.phase_id == candidate_parent_phase_id
    ]
    product_global = [
        i for i, grain in enumerate(grains)
        if grain.phase_id == candidate_product_phase_id
    ]
    global_to_local = {
        grain_id: local for local, grain_id in enumerate(product_global)
    }
    local_pairs = []
    for ga, gb in same_pairs:
        if ga in global_to_local and gb in global_to_local:
            local_pairs.append(
                (global_to_local[ga], global_to_local[gb])
            )

    role_cross_contacts = 0
    for ga, gb in cross_pairs:
        pa = grains[ga].phase_id
        pb = grains[gb].phase_id
        if {pa, pb} == {
            candidate_parent_phase_id,
            candidate_product_phase_id,
        }:
            role_cross_contacts += 1

    # A blind interphase relationship is not fitted if one candidate phase has
    # no retained grain or if the retained grain graph contains no interface
    # between the two phases.  Fitting a same-phase child network in that
    # situation could produce a mathematically good subgroup embedding with no
    # experimental evidence that the other phase participates at all.
    if (
        len(parent_global) == 0
        or len(product_global) < 3
        or role_cross_contacts == 0
    ):
        return {
            "candidate_parent_phase_id": candidate_parent_phase_id,
            "candidate_product_phase_id": candidate_product_phase_id,
            "status": "insufficient_evidence",
            "n_parent_grains": len(parent_global),
            "n_product_grains": len(product_global),
            "n_product_product_boundaries": len(local_pairs),
            "n_cross_phase_grain_contacts": role_cross_contacts,
            "n_fit_boundaries": 0,
            "n_holdout_boundaries": 0,
            "note": (
                "The retained grain graph does not contain both phases with "
                "at least one interphase grain contact; blind interphase OR "
                "inference is therefore not experimentally identifiable."
            ),
        }

    orientations = np.asarray(
        [grains[index].mean_orientation for index in product_global],
        dtype=float,
    )

    fit, holdout = deterministic_fit_holdout(local_pairs)
    if len(fit) < 4:
        return {
            "candidate_parent_phase_id": candidate_parent_phase_id,
            "candidate_product_phase_id": candidate_product_phase_id,
            "status": "insufficient_evidence",
            "n_parent_grains": len(parent_global),
            "n_product_grains": len(product_global),
            "n_product_product_boundaries": len(local_pairs),
            "n_cross_phase_grain_contacts": role_cross_contacts,
            "n_fit_boundaries": len(fit),
            "n_holdout_boundaries": len(holdout),
            "note": "Not enough product-grain boundary evidence for blind inversion.",
        }

    result = discover_orientation_relationship_blind(
        orientations,
        fit,
        phases[candidate_parent_phase_id],
        phases[candidate_product_phase_id],
        settings=blind_settings_for_map(seed),
    )

    row: dict[str, Any] = {
        "candidate_parent_phase_id": candidate_parent_phase_id,
        "candidate_product_phase_id": candidate_product_phase_id,
        "status": result.status,
        "identification_target": result.identification_target,
        "n_parent_grains": len(parent_global),
        "n_product_grains": len(product_global),
        "n_product_product_boundaries": len(local_pairs),
        "n_cross_phase_grain_contacts": role_cross_contacts,
        "n_fit_boundaries": len(fit),
        "n_holdout_boundaries": len(holdout),
        "evidence_note": result.evidence_note,
    }
    if result.best is None:
        return row

    row["fit"] = {
        "median_residual_deg": result.best.median_residual_deg,
        "p90_residual_deg": result.best.p90_residual_deg,
        "support_fraction": result.best.support_fraction,
        "n_variants": result.best.n_variants,
        "n_operator_classes": result.best.n_operator_classes,
        "n_distinct_accepted_operator_classes": (
            result.best.n_distinct_accepted_operator_classes
        ),
        "R_parent_from_product_representative": (
            result.best.R_parent_from_product
        ),
    }

    holdout_residuals = holdout_operator_residuals(
        orientations,
        holdout,
        phases[candidate_product_phase_id],
        phases[candidate_parent_phase_id],
        result.best.R_parent_from_product,
    )
    row["holdout"] = {
        "n": len(holdout_residuals),
        "median_residual_deg": (
            float(np.median(holdout_residuals))
            if len(holdout_residuals)
            else None
        ),
        "p90_residual_deg": (
            float(np.quantile(holdout_residuals, 0.90))
            if len(holdout_residuals)
            else None
        ),
        "support_fraction_le_3deg": (
            float(np.mean(holdout_residuals <= 3.0))
            if len(holdout_residuals)
            else None
        ),
    }

    row["cross_phase_validation"] = interphase_validation(
        grains,
        cross_pairs,
        candidate_parent_phase_id=candidate_parent_phase_id,
        candidate_product_phase_id=candidate_product_phase_id,
        parent_phase=phases[candidate_parent_phase_id],
        product_phase=phases[candidate_product_phase_id],
        recovered_R=result.best.R_parent_from_product,
        null_permutations=null_permutations,
        seed=seed + 1000003,
    )
    return row


def consensus_summary(
    threshold_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_role: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for threshold in threshold_results:
        for role in threshold["ordered_role_tests"]:
            key = (
                int(role["candidate_parent_phase_id"]),
                int(role["candidate_product_phase_id"]),
            )
            by_role.setdefault(key, []).append(role)

    summaries = []
    for key, rows in sorted(by_role.items()):
        usable = [row for row in rows if "fit" in row]
        holdout_values = [
            row["holdout"]["median_residual_deg"]
            for row in usable
            if row.get("holdout", {}).get("median_residual_deg") is not None
        ]
        cross_values = [
            row["cross_phase_validation"]["median_residual_deg"]
            for row in usable
            if row.get("cross_phase_validation", {}).get(
                "median_residual_deg"
            ) is not None
        ]
        p_values = [
            row["cross_phase_validation"]["empirical_p_value"]
            for row in usable
            if row.get("cross_phase_validation", {}).get(
                "empirical_p_value"
            ) is not None
        ]
        summaries.append(
            {
                "candidate_parent_phase_id": key[0],
                "candidate_product_phase_id": key[1],
                "thresholds_tested": len(rows),
                "thresholds_with_blind_solution": len(usable),
                "statuses": [row["status"] for row in rows],
                "median_of_fit_medians_deg": (
                    float(
                        np.median(
                            [row["fit"]["median_residual_deg"] for row in usable]
                        )
                    )
                    if usable
                    else None
                ),
                "median_of_holdout_medians_deg": (
                    float(np.median(holdout_values))
                    if holdout_values
                    else None
                ),
                "median_of_cross_phase_medians_deg": (
                    float(np.median(cross_values))
                    if cross_values
                    else None
                ),
                "median_cross_phase_empirical_p": (
                    float(np.median(p_values))
                    if p_values
                    else None
                ),
            }
        )

    # Evidence table only.  No automatic "winner" is declared.
    return {
        "ordered_role_evidence": summaries,
        "selection_policy": (
            "No role is forced automatically. Compare stability across "
            "segmentation thresholds, untouched holdout residuals, and "
            "independent cross-phase permutation evidence."
        ),
    }


def load_sealed_challenge(root: Path) -> tuple[
    dict[str, Any], EBSDMap, dict[int, EBSDPhase]
]:
    challenge_path = root / "challenge.json"
    observations_path = root / "observations.npz"
    if not challenge_path.is_file() or not observations_path.is_file():
        raise FileNotFoundError(
            "challenge directory must contain challenge.json and observations.npz"
        )

    challenge_sha = sha256_file(challenge_path)
    observations_sha = sha256_file(observations_path)
    if challenge_sha != EXPECTED_CHALLENGE_SHA256:
        raise RuntimeError(
            "challenge.json hash mismatch: sealed challenge was modified"
        )
    if observations_sha != EXPECTED_OBSERVATIONS_SHA256:
        raise RuntimeError(
            "observations.npz hash mismatch: sealed observations were modified"
        )

    challenge = json.loads(challenge_path.read_text())
    validate_zero_clue_contract(challenge)

    phase_specs = challenge["phases"]
    phases = {
        int(spec["id"]): phase_from_spec(spec)
        for spec in phase_specs
    }
    if len(phases) < 2:
        raise RuntimeError("role-free test requires at least two phases")

    with np.load(observations_path) as payload:
        orientations = np.asarray(payload["orientations"], dtype=float)
        phase_id = np.asarray(payload["phase_id"], dtype=int)
        indexed = np.asarray(payload["indexed"], dtype=bool)
        x = np.asarray(payload["x"], dtype=float)
        y = np.asarray(payload["y"], dtype=float)
        z = np.asarray(payload["z"], dtype=float)
        quality = {
            key[len("quality_"):]: np.asarray(payload[key])
            for key in payload.files
            if key.startswith("quality_")
        }

    # Unindexed finite orientations are deliberately retained but never used.
    metadata = dict(challenge.get("grid_metadata", {}))
    data = EBSDMap(
        orientations=orientations,
        phase_id=phase_id,
        indexed=indexed,
        x=x,
        y=y,
        z=z,
        quality=quality,
        metadata=metadata,
    )
    return challenge, data, phases


def main() -> int:
    if len(sys.argv) not in {2, 3}:
        print(
            "usage: python run_zero_clue_real_map.py "
            "data/real_blind_challenge [output.json]",
            file=sys.stderr,
        )
        return 2

    challenge_root = Path(sys.argv[1]).resolve()
    output_path = (
        Path(sys.argv[2]).resolve()
        if len(sys.argv) == 3
        else Path("zero_clue_real_map_report.json").resolve()
    )

    start_total = time.perf_counter()
    challenge, data, phases = load_sealed_challenge(challenge_root)
    settings = challenge["settings"]

    print("SEALED CHALLENGE VERIFIED")
    print("points:", data.n_points)
    print("anonymous phases:", sorted(phases))
    print("No material names / phase roles / OR supplied.")
    print()

    audit = audit_map(data)
    print("Building spatial neighbor graph...", flush=True)
    graph = build_neighbor_graph(data)
    edges = np.asarray(graph.edges, dtype=int)
    print(
        f"neighbor edges: {len(edges):,}; radius={graph.radius:g}",
        flush=True,
    )

    print("Precomputing exact same-phase edge disorientations once...", flush=True)
    t0 = time.perf_counter()
    edge_angles = exact_same_phase_edge_angles(
        data,
        phases,
        edges,
    )
    print(
        f"edge crystallography precompute: {time.perf_counter()-t0:.1f} s",
        flush=True,
    )

    thresholds = [
        float(x) for x in settings["segmentation_thresholds_deg"]
    ]
    minimum_points = int(settings["minimum_grain_points"])
    null_permutations = int(settings["null_permutations"])
    seed = int(settings["random_seed"])

    threshold_results = []
    phase_ids = sorted(phases)

    for threshold_index, threshold in enumerate(thresholds):
        print(
            f"\n=== SEGMENTATION THRESHOLD {threshold:g} deg ===",
            flush=True,
        )
        t_threshold = time.perf_counter()

        segmentation = segmentation_from_precomputed_edges(
            data,
            edges,
            edge_angles,
            threshold_deg=threshold,
            minimum_points=minimum_points,
        )
        grains = grain_records(data, phases, segmentation)
        same_pairs, cross_pairs = grain_adjacencies(
            data,
            segmentation.labels,
            edges,
        )

        phase_grain_counts = {
            phase_id: sum(
                grain.phase_id == phase_id for grain in grains
            )
            for phase_id in phase_ids
        }
        print(
            "grains:",
            ", ".join(
                f"phase_{pid}={phase_grain_counts[pid]}"
                for pid in phase_ids
            ),
            "| cross-phase grain contacts:",
            len(cross_pairs),
            flush=True,
        )

        ordered_role_tests = []
        for parent_id in phase_ids:
            for product_id in phase_ids:
                if parent_id == product_id:
                    continue
                print(
                    f"  testing anonymous role {parent_id} <- {product_id} ...",
                    flush=True,
                )
                role_seed = (
                    seed
                    + 1009 * threshold_index
                    + 9176 * parent_id
                    + 65537 * product_id
                )
                t_role = time.perf_counter()
                role = role_run(
                    grains,
                    same_pairs,
                    cross_pairs,
                    candidate_parent_phase_id=parent_id,
                    candidate_product_phase_id=product_id,
                    phases=phases,
                    null_permutations=null_permutations,
                    seed=role_seed,
                )
                role["elapsed_seconds"] = (
                    time.perf_counter() - t_role
                )
                ordered_role_tests.append(role)

                if "fit" in role:
                    h = role["holdout"]
                    c = role["cross_phase_validation"]
                    print(
                        "   ",
                        role["status"],
                        f"fit_med={role['fit']['median_residual_deg']:.3f}°",
                        (
                            f"holdout_med={h['median_residual_deg']:.3f}°"
                            if h["median_residual_deg"] is not None
                            else "holdout=NA"
                        ),
                        (
                            f"cross_med={c['median_residual_deg']:.3f}° "
                            f"p={c['empirical_p_value']:.4f}"
                            if c["median_residual_deg"] is not None
                            else "cross=NA"
                        ),
                        flush=True,
                    )
                else:
                    print("   ", role["status"], flush=True)

        threshold_results.append(
            {
                "threshold_deg": threshold,
                "n_grains": segmentation.n_grains,
                "phase_grain_counts": phase_grain_counts,
                "n_same_phase_grain_adjacencies": len(same_pairs),
                "n_cross_phase_grain_adjacencies": len(cross_pairs),
                "ordered_role_tests": ordered_role_tests,
                "elapsed_seconds": time.perf_counter() - t_threshold,
            }
        )

    report = {
        "schema_version": 1,
        "challenge_integrity": {
            "challenge_json_sha256": EXPECTED_CHALLENGE_SHA256,
            "observations_npz_sha256": EXPECTED_OBSERVATIONS_SHA256,
            "structural_zero_clue_contract": "PASS",
        },
        "inputs_visible_to_solver": {
            "anonymous_phase_ids": phase_ids,
            "phase_names_withheld": True,
            "parent_product_roles_withheld": True,
            "orientation_relationship_withheld": True,
            "correspondence_withheld": True,
            "expected_operator_families_withheld": True,
            "chemical_identity_withheld": True,
        },
        "map_audit": audit,
        "neighbor_graph": {
            "n_edges": len(edges),
            "radius": graph.radius,
            "coordinate_dimension": graph.coordinate_dimension,
        },
        "threshold_results": threshold_results,
        "consensus": consensus_summary(threshold_results),
        "elapsed_seconds": time.perf_counter() - start_total,
    }
    output_path.write_text(
        json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n"
    )

    print("\n=== ZERO-CLUE REAL-MAP RUN COMPLETE ===")
    print("report:", output_path)
    print(
        "No material identity, phase role, OR, correspondence, or expected "
        "operator answer was supplied."
    )
    print(
        "Do not reveal external answers before preserving this report."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
