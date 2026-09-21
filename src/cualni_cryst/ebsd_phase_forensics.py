from __future__ import annotations

"""Phase-survival forensics and role-free interphase OR concentration analysis.

This module exists to prevent a scientifically important failure mode:

    a phase is present in indexed EBSD pixels,
    but disappears after orientation segmentation + minimum-grain filtering,
    so downstream crystallographic inference silently sees zero grains.

The module performs a mandatory *pre-inference* audit at three levels:

1. raw same-phase spatial connected components (no orientation threshold);
2. orientation-connected components with minimum_points=1;
3. retention/survival under a declared grid of minimum-component sizes.

It also provides a role-free direct interphase relationship route based on
cross-phase interfaces.  This route does not need a parent/product assignment,
a literature OR, a correspondence, or child/child variants.  It estimates
whether measured cross-phase component pairs concentrate around a common
relative-orientation class under the complete left/right crystal-symmetry
quotient.

The direct relation is reported as

    R_phase_a_from_phase_b = g_a.T @ g_b,   phase_a < phase_b,

where every internal EBSD orientation obeys

    v_sample = g @ v_crystal.

No transformation direction is inferred from that definition.

Scientific safeguards
---------------------
* No phase can disappear silently: raw pixels, spatial components,
  orientation components, and retention fractions are all reported.
* The minimum-size cutoff is never auto-lowered.
* An interface fallback is evidence-only; it does not relabel noise as grains.
* Interface units are unique orientation-component pairs, not raw pixel edges,
  preventing one long interface from dominating only by pixel count.
* Discovery and validation are separated deterministically into FIT/HOLDOUT
  interface units.
* The fitted relation is evaluated on untouched holdout units.
* A pairing-permutation null test is run only on holdout data.
* The OR search uses the complete left/right proper-symmetry quotient.
* Threshold sensitivity is explicit; no threshold is silently selected.
"""

from dataclasses import dataclass
import hashlib
from math import ceil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .ebsd_analysis import (
    NeighborGraph,
    minimum_disorientation,
    segment_grains,
    symmetry_aligned_mean,
)
from .ebsd_map import EBSDMap, EBSDPhase
from .orientation_kernel import require_so3


@dataclass(frozen=True)
class PhaseForensicsSettings:
    orientation_thresholds_deg: tuple[float, ...] = (
        1.0,
        2.0,
        3.0,
        5.0,
        7.5,
        10.0,
    )
    minimum_component_sizes: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    relation_support_threshold_deg: float = 3.0
    relation_minimum_interface_units: int = 8
    relation_seed_cap: int = 128
    relation_refine_seeds: int = 6
    relation_trim_fraction: float = 0.70
    relation_huber_delta_deg: float = 2.0
    relation_local_radius_deg: float = 8.0
    relation_local_max_iterations: int = 45
    relation_minimum_holdout_support: float = 0.50
    relation_maximum_holdout_median_deg: float = 3.0
    null_permutations: int = 128
    null_significance_level: float = 0.05
    random_seed: int = 20260921
    mean_batch_size: int = 4096
    reliability_component_min_points: int = 2
    reliability_minimum_spatial_supported_pixel_fraction: float = 0.20
    reliability_minimum_orientation_supported_pixel_fraction: float = 0.20
    reliability_minimum_supported_components: int = 8

    def __post_init__(self) -> None:
        thresholds = tuple(float(x) for x in self.orientation_thresholds_deg)
        if not thresholds or any(x <= 0.0 for x in thresholds):
            raise ValueError("orientation_thresholds_deg must be positive")
        if any(b <= a for a, b in zip(thresholds, thresholds[1:])):
            raise ValueError(
                "orientation_thresholds_deg must be strictly increasing"
            )
        sizes = tuple(int(x) for x in self.minimum_component_sizes)
        if not sizes or any(x < 1 for x in sizes):
            raise ValueError("minimum_component_sizes must be >= 1")
        if any(b <= a for a, b in zip(sizes, sizes[1:])):
            raise ValueError(
                "minimum_component_sizes must be strictly increasing"
            )
        if self.relation_support_threshold_deg <= 0.0:
            raise ValueError("relation_support_threshold_deg must be positive")
        if self.relation_minimum_interface_units < 4:
            raise ValueError(
                "relation_minimum_interface_units must be >= 4"
            )
        if self.relation_seed_cap < 1 or self.relation_refine_seeds < 1:
            raise ValueError("relation seed counts must be positive")
        if self.relation_refine_seeds > self.relation_seed_cap:
            raise ValueError(
                "relation_refine_seeds cannot exceed relation_seed_cap"
            )
        if not (0.0 < self.relation_trim_fraction <= 1.0):
            raise ValueError("relation_trim_fraction must lie in (0,1]")
        if self.relation_huber_delta_deg <= 0.0:
            raise ValueError("relation_huber_delta_deg must be positive")
        if self.relation_local_radius_deg <= 0.0:
            raise ValueError("relation_local_radius_deg must be positive")
        if self.relation_local_max_iterations < 1:
            raise ValueError("relation_local_max_iterations must be >= 1")
        if not (0.0 <= self.relation_minimum_holdout_support <= 1.0):
            raise ValueError(
                "relation_minimum_holdout_support must lie in [0,1]"
            )
        if self.relation_maximum_holdout_median_deg <= 0.0:
            raise ValueError(
                "relation_maximum_holdout_median_deg must be positive"
            )
        if self.null_permutations < 16:
            raise ValueError("null_permutations must be >= 16")
        if not (0.0 < self.null_significance_level < 1.0):
            raise ValueError(
                "null_significance_level must lie strictly between 0 and 1"
            )
        if self.mean_batch_size < 1:
            raise ValueError("mean_batch_size must be positive")
        if self.reliability_component_min_points < 2:
            raise ValueError(
                "reliability_component_min_points must be >= 2"
            )
        for name, value in (
            (
                "reliability_minimum_spatial_supported_pixel_fraction",
                self.reliability_minimum_spatial_supported_pixel_fraction,
            ),
            (
                "reliability_minimum_orientation_supported_pixel_fraction",
                self.reliability_minimum_orientation_supported_pixel_fraction,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if self.reliability_minimum_supported_components < 1:
            raise ValueError(
                "reliability_minimum_supported_components must be >= 1"
            )


@dataclass(frozen=True)
class ComponentSegmentation:
    component_id: np.ndarray
    n_components: int
    threshold_deg: float | None
    component_sizes: np.ndarray
    component_phase_id: np.ndarray


@dataclass(frozen=True)
class PhaseSurvivalRecord:
    phase_id: int
    raw_indexed_pixels: int
    raw_spatial_components: int
    largest_raw_spatial_component: int
    raw_spatial_component_size_quantiles: Mapping[str, float]
    raw_components_ge_size: Mapping[int, int]
    raw_pixel_fraction_in_components_ge_size: Mapping[int, float]
    same_phase_edge_count: int
    same_phase_edge_misorientation_quantiles_deg: Mapping[str, float | None]
    quality_quantiles: Mapping[str, Mapping[str, float | None]]


@dataclass(frozen=True)
class ThresholdPhaseSurvival:
    threshold_deg: float
    phase_id: int
    orientation_components: int
    largest_orientation_component: int
    components_ge_size: Mapping[int, int]
    pixel_fraction_in_components_ge_size: Mapping[int, float]


@dataclass(frozen=True)
class PhaseReliabilityAssessment:
    phase_id: int
    threshold_deg: float
    interface_inference_allowed: bool
    minimum_component_points: int
    raw_supported_components: int
    raw_supported_pixel_fraction: float
    orientation_supported_components: int
    orientation_supported_pixel_fraction: float
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class GrainRouteReliabilityAssessment:
    phase_id: int
    allowed: bool
    indexed_pixels: int
    retained_pixels: int
    retained_pixel_fraction: float
    retained_grains: int
    largest_retained_grain: int
    minimum_required_grains: int
    reason_codes: tuple[str, ...]


class InsufficientPhaseEvidenceError(RuntimeError):
    """Experimental evidence is insufficient for the requested EBSD route."""


@dataclass(frozen=True)
class InterfaceUnit:
    phase_a: int
    phase_b: int
    component_a: int
    component_b: int
    n_pixel_edges: int
    g_a: np.ndarray
    g_b: np.ndarray

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (
            self.phase_a,
            self.phase_b,
            self.component_a,
            self.component_b,
        )

    @property
    def relation_a_from_b(self) -> np.ndarray:
        return self.g_a.T @ self.g_b


@dataclass(frozen=True)
class InterphaseRelationCandidate:
    R_phase_a_from_phase_b: np.ndarray
    fit_objective: float
    fit_median_deg: float
    fit_support_fraction: float
    holdout_median_deg: float | None
    holdout_p90_deg: float | None
    holdout_support_fraction: float | None
    null_empirical_p_value: float | None
    optimizer_success: bool
    optimizer_message: str


@dataclass(frozen=True)
class InterphaseRelationResult:
    phase_a: int
    phase_b: int
    status: str
    n_interface_units: int
    n_fit_units: int
    n_holdout_units: int
    best: InterphaseRelationCandidate | None
    alternatives: tuple[InterphaseRelationCandidate, ...]
    note: str


@dataclass(frozen=True)
class PairRouteEvidence:
    phase_a: int
    phase_b: int
    raw_cross_phase_pixel_edges: int
    raw_phase_a_pixels_touching_b: int
    raw_phase_b_pixels_touching_a: int
    raw_spatial_component_pairs: int
    orientation_interface_units: int
    direct_interface_route_possible: bool


@dataclass(frozen=True)
class ForensicThresholdResult:
    threshold_deg: float
    segmentation: ComponentSegmentation
    phase_survival: tuple[ThresholdPhaseSurvival, ...]
    phase_reliability: tuple[PhaseReliabilityAssessment, ...]
    pair_routes: tuple[PairRouteEvidence, ...]
    direct_relations: tuple[InterphaseRelationResult, ...]


@dataclass(frozen=True)
class PhaseForensicsReport:
    phase_survival: tuple[PhaseSurvivalRecord, ...]
    raw_pair_routes: tuple[PairRouteEvidence, ...]
    thresholds: tuple[ForensicThresholdResult, ...]
    warnings: tuple[str, ...]


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, value: int) -> int:
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

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


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    q = np.quantile(array, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    keys = ("min", "p05", "p25", "median", "p75", "p95", "max")
    return {key: float(value) for key, value in zip(keys, q, strict=True)}


def _rotation_quaternions(matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrices, dtype=float)
    if values.shape[-2:] != (3, 3):
        raise ValueError("rotation array must end in (3,3)")
    flat = values.reshape(-1, 3, 3)
    q = Rotation.from_matrix(flat).as_quat()
    q /= np.linalg.norm(q, axis=1)[:, None]
    return q.reshape(values.shape[:-2] + (4,))


def _angles_from_projective_dots(dots: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(dots, dtype=float), 0.0, 1.0)
    sine_half = np.sqrt(np.maximum(0.0, 1.0 - value * value))
    return np.degrees(2.0 * np.arctan2(sine_half, value))


def exact_same_phase_edge_angles(
    data: EBSDMap,
    phases: Mapping[int, EBSDPhase],
    neighbor_graph: NeighborGraph,
    *,
    batch_size: int = 20000,
) -> np.ndarray:
    """Vectorized exact same-phase disorientation for every spatial edge.

    Frozen scalar definition:

        min_{S1,S2 in G} angle(S1.T @ Delta @ S2)

    with Delta = g1.T @ g2.

    Because G is a group and SO(3) angle is determined by trace,

        tr(S1.T Delta S2) = tr(Delta S2 S1.T),

    the complete two-sided quotient is exactly equivalent to one loop over

        K = S2 S1.T in G.

    This removes redundant computation only; no symmetry operation is lost.
    """

    edges = np.asarray(neighbor_graph.edges, dtype=int)
    result = np.full(len(edges), np.nan, dtype=float)
    if len(edges) == 0:
        return result

    for phase_id, phase in phases.items():
        mask = (
            data.indexed[edges[:, 0]]
            & data.indexed[edges[:, 1]]
            & (data.phase_id[edges[:, 0]] == int(phase_id))
            & (data.phase_id[edges[:, 1]] == int(phase_id))
        )
        positions = np.flatnonzero(mask)
        if len(positions) == 0:
            continue
        symmetry = np.asarray(
            phase.proper_symmetry_cartesian, dtype=float
        )
        for start in range(0, len(positions), batch_size):
            local = positions[start : start + batch_size]
            a = edges[local, 0]
            b = edges[local, 1]
            raw = np.einsum(
                "nji,njk->nik",
                data.orientations[a],
                data.orientations[b],
                optimize=True,
            )
            products = np.einsum(
                "nij,sjk->nsik",
                raw,
                symmetry,
                optimize=True,
            )
            traces = np.trace(products, axis1=2, axis2=3)
            cosine = np.clip((traces - 1.0) * 0.5, -1.0, 1.0)
            result[local] = np.min(
                np.degrees(np.arccos(cosine)),
                axis=1,
            )
    return result


def _components_from_eligible_edges(
    data: EBSDMap,
    neighbor_graph: NeighborGraph,
    eligible_edges: np.ndarray,
) -> ComponentSegmentation:
    edges = np.asarray(neighbor_graph.edges, dtype=int)
    eligible = np.asarray(eligible_edges, dtype=bool)
    if len(eligible) != len(edges):
        raise ValueError("eligible_edges length mismatch")

    uf = _UnionFind(data.n_points)
    for index in np.flatnonzero(eligible):
        a, b = edges[int(index)]
        uf.union(int(a), int(b))

    members: dict[int, list[int]] = {}
    for point in np.flatnonzero(data.indexed):
        root = uf.find(int(point))
        members.setdefault(root, []).append(int(point))

    kept = list(members.values())
    kept.sort(key=lambda points: min(points))
    labels = np.full(data.n_points, -1, dtype=np.int64)
    sizes = np.empty(len(kept), dtype=np.int64)
    phase_ids = np.empty(len(kept), dtype=np.int64)
    for component_id, points in enumerate(kept):
        array = np.asarray(points, dtype=int)
        phases_here = np.unique(data.phase_id[array])
        if len(phases_here) != 1:
            raise AssertionError("component mixed phases")
        labels[array] = component_id
        sizes[component_id] = len(array)
        phase_ids[component_id] = int(phases_here[0])

    return ComponentSegmentation(
        component_id=labels,
        n_components=len(kept),
        threshold_deg=None,
        component_sizes=sizes,
        component_phase_id=phase_ids,
    )


def spatial_components(
    data: EBSDMap,
    neighbor_graph: NeighborGraph,
) -> ComponentSegmentation:
    edges = np.asarray(neighbor_graph.edges, dtype=int)
    if len(edges) == 0:
        eligible = np.empty(0, dtype=bool)
    else:
        eligible = (
            data.indexed[edges[:, 0]]
            & data.indexed[edges[:, 1]]
            & (data.phase_id[edges[:, 0]] == data.phase_id[edges[:, 1]])
        )
    return _components_from_eligible_edges(
        data,
        neighbor_graph,
        eligible,
    )


def orientation_components(
    data: EBSDMap,
    neighbor_graph: NeighborGraph,
    edge_angles_deg: np.ndarray,
    *,
    threshold_deg: float,
) -> ComponentSegmentation:
    if threshold_deg <= 0.0:
        raise ValueError("threshold_deg must be positive")
    edges = np.asarray(neighbor_graph.edges, dtype=int)
    angles = np.asarray(edge_angles_deg, dtype=float)
    if len(angles) != len(edges):
        raise ValueError("edge_angles_deg length mismatch")
    eligible = np.isfinite(angles) & (angles <= threshold_deg)
    result = _components_from_eligible_edges(
        data,
        neighbor_graph,
        eligible,
    )
    return ComponentSegmentation(
        component_id=result.component_id,
        n_components=result.n_components,
        threshold_deg=float(threshold_deg),
        component_sizes=result.component_sizes,
        component_phase_id=result.component_phase_id,
    )


def _survival_maps(
    sizes: np.ndarray,
    *,
    minimum_sizes: Sequence[int],
) -> tuple[dict[int, int], dict[int, float]]:
    values = np.asarray(sizes, dtype=int)
    total_pixels = int(np.sum(values))
    counts: dict[int, int] = {}
    fractions: dict[int, float] = {}
    for minimum in minimum_sizes:
        minimum = int(minimum)
        mask = values >= minimum
        counts[minimum] = int(np.count_nonzero(mask))
        fractions[minimum] = (
            float(np.sum(values[mask]) / total_pixels)
            if total_pixels
            else 0.0
        )
    return counts, fractions


def phase_survival_records(
    data: EBSDMap,
    phases: Mapping[int, EBSDPhase],
    neighbor_graph: NeighborGraph,
    spatial: ComponentSegmentation,
    edge_angles_deg: np.ndarray,
    *,
    minimum_sizes: Sequence[int],
) -> tuple[PhaseSurvivalRecord, ...]:
    edges = np.asarray(neighbor_graph.edges, dtype=int)
    output = []

    for phase_id in sorted(phases):
        raw_pixels = int(
            np.count_nonzero(data.indexed & (data.phase_id == phase_id))
        )
        component_mask = spatial.component_phase_id == phase_id
        sizes = spatial.component_sizes[component_mask]
        counts, fractions = _survival_maps(
            sizes,
            minimum_sizes=minimum_sizes,
        )

        same_edge_mask = (
            data.indexed[edges[:, 0]]
            & data.indexed[edges[:, 1]]
            & (data.phase_id[edges[:, 0]] == phase_id)
            & (data.phase_id[edges[:, 1]] == phase_id)
        )
        angles = edge_angles_deg[same_edge_mask]

        quality: dict[str, Mapping[str, float | None]] = {}
        point_mask = data.indexed & (data.phase_id == phase_id)
        for name, values in data.quality.items():
            array = np.asarray(values)
            if not np.issubdtype(array.dtype, np.number):
                continue
            quality[str(name)] = _quantiles(
                np.asarray(array[point_mask], dtype=float)
            )

        output.append(
            PhaseSurvivalRecord(
                phase_id=int(phase_id),
                raw_indexed_pixels=raw_pixels,
                raw_spatial_components=int(len(sizes)),
                largest_raw_spatial_component=(
                    int(np.max(sizes)) if len(sizes) else 0
                ),
                raw_spatial_component_size_quantiles={
                    key: float(value)
                    for key, value in _quantiles(sizes).items()
                    if value is not None
                },
                raw_components_ge_size=counts,
                raw_pixel_fraction_in_components_ge_size=fractions,
                same_phase_edge_count=int(np.count_nonzero(same_edge_mask)),
                same_phase_edge_misorientation_quantiles_deg=_quantiles(
                    angles
                ),
                quality_quantiles=quality,
            )
        )
    return tuple(output)


def threshold_phase_survival(
    segmentation: ComponentSegmentation,
    phase_ids: Iterable[int],
    *,
    minimum_sizes: Sequence[int],
) -> tuple[ThresholdPhaseSurvival, ...]:
    output = []
    if segmentation.threshold_deg is None:
        raise ValueError("orientation segmentation must have a threshold")
    for phase_id in sorted(int(x) for x in phase_ids):
        mask = segmentation.component_phase_id == phase_id
        sizes = segmentation.component_sizes[mask]
        counts, fractions = _survival_maps(
            sizes,
            minimum_sizes=minimum_sizes,
        )
        output.append(
            ThresholdPhaseSurvival(
                threshold_deg=float(segmentation.threshold_deg),
                phase_id=phase_id,
                orientation_components=int(len(sizes)),
                largest_orientation_component=(
                    int(np.max(sizes)) if len(sizes) else 0
                ),
                components_ge_size=counts,
                pixel_fraction_in_components_ge_size=fractions,
            )
        )
    return tuple(output)


def _project_so3(matrix: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(matrix, dtype=float))
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return require_so3(
        U @ correction @ Vt,
        tolerance=2.0e-9,
        name="component mean orientation",
    )


def symmetry_aligned_mean_vectorized(
    orientations: np.ndarray,
    symmetry: Sequence[np.ndarray],
    *,
    max_iterations: int = 50,
    convergence_deg: float = 1.0e-9,
    batch_size: int = 4096,
) -> np.ndarray:
    """Batched vectorization of the frozen symmetry-aligned matrix mean.

    This computes the same algorithm as ``ebsd_analysis.symmetry_aligned_mean``:
    nearest symmetry copy -> weighted matrix average -> SVD projection to SO(3).
    It is separated here so it can be cross-locked by regression tests.
    """

    values = np.asarray(orientations, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or len(values) == 0:
        raise ValueError("orientations must have shape (N,3,3), N>0")
    sym = np.asarray(symmetry, dtype=float)
    if sym.ndim != 3 or sym.shape[1:] != (3, 3) or len(sym) == 0:
        raise ValueError("symmetry must have shape (S,3,3), S>0")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    mean = values[0].copy()
    weights = np.full(len(values), 1.0 / len(values), dtype=float)

    for _ in range(max_iterations):
        raw_mean = np.zeros((3, 3), dtype=float)
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            local = values[start:stop]
            candidates = np.einsum(
                "nij,sjk->nsik",
                local,
                sym,
                optimize=True,
            )
            # angle is monotone decreasing with trace on SO(3)
            scores = np.einsum(
                "ij,nsij->ns",
                mean,
                candidates,
                optimize=True,
            )
            best = np.argmax(scores, axis=1)
            chosen = candidates[np.arange(len(local)), best]
            raw_mean += np.tensordot(
                weights[start:stop],
                chosen,
                axes=(0, 0),
            )
        updated = _project_so3(raw_mean)
        cosine = np.clip(
            (np.trace(mean.T @ updated) - 1.0) * 0.5,
            -1.0,
            1.0,
        )
        change = float(np.degrees(np.arccos(cosine)))
        mean = updated
        if change <= convergence_deg:
            break
    return mean


def component_mean_orientations(
    data: EBSDMap,
    phases: Mapping[int, EBSDPhase],
    segmentation: ComponentSegmentation,
    *,
    batch_size: int = 4096,
) -> np.ndarray:
    means = np.full((segmentation.n_components, 3, 3), np.nan, dtype=float)
    order = np.argsort(segmentation.component_id, kind="stable")
    labels_sorted = segmentation.component_id[order]
    valid_positions = labels_sorted >= 0
    order = order[valid_positions]
    labels_sorted = labels_sorted[valid_positions]

    if len(order) == 0:
        return means

    boundaries = np.flatnonzero(
        np.r_[True, labels_sorted[1:] != labels_sorted[:-1], True]
    )
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        component_id = int(labels_sorted[start])
        points = order[start:stop]
        phase_id = int(segmentation.component_phase_id[component_id])
        values = data.orientations[points]
        if len(values) == 1:
            means[component_id] = values[0]
        else:
            means[component_id] = symmetry_aligned_mean_vectorized(
                values,
                phases[phase_id].proper_symmetry_cartesian,
                batch_size=batch_size,
            )
    return means


def interface_units(
    data: EBSDMap,
    neighbor_graph: NeighborGraph,
    segmentation: ComponentSegmentation,
    component_means: np.ndarray,
) -> tuple[InterfaceUnit, ...]:
    edges = np.asarray(neighbor_graph.edges, dtype=int)
    grouped: dict[tuple[int, int, int, int], int] = {}

    for a, b in edges:
        a = int(a)
        b = int(b)
        if not data.indexed[a] or not data.indexed[b]:
            continue
        pa = int(data.phase_id[a])
        pb = int(data.phase_id[b])
        if pa == pb:
            continue
        ca = int(segmentation.component_id[a])
        cb = int(segmentation.component_id[b])
        if ca < 0 or cb < 0:
            continue

        if pa < pb:
            key = (pa, pb, ca, cb)
        else:
            key = (pb, pa, cb, ca)
        grouped[key] = grouped.get(key, 0) + 1

    output = []
    for key in sorted(grouped):
        pa, pb, ca, cb = key
        output.append(
            InterfaceUnit(
                phase_a=pa,
                phase_b=pb,
                component_a=ca,
                component_b=cb,
                n_pixel_edges=int(grouped[key]),
                g_a=component_means[ca].copy(),
                g_b=component_means[cb].copy(),
            )
        )
    return tuple(output)


def raw_pair_route_evidence(
    data: EBSDMap,
    neighbor_graph: NeighborGraph,
    spatial: ComponentSegmentation,
    *,
    minimum_interface_units: int,
) -> tuple[PairRouteEvidence, ...]:
    edges = np.asarray(neighbor_graph.edges, dtype=int)
    phase_ids = sorted(
        int(x) for x in np.unique(data.phase_id[data.indexed])
    )
    output = []

    for i, phase_a in enumerate(phase_ids):
        for phase_b in phase_ids[i + 1 :]:
            cross = (
                data.indexed[edges[:, 0]]
                & data.indexed[edges[:, 1]]
                & (
                    (
                        (data.phase_id[edges[:, 0]] == phase_a)
                        & (data.phase_id[edges[:, 1]] == phase_b)
                    )
                    | (
                        (data.phase_id[edges[:, 0]] == phase_b)
                        & (data.phase_id[edges[:, 1]] == phase_a)
                    )
                )
            )
            selected = edges[cross]
            pixels_a: set[int] = set()
            pixels_b: set[int] = set()
            component_pairs: set[tuple[int, int]] = set()

            for left, right in selected:
                left = int(left)
                right = int(right)
                if int(data.phase_id[left]) == phase_a:
                    pa, pb = left, right
                else:
                    pa, pb = right, left
                pixels_a.add(pa)
                pixels_b.add(pb)
                ca = int(spatial.component_id[pa])
                cb = int(spatial.component_id[pb])
                if ca >= 0 and cb >= 0:
                    component_pairs.add((ca, cb))

            output.append(
                PairRouteEvidence(
                    phase_a=phase_a,
                    phase_b=phase_b,
                    raw_cross_phase_pixel_edges=int(len(selected)),
                    raw_phase_a_pixels_touching_b=int(len(pixels_a)),
                    raw_phase_b_pixels_touching_a=int(len(pixels_b)),
                    raw_spatial_component_pairs=int(len(component_pairs)),
                    orientation_interface_units=0,
                    direct_interface_route_possible=(
                        len(component_pairs) >= minimum_interface_units
                    ),
                )
            )
    return tuple(output)


def threshold_pair_route_evidence(
    raw_routes: Sequence[PairRouteEvidence],
    units: Sequence[InterfaceUnit],
    *,
    minimum_interface_units: int,
) -> tuple[PairRouteEvidence, ...]:
    raw = tuple(raw_routes)
    counts: dict[tuple[int, int], int] = {}
    for unit in units:
        key = (unit.phase_a, unit.phase_b)
        counts[key] = counts.get(key, 0) + 1

    output = []
    for item in raw:
        n_units = counts.get((item.phase_a, item.phase_b), 0)
        output.append(
            PairRouteEvidence(
                phase_a=item.phase_a,
                phase_b=item.phase_b,
                raw_cross_phase_pixel_edges=item.raw_cross_phase_pixel_edges,
                raw_phase_a_pixels_touching_b=item.raw_phase_a_pixels_touching_b,
                raw_phase_b_pixels_touching_a=item.raw_phase_b_pixels_touching_a,
                raw_spatial_component_pairs=item.raw_spatial_component_pairs,
                orientation_interface_units=int(n_units),
                direct_interface_route_possible=(
                    n_units >= minimum_interface_units
                ),
            )
        )
    return tuple(output)


def assess_phase_interface_reliability(
    raw_record: PhaseSurvivalRecord,
    threshold_record: ThresholdPhaseSurvival,
    settings: PhaseForensicsSettings,
) -> PhaseReliabilityAssessment:
    """Evaluate evidence sufficiency before any direct interphase OR fit."""

    if raw_record.phase_id != threshold_record.phase_id:
        raise ValueError("phase reliability records refer to different phases")

    minimum = int(settings.reliability_component_min_points)
    raw_components = int(
        raw_record.raw_components_ge_size.get(minimum, 0)
    )
    raw_fraction = float(
        raw_record.raw_pixel_fraction_in_components_ge_size.get(
            minimum, 0.0
        )
    )
    orientation_components_count = int(
        threshold_record.components_ge_size.get(minimum, 0)
    )
    orientation_fraction = float(
        threshold_record.pixel_fraction_in_components_ge_size.get(
            minimum, 0.0
        )
    )

    reasons: list[str] = []
    if raw_record.raw_indexed_pixels <= 0:
        reasons.append("no_indexed_pixels")
    if (
        raw_fraction
        < settings.reliability_minimum_spatial_supported_pixel_fraction
    ):
        reasons.append(
            "spatial_supported_pixel_fraction_below_minimum"
        )
    if (
        orientation_fraction
        < settings.reliability_minimum_orientation_supported_pixel_fraction
    ):
        reasons.append(
            "orientation_supported_pixel_fraction_below_minimum"
        )
    if (
        orientation_components_count
        < settings.reliability_minimum_supported_components
    ):
        reasons.append("too_few_supported_orientation_components")

    return PhaseReliabilityAssessment(
        phase_id=raw_record.phase_id,
        threshold_deg=threshold_record.threshold_deg,
        interface_inference_allowed=not reasons,
        minimum_component_points=minimum,
        raw_supported_components=raw_components,
        raw_supported_pixel_fraction=raw_fraction,
        orientation_supported_components=orientation_components_count,
        orientation_supported_pixel_fraction=orientation_fraction,
        reason_codes=tuple(reasons),
    )


def filter_interface_units_for_reliability(
    units: Sequence[InterfaceUnit],
    segmentation: ComponentSegmentation,
    *,
    minimum_component_points: int,
) -> tuple[InterfaceUnit, ...]:
    """Keep only interfaces supported by nontrivial components on both sides."""

    if minimum_component_points < 2:
        raise ValueError("minimum_component_points must be >= 2")

    sizes = np.asarray(segmentation.component_sizes, dtype=int)
    output = []
    for unit in units:
        ca = int(unit.component_a)
        cb = int(unit.component_b)
        if not (0 <= ca < len(sizes) and 0 <= cb < len(sizes)):
            raise ValueError("interface unit references an invalid component")
        if (
            sizes[ca] >= minimum_component_points
            and sizes[cb] >= minimum_component_points
        ):
            output.append(unit)
    return tuple(output)


def assess_grain_route_reliability(
    data: EBSDMap,
    segmentation: GrainSegmentation,
    grains: Sequence[Grain],
    *,
    phase_id: int,
    minimum_required_grains: int = 2,
) -> GrainRouteReliabilityAssessment:
    """Mandatory gate for grain-level experimental inference."""

    if minimum_required_grains < 1:
        raise ValueError("minimum_required_grains must be >= 1")

    phase_mask = data.indexed & (data.phase_id == int(phase_id))
    indexed_pixels = int(np.count_nonzero(phase_mask))
    retained_mask = phase_mask & (segmentation.grain_id >= 0)
    retained_pixels = int(np.count_nonzero(retained_mask))
    retained_grains = [
        grain for grain in grains if grain.phase_id == int(phase_id)
    ]
    grain_count = len(retained_grains)
    largest = max(
        (int(grain.size) for grain in retained_grains),
        default=0,
    )
    fraction = (
        float(retained_pixels / indexed_pixels)
        if indexed_pixels
        else 0.0
    )

    reasons: list[str] = []
    if indexed_pixels == 0:
        reasons.append("no_indexed_pixels")
    if retained_pixels == 0:
        reasons.append("no_retained_pixels")
    if grain_count == 0:
        reasons.append("no_retained_grains")
    if grain_count < minimum_required_grains:
        reasons.append("too_few_retained_grains")

    return GrainRouteReliabilityAssessment(
        phase_id=int(phase_id),
        allowed=not reasons,
        indexed_pixels=indexed_pixels,
        retained_pixels=retained_pixels,
        retained_pixel_fraction=fraction,
        retained_grains=grain_count,
        largest_retained_grain=largest,
        minimum_required_grains=int(minimum_required_grains),
        reason_codes=tuple(reasons),
    )


def _candidate_equivalent_quaternions(
    candidate: np.ndarray,
    symmetry_a: Sequence[np.ndarray],
    symmetry_b: Sequence[np.ndarray],
) -> np.ndarray:
    C = require_so3(
        candidate,
        tolerance=2.0e-8,
        name="interphase relation candidate",
    )
    SA = np.asarray(symmetry_a, dtype=float)
    SB = np.asarray(symmetry_b, dtype=float)

    equivalents = np.einsum(
        "sji,jk,tkl->stil",
        SA,
        C,
        SB,
        optimize=True,
    ).reshape(-1, 3, 3)
    # Deliberately keep the complete generated set.  Exact duplicates are
    # harmless under max(projective quaternion dot), and avoiding Python-level
    # O((|G_a||G_b|)^2) duplicate scans is substantially faster inside local
    # optimization.  No symmetry element is removed.
    return _rotation_quaternions(equivalents)


def relation_residuals_deg(
    observed_relations: np.ndarray,
    candidate: np.ndarray,
    symmetry_a: Sequence[np.ndarray],
    symmetry_b: Sequence[np.ndarray],
) -> np.ndarray:
    observed = np.asarray(observed_relations, dtype=float)
    if observed.ndim != 3 or observed.shape[1:] != (3, 3):
        raise ValueError(
            "observed_relations must have shape (N,3,3)"
        )
    if len(observed) == 0:
        return np.empty(0, dtype=float)
    q_observed = _rotation_quaternions(observed)
    q_equivalent = _candidate_equivalent_quaternions(
        candidate,
        symmetry_a,
        symmetry_b,
    )
    best_dot = np.max(
        np.abs(q_observed @ q_equivalent.T),
        axis=1,
    )
    return _angles_from_projective_dots(best_dot)


def relation_quotient_distance_deg(
    first: np.ndarray,
    second: np.ndarray,
    symmetry_a: Sequence[np.ndarray],
    symmetry_b: Sequence[np.ndarray],
) -> float:
    return float(
        relation_residuals_deg(
            np.asarray(first, dtype=float).reshape(1, 3, 3),
            second,
            symmetry_a,
            symmetry_b,
        )[0]
    )


def _huber(values: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(np.asarray(values, dtype=float))
    quadratic = np.minimum(absolute, delta)
    linear = absolute - quadratic
    return 0.5 * quadratic**2 + delta * linear


def _robust_objective(
    residuals_deg: np.ndarray,
    *,
    trim_fraction: float,
    huber_delta_deg: float,
) -> float:
    values = np.sort(np.asarray(residuals_deg, dtype=float))
    if len(values) == 0:
        return float("inf")
    keep = max(1, int(ceil(trim_fraction * len(values))))
    return float(np.mean(_huber(values[:keep], huber_delta_deg)))


def _deterministic_split(
    units: Sequence[InterfaceUnit],
    *,
    holdout_fraction: float = 0.30,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(units) < 4:
        return tuple(range(len(units))), tuple()

    ordering = sorted(
        range(len(units)),
        key=lambda index: hashlib.sha256(
            repr(units[index].key).encode("utf-8")
        ).digest(),
    )
    n_holdout = max(1, int(round(len(units) * holdout_fraction)))
    n_holdout = min(n_holdout, len(units) - 3)
    holdout = tuple(sorted(ordering[:n_holdout]))
    holdout_set = set(holdout)
    fit = tuple(
        index for index in range(len(units))
        if index not in holdout_set
    )
    return fit, holdout


def _relations_from_units(
    units: Sequence[InterfaceUnit],
    indices: Sequence[int],
) -> np.ndarray:
    return np.asarray(
        [units[index].relation_a_from_b for index in indices],
        dtype=float,
    )


def discover_interphase_relation(
    units: Sequence[InterfaceUnit],
    phase_a: EBSDPhase,
    phase_b: EBSDPhase,
    *,
    settings: PhaseForensicsSettings | None = None,
) -> InterphaseRelationResult:
    settings = PhaseForensicsSettings() if settings is None else settings
    units = tuple(
        unit
        for unit in units
        if unit.phase_a == phase_a.phase_id
        and unit.phase_b == phase_b.phase_id
    )
    if phase_a.phase_id >= phase_b.phase_id:
        raise ValueError(
            "discover_interphase_relation expects phase_a.id < phase_b.id"
        )

    if len(units) < settings.relation_minimum_interface_units:
        return InterphaseRelationResult(
            phase_a=phase_a.phase_id,
            phase_b=phase_b.phase_id,
            status="insufficient_interface_evidence",
            n_interface_units=len(units),
            n_fit_units=0,
            n_holdout_units=0,
            best=None,
            alternatives=tuple(),
            note=(
                "Too few independent orientation-component interface units "
                "for a fit/holdout relation test."
            ),
        )

    fit_indices, holdout_indices = _deterministic_split(units)
    if len(fit_indices) < 3 or len(holdout_indices) < 2:
        return InterphaseRelationResult(
            phase_a=phase_a.phase_id,
            phase_b=phase_b.phase_id,
            status="insufficient_interface_evidence",
            n_interface_units=len(units),
            n_fit_units=len(fit_indices),
            n_holdout_units=len(holdout_indices),
            best=None,
            alternatives=tuple(),
            note="Fit/holdout split is too small for independent validation.",
        )

    fit_relations = _relations_from_units(units, fit_indices)
    holdout_relations = _relations_from_units(units, holdout_indices)

    seed_count = min(settings.relation_seed_cap, len(fit_relations))
    seed_positions = np.unique(
        np.linspace(
            0,
            len(fit_relations) - 1,
            seed_count,
            dtype=int,
        )
    )

    seed_records: list[tuple[float, float, int, np.ndarray]] = []
    for rank, position in enumerate(seed_positions):
        candidate = fit_relations[int(position)]
        residuals = relation_residuals_deg(
            fit_relations,
            candidate,
            phase_a.proper_symmetry_cartesian,
            phase_b.proper_symmetry_cartesian,
        )
        objective = _robust_objective(
            residuals,
            trim_fraction=settings.relation_trim_fraction,
            huber_delta_deg=settings.relation_huber_delta_deg,
        )
        support = float(
            np.mean(
                residuals <= settings.relation_support_threshold_deg
            )
        )
        seed_records.append(
            (objective, -support, rank, candidate)
        )

    seed_records.sort(key=lambda item: (item[0], item[1], item[2]))
    refine = seed_records[: settings.relation_refine_seeds]

    radius = np.deg2rad(settings.relation_local_radius_deg)
    optimized: list[
        tuple[float, float, np.ndarray, bool, str]
    ] = []

    for _, _, _, seed in refine:
        cache: dict[tuple[float, float, float], float] = {}

        def objective(vector: np.ndarray) -> float:
            v = np.asarray(vector, dtype=float).reshape(3)
            key = tuple(float(x) for x in np.round(v, 10))
            if key in cache:
                return cache[key]
            candidate = Rotation.from_rotvec(v).as_matrix() @ seed
            residuals = relation_residuals_deg(
                fit_relations,
                candidate,
                phase_a.proper_symmetry_cartesian,
                phase_b.proper_symmetry_cartesian,
            )
            value = _robust_objective(
                residuals,
                trim_fraction=settings.relation_trim_fraction,
                huber_delta_deg=settings.relation_huber_delta_deg,
            )
            cache[key] = value
            return value

        result = minimize(
            objective,
            np.zeros(3, dtype=float),
            method="Powell",
            bounds=[(-radius, radius)] * 3,
            options={
                "maxiter": int(settings.relation_local_max_iterations),
                "xtol": 1.0e-6,
                "ftol": 1.0e-8,
                "disp": False,
            },
        )
        candidate = (
            Rotation.from_rotvec(
                np.asarray(result.x, dtype=float)
            ).as_matrix()
            @ seed
        )
        residuals = relation_residuals_deg(
            fit_relations,
            candidate,
            phase_a.proper_symmetry_cartesian,
            phase_b.proper_symmetry_cartesian,
        )
        value = _robust_objective(
            residuals,
            trim_fraction=settings.relation_trim_fraction,
            huber_delta_deg=settings.relation_huber_delta_deg,
        )
        support = float(
            np.mean(
                residuals <= settings.relation_support_threshold_deg
            )
        )
        optimized.append(
            (
                value,
                -support,
                candidate,
                bool(result.success),
                str(result.message),
            )
        )

    optimized.sort(key=lambda item: (item[0], item[1]))
    unique: list[
        tuple[float, float, np.ndarray, bool, str]
    ] = []
    for record in optimized:
        candidate = record[2]
        if any(
            relation_quotient_distance_deg(
                candidate,
                existing[2],
                phase_a.proper_symmetry_cartesian,
                phase_b.proper_symmetry_cartesian,
            )
            <= 0.25
            for existing in unique
        ):
            continue
        unique.append(record)

    if not unique:
        return InterphaseRelationResult(
            phase_a=phase_a.phase_id,
            phase_b=phase_b.phase_id,
            status="no_finite_relation_candidate",
            n_interface_units=len(units),
            n_fit_units=len(fit_indices),
            n_holdout_units=len(holdout_indices),
            best=None,
            alternatives=tuple(),
            note="No finite candidate survived exact quotient deduplication.",
        )

    rng = np.random.default_rng(settings.random_seed)
    candidates: list[InterphaseRelationCandidate] = []

    for index, record in enumerate(unique):
        fit_objective, _, candidate, success, message = record
        fit_residuals = relation_residuals_deg(
            fit_relations,
            candidate,
            phase_a.proper_symmetry_cartesian,
            phase_b.proper_symmetry_cartesian,
        )
        holdout_residuals = relation_residuals_deg(
            holdout_relations,
            candidate,
            phase_a.proper_symmetry_cartesian,
            phase_b.proper_symmetry_cartesian,
        )

        null_p: float | None = None
        if index == 0 and len(holdout_indices) >= 2:
            g_a = np.asarray(
                [units[i].g_a for i in holdout_indices],
                dtype=float,
            )
            g_b = np.asarray(
                [units[i].g_b for i in holdout_indices],
                dtype=float,
            )
            observed_stat = float(np.median(holdout_residuals))
            null_statistics = np.empty(
                settings.null_permutations,
                dtype=float,
            )
            for permutation in range(settings.null_permutations):
                order = rng.permutation(len(g_b))
                shuffled_relations = np.einsum(
                    "nji,njk->nik",
                    g_a,
                    g_b[order],
                    optimize=True,
                )
                shuffled_residuals = relation_residuals_deg(
                    shuffled_relations,
                    candidate,
                    phase_a.proper_symmetry_cartesian,
                    phase_b.proper_symmetry_cartesian,
                )
                null_statistics[permutation] = float(
                    np.median(shuffled_residuals)
                )
            null_p = float(
                (
                    1
                    + np.count_nonzero(
                        null_statistics <= observed_stat
                    )
                )
                / (settings.null_permutations + 1)
            )

        candidates.append(
            InterphaseRelationCandidate(
                R_phase_a_from_phase_b=candidate.copy(),
                fit_objective=float(fit_objective),
                fit_median_deg=float(np.median(fit_residuals)),
                fit_support_fraction=float(
                    np.mean(
                        fit_residuals
                        <= settings.relation_support_threshold_deg
                    )
                ),
                holdout_median_deg=float(
                    np.median(holdout_residuals)
                ),
                holdout_p90_deg=float(
                    np.quantile(holdout_residuals, 0.90)
                ),
                holdout_support_fraction=float(
                    np.mean(
                        holdout_residuals
                        <= settings.relation_support_threshold_deg
                    )
                ),
                null_empirical_p_value=null_p,
                optimizer_success=success,
                optimizer_message=message,
            )
        )

    best = candidates[0]
    concentrated = (
        best.holdout_median_deg is not None
        and best.holdout_median_deg
        <= settings.relation_maximum_holdout_median_deg
        and best.holdout_support_fraction is not None
        and best.holdout_support_fraction
        >= settings.relation_minimum_holdout_support
        and best.null_empirical_p_value is not None
        and best.null_empirical_p_value
        <= settings.null_significance_level
    )

    return InterphaseRelationResult(
        phase_a=phase_a.phase_id,
        phase_b=phase_b.phase_id,
        status=(
            "concentrated_interphase_relation"
            if concentrated
            else "weak_or_unconfirmed_interphase_relation"
        ),
        n_interface_units=len(units),
        n_fit_units=len(fit_indices),
        n_holdout_units=len(holdout_indices),
        best=best,
        alternatives=tuple(candidates[1:]),
        note=(
            "Relation direction is only coordinate bookkeeping "
            "R_phase_a_from_phase_b with phase_a < phase_b; no "
            "transformation direction is inferred."
        ),
    )


def run_phase_forensics(
    data: EBSDMap,
    phases: Mapping[int, EBSDPhase],
    neighbor_graph: NeighborGraph,
    *,
    settings: PhaseForensicsSettings | None = None,
) -> PhaseForensicsReport:
    settings = PhaseForensicsSettings() if settings is None else settings

    observed_phase_ids = sorted(
        int(x) for x in np.unique(data.phase_id[data.indexed])
    )
    undefined = sorted(set(observed_phase_ids).difference(phases))
    if undefined:
        raise ValueError(
            f"indexed map contains undefined phase IDs: {undefined}"
        )

    edge_angles = exact_same_phase_edge_angles(
        data,
        phases,
        neighbor_graph,
    )
    spatial = spatial_components(data, neighbor_graph)
    internal_survival_sizes = tuple(
        sorted(
            set(int(x) for x in settings.minimum_component_sizes)
            | {int(settings.reliability_component_min_points)}
        )
    )
    phase_records = phase_survival_records(
        data,
        phases,
        neighbor_graph,
        spatial,
        edge_angles,
        minimum_sizes=internal_survival_sizes,
    )
    raw_routes = raw_pair_route_evidence(
        data,
        neighbor_graph,
        spatial,
        minimum_interface_units=settings.relation_minimum_interface_units,
    )

    threshold_results = []
    warnings: list[str] = []

    for phase_record in phase_records:
        if phase_record.raw_indexed_pixels <= 0:
            continue
        for minimum in settings.minimum_component_sizes:
            if minimum <= 1:
                continue
            if phase_record.raw_components_ge_size.get(minimum, 0) == 0:
                warnings.append(
                    f"phase {phase_record.phase_id}: indexed pixels are "
                    f"present but no raw spatial component reaches {minimum} "
                    f"points; a minimum_points>={minimum} grain pipeline "
                    "would erase this phase before orientation analysis."
                )

    for threshold in settings.orientation_thresholds_deg:
        segmentation = orientation_components(
            data,
            neighbor_graph,
            edge_angles,
            threshold_deg=threshold,
        )
        survival = threshold_phase_survival(
            segmentation,
            observed_phase_ids,
            minimum_sizes=internal_survival_sizes,
        )
        raw_by_phase = {
            item.phase_id: item for item in phase_records
        }
        reliability = tuple(
            assess_phase_interface_reliability(
                raw_by_phase[item.phase_id],
                item,
                settings,
            )
            for item in survival
        )
        reliability_by_phase = {
            item.phase_id: item for item in reliability
        }
        for record in survival:
            raw_pixels = int(
                np.count_nonzero(
                    data.indexed & (data.phase_id == record.phase_id)
                )
            )
            if raw_pixels <= 0:
                continue
            for minimum in settings.minimum_component_sizes:
                if minimum <= 1:
                    continue
                if record.components_ge_size.get(minimum, 0) == 0:
                    warnings.append(
                        f"phase {record.phase_id}: at orientation threshold "
                        f"{threshold:g} deg, minimum_points>={minimum} would "
                        "retain zero orientation components despite indexed "
                        "pixels being present."
                    )
        means = component_mean_orientations(
            data,
            phases,
            segmentation,
            batch_size=settings.mean_batch_size,
        )
        all_units = interface_units(
            data,
            neighbor_graph,
            segmentation,
            means,
        )
        units = filter_interface_units_for_reliability(
            all_units,
            segmentation,
            minimum_component_points=(
                settings.reliability_component_min_points
            ),
        )
        routes = threshold_pair_route_evidence(
            raw_routes,
            units,
            minimum_interface_units=settings.relation_minimum_interface_units,
        )

        direct_relations = []
        for route in routes:
            gate_a = reliability_by_phase[route.phase_a]
            gate_b = reliability_by_phase[route.phase_b]
            if (
                not gate_a.interface_inference_allowed
                or not gate_b.interface_inference_allowed
            ):
                reasons = []
                if not gate_a.interface_inference_allowed:
                    reasons.append(
                        f"phase {route.phase_a}: "
                        + ",".join(gate_a.reason_codes)
                    )
                if not gate_b.interface_inference_allowed:
                    reasons.append(
                        f"phase {route.phase_b}: "
                        + ",".join(gate_b.reason_codes)
                    )
                direct_relations.append(
                    InterphaseRelationResult(
                        phase_a=route.phase_a,
                        phase_b=route.phase_b,
                        status="rejected_by_phase_reliability_gate",
                        n_interface_units=route.orientation_interface_units,
                        n_fit_units=0,
                        n_holdout_units=0,
                        best=None,
                        alternatives=tuple(),
                        note=(
                            "Direct interphase optimization was not run. "
                            "Evidence gate: " + " | ".join(reasons)
                        ),
                    )
                )
                continue
            if not route.direct_interface_route_possible:
                direct_relations.append(
                    InterphaseRelationResult(
                        phase_a=route.phase_a,
                        phase_b=route.phase_b,
                        status="insufficient_interface_evidence",
                        n_interface_units=route.orientation_interface_units,
                        n_fit_units=0,
                        n_holdout_units=0,
                        best=None,
                        alternatives=tuple(),
                        note=(
                            "Not enough independent orientation-component "
                            "interface units for direct relation discovery."
                        ),
                    )
                )
                continue
            pair_units = tuple(
                unit
                for unit in units
                if unit.phase_a == route.phase_a
                and unit.phase_b == route.phase_b
            )
            direct_relations.append(
                discover_interphase_relation(
                    pair_units,
                    phases[route.phase_a],
                    phases[route.phase_b],
                    settings=settings,
                )
            )

        threshold_results.append(
            ForensicThresholdResult(
                threshold_deg=float(threshold),
                segmentation=segmentation,
                phase_survival=survival,
                phase_reliability=reliability,
                pair_routes=routes,
                direct_relations=tuple(direct_relations),
            )
        )

    return PhaseForensicsReport(
        phase_survival=phase_records,
        raw_pair_routes=raw_routes,
        thresholds=tuple(threshold_results),
        warnings=tuple(warnings),
    )


def compact_phase_forensics_report(
    report: PhaseForensicsReport,
) -> dict[str, Any]:
    """Return the human-readable view with all large label arrays removed."""

    return {
        "phase_survival": report.phase_survival,
        "raw_pair_routes": report.raw_pair_routes,
        "thresholds": [
            {
                "threshold_deg": item.threshold_deg,
                "segmentation": {
                    "n_components": item.segmentation.n_components,
                    "threshold_deg": item.segmentation.threshold_deg,
                },
                "phase_survival": item.phase_survival,
                "phase_reliability": item.phase_reliability,
                "pair_routes": item.pair_routes,
                "direct_relations": item.direct_relations,
            }
            for item in report.thresholds
        ],
        "warnings": report.warnings,
    }


def phase_forensics_array_payload(
    report: PhaseForensicsReport,
) -> dict[str, np.ndarray]:
    """Return reproducible large arrays for compressed NPZ persistence."""

    payload: dict[str, np.ndarray] = {}
    for item in report.thresholds:
        token = (
            f"{item.threshold_deg:g}"
            .replace("-", "m")
            .replace(".", "p")
        )
        prefix = f"threshold_{token}"
        payload[f"{prefix}_component_id"] = np.asarray(
            item.segmentation.component_id,
            dtype=np.int64,
        )
        payload[f"{prefix}_component_sizes"] = np.asarray(
            item.segmentation.component_sizes,
            dtype=np.int64,
        )
        payload[f"{prefix}_component_phase_id"] = np.asarray(
            item.segmentation.component_phase_id,
            dtype=np.int64,
        )
    return payload


def assert_vectorized_disorientation_parity(
    data: EBSDMap,
    phases: Mapping[int, EBSDPhase],
    neighbor_graph: NeighborGraph,
    *,
    maximum_edges_per_phase: int = 32,
    tolerance_deg: float = 2.0e-7,
) -> None:
    """Cross-lock vectorized edge math to the frozen scalar implementation."""

    fast = exact_same_phase_edge_angles(data, phases, neighbor_graph)
    edges = np.asarray(neighbor_graph.edges, dtype=int)

    for phase_id, phase in phases.items():
        positions = np.flatnonzero(
            data.indexed[edges[:, 0]]
            & data.indexed[edges[:, 1]]
            & (data.phase_id[edges[:, 0]] == phase_id)
            & (data.phase_id[edges[:, 1]] == phase_id)
        )
        if len(positions) == 0:
            continue
        selected = positions[
            np.unique(
                np.linspace(
                    0,
                    len(positions) - 1,
                    min(maximum_edges_per_phase, len(positions)),
                    dtype=int,
                )
            )
        ]
        for position in selected:
            a, b = edges[int(position)]
            reference = minimum_disorientation(
                data.orientations[int(a)],
                data.orientations[int(b)],
                phase.proper_symmetry_cartesian,
            ).angle_deg
            if not np.isclose(
                fast[int(position)],
                reference,
                atol=tolerance_deg,
                rtol=0.0,
            ):
                raise AssertionError(
                    "vectorized disorientation differs from frozen scalar "
                    f"definition at edge {int(position)}: "
                    f"{fast[int(position)]} vs {reference}"
                )


def assert_component_segmentation_parity(
    data: EBSDMap,
    phases: Mapping[int, EBSDPhase],
    neighbor_graph: NeighborGraph,
    edge_angles_deg: np.ndarray,
    *,
    threshold_deg: float,
) -> None:
    """Cross-lock threshold components to frozen segment_grains(minimum=1)."""

    fast = orientation_components(
        data,
        neighbor_graph,
        edge_angles_deg,
        threshold_deg=threshold_deg,
    )
    reference = segment_grains(
        data,
        phases,
        threshold_deg=threshold_deg,
        neighbor_graph=neighbor_graph,
        minimum_points=1,
    )

    # Labels may be numbered differently.  Compare partitions by mapping each
    # fast label to exactly one reference label and vice versa.
    mask = data.indexed
    fast_labels = fast.component_id[mask]
    reference_labels = reference.grain_id[mask]
    pairs = np.column_stack((fast_labels, reference_labels))

    mapping_fast: dict[int, int] = {}
    mapping_reference: dict[int, int] = {}
    for fast_id, reference_id in pairs:
        fast_id = int(fast_id)
        reference_id = int(reference_id)
        if fast_id in mapping_fast and mapping_fast[fast_id] != reference_id:
            raise AssertionError("fast component split differs from reference")
        if (
            reference_id in mapping_reference
            and mapping_reference[reference_id] != fast_id
        ):
            raise AssertionError("fast component merge differs from reference")
        mapping_fast[fast_id] = reference_id
        mapping_reference[reference_id] = fast_id


def assert_vectorized_mean_parity(
    orientations: np.ndarray,
    symmetry: Sequence[np.ndarray],
    *,
    tolerance_deg: float = 2.0e-7,
) -> None:
    values = np.asarray(orientations, dtype=float)
    if len(values) == 0:
        raise ValueError("orientations must not be empty")
    fast = symmetry_aligned_mean_vectorized(values, symmetry)
    reference = symmetry_aligned_mean(values, symmetry)
    residual = minimum_disorientation(
        fast,
        reference,
        symmetry,
    ).angle_deg
    if residual > tolerance_deg:
        raise AssertionError(
            "vectorized symmetry-aligned mean differs from frozen reference: "
            f"{residual} deg"
        )
