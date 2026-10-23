from __future__ import annotations

"""Symmetry-orbit modes, derived from calculated stretches and twin branches.

There is no alloy-name or literature-table logic here. Operator/inverse-family
records remain science provenance, but never define a displayed physical mode.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping

import numpy as np
import sympy as sp

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.lattice import metric_sqrt
from cualni_cryst.project_io import project_from_dict
from cualni_cryst.stretch import stretch_from_metrics
from .scientific_models import TwinFamilyReport, VariantPairRecord, PairTwinConstruction
from .mode_type_status import supported_classical_type


class ModeClassificationError(ValueError):
    """The mode presentation cannot be certified from this computed state."""


@dataclass(frozen=True)
class ModeEntry:
    pair: VariantPairRecord
    construction: PairTwinConstruction
    operator_family: str

    @property
    def evidence_category(self) -> str:
        # Provenance only; never infer a classical twin subtype.
        from .twin_evidence_status_2026_10_22 import category_for_construction
        return category_for_construction(self.construction)


@dataclass(frozen=True)
class PhysicalTwinMode:
    mode_id: str
    stretch_pairs: tuple[tuple[int, int], ...]
    entries: tuple[ModeEntry, ...]
    shear: float
    types: tuple[str, ...]

    def by_type(self, twin_type: str) -> tuple[ModeEntry, ...]:
        return tuple(x for x in self.entries if _display_type(x.construction) == twin_type)

    def by_evidence_category(self, category: str) -> tuple[ModeEntry, ...]:
        # Separate parent-symmetry exclusion from unresolved evidence.
        return tuple(x for x in self.entries if x.evidence_category == category)

    def evidence_counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for entry in self.entries:
            category = entry.evidence_category
            result[category] = result.get(category, 0) + 1
        return result


@dataclass(frozen=True)
class ModeCatalogue:
    modes: tuple[PhysicalTwinMode, ...]
    incompatible_pair_orbits: tuple[tuple[tuple[int, int], ...], ...]
    correspondence_pair_count: int
    stretch_pair_count: int
    mapped_stretch_count: int


def _display_type(branch: PairTwinConstruction) -> str:
    if branch.classification in ("Type I", "Type II", "Compound"):
        if not supported_classical_type(branch.classification, branch.classification_status):
            raise ModeClassificationError("A claimed twin type lacks direct cross-lock or proved symmetry-transport evidence")
        return branch.classification
    return "Unresolved"


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1.e-15))


def _stretch_variants_from_report(report: TwinFamilyReport, parent_metric: np.ndarray, product_metric: np.ndarray) -> dict[int, np.ndarray]:
    # Reconstruct from exact per-variant C and both measured/entered metrics.
    # Never use a material-specific tensor or load any benchmark answer.
    stretches: dict[int, np.ndarray] = {}
    for record in report.correspondence_variants:
        if record.stretch_variant_index is None:
            continue
        C = Correspondence(sp.Matrix([[sp.Rational(x) for x in row] for row in record.correspondence_M_from_A]))
        U = np.asarray(stretch_from_metrics(parent_metric, product_metric, C), float)
        idx = int(record.stretch_variant_index)
        if idx in stretches and _relative_error(stretches[idx], U) > 2.e-6:
            raise ModeClassificationError(f"Correspondence variants assigned to U{idx} disagree metrically")
        stretches.setdefault(idx, U)
    if len(stretches) != report.audit.stretch_variant_count:
        raise ModeClassificationError("Incomplete metric stretch inventory; cannot identify physical modes")
    return stretches


def _symmetry_permutations(stretches: dict[int, np.ndarray], parent: Any, metric: np.ndarray, tolerance: float) -> tuple[tuple[int, ...], ...]:
    B = np.asarray(metric_sqrt(metric), float)
    Binv = np.linalg.inv(B)
    ids = tuple(sorted(stretches))
    if ids != tuple(range(1, len(ids) + 1)):
        raise ModeClassificationError("Stretch IDs must be contiguous and one-based")
    # Uniqueness depends on actual separation, NOT on a fixed 1e-9 gap.
    orbit_gaps = [
        _relative_error(stretches[i], stretches[j])
        for p, i in enumerate(ids)
        for j in ids[p + 1:]
    ]
    if orbit_gaps:
        orbit_minimum_gap = min(orbit_gaps)
        if not np.isfinite(orbit_minimum_gap) or orbit_minimum_gap <= 128.0 * np.finfo(float).eps:
            raise ModeClassificationError(
                "UNRESOLVED_MODE_SYMMETRY: distinct stretch variants below "
                f"reliable binary64 resolution; gap={orbit_minimum_gap:.3e}"
            )
        permutation_tolerance = min(tolerance, orbit_minimum_gap / 8.0)
    else:
        permutation_tolerance = tolerance
    result: set[tuple[int, ...]] = set()
    for lattice_operation in parent.symmetry_matrices():
        Q = B @ np.asarray(lattice_operation, float) @ Binv
        if _relative_error(Q.T @ Q, np.eye(3)) > tolerance:
            raise ModeClassificationError("Declared parent symmetry is not metric preserving")
        perm: list[int] = []
        for idx in ids:
            target = Q @ stretches[idx] @ Q.T
            errors = sorted((_relative_error(target, U), other) for other, U in stretches.items())
            if errors[0][0] > permutation_tolerance:
                raise ModeClassificationError(
                    "Parent symmetry cannot uniquely identify the transformed "
                    "stretch at the resolvable orbit separation"
                )
            if len(errors) > 1 and errors[1][0] <= permutation_tolerance:
                raise ModeClassificationError("Multiple distinct stretch variants satisfy symmetry matching")
            perm.append(errors[0][1])
        if len(set(perm)) != len(ids):
            raise ModeClassificationError("Nonbijective parent symmetry action on stretch variants")
        result.add(tuple(perm))
    if not result:
        raise ModeClassificationError("No validated parent symmetry operations available")
    return tuple(sorted(result))


def _orbits(variant_count: int, permutations: tuple[tuple[int, ...], ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    remaining = set(combinations(range(1, variant_count + 1), 2))
    orbits = []
    while remaining:
        seed = min(remaining)
        orbit = {tuple(sorted((perm[seed[0] - 1], perm[seed[1] - 1]))) for perm in permutations}
        if not orbit.issubset(remaining):
            raise ModeClassificationError("Symmetry-pair orbits overlap or omit a calculated state")
        remaining -= orbit
        orbits.append(tuple(sorted(orbit)))
    return tuple(orbits)


def build_mode_catalogue(
    report: TwinFamilyReport,
    project_payload: Mapping[str, object],
    *,
    metric_tolerance: float = 2.e-6,
    shear_tolerance: float = 2.e-5,
) -> ModeCatalogue:
    """Generate physically twinnable pair orbits and retain all provenance.

    1. Recover U_j from EXACT correspondence matrices C_i and two metrics.
    2. Compute action of the parent point group on U_j by Q U_j Q.T.
    3. Orbit-partition unordered stretch-variant pairs, not operator families.
    4. Include a mode only if an exact rank-one branch exists for its pair orbit.
       Keep all incompatible orbits in the audit, never falsely call them twins.
    5. Preserve every verified Type I/II/Compound and every unresolved record.
    """
    if not (0 < metric_tolerance < 1.e-3 and 0 < shear_tolerance < 1.e-3):
        raise ValueError("Invalid mode-validation tolerances")
    state = project_from_dict(dict(project_payload), source="mode_catalogue")
    project = state.project
    t = project.transformation(report.transformation_id)
    parent = project.phase(t.parent_phase_id)
    product = project.phase(t.product_phase_id)
    Ma = np.asarray(parent.lattice.metric(), float)
    Mm = np.asarray(product.lattice.metric(), float)
    stretches = _stretch_variants_from_report(report, Ma, Mm)
    perms = _symmetry_permutations(stretches, parent, Ma, metric_tolerance)
    orbits = _orbits(len(stretches), perms)

    # A correspondence couple may map to a stretch couple shared by several
    # correspondence representatives. All source IDs and branches are retained.
    by_stretch: dict[tuple[int, int], list[ModeEntry]] = defaultdict(list)
    correspondence_pairs = set()
    for family in report.families:
        for pair in family.pair_records:
            ident = tuple(sorted((int(pair.variant_i), int(pair.variant_j))))
            if ident in correspondence_pairs:
                raise ModeClassificationError(f"Duplicate correspondence couple M{ident[0]}-M{ident[1]}")
            correspondence_pairs.add(ident)
            if pair.stretch_i is None or pair.stretch_j is None or pair.stretch_i == pair.stretch_j:
                if pair.constructions:
                    raise ModeClassificationError("A nonmapped or identical-stretch couple claims a twin")
                continue
            key = tuple(sorted((int(pair.stretch_i), int(pair.stretch_j))))
            for b in pair.constructions:
                by_stretch[key].append(ModeEntry(pair, b, family.family_id))

    modes = []
    incompatible = []
    for orbit in orbits:
        populated = [bool(by_stretch.get(pair)) for pair in orbit]
        if not any(populated):
            incompatible.append(orbit)
            continue
        if not all(populated):
            raise ModeClassificationError(
                "Symmetry-equivalent stretch pairs disagree about rank-one compatibility; "
                "investigate variant mapping and numerical conditioning"
            )
        entries = tuple(sorted((record for pair in orbit for record in by_stretch[pair]),
                               key=lambda e: (min(e.pair.variant_i, e.pair.variant_j),
                                              max(e.pair.variant_i, e.pair.variant_j), e.construction.branch,
                                              e.construction.construction_id)))
        shears = [float(e.construction.shear_magnitude) for e in entries]
        if not all(np.isfinite(s) and s > 0 for s in shears):
            raise ModeClassificationError("Nonfinite or nonpositive twin shear")
        nominal = float(np.median(shears))
        if any(abs(s - nominal) > shear_tolerance * max(1., nominal) for s in shears):
            raise ModeClassificationError("A putative symmetry orbit contains different physical shears")
        kinds = tuple(k for k in ("Compound", "Type I", "Type II", "Unresolved")
                      if any(_display_type(e.construction) == k for e in entries))
        modes.append((orbit, entries, nominal, kinds))

    # Scientific mode labels are generated from stable orbit order, NEVER from
    # a specific alloy's reference shears or paper's mode A/B/C/D names.
    output = tuple(PhysicalTwinMode(f"Mode {i}", orbit, entries, shear, types)
                   for i, (orbit, entries, shear, types) in enumerate(modes, 1))
    # An independent combinatorial completeness guard: n stretch variants
    # must generate exactly n(n-1)/2 unordered pairs. A pair classified
    # as both Type I and Type II is STILL one unordered variant pair.
    all_pairs = {pair for mode in output for pair in mode.stretch_pairs}
    non_twinning = {pair for orbit in incompatible for pair in orbit}
    required = len(stretches) * (len(stretches) - 1) // 2
    if (all_pairs & non_twinning or
            len(all_pairs | non_twinning) != required or
            sum(len(mode.stretch_pairs) for mode in output) +
            sum(len(orbit) for orbit in incompatible) != required):
        raise ModeClassificationError(
            "Mode and non-twinning orbits do not partition all unordered stretch pairs"
        )
    return ModeCatalogue(output, tuple(incompatible), len(correspondence_pairs),
                         sum(len(o) for o in orbits), len(stretches))
