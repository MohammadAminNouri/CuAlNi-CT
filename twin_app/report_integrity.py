from __future__ import annotations

"""Fail-closed structure and provenance checks on a finished twin-family report.

This does NOT independently solve PTMC, correspondence theory, or nonlinear
elasticity. The unchanged scientific certifier validates raw matrix identities
before this presentation boundary. Here we reject corrupt/ambiguous metadata,
missing family records and scientifically impossible display quantities.
"""

from math import isfinite, sqrt
from typing import Any, Iterable


class ReportIntegrityError(ValueError):
    """A report is unsafe to display or export as an internally coherent result."""


def _fail(where: str, why: str) -> None:
    raise ReportIntegrityError(f"{where}: {why}")


def _number(value: Any, where: str, *, nonnegative: bool = False) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReportIntegrityError(f"{where}: not a finite numerical value") from exc
    if not isfinite(v):
        _fail(where, "nonfinite numerical value")
    if nonnegative and v < 0:
        _fail(where, "negative magnitude or residual")
    return v


def _vector(values: Iterable[float], where: str, *, nonzero: bool = True) -> float:
    try:
        v = list(values)
    except TypeError as exc:
        raise ReportIntegrityError(f"{where}: expected three-vector") from exc
    if len(v) != 3:
        _fail(where, "expected exactly three components")
    norm = sqrt(sum(_number(x, where) ** 2 for x in v))
    if not isfinite(norm) or (nonzero and norm <= 1e-15):
        _fail(where, "zero or invalid physical vector")
    return norm


def validate_report_integrity(report: Any, *, tolerance: float = 1e-7) -> None:
    """Require a complete, unambiguous graph and finite, correctly scoped data.

    The tolerance only governs *structural* normalization and laminate fractions;
    it does not modify the backend's physical-compatibility tolerances.
    """
    if not (0 < tolerance <= 1e-4):
        raise ValueError("Invalid report-integrity tolerance")
    audit = report.audit
    count = int(audit.topology_variant_count)
    stretch_count = int(audit.stretch_variant_count)
    if count < 1 or stretch_count < 1:
        _fail("audit", "invalid variant counts")
    variants = tuple(report.correspondence_variants)
    indices = [int(v.variant_index) for v in variants]
    if len(variants) != count or sorted(indices) != list(range(1, count + 1)):
        _fail("correspondence", "variant IDs must be unique and cover the reported inventory")
    for v in variants:
        if v.stretch_variant_index is not None and not 1 <= int(v.stretch_variant_index) <= stretch_count:
            _fail(f"M{v.variant_index}", "stretch index is outside the reported U inventory")
        if v.stretch_mapping_residual is not None:
            _number(v.stretch_mapping_residual, f"M{v.variant_index} mapping residual", nonnegative=True)

    seen_family: set[str] = set()
    seen_pairs: set[tuple[int, int]] = set()
    for family in report.families:
        family_id = str(family.family_id)
        if not family_id or family_id in seen_family:
            _fail("families", "empty or duplicate family identifier")
        seen_family.add(family_id)
        declared: set[tuple[int, int]] = set()
        for pair in family.equivalent_pairs:
            if len(pair) != 2:
                _fail(family_id, "invalid correspondence-couple shape")
            i, j = sorted(map(int, pair))
            if not (1 <= i < j <= count):
                _fail(family_id, "invalid or self-correspondence couple")
            if (i, j) in seen_pairs or (i, j) in declared:
                _fail(family_id, "same correspondence couple appears in multiple locations")
            declared.add((i, j))
            seen_pairs.add((i, j))
        recorded = {(min(int(p.variant_i), int(p.variant_j)),
                     max(int(p.variant_i), int(p.variant_j))) for p in family.pair_records}
        if len(recorded) != len(family.pair_records) or recorded != declared:
            _fail(family_id, "computed pair records must match precisely the displayed couple inventory")
        for pair in family.pair_records:
            where = f"{family_id}:M{pair.variant_i}-M{pair.variant_j}"
            for branch in pair.constructions:
                branch_where = f"{where}:{branch.construction_id}"
                _number(branch.shear_magnitude, branch_where + " shear", nonnegative=True)
                _vector(branch.a_parent_cartesian, branch_where + " twin a")
                n_norm = _vector(branch.n_parent_cartesian, branch_where + " twin n")
                if abs(n_norm - 1.0) > tolerance:
                    _fail(branch_where, "twin reference normal is not unit length")
                _vector(branch.twin_plane_product_crystal, branch_where + " product K1")
                _vector(branch.shear_direction_product_crystal, branch_where + " product eta1")
                for name in ("rank_one_residual", "rotation_residual"):
                    _number(getattr(branch, name), branch_where + " " + name, nonnegative=True)
                for name in ("independent_outer_product_residual", "independent_shear_relative_residual",
                             "discrete_plane_angle_deg", "discrete_direction_angle_deg",
                             "discrete_shear_relative_residual"):
                    item = getattr(branch, name)
                    if item is not None:
                        _number(item, branch_where + " " + name, nonnegative=True)
                if branch.classification in {"Type I", "Type II", "Compound"} and "cross-locked" not in branch.classification_status.lower():
                    _fail(branch_where, "verified twin type is missing the independent cross-lock")
                for habit in branch.habit_solutions:
                    h_where = branch_where + f":habit{habit.habit_branch}"
                    lam = _number(habit.other_variant_volume_fraction, h_where + " lambda")
                    base = _number(habit.base_variant_volume_fraction, h_where + " base fraction")
                    if not (-tolerance <= lam <= 1 + tolerance and
                            -tolerance <= base <= 1 + tolerance and
                            abs(lam + base - 1.0) <= tolerance):
                        _fail(h_where, "laminate fractions must lie in [0,1] and sum to one")
                    m_norm = _vector(habit.habit_normal_parent_cartesian, h_where + " Cartesian m")
                    if abs(m_norm - 1.0) > tolerance:
                        _fail(h_where, "Cartesian habit normal is not unit length")
                    _vector(habit.shape_vector_parent_cartesian, h_where + " Cartesian b", nonzero=False)
                    _vector(habit.habit_plane_parent_crystal, h_where + " reciprocal m")
                    _vector(habit.shape_vector_parent_crystal, h_where + " direct b", nonzero=False)
                    for name in ("frame_plane_residual", "frame_shape_vector_residual",
                                 "rank_one_residual", "rotation_residual", "middle_stretch_residual"):
                        _number(getattr(habit, name), h_where + " " + name, nonnegative=True)
    for name in ("maximum_correspondence_to_stretch_residual",
                 "ptmc_maximum_residual", "ball_james_maximum_residual"):
        _number(getattr(audit, name), "audit " + name, nonnegative=True)
