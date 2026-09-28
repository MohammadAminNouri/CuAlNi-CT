from __future__ import annotations

"""Performance-hardened physical matching for the CT equivalence laboratory.

This module is deliberately downstream of :mod:`theory_equivalence` and does
not change any CT, Ball--James, PTMC, Mallard, or cofactor equations.  It adds
three things only:

* cached conversion of branch observables into the common physical frame;
* family-specific residual evaluation so unrelated observables are not
  recomputed for every candidate pair;
* exact/approximate separation for A/M equivalence: approximate CT CMC habit
  diagnostics are excluded from exact-equivalence assignment by construction;
* source-faithful A/M semantics inherited from ``PhysicalBranchMatcher``:
  theory-native vectors are converted to a common parent-identity rank-one
  observable only when their mathematical contract permits it.

The public residual definitions and tolerances remain those of
``theory_equivalence``.
"""

from dataclasses import replace
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .physical_validation import cross_validate_ct_mallard_ball_james
from .theory_equivalence import (
    AssignmentScales,
    BranchMatch,
    EquivalenceReport,
    EquivalenceTolerances,
    MatchCompleteness,
    MatchFamily,
    PhysicalBranchMatcher,
    PhysicalResiduals,
    UnmatchedBranch,
    _relative_scalar,
    _relative_tensor,
)
from .theory_unified import ComparisonRow, PredictionKind, TheoryKind, UnifiedTheoryReport
from .orientation_kernel import oriented_angle_deg, projective_angle_deg

Array = np.ndarray


class CachedPhysicalBranchMatcher(PhysicalBranchMatcher):
    """Physical matcher with row-local caches and family-specific evaluation.

    The scientific contract is unchanged from :class:`PhysicalBranchMatcher`.
    Caches are keyed by immutable ``row_id`` values inside one matcher instance.
    """

    def __init__(self, project: Any, transformation_id: str) -> None:
        super().__init__(project, transformation_id)
        self._habit_cache: dict[str, Array | None] = {}
        self._stored_shape_cache: dict[str, Array | None] = {}
        self._am_shape_cache: dict[str, Array | None] = {}
        self._am_rank_one_cache: dict[str, Array | None] = {}
        self._twin_normal_cache: dict[str, Array | None] = {}
        self._twin_direction_cache: dict[str, Array | None] = {}
        self._or_cache: dict[str, Array | None] = {}
        self._or_orbit_cache: dict[str, Array | None] = {}

    @staticmethod
    def _cached(
        cache: dict[str, Array | None],
        row: ComparisonRow,
        factory,
    ) -> Array | None:
        if row.row_id not in cache:
            value = factory()
            cache[row.row_id] = None if value is None else np.asarray(value, dtype=float)
        return cache[row.row_id]

    def _habit_normal(self, row: ComparisonRow) -> Array | None:
        return self._cached(self._habit_cache, row, lambda: super(CachedPhysicalBranchMatcher, self)._habit_normal(row))

    def _stored_shape_vector(self, row: ComparisonRow) -> Array | None:
        return self._cached(
            self._stored_shape_cache,
            row,
            lambda: super(CachedPhysicalBranchMatcher, self)._stored_shape_vector(row),
        )

    def _am_rank_one_shape_vector(self, row: ComparisonRow) -> Array | None:
        return self._cached(
            self._am_shape_cache,
            row,
            lambda: super(CachedPhysicalBranchMatcher, self)._am_rank_one_shape_vector(row),
        )

    def _am_rank_one_tensor(self, row: ComparisonRow) -> Array | None:
        return self._cached(
            self._am_rank_one_cache,
            row,
            lambda: super(CachedPhysicalBranchMatcher, self)._am_rank_one_tensor(row),
        )

    def _twin_normal(self, row: ComparisonRow) -> Array | None:
        return self._cached(self._twin_normal_cache, row, lambda: super(CachedPhysicalBranchMatcher, self)._twin_normal(row))

    def _twin_direction(self, row: ComparisonRow) -> Array | None:
        return self._cached(self._twin_direction_cache, row, lambda: super(CachedPhysicalBranchMatcher, self)._twin_direction(row))

    def _or_in_symmetric_metric_frame(self, row: ComparisonRow) -> Array | None:
        return self._cached(self._or_cache, row, lambda: super(CachedPhysicalBranchMatcher, self)._or_in_symmetric_metric_frame(row))

    @staticmethod
    def _dedupe_exact_or_rows(rows: list[ComparisonRow]) -> list[ComparisonRow]:
        """Collapse only numerically identical OR matrices.

        No symmetry-equivalence inference is performed here.  Distinct matrices
        remain distinct candidates even when their symmetry-reduced
        disorientation may later vanish.
        """
        unique: dict[bytes, ComparisonRow] = {}
        without_or: list[ComparisonRow] = []
        for row in rows:
            if row.or_parent_from_product is None:
                without_or.append(row)
                continue
            matrix = np.asarray(row.or_parent_from_product, dtype=float).reshape(3, 3)
            key = np.round(matrix, decimals=13).tobytes()
            unique.setdefault(key, row)
        return [*unique.values(), *without_or]

    def _or_orbit(self, row: ComparisonRow) -> Array | None:
        """All proper-symmetry representatives ``S_A R S_M`` for one OR."""
        if row.row_id in self._or_orbit_cache:
            return self._or_orbit_cache[row.row_id]
        R = self._or_in_symmetric_metric_frame(row)
        if R is None:
            self._or_orbit_cache[row.row_id] = None
            return None
        SA = np.stack(self.kernel.reference_proper_cartesian, axis=0)
        SM = np.stack(self.kernel.moving_proper_cartesian, axis=0)
        left = np.einsum("aij,jk->aik", SA, R)
        orbit = np.einsum("aij,bjk->abik", left, SM).reshape(-1, 3, 3)
        self._or_orbit_cache[row.row_id] = orbit
        return orbit

    @staticmethod
    def _angles_for_orbit_against_many(orbit: Array, right_rotations: Array) -> Array:
        """Vectorized principal SO(3) angle, minimized over one symmetry orbit."""
        right_t = np.swapaxes(right_rotations, 1, 2)
        delta = np.einsum("sij,rjk->srik", orbit, right_t)
        trace = np.trace(delta, axis1=-2, axis2=-1)
        cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        skew = 0.5 * np.stack(
            [
                delta[..., 2, 1] - delta[..., 1, 2],
                delta[..., 0, 2] - delta[..., 2, 0],
                delta[..., 1, 0] - delta[..., 0, 1],
            ],
            axis=-1,
        )
        sine = np.linalg.norm(skew, axis=-1)
        angles = np.degrees(np.arctan2(sine, cosine))
        return np.min(angles, axis=0)

    def _family_residuals(
        self,
        left: ComparisonRow,
        right: ComparisonRow,
        family: MatchFamily,
    ) -> PhysicalResiduals:
        if family is MatchFamily.AM_INTERFACE:
            h1 = self._habit_normal(left)
            h2 = self._habit_normal(right)
            # Compare the common macroscopic rank-one observable only after
            # theory-native vectors have been mapped into the same contract.
            # Exact CT uses the audited SMC->rank-one bridge; Ball--James and
            # true-IPS PTMC already expose b in RU-I=b⊗n. Approximate CT and
            # uniformly dilated PTMC rows intentionally return N/A here.
            s1 = self._am_rank_one_shape_vector(left)
            s2 = self._am_rank_one_shape_vector(right)
            t1 = self._am_rank_one_tensor(left)
            t2 = self._am_rank_one_tensor(right)
            d1 = None if s1 is None else s1 / np.linalg.norm(s1)
            d2 = None if s2 is None else s2 / np.linalg.norm(s2)
            m1 = None if s1 is None else float(np.linalg.norm(s1))
            m2 = None if s2 is None else float(np.linalg.norm(s2))
            return PhysicalResiduals(
                habit_plane_angle_deg=(
                    projective_angle_deg(h1, h2)
                    if h1 is not None and h2 is not None
                    else None
                ),
                shape_direction_projective_deg=(
                    projective_angle_deg(d1, d2)
                    if d1 is not None and d2 is not None
                    else None
                ),
                shape_direction_oriented_deg=(
                    oriented_angle_deg(d1, d2)
                    if d1 is not None and d2 is not None
                    else None
                ),
                shape_magnitude_relative=_relative_scalar(m1, m2),
                rank_one_tensor_relative=_relative_tensor(t1, t2),
            )

        if family is MatchFamily.MM_TWIN:
            n1 = self._twin_normal(left)
            n2 = self._twin_normal(right)
            mixed = self._mixed_reference_current_twin_direction_semantics(left, right)
            d1 = None if mixed else self._twin_direction(left)
            d2 = None if mixed else self._twin_direction(right)
            return PhysicalResiduals(
                twin_plane_angle_deg=(
                    projective_angle_deg(n1, n2)
                    if n1 is not None and n2 is not None
                    else None
                ),
                twin_direction_angle_deg=(
                    projective_angle_deg(d1, d2)
                    if d1 is not None and d2 is not None
                    else None
                ),
                shear_relative=_relative_scalar(left.shear_magnitude, right.shear_magnitude),
            )

        left_or = self._or_in_symmetric_metric_frame(left)
        right_or = self._or_in_symmetric_metric_frame(right)
        return PhysicalResiduals(
            or_disorientation_deg=(
                self.kernel.disorientation(left_or, right_or).angle_deg
                if left_or is not None and right_or is not None
                else None
            )
        )

    def residuals(self, left: ComparisonRow, right: ComparisonRow) -> PhysicalResiduals:
        """Full residual contract for experiment rows and direct callers, with caches."""
        return super().residuals(left, right)

    @staticmethod
    def _finalize_assignment(
        left_rows: list[ComparisonRow],
        right_rows: list[ComparisonRow],
        residual_grid: list[list[PhysicalResiduals]],
        cost: Array,
        *,
        family: MatchFamily,
        tolerances: EquivalenceTolerances,
    ) -> tuple[list[BranchMatch], list[UnmatchedBranch]]:
        assigned_left, assigned_right = linear_sum_assignment(cost)
        matches: list[BranchMatch] = []
        used_left: set[int] = set()
        used_right: set[int] = set()

        for i, j in zip(assigned_left.tolist(), assigned_right.tolist(), strict=True):
            residuals = residual_grid[i][j]
            components = PhysicalBranchMatcher._assignment_components(family, residuals)
            if not components or cost[i, j] >= 1.0e11:
                continue
            left = left_rows[i]
            right = right_rows[j]
            used_left.add(i)
            used_right.add(j)
            agreement = PhysicalBranchMatcher._agreement_flags(family, residuals, tolerances)
            required = PhysicalBranchMatcher._required_components(family)
            data = residuals.to_dict()
            missing = tuple(name for name in required if data[name] is None)
            completeness = MatchCompleteness.FULL if not missing else MatchCompleteness.PARTIAL
            full_agreement = (
                all(agreement.get(name, False) for name in required)
                if completeness is MatchCompleteness.FULL
                else None
            )
            matches.append(
                BranchMatch(
                    family=family,
                    left_theory=left.theory,
                    right_theory=right.theory,
                    left_row_id=left.row_id,
                    right_row_id=right.row_id,
                    left_branch=left.branch_label,
                    right_branch=right.branch_label,
                    residuals=residuals,
                    assignment_components=components,
                    required_components=required,
                    missing_required_components=missing,
                    completeness=completeness,
                    within_tolerance=agreement,
                    all_required_components_within_tolerance=full_agreement,
                )
            )

        unmatched = [
            UnmatchedBranch(family, row.theory, row.row_id, row.branch_label)
            for i, row in enumerate(left_rows)
            if i not in used_left
        ]
        unmatched.extend(
            UnmatchedBranch(family, row.theory, row.row_id, row.branch_label)
            for j, row in enumerate(right_rows)
            if j not in used_right
        )
        return matches, unmatched

    def match_pair(
        self,
        left_rows: list[ComparisonRow],
        right_rows: list[ComparisonRow],
        *,
        family: MatchFamily,
        scales: AssignmentScales,
        tolerances: EquivalenceTolerances = EquivalenceTolerances(),
    ) -> tuple[list[BranchMatch], list[UnmatchedBranch]]:
        if family is MatchFamily.ORIENTATION:
            left_rows = self._dedupe_exact_or_rows(left_rows)
            right_rows = self._dedupe_exact_or_rows(right_rows)

        if not left_rows or not right_rows:
            return [], [
                UnmatchedBranch(family, row.theory, row.row_id, row.branch_label)
                for row in (*left_rows, *right_rows)
            ]

        if family is MatchFamily.ORIENTATION:
            left_orbits = [self._or_orbit(row) for row in left_rows]
            right_rotations = [self._or_in_symmetric_metric_frame(row) for row in right_rows]
            cost = np.full((len(left_rows), len(right_rows)), 1.0e12, dtype=float)
            residual_grid = [
                [PhysicalResiduals() for _ in right_rows]
                for _ in left_rows
            ]
            valid_right = [i for i, matrix in enumerate(right_rotations) if matrix is not None]
            if valid_right:
                right_stack = np.stack([right_rotations[i] for i in valid_right], axis=0)
                for i, orbit in enumerate(left_orbits):
                    if orbit is None:
                        continue
                    angles = self._angles_for_orbit_against_many(orbit, right_stack)
                    for local_j, angle in enumerate(angles.tolist()):
                        j = valid_right[local_j]
                        residual = PhysicalResiduals(or_disorientation_deg=float(angle))
                        residual_grid[i][j] = residual
                        cost[i, j] = self._assignment_cost(family, residual, scales)
            return self._finalize_assignment(
                left_rows,
                right_rows,
                residual_grid,
                cost,
                family=family,
                tolerances=tolerances,
            )

        residual_grid: list[list[PhysicalResiduals]] = []
        cost = np.zeros((len(left_rows), len(right_rows)), dtype=float)
        for i, left in enumerate(left_rows):
            row_residuals = []
            for j, right in enumerate(right_rows):
                result = self._family_residuals(left, right, family)
                row_residuals.append(result)
                cost[i, j] = self._assignment_cost(family, result, scales)
            residual_grid.append(row_residuals)

        return self._finalize_assignment(
            left_rows,
            right_rows,
            residual_grid,
            cost,
            family=family,
            tolerances=tolerances,
        )


def build_equivalence_report_optimized(
    project: Any,
    transformation_id: str,
    unified_report: UnifiedTheoryReport,
    *,
    scales: AssignmentScales = AssignmentScales(),
    tolerances: EquivalenceTolerances = EquivalenceTolerances(),
) -> EquivalenceReport:
    """Build the exact-equivalence report without rerunning the independent M/M audit."""
    return CachedPhysicalBranchMatcher(project, transformation_id).build_report(
        unified_report,
        scales=scales,
        tolerances=tolerances,
        include_independent_mm_audit=False,
    )


def independent_mm_audit(project: Any, transformation_id: str) -> dict[str, Any]:
    """Run the authoritative CT ↔ Mallard ↔ Ball--James M/M validation once.

    This is intentionally independent of assignment tolerances and PTMC controls,
    so callers can cache it solely by the calculated crystallographic state.
    """
    matcher = CachedPhysicalBranchMatcher(project, transformation_id)
    audit = cross_validate_ct_mallard_ball_james(
        matcher.parent.lattice.metric(),
        matcher.product.lattice.metric(),
        list(matcher.kernel.G_A),
        list(matcher.kernel.G_M),
        matcher.transformation.correspondence,
    )
    return {
        "success": bool(audit.success),
        "relation_count": int(audit.relation_count),
        "ct_coverage_complete": bool(audit.ct_coverage_complete),
        "classification_exact": bool(audit.classification_exact),
        "max_ct_geometry_angle_deg": float(audit.max_ct_geometry_angle_deg),
        "max_ball_james_geometry_angle_deg": float(audit.max_ball_james_geometry_angle_deg),
        "max_shear_relative_residual": float(audit.max_shear_relative_residual),
        "max_rank_one_residual": float(audit.max_rank_one_residual),
        "max_rotation_residual": float(audit.max_rotation_residual),
    }


def attach_independent_mm_audit(
    report: EquivalenceReport,
    audit: dict[str, Any] | None,
) -> EquivalenceReport:
    """Return an immutable report with a separately cached M/M audit attached."""
    return replace(report, mm_independent_audit=audit)
