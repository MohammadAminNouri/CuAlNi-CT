
from __future__ import annotations

"""Branch-resolved physical comparison of independently generated theory predictions.

This module is deliberately *downstream* of :mod:`theory_unified`.  It never
constructs a CT, Ball--James or PTMC prediction.  It converts already generated
branches to common physical quantities and performs one-to-one branch
assignment.  The assignment cost is only a combinatorial device; scientific
residuals are always reported separately and no theory score/winner is
constructed.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .orientation import OrientationService
from .orientation_kernel import (
    OrientationKernel,
    oriented_angle_deg,
    projective_angle_deg,
)
from .representation import CartesianConvention
from .physical_validation import cross_validate_ct_mallard_ball_james
from .theory_observable_contracts import (
    AMRankOneBridgeAudit,
    ShapeVectorRole,
    audit_smc_rank_one_bridge,
    infer_native_shape_vector_role,
)
from .theory_unified import (
    ComparisonRow,
    PredictionKind,
    TheoryKind,
    UnifiedTheoryReport,
    exact_symmetry_group_for_ct,
)

Array = np.ndarray


class MatchFamily(str, Enum):
    AM_INTERFACE = "austenite_martensite_interface"
    MM_TWIN = "martensite_martensite_twin"
    ORIENTATION = "orientation_relationship"


class MatchCompleteness(str, Enum):
    """Whether the compared rows expose all observables required by the family contract."""

    FULL = "full"
    PARTIAL = "partial"


@dataclass(frozen=True)
class AssignmentScales:
    """Numerical scales used only to make heterogeneous assignment costs dimensionless.

    They are *not* pass/fail tolerances and are never collapsed into a reported
    scientific score.
    """

    angle_deg: float = 1.0
    relative_magnitude: float = 0.05
    or_angle_deg: float = 1.0

    def __post_init__(self) -> None:
        if min(self.angle_deg, self.relative_magnitude, self.or_angle_deg) <= 0.0:
            raise ValueError("All assignment scales must be positive")


@dataclass(frozen=True)
class EquivalenceTolerances:
    """Explicit component-wise tolerances for numerical branch agreement.

    These values classify whether *reported physical observables* agree within
    a requested numerical tolerance.  They do not turn the assignment cost into
    a theory score and they do not imply a general equivalence theorem.
    """

    plane_angle_deg: float = 1.0e-4
    direction_angle_deg: float = 1.0e-4
    relative_magnitude: float = 1.0e-6
    rank_one_tensor_relative: float = 1.0e-6
    or_disorientation_deg: float = 1.0e-4

    def __post_init__(self) -> None:
        if min(
            self.plane_angle_deg,
            self.direction_angle_deg,
            self.relative_magnitude,
            self.rank_one_tensor_relative,
            self.or_disorientation_deg,
        ) <= 0.0:
            raise ValueError("All equivalence tolerances must be positive")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PhysicalResiduals:
    habit_plane_angle_deg: float | None = None
    shape_direction_projective_deg: float | None = None
    shape_direction_oriented_deg: float | None = None
    shape_magnitude_relative: float | None = None
    rank_one_tensor_relative: float | None = None

    twin_plane_angle_deg: float | None = None
    twin_direction_angle_deg: float | None = None
    shear_relative: float | None = None

    or_disorientation_deg: float | None = None

    def available(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, value in asdict(self).items()
            if value is not None
        )

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass(frozen=True)
class BranchMatch:
    family: MatchFamily
    left_theory: TheoryKind
    right_theory: TheoryKind
    left_row_id: str
    right_row_id: str
    left_branch: str
    right_branch: str
    residuals: PhysicalResiduals
    assignment_components: tuple[str, ...]
    required_components: tuple[str, ...]
    missing_required_components: tuple[str, ...]
    completeness: MatchCompleteness
    within_tolerance: dict[str, bool] = field(default_factory=dict)
    all_required_components_within_tolerance: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "left_theory": self.left_theory.value,
            "right_theory": self.right_theory.value,
            "left_row_id": self.left_row_id,
            "right_row_id": self.right_row_id,
            "left_branch": self.left_branch,
            "right_branch": self.right_branch,
            "residuals": self.residuals.to_dict(),
            "assignment_components": list(self.assignment_components),
            "required_components": list(self.required_components),
            "missing_required_components": list(self.missing_required_components),
            "completeness": self.completeness.value,
            "within_tolerance": dict(self.within_tolerance),
            "all_required_components_within_tolerance": (
                self.all_required_components_within_tolerance
            ),
        }


@dataclass(frozen=True)
class UnmatchedBranch:
    family: MatchFamily
    theory: TheoryKind
    row_id: str
    branch: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "theory": self.theory.value,
            "row_id": self.row_id,
            "branch": self.branch,
        }


@dataclass(frozen=True)
class ExperimentResidual:
    observation_row_id: str
    theory_row_id: str
    theory: TheoryKind
    theory_branch: str
    residuals: PhysicalResiduals
    uncertainty_normalized: dict[str, float] = field(default_factory=dict)
    comparable_components: tuple[str, ...] = ()
    unavailable_components: dict[str, str] = field(default_factory=dict)

    @property
    def comparison_status(self) -> str:
        if self.comparable_components:
            return "compared"
        if self.unavailable_components:
            return "relevant_but_not_comparable"
        return "no_shared_observable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_row_id": self.observation_row_id,
            "theory_row_id": self.theory_row_id,
            "theory": self.theory.value,
            "theory_branch": self.theory_branch,
            "residuals": self.residuals.to_dict(),
            "uncertainty_normalized": dict(self.uncertainty_normalized),
            "comparable_components": list(self.comparable_components),
            "unavailable_components": dict(self.unavailable_components),
            "comparison_status": self.comparison_status,
        }


@dataclass(frozen=True)
class EquivalenceReport:
    transformation_id: str
    matches: tuple[BranchMatch, ...]
    unmatched: tuple[UnmatchedBranch, ...]
    experiment_residuals: tuple[ExperimentResidual, ...]
    mm_independent_audit: dict[str, Any] | None
    classification_checks: dict[str, Any] = field(default_factory=dict)
    tolerances: EquivalenceTolerances = EquivalenceTolerances()
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def matches_for(
        self,
        *,
        family: MatchFamily | str | None = None,
        left: TheoryKind | str | None = None,
        right: TheoryKind | str | None = None,
    ) -> tuple[BranchMatch, ...]:
        family_value = None if family is None else MatchFamily(family)
        left_value = None if left is None else TheoryKind(left)
        right_value = None if right is None else TheoryKind(right)
        return tuple(
            item
            for item in self.matches
            if (family_value is None or item.family is family_value)
            and (left_value is None or item.left_theory is left_value)
            and (right_value is None or item.right_theory is right_value)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "transformation_id": self.transformation_id,
            "matches": [item.to_dict() for item in self.matches],
            "unmatched": [item.to_dict() for item in self.unmatched],
            "experiment_residuals": [
                item.to_dict() for item in self.experiment_residuals
            ],
            "mm_independent_audit": self.mm_independent_audit,
            "classification_checks": dict(self.classification_checks),
            "tolerances": self.tolerances.to_dict(),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }


def _unit(value: Any) -> Array | None:
    if value is None:
        return None
    vector = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-15:
        return None
    return vector / norm


def _relative_scalar(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    a = float(first)
    b = float(second)
    if not np.isfinite(a) or not np.isfinite(b):
        return None
    return abs(a - b) / max(abs(a), abs(b), 1.0e-15)


def _relative_tensor(first: Array | None, second: Array | None) -> float | None:
    if first is None or second is None:
        return None
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return None
    scale = max(float(np.linalg.norm(a, ord="fro")), float(np.linalg.norm(b, ord="fro")), 1.0e-15)
    return float(np.linalg.norm(a - b, ord="fro") / scale)


class PhysicalBranchMatcher:
    """Convert unified-theory rows into common physical observables and match them."""

    def __init__(self, project: Any, transformation_id: str) -> None:
        project.validate().assert_passed()
        self.project = project
        self.transformation_id = transformation_id
        self.transformation = project.transformation(transformation_id)
        self.parent = project.phase(self.transformation.parent_phase_id)
        self.product = project.phase(self.transformation.product_phase_id)

        tolerance = max(
            project.numerical_policy.representation,
            project.numerical_policy.algebraic,
        )
        parent_group = exact_symmetry_group_for_ct(
            list(self.parent.symmetry_matrices()),
            self.parent.lattice.metric(),
            tolerance=tolerance,
            label=f"parent phase {self.parent.phase_id!r}",
        )
        product_group = exact_symmetry_group_for_ct(
            list(self.product.symmetry_matrices()),
            self.product.lattice.metric(),
            tolerance=tolerance,
            label=f"product phase {self.product.phase_id!r}",
        )
        self.kernel = OrientationKernel(
            self.parent.lattice.metric(),
            self.product.lattice.metric(),
            parent_group,
            product_group,
            self.transformation.correspondence,
            algebraic_tolerance=max(tolerance, 1.0e-10),
        )
        self.orientation_service = OrientationService(project)
        self._am_stretch = self.kernel.kinematics_from_correspondence().stretch_A
        self._am_bridge_guard_tolerance = 10.0 * max(
            project.numerical_policy.exact_eigenvalue,
            project.numerical_policy.representation,
            project.numerical_policy.algebraic,
        )

    def _habit_normal(self, row: ComparisonRow) -> Array | None:
        direct = _unit(row.habit_normal_parent_cartesian)
        if direct is not None:
            return direct
        if row.habit_plane_parent_crystal is None:
            return None
        return self.kernel.plane_normal_cartesian(
            np.asarray(row.habit_plane_parent_crystal, dtype=float),
            phase="A",
        )

    def _stored_shape_vector(self, row: ComparisonRow) -> Array | None:
        """Physicalize the vector stored by the native theory row, unchanged."""

        if row.shape_vector_parent_cartesian is not None:
            value = np.asarray(row.shape_vector_parent_cartesian, dtype=float).reshape(3)
            if np.all(np.isfinite(value)) and float(np.linalg.norm(value)) > 1.0e-15:
                return value
        if row.shape_vector_parent_crystal is None:
            return None
        coeffs = np.asarray(row.shape_vector_parent_crystal, dtype=float).reshape(3)
        value = self.kernel.B_A @ coeffs
        if not np.all(np.isfinite(value)) or float(np.linalg.norm(value)) <= 1.0e-15:
            return None
        return value

    @staticmethod
    def _shape_vector_role(row: ComparisonRow) -> ShapeVectorRole:
        """Return the producer-defined native vector contract, fail-closed."""

        return infer_native_shape_vector_role(
            theory=row.theory,
            prediction_kind=row.prediction_kind,
            exact=row.exact,
            metadata=row.metadata,
        )

    def _ct_am_bridge_audit(self, row: ComparisonRow) -> AMRankOneBridgeAudit | None:
        if self._shape_vector_role(row) is not ShapeVectorRole.CT_SMC_IPS_D:
            return None
        if row.prediction_kind is not PredictionKind.CT_AM_HABIT or row.exact is not True:
            return None
        stored = self._stored_shape_vector(row)
        normal = self._habit_normal(row)
        if stored is None or normal is None:
            return None
        audit = audit_smc_rank_one_bridge(stored, normal, self._am_stretch)
        if audit.maximum_residual > self._am_bridge_guard_tolerance:
            raise AssertionError(
                "Exact CT A/M SMC-to-rank-one observable bridge failed its "
                "independent deformation audit: "
                f"max residual={audit.maximum_residual:.3e}, "
                f"guard={self._am_bridge_guard_tolerance:.3e}"
            )
        return audit

    def _am_rank_one_shape_vector(self, row: ComparisonRow) -> Array | None:
        """Return b for the common exact ``R F-I=b⊗n`` A/M observable.

        Native theory vectors are never overwritten. Exact CT SMC ``d`` is
        converted only at this comparison boundary and only after an independent
        deformation audit. Ball--James and true-IPS PTMC rows already expose
        ``b``. Approximate CT, dilated PTMC and unspecified experiment vectors
        remain N/A rather than being guessed into equivalence.
        """

        role = self._shape_vector_role(row)
        if role is ShapeVectorRole.CT_SMC_IPS_D:
            audit = self._ct_am_bridge_audit(row)
            return None if audit is None else audit.rank_one_shape_cartesian
        if role is ShapeVectorRole.RANK_ONE_PARENT_IDENTITY_B:
            return self._stored_shape_vector(row)
        return None

    def _am_rank_one_tensor(self, row: ComparisonRow) -> Array | None:
        shape = self._am_rank_one_shape_vector(row)
        normal = self._habit_normal(row)
        if shape is None or normal is None:
            return None
        return np.outer(shape, normal)

    def _twin_normal(self, row: ComparisonRow) -> Array | None:
        direct = _unit(row.twin_normal_parent_cartesian)
        if direct is not None:
            return direct
        if row.twin_plane_parent_crystal is None:
            return None
        return self.kernel.plane_normal_cartesian(
            np.asarray(row.twin_plane_parent_crystal, dtype=float),
            phase="A",
        )

    def _twin_direction(self, row: ComparisonRow) -> Array | None:
        direct = _unit(row.twin_direction_parent_cartesian)
        if direct is not None:
            return direct
        if row.twin_direction_parent_crystal is None:
            return None
        return self.kernel.direction_cartesian(
            np.asarray(row.twin_direction_parent_crystal, dtype=float),
            phase="A",
        )

    def _or_in_symmetric_metric_frame(self, row: ComparisonRow) -> Array | None:
        """Return a physical OR in the kernel's symmetric-metric frame.

        The current unified CT closing-gap and PTMC adapters intentionally
        publish ``or_parent_from_product`` in the PTCLab Cartesian embedding
        (x||a, c in xz).  The orientation kernel's symmetry operators live in
        the symmetric-metric frame.  Re-expression is therefore mandatory
        before symmetry-reduced disorientation is evaluated.

        Experimental rows accepted by this comparison contract must likewise
        supply OR matrices in the PTCLab embedding.  The Streamlit EBSD bridge
        converts its symmetric-metric pipeline OR explicitly before creating
        the ExperimentalObservation.
        """
        if row.or_parent_from_product is None:
            return None
        state = self.orientation_service.state_from_matrix(
            self.parent.phase_id,
            self.product.phase_id,
            np.asarray(row.or_parent_from_product, dtype=float),
            orientation_id=f"equivalence_{row.row_id}",
            reference_convention=CartesianConvention.PTCLAB_A_X_C_XZ,
            moving_convention=CartesianConvention.PTCLAB_A_X_C_XZ,
            transformation_id=self.transformation_id,
        )
        symmetric = self.orientation_service.reexpress(
            state,
            CartesianConvention.SYMMETRIC_METRIC,
            CartesianConvention.SYMMETRIC_METRIC,
        )
        return np.asarray(symmetric.R_reference_from_moving, dtype=float)

    @staticmethod
    def _mixed_reference_current_twin_direction_semantics(
        left: ComparisonRow,
        right: ComparisonRow,
    ) -> bool:
        """Whether a direct twin-direction comparison would mix configurations.

        CT transformation-twin directions and the current ExperimentalObservation
        twin-direction fields are parent-reference crystallographic objects.
        Ball--James/PTMC rows expose the current-configuration rank-one shear
        direction.  Those are not the same physical object before the appropriate
        variant stretch is applied.

        The independent CT/Mallard/Ball--James M/M audit performs the required
        push-forward for theory/theory validation.  Experimental parent-reference
        directions are therefore compared directly to CT, but are left N/A against
        BJ/PTMC until an explicit current-configuration experimental direction or
        branch-specific push-forward is supplied.
        """
        kinds = {left.prediction_kind, right.prediction_kind}
        reference_kinds = {PredictionKind.CT_MM_TWIN, PredictionKind.EXPERIMENT}
        current_kinds = {PredictionKind.BALL_JAMES_MM, PredictionKind.PTMC_HABIT}
        return bool(kinds & reference_kinds) and bool(kinds & current_kinds)

    def residuals(self, left: ComparisonRow, right: ComparisonRow) -> PhysicalResiduals:
        habit_left = self._habit_normal(left)
        habit_right = self._habit_normal(right)
        shape_vector_left = self._am_rank_one_shape_vector(left)
        shape_vector_right = self._am_rank_one_shape_vector(right)
        shape_left = _unit(shape_vector_left)
        shape_right = _unit(shape_vector_right)
        tensor_left = self._am_rank_one_tensor(left)
        tensor_right = self._am_rank_one_tensor(right)
        twin_normal_left = self._twin_normal(left)
        twin_normal_right = self._twin_normal(right)
        mixed_twin_semantics = self._mixed_reference_current_twin_direction_semantics(left, right)
        twin_dir_left = None if mixed_twin_semantics else self._twin_direction(left)
        twin_dir_right = None if mixed_twin_semantics else self._twin_direction(right)

        habit_angle = (
            projective_angle_deg(habit_left, habit_right)
            if habit_left is not None and habit_right is not None
            else None
        )
        shape_projective = (
            projective_angle_deg(shape_left, shape_right)
            if shape_left is not None and shape_right is not None
            else None
        )
        shape_oriented = (
            oriented_angle_deg(shape_left, shape_right)
            if shape_left is not None and shape_right is not None
            else None
        )
        twin_plane = (
            projective_angle_deg(twin_normal_left, twin_normal_right)
            if twin_normal_left is not None and twin_normal_right is not None
            else None
        )
        twin_direction = (
            projective_angle_deg(twin_dir_left, twin_dir_right)
            if twin_dir_left is not None and twin_dir_right is not None
            else None
        )

        or_residual = None
        left_or = self._or_in_symmetric_metric_frame(left)
        right_or = self._or_in_symmetric_metric_frame(right)
        if left_or is not None and right_or is not None:
            or_residual = self.kernel.disorientation(left_or, right_or).angle_deg

        return PhysicalResiduals(
            habit_plane_angle_deg=habit_angle,
            shape_direction_projective_deg=shape_projective,
            shape_direction_oriented_deg=shape_oriented,
            shape_magnitude_relative=_relative_scalar(
                None
                if shape_vector_left is None
                else float(np.linalg.norm(shape_vector_left)),
                None
                if shape_vector_right is None
                else float(np.linalg.norm(shape_vector_right)),
            ),
            rank_one_tensor_relative=_relative_tensor(tensor_left, tensor_right),
            twin_plane_angle_deg=twin_plane,
            twin_direction_angle_deg=twin_direction,
            shear_relative=_relative_scalar(
                left.shear_magnitude,
                right.shear_magnitude,
            ),
            or_disorientation_deg=or_residual,
        )

    @staticmethod
    def _required_components(family: MatchFamily) -> tuple[str, ...]:
        if family is MatchFamily.AM_INTERFACE:
            # Plane + complete rank-one jump tensor are the minimal full A/M contract.
            return ("habit_plane_angle_deg", "rank_one_tensor_relative")
        if family is MatchFamily.MM_TWIN:
            # A full M/M comparison requires the physical plane, direction and shear.
            return ("twin_plane_angle_deg", "twin_direction_angle_deg", "shear_relative")
        return ("or_disorientation_deg",)

    @staticmethod
    def _assignment_components(
        family: MatchFamily,
        residuals: PhysicalResiduals,
    ) -> tuple[str, ...]:
        if family is MatchFamily.AM_INTERFACE:
            preferred = ("habit_plane_angle_deg", "rank_one_tensor_relative")
        elif family is MatchFamily.MM_TWIN:
            preferred = (
                "twin_plane_angle_deg",
                "twin_direction_angle_deg",
                "shear_relative",
            )
        else:
            preferred = ("or_disorientation_deg",)
        data = residuals.to_dict()
        return tuple(name for name in preferred if data[name] is not None)

    @staticmethod
    def _agreement_flags(
        family: MatchFamily,
        residuals: PhysicalResiduals,
        tolerances: EquivalenceTolerances,
    ) -> dict[str, bool]:
        data = residuals.to_dict()
        thresholds = {
            "habit_plane_angle_deg": tolerances.plane_angle_deg,
            "shape_direction_projective_deg": tolerances.direction_angle_deg,
            "shape_magnitude_relative": tolerances.relative_magnitude,
            "rank_one_tensor_relative": tolerances.rank_one_tensor_relative,
            "twin_plane_angle_deg": tolerances.plane_angle_deg,
            "twin_direction_angle_deg": tolerances.direction_angle_deg,
            "shear_relative": tolerances.relative_magnitude,
            "or_disorientation_deg": tolerances.or_disorientation_deg,
        }
        components = PhysicalBranchMatcher._assignment_components(family, residuals)
        return {
            name: bool(float(data[name]) <= thresholds[name])
            for name in components
            if data[name] is not None
        }

    @staticmethod
    def _assignment_cost(
        family: MatchFamily,
        residuals: PhysicalResiduals,
        scales: AssignmentScales,
    ) -> float:
        components = PhysicalBranchMatcher._assignment_components(family, residuals)
        if not components:
            return 1.0e12
        data = residuals.to_dict()
        values = []
        for name in components:
            value = float(data[name])  # type: ignore[arg-type]
            if name.endswith("_deg"):
                denominator = (
                    scales.or_angle_deg
                    if name == "or_disorientation_deg"
                    else scales.angle_deg
                )
            else:
                denominator = scales.relative_magnitude
            values.append(value / denominator)
        # This number is only used by the assignment algorithm.  It is never
        # exported as a physical score.
        return float(sum(item * item for item in values) / len(values))

    @staticmethod
    def _rows(
        report: UnifiedTheoryReport,
        theory: TheoryKind,
        kinds: Iterable[PredictionKind],
    ) -> list[ComparisonRow]:
        allowed = set(kinds)
        return [
            row
            for row in report.rows
            if row.theory is theory and row.prediction_kind in allowed
        ]

    @staticmethod
    def _dedupe_ptmc_twins(rows: list[ComparisonRow]) -> list[ComparisonRow]:
        """PTMC habit branches may repeat one LIS twin; retain one row per twin relation."""
        unique: dict[tuple[Any, Any, Any], ComparisonRow] = {}
        no_key: list[ComparisonRow] = []
        for row in rows:
            key = (
                row.metadata.get("base_variant_index"),
                row.metadata.get("other_variant_index"),
                row.metadata.get("twin_branch"),
            )
            if key[1] is None or key[2] is None:
                no_key.append(row)
            else:
                unique.setdefault(key, row)
        return [*unique.values(), *no_key]

    def match_pair(
        self,
        left_rows: list[ComparisonRow],
        right_rows: list[ComparisonRow],
        *,
        family: MatchFamily,
        scales: AssignmentScales,
        tolerances: EquivalenceTolerances = EquivalenceTolerances(),
    ) -> tuple[list[BranchMatch], list[UnmatchedBranch]]:
        if not left_rows or not right_rows:
            unmatched = [
                UnmatchedBranch(family, row.theory, row.row_id, row.branch_label)
                for row in (*left_rows, *right_rows)
            ]
            return [], unmatched

        residual_grid: list[list[PhysicalResiduals]] = []
        cost = np.zeros((len(left_rows), len(right_rows)), dtype=float)
        for i, left in enumerate(left_rows):
            row_residuals = []
            for j, right in enumerate(right_rows):
                result = self.residuals(left, right)
                row_residuals.append(result)
                cost[i, j] = self._assignment_cost(family, result, scales)
            residual_grid.append(row_residuals)

        assigned_left, assigned_right = linear_sum_assignment(cost)
        matches: list[BranchMatch] = []
        used_left: set[int] = set()
        used_right: set[int] = set()
        for i, j in zip(assigned_left.tolist(), assigned_right.tolist(), strict=True):
            residuals = residual_grid[i][j]
            components = self._assignment_components(family, residuals)
            # A pair with no comparable physical observable is not a match.
            if not components or cost[i, j] >= 1.0e11:
                continue
            left = left_rows[i]
            right = right_rows[j]
            used_left.add(i)
            used_right.add(j)
            agreement = self._agreement_flags(family, residuals, tolerances)
            required = self._required_components(family)
            data = residuals.to_dict()
            missing = tuple(name for name in required if data[name] is None)
            completeness = (
                MatchCompleteness.FULL if not missing else MatchCompleteness.PARTIAL
            )
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

    @staticmethod
    def _declared_observable_overlap(
        observation: ComparisonRow,
        prediction: ComparisonRow,
    ) -> tuple[str, ...]:
        """Observable families explicitly present on both rows before frame semantics.

        This differs from :meth:`PhysicalResiduals.available`: a shared declared
        observable can be intentionally *not comparable* (for example a
        parent-reference experimental twin direction versus a Ball--James
        current-configuration shear direction).  Such a pair must be retained
        in the report as N/A with an explanation rather than silently dropped.
        """

        def any_present(row: ComparisonRow, names: tuple[str, ...]) -> bool:
            return any(getattr(row, name) is not None for name in names)

        families = {
            "orientation_relationship": ("or_parent_from_product",),
            "habit_plane": (
                "habit_normal_parent_cartesian",
                "habit_plane_parent_crystal",
            ),
            "twin_plane": (
                "twin_normal_parent_cartesian",
                "twin_plane_parent_crystal",
            ),
            "twin_direction": (
                "twin_direction_parent_cartesian",
                "twin_direction_parent_crystal",
            ),
            "shear_magnitude": ("shear_magnitude",),
            "shape_vector": (
                "shape_vector_parent_cartesian",
                "shape_vector_parent_crystal",
            ),
            "shape_magnitude": ("shape_vector_magnitude",),
        }
        return tuple(
            family
            for family, names in families.items()
            if any_present(observation, names) and any_present(prediction, names)
        )

    def _unavailable_experiment_components(
        self,
        observation: ComparisonRow,
        prediction: ComparisonRow,
        declared_overlap: tuple[str, ...],
        residuals: PhysicalResiduals,
    ) -> dict[str, str]:
        """Explain shared observables intentionally excluded from comparison."""

        unavailable: dict[str, str] = {}
        if (
            "twin_direction" in declared_overlap
            and residuals.twin_direction_angle_deg is None
            and self._mixed_reference_current_twin_direction_semantics(
                observation, prediction
            )
        ):
            unavailable["twin_direction_angle_deg"] = (
                "Experimental/CT twin direction is a parent-reference object, "
                "whereas Ball-James/PTMC exposes the current-configuration "
                "rank-one shear direction. No residual is formed without an "
                "explicit branch-specific push-forward or a current-configuration "
                "experimental direction."
            )
        if (
            "shape_vector" in declared_overlap
            and residuals.rank_one_tensor_relative is None
            and self._shape_vector_role(observation) is ShapeVectorRole.UNSPECIFIED
        ):
            unavailable["rank_one_tensor_relative"] = (
                "Experimental shape-vector semantics are unspecified. The comparison "
                "engine will not guess whether it is CT SMC d, a parent-identity "
                "rank-one b, or a dilated-plane vector."
            )
        return unavailable

    def experiment_residuals(
        self,
        report: UnifiedTheoryReport,
    ) -> tuple[ExperimentResidual, ...]:
        experiments = [
            row for row in report.rows if row.theory is TheoryKind.EXPERIMENT
        ]
        predictions = [
            row for row in report.rows if row.theory is not TheoryKind.EXPERIMENT
        ]
        output: list[ExperimentResidual] = []
        for observation in experiments:
            uncertainties = {
                key.removeprefix("uncertainty_"): float(value)
                for key, value in observation.residuals.items()
                if key.startswith("uncertainty_") and float(value) > 0.0
            }
            for prediction in predictions:
                declared_overlap = self._declared_observable_overlap(
                    observation, prediction
                )
                if not declared_overlap:
                    continue
                residuals = self.residuals(observation, prediction)
                unavailable = self._unavailable_experiment_components(
                    observation, prediction, declared_overlap, residuals
                )
                # Keep relevant pairs even when every shared observable is
                # intentionally N/A because its configurations are incompatible.
                # Silent omission would falsely look like "no experimental datum".
                comparable = residuals.available()
                if not comparable and not unavailable:
                    continue
                normalized: dict[str, float] = {}
                mapping = {
                    "or_disorientation_deg": "orientation_deg",
                    "habit_plane_angle_deg": "habit_plane_deg",
                    "twin_plane_angle_deg": "twin_plane_deg",
                    "twin_direction_angle_deg": "twin_direction_deg",
                    "shape_direction_oriented_deg": "shape_direction_deg",
                    "shear_relative": "shear_relative",
                    "shape_magnitude_relative": "shape_magnitude_relative",
                }
                data = residuals.to_dict()
                for residual_name, uncertainty_name in mapping.items():
                    value = data[residual_name]
                    sigma = uncertainties.get(uncertainty_name)
                    if value is not None and sigma is not None:
                        normalized[residual_name] = float(value) / sigma
                output.append(
                    ExperimentResidual(
                        observation_row_id=observation.row_id,
                        theory_row_id=prediction.row_id,
                        theory=prediction.theory,
                        theory_branch=prediction.branch_label,
                        residuals=residuals,
                        uncertainty_normalized=normalized,
                        comparable_components=comparable,
                        unavailable_components=unavailable,
                    )
                )
        return tuple(output)

    def build_report(
        self,
        report: UnifiedTheoryReport,
        *,
        scales: AssignmentScales = AssignmentScales(),
        tolerances: EquivalenceTolerances = EquivalenceTolerances(),
        include_independent_mm_audit: bool = True,
    ) -> EquivalenceReport:
        matches: list[BranchMatch] = []
        unmatched: list[UnmatchedBranch] = []
        warnings: list[str] = []

        comparisons = [
            (
                MatchFamily.AM_INTERFACE,
                TheoryKind.CAYRON_CT,
                (PredictionKind.CT_AM_HABIT,),
                TheoryKind.BALL_JAMES,
                (PredictionKind.BALL_JAMES_AM,),
            ),
            (
                MatchFamily.AM_INTERFACE,
                TheoryKind.CAYRON_CT,
                (PredictionKind.CT_AM_HABIT,),
                TheoryKind.PTMC,
                (PredictionKind.PTMC_HABIT,),
            ),
            (
                MatchFamily.MM_TWIN,
                TheoryKind.CAYRON_CT,
                (PredictionKind.CT_MM_TWIN,),
                TheoryKind.BALL_JAMES,
                (PredictionKind.BALL_JAMES_MM,),
            ),
            (
                MatchFamily.MM_TWIN,
                TheoryKind.CAYRON_CT,
                (PredictionKind.CT_MM_TWIN,),
                TheoryKind.PTMC,
                (PredictionKind.PTMC_HABIT,),
            ),
            (
                MatchFamily.ORIENTATION,
                TheoryKind.CAYRON_CT,
                (PredictionKind.CT_CLOSING_GAP_OR,),
                TheoryKind.PTMC,
                (PredictionKind.PTMC_HABIT,),
            ),
        ]

        for family, left_theory, left_kinds, right_theory, right_kinds in comparisons:
            left_rows = self._rows(report, left_theory, left_kinds)
            right_rows = self._rows(report, right_theory, right_kinds)
            if family is MatchFamily.AM_INTERFACE:
                left_rows = [row for row in left_rows if row.exact is True]
                right_rows = [row for row in right_rows if row.exact is True]
            if family is MatchFamily.MM_TWIN and right_theory is TheoryKind.PTMC:
                right_rows = self._dedupe_ptmc_twins(right_rows)
            pair_matches, pair_unmatched = self.match_pair(
                left_rows,
                right_rows,
                family=family,
                scales=scales,
                tolerances=tolerances,
            )
            matches.extend(pair_matches)
            unmatched.extend(pair_unmatched)

        independent_audit: dict[str, Any] | None = None
        if include_independent_mm_audit:
            try:
                tolerance = max(
                    self.project.numerical_policy.representation,
                    self.project.numerical_policy.algebraic,
                )
                parent_group = exact_symmetry_group_for_ct(
                    list(self.parent.symmetry_matrices()),
                    self.parent.lattice.metric(),
                    tolerance=tolerance,
                    label="parent",
                )
                product_group = exact_symmetry_group_for_ct(
                    list(self.product.symmetry_matrices()),
                    self.product.lattice.metric(),
                    tolerance=tolerance,
                    label="product",
                )
                audit = cross_validate_ct_mallard_ball_james(
                    self.parent.lattice.metric(),
                    self.product.lattice.metric(),
                    parent_group,
                    product_group,
                    self.transformation.correspondence,
                )
                independent_audit = {
                    "success": audit.success,
                    "relation_count": audit.relation_count,
                    "ct_coverage_complete": audit.ct_coverage_complete,
                    "classification_exact": audit.classification_exact,
                    "max_ct_geometry_angle_deg": audit.max_ct_geometry_angle_deg,
                    "max_ball_james_geometry_angle_deg": (
                        audit.max_ball_james_geometry_angle_deg
                    ),
                    "max_shear_relative_residual": audit.max_shear_relative_residual,
                    "max_rank_one_residual": audit.max_rank_one_residual,
                    "max_rotation_residual": audit.max_rotation_residual,
                }
            except (AssertionError, ValueError, np.linalg.LinAlgError) as exc:
                warnings.append(
                    "Independent CT/Mallard/Ball-James M/M audit was not "
                    f"evaluable for this state: {type(exc).__name__}: {exc}"
                )

        ct_am_rows = self._rows(
            report, TheoryKind.CAYRON_CT, (PredictionKind.CT_AM_HABIT,)
        )
        bj_am_rows = self._rows(
            report, TheoryKind.BALL_JAMES, (PredictionKind.BALL_JAMES_AM,)
        )
        ptmc_habit_rows = self._rows(
            report, TheoryKind.PTMC, (PredictionKind.PTMC_HABIT,)
        )
        ct_am_exact = bool(report.ct_report.analysis.exact_compatible)
        classification_checks = {
            "ct_am_exact_compatible": ct_am_exact,
            "ct_am_habit_branch_count": len(ct_am_rows),
            "ball_james_am_branch_count": len(bj_am_rows),
            "ct_vs_ball_james_am_existence_agreement": (
                ct_am_exact == bool(bj_am_rows)
            ),
            "ptmc_habit_branch_count": len(ptmc_habit_rows),
            "note": (
                "CT/Ball-James existence agreement compares independent A/M "
                "compatibility classifications. PTMC branch count is reported "
                "descriptively because PTMC also depends on the selected lattice-"
                "invariant-shear hypothesis."
            ),
        }

        return EquivalenceReport(
            transformation_id=self.transformation_id,
            matches=tuple(matches),
            unmatched=tuple(unmatched),
            experiment_residuals=self.experiment_residuals(report),
            mm_independent_audit=independent_audit,
            classification_checks=classification_checks,
            tolerances=tolerances,
            warnings=tuple(warnings),
            notes=(
                "Branch assignment is global one-to-one; its internal cost is not a scientific score.",
                "CT/Ball-James/PTMC native residuals remain separate from cross-theory physical residuals.",
                (
                    "CT and ExperimentalObservation twin-direction fields are parent-"
                    "reference objects, whereas Ball-James/PTMC rank-one shear directions "
                    "are current-configuration objects. Mixed reference/current direction "
                    "residuals are therefore N/A; the independent M/M audit performs the "
                    "required U_j push-forward for CT/Mallard/Ball-James validation."
                ),
                (
                    "Unified CT closing-gap and PTMC OR matrices are re-expressed from "
                    "the PTCLab Cartesian convention into the symmetric-metric frame "
                    "before symmetry-reduced disorientation is computed."
                ),
            ),
        )


def build_equivalence_report(
    project: Any,
    transformation_id: str,
    unified_report: UnifiedTheoryReport,
    *,
    scales: AssignmentScales = AssignmentScales(),
    tolerances: EquivalenceTolerances = EquivalenceTolerances(),
    include_independent_mm_audit: bool = True,
) -> EquivalenceReport:
    """Convenience entry point for the branch-resolved physical comparison."""
    return PhysicalBranchMatcher(project, transformation_id).build_report(
        unified_report,
        scales=scales,
        tolerances=tolerances,
        include_independent_mm_audit=include_independent_mm_audit,
    )


def double_shear_comparison_rows(report: Any) -> tuple[ComparisonRow, ...]:
    """Expose double-shear habit branches in the common physical-row contract.

    This adapter is intentionally downstream of the double-shear solver.  It
    does not alter the double-shear equations and does not add the rows to a
    :class:`UnifiedTheoryReport` automatically; callers decide explicitly when
    an extended-PTMC comparison is desired.
    """

    rows: list[ComparisonRow] = []
    for solution_index, solution in enumerate(report.solutions):
        for habit in solution.habit_connections:
            rows.append(
                ComparisonRow(
                    row_id=(
                        f"ptmc_double_shear_{solution_index}_"
                        f"{int(habit.branch):+d}"
                    ),
                    theory=TheoryKind.PTMC,
                    prediction_kind=PredictionKind.PTMC_HABIT,
                    branch_label=(
                        "double-shear PTMC "
                        f"f1={solution.first_parameter:.12g}, "
                        f"f2={solution.second_parameter:.12g}, "
                        f"habit branch {int(habit.branch):+d}"
                    ),
                    exact=bool(abs(float(report.dilatational_factor) - 1.0) <= 1.0e-12),
                    rotation_matrix=tuple(
                        tuple(float(x) for x in row)
                        for row in np.asarray(habit.rotation, dtype=float)
                    ),
                    rotation_role="double-shear PTMC habit rotation",
                    habit_normal_parent_cartesian=tuple(
                        float(x)
                        for x in np.asarray(
                            habit.habit_normal_parent_cartesian,
                            dtype=float,
                        )
                    ),
                    shape_vector_parent_cartesian=tuple(
                        float(x)
                        for x in np.asarray(
                            habit.shape_vector_parent_cartesian,
                            dtype=float,
                        )
                    ),
                    shape_vector_magnitude=float(habit.shape_vector_magnitude),
                    residuals={
                        "middle_stretch": float(solution.middle_stretch_residual),
                        "compatibility_determinant": float(solution.determinant_residual),
                        "rank_one": float(habit.residual),
                    },
                    metadata={
                        "ptmc_order": "double_shear",
                        "composition": solution.composition.value,
                        "first_parameter": float(solution.first_parameter),
                        "second_parameter": float(solution.second_parameter),
                        "dilatational_factor": float(report.dilatational_factor),
                        "true_invariant_plane": bool(
                            abs(float(report.dilatational_factor) - 1.0) <= 1.0e-12
                        ),
                    },
                    provenance="Explicit two-LIS PTMC compatibility",
                    notes=(
                        "No OR is reported by this generic adapter because the "
                        "two-LIS solver is intentionally independent of a specific "
                        "product-variant orientation construction."
                    ),
                )
            )
    return tuple(rows)
