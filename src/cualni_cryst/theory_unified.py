from __future__ import annotations

"""Unified, branch-preserving comparison layer for CT, Ball--James, PTMC, experiment.

This module is an orchestrator only.  It does not reimplement any theory
equations and does not allow one theory backend to call another.  All
predictors consume the same ProjectState transformation; theory-specific
auxiliary hypotheses (for example a PTMC slip system or a CT natural OR) are
carried explicitly and never folded into the shared crystallographic input.

Rows preserve every discrete branch.  Missing observables remain ``None``
(rendered as N/A by :meth:`UnifiedTheoryReport.table_rows`) rather than being
invented.  Residuals remain named quantities with their native units; there is
deliberately no scalar "winner score".
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import sympy as sp

from .ball_james_adapter import BallJamesAdapter, BallJamesReport
from .ct import (
    CTAMResult,
    analyze_austenite_martensite,
    ct_supercompatibility_residual,
    ips_shear_from_habit_plane,
)
from .ct_orientation import CayronOrientationAdapter
from .group_theory import correspondence_groupoid, validate_group
from .lattice import metric_norm
from .ptmc_adapter import (
    PTMCAdapter,
    PTMCReport,
    PTMCSlipSystemInput,
    PTMCTwinPairInput,
    PTMCTwinPlaneInput,
)
from .twinning_ct import CTTwin, twins_from_operator

Array = np.ndarray
Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


class TheoryKind(str, Enum):
    CAYRON_CT = "cayron_ct"
    BALL_JAMES = "ball_james"
    PTMC = "ptmc"
    EXPERIMENT = "experiment"


class PredictionKind(str, Enum):
    CT_AM_HABIT = "ct_am_habit"
    CT_AM_DEGENERACY = "ct_am_degeneracy"
    CT_MM_TWIN = "ct_mm_twin"
    CT_CLOSING_GAP_OR = "ct_closing_gap_or"
    CT_SUPERCOMPATIBILITY = "ct_supercompatibility"
    BALL_JAMES_AM = "ball_james_am"
    BALL_JAMES_MM = "ball_james_mm"
    PTMC_HABIT = "ptmc_habit"
    PTMC_CONTINUUM = "ptmc_continuum"
    PTMC_PARAMETER_DIAGNOSTIC = "ptmc_parameter_diagnostic"
    EXPERIMENT = "experiment"


class PTMCMode(str, Enum):
    NONE = "none"
    ALL_TWINNING = "all_twinning"
    TWIN_PLANE = "twin_plane"
    TWIN_PAIR = "twin_pair"
    SLIP = "slip"


@dataclass(frozen=True)
class ExperimentalObservation:
    """Generic experiment row.

    This is deliberately convention-explicit and theory-neutral.  The EBSD
    milestone can later populate the same object after convention-aware
    orientation processing.
    """

    observation_id: str
    label: str = ""
    or_parent_from_product: Matrix3 | None = None
    habit_plane_parent_crystal: Vector3 | None = None
    habit_normal_parent_cartesian: Vector3 | None = None
    twin_plane_parent_crystal: Vector3 | None = None
    twin_plane_product_crystal: Vector3 | None = None
    twin_normal_parent_cartesian: Vector3 | None = None
    twin_direction_parent_crystal: Vector3 | None = None
    twin_direction_parent_cartesian: Vector3 | None = None
    shear_magnitude: float | None = None
    shape_vector_parent_crystal: Vector3 | None = None
    shape_vector_parent_cartesian: Vector3 | None = None
    shape_vector_magnitude: float | None = None
    uncertainty: dict[str, float] = field(default_factory=dict)
    provenance: str = "user_supplied_experiment"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.observation_id.strip():
            raise ValueError("observation_id must be non-empty")
        for name in ("shear_magnitude", "shape_vector_magnitude"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and nonnegative")
        for key, value in self.uncertainty.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Uncertainty {key!r} must be finite and nonnegative")


@dataclass(frozen=True)
class ComparisonRow:
    row_id: str
    theory: TheoryKind
    prediction_kind: PredictionKind
    branch_label: str
    exact: bool | None = None

    or_parent_from_product: Matrix3 | None = None
    rotation_matrix: Matrix3 | None = None
    rotation_role: str = ""

    habit_plane_parent_crystal: Vector3 | None = None
    habit_normal_parent_cartesian: Vector3 | None = None

    twin_plane_parent_crystal: Vector3 | None = None
    twin_plane_product_crystal: Vector3 | None = None
    twin_normal_parent_cartesian: Vector3 | None = None
    twin_direction_parent_crystal: Vector3 | None = None
    twin_direction_parent_cartesian: Vector3 | None = None

    shear_magnitude: float | None = None

    shape_vector_parent_crystal: Vector3 | None = None
    shape_vector_parent_cartesian: Vector3 | None = None
    shape_vector_magnitude: float | None = None

    residuals: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.row_id.strip():
            raise ValueError("row_id must be non-empty")
        if not self.branch_label.strip():
            raise ValueError("branch_label must be non-empty")
        for key, value in self.residuals.items():
            if not np.isfinite(value):
                raise ValueError(f"Residual {key!r} must be finite")
        for name in ("shear_magnitude", "shape_vector_magnitude"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class UnifiedTheoryReport:
    transformation_id: str
    parent_phase_id: str
    product_phase_id: str
    rows: tuple[ComparisonRow, ...]
    ct_report: CTAMResult
    ball_james_report: BallJamesReport
    ptmc_report: PTMCReport | None
    warnings: tuple[str, ...]
    notes: tuple[str, ...]

    def rows_for(self, theory: TheoryKind | str) -> tuple[ComparisonRow, ...]:
        selected = TheoryKind(theory)
        return tuple(row for row in self.rows if row.theory is selected)

    def table_rows(self, *, na: str = "N/A") -> list[dict[str, Any]]:
        """Return branch-preserving table rows with explicit N/A values.

        Residuals remain a named dictionary because degrees, dimensionless
        algebraic residuals and cofactor margins must never be merged into one
        score.
        """

        def show(value: Any) -> Any:
            if value is None:
                return na
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, tuple):
                return list(value)
            return value

        return [
            {
                "theory": row.theory.value,
                "kind": row.prediction_kind.value,
                "branch": row.branch_label,
                "exact": show(row.exact),
                "OR_parent_from_product": show(row.or_parent_from_product),
                "rotation": show(row.rotation_matrix),
                "rotation_role": row.rotation_role or na,
                "habit_plane_parent": show(row.habit_plane_parent_crystal),
                "twin_plane_parent": show(row.twin_plane_parent_crystal),
                "twin_plane_product": show(row.twin_plane_product_crystal),
                "twin_direction_parent": show(row.twin_direction_parent_crystal),
                "shear": show(row.shear_magnitude),
                "shape_vector_parent": show(row.shape_vector_parent_crystal),
                "shape_magnitude": show(row.shape_vector_magnitude),
                "residuals": dict(row.residuals) if row.residuals else na,
                "provenance": row.provenance or na,
            }
            for row in self.rows
        ]

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {str(key): convert(item) for key, item in value.items()}
            if hasattr(value, "__dataclass_fields__"):
                return convert(asdict(value))
            return value

        return {
            "transformation_id": self.transformation_id,
            "parent_phase_id": self.parent_phase_id,
            "product_phase_id": self.product_phase_id,
            "rows": [convert(row) for row in self.rows],
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "raw_reports": {
                "ct": convert(self.ct_report),
                "ball_james": convert(self.ball_james_report),
                "ptmc": None
                if self.ptmc_report is None
                else self.ptmc_report.to_dict(),
            },
        }


def _vector3(value: Any) -> Vector3:
    array = np.asarray(value, dtype=float).reshape(3)
    return tuple(float(x) for x in array)


def _matrix3(value: Any) -> Matrix3:
    array = np.asarray(value, dtype=float).reshape(3, 3)
    return tuple(tuple(float(x) for x in row) for row in array)  # type: ignore[return-value]


def _unit(value: Any) -> Array:
    array = np.asarray(value, dtype=float).reshape(3)
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-15:
        raise ValueError("Cannot normalize a zero vector")
    return array / norm


def _native_residuals(**values: float | None) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in values.items()
        if value is not None and np.isfinite(value)
    }


def exact_symmetry_group_for_ct(
    matrices: tuple[np.ndarray, ...] | list[np.ndarray],
    metric: np.ndarray,
    *,
    tolerance: float,
    label: str,
    max_denominator: int = 1_000_000,
) -> list[sp.Matrix]:
    """Recover an exact crystallographic group from ProjectState float storage.

    ``ProjectState`` intentionally freezes symmetry matrices as finite floats
    for generic numerical services.  Cayron's finite-group topology is
    different: subgroup/coset/double-coset membership must be exact.  Feeding
    SymPy ``Float(1.0)`` matrices directly into that engine is invalid because
    exact ``Integer(1)`` identity/group keys no longer compare as the same
    symbolic objects.

    Crystallographic direct-coordinate symmetry matrices are lattice
    automorphisms and therefore have exact rational coordinates in the stored
    lattice basis for the supported project workflow.  Each stored coefficient
    is reconstructed to the nearest bounded rational only when that
    reconstruction lies within the project's representation tolerance.  The
    resulting set is then required to form an exact finite group and to
    preserve the supplied metric numerically.

    Nothing is silently coerced: if exact recovery is ambiguous or fails group
    closure, CT topology is refused with a diagnostic rather than computed from
    approximate group membership.
    """

    if tolerance <= 0.0:
        raise ValueError("symmetry exactification tolerance must be positive")
    if max_denominator < 1:
        raise ValueError("max_denominator must be positive")
    if not matrices:
        raise ValueError(f"{label} symmetry group is empty")

    M = np.asarray(metric, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(M)):
        raise ValueError(f"{label} metric contains non-finite values")

    exact: list[sp.Matrix] = []
    for matrix_index, matrix in enumerate(matrices):
        raw = np.asarray(matrix, dtype=float)
        if raw.shape != (3, 3) or not np.all(np.isfinite(raw)):
            raise ValueError(
                f"{label} symmetry[{matrix_index}] must be a finite 3x3 matrix"
            )

        recovered: list[sp.Expr] = []
        for value in raw.reshape(-1):
            candidate = sp.Rational(str(float(value))).limit_denominator(
                max_denominator
            )
            error = abs(float(candidate) - float(value))
            scale = max(abs(float(value)), 1.0)
            if error > tolerance * scale:
                raise ValueError(
                    f"{label} symmetry[{matrix_index}] coefficient "
                    f"{float(value):.16g} cannot be recovered as a bounded exact "
                    f"rational within tolerance {tolerance:.3e}"
                )
            recovered.append(candidate)

        exact_matrix = sp.Matrix(3, 3, recovered)
        exact.append(exact_matrix)

    keys = [tuple(sp.simplify(value) for value in matrix) for matrix in exact]
    if len(keys) != len(set(keys)):
        raise ValueError(
            f"{label} symmetry exactification collapsed distinct stored operations"
        )

    try:
        validate_group(exact)
    except ValueError as exc:
        raise ValueError(
            f"{label} stored symmetry operators cannot be recovered as an exact "
            "finite crystallographic group; CT subgroup/coset topology requires "
            "exact group operations"
        ) from exc

    scale = max(float(np.linalg.norm(M)), 1.0)
    for matrix_index, operation in enumerate(exact):
        G = np.asarray(operation, dtype=float)
        residual = float(np.linalg.norm(G.T @ M @ G - M) / scale)
        if residual > tolerance:
            raise ValueError(
                f"{label} exactified symmetry[{matrix_index}] does not preserve "
                f"the supplied metric; residual={residual:.3e}"
            )

    return exact


class TheoryComparisonAdapter:
    """Run CT, Ball--James and PTMC on one frozen ProjectState transformation."""

    def __init__(self, project: Any, transformation_id: str) -> None:
        project.validate().assert_passed()
        self.project = project
        self.transformation_id = transformation_id
        self.transformation = project.transformation(transformation_id)
        self.parent = project.phase(self.transformation.parent_phase_id)
        self.product = project.phase(self.transformation.product_phase_id)

    def _ct_rows(
        self,
        *,
        natural_orientation: Any | None,
        include_closing_gap: bool,
        include_supercompatibility: bool,
    ) -> tuple[CTAMResult, list[ComparisonRow]]:
        M_a = self.parent.lattice.metric()
        M_m = self.product.lattice.metric()
        correspondence = self.transformation.correspondence
        tol = self.project.numerical_policy.exact_eigenvalue

        am = analyze_austenite_martensite(M_a, M_m, correspondence, tol=tol)
        rows: list[ComparisonRow] = []

        habit_planes: list[tuple[np.ndarray, bool, str, float]] = []
        if am.exact_habit_planes:
            for index, plane in enumerate(am.exact_habit_planes):
                habit_planes.append(
                    (
                        np.asarray(plane, dtype=float),
                        True,
                        f"exact_{index}",
                        float(am.analysis.nearest_zero_residual),
                    )
                )
        elif am.approximate.candidate_planes:
            for index, plane in enumerate(am.approximate.candidate_planes):
                habit_planes.append(
                    (
                        np.asarray(plane, dtype=float),
                        False,
                        f"approximate_{index}",
                        float(am.approximate.residual),
                    )
                )

        for plane, exact, label, residual in habit_planes:
            d_a = ips_shear_from_habit_plane(plane, M_a, M_m, correspondence)
            rows.append(
                ComparisonRow(
                    row_id=f"ct_am_habit_{label}",
                    theory=TheoryKind.CAYRON_CT,
                    prediction_kind=PredictionKind.CT_AM_HABIT,
                    branch_label=f"CT A/M habit {label}",
                    exact=exact,
                    habit_plane_parent_crystal=_vector3(plane),
                    shear_magnitude=float(metric_norm(d_a, M_a)),
                    shape_vector_parent_crystal=_vector3(d_a),
                    shape_vector_magnitude=float(metric_norm(d_a, M_a)),
                    residuals=_native_residuals(
                        cmc_nearest_zero_dimensionless=residual
                    ),
                    metadata={
                        "cmc_degeneracy_order": am.analysis.degeneracy_order,
                        "cmc_inertia": am.analysis.inertia,
                        "approximate_admissible_signature": (
                            am.approximate.admissible_signature
                        ),
                    },
                    provenance="Cayron CMC/SMC",
                    notes=(
                        "Approximate rows are nearest-zero diagnostics, not exact CT "
                        "compatibility predictions."
                        if not exact
                        else ""
                    ),
                )
            )

        if not habit_planes:
            rows.append(
                ComparisonRow(
                    row_id="ct_am_degeneracy",
                    theory=TheoryKind.CAYRON_CT,
                    prediction_kind=PredictionKind.CT_AM_DEGENERACY,
                    branch_label="CT A/M compatibility state",
                    exact=bool(am.analysis.exact_compatible),
                    residuals=_native_residuals(
                        cmc_nearest_zero_dimensionless=(
                            am.analysis.nearest_zero_residual
                        )
                    ),
                    metadata={
                        "cmc_degeneracy_order": am.analysis.degeneracy_order,
                        "cmc_inertia": am.analysis.inertia,
                        "reason": am.analysis.reason,
                    },
                    provenance="Cayron CMC",
                    notes=(
                        "No finite habit-plane list exists for third-order "
                        "degeneracy, or no admissible exact/approximate plane exists."
                    ),
                )
            )

        symmetry_tolerance = max(
            self.project.numerical_policy.representation,
            self.project.numerical_policy.algebraic,
        )
        parent_exact_symmetry = exact_symmetry_group_for_ct(
            list(self.parent.symmetry_matrices()),
            M_a,
            tolerance=symmetry_tolerance,
            label=f"parent phase {self.parent.phase_id!r}",
        )
        product_exact_symmetry = exact_symmetry_group_for_ct(
            list(self.product.symmetry_matrices()),
            M_m,
            tolerance=symmetry_tolerance,
            label=f"product phase {self.product.phase_id!r}",
        )
        groupoid = correspondence_groupoid(
            parent_exact_symmetry,
            product_exact_symmetry,
            correspondence,
        )
        orientation_adapter = (
            CayronOrientationAdapter(
                self._orientation_service(), self.transformation_id
            )
            if include_closing_gap
            else None
        )

        all_twins: list[tuple[int, int, CTTwin]] = []
        for operator_index, operator in enumerate(groupoid.operators):
            operator_twins = twins_from_operator(operator, M_a, M_m, correspondence)
            for twin_index, twin in enumerate(operator_twins):
                all_twins.append((operator_index, twin_index, twin))
                rows.append(
                    ComparisonRow(
                        row_id=f"ct_twin_{operator_index}_{twin_index}",
                        theory=TheoryKind.CAYRON_CT,
                        prediction_kind=PredictionKind.CT_MM_TWIN,
                        branch_label=(
                            f"CT operator {operator_index} twin {twin_index} "
                            f"Type-{twin.kind}"
                        ),
                        exact=True,
                        twin_plane_parent_crystal=_vector3(twin.plane_a),
                        twin_direction_parent_crystal=_vector3(twin.direction_a),
                        shear_magnitude=float(abs(twin.shear)),
                        metadata={
                            "operator_index": operator_index,
                            "twin_index": twin_index,
                            "twin_kind": twin.kind,
                            "rational_element": twin.rational_element,
                            "plane_product_crystal": _vector3(twin.plane_m),
                            "direction_product_crystal": _vector3(twin.direction_m),
                        },
                        provenance="Cayron exact Type-I/II transformation twin",
                    )
                )

                if orientation_adapter is not None:
                    report = orientation_adapter.closing_gap_from_twin(
                        twin,
                        natural_orientation=natural_orientation,
                        orientation_id_prefix=(
                            f"compare_ct_op{operator_index}_twin{twin_index}"
                        ),
                    )
                    for candidate in report.candidates:
                        rows.append(
                            ComparisonRow(
                                row_id=(
                                    f"ct_or_{operator_index}_{twin_index}_"
                                    f"{candidate.index}"
                                ),
                                theory=TheoryKind.CAYRON_CT,
                                prediction_kind=PredictionKind.CT_CLOSING_GAP_OR,
                                branch_label=(
                                    f"CT closing-gap op {operator_index}, "
                                    f"twin {twin_index}, candidate {candidate.index}"
                                ),
                                exact=True,
                                or_parent_from_product=_matrix3(
                                    candidate.orientation.R_reference_from_moving
                                ),
                                rotation_matrix=_matrix3(
                                    candidate.orientation.R_reference_from_moving
                                ),
                                rotation_role="Cayron closing-gap OR parent_from_product",
                                twin_plane_parent_crystal=_vector3(twin.plane_a),
                                twin_direction_parent_crystal=_vector3(
                                    twin.direction_a
                                ),
                                shear_magnitude=float(abs(twin.shear)),
                                residuals=_native_residuals(
                                    correspondence_plane_deg=(
                                        report.correspondence_plane_residual_deg
                                    ),
                                    correspondence_direction_deg=(
                                        report.correspondence_direction_residual_deg
                                    ),
                                    intercorrespondence=(
                                        report.intercorrespondence_residual
                                    ),
                                    direction_parallelism_deg=(
                                        candidate.direction_parallelism_residual_deg
                                    ),
                                    plane_parallelism_deg=(
                                        candidate.plane_parallelism_residual_deg
                                    ),
                                    rotation=candidate.rotation_residual,
                                    natural_or_disorientation_deg=(
                                        candidate.symmetry_reduced_deviation_from_natural_deg
                                    ),
                                ),
                                metadata={
                                    "operator_index": operator_index,
                                    "twin_index": twin_index,
                                    "twin_kind": twin.kind,
                                    "candidate_index": candidate.index,
                                    "selected_within_twin": (
                                        report.selected_candidate_index
                                        == candidate.index
                                    ),
                                    "selection_tie_indices": (
                                        report.selection_tie_indices
                                    ),
                                },
                                provenance="Cayron exact closing-gap construction",
                            )
                        )

        if include_supercompatibility and am.exact_habit_planes:
            for habit_index, habit in enumerate(am.exact_habit_planes):
                d_a = ips_shear_from_habit_plane(habit, M_a, M_m, correspondence)
                for operator_index, twin_index, twin in all_twins:
                    residual = ct_supercompatibility_residual(
                        habit,
                        d_a,
                        twin.plane_a,
                        twin.direction_a,
                        twin.shear,
                        M_a,
                    )
                    rows.append(
                        ComparisonRow(
                            row_id=(
                                f"ct_super_{habit_index}_{operator_index}_{twin_index}"
                            ),
                            theory=TheoryKind.CAYRON_CT,
                            prediction_kind=PredictionKind.CT_SUPERCOMPATIBILITY,
                            branch_label=(
                                f"CT supercompatibility habit {habit_index}, "
                                f"operator {operator_index}, twin {twin_index}"
                            ),
                            exact=True,
                            habit_plane_parent_crystal=_vector3(habit),
                            twin_plane_parent_crystal=_vector3(twin.plane_a),
                            twin_direction_parent_crystal=_vector3(twin.direction_a),
                            shear_magnitude=float(abs(twin.shear)),
                            shape_vector_parent_crystal=_vector3(d_a),
                            shape_vector_magnitude=float(metric_norm(d_a, M_a)),
                            residuals=_native_residuals(
                                ct_supercompatibility_dimensionless=residual
                            ),
                            metadata={
                                "habit_index": habit_index,
                                "operator_index": operator_index,
                                "twin_index": twin_index,
                                "twin_kind": twin.kind,
                            },
                            provenance="Cayron A/M/M shear-shear compatibility",
                        )
                    )

        return am, rows

    def _orientation_service(self) -> Any:
        from .orientation import OrientationService

        return OrientationService(self.project)

    def _ball_james_rows(
        self, *, fraction_samples: int
    ) -> tuple[BallJamesReport, list[ComparisonRow]]:
        report = BallJamesAdapter(self.project, self.transformation_id).analyze(
            fraction_samples=fraction_samples
        )
        rows: list[ComparisonRow] = []

        for branch in report.austenite_martensite_branches:
            rows.append(
                ComparisonRow(
                    row_id=(f"bj_am_{branch.variant_index}_{branch.branch}"),
                    theory=TheoryKind.BALL_JAMES,
                    prediction_kind=PredictionKind.BALL_JAMES_AM,
                    branch_label=(
                        f"Ball-James A/M variant {branch.variant_index}, "
                        f"branch {branch.branch:+d}"
                    ),
                    exact=True,
                    rotation_matrix=_matrix3(branch.rotation),
                    rotation_role=(
                        "rank-one habit rotation in parent metric-whitened frame; "
                        "not silently relabelled as an OR"
                    ),
                    habit_plane_parent_crystal=_vector3(
                        branch.habit_plane_parent_crystal
                    ),
                    habit_normal_parent_cartesian=_vector3(
                        branch.habit_normal_parent_cartesian
                    ),
                    shape_vector_parent_crystal=_vector3(
                        branch.shape_strain_parent_crystal
                    ),
                    shape_vector_parent_cartesian=_vector3(branch.shape_strain),
                    shape_vector_magnitude=float(np.linalg.norm(branch.shape_strain)),
                    residuals=_native_residuals(
                        rank_one=branch.rank_one_residual,
                        rotation=branch.rotation_residual,
                    ),
                    metadata={
                        "variant_index": branch.variant_index,
                        "branch": branch.branch,
                    },
                    provenance="Ball-James single-variant rank-one compatibility",
                )
            )

        for index, branch in enumerate(report.martensite_twin_branches):
            cofactor = branch.cofactor
            rows.append(
                ComparisonRow(
                    row_id=(
                        f"bj_mm_{branch.base_variant_index}_"
                        f"{branch.other_variant_index}_{branch.branch}_{index}"
                    ),
                    theory=TheoryKind.BALL_JAMES,
                    prediction_kind=PredictionKind.BALL_JAMES_MM,
                    branch_label=(
                        f"Ball-James M/M {branch.base_variant_index}->"
                        f"{branch.other_variant_index}, branch {branch.branch:+d}"
                    ),
                    exact=True,
                    rotation_matrix=_matrix3(branch.rotation),
                    rotation_role="M/M rank-one rotation in parent orthonormal frame",
                    twin_normal_parent_cartesian=_vector3(branch.n_reference),
                    twin_direction_parent_cartesian=_vector3(_unit(branch.a)),
                    shear_magnitude=float(branch.shear_magnitude),
                    residuals=_native_residuals(
                        rank_one=branch.rank_one_residual,
                        rotation=branch.rotation_residual,
                        cc1_lambda2_minus_one=cofactor.cc1_lambda2_minus_one,
                        cc2=cofactor.cc2_value,
                        cc3_margin=cofactor.cc3_margin,
                        all_fraction_lambda2_max=(
                            cofactor.sampled_all_fraction_max_lambda2_residual
                        ),
                    ),
                    metadata={
                        "base_variant_index": branch.base_variant_index,
                        "other_variant_index": branch.other_variant_index,
                        "branch": branch.branch,
                        "cofactor_cc1_satisfied": cofactor.cc1_satisfied,
                        "cofactor_cc2_satisfied": cofactor.cc2_satisfied,
                        "cofactor_cc3_satisfied": cofactor.cc3_satisfied,
                        "cofactor_all_satisfied": cofactor.all_satisfied,
                        "mallard_matches": [
                            {
                                "parent_symmetry_index": match.parent_symmetry_index,
                                "kind": match.kind,
                                "rotation_residual": (
                                    match.rotation_residual_to_general
                                ),
                                "outer_product_residual": (
                                    match.outer_product_residual_to_general
                                ),
                                "shear_relative_residual": (
                                    match.shear_relative_residual
                                ),
                            }
                            for match in branch.mallard_matches
                        ],
                    },
                    provenance="Ball-James general M/M rank-one compatibility",
                )
            )

        return report, rows

    def _ptmc_report(
        self,
        *,
        mode: PTMCMode,
        request: (PTMCSlipSystemInput | PTMCTwinPlaneInput | PTMCTwinPairInput | None),
        base_variant_index: int | None,
        dilatational_factor: float,
    ) -> PTMCReport | None:
        if mode is PTMCMode.NONE:
            if request is not None:
                raise ValueError("ptmc_request must be None when ptmc_mode='none'")
            return None

        adapter = PTMCAdapter(self.project, self.transformation_id)
        if mode is PTMCMode.ALL_TWINNING:
            if request is not None:
                raise ValueError(
                    "ptmc_request must be None for ptmc_mode='all_twinning'"
                )
            return adapter.analyze_all_twinning(
                base_variant_index=base_variant_index,
                dilatational_factor=dilatational_factor,
            )
        if mode is PTMCMode.TWIN_PLANE:
            if not isinstance(request, PTMCTwinPlaneInput):
                raise TypeError("ptmc_mode='twin_plane' requires PTMCTwinPlaneInput")
            return adapter.analyze_twin_plane(
                request, dilatational_factor=dilatational_factor
            )
        if mode is PTMCMode.TWIN_PAIR:
            if not isinstance(request, PTMCTwinPairInput):
                raise TypeError("ptmc_mode='twin_pair' requires PTMCTwinPairInput")
            return adapter.analyze_twin_pair(
                request, dilatational_factor=dilatational_factor
            )
        if mode is PTMCMode.SLIP:
            if not isinstance(request, PTMCSlipSystemInput):
                raise TypeError("ptmc_mode='slip' requires PTMCSlipSystemInput")
            return adapter.analyze_slip(
                request, dilatational_factor=dilatational_factor
            )
        raise AssertionError(f"Unhandled PTMC mode {mode}")

    @staticmethod
    def _ptmc_rows(report: PTMCReport) -> list[ComparisonRow]:
        rows: list[ComparisonRow] = []
        relation_index = {
            (
                relation.base_variant_index,
                relation.other_variant_index,
                relation.branch,
            ): relation
            for relation in report.twin_relations
        }

        for index, solution in enumerate(report.solutions):
            relation = None
            if (
                solution.other_variant_index is not None
                and solution.twin_branch is not None
            ):
                relation = relation_index.get(
                    (
                        solution.base_variant_index,
                        solution.other_variant_index,
                        solution.twin_branch,
                    )
                )

            rows.append(
                ComparisonRow(
                    row_id=f"ptmc_solution_{index}",
                    theory=TheoryKind.PTMC,
                    prediction_kind=PredictionKind.PTMC_HABIT,
                    branch_label=(
                        f"PTMC {solution.lis_type} base "
                        f"{solution.base_variant_index}, "
                        f"{solution.parameter_name}={solution.parameter_value:.12g}, "
                        f"habit branch {solution.habit_branch:+d}"
                    ),
                    exact=bool(solution.true_invariant_plane),
                    or_parent_from_product=_matrix3(
                        solution.or_parent_from_product_base_ptclab
                    ),
                    rotation_matrix=_matrix3(solution.habit_rotation),
                    rotation_role="PTMC habit rotation",
                    habit_plane_parent_crystal=_vector3(
                        solution.habit_plane_parent_crystal
                    ),
                    habit_normal_parent_cartesian=_vector3(
                        solution.habit_normal_parent_cartesian
                    ),
                    twin_plane_product_crystal=(
                        None
                        if relation is None
                        else _vector3(relation.twin_plane_product_crystal_base)
                    ),
                    twin_normal_parent_cartesian=(
                        None if relation is None else _vector3(relation.n_reference)
                    ),
                    twin_direction_parent_cartesian=(
                        None if relation is None else _vector3(_unit(relation.a))
                    ),
                    shear_magnitude=float(solution.lattice_invariant_shear_magnitude),
                    shape_vector_parent_cartesian=_vector3(solution.rank_one_vector),
                    shape_vector_magnitude=float(solution.shape_vector_magnitude),
                    residuals=_native_residuals(
                        middle_stretch=solution.middle_stretch_residual,
                        rank_one=solution.rank_one_residual,
                        rotation=solution.rotation_orthogonality_residual,
                        or_base=solution.or_base_residual,
                        or_other=solution.or_other_residual,
                    ),
                    metadata={
                        "lis_type": solution.lis_type,
                        "base_variant_index": solution.base_variant_index,
                        "other_variant_index": solution.other_variant_index,
                        "twin_branch": solution.twin_branch,
                        "parameter_name": solution.parameter_name,
                        "parameter_value": solution.parameter_value,
                        "dilatational_factor": solution.dilatational_factor,
                        "true_invariant_plane": solution.true_invariant_plane,
                        "base_variant_volume_fraction": (
                            solution.base_variant_volume_fraction
                        ),
                        "other_variant_volume_fraction": (
                            solution.other_variant_volume_fraction
                        ),
                        "tangential_shape_shear_magnitude": (
                            solution.tangential_shape_shear_magnitude
                        ),
                        "normal_shape_component": solution.normal_shape_component,
                    },
                    provenance="Classical single-shear PTMC",
                )
            )

        for index, family in enumerate(report.continuous_families):
            rows.append(
                ComparisonRow(
                    row_id=f"ptmc_continuum_{index}",
                    theory=TheoryKind.PTMC,
                    prediction_kind=PredictionKind.PTMC_CONTINUUM,
                    branch_label=(
                        f"PTMC {family.lis_type} continuous family "
                        f"base {family.base_variant_index}"
                    ),
                    exact=True,
                    metadata={
                        "lis_type": family.lis_type,
                        "base_variant_index": family.base_variant_index,
                        "other_variant_index": family.other_variant_index,
                        "twin_branch": family.twin_branch,
                        "parameter_name": family.parameter_name,
                        "domain": family.domain,
                        "dilatational_factor": family.dilatational_factor,
                        "polynomial_coefficients_ascending": (
                            family.polynomial_coefficients_ascending
                        ),
                        "polynomial_verification_residual": (
                            family.polynomial_verification_residual
                        ),
                    },
                    residuals=_native_residuals(
                        polynomial=family.polynomial_verification_residual
                    ),
                    provenance="Classical PTMC exact continuum",
                    notes=family.note,
                )
            )

        for index, roots in enumerate(report.parameter_roots):
            rows.append(
                ComparisonRow(
                    row_id=f"ptmc_diagnostic_{index}",
                    theory=TheoryKind.PTMC,
                    prediction_kind=PredictionKind.PTMC_PARAMETER_DIAGNOSTIC,
                    branch_label=(
                        f"PTMC parameter diagnostic {index}: {roots.parameter_name}"
                    ),
                    exact=None,
                    residuals=_native_residuals(
                        nearest_middle_stretch=(roots.nearest_middle_stretch_residual),
                        polynomial=roots.polynomial_verification_residual,
                    ),
                    metadata={
                        "parameter_name": roots.parameter_name,
                        "discrete_roots": roots.discrete_roots,
                        "continuum": roots.continuum,
                        "domain": roots.domain,
                        "nearest_parameter": roots.nearest_parameter,
                    },
                    provenance="Classical PTMC parameter/root audit",
                    notes=(
                        "This row preserves the parameter-solving audit and is not "
                        "itself a habit-plane prediction."
                    ),
                )
            )

        return rows

    @staticmethod
    def _experiment_rows(
        observations: tuple[ExperimentalObservation, ...],
    ) -> list[ComparisonRow]:
        rows: list[ComparisonRow] = []
        for observation in observations:
            rows.append(
                ComparisonRow(
                    row_id=f"experiment_{observation.observation_id}",
                    theory=TheoryKind.EXPERIMENT,
                    prediction_kind=PredictionKind.EXPERIMENT,
                    branch_label=observation.label
                    or f"Experiment {observation.observation_id}",
                    exact=None,
                    or_parent_from_product=observation.or_parent_from_product,
                    habit_plane_parent_crystal=(observation.habit_plane_parent_crystal),
                    habit_normal_parent_cartesian=(
                        observation.habit_normal_parent_cartesian
                    ),
                    twin_plane_parent_crystal=(observation.twin_plane_parent_crystal),
                    twin_plane_product_crystal=(observation.twin_plane_product_crystal),
                    twin_normal_parent_cartesian=(
                        observation.twin_normal_parent_cartesian
                    ),
                    twin_direction_parent_crystal=(
                        observation.twin_direction_parent_crystal
                    ),
                    twin_direction_parent_cartesian=(
                        observation.twin_direction_parent_cartesian
                    ),
                    shear_magnitude=observation.shear_magnitude,
                    shape_vector_parent_crystal=(
                        observation.shape_vector_parent_crystal
                    ),
                    shape_vector_parent_cartesian=(
                        observation.shape_vector_parent_cartesian
                    ),
                    shape_vector_magnitude=observation.shape_vector_magnitude,
                    residuals={
                        f"uncertainty_{key}": float(value)
                        for key, value in observation.uncertainty.items()
                    },
                    provenance=observation.provenance,
                    notes=observation.notes,
                )
            )
        return rows

    def compare(
        self,
        *,
        ptmc_mode: PTMCMode | str = PTMCMode.ALL_TWINNING,
        ptmc_request: (
            PTMCSlipSystemInput | PTMCTwinPlaneInput | PTMCTwinPairInput | None
        ) = None,
        ptmc_base_variant_index: int | None = None,
        ptmc_dilatational_factor: float = 1.0,
        natural_orientation: Any | None = None,
        include_ct_closing_gap: bool = True,
        include_ct_supercompatibility: bool = True,
        ball_james_fraction_samples: int = 101,
        experiments: tuple[ExperimentalObservation, ...] = (),
    ) -> UnifiedTheoryReport:
        if ball_james_fraction_samples < 2:
            raise ValueError("ball_james_fraction_samples must be >= 2")
        if ptmc_dilatational_factor <= 0.0:
            raise ValueError("ptmc_dilatational_factor must be positive")
        if natural_orientation is not None and not include_ct_closing_gap:
            raise ValueError("natural_orientation requires include_ct_closing_gap=True")

        mode = PTMCMode(ptmc_mode)
        if ptmc_base_variant_index is not None and mode is not PTMCMode.ALL_TWINNING:
            raise ValueError(
                "ptmc_base_variant_index is only valid for ptmc_mode='all_twinning'; "
                "other PTMC modes carry their variant selection in ptmc_request"
            )

        ct_report, rows = self._ct_rows(
            natural_orientation=natural_orientation,
            include_closing_gap=include_ct_closing_gap,
            include_supercompatibility=include_ct_supercompatibility,
        )

        ball_james_report, ball_james_rows = self._ball_james_rows(
            fraction_samples=ball_james_fraction_samples
        )
        rows.extend(ball_james_rows)

        ptmc_report = self._ptmc_report(
            mode=mode,
            request=ptmc_request,
            base_variant_index=ptmc_base_variant_index,
            dilatational_factor=ptmc_dilatational_factor,
        )
        if ptmc_report is not None:
            rows.extend(self._ptmc_rows(ptmc_report))

        rows.extend(self._experiment_rows(experiments))

        ids = [row.row_id for row in rows]
        if len(ids) != len(set(ids)):
            raise AssertionError("Unified comparison generated duplicate row_id values")

        warnings: list[str] = []
        if ptmc_report is None:
            warnings.append("PTMC was explicitly disabled for this comparison.")
        if natural_orientation is None and include_ct_closing_gap:
            warnings.append(
                "CT closing-gap candidates are all retained because no natural OR "
                "was supplied; no preferred CT OR is selected."
            )
        if not ct_report.exact_habit_planes:
            warnings.append(
                "CT has no exact finite A/M habit-plane branch for these inputs; "
                "approximate CMC diagnostics, when admissible, remain explicitly "
                "marked approximate."
            )

        notes = (
            "All three theory branches consume the same ProjectState transformation.",
            (
                "PTMC LIS data and a CT natural OR are theory-specific auxiliary "
                "hypotheses, not shared crystallographic inputs."
            ),
            (
                "Residuals retain names and native units; this report deliberately "
                "does not compute an overall winner score."
            ),
            (
                "Ball-James rank-one rotations are not silently relabelled as physical "
                "parent/product OR matrices."
            ),
            "Missing observables are represented as None / N/A.",
        )

        return UnifiedTheoryReport(
            transformation_id=self.transformation_id,
            parent_phase_id=self.parent.phase_id,
            product_phase_id=self.product.phase_id,
            rows=tuple(rows),
            ct_report=ct_report,
            ball_james_report=ball_james_report,
            ptmc_report=ptmc_report,
            warnings=tuple(warnings),
            notes=notes,
        )
