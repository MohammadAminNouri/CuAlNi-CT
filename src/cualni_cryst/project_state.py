from __future__ import annotations

"""Immutable scientific/project state for crystallographic calculations.

The state layer connects typed crystallographic objects to the lattice,
symmetry, correspondence, provenance, numerical-policy, composition and
thermomechanical context they belong to.

It deliberately contains no GUI state and no heavy transformation-theory
workflow.  Future calculation services consume this state rather than reading
values directly from widgets.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import sympy as sp

from .correspondence import Correspondence
from .crystal_objects import CrystalBasisRef, Direction, ObjectProvenance, Plane
from .lattice import Lattice
from .numerics import DEFAULT_NUMERICAL_POLICY, NumericalPolicy
from .provenance import DataStatus, SourceRef
from .representation import (
    CartesianConvention,
    RepresentationBridge,
)

Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def _freeze_matrix3(matrix: object) -> Matrix3:
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 matrix; got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Matrix contains non-finite values")
    if abs(float(np.linalg.det(arr))) <= 1.0e-14:
        raise ValueError("Matrix must be invertible")
    return tuple(tuple(float(x) for x in row) for row in arr)  # type: ignore[return-value]


def _matrix3_array(matrix: Matrix3) -> np.ndarray:
    return np.asarray(matrix, dtype=float)


def _relative_matrix_residual(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    scale = max(float(np.linalg.norm(lhs)), float(np.linalg.norm(rhs)), 1.0)
    return float(np.linalg.norm(lhs - rhs) / scale)


def _sympy_vector_to_tuple(vector: sp.Matrix) -> tuple[float, float, float]:
    arr = np.asarray(vector, dtype=float).reshape(3)
    return tuple(float(x) for x in arr)


class CompositionBasis(str, Enum):
    """Composition scale; conversions are never performed silently."""

    WEIGHT_PERCENT = "weight_percent"
    ATOMIC_PERCENT = "atomic_percent"
    WEIGHT_FRACTION = "weight_fraction"
    ATOMIC_FRACTION = "atomic_fraction"

    @property
    def target_total(self) -> float:
        if self in {self.WEIGHT_PERCENT, self.ATOMIC_PERCENT}:
            return 100.0
        return 1.0


class OrientationDefinition(str, Enum):
    """How an OrientationState was constructed."""

    USER_MATRIX = "user_matrix"
    AXIS_ANGLE = "axis_angle"
    EULER_ZXZ_ACTIVE = "euler_zxz_active"
    EULER_ZXZ_PASSIVE = "euler_zxz_passive"
    CARTESIAN_HAMILTON_QUATERNION = "cartesian_hamilton_quaternion"
    PARALLELISMS = "crystallographic_parallelisms"
    POLAR_CORRESPONDENCE = "polar_correspondence"


class OrientationTheoryOrigin(str, Enum):
    """Scientific origin of an OR candidate.

    Cayron/Ball-James/PTMC entries are schema contracts only; their theory
    adapters are not silently implemented by this state class.
    """

    USER_DEFINED = "user_defined"
    EXPERIMENTAL = "experimental"
    LITERATURE = "literature"
    POLAR_CORRESPONDENCE = "polar_correspondence"
    CAYRON_CT = "cayron_ct"
    BALL_JAMES = "ball_james"
    PTMC = "ptmc"


@dataclass(frozen=True)
class StateProvenance:
    """Provenance for phase/material/transformation state."""

    status: DataStatus = DataStatus.UNVERIFIED
    source_key: str = ""
    uncertainty: str = ""
    notes: str = ""


@dataclass(frozen=True)
class CompositionComponent:
    element: str
    value: float

    def __post_init__(self) -> None:
        symbol = self.element.strip()
        if not symbol:
            raise ValueError("Composition element must be non-empty")
        if not np.isfinite(self.value) or self.value < 0.0:
            raise ValueError("Composition values must be finite and non-negative")
        object.__setattr__(self, "element", symbol)


@dataclass(frozen=True)
class Composition:
    """Composition with an optional explicit balance element.

    No atomic<->weight conversion is attempted because that requires an
    additional molar-mass model and is not a mere unit conversion.
    """

    components: tuple[CompositionComponent, ...]
    basis: CompositionBasis
    balance_element: str = ""
    provenance: StateProvenance = field(default_factory=StateProvenance)

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("Composition must contain at least one component")

        names = [component.element for component in self.components]
        if len(names) != len(set(names)):
            raise ValueError("Composition contains duplicate element names")

        balance = self.balance_element.strip()
        if balance and balance in set(names):
            raise ValueError("balance_element must not duplicate an explicit component")

        total = sum(component.value for component in self.components)
        target = self.basis.target_total
        tolerance = 1.0e-8 * max(target, 1.0)

        if balance:
            if total > target + tolerance:
                raise ValueError(
                    f"Explicit composition total {total} exceeds {target} "
                    f"for {self.basis.value}"
                )
        elif abs(total - target) > tolerance:
            raise ValueError(
                f"Composition total must equal {target} for {self.basis.value}; got {total}"
            )

        object.__setattr__(self, "balance_element", balance)

    def resolved_components(self) -> tuple[CompositionComponent, ...]:
        if not self.balance_element:
            return self.components

        remainder = self.basis.target_total - sum(
            component.value for component in self.components
        )
        return (*self.components, CompositionComponent(self.balance_element, remainder))

    def value_for(self, element: str) -> float:
        key = element.strip()
        for component in self.resolved_components():
            if component.element == key:
                return component.value
        raise KeyError(f"Element {element!r} is not present in the composition")

    def to_dict(self) -> dict[str, object]:
        return {
            "basis": self.basis.value,
            "input_components": [
                {"element": component.element, "value": component.value}
                for component in self.components
            ],
            "balance_element": self.balance_element,
            "resolved_components": [
                {"element": component.element, "value": component.value}
                for component in self.resolved_components()
            ],
            "status": self.provenance.status.value,
            "source_key": self.provenance.source_key,
            "uncertainty": self.provenance.uncertainty,
            "notes": self.provenance.notes,
        }


@dataclass(frozen=True)
class ThermomechanicalState:
    """Specimen state without inventing missing experimental information."""

    temperature_k: float | None = None
    cauchy_stress_mpa: Matrix3 | None = None
    history: str = ""
    provenance: StateProvenance = field(default_factory=StateProvenance)

    def __post_init__(self) -> None:
        if self.temperature_k is not None and (
            not np.isfinite(self.temperature_k) or self.temperature_k <= 0.0
        ):
            raise ValueError("Absolute temperature must be finite and > 0 K")

        if self.cauchy_stress_mpa is not None:
            stress = _freeze_matrix3(self.cauchy_stress_mpa)
            arr = _matrix3_array(stress)
            if not np.allclose(arr, arr.T, atol=1.0e-10, rtol=1.0e-10):
                raise ValueError(
                    "Cauchy stress must be symmetric in the classical "
                    "non-polar continuum model"
                )
            object.__setattr__(self, "cauchy_stress_mpa", stress)

    def to_dict(self) -> dict[str, object]:
        return {
            "temperature_k": self.temperature_k,
            "cauchy_stress_mpa": (
                [list(row) for row in self.cauchy_stress_mpa]
                if self.cauchy_stress_mpa is not None
                else None
            ),
            "history": self.history,
            "status": self.provenance.status.value,
            "source_key": self.provenance.source_key,
            "uncertainty": self.provenance.uncertainty,
            "notes": self.provenance.notes,
        }


@dataclass(frozen=True)
class PhaseState:
    """One physical phase plus one explicit crystallographic cell/basis."""

    phase_id: str
    label: str
    physical_phase: str
    cell_representation: str
    lattice: Lattice
    basis: CrystalBasisRef
    point_group_symbol: str
    symmetry_operators: tuple[Matrix3, ...] = ()
    provenance: StateProvenance = field(default_factory=StateProvenance)

    def __post_init__(self) -> None:
        phase_id = self.phase_id.strip()
        if not phase_id:
            raise ValueError("phase_id must be non-empty")
        if self.basis.phase_id != phase_id:
            raise ValueError(
                "PhaseState.phase_id and CrystalBasisRef.phase_id must match exactly"
            )
        if (
            self.basis.cell_representation
            and self.cell_representation
            and self.basis.cell_representation != self.cell_representation
        ):
            raise ValueError(
                "CrystalBasisRef.cell_representation must match PhaseState "
                "cell_representation"
            )

        frozen_ops = tuple(
            _freeze_matrix3(operator) for operator in self.symmetry_operators
        )
        object.__setattr__(self, "phase_id", phase_id)
        object.__setattr__(self, "symmetry_operators", frozen_ops)

    def symmetry_matrices(self) -> tuple[np.ndarray, ...]:
        return tuple(_matrix3_array(operator) for operator in self.symmetry_operators)

    def to_dict(self) -> dict[str, object]:
        return {
            "phase_id": self.phase_id,
            "label": self.label,
            "physical_phase": self.physical_phase,
            "cell_representation": self.cell_representation,
            "basis": {
                "phase_id": self.basis.phase_id,
                "basis_id": self.basis.basis_id,
                "cell_representation": self.basis.cell_representation,
            },
            "lattice": {
                "a": self.lattice.a,
                "b": self.lattice.b,
                "c": self.lattice.c,
                "alpha_deg": self.lattice.alpha_deg,
                "beta_deg": self.lattice.beta_deg,
                "gamma_deg": self.lattice.gamma_deg,
                "label": self.lattice.label,
                "length_unit": self.lattice.length_unit,
            },
            "point_group_symbol": self.point_group_symbol,
            "symmetry_order": len(self.symmetry_operators),
            "status": self.provenance.status.value,
            "source_key": self.provenance.source_key,
            "uncertainty": self.provenance.uncertainty,
            "notes": self.provenance.notes,
        }


@dataclass(frozen=True)
class TransformationState:
    """Explicit relation between two registered phase/basis states."""

    transformation_id: str
    label: str
    parent_phase_id: str
    product_phase_id: str
    correspondence: Correspondence
    parent_cartesian_convention: CartesianConvention = (
        CartesianConvention.SYMMETRIC_METRIC
    )
    product_cartesian_convention: CartesianConvention = (
        CartesianConvention.SYMMETRIC_METRIC
    )
    provenance: StateProvenance = field(default_factory=StateProvenance)

    def __post_init__(self) -> None:
        transformation_id = self.transformation_id.strip()
        parent = self.parent_phase_id.strip()
        product = self.product_phase_id.strip()

        if not transformation_id:
            raise ValueError("transformation_id must be non-empty")
        if not parent or not product:
            raise ValueError("parent_phase_id and product_phase_id must be non-empty")
        if parent == product:
            raise ValueError("A transformation must connect distinct registered phases")

        object.__setattr__(self, "transformation_id", transformation_id)
        object.__setattr__(self, "parent_phase_id", parent)
        object.__setattr__(self, "product_phase_id", product)

    def to_dict(self) -> dict[str, object]:
        exact_matrix = [
            [str(self.correspondence.C_M_from_A[i, j]) for j in range(3)]
            for i in range(3)
        ]
        return {
            "transformation_id": self.transformation_id,
            "label": self.label,
            "parent_phase_id": self.parent_phase_id,
            "product_phase_id": self.product_phase_id,
            "correspondence_M_from_A_exact": exact_matrix,
            "correspondence_label": self.correspondence.label,
            "parent_cartesian_convention": self.parent_cartesian_convention.value,
            "product_cartesian_convention": self.product_cartesian_convention.value,
            "status": self.provenance.status.value,
            "source_key": self.provenance.source_key,
            "uncertainty": self.provenance.uncertainty,
            "notes": self.provenance.notes,
        }


@dataclass(frozen=True)
class OrientationState:
    """One proper physical orientation relationship between two phases.

    Convention:
        x_reference = R_reference_from_moving @ x_moving

    This is deliberately not a crystallographic correspondence matrix.
    """

    orientation_id: str
    label: str
    reference_phase_id: str
    moving_phase_id: str
    R_reference_from_moving: Matrix3
    reference_cartesian_convention: CartesianConvention = (
        CartesianConvention.PTCLAB_A_X_C_XZ
    )
    moving_cartesian_convention: CartesianConvention = (
        CartesianConvention.PTCLAB_A_X_C_XZ
    )
    definition_method: OrientationDefinition = OrientationDefinition.USER_MATRIX
    theory_origin: OrientationTheoryOrigin = OrientationTheoryOrigin.USER_DEFINED
    provenance: StateProvenance = field(default_factory=StateProvenance)
    notes: str = ""
    transformation_id: str = ""

    def __post_init__(self) -> None:
        orientation_id = self.orientation_id.strip()
        reference = self.reference_phase_id.strip()
        moving = self.moving_phase_id.strip()
        transformation_id = self.transformation_id.strip()
        if not orientation_id:
            raise ValueError("orientation_id must be non-empty")
        if not reference or not moving:
            raise ValueError("reference_phase_id and moving_phase_id must be non-empty")
        if reference == moving:
            raise ValueError("An OrientationState must connect two distinct phases")

        matrix = _freeze_matrix3(self.R_reference_from_moving)
        array = _matrix3_array(matrix)
        orthogonality = _relative_matrix_residual(
            array.T @ array,
            np.eye(3),
        )
        determinant = float(np.linalg.det(array))
        residual = max(orthogonality, abs(determinant - 1.0))
        if residual > 1.0e-8:
            raise ValueError(
                "R_reference_from_moving must be a proper rotation; "
                f"residual={residual:.3e}, det={determinant:.12g}"
            )

        object.__setattr__(self, "orientation_id", orientation_id)
        object.__setattr__(self, "reference_phase_id", reference)
        object.__setattr__(self, "moving_phase_id", moving)
        object.__setattr__(self, "transformation_id", transformation_id)
        object.__setattr__(self, "R_reference_from_moving", matrix)
        object.__setattr__(
            self,
            "reference_cartesian_convention",
            CartesianConvention(self.reference_cartesian_convention),
        )
        object.__setattr__(
            self,
            "moving_cartesian_convention",
            CartesianConvention(self.moving_cartesian_convention),
        )
        object.__setattr__(
            self,
            "definition_method",
            OrientationDefinition(self.definition_method),
        )
        object.__setattr__(
            self,
            "theory_origin",
            OrientationTheoryOrigin(self.theory_origin),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "orientation_id": self.orientation_id,
            "label": self.label,
            "reference_phase_id": self.reference_phase_id,
            "moving_phase_id": self.moving_phase_id,
            "R_reference_from_moving": [
                list(row) for row in self.R_reference_from_moving
            ],
            "reference_cartesian_convention": (
                self.reference_cartesian_convention.value
            ),
            "moving_cartesian_convention": (self.moving_cartesian_convention.value),
            "definition_method": self.definition_method.value,
            "theory_origin": self.theory_origin.value,
            "transformation_id": self.transformation_id,
            "status": self.provenance.status.value,
            "source_key": self.provenance.source_key,
            "uncertainty": self.provenance.uncertainty,
            "provenance_notes": self.provenance.notes,
            "notes": self.notes,
        }


class ValidationSeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass(frozen=True)
class ValidationIssue:
    severity: ValidationSeverity
    code: str
    message: str


@dataclass(frozen=True)
class ProjectValidationReport:
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(
            issue for issue in self.issues if issue.severity is ValidationSeverity.ERROR
        )

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(
            issue
            for issue in self.issues
            if issue.severity is ValidationSeverity.WARNING
        )

    @property
    def passed(self) -> bool:
        return not self.errors

    def assert_passed(self) -> None:
        if self.errors:
            summary = "; ".join(
                f"{issue.code}: {issue.message}" for issue in self.errors
            )
            raise AssertionError(
                f"Project scientific-state validation failed: {summary}"
            )


@dataclass(frozen=True)
class ProjectState:
    """Single source of truth for a reproducible crystallographic project."""

    project_id: str
    title: str
    phases: tuple[PhaseState, ...]
    transformations: tuple[TransformationState, ...] = ()
    sources: tuple[SourceRef, ...] = ()
    composition: Composition | None = None
    thermomechanical_state: ThermomechanicalState = field(
        default_factory=ThermomechanicalState
    )
    numerical_policy: NumericalPolicy = DEFAULT_NUMERICAL_POLICY
    notes: str = ""
    orientations: tuple[OrientationState, ...] = ()

    def __post_init__(self) -> None:
        project_id = self.project_id.strip()
        if not project_id:
            raise ValueError("project_id must be non-empty")
        if not self.phases:
            raise ValueError("A project must contain at least one PhaseState")

        phase_ids = [phase.phase_id for phase in self.phases]
        if len(phase_ids) != len(set(phase_ids)):
            raise ValueError("Project contains duplicate phase_id values")

        basis_refs = [phase.basis for phase in self.phases]
        if len(basis_refs) != len(set(basis_refs)):
            raise ValueError("Project contains duplicate CrystalBasisRef identities")

        transformation_ids = [
            transformation.transformation_id for transformation in self.transformations
        ]
        if len(transformation_ids) != len(set(transformation_ids)):
            raise ValueError("Project contains duplicate transformation_id values")

        known_phases = set(phase_ids)
        for transformation in self.transformations:
            if transformation.parent_phase_id not in known_phases:
                raise ValueError(
                    f"Transformation {transformation.transformation_id!r} references "
                    f"unknown parent phase {transformation.parent_phase_id!r}"
                )
            if transformation.product_phase_id not in known_phases:
                raise ValueError(
                    f"Transformation {transformation.transformation_id!r} references "
                    f"unknown product phase {transformation.product_phase_id!r}"
                )

        source_keys = [source.key for source in self.sources]
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("Project contains duplicate SourceRef keys")

        orientation_ids = [
            orientation.orientation_id for orientation in self.orientations
        ]
        if len(orientation_ids) != len(set(orientation_ids)):
            raise ValueError("Project contains duplicate orientation_id values")

        for orientation in self.orientations:
            if orientation.reference_phase_id not in known_phases:
                raise ValueError(
                    f"Orientation {orientation.orientation_id!r} references "
                    f"unknown reference phase {orientation.reference_phase_id!r}"
                )
            if orientation.moving_phase_id not in known_phases:
                raise ValueError(
                    f"Orientation {orientation.orientation_id!r} references "
                    f"unknown moving phase {orientation.moving_phase_id!r}"
                )

            if orientation.transformation_id:
                bound = next(
                    (
                        transformation
                        for transformation in self.transformations
                        if transformation.transformation_id
                        == orientation.transformation_id
                    ),
                    None,
                )
                if bound is None:
                    raise ValueError(
                        f"Orientation {orientation.orientation_id!r} is bound to "
                        f"unknown transformation "
                        f"{orientation.transformation_id!r}"
                    )
                if (
                    bound.parent_phase_id != orientation.reference_phase_id
                    or bound.product_phase_id != orientation.moving_phase_id
                ):
                    raise ValueError(
                        f"Orientation {orientation.orientation_id!r} binding "
                        f"{orientation.transformation_id!r} has incompatible "
                        "parent/product phase endpoints"
                    )

        object.__setattr__(self, "project_id", project_id)

    def phase(self, phase_id: str) -> PhaseState:
        for phase in self.phases:
            if phase.phase_id == phase_id:
                return phase
        raise KeyError(f"Unknown phase_id {phase_id!r}")

    def transformation(self, transformation_id: str) -> TransformationState:
        for transformation in self.transformations:
            if transformation.transformation_id == transformation_id:
                return transformation
        raise KeyError(f"Unknown transformation_id {transformation_id!r}")

    def orientation(self, orientation_id: str) -> OrientationState:
        for orientation in self.orientations:
            if orientation.orientation_id == orientation_id:
                return orientation
        raise KeyError(f"Unknown orientation_id {orientation_id!r}")

    def phase_for_basis(self, basis: CrystalBasisRef) -> PhaseState:
        for phase in self.phases:
            if phase.basis == basis:
                return phase
        raise KeyError(
            "No registered PhaseState matches basis "
            f"{basis.phase_id!r}/{basis.basis_id!r}/{basis.cell_representation!r}"
        )

    def lattice_for(self, obj: Direction | Plane) -> Lattice:
        return self.phase_for_basis(obj.basis).lattice

    def bridge(self, transformation_id: str) -> RepresentationBridge:
        transformation = self.transformation(transformation_id)
        parent = self.phase(transformation.parent_phase_id)
        product = self.phase(transformation.product_phase_id)
        return RepresentationBridge(
            parent.lattice,
            product.lattice,
            transformation.correspondence,
            transformation.parent_cartesian_convention,
            transformation.product_cartesian_convention,
        )

    @staticmethod
    def _mapped_provenance(
        obj: Direction | Plane,
        transformation: TransformationState,
        direction_text: str,
    ) -> ObjectProvenance:
        source_key = (
            transformation.provenance.source_key
            or obj.provenance.source_key
            or transformation.correspondence.source
        )
        return ObjectProvenance(
            status=DataStatus.COMPUTATION_DERIVED,
            source_key=source_key,
            uncertainty=obj.provenance.uncertainty,
            notes=(
                f"{direction_text} via {transformation.transformation_id}; "
                f"input_status={obj.provenance.status.value}. "
                f"{obj.provenance.notes}".strip()
            ),
        )

    def map_direction(
        self,
        transformation_id: str,
        direction: Direction,
    ) -> Direction:
        transformation = self.transformation(transformation_id)
        parent = self.phase(transformation.parent_phase_id)
        product = self.phase(transformation.product_phase_id)

        if direction.basis == parent.basis:
            mapped = transformation.correspondence.map_direction_A_to_M(direction.array)
            target_basis = product.basis
            text = f"Mapped {parent.phase_id} -> {product.phase_id}"
        elif direction.basis == product.basis:
            mapped = transformation.correspondence.map_direction_M_to_A(direction.array)
            target_basis = parent.basis
            text = f"Mapped {product.phase_id} -> {parent.phase_id}"
        else:
            raise ValueError(
                "Direction basis does not belong to either endpoint of "
                f"transformation {transformation_id!r}"
            )

        return Direction(
            _sympy_vector_to_tuple(mapped),
            target_basis,
            label=direction.label,
            provenance=self._mapped_provenance(direction, transformation, text),
        )

    def map_plane(
        self,
        transformation_id: str,
        plane: Plane,
    ) -> Plane:
        transformation = self.transformation(transformation_id)
        parent = self.phase(transformation.parent_phase_id)
        product = self.phase(transformation.product_phase_id)

        if plane.basis == parent.basis:
            mapped = transformation.correspondence.map_plane_A_to_M(plane.array)
            target_basis = product.basis
            text = f"Mapped {parent.phase_id} -> {product.phase_id}"
        elif plane.basis == product.basis:
            mapped = transformation.correspondence.map_plane_M_to_A(plane.array)
            target_basis = parent.basis
            text = f"Mapped {product.phase_id} -> {parent.phase_id}"
        else:
            raise ValueError(
                "Plane basis does not belong to either endpoint of "
                f"transformation {transformation_id!r}"
            )

        return Plane(
            _sympy_vector_to_tuple(mapped),
            target_basis,
            label=plane.label,
            provenance=self._mapped_provenance(plane, transformation, text),
        )

    def validate(self) -> ProjectValidationReport:
        """Validate reference integrity, symmetry metrics and frame parity.

        This is a state-integrity audit.  Heavy CT/PTMC/cofactor calculations
        remain in the calculation layer.
        """

        issues: list[ValidationIssue] = []
        known_sources = {source.key for source in self.sources}

        def check_source(owner: str, provenance: StateProvenance) -> None:
            if provenance.source_key and provenance.source_key not in known_sources:
                issues.append(
                    ValidationIssue(
                        ValidationSeverity.ERROR,
                        "UNKNOWN_SOURCE_KEY",
                        f"{owner} references unknown source {provenance.source_key!r}",
                    )
                )

        if self.composition is not None:
            check_source("composition", self.composition.provenance)
        check_source("thermomechanical_state", self.thermomechanical_state.provenance)

        for phase in self.phases:
            check_source(f"phase:{phase.phase_id}", phase.provenance)

            if not phase.symmetry_operators:
                issues.append(
                    ValidationIssue(
                        ValidationSeverity.WARNING,
                        "NO_SYMMETRY_OPERATORS",
                        f"Phase {phase.phase_id!r} has no registered symmetry operators",
                    )
                )
                continue

            metric = phase.lattice.metric()
            for index, operator in enumerate(phase.symmetry_matrices()):
                det = float(np.linalg.det(operator))
                if abs(abs(det) - 1.0) > self.numerical_policy.algebraic:
                    issues.append(
                        ValidationIssue(
                            ValidationSeverity.ERROR,
                            "SYMMETRY_DETERMINANT",
                            f"{phase.phase_id} symmetry[{index}] det={det:.12g}, "
                            "expected ±1",
                        )
                    )

                residual = _relative_matrix_residual(
                    operator.T @ metric @ operator,
                    metric,
                )
                if residual > self.numerical_policy.representation:
                    issues.append(
                        ValidationIssue(
                            ValidationSeverity.ERROR,
                            "SYMMETRY_METRIC_MISMATCH",
                            f"{phase.phase_id} symmetry[{index}] does not preserve "
                            f"the phase metric; residual={residual:.3e}",
                        )
                    )

        for transformation in self.transformations:
            check_source(
                f"transformation:{transformation.transformation_id}",
                transformation.provenance,
            )
            parity = self.bridge(transformation.transformation_id).audit()
            if parity.maximum_residual > self.numerical_policy.representation:
                issues.append(
                    ValidationIssue(
                        ValidationSeverity.ERROR,
                        "REPRESENTATION_PARITY",
                        f"{transformation.transformation_id} metric/Cartesian "
                        f"parity residual={parity.maximum_residual:.3e}",
                    )
                )

        for orientation in self.orientations:
            check_source(
                f"orientation:{orientation.orientation_id}",
                orientation.provenance,
            )
            matrix = np.asarray(
                orientation.R_reference_from_moving,
                dtype=float,
            )
            residual = max(
                _relative_matrix_residual(matrix.T @ matrix, np.eye(3)),
                abs(float(np.linalg.det(matrix)) - 1.0),
            )
            if residual > self.numerical_policy.representation:
                issues.append(
                    ValidationIssue(
                        ValidationSeverity.ERROR,
                        "ORIENTATION_NOT_SO3",
                        f"{orientation.orientation_id} is not a proper "
                        f"rotation within project tolerance; residual="
                        f"{residual:.3e}",
                    )
                )

        return ProjectValidationReport(tuple(issues))

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "title": self.title,
            "composition": self.composition.to_dict() if self.composition else None,
            "thermomechanical_state": self.thermomechanical_state.to_dict(),
            "phases": [phase.to_dict() for phase in self.phases],
            "transformations": [
                transformation.to_dict() for transformation in self.transformations
            ],
            "orientations": [
                orientation.to_dict() for orientation in self.orientations
            ],
            "sources": [
                {
                    "key": source.key,
                    "citation": source.citation,
                    "doi": source.doi,
                    "url": source.url,
                    "pages": source.pages,
                    "equations": list(source.equations),
                    "notes": source.notes,
                }
                for source in self.sources
            ],
            "numerical_policy": {
                "algebraic": self.numerical_policy.algebraic,
                "representation": self.numerical_policy.representation,
                "exact_eigenvalue": self.numerical_policy.exact_eigenvalue,
                "rank_one": self.numerical_policy.rank_one,
                "projective_angle_deg": self.numerical_policy.projective_angle_deg,
            },
            "notes": self.notes,
        }


def _freeze_symmetry_operators(operators: tuple[sp.Matrix, ...]) -> tuple[Matrix3, ...]:
    return tuple(
        _freeze_matrix3(np.asarray(operator, dtype=float)) for operator in operators
    )


def james_hane_6m_reference_project() -> ProjectState:
    """Build the literature benchmark without inventing missing T/stress data."""

    from .cualni_models import do3_to_6m_branch, james_hane_6m_example_lattices

    parent_lattice, product_lattice = james_hane_6m_example_lattices()
    branch = do3_to_6m_branch()

    source = SourceRef(
        key="james_hane2000",
        citation=(
            "R. D. James and K. F. Hane, Martensitic transformations and "
            "shape-memory materials, Acta Materialia 48 (2000) 197–222."
        ),
        notes=(
            "Cu-Al-Ni lattice benchmark and 6M compatibility discussion. "
            "Publication-grade page/equation pinpointing remains a separate task."
        ),
    )

    parent_basis = CrystalBasisRef(
        "austenite_do3",
        "cubic_DO3",
        "DO3",
    )
    product_basis = CrystalBasisRef(
        "martensite_long_period",
        "reference_6M_unique_b",
        "6M",
    )

    parent = PhaseState(
        phase_id="austenite_do3",
        label="DO3 austenite",
        physical_phase="DO3 ordered cubic austenite",
        cell_representation="DO3",
        lattice=parent_lattice,
        basis=parent_basis,
        point_group_symbol="m-3m",
        symmetry_operators=_freeze_symmetry_operators(branch.parent_point_group),
        provenance=StateProvenance(
            DataStatus.SOURCE_MEASURED,
            "james_hane2000",
            notes="Lattice parameter from the rounded literature benchmark.",
        ),
    )

    product = PhaseState(
        phase_id="martensite_long_period",
        label="Long-period martensite represented as 6M",
        physical_phase="long-period Cu-Al-Ni martensite",
        cell_representation="6M",
        lattice=product_lattice,
        basis=product_basis,
        point_group_symbol="2/m (unique b)",
        symmetry_operators=_freeze_symmetry_operators(branch.product_point_group),
        provenance=StateProvenance(
            DataStatus.SOURCE_MEASURED,
            "james_hane2000",
            notes=(
                "Physical martensite and computational 6M cell representation "
                "are stored separately."
            ),
        ),
    )

    transformation = TransformationState(
        transformation_id="do3_to_6m_reference",
        label="DO3 -> 6M selected reference correspondence",
        parent_phase_id=parent.phase_id,
        product_phase_id=product.phase_id,
        correspondence=branch.correspondence,
        provenance=StateProvenance(
            DataStatus.SOURCE_INTERPRETATION,
            "james_hane2000",
            notes=branch.correspondence.derivation,
        ),
    )

    composition = Composition(
        (
            CompositionComponent("Al", 14.0),
            CompositionComponent("Ni", 4.0),
        ),
        CompositionBasis.WEIGHT_PERCENT,
        balance_element="Cu",
        provenance=StateProvenance(
            DataStatus.SOURCE_MEASURED,
            "james_hane2000",
            notes="Cu balance is explicitly resolved, not silently assumed.",
        ),
    )

    return ProjectState(
        project_id="james_hane_cualni_6m_reference",
        title="James-Hane Cu-Al-Ni DO3 -> 6M reproducibility benchmark",
        phases=(parent, product),
        transformations=(transformation,),
        sources=(source,),
        composition=composition,
        thermomechanical_state=ThermomechanicalState(
            history="Temperature, applied stress and processing history are not invented.",
            provenance=StateProvenance(
                DataStatus.NOT_EVALUABLE,
                notes="No specimen-specific thermomechanical state supplied here.",
            ),
        ),
        notes=(
            "Reference/reproducibility project only. Rounded literature values "
            "must not be treated as an unknown specimen's defaults."
        ),
    )
