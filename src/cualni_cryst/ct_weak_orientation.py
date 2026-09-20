from __future__ import annotations

"""Cayron-CT binding for axial weak twins and weak closing-gap ORs.

The generic reticular weak-twin mathematics lives in :mod:`weak_twins` and
knows nothing about correspondence groups or parent/product phases. This
adapter supplies the CT-specific pieces only:

* exact parent symmetry operation ``G_A``;
* inherited product inter-correspondence ``C_int = C G_A C^{-1}``;
* exact inherited rational axis;
* weak-plane enumeration in the product Bravais lattice;
* proper parent/product closing-gap ORs generated from the inherited axis and
  a selected weak plane of the base product variant;
* optional symmetry-reduced comparison with an explicitly supplied natural OR.

No polar, Ball-James, PTMC, or experimental orientation is silently reused as
Cayron's answer.
"""

from dataclasses import dataclass

import numpy as np
import sympy as sp

from .orientation import OrientationService, rotation_audit
from .project_state import (
    OrientationDefinition,
    OrientationState,
    OrientationTheoryOrigin,
    StateProvenance,
)
from .provenance import DataStatus
from .representation import CartesianConvention, CartesianFrame
from .weak_twins import (
    AxialWeakTwinResult,
    BravaisNodeBasis,
    enumerate_ct_constrained_weak_planes,
    primitive_integer_direction,
)

Vector3 = tuple[float, float, float]


def _relative_residual(lhs: np.ndarray, rhs: np.ndarray) -> float:
    left = np.asarray(lhs, dtype=float)
    right = np.asarray(rhs, dtype=float)
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1.0)
    return float(np.linalg.norm(left - right) / scale)


def _normalize(vector: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(vector, dtype=float).reshape(3)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite entries")
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-15:
        raise ValueError(f"{name} must be nonzero")
    return array / norm


def _projective_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    a = _normalize(first, name="first vector")
    b = _normalize(second, name="second vector")
    cosine = abs(float(a @ b))
    sine = float(np.linalg.norm(np.cross(a, b)))
    return float(np.degrees(np.arctan2(sine, cosine)))


def _matrix_tuple(matrix: np.ndarray) -> tuple[Vector3, Vector3, Vector3]:
    array = np.asarray(matrix, dtype=float).reshape(3, 3)
    return tuple(tuple(float(value) for value in row) for row in array)  # type: ignore[return-value]


def _pair_frame(direction: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, float]:
    d = _normalize(direction, name="direction")
    n = _normalize(normal, name="plane normal")
    incidence = abs(float(d @ n))
    if incidence > 1.0e-9:
        raise ValueError(
            "physical axis is not incident in the weak plane; "
            f"orthogonality residual={incidence:.3e}"
        )
    d = _normalize(d - float(d @ n) * n, name="in-plane direction")
    second = _normalize(np.cross(n, d), name="frame transverse axis")
    frame = np.column_stack((d, second, n))
    audit = rotation_audit(frame)
    if audit.maximum_residual > 1.0e-12:
        raise AssertionError("weak closing-gap frame is not SO(3)")
    return frame, incidence


def _unique_positive_axis(operation: sp.Matrix) -> tuple[int, int, int]:
    nullspace = (sp.Matrix(operation) - sp.eye(3)).nullspace()
    if len(nullspace) != 1:
        raise ValueError(
            "weak CT route requires a parent symmetry with one unique +1 "
            f"eigenaxis; nullity={len(nullspace)}"
        )
    return primitive_integer_direction(nullspace[0])


def _registered_operation_residual(
    operation: np.ndarray, registered: tuple[np.ndarray, ...]
) -> float:
    return min(_relative_residual(operation, candidate) for candidate in registered)


@dataclass(frozen=True)
class WeakClosingGapCandidate:
    weak_twin_index: int
    orientation: OrientationState
    axis_sign: int
    plane_sign: int
    direction_parallelism_residual_deg: float
    plane_parallelism_residual_deg: float
    rotation_residual: float
    raw_deviation_from_natural_deg: float | None
    symmetry_reduced_deviation_from_natural_deg: float | None


@dataclass(frozen=True)
class CTWeakOrientationReport:
    transformation_id: str
    parent_axis: tuple[int, int, int]
    product_axis: tuple[int, int, int]
    intercorrespondence_exact: tuple[tuple[str, str, str], ...]
    weak_twins: tuple[AxialWeakTwinResult, ...]
    orientation_candidates: tuple[WeakClosingGapCandidate, ...]
    selected_candidate_index: int | None
    natural_orientation_id: str
    selection_rule: str
    warnings: tuple[str, ...]

    @property
    def selected(self) -> WeakClosingGapCandidate | None:
        if self.selected_candidate_index is None:
            return None
        return self.orientation_candidates[self.selected_candidate_index]


class CTWeakOrientationAdapter:
    """Dedicated weak-twin extension of the exact Cayron orientation adapter."""

    def __init__(
        self,
        service: OrientationService,
        transformation_id: str,
        *,
        reference_convention: CartesianConvention | None = None,
        moving_convention: CartesianConvention | None = None,
        algebraic_tolerance: float = 1.0e-10,
        selection_tolerance_deg: float = 1.0e-7,
    ) -> None:
        if algebraic_tolerance <= 0.0:
            raise ValueError("algebraic_tolerance must be positive")
        if selection_tolerance_deg <= 0.0:
            raise ValueError("selection_tolerance_deg must be positive")

        self.service = service
        self.transformation_id = transformation_id
        self.transformation = service.project.transformation(transformation_id)
        self.parent = service.project.phase(self.transformation.parent_phase_id)
        self.product = service.project.phase(self.transformation.product_phase_id)
        self.reference_convention = (
            CartesianConvention(reference_convention)
            if reference_convention is not None
            else self.transformation.parent_cartesian_convention
        )
        self.moving_convention = (
            CartesianConvention(moving_convention)
            if moving_convention is not None
            else self.transformation.product_cartesian_convention
        )
        self.algebraic_tolerance = float(algebraic_tolerance)
        self.selection_tolerance_deg = float(selection_tolerance_deg)

    def _validate_parent_operation(self, parent_operation: sp.Matrix) -> sp.Matrix:
        exact = sp.Matrix(parent_operation)
        if exact.shape != (3, 3):
            raise ValueError("parent_operation must be 3x3")

        numeric = np.asarray(exact, dtype=float)
        metric = self.parent.lattice.metric()
        preservation = _relative_residual(numeric.T @ metric @ numeric, metric)
        if preservation > self.algebraic_tolerance:
            raise ValueError(
                "parent_operation does not preserve the registered parent metric: "
                f"residual={preservation:.3e}"
            )
        if float(np.linalg.det(numeric)) <= 0.0:
            raise ValueError(
                "axial weak CT route requires a proper parent rotation; mirrors "
                "belong to the exact Type-I route"
            )
        if exact**2 == sp.eye(3):
            raise ValueError(
                "parent twofold belongs to the exact Type-II route, not the weak route"
            )

        registered = self.parent.symmetry_matrices()
        if not registered:
            raise ValueError(
                "parent phase has no registered crystallographic symmetries"
            )
        residual = _registered_operation_residual(numeric, registered)
        if residual > self.algebraic_tolerance:
            raise ValueError(
                "parent_operation is not a registered parent symmetry: "
                f"nearest residual={residual:.3e}"
            )
        _unique_positive_axis(exact)
        return exact

    def _validate_natural_orientation(self, state: OrientationState) -> None:
        if state.reference_phase_id != self.parent.phase_id:
            raise ValueError(
                "natural OR reference phase must be the transformation parent"
            )
        if state.moving_phase_id != self.product.phase_id:
            raise ValueError(
                "natural OR moving phase must be the transformation product"
            )
        if (
            state.transformation_id
            and state.transformation_id != self.transformation_id
        ):
            raise ValueError("natural OR is bound to a different transformation")

    def _closing_gap_candidates(
        self,
        weak_twins: tuple[AxialWeakTwinResult, ...],
        *,
        parent_axis: sp.Matrix,
        product_axis: sp.Matrix,
        base_correspondence: sp.Matrix,
        natural_orientation: OrientationState | None,
    ) -> tuple[WeakClosingGapCandidate, ...]:
        parent_frame = CartesianFrame(self.parent.lattice, self.reference_convention)
        product_frame = CartesianFrame(self.product.lattice, self.moving_convention)

        axis_a_cart = parent_frame.direct_to_cartesian(
            np.asarray(parent_axis, dtype=float).reshape(3), normalize=True
        )
        axis_m_cart = product_frame.direct_to_cartesian(
            np.asarray(product_axis, dtype=float).reshape(3), normalize=True
        )

        candidates: list[WeakClosingGapCandidate] = []
        seen: list[np.ndarray] = []

        for weak_index, weak in enumerate(weak_twins):
            # p1 is the weak plane of the base product variant whose A->M
            # correspondence is the registered transformation correspondence.
            p_m = sp.Matrix(weak.plane1_conventional)
            p_a = base_correspondence.T * p_m
            p_a_array = np.asarray(p_a, dtype=float).reshape(3)
            p_m_array = np.asarray(p_m, dtype=float).reshape(3)

            n_a_cart = parent_frame.plane_to_cartesian(p_a_array, normalize=True)
            n_m_cart = product_frame.plane_to_cartesian(p_m_array, normalize=True)
            source_frame, _ = _pair_frame(axis_m_cart, n_m_cart)

            for axis_sign in (-1, 1):
                for plane_sign in (-1, 1):
                    target_frame, _ = _pair_frame(
                        axis_sign * axis_a_cart,
                        plane_sign * n_a_cart,
                    )
                    rotation = target_frame @ source_frame.T
                    if any(
                        _relative_residual(rotation, previous)
                        <= self.algebraic_tolerance
                        for previous in seen
                    ):
                        continue
                    seen.append(rotation)

                    state = self.service.state_from_matrix(
                        self.parent.phase_id,
                        self.product.phase_id,
                        rotation,
                        orientation_id=(
                            f"ct_weak_{self.transformation_id}_{weak_index}_"
                            f"{len(candidates)}"
                        ),
                        label=f"CT weak closing-gap OR {weak_index}",
                        reference_convention=self.reference_convention,
                        moving_convention=self.moving_convention,
                        definition_method=OrientationDefinition.PARALLELISMS,
                        theory_origin=OrientationTheoryOrigin.CAYRON_CT,
                        transformation_id=self.transformation_id,
                        provenance=StateProvenance(
                            status=DataStatus.COMPUTATION_DERIVED,
                            notes=(
                                "Cayron axial weak-twin route: inherited rational "
                                "axis plus CT-constrained weak plane."
                            ),
                        ),
                        notes=(
                            "Proper parent/product closing-gap OR. Reticular weak-"
                            "twin T/C/F remain separately reported and are not "
                            "stored as a physical OrientationState."
                        ),
                    )

                    direction_residual = _projective_angle_deg(
                        rotation @ axis_m_cart, axis_a_cart
                    )
                    plane_residual = _projective_angle_deg(
                        rotation @ n_m_cart, n_a_cart
                    )
                    raw: float | None = None
                    reduced: float | None = None
                    if natural_orientation is not None:
                        comparison = self.service.compare_orientations(
                            state, natural_orientation
                        )
                        raw = comparison.raw_misorientation_deg
                        reduced = comparison.symmetry_reduced_disorientation_deg

                    candidates.append(
                        WeakClosingGapCandidate(
                            weak_twin_index=weak_index,
                            orientation=state,
                            axis_sign=axis_sign,
                            plane_sign=plane_sign,
                            direction_parallelism_residual_deg=direction_residual,
                            plane_parallelism_residual_deg=plane_residual,
                            rotation_residual=rotation_audit(rotation).maximum_residual,
                            raw_deviation_from_natural_deg=raw,
                            symmetry_reduced_deviation_from_natural_deg=reduced,
                        )
                    )

        return tuple(candidates)

    def from_parent_operation(
        self,
        parent_operation: sp.Matrix,
        *,
        product_node_basis: BravaisNodeBasis,
        max_plane_index: int = 6,
        maximum_generalized_shear: float | None = 0.3,
        max_weak_twins: int | None = 32,
        natural_orientation: OrientationState | None = None,
    ) -> CTWeakOrientationReport:
        """Compute CT-constrained axial weak twins for one non-twofold operator."""

        G = self._validate_parent_operation(parent_operation)
        if natural_orientation is not None:
            self._validate_natural_orientation(natural_orientation)

        C = sp.Matrix(self.transformation.correspondence.C_M_from_A)
        parent_axis_tuple = _unique_positive_axis(G)
        parent_axis = sp.Matrix(parent_axis_tuple)
        product_axis = C * parent_axis
        product_axis_tuple = primitive_integer_direction(product_axis)

        C_int = sp.simplify(C * G * C.inv())
        if C_int * product_axis != product_axis:
            raise AssertionError(
                "CT inter-correspondence did not preserve inherited axis"
            )

        weak_twins = enumerate_ct_constrained_weak_planes(
            self.product.lattice.metric(),
            C_int,
            product_axis,
            node_basis=product_node_basis,
            max_plane_index=max_plane_index,
            maximum_generalized_shear=maximum_generalized_shear,
        )
        if max_weak_twins is not None:
            if max_weak_twins < 1:
                raise ValueError("max_weak_twins must be >= 1 or None")
            weak_twins = weak_twins[:max_weak_twins]

        orientations = self._closing_gap_candidates(
            weak_twins,
            parent_axis=parent_axis,
            product_axis=product_axis,
            base_correspondence=C,
            natural_orientation=natural_orientation,
        )

        selected_index: int | None = None
        natural_id = ""
        if natural_orientation is not None and orientations:
            natural_id = natural_orientation.orientation_id
            best = min(
                float(candidate.symmetry_reduced_deviation_from_natural_deg)
                for candidate in orientations
                if candidate.symmetry_reduced_deviation_from_natural_deg is not None
            )
            ties = [
                index
                for index, candidate in enumerate(orientations)
                if candidate.symmetry_reduced_deviation_from_natural_deg is not None
                and abs(
                    float(candidate.symmetry_reduced_deviation_from_natural_deg) - best
                )
                <= self.selection_tolerance_deg
            ]
            selected_index = min(
                ties,
                key=lambda index: (
                    float(orientations[index].raw_deviation_from_natural_deg),
                    orientations[index].weak_twin_index,
                    index,
                ),
            )

        warnings = [
            (
                "This is a CT-constrained weak-plane search for a known exact "
                "inter-correspondence. It does not claim to reproduce every "
                "internal GenOVa generic-supercell enumeration choice."
            ),
            (
                "The product Bravais node basis is explicit and mandatory; no "
                "centering or primitive-cell assumption is inferred from a phase name."
            ),
            (
                "Reticular weak-twin T may be improper, as allowed by Cayron's "
                "reticular formulation. Physical parent/product OR candidates are "
                "stored separately and are always proper rotations in SO(3)."
            ),
        ]

        selection_rule = (
            "No natural OR supplied: weak twins are ordered by generalized shear; "
            "no physical closing-gap OR is selected."
            if natural_orientation is None
            else (
                "Select minimum symmetry-reduced disorientation to the explicitly "
                "supplied natural OR; ties are resolved by raw disorientation and "
                "stable weak-twin/index order."
            )
        )

        return CTWeakOrientationReport(
            transformation_id=self.transformation_id,
            parent_axis=parent_axis_tuple,
            product_axis=product_axis_tuple,
            intercorrespondence_exact=tuple(
                tuple(str(C_int[i, j]) for j in range(3)) for i in range(3)
            ),
            weak_twins=weak_twins,
            orientation_candidates=orientations,
            selected_candidate_index=selected_index,
            natural_orientation_id=natural_id,
            selection_rule=selection_rule,
            warnings=tuple(warnings),
        )
