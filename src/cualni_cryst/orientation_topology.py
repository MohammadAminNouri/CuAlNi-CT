from __future__ import annotations

"""Cayron-compatible orientation-variant topology.

This module deliberately keeps two layers separate:

1. full crystallographic point groups, used for Cayron-style intersection
   subgroups, left cosets and double cosets;
2. proper SO(3) point-group parts, used to emit physical orientation matrices.

For an OR stored as

    x_A = R_A_from_M @ x_M,

the orientation intersection subgroup in the parent is

    H_T^A = G_A ∩ R_A_from_M G_M R_A_from_M^-1.

Distinct orientation variants are left cosets G_A / H_T^A.  They are *not*
the raw Cartesian matrices obtained from all left/right symmetry products.
"""

from dataclasses import dataclass

import numpy as np
import sympy as sp

from .group_theory import correspondence_subgroup, double_cosets
from .project_state import OrientationState, PhaseState, ProjectState
from .representation import CartesianConvention, CartesianFrame
from .symmetry import matrix_key

Vector3 = tuple[float, float, float]
Matrix3Tuple = tuple[Vector3, Vector3, Vector3]


def _matrix_tuple(matrix: np.ndarray) -> Matrix3Tuple:
    arr = np.asarray(matrix, dtype=float).reshape(3, 3)
    return tuple(tuple(float(value) for value in row) for row in arr)  # type: ignore[return-value]


def _relative_residual(lhs: np.ndarray, rhs: np.ndarray) -> float:
    a = np.asarray(lhs, dtype=float)
    b = np.asarray(rhs, dtype=float)
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0)
    return float(np.linalg.norm(a - b) / scale)


def _rotation_angle_deg(matrix: np.ndarray) -> float:
    R = np.asarray(matrix, dtype=float).reshape(3, 3)
    cosine = float(np.clip((float(np.trace(R)) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _rotation_axis(matrix: np.ndarray) -> Vector3:
    R = np.asarray(matrix, dtype=float).reshape(3, 3)
    angle = _rotation_angle_deg(R)
    if angle <= 1.0e-10:
        return (1.0, 0.0, 0.0)
    if abs(180.0 - angle) <= 1.0e-7:
        values, vectors = np.linalg.eigh(0.5 * (R + np.eye(3)))
        axis = np.real(vectors[:, int(np.argmax(values))])
    else:
        axis = np.array(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ],
            dtype=float,
        )
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-14:
        return (1.0, 0.0, 0.0)
    axis = axis / norm
    nonzero = np.flatnonzero(np.abs(axis) > 1.0e-12)
    if nonzero.size and axis[int(nonzero[0])] < 0.0:
        axis = -axis
    axis[np.abs(axis) < 1.0e-14] = 0.0
    return tuple(float(value) for value in axis)  # type: ignore[return-value]


def _sympy_exact_matrix(matrix: np.ndarray) -> sp.Matrix:
    arr = np.asarray(matrix, dtype=float).reshape(3, 3)
    values: list[list[sp.Expr]] = []
    for row in arr:
        exact_row: list[sp.Expr] = []
        for value in row:
            nearest = round(float(value))
            if abs(float(value) - nearest) <= 1.0e-12:
                exact_row.append(sp.Integer(nearest))
            else:
                exact_row.append(sp.nsimplify(float(value)))
        values.append(exact_row)
    return sp.Matrix(values)


def _numeric_key(matrix: np.ndarray, digits: int = 12) -> tuple[float, ...]:
    arr = np.asarray(matrix, dtype=float).reshape(3, 3).copy()
    arr[np.abs(arr) < 10.0 ** (-digits)] = 0.0
    return tuple(float(value) for value in np.round(arr, digits).ravel())


@dataclass(frozen=True)
class SymmetryCartesianElement:
    phase_symmetry_index: int
    crystallographic_matrix: Matrix3Tuple
    cartesian_matrix: Matrix3Tuple
    determinant: int

    def to_dict(self) -> dict[str, object]:
        return {
            "phase_symmetry_index": self.phase_symmetry_index,
            "crystallographic_matrix": [
                list(row) for row in self.crystallographic_matrix
            ],
            "cartesian_matrix": [list(row) for row in self.cartesian_matrix],
            "determinant": self.determinant,
        }


@dataclass(frozen=True)
class OrientationIntersectionElement:
    reference_symmetry_index: int
    moving_symmetry_index: int
    reference_crystallographic_matrix: Matrix3Tuple
    residual: float

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_symmetry_index": self.reference_symmetry_index,
            "moving_symmetry_index": self.moving_symmetry_index,
            "reference_crystallographic_matrix": [
                list(row) for row in self.reference_crystallographic_matrix
            ],
            "residual": self.residual,
        }


@dataclass(frozen=True)
class OrientationVariantClass:
    index: int
    representative_reference_symmetry_index: int
    reference_coset_symmetry_indices: tuple[int, ...]
    matrix_reference_from_moving: Matrix3Tuple
    equivalent_proper_matrix_count: int
    raw_representative_angle_from_base_deg: float
    crystallographic_disorientation_from_base_deg: float

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "representative_reference_symmetry_index": (
                self.representative_reference_symmetry_index
            ),
            "reference_coset_symmetry_indices": list(
                self.reference_coset_symmetry_indices
            ),
            "matrix_reference_from_moving": [
                list(row) for row in self.matrix_reference_from_moving
            ],
            "equivalent_proper_matrix_count": self.equivalent_proper_matrix_count,
            "raw_representative_angle_from_base_deg": (
                self.raw_representative_angle_from_base_deg
            ),
            "crystallographic_disorientation_from_base_deg": (
                self.crystallographic_disorientation_from_base_deg
            ),
        }


@dataclass(frozen=True)
class OrientationOperatorClass:
    index: int
    parent_double_coset_symmetry_indices: tuple[int, ...]
    size: int
    minimum_crystallographic_disorientation_deg: float
    representative_axis_moving_cartesian: Vector3
    contains_parent_reflection: bool
    contains_parent_180_rotation: bool
    cayron_class: str

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "parent_double_coset_symmetry_indices": list(
                self.parent_double_coset_symmetry_indices
            ),
            "size": self.size,
            "minimum_crystallographic_disorientation_deg": (
                self.minimum_crystallographic_disorientation_deg
            ),
            "representative_axis_moving_cartesian": list(
                self.representative_axis_moving_cartesian
            ),
            "contains_parent_reflection": self.contains_parent_reflection,
            "contains_parent_180_rotation": self.contains_parent_180_rotation,
            "cayron_class": self.cayron_class,
        }


@dataclass(frozen=True)
class CayronTopologyAudit:
    full_reference_group_order: int
    full_moving_group_order: int
    proper_reference_group_order: int
    proper_moving_group_order: int
    full_orientation_intersection_order: int
    proper_orientation_intersection_order: int
    full_orientation_variant_count: int
    proper_orientation_variant_count: int
    full_orientation_operator_count: int
    proper_orientation_operator_count: int
    correspondence_intersection_order: int | None
    correspondence_variant_count: int | None
    correspondence_operator_count: int | None
    orientation_correspondence_intersections_equal: bool | None
    one_to_one_correspondence_orientation_topology: bool | None
    maximum_orientation_intersection_residual: float
    note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "full_reference_group_order": self.full_reference_group_order,
            "full_moving_group_order": self.full_moving_group_order,
            "proper_reference_group_order": self.proper_reference_group_order,
            "proper_moving_group_order": self.proper_moving_group_order,
            "full_orientation_intersection_order": (
                self.full_orientation_intersection_order
            ),
            "proper_orientation_intersection_order": (
                self.proper_orientation_intersection_order
            ),
            "full_orientation_variant_count": self.full_orientation_variant_count,
            "proper_orientation_variant_count": self.proper_orientation_variant_count,
            "full_orientation_operator_count": self.full_orientation_operator_count,
            "proper_orientation_operator_count": self.proper_orientation_operator_count,
            "correspondence_intersection_order": self.correspondence_intersection_order,
            "correspondence_variant_count": self.correspondence_variant_count,
            "correspondence_operator_count": self.correspondence_operator_count,
            "orientation_correspondence_intersections_equal": (
                self.orientation_correspondence_intersections_equal
            ),
            "one_to_one_correspondence_orientation_topology": (
                self.one_to_one_correspondence_orientation_topology
            ),
            "maximum_orientation_intersection_residual": (
                self.maximum_orientation_intersection_residual
            ),
            "note": self.note,
        }


@dataclass(frozen=True)
class OrientationTopologyResult:
    proper_intersection: tuple[OrientationIntersectionElement, ...]
    full_intersection: tuple[OrientationIntersectionElement, ...]
    variants: tuple[OrientationVariantClass, ...]
    operators: tuple[OrientationOperatorClass, ...]
    audit: CayronTopologyAudit

    def to_dict(self) -> dict[str, object]:
        return {
            "proper_intersection": [
                item.to_dict() for item in self.proper_intersection
            ],
            "full_intersection": [item.to_dict() for item in self.full_intersection],
            "variants": [item.to_dict() for item in self.variants],
            "operators": [item.to_dict() for item in self.operators],
            "audit": self.audit.to_dict(),
        }


def _cartesian_symmetries(
    phase: PhaseState,
    convention: CartesianConvention,
    *,
    proper_only: bool,
) -> tuple[SymmetryCartesianElement, ...]:
    frame = CartesianFrame(phase.lattice, convention)
    result: list[SymmetryCartesianElement] = []
    for index, operator in enumerate(phase.symmetry_matrices()):
        determinant = round(float(np.linalg.det(operator)))
        if proper_only and determinant != 1:
            continue
        cartesian = frame.operator_to_cartesian(operator)
        orthogonality = _relative_residual(cartesian.T @ cartesian, np.eye(3))
        determinant_cart = float(np.linalg.det(cartesian))
        if max(orthogonality, abs(abs(determinant_cart) - 1.0)) > 1.0e-8:
            raise AssertionError(
                f"Symmetry {phase.phase_id}[{index}] is not an orthogonal "
                f"Cartesian isometry; residual={orthogonality:.3e}."
            )
        result.append(
            SymmetryCartesianElement(
                phase_symmetry_index=index,
                crystallographic_matrix=_matrix_tuple(operator),
                cartesian_matrix=_matrix_tuple(cartesian),
                determinant=determinant,
            )
        )
    if not result:
        raise ValueError(f"Phase {phase.phase_id!r} has no requested symmetries.")
    return tuple(result)


def _intersection(
    reference_ops: tuple[SymmetryCartesianElement, ...],
    moving_ops: tuple[SymmetryCartesianElement, ...],
    R_reference_from_moving: np.ndarray,
    *,
    tolerance: float,
) -> tuple[OrientationIntersectionElement, ...]:
    found: list[OrientationIntersectionElement] = []
    R = np.asarray(R_reference_from_moving, dtype=float).reshape(3, 3)
    for reference in reference_ops:
        A = np.asarray(reference.cartesian_matrix, dtype=float)
        matches: list[tuple[float, SymmetryCartesianElement]] = []
        for moving in moving_ops:
            M = np.asarray(moving.cartesian_matrix, dtype=float)
            residual = _relative_residual(A, R @ M @ R.T)
            if residual <= tolerance:
                matches.append((residual, moving))
        if matches:
            residual, moving = min(matches, key=lambda item: item[0])
            found.append(
                OrientationIntersectionElement(
                    reference_symmetry_index=reference.phase_symmetry_index,
                    moving_symmetry_index=moving.phase_symmetry_index,
                    reference_crystallographic_matrix=(
                        reference.crystallographic_matrix
                    ),
                    residual=residual,
                )
            )
    return tuple(found)


def _find_matrix_index(
    matrix: np.ndarray,
    group: tuple[SymmetryCartesianElement, ...],
    *,
    tolerance: float,
) -> int:
    matches = [
        (
            _relative_residual(
                matrix,
                np.asarray(element.cartesian_matrix, dtype=float),
            ),
            element.phase_symmetry_index,
        )
        for element in group
    ]
    residual, index = min(matches, key=lambda item: item[0])
    if residual > tolerance:
        raise AssertionError(
            f"Group product did not match a registered symmetry; residual={residual:.3e}."
        )
    return index


def _group_by_index(
    group: tuple[SymmetryCartesianElement, ...],
) -> dict[int, SymmetryCartesianElement]:
    return {element.phase_symmetry_index: element for element in group}


def _left_cosets(
    group: tuple[SymmetryCartesianElement, ...],
    subgroup: tuple[OrientationIntersectionElement, ...],
    *,
    tolerance: float,
) -> tuple[tuple[int, ...], ...]:
    by_index = _group_by_index(group)
    H = [
        np.asarray(by_index[item.reference_symmetry_index].cartesian_matrix)
        for item in subgroup
    ]
    remaining = set(by_index)
    cosets: list[tuple[int, ...]] = []
    while remaining:
        representative_index = min(remaining)
        representative = np.asarray(
            by_index[representative_index].cartesian_matrix,
            dtype=float,
        )
        members = tuple(
            sorted(
                {
                    _find_matrix_index(
                        representative @ h,
                        group,
                        tolerance=tolerance,
                    )
                    for h in H
                }
            )
        )
        cosets.append(members)
        remaining.difference_update(members)
    return tuple(cosets)


def _double_cosets(
    group: tuple[SymmetryCartesianElement, ...],
    subgroup: tuple[OrientationIntersectionElement, ...],
    *,
    tolerance: float,
) -> tuple[tuple[int, ...], ...]:
    by_index = _group_by_index(group)
    H = [
        np.asarray(by_index[item.reference_symmetry_index].cartesian_matrix)
        for item in subgroup
    ]
    remaining = set(by_index)
    result: list[tuple[int, ...]] = []
    while remaining:
        representative_index = min(remaining)
        representative = np.asarray(
            by_index[representative_index].cartesian_matrix,
            dtype=float,
        )
        members = tuple(
            sorted(
                {
                    _find_matrix_index(
                        h_left @ representative @ h_right,
                        group,
                        tolerance=tolerance,
                    )
                    for h_left in H
                    for h_right in H
                }
            )
        )
        result.append(members)
        remaining.difference_update(members)
    return tuple(result)


def _identity_index(group: tuple[SymmetryCartesianElement, ...]) -> int:
    return min(
        group,
        key=lambda item: _relative_residual(
            np.asarray(item.cartesian_matrix, dtype=float),
            np.eye(3),
        ),
    ).phase_symmetry_index


def _crystallographic_disorientation(
    first: np.ndarray,
    second: np.ndarray,
    moving_proper_ops: tuple[SymmetryCartesianElement, ...],
) -> tuple[float, np.ndarray]:
    delta = np.asarray(first, dtype=float).T @ np.asarray(second, dtype=float)
    best_angle = float("inf")
    best = np.eye(3)
    for left in moving_proper_ops:
        S_left = np.asarray(left.cartesian_matrix, dtype=float)
        for right in moving_proper_ops:
            S_right = np.asarray(right.cartesian_matrix, dtype=float)
            candidate = S_left.T @ delta @ S_right
            angle = _rotation_angle_deg(candidate)
            if angle < best_angle:
                best_angle = angle
                best = candidate
    return best_angle, best


def _matching_transformation(project: ProjectState, state: OrientationState):
    matches = [
        transformation
        for transformation in project.transformations
        if transformation.parent_phase_id == state.reference_phase_id
        and transformation.product_phase_id == state.moving_phase_id
    ]
    return matches[0] if len(matches) == 1 else None


def _exact_correspondence_audit(
    project: ProjectState,
    state: OrientationState,
    full_orientation_intersection: tuple[OrientationIntersectionElement, ...],
) -> tuple[int | None, int | None, int | None, bool | None]:
    transformation = _matching_transformation(project, state)
    if transformation is None:
        return None, None, None, None

    reference = project.phase(state.reference_phase_id)
    moving = project.phase(state.moving_phase_id)
    G_reference = [
        _sympy_exact_matrix(matrix) for matrix in reference.symmetry_matrices()
    ]
    G_moving = [_sympy_exact_matrix(matrix) for matrix in moving.symmetry_matrices()]
    H_C = correspondence_subgroup(
        G_reference,
        G_moving,
        transformation.correspondence,
    )
    operators_C = double_cosets(G_reference, H_C)

    H_C_keys = {matrix_key(matrix) for matrix in H_C}
    H_T_keys = {
        matrix_key(
            _sympy_exact_matrix(np.asarray(item.reference_crystallographic_matrix))
        )
        for item in full_orientation_intersection
    }
    equal = H_C_keys == H_T_keys
    return (
        len(H_C),
        len(G_reference) // len(H_C),
        len(operators_C),
        equal,
    )


def build_orientation_topology(
    project: ProjectState,
    state: OrientationState,
) -> OrientationTopologyResult:
    """Build Cayron-style H_T, cosets, double cosets and proper OR representatives."""

    reference = project.phase(state.reference_phase_id)
    moving = project.phase(state.moving_phase_id)
    R = np.asarray(state.R_reference_from_moving, dtype=float)
    tolerance = max(float(project.numerical_policy.representation), 1.0e-12)

    reference_full = _cartesian_symmetries(
        reference,
        state.reference_cartesian_convention,
        proper_only=False,
    )
    moving_full = _cartesian_symmetries(
        moving,
        state.moving_cartesian_convention,
        proper_only=False,
    )
    reference_proper = tuple(item for item in reference_full if item.determinant == 1)
    moving_proper = tuple(item for item in moving_full if item.determinant == 1)

    H_T_full = _intersection(
        reference_full,
        moving_full,
        R,
        tolerance=tolerance,
    )
    H_T_proper = _intersection(
        reference_proper,
        moving_proper,
        R,
        tolerance=tolerance,
    )
    if not H_T_full or not H_T_proper:
        raise AssertionError("Orientation intersection subgroup unexpectedly empty.")

    proper_cosets = _left_cosets(
        reference_proper,
        H_T_proper,
        tolerance=tolerance,
    )
    full_cosets = _left_cosets(
        reference_full,
        H_T_full,
        tolerance=tolerance,
    )
    proper_double_cosets = _double_cosets(
        reference_proper,
        H_T_proper,
        tolerance=tolerance,
    )
    full_double_cosets = _double_cosets(
        reference_full,
        H_T_full,
        tolerance=tolerance,
    )

    reference_proper_by_index = _group_by_index(reference_proper)
    moving_identity_index = _identity_index(moving_proper)
    _ = moving_identity_index  # identity is implicit in emitted representatives

    base = R
    variants: list[OrientationVariantClass] = []
    for index, coset in enumerate(proper_cosets, start=1):
        representative_index = min(coset)
        g = np.asarray(
            reference_proper_by_index[representative_index].cartesian_matrix,
            dtype=float,
        )
        representative = g @ R
        disorientation, _delta = _crystallographic_disorientation(
            base,
            representative,
            moving_proper,
        )
        variants.append(
            OrientationVariantClass(
                index=index,
                representative_reference_symmetry_index=representative_index,
                reference_coset_symmetry_indices=coset,
                matrix_reference_from_moving=_matrix_tuple(representative),
                equivalent_proper_matrix_count=len(coset),
                raw_representative_angle_from_base_deg=_rotation_angle_deg(
                    representative @ base.T
                ),
                crystallographic_disorientation_from_base_deg=disorientation,
            )
        )

    # Cayron's ambivalent/polar classification is a full-point-group property.
    full_by_index = _group_by_index(reference_full)
    operators: list[OrientationOperatorClass] = []
    for index, members in enumerate(full_double_cosets, start=1):
        contains_reflection = False
        contains_pi_rotation = False
        for member_index in members:
            exact = _sympy_exact_matrix(
                np.asarray(full_by_index[member_index].crystallographic_matrix)
            )
            determinant = sp.simplify(exact.det())
            order_two = sp.simplify(exact * exact - sp.eye(3)) == sp.zeros(3)
            trace = sp.simplify(sp.trace(exact))
            # In 3-D an order-2 orthogonal operation with eigenvalues
            # (1,1,-1) is a reflection; (1,-1,-1) is a pi rotation.
            contains_reflection = contains_reflection or bool(
                order_two and determinant == -1 and trace == 1
            )
            contains_pi_rotation = contains_pi_rotation or bool(
                order_two and determinant == 1 and trace == -1
            )

        # Minimum physical disorientation for this operator, evaluated from
        # all proper members that belong to the same full double coset.
        proper_members = [
            member_index
            for member_index in members
            if member_index in reference_proper_by_index
        ]
        candidate_pairs: list[tuple[float, np.ndarray]] = []
        for member_index in proper_members:
            g = np.asarray(
                reference_proper_by_index[member_index].cartesian_matrix,
                dtype=float,
            )
            angle, delta = _crystallographic_disorientation(
                base,
                g @ base,
                moving_proper,
            )
            candidate_pairs.append((angle, delta))
        if candidate_pairs:
            minimum_angle, minimum_delta = min(
                candidate_pairs, key=lambda item: item[0]
            )
            axis = _rotation_axis(minimum_delta)
        else:
            minimum_angle = float("nan")
            axis = (float("nan"), float("nan"), float("nan"))

        identity_index_full = _identity_index(reference_full)
        is_identity_operator = identity_index_full in members
        if is_identity_operator:
            # H_T itself is the identity relation between one variant and itself;
            # common two-fold elements inside H_T must not be advertised as
            # transformation-twin candidates between distinct variants.
            cayron_class = "identity"
            contains_reflection_reported = False
            contains_pi_rotation_reported = False
        else:
            cayron_class = (
                "ambivalent" if contains_reflection or contains_pi_rotation else "polar"
            )
            contains_reflection_reported = contains_reflection
            contains_pi_rotation_reported = contains_pi_rotation

        operators.append(
            OrientationOperatorClass(
                index=index,
                parent_double_coset_symmetry_indices=members,
                size=len(members),
                minimum_crystallographic_disorientation_deg=minimum_angle,
                representative_axis_moving_cartesian=axis,
                contains_parent_reflection=contains_reflection_reported,
                contains_parent_180_rotation=contains_pi_rotation_reported,
                cayron_class=cayron_class,
            )
        )

    H_C_order, N_C, O_C, H_equal = _exact_correspondence_audit(
        project,
        state,
        H_T_full,
    )
    one_to_one = None
    if H_equal is not None and N_C is not None and O_C is not None:
        one_to_one = bool(
            H_equal and N_C == len(full_cosets) and O_C == len(full_double_cosets)
        )

    max_intersection_residual = max(item.residual for item in (*H_T_full, *H_T_proper))
    audit = CayronTopologyAudit(
        full_reference_group_order=len(reference_full),
        full_moving_group_order=len(moving_full),
        proper_reference_group_order=len(reference_proper),
        proper_moving_group_order=len(moving_proper),
        full_orientation_intersection_order=len(H_T_full),
        proper_orientation_intersection_order=len(H_T_proper),
        full_orientation_variant_count=len(full_cosets),
        proper_orientation_variant_count=len(proper_cosets),
        full_orientation_operator_count=len(full_double_cosets),
        proper_orientation_operator_count=len(proper_double_cosets),
        correspondence_intersection_order=H_C_order,
        correspondence_variant_count=N_C,
        correspondence_operator_count=O_C,
        orientation_correspondence_intersections_equal=H_equal,
        one_to_one_correspondence_orientation_topology=one_to_one,
        maximum_orientation_intersection_residual=max_intersection_residual,
        note=(
            "Full point groups follow Cayron's crystallographic coset/double-coset "
            "construction. Proper subgroups are used only to emit SO(3) orientation "
            "matrices. Equality H_T=H_C, when reported, is a computed result for this "
            "OR candidate and transformation state, not a general assumption."
        ),
    )
    return OrientationTopologyResult(
        proper_intersection=H_T_proper,
        full_intersection=H_T_full,
        variants=tuple(variants),
        operators=tuple(operators),
        audit=audit,
    )
