from __future__ import annotations

"""Orientation-relationship engine for CuAlNi-CT.

Central convention:

    x_reference = R_reference_from_moving @ x_moving

R is a proper physical Cartesian rotation. It is deliberately distinct from
the crystallographic correspondence matrix C.
"""

import argparse
import json
import math
import re
from dataclasses import dataclass, replace
from enum import Enum
from itertools import product

import numpy as np

from .crystal_objects import Direction, ObjectProvenance, Plane
from .crystallography_console import (
    ConsoleInputKind,
    CrystallographyConsole,
    parse_crystal_input,
)
from .orientation_topology import (
    CayronTopologyAudit,
    OrientationOperatorClass,
    build_orientation_topology,
)
from .project_state import (
    OrientationDefinition,
    OrientationState,
    OrientationTheoryOrigin,
    PhaseState,
    ProjectState,
    StateProvenance,
    james_hane_6m_reference_project,
)
from .provenance import DataStatus
from .representation import CartesianConvention, CartesianFrame, frame_rotation

Vector3 = tuple[float, float, float]
Quaternion4 = tuple[float, float, float, float]
CrystalObject = Direction | Plane


class EulerConvention(str, Enum):
    """Explicit ZXZ conventions, never an unnamed vendor convention."""

    ZXZ_ACTIVE = "zxz_active"
    ZXZ_PASSIVE = "zxz_passive"


@dataclass(frozen=True)
class RotationAudit:
    orthogonality_residual: float
    determinant: float
    determinant_residual: float

    @property
    def maximum_residual(self) -> float:
        return max(self.orthogonality_residual, self.determinant_residual)

    def to_dict(self) -> dict[str, float]:
        return {
            "orthogonality_residual": self.orthogonality_residual,
            "determinant": self.determinant,
            "determinant_residual": self.determinant_residual,
            "maximum_residual": self.maximum_residual,
        }


@dataclass(frozen=True)
class AxisAngle:
    axis: Vector3
    angle_deg: float

    def to_dict(self) -> dict[str, object]:
        return {"axis": list(self.axis), "angle_deg": self.angle_deg}


@dataclass(frozen=True)
class EulerAngles:
    convention: EulerConvention
    phi1_deg: float
    Phi_deg: float
    phi2_deg: float
    singular: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "convention": self.convention.value,
            "phi1_deg": self.phi1_deg,
            "Phi_deg": self.Phi_deg,
            "phi2_deg": self.phi2_deg,
            "singular": self.singular,
        }


@dataclass(frozen=True)
class OrientationRepresentation:
    reference_convention: str
    moving_convention: str
    matrix_reference_from_moving: tuple[Vector3, Vector3, Vector3]
    audit: RotationAudit

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_convention": self.reference_convention,
            "moving_convention": self.moving_convention,
            "matrix_reference_from_moving": [
                list(row) for row in self.matrix_reference_from_moving
            ],
            "audit": self.audit.to_dict(),
        }


@dataclass(frozen=True)
class OrientationParityAudit:
    maximum_direction_mapping_residual: float
    maximum_plane_mapping_residual: float
    maximum_rotation_residual: float

    @property
    def maximum_residual(self) -> float:
        return max(
            self.maximum_direction_mapping_residual,
            self.maximum_plane_mapping_residual,
            self.maximum_rotation_residual,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "maximum_direction_mapping_residual": (
                self.maximum_direction_mapping_residual
            ),
            "maximum_plane_mapping_residual": self.maximum_plane_mapping_residual,
            "maximum_rotation_residual": self.maximum_rotation_residual,
            "maximum_residual": self.maximum_residual,
        }


@dataclass(frozen=True)
class OrientationVariant:
    """One crystallographically distinct proper orientation-variant class.

    The class is a left coset of the proper orientation intersection subgroup.
    It is not one raw left/right symmetry-product matrix.
    """

    index: int
    reference_symmetry_index: int
    moving_symmetry_index: int
    matrix_reference_from_moving: tuple[Vector3, Vector3, Vector3]
    misorientation_from_base_deg: float
    reference_coset_symmetry_indices: tuple[int, ...] = ()
    equivalent_proper_matrix_count: int = 1
    raw_representative_angle_from_base_deg: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "reference_symmetry_index": self.reference_symmetry_index,
            "moving_symmetry_index": self.moving_symmetry_index,
            "matrix_reference_from_moving": [
                list(row) for row in self.matrix_reference_from_moving
            ],
            "misorientation_from_base_deg": self.misorientation_from_base_deg,
            "reference_coset_symmetry_indices": list(
                self.reference_coset_symmetry_indices
            ),
            "equivalent_proper_matrix_count": self.equivalent_proper_matrix_count,
            "raw_representative_angle_from_base_deg": (
                self.raw_representative_angle_from_base_deg
            ),
        }


@dataclass(frozen=True)
class OrientationReport:
    state: OrientationState
    audit: RotationAudit
    axis_angle: AxisAngle
    quaternion_wxyz: Quaternion4
    euler_zxz_active: EulerAngles
    euler_zxz_passive: EulerAngles
    representations: tuple[OrientationRepresentation, ...]
    parity: OrientationParityAudit
    proper_reference_symmetry_order: int
    proper_moving_symmetry_order: int
    full_reference_symmetry_order: int
    full_moving_symmetry_order: int
    proper_orientation_intersection_order: int
    full_orientation_intersection_order: int
    orientation_variant_count: int
    orientation_operator_count: int
    cayron_topology_audit: CayronTopologyAudit
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.to_dict(),
            "audit": self.audit.to_dict(),
            "axis_angle": self.axis_angle.to_dict(),
            "quaternion_wxyz": list(self.quaternion_wxyz),
            "euler_zxz_active": self.euler_zxz_active.to_dict(),
            "euler_zxz_passive": self.euler_zxz_passive.to_dict(),
            "representations": [item.to_dict() for item in self.representations],
            "parity": self.parity.to_dict(),
            "proper_reference_symmetry_order": self.proper_reference_symmetry_order,
            "proper_moving_symmetry_order": self.proper_moving_symmetry_order,
            "full_reference_symmetry_order": self.full_reference_symmetry_order,
            "full_moving_symmetry_order": self.full_moving_symmetry_order,
            "proper_orientation_intersection_order": (
                self.proper_orientation_intersection_order
            ),
            "full_orientation_intersection_order": (
                self.full_orientation_intersection_order
            ),
            "orientation_variant_count": self.orientation_variant_count,
            "orientation_operator_count": self.orientation_operator_count,
            "cayron_topology_audit": self.cayron_topology_audit.to_dict(),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ParallelismCandidate:
    state: OrientationState
    first_parallelism_residual_deg: float
    second_parallelism_residual_deg: float

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.to_dict(),
            "first_parallelism_residual_deg": self.first_parallelism_residual_deg,
            "second_parallelism_residual_deg": self.second_parallelism_residual_deg,
        }


@dataclass(frozen=True)
class ParallelismSolveReport:
    reference_phase_id: str
    moving_phase_id: str
    reference_first: str
    moving_first: str
    reference_second: str
    moving_second: str
    unoriented_directions: bool
    reference_internal_angle_deg: float
    moving_internal_angle_deg: float
    internal_angle_mismatch_deg: float
    candidates: tuple[ParallelismCandidate, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_phase_id": self.reference_phase_id,
            "moving_phase_id": self.moving_phase_id,
            "reference_first": self.reference_first,
            "moving_first": self.moving_first,
            "reference_second": self.reference_second,
            "moving_second": self.moving_second,
            "unoriented_directions": self.unoriented_directions,
            "reference_internal_angle_deg": self.reference_internal_angle_deg,
            "moving_internal_angle_deg": self.moving_internal_angle_deg,
            "internal_angle_mismatch_deg": self.internal_angle_mismatch_deg,
            "candidate_count": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class OrientationComparisonReport:
    first_orientation_id: str
    second_orientation_id: str
    raw_misorientation_deg: float
    symmetry_reduced_disorientation_deg: float
    best_first_variant: int
    best_second_variant: int
    best_delta_axis_reference_cartesian: Vector3
    note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "first_orientation_id": self.first_orientation_id,
            "second_orientation_id": self.second_orientation_id,
            "raw_misorientation_deg": self.raw_misorientation_deg,
            "symmetry_reduced_disorientation_deg": (
                self.symmetry_reduced_disorientation_deg
            ),
            "best_first_variant": self.best_first_variant,
            "best_second_variant": self.best_second_variant,
            "best_delta_axis_reference_cartesian": list(
                self.best_delta_axis_reference_cartesian
            ),
            "note": self.note,
        }


@dataclass(frozen=True)
class TwoPhaseAngleReport:
    orientation_id: str
    reference_object: str
    moving_object: str
    relation: str
    angle_deg: float
    projective: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "orientation_id": self.orientation_id,
            "reference_object": self.reference_object,
            "moving_object": self.moving_object,
            "relation": self.relation,
            "angle_deg": self.angle_deg,
            "projective": self.projective,
        }


@dataclass(frozen=True)
class GeometricMisfitReport:
    orientation_id: str
    reference_object: str
    moving_object: str
    quantity: str
    reference_value: float
    moving_value: float
    signed_relative_percent: float
    absolute_relative_percent: float
    unit: str

    def to_dict(self) -> dict[str, object]:
        return {
            "orientation_id": self.orientation_id,
            "reference_object": self.reference_object,
            "moving_object": self.moving_object,
            "quantity": self.quantity,
            "reference_value": self.reference_value,
            "moving_value": self.moving_value,
            "signed_relative_percent": self.signed_relative_percent,
            "absolute_relative_percent": self.absolute_relative_percent,
            "unit": self.unit,
        }


def _matrix_tuple(matrix: np.ndarray) -> tuple[Vector3, Vector3, Vector3]:
    arr = np.asarray(matrix, dtype=float).reshape(3, 3)
    return tuple(tuple(float(value) for value in row) for row in arr)  # type: ignore[return-value]


def _vector_tuple(vector: np.ndarray) -> Vector3:
    arr = np.asarray(vector, dtype=float).reshape(3)
    return tuple(float(value) for value in arr)  # type: ignore[return-value]


def _relative_residual(lhs: np.ndarray, rhs: np.ndarray) -> float:
    a = np.asarray(lhs, dtype=float)
    b = np.asarray(rhs, dtype=float)
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0)
    return float(np.linalg.norm(a - b) / scale)


def _normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1.0e-15:
        raise ValueError("Cannot normalize a zero vector.")
    return arr / norm


def _clip(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def _angle_deg(a: np.ndarray, b: np.ndarray, *, projective: bool = False) -> float:
    cosine = float(_normalize(a) @ _normalize(b))
    if projective:
        cosine = abs(cosine)
    return float(np.rad2deg(np.arccos(_clip(cosine))))


def rotation_audit(matrix: np.ndarray) -> RotationAudit:
    R = np.asarray(matrix, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"Rotation matrix must be 3x3; got {R.shape}.")
    if not np.all(np.isfinite(R)):
        raise ValueError("Rotation matrix contains non-finite values.")

    orthogonality = _relative_residual(R.T @ R, np.eye(3))
    determinant = float(np.linalg.det(R))
    return RotationAudit(
        orthogonality_residual=orthogonality,
        determinant=determinant,
        determinant_residual=abs(determinant - 1.0),
    )


def closest_proper_rotation(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Project a finite 3x3 matrix onto SO(3); never used silently."""

    A = np.asarray(matrix, dtype=float)
    if A.shape != (3, 3) or not np.all(np.isfinite(A)):
        raise ValueError("Input must be a finite 3x3 matrix.")

    U, _, Vt = np.linalg.svd(A)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ correction @ Vt
    return R, _relative_residual(A, R)


def _rx(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=float,
    )


def _rz(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def matrix_from_axis_angle(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis_unit = _normalize(axis)
    angle = np.deg2rad(float(angle_deg))
    x, y, z = axis_unit
    K = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=float,
    )
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def axis_angle_from_matrix(matrix: np.ndarray) -> AxisAngle:
    R = np.asarray(matrix, dtype=float).reshape(3, 3)
    audit = rotation_audit(R)
    if audit.maximum_residual > 1.0e-8:
        raise ValueError(
            "axis_angle_from_matrix requires a proper rotation; "
            f"residual={audit.maximum_residual:.3e}."
        )

    angle_rad = math.acos(_clip((float(np.trace(R)) - 1.0) / 2.0))
    angle_deg = math.degrees(angle_rad)
    if angle_deg <= 1.0e-10:
        return AxisAngle((1.0, 0.0, 0.0), 0.0)

    if abs(180.0 - angle_deg) <= 1.0e-7:
        symmetric = 0.5 * (R + np.eye(3))
        values, vectors = np.linalg.eigh(symmetric)
        axis = _normalize(np.real(vectors[:, int(np.argmax(values))]))
        nonzero = np.flatnonzero(np.abs(axis) > 1.0e-12)
        if nonzero.size and axis[int(nonzero[0])] < 0.0:
            axis = -axis
        return AxisAngle(_vector_tuple(axis), 180.0)

    denominator = 2.0 * math.sin(angle_rad)
    axis = (
        np.array(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ],
            dtype=float,
        )
        / denominator
    )
    return AxisAngle(_vector_tuple(_normalize(axis)), angle_deg)


def matrix_from_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-15:
        raise ValueError("Quaternion must be non-zero.")
    w, x, y, z = q / norm

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )


def quaternion_wxyz_from_matrix(matrix: np.ndarray) -> Quaternion4:
    result = axis_angle_from_matrix(matrix)
    half = math.radians(result.angle_deg) / 2.0
    axis = np.asarray(result.axis, dtype=float)
    quaternion = np.concatenate(([math.cos(half)], axis * math.sin(half)))
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    if abs(quaternion[0]) <= 1.0e-14:
        nonzero = np.flatnonzero(np.abs(quaternion[1:]) > 1.0e-12)
        if nonzero.size and quaternion[1 + int(nonzero[0])] < 0.0:
            quaternion = -quaternion
    return tuple(float(value) for value in quaternion)  # type: ignore[return-value]


def matrix_from_euler_zxz(
    phi1_deg: float,
    Phi_deg: float,
    phi2_deg: float,
    convention: EulerConvention = EulerConvention.ZXZ_ACTIVE,
) -> np.ndarray:
    convention = EulerConvention(convention)
    active = _rz(phi1_deg) @ _rx(Phi_deg) @ _rz(phi2_deg)
    return active if convention is EulerConvention.ZXZ_ACTIVE else active.T


def euler_zxz_from_matrix(
    matrix: np.ndarray,
    convention: EulerConvention = EulerConvention.ZXZ_ACTIVE,
) -> EulerAngles:
    convention = EulerConvention(convention)
    R = np.asarray(matrix, dtype=float).reshape(3, 3)
    if convention is EulerConvention.ZXZ_PASSIVE:
        R = R.T

    audit = rotation_audit(R)
    if audit.maximum_residual > 1.0e-8:
        raise ValueError(
            "Euler extraction requires a proper rotation; "
            f"residual={audit.maximum_residual:.3e}."
        )

    Phi = math.degrees(math.acos(_clip(float(R[2, 2]))))
    sin_Phi = math.sin(math.radians(Phi))
    singular = abs(sin_Phi) <= 1.0e-10

    if not singular:
        phi1 = math.degrees(math.atan2(float(R[0, 2]), -float(R[1, 2])))
        phi2 = math.degrees(math.atan2(float(R[2, 0]), float(R[2, 1])))
    elif float(R[2, 2]) >= 0.0:
        phi1 = math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0])))
        Phi = 0.0
        phi2 = 0.0
    else:
        phi1 = math.degrees(math.atan2(float(R[0, 1]), float(R[0, 0])))
        Phi = 180.0
        phi2 = 0.0

    return EulerAngles(
        convention=convention,
        phi1_deg=phi1 % 360.0,
        Phi_deg=Phi,
        phi2_deg=phi2 % 360.0,
        singular=singular,
    )


def misorientation_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (
        np.asarray(first, dtype=float).reshape(3, 3)
        @ np.asarray(
            second,
            dtype=float,
        )
        .reshape(3, 3)
        .T
    )


def misorientation_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    return axis_angle_from_matrix(misorientation_matrix(first, second)).angle_deg


def _parse_numeric_vector(text: str, size: int) -> np.ndarray:
    cleaned = text.strip().replace(",", " ").replace(";", " ")
    tokens = cleaned.split()
    if len(tokens) != size:
        raise ValueError(f"Expected {size} numeric values; got {len(tokens)}.")
    values = np.asarray([float(token) for token in tokens], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Numeric input contains non-finite values.")
    return values


def _parse_matrix(text: str) -> np.ndarray:
    rows = [row.strip() for row in text.strip().split(";") if row.strip()]
    if len(rows) == 3:
        return np.vstack([_parse_numeric_vector(row, 3) for row in rows])
    return _parse_numeric_vector(text, 9).reshape(3, 3)


def _normalized_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _state_matrix(state: OrientationState) -> np.ndarray:
    return np.asarray(state.R_reference_from_moving, dtype=float)


def _object_physical_vector(
    obj: CrystalObject,
    phase: PhaseState,
    convention: CartesianConvention,
) -> np.ndarray:
    frame = CartesianFrame(phase.lattice, convention)
    if isinstance(obj, Direction):
        return frame.direct_to_cartesian(obj.array, normalize=True)
    return frame.plane_to_cartesian(obj.array, normalize=True)


def _object_sign_options(
    obj: CrystalObject,
    *,
    unoriented_directions: bool,
) -> tuple[float, ...]:
    if isinstance(obj, Plane) or unoriented_directions:
        return (-1.0, 1.0)
    return (1.0,)


def _triad_from_two_vectors(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    e1 = _normalize(first)
    second_perpendicular = second - float(second @ e1) * e1
    e2 = _normalize(second_perpendicular)
    e3 = _normalize(np.cross(e1, e2))
    return np.column_stack((e1, e2, e3))


def _rotation_from_two_vector_pairs(
    source_first: np.ndarray,
    source_second: np.ndarray,
    target_first: np.ndarray,
    target_second: np.ndarray,
    *,
    angle_tolerance_deg: float,
) -> np.ndarray:
    source_angle = _angle_deg(source_first, source_second)
    target_angle = _angle_deg(target_first, target_second)
    if abs(source_angle - target_angle) > angle_tolerance_deg:
        raise ValueError(
            "The two vector pairs do not have the same internal angle: "
            f"source={source_angle:.12g} deg, target={target_angle:.12g} deg."
        )

    source_triad = _triad_from_two_vectors(source_first, source_second)
    target_triad = _triad_from_two_vectors(target_first, target_second)
    R = target_triad @ source_triad.T
    if rotation_audit(R).maximum_residual > 1.0e-10:
        raise AssertionError("Two-vector construction failed to produce SO(3).")
    return R


class OrientationService:
    """Phase-aware OR service with explicit correspondence/orientation separation."""

    def __init__(self, project: ProjectState):
        self.project = project
        project.validate().assert_passed()
        self.console = CrystallographyConsole(project)

    def resolve_phase(self, query: str) -> PhaseState:
        return self.console.resolve_phase(query)

    @staticmethod
    def _state_id(
        reference: PhaseState,
        moving: PhaseState,
        method: str,
        requested: str,
    ) -> str:
        if requested.strip():
            return requested.strip()
        return (
            "or_"
            + _normalized_key(reference.phase_id)
            + "_from_"
            + _normalized_key(moving.phase_id)
            + "_"
            + _normalized_key(method)
        )

    def state_from_matrix(
        self,
        reference_phase: str,
        moving_phase: str,
        matrix_reference_from_moving: np.ndarray,
        *,
        orientation_id: str = "",
        label: str = "",
        reference_convention: CartesianConvention = CartesianConvention.PTCLAB_A_X_C_XZ,
        moving_convention: CartesianConvention = CartesianConvention.PTCLAB_A_X_C_XZ,
        definition_method: OrientationDefinition = OrientationDefinition.USER_MATRIX,
        theory_origin: OrientationTheoryOrigin = OrientationTheoryOrigin.USER_DEFINED,
        provenance: StateProvenance | None = None,
        transformation_id: str = "",
        repair: bool = False,
        maximum_repair_residual: float = 1.0e-3,
        notes: str = "",
    ) -> OrientationState:
        reference = self.resolve_phase(reference_phase)
        moving = self.resolve_phase(moving_phase)
        if reference.phase_id == moving.phase_id:
            raise ValueError("An OR must connect two distinct registered phases.")

        R = np.asarray(matrix_reference_from_moving, dtype=float).reshape(3, 3)
        audit = rotation_audit(R)
        repair_note = ""

        if audit.maximum_residual > self.project.numerical_policy.representation:
            if not repair:
                raise ValueError(
                    "Input matrix is not a proper rotation within project tolerance; "
                    f"residual={audit.maximum_residual:.3e}. Use explicit repair=True."
                )
            projected, correction = closest_proper_rotation(R)
            if correction > maximum_repair_residual:
                raise ValueError(
                    "Requested SO(3) repair is too large: "
                    f"{correction:.3e} > {maximum_repair_residual:.3e}."
                )
            R = projected
            repair_note = f" Explicit SO(3) projection applied; correction residual={correction:.3e}."

        method = OrientationDefinition(definition_method)
        origin = OrientationTheoryOrigin(theory_origin)
        return OrientationState(
            orientation_id=self._state_id(
                reference,
                moving,
                method.value,
                orientation_id,
            ),
            label=label or f"{reference.label} <- {moving.label} OR",
            reference_phase_id=reference.phase_id,
            moving_phase_id=moving.phase_id,
            R_reference_from_moving=_matrix_tuple(R),
            reference_cartesian_convention=CartesianConvention(reference_convention),
            moving_cartesian_convention=CartesianConvention(moving_convention),
            definition_method=method,
            theory_origin=origin,
            provenance=provenance or StateProvenance(),
            transformation_id=transformation_id,
            notes=(notes + repair_note).strip(),
        )

    def state_from_axis_angle(
        self,
        reference_phase: str,
        moving_phase: str,
        axis_reference_cartesian: np.ndarray,
        angle_deg: float,
        **kwargs: object,
    ) -> OrientationState:
        return self.state_from_matrix(
            reference_phase,
            moving_phase,
            matrix_from_axis_angle(axis_reference_cartesian, angle_deg),
            definition_method=OrientationDefinition.AXIS_ANGLE,
            **kwargs,
        )

    def state_from_reference_crystal_axis(
        self,
        reference_phase: str,
        moving_phase: str,
        axis_direction_text: str,
        angle_deg: float,
        *,
        reference_convention: CartesianConvention = CartesianConvention.PTCLAB_A_X_C_XZ,
        **kwargs: object,
    ) -> OrientationState:
        reference = self.resolve_phase(reference_phase)
        parsed = parse_crystal_input(axis_direction_text)
        if parsed.kind is not ConsoleInputKind.DIRECTION:
            raise ValueError("Axis must be a direct direction [uvw].")
        direction = Direction(parsed.indices, reference.basis)
        axis = CartesianFrame(
            reference.lattice,
            reference_convention,
        ).direct_to_cartesian(direction.array, normalize=True)
        return self.state_from_axis_angle(
            reference.phase_id,
            moving_phase,
            axis,
            angle_deg,
            reference_convention=reference_convention,
            **kwargs,
        )

    def state_from_quaternion(
        self,
        reference_phase: str,
        moving_phase: str,
        quaternion_wxyz: np.ndarray,
        **kwargs: object,
    ) -> OrientationState:
        return self.state_from_matrix(
            reference_phase,
            moving_phase,
            matrix_from_quaternion_wxyz(quaternion_wxyz),
            definition_method=OrientationDefinition.CARTESIAN_HAMILTON_QUATERNION,
            **kwargs,
        )

    def state_from_euler(
        self,
        reference_phase: str,
        moving_phase: str,
        phi1_deg: float,
        Phi_deg: float,
        phi2_deg: float,
        *,
        convention: EulerConvention = EulerConvention.ZXZ_ACTIVE,
        **kwargs: object,
    ) -> OrientationState:
        euler_convention = EulerConvention(convention)
        definition = (
            OrientationDefinition.EULER_ZXZ_ACTIVE
            if euler_convention is EulerConvention.ZXZ_ACTIVE
            else OrientationDefinition.EULER_ZXZ_PASSIVE
        )
        return self.state_from_matrix(
            reference_phase,
            moving_phase,
            matrix_from_euler_zxz(
                phi1_deg,
                Phi_deg,
                phi2_deg,
                euler_convention,
            ),
            definition_method=definition,
            **kwargs,
        )

    def reexpress(
        self,
        state: OrientationState,
        reference_convention: CartesianConvention,
        moving_convention: CartesianConvention,
    ) -> OrientationState:
        reference = self.project.phase(state.reference_phase_id)
        moving = self.project.phase(state.moving_phase_id)

        Q_reference = frame_rotation(
            CartesianFrame(
                reference.lattice,
                state.reference_cartesian_convention,
            ),
            CartesianFrame(reference.lattice, reference_convention),
        )
        Q_moving = frame_rotation(
            CartesianFrame(
                moving.lattice,
                state.moving_cartesian_convention,
            ),
            CartesianFrame(moving.lattice, moving_convention),
        )
        R_new = Q_reference @ _state_matrix(state) @ Q_moving.T

        return replace(
            state,
            R_reference_from_moving=_matrix_tuple(R_new),
            reference_cartesian_convention=CartesianConvention(reference_convention),
            moving_cartesian_convention=CartesianConvention(moving_convention),
        )

    def map_direction(self, state: OrientationState, direction: Direction) -> Direction:
        reference = self.project.phase(state.reference_phase_id)
        moving = self.project.phase(state.moving_phase_id)
        R = _state_matrix(state)
        reference_frame = CartesianFrame(
            reference.lattice,
            state.reference_cartesian_convention,
        )
        moving_frame = CartesianFrame(
            moving.lattice,
            state.moving_cartesian_convention,
        )

        if direction.basis == moving.basis:
            mapped = reference_frame.direct_from_cartesian(
                R @ moving_frame.direct_to_cartesian(direction.array)
            )
            target_basis = reference.basis
            arrow = f"{moving.phase_id} -> {reference.phase_id}"
        elif direction.basis == reference.basis:
            mapped = moving_frame.direct_from_cartesian(
                R.T @ reference_frame.direct_to_cartesian(direction.array)
            )
            target_basis = moving.basis
            arrow = f"{reference.phase_id} -> {moving.phase_id}"
        else:
            raise ValueError("Direction basis is not an endpoint of the OR.")

        return Direction(
            _vector_tuple(mapped),
            target_basis,
            label=direction.label,
            provenance=ObjectProvenance(
                status=DataStatus.COMPUTATION_DERIVED,
                source_key=state.provenance.source_key,
                uncertainty=direction.provenance.uncertainty,
                notes=(
                    f"Physical OR mapping {arrow} via {state.orientation_id}; "
                    "this is not correspondence mapping."
                ),
            ),
        )

    def map_plane(self, state: OrientationState, plane: Plane) -> Plane:
        reference = self.project.phase(state.reference_phase_id)
        moving = self.project.phase(state.moving_phase_id)
        R = _state_matrix(state)
        reference_frame = CartesianFrame(
            reference.lattice,
            state.reference_cartesian_convention,
        )
        moving_frame = CartesianFrame(
            moving.lattice,
            state.moving_cartesian_convention,
        )

        if plane.basis == moving.basis:
            mapped = reference_frame.plane_from_cartesian(
                R @ moving_frame.plane_to_cartesian(plane.array)
            )
            target_basis = reference.basis
            arrow = f"{moving.phase_id} -> {reference.phase_id}"
        elif plane.basis == reference.basis:
            mapped = moving_frame.plane_from_cartesian(
                R.T @ reference_frame.plane_to_cartesian(plane.array)
            )
            target_basis = moving.basis
            arrow = f"{reference.phase_id} -> {moving.phase_id}"
        else:
            raise ValueError("Plane basis is not an endpoint of the OR.")

        return Plane(
            _vector_tuple(mapped),
            target_basis,
            label=plane.label,
            provenance=ObjectProvenance(
                status=DataStatus.COMPUTATION_DERIVED,
                source_key=state.provenance.source_key,
                uncertainty=plane.provenance.uncertainty,
                notes=(
                    f"Physical OR mapping {arrow} via {state.orientation_id}; "
                    "this is not correspondence mapping."
                ),
            ),
        )

    def _mapped_coefficients(
        self,
        state: OrientationState,
        *,
        direction: bool,
        values: np.ndarray,
    ) -> np.ndarray:
        moving = self.project.phase(state.moving_phase_id)
        if direction:
            return self.map_direction(
                state,
                Direction(_vector_tuple(values), moving.basis),
            ).array
        return self.map_plane(
            state,
            Plane(_vector_tuple(values), moving.basis),
        ).array

    def representation_parity(
        self,
        state: OrientationState,
    ) -> OrientationParityAudit:
        test_direction = np.array([1.0, 2.0, -1.0])
        test_plane = np.array([2.0, -1.0, 1.0])
        base_direction = self._mapped_coefficients(
            state,
            direction=True,
            values=test_direction,
        )
        base_plane = self._mapped_coefficients(
            state,
            direction=False,
            values=test_plane,
        )

        direction_residuals: list[float] = []
        plane_residuals: list[float] = []
        rotation_residuals: list[float] = []

        for reference_convention in CartesianConvention:
            for moving_convention in CartesianConvention:
                current = self.reexpress(
                    state,
                    reference_convention,
                    moving_convention,
                )
                direction_residuals.append(
                    _relative_residual(
                        base_direction,
                        self._mapped_coefficients(
                            current,
                            direction=True,
                            values=test_direction,
                        ),
                    )
                )
                plane_residuals.append(
                    _relative_residual(
                        base_plane,
                        self._mapped_coefficients(
                            current,
                            direction=False,
                            values=test_plane,
                        ),
                    )
                )
                rotation_residuals.append(
                    rotation_audit(_state_matrix(current)).maximum_residual
                )

        return OrientationParityAudit(
            maximum_direction_mapping_residual=max(direction_residuals),
            maximum_plane_mapping_residual=max(plane_residuals),
            maximum_rotation_residual=max(rotation_residuals),
        )

    def proper_symmetry_cartesian(
        self,
        phase: PhaseState,
        convention: CartesianConvention,
    ) -> tuple[tuple[int, np.ndarray], ...]:
        frame = CartesianFrame(phase.lattice, convention)
        proper: list[tuple[int, np.ndarray]] = []
        for index, operator in enumerate(phase.symmetry_matrices()):
            if float(np.linalg.det(operator)) <= 0.0:
                continue
            cartesian = frame.operator_to_cartesian(operator)
            audit = rotation_audit(cartesian)
            if audit.maximum_residual > self.project.numerical_policy.representation:
                raise AssertionError(
                    f"Proper symmetry {phase.phase_id}[{index}] is not SO(3); "
                    f"residual={audit.maximum_residual:.3e}."
                )
            proper.append((index, cartesian))
        if not proper:
            raise ValueError(f"Phase {phase.phase_id!r} has no proper symmetry.")
        return tuple(proper)

    def topology(self, state: OrientationState):
        """Return Cayron-style H_T, cosets, double cosets and audit data."""

        return build_orientation_topology(self.project, state)

    def variants(self, state: OrientationState) -> tuple[OrientationVariant, ...]:
        """Return one SO(3) representative per proper orientation left coset.

        The previous implementation emitted every distinct raw product
        S_A R S_M^-1.  That over-counts crystallographically equivalent
        matrices.  Cayron's definition is the left-coset quotient G_A/H_T.
        """

        topology = self.topology(state)
        return tuple(
            OrientationVariant(
                index=item.index,
                reference_symmetry_index=(item.representative_reference_symmetry_index),
                moving_symmetry_index=-1,
                matrix_reference_from_moving=item.matrix_reference_from_moving,
                misorientation_from_base_deg=(
                    item.crystallographic_disorientation_from_base_deg
                ),
                reference_coset_symmetry_indices=(
                    item.reference_coset_symmetry_indices
                ),
                equivalent_proper_matrix_count=(item.equivalent_proper_matrix_count),
                raw_representative_angle_from_base_deg=(
                    item.raw_representative_angle_from_base_deg
                ),
            )
            for item in topology.variants
        )

    def operators(
        self,
        state: OrientationState,
    ) -> tuple[OrientationOperatorClass, ...]:
        """Return Cayron-style full-point-group orientation double cosets."""

        return self.topology(state).operators

    def report(self, state: OrientationState) -> OrientationReport:
        representations: list[OrientationRepresentation] = []
        for reference_convention in CartesianConvention:
            for moving_convention in CartesianConvention:
                current = self.reexpress(
                    state,
                    reference_convention,
                    moving_convention,
                )
                matrix = _state_matrix(current)
                representations.append(
                    OrientationRepresentation(
                        reference_convention=reference_convention.value,
                        moving_convention=moving_convention.value,
                        matrix_reference_from_moving=_matrix_tuple(matrix),
                        audit=rotation_audit(matrix),
                    )
                )

        self.project.phase(state.reference_phase_id)
        self.project.phase(state.moving_phase_id)
        topology = self.topology(state)
        active_euler = euler_zxz_from_matrix(
            _state_matrix(state),
            EulerConvention.ZXZ_ACTIVE,
        )
        passive_euler = euler_zxz_from_matrix(
            _state_matrix(state),
            EulerConvention.ZXZ_PASSIVE,
        )
        warnings = [
            (
                "Quaternion output is a standard Cartesian Hamilton quaternion "
                "(w,x,y,z), not Cayron's crystallographic metric quaternion."
            ),
            (
                "Axis-angle, Euler angles and the displayed Hamilton quaternion "
                "parameterize this chosen parent/product Cartesian-frame pair. "
                "They are not invariant under independent re-expression of the "
                "two phase frames; physical mapping and symmetry-reduced OR "
                "comparison are the invariant checks."
            ),
            (
                "Orientation variants are Cayron-style left cosets G_A/H_T; "
                "orientation operators are double cosets H_T g H_T. Raw "
                "S_A R S_M^-1 matrices are not counted as distinct variants."
            ),
        ]
        if state.theory_origin is OrientationTheoryOrigin.POLAR_CORRESPONDENCE:
            warnings.append(
                "The polar rotation of the correspondence deformation is a "
                "finite-strain rotation candidate; it is not automatically "
                "Cayron's orientation matrix T, a Ball-James rotation, PTMC OR, "
                "or an experimental OR."
            )
        if active_euler.singular or passive_euler.singular:
            warnings.append(
                "Euler representation is singular at Phi=0 or 180 deg; matrix, "
                "axis-angle and quaternion remain well-defined."
            )

        return OrientationReport(
            state=state,
            audit=rotation_audit(_state_matrix(state)),
            axis_angle=axis_angle_from_matrix(_state_matrix(state)),
            quaternion_wxyz=quaternion_wxyz_from_matrix(_state_matrix(state)),
            euler_zxz_active=active_euler,
            euler_zxz_passive=passive_euler,
            representations=tuple(representations),
            parity=self.representation_parity(state),
            proper_reference_symmetry_order=(
                topology.audit.proper_reference_group_order
            ),
            proper_moving_symmetry_order=topology.audit.proper_moving_group_order,
            full_reference_symmetry_order=topology.audit.full_reference_group_order,
            full_moving_symmetry_order=topology.audit.full_moving_group_order,
            proper_orientation_intersection_order=(
                topology.audit.proper_orientation_intersection_order
            ),
            full_orientation_intersection_order=(
                topology.audit.full_orientation_intersection_order
            ),
            orientation_variant_count=(topology.audit.proper_orientation_variant_count),
            orientation_operator_count=topology.audit.full_orientation_operator_count,
            cayron_topology_audit=topology.audit,
            warnings=tuple(warnings),
        )

    def state_from_parallelisms(
        self,
        reference_phase: str,
        moving_phase: str,
        reference_first: str,
        moving_first: str,
        reference_second: str,
        moving_second: str,
        *,
        orientation_id: str = "",
        label: str = "",
        unoriented_directions: bool = True,
        reference_convention: CartesianConvention = CartesianConvention.PTCLAB_A_X_C_XZ,
        moving_convention: CartesianConvention = CartesianConvention.PTCLAB_A_X_C_XZ,
        theory_origin: OrientationTheoryOrigin = OrientationTheoryOrigin.USER_DEFINED,
        provenance: StateProvenance | None = None,
    ) -> ParallelismSolveReport:
        reference = self.resolve_phase(reference_phase)
        moving = self.resolve_phase(moving_phase)

        parsed = [
            parse_crystal_input(reference_first),
            parse_crystal_input(moving_first),
            parse_crystal_input(reference_second),
            parse_crystal_input(moving_second),
        ]
        ref_first_p, mov_first_p, ref_second_p, mov_second_p = parsed

        if ref_first_p.kind is not mov_first_p.kind:
            raise ValueError("First parallelism must compare like object types.")
        if ref_second_p.kind is not mov_second_p.kind:
            raise ValueError("Second parallelism must compare like object types.")

        def make_object(item: object, phase: PhaseState) -> CrystalObject:
            if not hasattr(item, "kind") or not hasattr(item, "indices"):
                raise TypeError("Parsed crystallographic object is invalid.")
            if item.kind is ConsoleInputKind.DIRECTION:
                return Direction(item.indices, phase.basis)
            return Plane(item.indices, phase.basis)

        ref_first_obj = make_object(ref_first_p, reference)
        mov_first_obj = make_object(mov_first_p, moving)
        ref_second_obj = make_object(ref_second_p, reference)
        mov_second_obj = make_object(mov_second_p, moving)

        ref_first_v = _object_physical_vector(
            ref_first_obj,
            reference,
            reference_convention,
        )
        mov_first_v = _object_physical_vector(
            mov_first_obj,
            moving,
            moving_convention,
        )
        ref_second_v = _object_physical_vector(
            ref_second_obj,
            reference,
            reference_convention,
        )
        mov_second_v = _object_physical_vector(
            mov_second_obj,
            moving,
            moving_convention,
        )

        ref_internal = _angle_deg(ref_first_v, ref_second_v, projective=True)
        mov_internal = _angle_deg(mov_first_v, mov_second_v, projective=True)
        internal_mismatch = abs(ref_internal - mov_internal)

        if (
            abs(float(ref_first_v @ ref_second_v)) >= 1.0 - 1.0e-12
            or abs(float(mov_first_v @ mov_second_v)) >= 1.0 - 1.0e-12
        ):
            raise ValueError("Defining object pairs must be non-collinear.")

        sign_sets = (
            _object_sign_options(
                ref_first_obj,
                unoriented_directions=unoriented_directions,
            ),
            _object_sign_options(
                mov_first_obj,
                unoriented_directions=unoriented_directions,
            ),
            _object_sign_options(
                ref_second_obj,
                unoriented_directions=unoriented_directions,
            ),
            _object_sign_options(
                mov_second_obj,
                unoriented_directions=unoriented_directions,
            ),
        )
        tolerance_deg = self.project.numerical_policy.projective_angle_deg
        unique_matrices: list[np.ndarray] = []

        for sr1, sm1, sr2, sm2 in product(*sign_sets):
            target_first = sr1 * ref_first_v
            source_first = sm1 * mov_first_v
            target_second = sr2 * ref_second_v
            source_second = sm2 * mov_second_v

            if (
                abs(
                    _angle_deg(source_first, source_second)
                    - _angle_deg(target_first, target_second)
                )
                > tolerance_deg
            ):
                continue

            candidate = _rotation_from_two_vector_pairs(
                source_first,
                source_second,
                target_first,
                target_second,
                angle_tolerance_deg=tolerance_deg,
            )
            if any(
                _relative_residual(candidate, existing)
                <= self.project.numerical_policy.representation
                for existing in unique_matrices
            ):
                continue
            unique_matrices.append(candidate)

        if not unique_matrices:
            raise ValueError(
                "No exact proper rotation satisfies both parallelisms. "
                f"Acute internal-angle mismatch={internal_mismatch:.12g} deg."
            )

        def residual(
            R: np.ndarray,
            reference_object: CrystalObject,
            reference_vector: np.ndarray,
            moving_vector: np.ndarray,
        ) -> float:
            projective = isinstance(reference_object, Plane) or (
                isinstance(reference_object, Direction) and unoriented_directions
            )
            return _angle_deg(
                reference_vector,
                R @ moving_vector,
                projective=projective,
            )

        candidates: list[ParallelismCandidate] = []
        for index, matrix in enumerate(unique_matrices, start=1):
            state = self.state_from_matrix(
                reference.phase_id,
                moving.phase_id,
                matrix,
                orientation_id=(
                    f"{orientation_id}_{index:02d}"
                    if orientation_id
                    else f"or_parallel_{index:02d}"
                ),
                label=label or "OR from two crystallographic parallelisms",
                reference_convention=reference_convention,
                moving_convention=moving_convention,
                definition_method=OrientationDefinition.PARALLELISMS,
                theory_origin=theory_origin,
                provenance=provenance,
                notes=(
                    "Plane signs are intrinsically unoriented; direction signs "
                    + (
                        "are treated as axes."
                        if unoriented_directions
                        else "are treated as oriented vectors."
                    )
                ),
            )
            candidates.append(
                ParallelismCandidate(
                    state=state,
                    first_parallelism_residual_deg=residual(
                        matrix,
                        ref_first_obj,
                        ref_first_v,
                        mov_first_v,
                    ),
                    second_parallelism_residual_deg=residual(
                        matrix,
                        ref_second_obj,
                        ref_second_v,
                        mov_second_v,
                    ),
                )
            )

        return ParallelismSolveReport(
            reference_phase_id=reference.phase_id,
            moving_phase_id=moving.phase_id,
            reference_first=ref_first_p.canonical_text,
            moving_first=mov_first_p.canonical_text,
            reference_second=ref_second_p.canonical_text,
            moving_second=mov_second_p.canonical_text,
            unoriented_directions=unoriented_directions,
            reference_internal_angle_deg=ref_internal,
            moving_internal_angle_deg=mov_internal,
            internal_angle_mismatch_deg=internal_mismatch,
            candidates=tuple(candidates),
        )

    def polar_orientation(
        self,
        transformation_id: str,
        *,
        orientation_id: str = "",
    ) -> OrientationState:
        transformation = self.project.transformation(transformation_id)
        bridge = self.project.bridge(transformation_id)
        R_product_from_parent = bridge.polar_rotation_cartesian()

        return self.state_from_matrix(
            transformation.parent_phase_id,
            transformation.product_phase_id,
            R_product_from_parent.T,
            orientation_id=(
                orientation_id or f"polar_{transformation.transformation_id}"
            ),
            label="Polar rotation of correspondence-derived deformation gradient",
            reference_convention=transformation.parent_cartesian_convention,
            moving_convention=transformation.product_cartesian_convention,
            definition_method=OrientationDefinition.POLAR_CORRESPONDENCE,
            theory_origin=OrientationTheoryOrigin.POLAR_CORRESPONDENCE,
            transformation_id=transformation.transformation_id,
            provenance=StateProvenance(
                DataStatus.COMPUTATION_DERIVED,
                transformation.provenance.source_key,
                notes=(
                    "Derived from F = B_M C B_A^-1 and F = R_polar U; "
                    "stored OR is R_A<-M = R_polar^T."
                ),
            ),
            notes=(
                "Finite-strain polar rotation from correspondence and metrics; "
                "not silently identified with Cayron T, Ball-James R, PTMC OR, "
                "or experiment."
            ),
        )

    def compare_orientations(
        self,
        first: OrientationState,
        second: OrientationState,
    ) -> OrientationComparisonReport:
        if (
            first.reference_phase_id != second.reference_phase_id
            or first.moving_phase_id != second.moving_phase_id
        ):
            raise ValueError("OR comparison requires the same ordered phase pair.")

        second_common = self.reexpress(
            second,
            first.reference_cartesian_convention,
            first.moving_cartesian_convention,
        )
        raw_angle = misorientation_angle_deg(
            _state_matrix(first),
            _state_matrix(second_common),
        )

        first_variants = self.variants(first)
        second_variants = self.variants(second_common)

        # A crystallographic OR is a quotient object. A moving-phase crystal
        # symmetry changes only the representative of the same product
        # orientation:
        #
        #     R ~ R S_M^{-1}.
        #
        # Consequently the symmetry-reduced distance between two ORs must
        # minimize not only over parent-derived orientation variants, but also
        # over the relative proper moving-phase symmetry:
        #
        #     min angle(A S_M B^T).
        #
        # The former implementation compared only A B^T for the chosen coset
        # representatives. That quantity depends on the arbitrary moving
        # crystal representative and can overestimate the true quotient
        # disorientation.
        moving_phase = self.project.phase(first.moving_phase_id)
        moving_symmetries = self.proper_symmetry_cartesian(
            moving_phase,
            first.moving_cartesian_convention,
        )

        best_angle = float("inf")
        best_first = 0
        best_second = 0
        best_delta = np.eye(3)
        best_key: tuple[float, int, int, int] | None = None

        for first_variant in first_variants:
            A = np.asarray(
                first_variant.matrix_reference_from_moving,
                dtype=float,
            )
            for second_variant in second_variants:
                B = np.asarray(
                    second_variant.matrix_reference_from_moving,
                    dtype=float,
                )
                for moving_symmetry_index, S_M in moving_symmetries:
                    delta = A @ np.asarray(S_M, dtype=float) @ B.T

                    # Stable principal SO(3) angle. Using atan2 avoids the
                    # loss of sensitivity of acos((tr(R)-1)/2) near 0 deg,
                    # while remaining well-conditioned near 180 deg.
                    skew = 0.5 * np.array(
                        [
                            delta[2, 1] - delta[1, 2],
                            delta[0, 2] - delta[2, 0],
                            delta[1, 0] - delta[0, 1],
                        ],
                        dtype=float,
                    )
                    sine = float(np.linalg.norm(skew))
                    cosine = float(
                        np.clip((float(np.trace(delta)) - 1.0) / 2.0, -1.0, 1.0)
                    )
                    angle = math.degrees(math.atan2(sine, cosine))

                    key = (
                        angle,
                        first_variant.index,
                        second_variant.index,
                        moving_symmetry_index,
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        best_angle = angle
                        best_first = first_variant.index
                        best_second = second_variant.index
                        best_delta = delta

        return OrientationComparisonReport(
            first_orientation_id=first.orientation_id,
            second_orientation_id=second.orientation_id,
            raw_misorientation_deg=raw_angle,
            symmetry_reduced_disorientation_deg=best_angle,
            best_first_variant=best_first,
            best_second_variant=best_second,
            best_delta_axis_reference_cartesian=axis_angle_from_matrix(best_delta).axis,
            note=(
                "OR agreement is one observable only; it does not by itself "
                "establish whether a complete transformation theory is correct."
            ),
        )

    def compare_two_phase_objects(
        self,
        state: OrientationState,
        reference_text: str,
        moving_text: str,
        *,
        projective: bool = True,
    ) -> TwoPhaseAngleReport:
        reference = self.project.phase(state.reference_phase_id)
        moving = self.project.phase(state.moving_phase_id)
        ref_parsed = parse_crystal_input(reference_text)
        mov_parsed = parse_crystal_input(moving_text)

        ref_obj: CrystalObject = (
            Direction(ref_parsed.indices, reference.basis)
            if ref_parsed.kind is ConsoleInputKind.DIRECTION
            else Plane(ref_parsed.indices, reference.basis)
        )
        mov_obj: CrystalObject = (
            Direction(mov_parsed.indices, moving.basis)
            if mov_parsed.kind is ConsoleInputKind.DIRECTION
            else Plane(mov_parsed.indices, moving.basis)
        )

        ref_vector = _object_physical_vector(
            ref_obj,
            reference,
            state.reference_cartesian_convention,
        )
        mov_vector = _state_matrix(state) @ _object_physical_vector(
            mov_obj,
            moving,
            state.moving_cartesian_convention,
        )

        if isinstance(ref_obj, Direction) and isinstance(mov_obj, Direction):
            relation = "direction-direction"
            angle = _angle_deg(ref_vector, mov_vector, projective=projective)
        elif isinstance(ref_obj, Plane) and isinstance(mov_obj, Plane):
            relation = "plane-normal/plane-normal"
            angle = _angle_deg(ref_vector, mov_vector, projective=projective)
        else:
            relation = "direction-plane"
            cosine_to_normal = abs(
                float(_normalize(ref_vector) @ _normalize(mov_vector))
            )
            angle = abs(90.0 - math.degrees(math.acos(_clip(cosine_to_normal))))
            projective = True

        return TwoPhaseAngleReport(
            orientation_id=state.orientation_id,
            reference_object=ref_parsed.canonical_text,
            moving_object=mov_parsed.canonical_text,
            relation=relation,
            angle_deg=angle,
            projective=projective,
        )

    def geometric_misfit(
        self,
        state: OrientationState,
        reference_text: str,
        moving_text: str,
    ) -> GeometricMisfitReport:
        reference = self.project.phase(state.reference_phase_id)
        moving = self.project.phase(state.moving_phase_id)
        ref_parsed = parse_crystal_input(reference_text)
        mov_parsed = parse_crystal_input(moving_text)

        if ref_parsed.kind is not mov_parsed.kind:
            raise ValueError("Misfit requires direction-direction or plane-plane.")

        if reference.lattice.length_unit != moving.lattice.length_unit:
            raise ValueError(
                "Cross-phase misfit requires matching explicit length units."
            )

        if ref_parsed.kind is ConsoleInputKind.DIRECTION:
            reference_value = Direction(
                ref_parsed.indices,
                reference.basis,
            ).length(reference.lattice)
            moving_value = Direction(
                mov_parsed.indices,
                moving.basis,
            ).length(moving.lattice)
            quantity = "direct_length"
        else:
            reference_value = Plane(
                ref_parsed.indices,
                reference.basis,
            ).spacing(reference.lattice)
            moving_value = Plane(
                mov_parsed.indices,
                moving.basis,
            ).spacing(moving.lattice)
            quantity = "interplanar_spacing"

        signed = 100.0 * (moving_value - reference_value) / reference_value
        return GeometricMisfitReport(
            orientation_id=state.orientation_id,
            reference_object=ref_parsed.canonical_text,
            moving_object=mov_parsed.canonical_text,
            quantity=quantity,
            reference_value=reference_value,
            moving_value=moving_value,
            signed_relative_percent=signed,
            absolute_relative_percent=abs(signed),
            unit=reference.lattice.length_unit,
        )


class OrientationRenderer:
    """Readable renderer with conventions and scientific warnings visible."""

    @staticmethod
    def _matrix(matrix: np.ndarray) -> list[str]:
        values = np.asarray(matrix, dtype=float).copy()
        values[np.abs(values) < 1.0e-14] = 0.0
        return [
            "    [" + "  ".join(f"{value: .10g}" for value in row) + "]"
            for row in values
        ]

    def orientation(
        self,
        report: OrientationReport,
        *,
        show_representations: bool = False,
        variants: tuple[OrientationVariant, ...] = (),
        operators: tuple[OrientationOperatorClass, ...] = (),
    ) -> str:
        state = report.state
        title = (
            "POLAR ROTATION CANDIDATE"
            if state.theory_origin is OrientationTheoryOrigin.POLAR_CORRESPONDENCE
            else "ORIENTATION RELATIONSHIP"
        )
        lines = [
            "=" * 88,
            f"{title}  |  {state.orientation_id}",
            "=" * 88,
            "CONVENTION",
            (
                f"  x_{state.reference_phase_id} = "
                f"R_{state.reference_phase_id}<-{state.moving_phase_id} "
                f"x_{state.moving_phase_id}"
            ),
            "  IMPORTANT: physical orientation R is not crystallographic correspondence C.",
            "",
            f"reference phase   : {state.reference_phase_id}",
            f"moving phase      : {state.moving_phase_id}",
            f"definition        : {state.definition_method.value}",
            f"theory origin     : {state.theory_origin.value}",
            f"reference frame   : {state.reference_cartesian_convention.value}",
            f"moving frame      : {state.moving_cartesian_convention.value}",
            "",
            "R_reference<-moving",
            *self._matrix(_state_matrix(state)),
            "",
            "ROTATION AUDIT",
            f"  det(R)                    : {report.audit.determinant:.12g}",
            (
                f"  ||R^T R - I|| relative    : "
                f"{report.audit.orthogonality_residual:.3e}"
            ),
            f"  |det(R)-1|                : {report.audit.determinant_residual:.3e}",
            "",
            "COORDINATE AXIS / ANGLE  (current reference Cartesian frame)",
            f"  axis                       : {np.array(report.axis_angle.axis)}",
            f"  angle                      : {report.axis_angle.angle_deg:.12g} deg",
            "",
            "STANDARD CARTESIAN HAMILTON QUATERNION",
            f"  (w,x,y,z)                  : {report.quaternion_wxyz}",
            "",
            "EULER ZXZ — EXPLICIT FORMULAS",
            (
                "  active R=Rz(phi1)Rx(Phi)Rz(phi2): "
                f"({report.euler_zxz_active.phi1_deg:.9g}, "
                f"{report.euler_zxz_active.Phi_deg:.9g}, "
                f"{report.euler_zxz_active.phi2_deg:.9g}) deg"
            ),
            (
                "  passive = active^T: "
                f"({report.euler_zxz_passive.phi1_deg:.9g}, "
                f"{report.euler_zxz_passive.Phi_deg:.9g}, "
                f"{report.euler_zxz_passive.phi2_deg:.9g}) deg"
            ),
            "",
            "SYMMETRY / REPRESENTATION",
            (
                f"  full point-group orders     : "
                f"{report.full_reference_symmetry_order} x "
                f"{report.full_moving_symmetry_order}"
            ),
            (
                f"  proper SO(3) subgroup orders: "
                f"{report.proper_reference_symmetry_order} x "
                f"{report.proper_moving_symmetry_order}"
            ),
            (
                f"  H_T orders (full / proper)  : "
                f"{report.full_orientation_intersection_order} / "
                f"{report.proper_orientation_intersection_order}"
            ),
            f"  distinct OR variants        : {report.orientation_variant_count}",
            f"  orientation operators       : {report.orientation_operator_count}",
            f"  max representation parity   : {report.parity.maximum_residual:.3e}",
        ]

        topology_audit = report.cayron_topology_audit
        lines.extend(
            [
                "",
                "CAYRON TOPOLOGY AUDIT",
                (
                    f"  N_T full / proper          : "
                    f"{topology_audit.full_orientation_variant_count} / "
                    f"{topology_audit.proper_orientation_variant_count}"
                ),
                (
                    f"  O_T full / proper          : "
                    f"{topology_audit.full_orientation_operator_count} / "
                    f"{topology_audit.proper_orientation_operator_count}"
                ),
                (
                    "  H_C / N_C / O_C             : "
                    + (
                        "n/a"
                        if topology_audit.correspondence_intersection_order is None
                        else (
                            f"{topology_audit.correspondence_intersection_order} / "
                            f"{topology_audit.correspondence_variant_count} / "
                            f"{topology_audit.correspondence_operator_count}"
                        )
                    )
                ),
                (
                    "  H_T == H_C                  : "
                    + (
                        "n/a"
                        if topology_audit.orientation_correspondence_intersections_equal
                        is None
                        else (
                            "YES"
                            if topology_audit.orientation_correspondence_intersections_equal
                            else "NO"
                        )
                    )
                ),
                (
                    "  one-to-one C/T topology     : "
                    + (
                        "n/a"
                        if topology_audit.one_to_one_correspondence_orientation_topology
                        is None
                        else (
                            "YES"
                            if topology_audit.one_to_one_correspondence_orientation_topology
                            else "NO"
                        )
                    )
                ),
                (
                    f"  H_T numerical residual      : "
                    f"{topology_audit.maximum_orientation_intersection_residual:.3e}"
                ),
            ]
        )

        ptclab = next(
            (
                item
                for item in report.representations
                if item.reference_convention
                == CartesianConvention.PTCLAB_A_X_C_XZ.value
                and item.moving_convention == CartesianConvention.PTCLAB_A_X_C_XZ.value
            ),
            None,
        )
        if ptclab is not None:
            lines.extend(
                [
                    "",
                    "PTCLAB-COMPATIBLE CARTESIAN MATRIX  (x||a, c in xz)",
                    *self._matrix(np.asarray(ptclab.matrix_reference_from_moving)),
                ]
            )

        if show_representations:
            lines.extend(["", "ALL 3 x 3 CARTESIAN REPRESENTATION PAIRS"])
            for item in report.representations:
                lines.append(
                    f"  {item.reference_convention} <- {item.moving_convention}"
                )
                lines.extend(
                    self._matrix(np.asarray(item.matrix_reference_from_moving))
                )

        if variants:
            lines.extend(
                [
                    "",
                    "ORIENTATION VARIANTS  (proper left cosets G_A^+ / H_T^+)",
                    (
                        "  idx   g_ref   parent-coset   raw rep angle   "
                        "crystal disorientation"
                    ),
                ]
            )
            for variant in variants:
                lines.append(
                    f"  {variant.index:>3d}   "
                    f"{variant.reference_symmetry_index:>5d}   "
                    f"{variant.reference_coset_symmetry_indices!s:>13s}   "
                    f"{variant.raw_representative_angle_from_base_deg:>13.7g}   "
                    f"{variant.misorientation_from_base_deg:>20.7g}"
                )
            lines.append(
                "  NOTE: each row is one crystallographic orientation class; "
                "daughter-symmetry-equivalent matrices are not double-counted."
            )

        if operators:
            lines.extend(
                [
                    "",
                    "ORIENTATION OPERATORS  (exact full double cosets H_T g H_T)",
                    (
                        "  idx  inv  size   class        Type-I? Type-II?   "
                        "repr. disorientation (deg)"
                    ),
                ]
            )
            for operator in operators:
                lines.append(
                    f"  {operator.index:>3d}  "
                    f"{operator.inverse_operator_index:>3d}  "
                    f"{operator.size:>4d}   "
                    f"{operator.cayron_class:<11s} "
                    f"{operator.contains_parent_reflection!s:<7s} "
                    f"{operator.contains_parent_180_rotation!s:<8s} "
                    f"{operator.minimum_crystallographic_disorientation_deg:>17.9g}"
                )
            lines.append(
                "  NOTE: operator identity is the exact double coset; "
                "disorientation is only a representative summary."
            )

        if report.warnings:
            lines.extend(["", "SCIENTIFIC NOTES"])
            lines.extend(f"  - {warning}" for warning in report.warnings)
        return "\n".join(lines)

    def parallelisms(self, report: ParallelismSolveReport) -> str:
        lines = [
            "=" * 88,
            "OR FROM CRYSTALLOGRAPHIC PARALLELISMS",
            "=" * 88,
            (
                f"phase pair        : {report.reference_phase_id} <- "
                f"{report.moving_phase_id}"
            ),
            f"pair 1            : {report.reference_first} || {report.moving_first}",
            f"pair 2            : {report.reference_second} || {report.moving_second}",
            (
                f"direction signs   : "
                f"{'axial / unoriented' if report.unoriented_directions else 'oriented'}"
            ),
            (
                f"internal angles   : reference="
                f"{report.reference_internal_angle_deg:.12g} deg, moving="
                f"{report.moving_internal_angle_deg:.12g} deg"
            ),
            f"acute mismatch    : {report.internal_angle_mismatch_deg:.3e} deg",
            f"exact candidates  : {len(report.candidates)}",
            "",
            "Sign ambiguity is exposed rather than silently choosing one OR.",
        ]
        for index, candidate in enumerate(report.candidates, start=1):
            lines.extend(
                [
                    "",
                    f"CANDIDATE {index}: {candidate.state.orientation_id}",
                    (
                        f"  residual pair 1 : "
                        f"{candidate.first_parallelism_residual_deg:.3e} deg"
                    ),
                    (
                        f"  residual pair 2 : "
                        f"{candidate.second_parallelism_residual_deg:.3e} deg"
                    ),
                    *self._matrix(_state_matrix(candidate.state)),
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def comparison(report: OrientationComparisonReport) -> str:
        return "\n".join(
            [
                "=" * 88,
                "ORIENTATION COMPARISON",
                "=" * 88,
                (
                    f"first / second    : {report.first_orientation_id} / "
                    f"{report.second_orientation_id}"
                ),
                f"raw misorientation: {report.raw_misorientation_deg:.12g} deg",
                (
                    f"symmetry-reduced  : "
                    f"{report.symmetry_reduced_disorientation_deg:.12g} deg"
                ),
                (
                    f"best variants     : "
                    f"{report.best_first_variant} / {report.best_second_variant}"
                ),
                f"best delta axis    : {report.best_delta_axis_reference_cartesian}",
                "",
                report.note,
            ]
        )


def _add_pair_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reference", required=True)
    parser.add_argument("--moving", required=True)
    parser.add_argument("--id", default="")
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--reference-convention",
        choices=tuple(item.value for item in CartesianConvention),
        default=CartesianConvention.PTCLAB_A_X_C_XZ.value,
    )
    parser.add_argument(
        "--moving-convention",
        choices=tuple(item.value for item in CartesianConvention),
        default=CartesianConvention.PTCLAB_A_X_C_XZ.value,
    )
    parser.add_argument("--representations", action="store_true")
    parser.add_argument("--variants", action="store_true")
    parser.add_argument("--operators", action="store_true")
    parser.add_argument("--json", action="store_true")


def _add_display_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--representations", action="store_true")
    parser.add_argument("--variants", action="store_true")
    parser.add_argument("--operators", action="store_true")
    parser.add_argument("--json", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CuAlNi-CT OR workbench. Physical orientation R and correspondence C "
            "remain distinct."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    polar = subparsers.add_parser("polar")
    polar.add_argument("--transformation", required=True)
    polar.add_argument("--id", default="")
    _add_display_arguments(polar)

    matrix = subparsers.add_parser("matrix")
    _add_pair_arguments(matrix)
    matrix.add_argument("--matrix", required=True)
    matrix.add_argument("--repair", action="store_true")
    matrix.add_argument("--max-repair", type=float, default=1.0e-3)

    axis = subparsers.add_parser("axis-angle")
    _add_pair_arguments(axis)
    axis.add_argument("--axis", required=True)
    axis.add_argument("--angle", required=True, type=float)

    euler = subparsers.add_parser("euler")
    _add_pair_arguments(euler)
    euler.add_argument("--phi1", required=True, type=float)
    euler.add_argument("--Phi", required=True, type=float)
    euler.add_argument("--phi2", required=True, type=float)
    euler.add_argument(
        "--euler-convention",
        choices=tuple(item.value for item in EulerConvention),
        default=EulerConvention.ZXZ_ACTIVE.value,
    )

    quaternion = subparsers.add_parser("quaternion")
    _add_pair_arguments(quaternion)
    quaternion.add_argument("--q", required=True)

    parallel = subparsers.add_parser("parallel")
    _add_pair_arguments(parallel)
    parallel.add_argument("--ref1", required=True)
    parallel.add_argument("--mov1", required=True)
    parallel.add_argument("--ref2", required=True)
    parallel.add_argument("--mov2", required=True)
    parallel.add_argument("--oriented-directions", action="store_true")

    compare_polar = subparsers.add_parser("compare-polar")
    compare_polar.add_argument("--transformation", required=True)
    compare_polar.add_argument("--matrix", required=True)
    compare_polar.add_argument("--repair", action="store_true")
    compare_polar.add_argument("--json", action="store_true")
    return parser


def _render_state(
    service: OrientationService,
    state: OrientationState,
    *,
    representations: bool,
    variants_requested: bool,
    operators_requested: bool,
    json_output: bool,
) -> str:
    report = service.report(state)
    variants = service.variants(state) if variants_requested else ()
    operators = service.operators(state) if operators_requested else ()
    if json_output:
        payload = report.to_dict()
        if variants_requested:
            payload["variants"] = [variant.to_dict() for variant in variants]
        if operators_requested:
            payload["operators"] = [operator.to_dict() for operator in operators]
        return json.dumps(payload, indent=2)
    return OrientationRenderer().orientation(
        report,
        show_representations=representations,
        variants=variants,
        operators=operators,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    service = OrientationService(james_hane_6m_reference_project())

    if args.command == "polar":
        state = service.polar_orientation(args.transformation, orientation_id=args.id)
        print(
            _render_state(
                service,
                state,
                representations=args.representations,
                variants_requested=args.variants,
                operators_requested=args.operators,
                json_output=args.json,
            )
        )
        return 0

    if args.command == "matrix":
        state = service.state_from_matrix(
            args.reference,
            args.moving,
            _parse_matrix(args.matrix),
            orientation_id=args.id,
            label=args.label,
            reference_convention=CartesianConvention(args.reference_convention),
            moving_convention=CartesianConvention(args.moving_convention),
            repair=args.repair,
            maximum_repair_residual=args.max_repair,
        )
        print(
            _render_state(
                service,
                state,
                representations=args.representations,
                variants_requested=args.variants,
                operators_requested=args.operators,
                json_output=args.json,
            )
        )
        return 0

    if args.command == "axis-angle":
        common = {
            "orientation_id": args.id,
            "label": args.label,
            "reference_convention": CartesianConvention(args.reference_convention),
            "moving_convention": CartesianConvention(args.moving_convention),
        }
        if args.axis.strip().startswith("["):
            state = service.state_from_reference_crystal_axis(
                args.reference,
                args.moving,
                args.axis,
                args.angle,
                **common,
            )
        else:
            state = service.state_from_axis_angle(
                args.reference,
                args.moving,
                _parse_numeric_vector(args.axis, 3),
                args.angle,
                **common,
            )
        print(
            _render_state(
                service,
                state,
                representations=args.representations,
                variants_requested=args.variants,
                operators_requested=args.operators,
                json_output=args.json,
            )
        )
        return 0

    if args.command == "euler":
        state = service.state_from_euler(
            args.reference,
            args.moving,
            args.phi1,
            args.Phi,
            args.phi2,
            convention=EulerConvention(args.euler_convention),
            orientation_id=args.id,
            label=args.label,
            reference_convention=CartesianConvention(args.reference_convention),
            moving_convention=CartesianConvention(args.moving_convention),
        )
        print(
            _render_state(
                service,
                state,
                representations=args.representations,
                variants_requested=args.variants,
                operators_requested=args.operators,
                json_output=args.json,
            )
        )
        return 0

    if args.command == "quaternion":
        state = service.state_from_quaternion(
            args.reference,
            args.moving,
            _parse_numeric_vector(args.q, 4),
            orientation_id=args.id,
            label=args.label,
            reference_convention=CartesianConvention(args.reference_convention),
            moving_convention=CartesianConvention(args.moving_convention),
        )
        print(
            _render_state(
                service,
                state,
                representations=args.representations,
                variants_requested=args.variants,
                operators_requested=args.operators,
                json_output=args.json,
            )
        )
        return 0

    if args.command == "parallel":
        report = service.state_from_parallelisms(
            args.reference,
            args.moving,
            args.ref1,
            args.mov1,
            args.ref2,
            args.mov2,
            orientation_id=args.id,
            label=args.label,
            unoriented_directions=not args.oriented_directions,
            reference_convention=CartesianConvention(args.reference_convention),
            moving_convention=CartesianConvention(args.moving_convention),
        )
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            renderer = OrientationRenderer()
            print(renderer.parallelisms(report))
            if args.representations or args.variants:
                for candidate in report.candidates:
                    print()
                    print(
                        _render_state(
                            service,
                            candidate.state,
                            representations=args.representations,
                            variants_requested=args.variants,
                            operators_requested=args.operators,
                            json_output=False,
                        )
                    )
        return 0

    if args.command == "compare-polar":
        polar = service.polar_orientation(args.transformation)
        transformation = service.project.transformation(args.transformation)
        user = service.state_from_matrix(
            transformation.parent_phase_id,
            transformation.product_phase_id,
            _parse_matrix(args.matrix),
            orientation_id="comparison_input",
            reference_convention=transformation.parent_cartesian_convention,
            moving_convention=transformation.product_cartesian_convention,
            repair=args.repair,
        )
        comparison = service.compare_orientations(user, polar)
        print(
            json.dumps(comparison.to_dict(), indent=2)
            if args.json
            else OrientationRenderer.comparison(comparison)
        )
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
