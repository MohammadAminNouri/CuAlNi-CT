from __future__ import annotations

"""Metric-native orientation kernel for arbitrary 3-D crystallographic phase pairs.

This module is a low-level mathematical kernel.  It is deliberately independent
of project/UI state and keeps the following objects distinct:

    C  crystallographic correspondence (crystal coordinates)
    F  correspondence-induced Cartesian deformation bridge
    U  right stretch from F = R_M<-A U
    R  proper Cartesian orientation / polar rotation
    H_T orientation intersection subgroup
    H_C correspondence intersection subgroup

The kernel accepts any finite, physically admissible 3-D crystallographic
input:

* symmetric-positive-definite parent/product metrics;
* exact finite parent/product point groups in the supplied crystal bases;
* an optional exact invertible correspondence;
* an independently supplied proper orientation relationship.

Invalid, internally inconsistent, handedness-ambiguous, or underdetermined
inputs fail explicitly rather than being silently repaired.

Orientation convention
----------------------
For a reference phase A and moving phase M,

    x_A = R_A_from_M @ x_M.

The canonical Cartesian embedding used here is the symmetric metric embedding

    B_A = M_A^(1/2),  B_M = M_M^(1/2),

so x = B u for direct crystal coordinates u.  Plane covectors p map to
Cartesian normals by B^(-T) p.

An exact crystal-basis change B_crystal' = B_crystal P obeys

    M' = P^T M P
    g' = P^(-1) g P
    C' = P_M^(-1) C P_A.

Because the canonical symmetric embedding changes its Cartesian gauge after a
crystal-basis change, the physical gauge rotations are tracked explicitly.
"""

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import sympy as sp

from .correspondence import Correspondence
from .group_theory import (
    burnside_double_coset_count,
    correspondence_subgroup,
    double_cosets,
    left_cosets,
    operator_adjacency,
    representatives,
    validate_group,
)
from .lattice import metric_sqrt
from .stretch import (
    metric_native_stretch_spectrum,
    positive_definite_sqrt,
    stretch_from_metrics,
)
from .symmetry import matrix_key
from .twinning_ct import CTTwin

CrystalVectorKind = Literal["direction", "plane"]


def _relative_residual(lhs: np.ndarray, rhs: np.ndarray) -> float:
    left = np.asarray(lhs, dtype=float)
    right = np.asarray(rhs, dtype=float)
    scale = max(
        float(np.linalg.norm(left, ord="fro") if left.ndim == 2 else np.linalg.norm(left)),
        float(np.linalg.norm(right, ord="fro") if right.ndim == 2 else np.linalg.norm(right)),
        1.0,
    )
    return float(np.linalg.norm(left - right) / scale)


def _as_spd(metric: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(metric, dtype=float)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    symmetric = 0.5 * (value + value.T)
    if _relative_residual(value, symmetric) > 1.0e-12:
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if float(np.min(eigenvalues)) <= 0.0:
        raise ValueError(
            f"{name} must be positive definite; eigenvalues={eigenvalues}"
        )
    return symmetric


def _as_exact_group(
    group: Sequence[sp.Matrix] | Iterable[sp.Matrix],
    *,
    name: str,
) -> tuple[sp.Matrix, ...]:
    result = tuple(sp.Matrix(item) for item in group)
    if not result:
        raise ValueError(f"{name} must not be empty")
    try:
        validate_group(list(result))
    except Exception as exc:
        raise ValueError(f"{name} is not an exact finite matrix group: {exc}") from exc
    return result


def _unit(vector: np.ndarray, *, name: str = "vector") -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-15:
        raise ValueError(f"{name} must be nonzero")
    return value / norm


@dataclass(frozen=True)
class SO3Audit:
    orthogonality_residual: float
    determinant: float
    determinant_residual: float

    @property
    def maximum_residual(self) -> float:
        return max(self.orthogonality_residual, self.determinant_residual)


def so3_audit(matrix: np.ndarray) -> SO3Audit:
    value = np.asarray(matrix, dtype=float)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    orthogonality = float(
        np.linalg.norm(value.T @ value - np.eye(3), ord="fro")
    )
    determinant = float(np.linalg.det(value))
    return SO3Audit(
        orthogonality_residual=orthogonality,
        determinant=determinant,
        determinant_residual=abs(determinant - 1.0),
    )


def require_so3(
    matrix: np.ndarray,
    *,
    tolerance: float = 1.0e-9,
    name: str = "orientation",
) -> np.ndarray:
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    value = np.asarray(matrix, dtype=float).reshape(3, 3)
    audit = so3_audit(value)
    if audit.maximum_residual > tolerance:
        raise ValueError(
            f"{name} must be a proper rotation in SO(3); "
            f"orthogonality={audit.orthogonality_residual:.3e}, "
            f"det={audit.determinant:.16g}"
        )
    return value


def rotation_angle_deg(matrix: np.ndarray, *, tolerance: float = 1.0e-8) -> float:
    """Stable principal SO(3) angle in [0, 180] degrees.

    ``atan2(sin(theta), cos(theta))`` is used instead of arccos(trace) alone,
    retaining sensitivity near zero while remaining stable near 180 degrees.
    """

    R = require_so3(matrix, tolerance=tolerance, name="relative rotation")
    skew = 0.5 * np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=float,
    )
    sine = float(np.linalg.norm(skew))
    cosine = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arctan2(sine, cosine)))


@dataclass(frozen=True)
class RotationAxisAngle:
    axis: tuple[float, float, float]
    angle_deg: float


def rotation_axis_angle(
    matrix: np.ndarray,
    *,
    tolerance: float = 1.0e-8,
) -> RotationAxisAngle:
    R = require_so3(matrix, tolerance=tolerance, name="rotation")
    angle = rotation_angle_deg(R, tolerance=tolerance)

    if angle <= 1.0e-10:
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
    elif abs(180.0 - angle) <= 1.0e-6:
        symmetric = 0.5 * (R + np.eye(3))
        values, vectors = np.linalg.eigh(symmetric)
        axis = _unit(vectors[:, int(np.argmax(values))], name="180-degree axis")
    else:
        axis = _unit(
            np.array(
                [
                    R[2, 1] - R[1, 2],
                    R[0, 2] - R[2, 0],
                    R[1, 0] - R[0, 1],
                ],
                dtype=float,
            ),
            name="rotation axis",
        )

    nonzero = np.flatnonzero(np.abs(axis) > 1.0e-12)
    if len(nonzero) and axis[int(nonzero[0])] < 0.0:
        axis = -axis
    axis[np.abs(axis) < 1.0e-14] = 0.0
    return RotationAxisAngle(
        axis=tuple(float(item) for item in axis),
        angle_deg=float(angle),
    )


def projective_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    a = _unit(first, name="first vector")
    b = _unit(second, name="second vector")
    sine = float(np.linalg.norm(np.cross(a, b)))
    cosine = abs(float(a @ b))
    return float(np.degrees(np.arctan2(sine, cosine)))


def oriented_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    a = _unit(first, name="first vector")
    b = _unit(second, name="second vector")
    sine = float(np.linalg.norm(np.cross(a, b)))
    cosine = float(np.clip(a @ b, -1.0, 1.0))
    return float(np.degrees(np.arctan2(sine, cosine)))


def metric_projective_residual(
    first: np.ndarray,
    second: np.ndarray,
    metric: np.ndarray,
    *,
    reciprocal: bool,
) -> float:
    """Stable projective chord residual in a direct/reciprocal metric."""

    M = _as_spd(metric, name="metric")
    G = np.linalg.solve(M, np.eye(3)) if reciprocal else M
    G = 0.5 * (G + G.T)
    L = np.linalg.cholesky(G)

    def embedded(value: np.ndarray) -> np.ndarray:
        return _unit(
            L.T @ np.asarray(value, dtype=float).reshape(3),
            name="metric-embedded vector",
        )

    a = embedded(first)
    b = embedded(second)
    return float(min(np.linalg.norm(a - b), np.linalg.norm(a + b)))


@dataclass(frozen=True)
class KinematicDecomposition:
    """Strict separation of C, F, U and polar orientation."""

    correspondence_M_from_A: np.ndarray
    deformation_M_from_A: np.ndarray
    stretch_A: np.ndarray
    polar_rotation_M_from_A: np.ndarray
    polar_orientation_A_from_M: np.ndarray
    principal_stretches: np.ndarray
    deformation_reconstruction_residual: float
    stretch_symmetry_residual: float
    polar_rotation_residual: float
    stretch_crosscheck_residual: float
    metric_spectrum_residual: float

    @property
    def maximum_residual(self) -> float:
        return max(
            self.deformation_reconstruction_residual,
            self.stretch_symmetry_residual,
            self.polar_rotation_residual,
            self.stretch_crosscheck_residual,
            self.metric_spectrum_residual,
        )


@dataclass(frozen=True)
class IntersectionMatch:
    reference_index: int
    moving_index: int
    residual: float


@dataclass(frozen=True)
class KernelTopology:
    full_intersection: tuple[sp.Matrix, ...]
    proper_intersection: tuple[sp.Matrix, ...]
    full_matches: tuple[IntersectionMatch, ...]
    proper_matches: tuple[IntersectionMatch, ...]
    full_variants: tuple[tuple[sp.Matrix, ...], ...]
    proper_variants: tuple[tuple[sp.Matrix, ...], ...]
    full_operators: tuple[tuple[sp.Matrix, ...], ...]
    proper_operators: tuple[tuple[sp.Matrix, ...], ...]
    full_adjacency: tuple[tuple[int, ...], ...]
    proper_adjacency: tuple[tuple[int, ...], ...]
    full_burnside_count: int
    proper_burnside_count: int
    correspondence_intersection: tuple[sp.Matrix, ...] | None
    orientation_correspondence_intersections_equal: bool | None
    maximum_intersection_residual: float

    @property
    def full_variant_count(self) -> int:
        return len(self.full_variants)

    @property
    def proper_variant_count(self) -> int:
        return len(self.proper_variants)

    @property
    def full_operator_count(self) -> int:
        return len(self.full_operators)

    @property
    def proper_operator_count(self) -> int:
        return len(self.proper_operators)


@dataclass(frozen=True)
class DisorientationResult:
    angle_deg: float
    axis_reference_cartesian: tuple[float, float, float]
    reference_symmetry_index: int
    moving_symmetry_index: int
    delta: np.ndarray


@dataclass(frozen=True)
class ParallelismCandidate:
    index: int
    R_A_from_M: np.ndarray
    first_residual_deg: float
    second_residual_deg: float
    first_sign: int
    second_sign: int


@dataclass(frozen=True)
class ParallelismReport:
    candidates: tuple[ParallelismCandidate, ...]
    reference_internal_angle_deg: float
    moving_internal_angle_deg: float
    projective_first: bool
    projective_second: bool


@dataclass(frozen=True)
class ClosingGapCandidate:
    index: int
    R_A_from_M: np.ndarray
    direction_parallelism_residual_deg: float
    plane_parallelism_residual_deg: float
    raw_deviation_from_natural_deg: float | None
    symmetry_reduced_deviation_from_natural_deg: float | None


@dataclass(frozen=True)
class ClosingGapReport:
    twin_kind: str
    classification: str
    candidates: tuple[ClosingGapCandidate, ...]
    selected_candidate_index: int | None
    selection_tie_indices: tuple[int, ...]
    intercorrespondence_residual: float
    plane_correspondence_residual: float
    direction_correspondence_residual: float

    @property
    def selected(self) -> ClosingGapCandidate | None:
        if self.selected_candidate_index is None:
            return None
        return self.candidates[self.selected_candidate_index]


@dataclass(frozen=True)
class BasisGauge:
    """Gauge relating canonical symmetric Cartesian frames before/after rebasing."""

    P_A: sp.Matrix
    P_M: sp.Matrix
    Q_A: np.ndarray
    Q_M: np.ndarray
    determinant_P_A: int
    determinant_P_M: int
    Q_A_orthogonality_residual: float
    Q_M_orthogonality_residual: float

    @property
    def relative_orientation_parity(self) -> int:
        return self.determinant_P_A * self.determinant_P_M

    def transform_orientation(
        self,
        R_A_from_M: np.ndarray,
        *,
        require_proper: bool = True,
    ) -> np.ndarray:
        R = np.asarray(R_A_from_M, dtype=float).reshape(3, 3)
        transformed = self.Q_A @ R @ self.Q_M.T
        if require_proper:
            if self.relative_orientation_parity != 1:
                raise ValueError(
                    "Parent/product basis changes have opposite handedness. "
                    "The canonical-frame matrix is then parity reversing and "
                    "cannot be represented as an SO(3) orientation without an "
                    "explicit handedness convention."
                )
            require_so3(
                transformed,
                tolerance=2.0e-8,
                name="rebased orientation",
            )
        return transformed

    def transform_deformation(self, F_M_from_A: np.ndarray) -> np.ndarray:
        return self.Q_M @ np.asarray(F_M_from_A, dtype=float) @ self.Q_A.T

    def transform_reference_stretch(self, U_A: np.ndarray) -> np.ndarray:
        return self.Q_A @ np.asarray(U_A, dtype=float) @ self.Q_A.T


class OrientationKernel:
    """Generic metric-native orientation mathematics for one phase pair."""

    def __init__(
        self,
        M_A: np.ndarray,
        M_M: np.ndarray,
        G_A: Sequence[sp.Matrix] | Iterable[sp.Matrix],
        G_M: Sequence[sp.Matrix] | Iterable[sp.Matrix],
        correspondence: Correspondence | None = None,
        *,
        algebraic_tolerance: float = 1.0e-9,
    ) -> None:
        if algebraic_tolerance <= 0.0:
            raise ValueError("algebraic_tolerance must be positive")

        self.M_A = _as_spd(M_A, name="M_A")
        self.M_M = _as_spd(M_M, name="M_M")
        self.G_A = _as_exact_group(G_A, name="G_A")
        self.G_M = _as_exact_group(G_M, name="G_M")
        self.correspondence = correspondence
        self.algebraic_tolerance = float(algebraic_tolerance)

        self.B_A = np.asarray(metric_sqrt(self.M_A), dtype=float)
        self.B_M = np.asarray(metric_sqrt(self.M_M), dtype=float)
        self.B_A_inv = np.linalg.inv(self.B_A)
        self.B_M_inv = np.linalg.inv(self.B_M)

        self._validate_group_metric(self.G_A, self.M_A, name="G_A")
        self._validate_group_metric(self.G_M, self.M_M, name="G_M")

        self._A_cart = self._physical_group(self.G_A, self.B_A, name="G_A")
        self._M_cart = self._physical_group(self.G_M, self.B_M, name="G_M")

        self.G_A_proper = tuple(
            g for g in self.G_A if sp.simplify(g.det()) == 1
        )
        self.G_M_proper = tuple(
            g for g in self.G_M if sp.simplify(g.det()) == 1
        )
        if not self.G_A_proper or not self.G_M_proper:
            raise ValueError("Both phases must contain the identity proper rotation")

        validate_group(list(self.G_A_proper))
        validate_group(list(self.G_M_proper))
        self._A_proper_cart = self._physical_group(
            self.G_A_proper, self.B_A, name="G_A proper"
        )
        self._M_proper_cart = self._physical_group(
            self.G_M_proper, self.B_M, name="G_M proper"
        )

        if correspondence is not None:
            # Construction of Correspondence already enforces exact invertibility.
            _ = correspondence.C_M_from_A

    @staticmethod
    def _validate_group_metric(
        group: tuple[sp.Matrix, ...],
        metric: np.ndarray,
        *,
        name: str,
    ) -> None:
        scale = max(float(np.linalg.norm(metric, ord="fro")), 1.0)
        worst = 0.0
        for index, operation in enumerate(group):
            G = np.asarray(operation, dtype=float)
            residual = float(
                np.linalg.norm(G.T @ metric @ G - metric, ord="fro") / scale
            )
            worst = max(worst, residual)
            if residual > 1.0e-9:
                raise ValueError(
                    f"{name}[{index}] does not preserve the supplied metric; "
                    f"residual={residual:.3e}"
                )
        if not np.isfinite(worst):
            raise ValueError(f"{name} metric-preservation audit is non-finite")

    @staticmethod
    def _physical_group(
        group: tuple[sp.Matrix, ...],
        B: np.ndarray,
        *,
        name: str,
    ) -> tuple[np.ndarray, ...]:
        Binv = np.linalg.inv(B)
        result = []
        for index, operation in enumerate(group):
            Q = B @ np.asarray(operation, dtype=float) @ Binv
            orth = float(np.linalg.norm(Q.T @ Q - np.eye(3), ord="fro"))
            determinant = float(np.linalg.det(Q))
            if max(orth, abs(abs(determinant) - 1.0)) > 2.0e-8:
                raise ValueError(
                    f"{name}[{index}] is not an O(3) isometry in the physical "
                    f"frame; orthogonality={orth:.3e}, det={determinant:.16g}"
                )
            result.append(Q)
        return tuple(result)

    @property
    def reference_proper_cartesian(self) -> tuple[np.ndarray, ...]:
        return self._A_proper_cart

    @property
    def moving_proper_cartesian(self) -> tuple[np.ndarray, ...]:
        return self._M_proper_cart

    def direction_cartesian(
        self,
        coordinates: np.ndarray,
        *,
        phase: Literal["A", "M"],
    ) -> np.ndarray:
        B = self.B_A if phase == "A" else self.B_M
        return _unit(B @ np.asarray(coordinates, dtype=float).reshape(3))

    def plane_normal_cartesian(
        self,
        covector: np.ndarray,
        *,
        phase: Literal["A", "M"],
    ) -> np.ndarray:
        B = self.B_A if phase == "A" else self.B_M
        return _unit(
            np.linalg.solve(B.T, np.asarray(covector, dtype=float).reshape(3))
        )

    def crystal_vector_cartesian(
        self,
        coordinates: np.ndarray,
        *,
        phase: Literal["A", "M"],
        kind: CrystalVectorKind,
    ) -> np.ndarray:
        if kind == "direction":
            return self.direction_cartesian(coordinates, phase=phase)
        if kind == "plane":
            return self.plane_normal_cartesian(coordinates, phase=phase)
        raise ValueError(f"Unknown crystallographic vector kind {kind!r}")

    def kinematics_from_correspondence(self) -> KinematicDecomposition:
        """Build F, U and polar R without identifying any of them with C."""

        if self.correspondence is None:
            raise ValueError("A correspondence is required for C/F/U/R kinematics")

        C = np.asarray(self.correspondence.C_M_from_A, dtype=float)
        F = self.B_M @ C @ self.B_A_inv
        determinant = float(np.linalg.det(F))
        if determinant <= 0.0:
            raise ValueError(
                "The correspondence-induced Cartesian deformation has "
                f"det(F)={determinant:.16g} <= 0. A proper polar orientation "
                "cannot be defined without an explicit handedness correction."
            )

        U = positive_definite_sqrt(F.T @ F)
        R_M_from_A = F @ np.linalg.inv(U)
        audit_R = so3_audit(R_M_from_A)
        if audit_R.maximum_residual > 2.0e-9:
            raise AssertionError(
                "Polar factor failed SO(3) audit: "
                f"{audit_R.maximum_residual:.3e}"
            )

        reconstruction = _relative_residual(F, R_M_from_A @ U)
        symmetry = _relative_residual(U, U.T)
        U_independent = stretch_from_metrics(
            self.M_A, self.M_M, self.correspondence
        )
        stretch_crosscheck = _relative_residual(U, U_independent)

        metric_spectrum = metric_native_stretch_spectrum(
            self.M_A, self.M_M, self.correspondence
        )
        principal = np.sort(np.linalg.eigvalsh(U))
        spectrum_residual = float(
            np.max(np.abs(principal - np.sort(metric_spectrum.lambdas)))
        )

        return KinematicDecomposition(
            correspondence_M_from_A=C,
            deformation_M_from_A=F,
            stretch_A=U,
            polar_rotation_M_from_A=R_M_from_A,
            polar_orientation_A_from_M=R_M_from_A.T,
            principal_stretches=principal,
            deformation_reconstruction_residual=reconstruction,
            stretch_symmetry_residual=symmetry,
            polar_rotation_residual=audit_R.maximum_residual,
            stretch_crosscheck_residual=stretch_crosscheck,
            metric_spectrum_residual=spectrum_residual,
        )

    @staticmethod
    def _intersection_subgroup(
        exact_A: tuple[sp.Matrix, ...],
        cart_A: tuple[np.ndarray, ...],
        exact_M: tuple[sp.Matrix, ...],
        cart_M: tuple[np.ndarray, ...],
        R_A_from_M: np.ndarray,
        *,
        tolerance: float,
    ) -> tuple[tuple[sp.Matrix, ...], tuple[IntersectionMatch, ...], float]:
        found = []
        matches_out = []
        maximum = 0.0

        for i, (gA, QA) in enumerate(zip(exact_A, cart_A, strict=True)):
            matches = []
            for j, QM in enumerate(cart_M):
                residual = _relative_residual(
                    QA,
                    R_A_from_M @ QM @ R_A_from_M.T,
                )
                if residual <= tolerance:
                    matches.append((residual, j))
            if len(matches) > 1:
                matches.sort()
                raise ValueError(
                    "Orientation-intersection membership is numerically "
                    f"ambiguous for parent symmetry {i}: {matches[:3]}. "
                    "Use a tighter tolerance or higher-precision input."
                )
            if matches:
                residual, j = matches[0]
                found.append(gA)
                matches_out.append(IntersectionMatch(i, j, residual))
                maximum = max(maximum, residual)

        if not found:
            raise AssertionError("Identity was not recovered in orientation intersection")
        try:
            validate_group(list(found))
        except Exception as exc:
            raise ValueError(
                "Numerically identified H_T does not close exactly as a subgroup. "
                "The OR is likely too close to a symmetry boundary for the "
                f"configured tolerance ({tolerance:.3e})."
            ) from exc

        return tuple(found), tuple(matches_out), maximum

    def topology(
        self,
        R_A_from_M: np.ndarray,
        *,
        tolerance: float | None = None,
    ) -> KernelTopology:
        """Cayron-style H_T / variants / double-coset topology.

        Intersection membership is discovered in physical Cartesian space,
        while subgroup closure, cosets, double cosets, adjacency and Burnside
        counting are then carried out with the exact crystallographic matrices.
        """

        tol = (
            self.algebraic_tolerance if tolerance is None else float(tolerance)
        )
        if tol <= 0.0:
            raise ValueError("tolerance must be positive")
        R = require_so3(R_A_from_M, tolerance=max(tol, 1.0e-9))

        H_full, matches_full, max_full = self._intersection_subgroup(
            self.G_A,
            self._A_cart,
            self.G_M,
            self._M_cart,
            R,
            tolerance=tol,
        )
        H_proper, matches_proper, max_proper = self._intersection_subgroup(
            self.G_A_proper,
            self._A_proper_cart,
            self.G_M_proper,
            self._M_proper_cart,
            R,
            tolerance=tol,
        )

        variants_full = tuple(
            tuple(item)
            for item in left_cosets(list(self.G_A), list(H_full))
        )
        variants_proper = tuple(
            tuple(item)
            for item in left_cosets(list(self.G_A_proper), list(H_proper))
        )
        operators_full = tuple(
            tuple(item)
            for item in double_cosets(list(self.G_A), list(H_full))
        )
        operators_proper = tuple(
            tuple(item)
            for item in double_cosets(list(self.G_A_proper), list(H_proper))
        )

        adjacency_full = tuple(
            tuple(row)
            for row in operator_adjacency(
                [list(item) for item in variants_full],
                [list(item) for item in operators_full],
            )
        )
        adjacency_proper = tuple(
            tuple(row)
            for row in operator_adjacency(
                [list(item) for item in variants_proper],
                [list(item) for item in operators_proper],
            )
        )

        burnside_full = burnside_double_coset_count(
            list(self.G_A), list(H_full)
        )
        burnside_proper = burnside_double_coset_count(
            list(self.G_A_proper), list(H_proper)
        )
        if burnside_full != len(operators_full):
            raise AssertionError("Full H_T\\G_A/H_T Burnside cross-check failed")
        if burnside_proper != len(operators_proper):
            raise AssertionError("Proper H_T\\G_A/H_T Burnside cross-check failed")

        H_C = None
        equal = None
        if self.correspondence is not None:
            H_C_list = correspondence_subgroup(
                list(self.G_A), list(self.G_M), self.correspondence
            )
            H_C = tuple(H_C_list)
            equal = {
                matrix_key(item) for item in H_C
            } == {
                matrix_key(item) for item in H_full
            }

        return KernelTopology(
            full_intersection=H_full,
            proper_intersection=H_proper,
            full_matches=matches_full,
            proper_matches=matches_proper,
            full_variants=variants_full,
            proper_variants=variants_proper,
            full_operators=operators_full,
            proper_operators=operators_proper,
            full_adjacency=adjacency_full,
            proper_adjacency=adjacency_proper,
            full_burnside_count=burnside_full,
            proper_burnside_count=burnside_proper,
            correspondence_intersection=H_C,
            orientation_correspondence_intersections_equal=equal,
            maximum_intersection_residual=max(max_full, max_proper),
        )

    def disorientation(
        self,
        first_R_A_from_M: np.ndarray,
        second_R_A_from_M: np.ndarray,
    ) -> DisorientationResult:
        """Symmetry-reduced OR-to-OR disorientation.

        For equivalent OR representatives

            R' = S_A R S_M^{-1},

        independent symmetry choices for the two ORs reduce to minimizing

            angle(S_A R1 S_M R2^T)

        over the proper parent and moving point groups.
        """

        R1 = require_so3(first_R_A_from_M, name="first orientation")
        R2 = require_so3(second_R_A_from_M, name="second orientation")

        best = None
        for i, SA in enumerate(self._A_proper_cart):
            for j, SM in enumerate(self._M_proper_cart):
                delta = SA @ R1 @ SM @ R2.T
                angle = rotation_angle_deg(delta, tolerance=2.0e-8)
                key = (angle, i, j)
                if best is None or key < best[0]:
                    best = (key, delta)

        assert best is not None
        (angle, i, j), delta = best
        axis_angle = rotation_axis_angle(delta, tolerance=2.0e-8)
        return DisorientationResult(
            angle_deg=float(angle),
            axis_reference_cartesian=axis_angle.axis,
            reference_symmetry_index=int(i),
            moving_symmetry_index=int(j),
            delta=np.asarray(delta, dtype=float),
        )

    @staticmethod
    def _two_vector_frame(
        first: np.ndarray,
        second: np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        e1 = _unit(first, name=f"{name} first vector")
        second_raw = _unit(second, name=f"{name} second vector")
        projected = second_raw - float(second_raw @ e1) * e1
        if float(np.linalg.norm(projected)) <= 1.0e-12:
            raise ValueError(
                f"{name} parallelisms are collinear/underdetermined; "
                "two independent vectors are required to define a unique OR."
            )
        e2 = _unit(projected, name=f"{name} transverse vector")
        e3 = _unit(np.cross(e1, e2), name=f"{name} frame normal")
        frame = np.column_stack((e1, e2, e3))
        require_so3(frame, tolerance=2.0e-10, name=f"{name} frame")
        return frame

    @staticmethod
    def _vector_residual_deg(
        mapped: np.ndarray,
        target: np.ndarray,
        *,
        projective: bool,
    ) -> float:
        return (
            projective_angle_deg(mapped, target)
            if projective
            else oriented_angle_deg(mapped, target)
        )

    def parallelism_orientations(
        self,
        reference_first: np.ndarray,
        moving_first: np.ndarray,
        reference_second: np.ndarray,
        moving_second: np.ndarray,
        *,
        reference_first_kind: CrystalVectorKind,
        moving_first_kind: CrystalVectorKind,
        reference_second_kind: CrystalVectorKind,
        moving_second_kind: CrystalVectorKind,
        projective_first: bool = True,
        projective_second: bool = True,
        tolerance_deg: float = 1.0e-7,
    ) -> ParallelismReport:
        """Solve two exact crystallographic parallelisms without hidden branch choice."""

        if tolerance_deg <= 0.0:
            raise ValueError("tolerance_deg must be positive")

        a1 = self.crystal_vector_cartesian(
            reference_first, phase="A", kind=reference_first_kind
        )
        m1 = self.crystal_vector_cartesian(
            moving_first, phase="M", kind=moving_first_kind
        )
        a2 = self.crystal_vector_cartesian(
            reference_second, phase="A", kind=reference_second_kind
        )
        m2 = self.crystal_vector_cartesian(
            moving_second, phase="M", kind=moving_second_kind
        )

        ref_internal = oriented_angle_deg(a1, a2)
        mov_internal = oriented_angle_deg(m1, m2)

        source_frame = self._two_vector_frame(m1, m2, name="moving")
        sign1_values = (-1, 1) if projective_first else (1,)
        sign2_values = (-1, 1) if projective_second else (1,)

        candidates = []
        seen = []
        for sign1 in sign1_values:
            for sign2 in sign2_values:
                target_frame = self._two_vector_frame(
                    sign1 * a1,
                    sign2 * a2,
                    name="reference",
                )
                R = target_frame @ source_frame.T
                require_so3(R, tolerance=2.0e-10, name="parallelism OR")
                r1 = self._vector_residual_deg(
                    R @ m1, a1, projective=projective_first
                )
                r2 = self._vector_residual_deg(
                    R @ m2, a2, projective=projective_second
                )
                if max(r1, r2) > tolerance_deg:
                    continue

                if any(
                    rotation_angle_deg(R @ old.T, tolerance=2.0e-8)
                    <= tolerance_deg
                    for old in seen
                ):
                    continue
                seen.append(R)
                candidates.append(
                    ParallelismCandidate(
                        index=len(candidates),
                        R_A_from_M=R,
                        first_residual_deg=r1,
                        second_residual_deg=r2,
                        first_sign=int(sign1),
                        second_sign=int(sign2),
                    )
                )

        if not candidates:
            raise ValueError(
                "No exact proper rotation satisfies the supplied two "
                "parallelisms within tolerance. The internal angles are "
                f"{ref_internal:.12g} deg (reference) and "
                f"{mov_internal:.12g} deg (moving)."
            )

        return ParallelismReport(
            candidates=tuple(candidates),
            reference_internal_angle_deg=ref_internal,
            moving_internal_angle_deg=mov_internal,
            projective_first=bool(projective_first),
            projective_second=bool(projective_second),
        )

    def _validate_twin_provenance(
        self,
        twin: CTTwin,
    ) -> tuple[float, float, float]:
        if self.correspondence is None:
            raise ValueError("A correspondence is required for CT closing-gap ORs")

        parent_keys = {matrix_key(item) for item in self.G_A}
        if matrix_key(twin.parent_symmetry) not in parent_keys:
            raise ValueError(
                "CTTwin parent_symmetry is not a registered parent symmetry"
            )

        C_exact = self.correspondence.C_M_from_A
        expected_Cint = sp.simplify(
            C_exact * twin.parent_symmetry * C_exact.inv()
        )
        inter_res = _relative_residual(
            np.asarray(expected_Cint, dtype=float),
            twin.intercorrespondence,
        )

        expected_plane = np.asarray(
            C_exact.inv().T * sp.Matrix(twin.plane_a),
            dtype=float,
        ).reshape(3)
        expected_direction = np.asarray(
            C_exact * sp.Matrix(twin.direction_a),
            dtype=float,
        ).reshape(3)

        plane_res = metric_projective_residual(
            expected_plane,
            twin.plane_m,
            self.M_M,
            reciprocal=True,
        )
        direction_res = metric_projective_residual(
            expected_direction,
            twin.direction_m,
            self.M_M,
            reciprocal=False,
        )

        limit = max(self.algebraic_tolerance * 100.0, 2.0e-8)
        if max(inter_res, plane_res, direction_res) > limit:
            raise ValueError(
                "CTTwin provenance does not match this orientation kernel: "
                f"Cint={inter_res:.3e}, plane={plane_res:.3e}, "
                f"direction={direction_res:.3e}"
            )
        return inter_res, plane_res, direction_res

    def closing_gap_orientations(
        self,
        twin: CTTwin,
        *,
        natural_orientation: np.ndarray | None = None,
        tolerance_deg: float = 1.0e-7,
        selection_tolerance_deg: float = 1.0e-7,
    ) -> ClosingGapReport:
        """Generate every exact Type-I/II projective closing-gap OR branch."""

        inter_res, plane_res, direction_res = self._validate_twin_provenance(twin)

        parallel = self.parallelism_orientations(
            twin.direction_a,
            twin.direction_m,
            twin.plane_a,
            twin.plane_m,
            reference_first_kind="direction",
            moving_first_kind="direction",
            reference_second_kind="plane",
            moving_second_kind="plane",
            projective_first=True,
            projective_second=True,
            tolerance_deg=tolerance_deg,
        )

        natural = None
        if natural_orientation is not None:
            natural = require_so3(
                natural_orientation,
                tolerance=2.0e-9,
                name="natural orientation",
            )

        candidates = []
        for item in parallel.candidates:
            raw = None
            reduced = None
            if natural is not None:
                raw = rotation_angle_deg(
                    item.R_A_from_M @ natural.T,
                    tolerance=2.0e-8,
                )
                reduced = self.disorientation(
                    item.R_A_from_M, natural
                ).angle_deg

            candidates.append(
                ClosingGapCandidate(
                    index=item.index,
                    R_A_from_M=item.R_A_from_M,
                    direction_parallelism_residual_deg=item.first_residual_deg,
                    plane_parallelism_residual_deg=item.second_residual_deg,
                    raw_deviation_from_natural_deg=raw,
                    symmetry_reduced_deviation_from_natural_deg=reduced,
                )
            )

        selected = None
        ties: tuple[int, ...] = ()
        if natural is not None:
            minimum = min(
                float(item.symmetry_reduced_deviation_from_natural_deg)
                for item in candidates
                if item.symmetry_reduced_deviation_from_natural_deg is not None
            )
            tied = tuple(
                item.index
                for item in candidates
                if item.symmetry_reduced_deviation_from_natural_deg is not None
                and abs(
                    float(item.symmetry_reduced_deviation_from_natural_deg)
                    - minimum
                )
                <= selection_tolerance_deg
            )
            selected = min(
                tied,
                key=lambda index: (
                    float(candidates[index].raw_deviation_from_natural_deg),
                    index,
                ),
            )
            ties = tied

        return ClosingGapReport(
            twin_kind=twin.kind,
            classification=twin.classification,
            candidates=tuple(candidates),
            selected_candidate_index=selected,
            selection_tie_indices=ties,
            intercorrespondence_residual=inter_res,
            plane_correspondence_residual=plane_res,
            direction_correspondence_residual=direction_res,
        )

    @staticmethod
    def _as_unimodular(
        matrix: sp.Matrix,
        *,
        name: str,
    ) -> tuple[sp.Matrix, int]:
        P = sp.Matrix(matrix)
        if P.shape != (3, 3):
            raise ValueError(f"{name} must be 3x3")
        if any(entry.is_integer is not True for entry in P):
            raise ValueError(f"{name} must contain exact integers")
        determinant = sp.simplify(P.det())
        if determinant not in (sp.Integer(1), sp.Integer(-1)):
            raise ValueError(
                f"{name} must be unimodular with determinant +/-1; "
                f"det={determinant}"
            )
        return P, int(determinant)

    def rebase(
        self,
        P_A: sp.Matrix,
        P_M: sp.Matrix,
    ) -> tuple["OrientationKernel", BasisGauge]:
        """Re-express the complete crystallographic state in new exact bases."""

        P_A, det_A = self._as_unimodular(P_A, name="P_A")
        P_M, det_M = self._as_unimodular(P_M, name="P_M")

        Paf = np.asarray(P_A, dtype=float)
        Pmf = np.asarray(P_M, dtype=float)
        M_A_prime = Paf.T @ self.M_A @ Paf
        M_M_prime = Pmf.T @ self.M_M @ Pmf

        P_A_inv = P_A.inv()
        P_M_inv = P_M.inv()
        G_A_prime = tuple(
            sp.simplify(P_A_inv * g * P_A) for g in self.G_A
        )
        G_M_prime = tuple(
            sp.simplify(P_M_inv * g * P_M) for g in self.G_M
        )

        C_prime = None
        if self.correspondence is not None:
            C_prime = Correspondence(
                sp.simplify(
                    P_M_inv * self.correspondence.C_M_from_A * P_A
                ),
                label=f"{self.correspondence.label} [rebased]",
                source=self.correspondence.source,
                derivation=self.correspondence.derivation,
            )

        rebased = OrientationKernel(
            M_A_prime,
            M_M_prime,
            G_A_prime,
            G_M_prime,
            C_prime,
            algebraic_tolerance=self.algebraic_tolerance,
        )

        Q_A = rebased.B_A @ np.linalg.inv(self.B_A @ Paf)
        Q_M = rebased.B_M @ np.linalg.inv(self.B_M @ Pmf)
        QA_orth = float(np.linalg.norm(Q_A.T @ Q_A - np.eye(3), ord="fro"))
        QM_orth = float(np.linalg.norm(Q_M.T @ Q_M - np.eye(3), ord="fro"))
        if max(QA_orth, QM_orth) > 2.0e-8:
            raise AssertionError(
                "Basis-gauge transformation failed orthogonality audit"
            )
        if abs(float(np.linalg.det(Q_A)) - det_A) > 2.0e-8:
            raise AssertionError("Reference gauge parity does not match basis parity")
        if abs(float(np.linalg.det(Q_M)) - det_M) > 2.0e-8:
            raise AssertionError("Moving gauge parity does not match basis parity")

        return rebased, BasisGauge(
            P_A=P_A,
            P_M=P_M,
            Q_A=Q_A,
            Q_M=Q_M,
            determinant_P_A=det_A,
            determinant_P_M=det_M,
            Q_A_orthogonality_residual=QA_orth,
            Q_M_orthogonality_residual=QM_orth,
        )
