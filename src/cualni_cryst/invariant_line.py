
from __future__ import annotations

"""Direct/reciprocal planar invariant-line model.

The formulation follows the inputs described in the PTCLab v1.18.8 manual:
a rational plane (direct-space model) or a common zone axis (reciprocal-space
model), plus two correlated vector pairs.  PTCLab's source equations are not
published in that manual, so this module does not claim source-code identity.
Instead it solves the explicit planar kinematic condition

    det(R(phi) D - I) = 0,

where ``D`` is the 2-D correspondence deformation built from the two correlated
pairs and ``R(phi)`` is the remaining rotation about the constrained plane
normal/zone axis.  Every real branch is returned and independently checked.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np

from .lattice import metric_sqrt
from .orientation_kernel import require_so3, rotation_angle_deg


class InvariantSpace(str, Enum):
    DIRECT = "direct"
    RECIPROCAL = "reciprocal"


@dataclass(frozen=True)
class InvariantLineSolution:
    branch: int
    space: InvariantSpace
    rotation_about_constraint_deg: float
    R_parent_from_product: np.ndarray
    invariant_vector_parent_cartesian: np.ndarray
    invariant_parent_crystal: np.ndarray
    invariant_product_crystal: np.ndarray
    planar_deformation: np.ndarray
    planar_principal_stretches: tuple[float, float]
    invariant_residual: float
    determinant_residual: float
    rotation_residual: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "space": self.space.value,
            "rotation_about_constraint_deg": self.rotation_about_constraint_deg,
            "R_parent_from_product": self.R_parent_from_product.tolist(),
            "invariant_vector_parent_cartesian": (
                self.invariant_vector_parent_cartesian.tolist()
            ),
            "invariant_parent_crystal": self.invariant_parent_crystal.tolist(),
            "invariant_product_crystal": self.invariant_product_crystal.tolist(),
            "planar_deformation": self.planar_deformation.tolist(),
            "planar_principal_stretches": list(self.planar_principal_stretches),
            "invariant_residual": self.invariant_residual,
            "determinant_residual": self.determinant_residual,
            "rotation_residual": self.rotation_residual,
        }


@dataclass(frozen=True)
class InvariantLineReport:
    space: InvariantSpace
    parent_constraint_cartesian: np.ndarray
    product_constraint_cartesian: np.ndarray
    baseline_rotation: np.ndarray
    correlated_pair_residuals: tuple[float, float]
    planar_correspondence: np.ndarray
    solutions: tuple[InvariantLineSolution, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "space": self.space.value,
            "parent_constraint_cartesian": self.parent_constraint_cartesian.tolist(),
            "product_constraint_cartesian": self.product_constraint_cartesian.tolist(),
            "baseline_rotation": self.baseline_rotation.tolist(),
            "correlated_pair_residuals": list(self.correlated_pair_residuals),
            "planar_correspondence": self.planar_correspondence.tolist(),
            "solutions": [solution.to_dict() for solution in self.solutions],
            "notes": list(self.notes),
        }


def _unit(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    norm = float(np.linalg.norm(result))
    if norm <= 1.0e-15:
        raise ValueError(f"{name} must be nonzero")
    return result / norm


def _projective_coefficients(value: np.ndarray) -> np.ndarray:
    out = np.asarray(value, dtype=float).reshape(3)
    scale = float(np.max(np.abs(out)))
    if scale <= 1.0e-15:
        raise ValueError("Cannot index a zero vector")
    out = out / scale
    nz = np.flatnonzero(np.abs(out) > 1.0e-12)
    if len(nz) and out[int(nz[0])] < 0.0:
        out = -out
    out[np.abs(out) < 1.0e-13] = 0.0
    return out


def _minimal_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Proper minimal-angle rotation carrying source to target."""
    a = _unit(source, name="source constraint")
    b = _unit(target, name="target constraint")
    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(a @ b, -1.0, 1.0))
    if sine <= 1.0e-14:
        if cosine > 0.0:
            return np.eye(3)
        # 180 degree rotation: choose a deterministic axis perpendicular to a.
        trial = np.array([1.0, 0.0, 0.0])
        if abs(float(a @ trial)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(a, trial), name="antiparallel rotation axis")
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    axis = cross / sine
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    angle = np.arctan2(sine, cosine)
    return (
        np.eye(3)
        + np.sin(angle) * K
        + (1.0 - np.cos(angle)) * (K @ K)
    )


def _rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    n = _unit(axis, name="constraint axis")
    K = np.array(
        [
            [0.0, -n[2], n[1]],
            [n[2], 0.0, -n[0]],
            [-n[1], n[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _plane_basis(normal: np.ndarray) -> np.ndarray:
    n = _unit(normal, name="plane normal")
    trial = np.array([1.0, 0.0, 0.0])
    if abs(float(n @ trial)) > 0.85:
        trial = np.array([0.0, 1.0, 0.0])
    e1 = _unit(trial - float(trial @ n) * n, name="plane basis e1")
    e2 = _unit(np.cross(n, e1), name="plane basis e2")
    return np.column_stack((e1, e2))


def _physical_direct(lattice: Any, coeffs: Sequence[float]) -> np.ndarray:
    B = np.asarray(metric_sqrt(lattice.metric()), dtype=float)
    return B @ np.asarray(coeffs, dtype=float).reshape(3)


def _physical_reciprocal(lattice: Any, coeffs: Sequence[float]) -> np.ndarray:
    B = np.asarray(metric_sqrt(lattice.metric()), dtype=float)
    return np.linalg.solve(B.T, np.asarray(coeffs, dtype=float).reshape(3))


def _planar_roots(D: np.ndarray, *, tolerance: float) -> tuple[float, ...]:
    D = np.asarray(D, dtype=float).reshape(2, 2)
    A = float(np.trace(D))
    B = float(D[0, 1] - D[1, 0])
    C = float(np.linalg.det(D) + 1.0)
    radius = float(np.hypot(A, B))
    if radius <= 1.0e-15:
        return tuple()
    ratio = C / radius
    if ratio > 1.0 + tolerance or ratio < -1.0 - tolerance:
        return tuple()
    ratio = float(np.clip(ratio, -1.0, 1.0))
    phase = float(np.arctan2(B, A))
    alpha = float(np.arccos(ratio))
    candidates = [phase + alpha, phase - alpha]
    canonical = []
    for angle in candidates:
        wrapped = float((angle + np.pi) % (2.0 * np.pi) - np.pi)
        if not any(abs(wrapped - old) <= 1.0e-10 for old in canonical):
            canonical.append(wrapped)
    return tuple(sorted(canonical))


def _solve(
    parent_lattice: Any,
    product_lattice: Any,
    *,
    space: InvariantSpace,
    parent_constraint: Sequence[float],
    product_constraint: Sequence[float],
    parent_vectors: Sequence[Sequence[float]],
    product_vectors: Sequence[Sequence[float]],
    tolerance: float,
) -> InvariantLineReport:
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if len(parent_vectors) != 2 or len(product_vectors) != 2:
        raise ValueError("Exactly two correlated vector pairs are required")

    if space is InvariantSpace.DIRECT:
        # Constraint is a plane, hence represented physically by its reciprocal normal.
        nA = _unit(
            _physical_reciprocal(parent_lattice, parent_constraint),
            name="parent plane normal",
        )
        nM = _unit(
            _physical_reciprocal(product_lattice, product_constraint),
            name="product plane normal",
        )
        parent_phys = [
            _physical_direct(parent_lattice, item) for item in parent_vectors
        ]
        product_phys = [
            _physical_direct(product_lattice, item) for item in product_vectors
        ]
    else:
        # Reciprocal correlated vectors are restricted to the plane normal to a
        # common direct-space zone axis.
        nA = _unit(
            _physical_direct(parent_lattice, parent_constraint),
            name="parent zone axis",
        )
        nM = _unit(
            _physical_direct(product_lattice, product_constraint),
            name="product zone axis",
        )
        parent_phys = [
            _physical_reciprocal(parent_lattice, item) for item in parent_vectors
        ]
        product_phys = [
            _physical_reciprocal(product_lattice, item) for item in product_vectors
        ]

    # All correlated vectors must belong to the constrained 2-D subspaces.
    incidence = []
    for index, vector in enumerate(parent_phys):
        residual = abs(float(_unit(vector, name=f"parent pair {index}") @ nA))
        if residual > tolerance:
            raise ValueError(
                f"Parent correlated vector {index} is not in the constrained plane; "
                f"normalized incidence={residual:.3e}"
            )
        incidence.append(residual)
    for index, vector in enumerate(product_phys):
        residual = abs(float(_unit(vector, name=f"product pair {index}") @ nM))
        if residual > tolerance:
            raise ValueError(
                f"Product correlated vector {index} is not in the constrained plane; "
                f"normalized incidence={residual:.3e}"
            )
        incidence[index] = max(incidence[index], residual)

    R0 = _minimal_rotation(nM, nA)
    require_so3(R0, tolerance=2.0e-9, name="baseline constraint rotation")
    E = _plane_basis(nA)

    A = np.column_stack([E.T @ vector for vector in parent_phys])
    M = np.column_stack([E.T @ (R0 @ vector) for vector in product_phys])
    if abs(float(np.linalg.det(A))) <= tolerance:
        raise ValueError("Parent correlated vectors are collinear/underdetermined")
    if abs(float(np.linalg.det(M))) <= tolerance:
        raise ValueError("Product correlated vectors are collinear/underdetermined")

    D = M @ np.linalg.inv(A)
    roots = _planar_roots(D, tolerance=max(tolerance, 1.0e-12))
    solutions = []
    B_A = np.asarray(metric_sqrt(parent_lattice.metric()), dtype=float)
    B_M = np.asarray(metric_sqrt(product_lattice.metric()), dtype=float)

    for branch, phi in enumerate(roots):
        c, s = float(np.cos(phi)), float(np.sin(phi))
        R2 = np.array([[c, -s], [s, c]])
        F2 = R2 @ D
        _, _, vh = np.linalg.svd(F2 - np.eye(2))
        l2 = vh[-1]
        l2 = l2 / np.linalg.norm(l2)
        residual = float(np.linalg.norm(F2 @ l2 - l2))
        determinant_residual = abs(float(np.linalg.det(F2 - np.eye(2))))

        Q = _rotation_about_axis(nA, phi)
        R = Q @ R0
        rot_residual = max(
            float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro")),
            abs(float(np.linalg.det(R)) - 1.0),
        )
        line = _unit(E @ l2, name="invariant vector")

        if space is InvariantSpace.DIRECT:
            parent_index = _projective_coefficients(np.linalg.solve(B_A, line))
            product_index = _projective_coefficients(
                np.linalg.solve(B_M, R.T @ line)
            )
        else:
            # The invariant reciprocal vector is a physical plane normal.
            parent_index = _projective_coefficients(B_A.T @ line)
            product_index = _projective_coefficients(B_M.T @ (R.T @ line))

        stretches = tuple(
            float(value) for value in np.sort(np.linalg.svd(F2, compute_uv=False))
        )
        solutions.append(
            InvariantLineSolution(
                branch=branch,
                space=space,
                rotation_about_constraint_deg=float(np.degrees(phi)),
                R_parent_from_product=R,
                invariant_vector_parent_cartesian=line,
                invariant_parent_crystal=parent_index,
                invariant_product_crystal=product_index,
                planar_deformation=F2,
                planar_principal_stretches=stretches,  # type: ignore[arg-type]
                invariant_residual=residual,
                determinant_residual=determinant_residual,
                rotation_residual=rot_residual,
            )
        )

    return InvariantLineReport(
        space=space,
        parent_constraint_cartesian=nA,
        product_constraint_cartesian=nM,
        baseline_rotation=R0,
        correlated_pair_residuals=(float(incidence[0]), float(incidence[1])),
        planar_correspondence=D,
        solutions=tuple(solutions),
        notes=(
            "Two correlated pairs define the in-plane deformation independently of CT/PTMC.",
            "Every real rotation branch satisfying det(R(phi)D-I)=0 is retained.",
            (
                "The PTCLab manual specifies this input/output structure but does not "
                "publish its source equations; this implementation is an explicit "
                "planar kinematic invariant-line formulation, not a claim of source-code identity."
            ),
        ),
    )


def solve_direct_invariant_line(
    parent_lattice: Any,
    product_lattice: Any,
    *,
    parent_plane: Sequence[float],
    product_plane: Sequence[float],
    parent_vectors: Sequence[Sequence[float]],
    product_vectors: Sequence[Sequence[float]],
    tolerance: float = 1.0e-8,
) -> InvariantLineReport:
    return _solve(
        parent_lattice,
        product_lattice,
        space=InvariantSpace.DIRECT,
        parent_constraint=parent_plane,
        product_constraint=product_plane,
        parent_vectors=parent_vectors,
        product_vectors=product_vectors,
        tolerance=tolerance,
    )


def solve_reciprocal_invariant_line(
    parent_lattice: Any,
    product_lattice: Any,
    *,
    parent_zone_axis: Sequence[float],
    product_zone_axis: Sequence[float],
    parent_g_vectors: Sequence[Sequence[float]],
    product_g_vectors: Sequence[Sequence[float]],
    tolerance: float = 1.0e-8,
) -> InvariantLineReport:
    return _solve(
        parent_lattice,
        product_lattice,
        space=InvariantSpace.RECIPROCAL,
        parent_constraint=parent_zone_axis,
        product_constraint=product_zone_axis,
        parent_vectors=parent_g_vectors,
        product_vectors=product_g_vectors,
        tolerance=tolerance,
    )
