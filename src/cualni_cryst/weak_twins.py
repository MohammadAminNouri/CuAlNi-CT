from __future__ import annotations

"""Metric-native axial weak-twin crystallography after Cayron (2022).

This module deliberately distinguishes:

* crystallographic correspondence ``C``;
* reticular orientation/isometry ``T``;
* active lattice distortion ``F = T^{-1} C``.

It implements two defensible tasks:

1. exact evaluation of a *specified* axial weak-twin correspondence and weak
   plane pair using Cayron's Eqs. (1)--(7);
2. CT-constrained low-index weak-plane enumeration when the inter-
   correspondence is already known independently (for example from a
   correspondence operator).

It does *not* claim to reverse-engineer every undocumented implementation
choice inside GenOVa's generic A/B/C/D supercell search.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from math import gcd

import numpy as np
import sympy as sp
from sympy.matrices.normalforms import smith_normal_form
from sympy.polys.domains import ZZ

ArrayLike3 = Iterable[float | int]


def _relative_residual(lhs: np.ndarray, rhs: np.ndarray) -> float:
    left = np.asarray(lhs, dtype=float)
    right = np.asarray(rhs, dtype=float)
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1.0)
    return float(np.linalg.norm(left - right) / scale)


def _validated_metric(metric: np.ndarray, *, name: str = "metric") -> np.ndarray:
    matrix = np.asarray(metric, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    symmetric = 0.5 * (matrix + matrix.T)
    if _relative_residual(matrix, symmetric) > 1.0e-12:
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if float(np.min(eigenvalues)) <= 0.0:
        raise ValueError(f"{name} must be positive definite")
    return symmetric


def _metric_basis(metric: np.ndarray) -> np.ndarray:
    """Return B with B.T @ B == M."""

    matrix = _validated_metric(metric)
    return np.linalg.cholesky(matrix).T


def _sympy_rational_matrix(matrix: object, *, tolerance: float = 1.0e-12) -> sp.Matrix:
    """Convert a finite 3x3 matrix to exact rationals without silent guessing.

    Exact SymPy rationals/integers are preserved. Floating values are accepted
    only when a modest-denominator rational reproduces them within ``tolerance``.
    """

    source = sp.Matrix(matrix)
    if source.shape != (3, 3):
        raise ValueError(f"matrix must be 3x3; got {source.shape}")

    exact: list[sp.Rational] = []
    for value in source:
        if value.is_Rational:
            exact.append(sp.Rational(value))
            continue

        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("matrix contains non-finite entries")
        candidate = sp.Rational(str(numeric)).limit_denominator(4096)
        if abs(float(candidate) - numeric) > tolerance:
            raise ValueError(
                "matrix entry is not safely recoverable as a small rational: "
                f"{numeric:.16g}"
            )
        exact.append(candidate)

    return sp.Matrix(3, 3, exact)


def _sympy_rational_vector(
    vector: object,
    *,
    name: str = "crystallographic vector",
    tolerance: float = 1.0e-12,
) -> sp.Matrix:
    """Return a 3x1 exact rational crystallographic vector.

    Exact SymPy integer/rational data are preserved. Floating values are accepted
    only when the same small-rational recovery contract used for correspondence
    matrices succeeds. This keeps direct/reciprocal incidence and projective
    plane relations exact before any Cartesian/metric floating calculation.
    """

    source = sp.Matrix(vector)
    if source.shape == (1, 3):
        source = source.T
    if source.shape != (3, 1):
        raise ValueError(
            f"{name} must contain exactly three entries; got {source.shape}"
        )

    exact: list[sp.Rational] = []
    for value in source:
        if value.is_Rational:
            exact.append(sp.Rational(value))
            continue

        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{name} contains non-finite entries")
        candidate = sp.Rational(str(numeric)).limit_denominator(4096)
        if abs(float(candidate) - numeric) > tolerance:
            raise ValueError(
                f"{name} entry is not safely recoverable as a small rational: "
                f"{numeric:.16g}"
            )
        exact.append(candidate)

    return sp.Matrix(exact)


def _exact_zero_vector(vector: object) -> bool:
    return all(sp.simplify(value) == 0 for value in sp.Matrix(vector))


def _exact_projectively_parallel(first: object, second: object) -> bool:
    """Exact 3-D projective parallelism for rational reciprocal/direct vectors."""

    a = _sympy_rational_vector(first, name="first projective vector")
    b = _sympy_rational_vector(second, name="second projective vector")
    if _exact_zero_vector(a) or _exact_zero_vector(b):
        raise ValueError("projective vectors must be nonzero")
    return _exact_zero_vector(a.cross(b))


def _exact_incident(plane: object, direction: object) -> bool:
    p = _sympy_rational_vector(plane, name="plane covector")
    u = _sympy_rational_vector(direction, name="direct direction")
    return sp.simplify((p.T * u)[0]) == 0


def _primitive_integer_vector(
    vector: sp.Matrix, *, projective: bool
) -> tuple[int, int, int]:
    values = [sp.Rational(item) for item in sp.Matrix(vector)]
    if len(values) != 3:
        raise ValueError("vector must contain exactly three entries")
    if all(value == 0 for value in values):
        raise ValueError("zero crystallographic vector is invalid")

    denominator = 1
    for value in values:
        denominator = int(sp.ilcm(denominator, int(value.q)))
    integers = [int(value * denominator) for value in values]

    divisor = 0
    for value in integers:
        divisor = gcd(divisor, abs(value))
    integers = [value // divisor for value in integers]

    if projective:
        first_nonzero = next(value for value in integers if value != 0)
        if first_nonzero < 0:
            integers = [-value for value in integers]

    return tuple(integers)  # type: ignore[return-value]


def _vector_tuple(vector: np.ndarray) -> tuple[float, float, float]:
    array = np.asarray(vector, dtype=float).reshape(3)
    return tuple(float(value) for value in array)  # type: ignore[return-value]


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    array = np.asarray(matrix, dtype=float).reshape(3, 3)
    return tuple(tuple(float(value) for value in row) for row in array)


@dataclass(frozen=True)
class BravaisNodeBasis:
    """Explicit primitive-node basis inside a chosen conventional cell.

    ``P_conventional_from_primitive`` means

        u_conventional = P @ u_primitive.

    Reciprocal covectors transform dually:

        p_primitive = P.T @ p_conventional.
    """

    P_conventional_from_primitive: sp.Matrix
    label: str = "explicit primitive-node basis"

    def __post_init__(self) -> None:
        matrix = _sympy_rational_matrix(self.P_conventional_from_primitive)
        if matrix.det() == 0:
            raise ValueError("primitive-node basis must be invertible")
        object.__setattr__(self, "P_conventional_from_primitive", matrix)

    @classmethod
    def primitive_conventional(cls) -> BravaisNodeBasis:
        return cls(sp.eye(3), "conventional cell explicitly treated as primitive")

    @classmethod
    def c_centered_unique_b(cls) -> BravaisNodeBasis:
        """Primitive node basis for conventional C-centering (1/2,1/2,0)."""

        return cls(
            sp.Matrix(
                [
                    [sp.Rational(1, 2), -sp.Rational(1, 2), 0],
                    [sp.Rational(1, 2), sp.Rational(1, 2), 0],
                    [0, 0, 1],
                ]
            ),
            "C-centered conventional cell, unique-b compatible node basis",
        )

    @property
    def determinant(self) -> sp.Rational:
        return sp.Rational(self.P_conventional_from_primitive.det())

    def metric_primitive(self, metric_conventional: np.ndarray) -> np.ndarray:
        metric = _validated_metric(metric_conventional)
        P = np.asarray(self.P_conventional_from_primitive, dtype=float)
        return P.T @ metric @ P

    def direct_to_primitive(self, vector_conventional: ArrayLike3) -> sp.Matrix:
        return self.P_conventional_from_primitive.inv() * sp.Matrix(vector_conventional)

    def direct_to_conventional(self, vector_primitive: ArrayLike3) -> sp.Matrix:
        return self.P_conventional_from_primitive * sp.Matrix(vector_primitive)

    def plane_to_primitive(self, plane_conventional: ArrayLike3) -> sp.Matrix:
        return self.P_conventional_from_primitive.T * sp.Matrix(plane_conventional)

    def plane_to_conventional(self, plane_primitive: ArrayLike3) -> sp.Matrix:
        return self.P_conventional_from_primitive.inv().T * sp.Matrix(plane_primitive)

    def operator_to_primitive(self, operator_conventional: object) -> sp.Matrix:
        P = self.P_conventional_from_primitive
        return P.inv() * _sympy_rational_matrix(operator_conventional) * P

    def operator_to_conventional(self, operator_primitive: object) -> sp.Matrix:
        P = self.P_conventional_from_primitive
        return P * _sympy_rational_matrix(operator_primitive) * P.inv()


@dataclass(frozen=True)
class WeakTwinResiduals:
    axis_correspondence: float
    plane_correspondence: float
    metric_isometry: float
    axis_orientation: float
    plane_orientation: float
    factorization: float

    @property
    def maximum(self) -> float:
        return max(
            self.axis_correspondence,
            self.plane_correspondence,
            self.metric_isometry,
            self.axis_orientation,
            self.plane_orientation,
            self.factorization,
        )


@dataclass(frozen=True)
class ReticularOrientationBranch:
    T_2_from_1: tuple[tuple[float, float, float], ...]
    determinant: float
    generalized_shear: float
    distortion_F1: tuple[tuple[float, float, float], ...]
    residuals: WeakTwinResiduals


@dataclass(frozen=True)
class AxialWeakTwinResult:
    axis_primitive: tuple[int, int, int]
    plane1_primitive: tuple[int, int, int]
    plane2_primitive: tuple[int, int, int]
    axis_conventional: tuple[float, float, float]
    plane1_conventional: tuple[float, float, float]
    plane2_conventional: tuple[float, float, float]
    correspondence_primitive_exact: tuple[tuple[str, str, str], ...]
    correspondence_conventional_exact: tuple[tuple[str, str, str], ...]
    generalized_twin_index: int
    generalized_strain: float
    branches: tuple[ReticularOrientationBranch, ...]
    selected_branch_index: int
    note: str

    @property
    def selected(self) -> ReticularOrientationBranch:
        return self.branches[self.selected_branch_index]


def generalized_twin_index(correspondence_primitive: object) -> int:
    """Exact common-sublattice index for a rational 3-D correspondence.

    If ``C = A/d`` with integer ``A`` and common denominator ``d``, the domain
    common sublattice is ``{x in Z^3 | A x == 0 (mod d)}``. Smith normal form
    gives the exact index

        q_g = product_i d / gcd(d, |s_i|),

    where ``s_i`` are the Smith invariants of ``A``.
    """

    C = _sympy_rational_matrix(correspondence_primitive)
    if C.det() == 0:
        raise ValueError("correspondence must be invertible")

    denominator = 1
    for value in C:
        denominator = int(sp.ilcm(denominator, int(sp.Rational(value).q)))
    integer_matrix = (denominator * C).applyfunc(int)
    smith = smith_normal_form(integer_matrix, domain=ZZ)

    index = 1
    for diagonal in (smith[0, 0], smith[1, 1], smith[2, 2]):
        invariant = abs(int(diagonal))
        index *= denominator // gcd(denominator, invariant)
    return int(index)


def generalized_strain(metric: np.ndarray, correspondence: object) -> float:
    """Cayron generalized strain, Eq. (6), in a metric-orthonormal frame."""

    M = _validated_metric(metric)
    C = np.asarray(_sympy_rational_matrix(correspondence), dtype=float)
    B = _metric_basis(M)

    physical = np.linalg.solve(B.T, (B @ C).T).T
    value = float(np.sum(physical * physical) - 3.0)
    scale = max(float(np.sum(physical * physical)), 3.0, 1.0)
    roundoff = 256.0 * np.finfo(float).eps * scale
    if value < -roundoff:
        raise ValueError(f"generalized strain^2 is negative: {value:.16g}")
    return float(np.sqrt(max(0.0, value)))


def generalized_shear(metric: np.ndarray, distortion: np.ndarray) -> float:
    """Cayron generalized shear, Eq. (5), without explicit metric inversion."""

    M = _validated_metric(metric)
    F = np.asarray(distortion, dtype=float).reshape(3, 3)
    B = _metric_basis(M)
    delta = F - np.eye(3)

    physical = np.linalg.solve(B.T, (B @ delta).T).T
    value = float(np.sum(physical * physical))
    if value < 0.0:
        raise AssertionError("squared Frobenius norm became negative")
    return float(np.sqrt(value))


def _cayron_supT_basis(
    metric: np.ndarray,
    axis: np.ndarray,
    plane: np.ndarray,
    *,
    normal_sign: int,
    transverse_sign: int,
) -> np.ndarray:
    if normal_sign not in {-1, 1} or transverse_sign not in {-1, 1}:
        raise ValueError("basis signs must be ±1")

    M = _validated_metric(metric)
    M_inv = np.linalg.inv(M)
    u = np.asarray(axis, dtype=float).reshape(3)
    p = np.asarray(plane, dtype=float).reshape(3)

    u_norm = float(np.sqrt(u @ M @ u))
    p_norm = float(np.sqrt(p @ M_inv @ p))
    if u_norm <= 1.0e-15 or p_norm <= 1.0e-15:
        raise ValueError("axis and plane must be nonzero")

    u_unit = u / u_norm
    p_unit = p / p_norm
    incidence = abs(float(p_unit @ u_unit))
    if incidence > 1.0e-10:
        raise ValueError(
            "axial direction must lie in weak plane; "
            f"normalized incidence residual={incidence:.3e}"
        )

    n = M_inv @ p_unit
    transverse = M_inv @ np.cross(u_unit, n)
    if float(np.sqrt(transverse @ M @ transverse)) <= 1.0e-15:
        raise ValueError("Cayron supT basis collapsed")

    basis = np.column_stack(
        (
            u_unit,
            normal_sign * n,
            transverse_sign * transverse,
        )
    )
    if abs(float(np.linalg.det(basis))) <= 1.0e-14:
        raise ValueError("Cayron supT basis is singular")
    return basis


def _orientation_branches(
    metric: np.ndarray,
    axis: np.ndarray,
    plane1: np.ndarray,
    plane2: np.ndarray,
    correspondence: np.ndarray,
) -> tuple[ReticularOrientationBranch, ...]:
    """Construct Cayron Eq. (3) branches after exact crystallographic audit."""

    M = _validated_metric(metric)
    C = np.asarray(correspondence, dtype=float).reshape(3, 3)

    unique: list[np.ndarray] = []
    branches: list[ReticularOrientationBranch] = []

    for n1_sign in (-1, 1):
        for t1_sign in (-1, 1):
            basis1 = _cayron_supT_basis(
                M,
                axis,
                plane1,
                normal_sign=n1_sign,
                transverse_sign=t1_sign,
            )
            for n2_sign in (-1, 1):
                for t2_sign in (-1, 1):
                    basis2 = _cayron_supT_basis(
                        M,
                        axis,
                        plane2,
                        normal_sign=n2_sign,
                        transverse_sign=t2_sign,
                    )

                    T = np.linalg.solve(basis1.T, basis2.T).T
                    if any(
                        _relative_residual(T, previous) <= 1.0e-12
                        for previous in unique
                    ):
                        continue
                    unique.append(T)

                    F = np.linalg.solve(T, C)
                    p2_pred = np.linalg.solve(
                        T.T,
                        np.asarray(plane1, dtype=float),
                    )

                    residuals = WeakTwinResiduals(
                        axis_correspondence=0.0,
                        plane_correspondence=0.0,
                        metric_isometry=_relative_residual(T.T @ M @ T, M),
                        axis_orientation=_relative_residual(
                            T @ np.asarray(axis, dtype=float),
                            np.asarray(axis, dtype=float),
                        ),
                        plane_orientation=_projective_vector_residual(
                            p2_pred,
                            np.asarray(plane2, dtype=float),
                        ),
                        factorization=_relative_residual(T @ F, C),
                    )
                    branches.append(
                        ReticularOrientationBranch(
                            T_2_from_1=_matrix_tuple(T),
                            determinant=float(np.linalg.det(T)),
                            generalized_shear=generalized_shear(M, F),
                            distortion_F1=_matrix_tuple(F),
                            residuals=residuals,
                        )
                    )

    branches.sort(
        key=lambda branch: (
            branch.generalized_shear,
            branch.residuals.maximum,
            -branch.determinant,
        )
    )
    return tuple(branches)


def _projective_vector_residual(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=float).reshape(3)
    b = np.asarray(second, dtype=float).reshape(3)
    if float(np.linalg.norm(a)) <= 1.0e-15 or float(np.linalg.norm(b)) <= 1.0e-15:
        raise ValueError("projective vectors must be nonzero")
    a /= float(np.linalg.norm(a))
    b /= float(np.linalg.norm(b))
    return min(float(np.linalg.norm(a - b)), float(np.linalg.norm(a + b)))


def evaluate_axial_weak_twin(
    metric_conventional: np.ndarray,
    correspondence_conventional: object,
    axis_conventional: ArrayLike3,
    plane1_conventional: ArrayLike3,
    plane2_conventional: ArrayLike3,
    *,
    node_basis: BravaisNodeBasis,
) -> AxialWeakTwinResult:
    """Evaluate one specified axial weak twin with exact-then-metric audit."""

    M_c = _validated_metric(metric_conventional, name="metric_conventional")
    C_c = _sympy_rational_matrix(correspondence_conventional)
    P = node_basis.P_conventional_from_primitive

    M_p = node_basis.metric_primitive(M_c)
    C_p = sp.simplify(P.inv() * C_c * P)

    # Rationalize crystallographic indices before basis conversion so exact
    # incidence/invariance/reciprocal statements never depend on float noise.
    u_c_exact = _sympy_rational_vector(
        axis_conventional,
        name="conventional invariant axis",
    )
    p1_c_exact = _sympy_rational_vector(
        plane1_conventional,
        name="conventional weak plane p1",
    )
    p2_c_exact = _sympy_rational_vector(
        plane2_conventional,
        name="conventional weak plane p2",
    )
    u_p_exact = sp.simplify(P.inv() * u_c_exact)
    p1_p_exact = sp.simplify(P.T * p1_c_exact)
    p2_p_exact = sp.simplify(P.T * p2_c_exact)

    determinant = sp.simplify(C_p.det())
    if determinant == 0:
        raise ValueError("correspondence must be invertible")
    if sp.Abs(determinant) != 1:
        raise ValueError(
            "Cayron axial weak-twin supercells must have equal volume: "
            f"|det(C)| must equal 1 exactly; got det(C)={determinant}"
        )

    if not _exact_zero_vector(C_p * u_p_exact - u_p_exact):
        raise ValueError(
            "supplied axis is not invariant under the correspondence (exact rational check; C u != u)"
        )
    if not _exact_incident(p1_p_exact, u_p_exact):
        raise ValueError("axial direction is not exactly incident in weak plane p1")
    if not _exact_incident(p2_p_exact, u_p_exact):
        raise ValueError("axial direction is not exactly incident in weak plane p2")

    p2_from_C_exact = sp.simplify(C_p.inv().T * p1_p_exact)
    if not _exact_projectively_parallel(p2_from_C_exact, p2_p_exact):
        raise ValueError(
            "supplied weak planes are not exactly related projectively by C^{-T}"
        )

    u_p = np.asarray(u_p_exact, dtype=float).reshape(3)
    p1_p = np.asarray(p1_p_exact, dtype=float).reshape(3)
    p2_p = np.asarray(p2_p_exact, dtype=float).reshape(3)
    C_p_float = np.asarray(C_p, dtype=float)

    branches = _orientation_branches(M_p, u_p, p1_p, p2_p, C_p_float)
    if not branches:
        raise AssertionError("no reticular orientation branches were generated")

    p1_conv_exact = node_basis.plane_to_conventional(p1_p_exact)
    p2_conv_exact = node_basis.plane_to_conventional(p2_p_exact)
    u_conv_exact = node_basis.direct_to_conventional(u_p_exact)

    C_c_from_p = sp.simplify(P * C_p * P.inv())
    if C_c_from_p != C_c:
        raise AssertionError("primitive/conventional correspondence roundtrip failed")

    result = AxialWeakTwinResult(
        axis_primitive=_primitive_integer_vector(u_p_exact, projective=False),
        plane1_primitive=_primitive_integer_vector(p1_p_exact, projective=True),
        plane2_primitive=_primitive_integer_vector(p2_p_exact, projective=True),
        axis_conventional=_vector_tuple(np.asarray(u_conv_exact, dtype=float)),
        plane1_conventional=_vector_tuple(np.asarray(p1_conv_exact, dtype=float)),
        plane2_conventional=_vector_tuple(np.asarray(p2_conv_exact, dtype=float)),
        correspondence_primitive_exact=tuple(
            tuple(str(C_p[i, j]) for j in range(3)) for i in range(3)
        ),
        correspondence_conventional_exact=tuple(
            tuple(str(C_c[i, j]) for j in range(3)) for i in range(3)
        ),
        generalized_twin_index=generalized_twin_index(C_p),
        generalized_strain=generalized_strain(M_p, C_p),
        branches=branches,
        selected_branch_index=0,
        note=(
            "Cayron-2022 reticular weak-twin evaluation. Exact rational "
            "crystallographic identities are proved before floating metric "
            "calculations. The selected branch minimizes generalized shear "
            "among Eq. (3) sign branches. Improper T is retained if selected."
        ),
    )

    eps2_c = result.generalized_strain**2
    B = _metric_basis(M_p)
    for branch in result.branches:
        F = np.asarray(branch.distortion_F1, dtype=float)
        physical_f = np.linalg.solve(B.T, (B @ F).T).T
        eps2_f = float(np.sum(physical_f * physical_f) - 3.0)
        scale = max(abs(eps2_c), abs(eps2_f), 1.0)
        if abs(eps2_c - eps2_f) / scale > 5.0e-10:
            raise AssertionError(
                "Cayron generalized-strain equivalence Eq. (6)==Eq. (7) failed: "
                f"C-based={eps2_c:.16g}, F-based={eps2_f:.16g}"
            )

    return result


def enumerate_ct_constrained_weak_planes(
    metric_conventional: np.ndarray,
    correspondence_conventional: object,
    axis_conventional: ArrayLike3,
    *,
    node_basis: BravaisNodeBasis,
    max_plane_index: int = 6,
    maximum_generalized_shear: float | None = None,
) -> tuple[AxialWeakTwinResult, ...]:
    """Enumerate low-index axial weak planes for an already-known C_int.

    This is a CT-constrained search, not a claim to reproduce GenOVa's generic
    supercell-discovery order. For each primitive reciprocal plane ``p1`` that
    contains the invariant axis, ``p2 = C^{-T} p1`` is reduced exactly. Results
    are ranked by generalized shear and then by plane-index complexity.
    """

    if max_plane_index < 1:
        raise ValueError("max_plane_index must be >= 1")
    if maximum_generalized_shear is not None and maximum_generalized_shear <= 0.0:
        raise ValueError("maximum_generalized_shear must be positive")

    C_c = _sympy_rational_matrix(correspondence_conventional)
    P = node_basis.P_conventional_from_primitive
    C_p = P.inv() * C_c * P
    u_p = node_basis.direct_to_primitive(axis_conventional)

    if C_p * u_p != u_p:
        raise ValueError(
            "axis must be an exact eigenvector of C_int with eigenvalue +1"
        )

    results: list[AxialWeakTwinResult] = []
    seen_pairs: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()

    for h in range(-max_plane_index, max_plane_index + 1):
        for k in range(-max_plane_index, max_plane_index + 1):
            for l in range(-max_plane_index, max_plane_index + 1):
                if h == k == l == 0:
                    continue
                p1 = sp.Matrix([h, k, l])
                try:
                    p1_key = _primitive_integer_vector(p1, projective=True)
                except ValueError:
                    continue
                if (h, k, l) != p1_key:
                    continue
                if (p1.T * u_p)[0] != 0:
                    continue

                p2_rational = C_p.inv().T * p1
                p2_key = _primitive_integer_vector(p2_rational, projective=True)
                pair_key = (p1_key, p2_key)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                p1_conv = node_basis.plane_to_conventional(p1)
                p2_conv = node_basis.plane_to_conventional(sp.Matrix(p2_key))
                result = evaluate_axial_weak_twin(
                    metric_conventional,
                    C_c,
                    axis_conventional,
                    p1_conv,
                    p2_conv,
                    node_basis=node_basis,
                )
                if (
                    maximum_generalized_shear is not None
                    and result.selected.generalized_shear > maximum_generalized_shear
                ):
                    continue
                results.append(result)

    results.sort(
        key=lambda item: (
            item.selected.generalized_shear,
            sum(abs(value) for value in item.plane1_primitive)
            + sum(abs(value) for value in item.plane2_primitive),
            item.plane1_primitive,
            item.plane2_primitive,
        )
    )
    return tuple(results)


def primitive_integer_direction(vector: object) -> tuple[int, int, int]:
    """Canonical primitive integer representative of a rational direct vector."""

    return _primitive_integer_vector(sp.Matrix(vector), projective=False)


def primitive_integer_plane(vector: object) -> tuple[int, int, int]:
    """Canonical projective primitive integer representative of a plane covector."""

    return _primitive_integer_vector(sp.Matrix(vector), projective=True)
