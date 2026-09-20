from __future__ import annotations

"""Cayron CT transformation-twin calculations.

The implementation is metric-native and does not assume a cubic parent basis.
Parent symmetry matrices are validated as crystallographic isometries of the
actual parent metric.  Type-I and Type-II routes are identified from generic
order-two isometries:

- mirror reflection: det = -1, tr = +1, G^2 = I;
- proper 180-degree rotation: det = +1, tr = -1, G^2 = I.

This keeps the existing public API while removing the earlier dependency on a
cubic signed-permutation classifier.
"""

from dataclasses import dataclass

import numpy as np
import sympy as sp

from .correspondence import Correspondence
from .lattice import (
    metric_norm,
    normalize_direct,
    normalize_plane,
    plane_to_unit_normal,
)


@dataclass(frozen=True)
class CTTwin:
    kind: str
    parent_symmetry: sp.Matrix
    plane_m: np.ndarray
    direction_m: np.ndarray
    shear: float
    plane_a: np.ndarray
    direction_a: np.ndarray
    intercorrespondence: np.ndarray
    rational_element: str
    equation_note: str


def _relative_residual(lhs: np.ndarray, rhs: np.ndarray) -> float:
    left = np.asarray(lhs, dtype=float)
    right = np.asarray(rhs, dtype=float)
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1.0)
    return float(np.linalg.norm(left - right) / scale)


def _as_metric(metric: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(metric, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    symmetric = 0.5 * (matrix + matrix.T)
    if _relative_residual(matrix, symmetric) > 1.0e-12:
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if float(np.min(eigenvalues)) <= 0.0:
        raise ValueError(f"{name} must be positive definite; eigenvalues={eigenvalues}")
    return symmetric


def _as_parent_symmetry(
    operation: sp.Matrix,
    metric_a: np.ndarray,
    *,
    tol: float,
) -> np.ndarray:
    matrix = np.asarray(sp.Matrix(operation), dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Parent symmetry must be a finite 3x3 matrix")

    metric = _as_metric(metric_a, name="M_a")
    preservation = _relative_residual(matrix.T @ metric @ matrix, metric)
    if preservation > tol:
        raise ValueError(
            "Parent operation does not preserve the supplied parent metric: "
            f"residual={preservation:.3e}"
        )
    return matrix


def classify_parent_order_two_isometry(
    operation: sp.Matrix,
    metric_a: np.ndarray,
    *,
    tol: float = 1.0e-10,
) -> str:
    """Classify a generic crystallographic order-two parent isometry."""

    matrix = _as_parent_symmetry(operation, metric_a, tol=tol)
    involution = _relative_residual(matrix @ matrix, np.eye(3))
    if involution > tol:
        return "other"

    determinant = float(np.linalg.det(matrix))
    trace = float(np.trace(matrix))

    if abs(determinant + 1.0) <= 10.0 * tol and abs(trace - 1.0) <= 10.0 * tol:
        return "reflection"
    if abs(determinant - 1.0) <= 10.0 * tol and abs(trace + 1.0) <= 10.0 * tol:
        return "twofold"
    if _relative_residual(matrix, np.eye(3)) <= tol:
        return "identity"
    if _relative_residual(matrix, -np.eye(3)) <= tol:
        return "inversion"
    return "other"


def _eigenvector(
    matrix: np.ndarray,
    eigenvalue: float,
    *,
    name: str,
    tol: float = 1.0e-10,
) -> np.ndarray:
    """Extract a robust real eigenvector via a smallest-singular-vector solve."""

    operator = np.asarray(matrix, dtype=float) - float(eigenvalue) * np.eye(3)
    _, _, vh = np.linalg.svd(operator)
    vector = np.asarray(vh[-1], dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-14:
        raise ValueError(f"Failed to extract {name}")

    vector /= norm
    residual = float(np.linalg.norm(operator @ vector))
    scale = max(float(np.linalg.norm(operator)), 1.0)
    if residual / scale > 100.0 * tol:
        raise ValueError(
            f"Failed to extract {name}; eigenvector residual={residual / scale:.3e}"
        )
    return vector


def _nonnegative_shear_squared(value: float, *, tol: float, label: str) -> float:
    if value < -tol:
        raise ValueError(f"Negative {label} shear^2 {value:.16g}")
    return max(0.0, float(value))


def type_i_from_parent_reflection(
    reflection_a: sp.Matrix,
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
    tol: float = 1.0e-10,
) -> CTTwin:
    """Cayron Type-I twin from a generic parent mirror.

    Implements the metric/correspondence equations restated by Cayron (2026,
    Eqs. 18-21).  The rational K1 plane is inherited by correspondence; shear
    amplitude and eta1 depend on the martensite metric.
    """

    metric_a = _as_metric(M_a, name="M_a")
    metric_m = _as_metric(M_m, name="M_m")
    kind = classify_parent_order_two_isometry(reflection_a, metric_a, tol=tol)
    if kind != "reflection":
        raise ValueError(
            "Parent operation is not a metric-preserving mirror reflection "
            f"(classified as {kind!r})"
        )

    G = np.asarray(reflection_a, dtype=float)
    p_a_raw = _eigenvector(
        G.T,
        -1.0,
        name="parent reflection-plane covector",
        tol=tol,
    )
    p_a = normalize_plane(p_a_raw, metric_a)

    C = np.asarray(correspondence.C_M_from_A, dtype=float)
    C_inv = np.linalg.inv(C)

    p_m = normalize_plane(C_inv.T @ p_a, metric_m)
    n_m = plane_to_unit_normal(p_m, metric_m)
    C_int = C @ G @ C_inv

    shear_squared = float(
        np.trace(C_int.T @ metric_m @ C_int @ np.linalg.inv(metric_m)) - 3.0
    )
    shear = float(
        np.sqrt(
            _nonnegative_shear_squared(
                shear_squared,
                tol=tol,
                label="CT Type-I",
            )
        )
    )

    direction_raw = -(C_int + np.eye(3)) @ n_m
    if metric_norm(direction_raw, metric_m) <= tol:
        raise ValueError("Collapsed CT Type-I shear vector")
    eta1_m = normalize_direct(direction_raw, metric_m)
    eta1_a = normalize_direct(C_inv @ eta1_m, metric_a)

    incidence_m = abs(float(p_m @ eta1_m))
    incidence_a = abs(float(p_a @ eta1_a))
    if max(incidence_m, incidence_a) > 100.0 * tol:
        raise AssertionError(
            "CT Type-I plane/direction incidence failed: "
            f"parent={incidence_a:.3e}, martensite={incidence_m:.3e}"
        )

    return CTTwin(
        kind="I",
        parent_symmetry=sp.Matrix(reflection_a),
        plane_m=p_m,
        direction_m=eta1_m,
        shear=shear,
        plane_a=p_a,
        direction_a=eta1_a,
        intercorrespondence=C_int,
        rational_element="K1 is rational/correspondence-derived",
        equation_note=(
            "Cayron Type-I: parent mirror -> rational K1; shear and eta1 "
            "depend on the martensite metric."
        ),
    )


def type_ii_from_parent_twofold(
    rotation_a: sp.Matrix,
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
    tol: float = 1.0e-10,
) -> CTTwin:
    """Cayron Type-II twin from a generic proper parent twofold.

    Implements the metric/correspondence equations restated by Cayron (2026,
    Eqs. 22-25).  The rational eta2 direction is inherited by correspondence;
    shear amplitude and K2 depend on the martensite metric.
    """

    metric_a = _as_metric(M_a, name="M_a")
    metric_m = _as_metric(M_m, name="M_m")
    kind = classify_parent_order_two_isometry(rotation_a, metric_a, tol=tol)
    if kind != "twofold":
        raise ValueError(
            "Parent operation is not a metric-preserving proper twofold "
            f"(classified as {kind!r})"
        )

    G = np.asarray(rotation_a, dtype=float)
    axis_a_raw = _eigenvector(
        G,
        1.0,
        name="parent twofold axis",
        tol=tol,
    )
    eta2_a = normalize_direct(axis_a_raw, metric_a)

    C = np.asarray(correspondence.C_M_from_A, dtype=float)
    C_inv = np.linalg.inv(C)

    eta2_m = normalize_direct(C @ eta2_a, metric_m)
    C_int = C @ G @ C_inv

    shear_squared = float(
        np.trace(C_int @ np.linalg.inv(metric_m) @ C_int.T @ metric_m) - 3.0
    )
    shear = float(
        np.sqrt(
            _nonnegative_shear_squared(
                shear_squared,
                tol=tol,
                label="CT Type-II",
            )
        )
    )

    p_m = metric_m @ eta2_m
    C_star = np.linalg.inv(C_int).T
    K2_raw = -(C_star - np.eye(3)) @ p_m
    if float(np.linalg.norm(K2_raw)) <= tol:
        raise ValueError("Collapsed CT Type-II K2")

    K2_m = normalize_plane(K2_raw, metric_m)
    K2_a = normalize_plane(C.T @ K2_m, metric_a)

    incidence_m = abs(float(K2_m @ eta2_m))
    incidence_a = abs(float(K2_a @ eta2_a))
    if max(incidence_m, incidence_a) > 100.0 * tol:
        raise AssertionError(
            "CT Type-II plane/direction incidence failed: "
            f"parent={incidence_a:.3e}, martensite={incidence_m:.3e}"
        )

    return CTTwin(
        kind="II",
        parent_symmetry=sp.Matrix(rotation_a),
        plane_m=K2_m,
        direction_m=eta2_m,
        shear=shear,
        plane_a=K2_a,
        direction_a=eta2_a,
        intercorrespondence=C_int,
        rational_element="eta2 is rational/correspondence-derived",
        equation_note=(
            "Cayron Type-II: parent proper twofold -> rational eta2; "
            "shear and K2 depend on the martensite metric."
        ),
    )


def twins_from_operator(
    operator: list[sp.Matrix],
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
) -> list[CTTwin]:
    """Calculate CT Type-I/II solutions generated by one parent double coset.

    The operator may come from any crystallographic parent point group.  The
    full group is appropriate: mirrors feed the Type-I route and proper
    twofolds feed the Type-II route.  Symmetry-equivalent duplicates are
    intentionally retained so exact parent-symmetry provenance is not lost.
    """

    metric_a = _as_metric(M_a, name="M_a")
    out: list[CTTwin] = []

    for operation in operator:
        kind = classify_parent_order_two_isometry(operation, metric_a)
        try:
            if kind == "reflection":
                twin = type_i_from_parent_reflection(
                    operation,
                    M_a,
                    M_m,
                    correspondence,
                )
            elif kind == "twofold":
                twin = type_ii_from_parent_twofold(
                    operation,
                    M_a,
                    M_m,
                    correspondence,
                )
            else:
                continue
        except ValueError as exc:
            # Elements of the correspondence subgroup can map the reference
            # variant onto itself, producing zero-shear/collapsed relations.
            # They are not M/M twin boundaries.
            if "Collapsed CT" in str(exc):
                continue
            raise

        if twin.shear > 1.0e-12:
            out.append(twin)

    return out
