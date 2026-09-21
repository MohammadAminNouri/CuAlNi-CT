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

from dataclasses import dataclass, replace

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
    # ``kind`` is deliberately the construction route, not the final physical
    # classification. A compound twin admits both Type-I and Type-II
    # descriptions, so overwriting kind would destroy exact provenance.
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
    classification: str = ""
    representations: tuple[str, ...] = ()
    classification_note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"I", "II"}:
            raise ValueError(f"Unknown CT twin construction route {self.kind!r}")

        classification = self.classification or f"type_{self.kind}"
        if classification not in {"type_I", "type_II", "compound"}:
            raise ValueError(f"Unknown CT twin classification {classification!r}")

        representations = self.representations or (self.kind,)
        if any(item not in {"I", "II"} for item in representations):
            raise ValueError(f"Invalid CT twin representations {representations!r}")
        if classification == "compound":
            representations = ("I", "II")

        object.__setattr__(self, "classification", classification)
        object.__setattr__(self, "representations", tuple(representations))

    @property
    def compound(self) -> bool:
        return self.classification == "compound"


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
    """Classify an order-two crystallographic isometry."""

    _as_parent_symmetry(operation, metric_a, tol=tol)
    exact = sp.Matrix(operation)
    if exact.shape != (3, 3):
        raise ValueError("Parent symmetry must be 3x3")

    if sp.simplify(exact * exact - sp.eye(3)) != sp.zeros(3):
        return "other"

    determinant = sp.simplify(exact.det())
    trace = sp.simplify(sp.trace(exact))

    if determinant == -1 and trace == 1:
        return "reflection"
    if determinant == 1 and trace == -1:
        return "twofold"
    if sp.simplify(exact - sp.eye(3)) == sp.zeros(3):
        return "identity"
    if sp.simplify(exact + sp.eye(3)) == sp.zeros(3):
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


def _exact_eigenvector(
    operation: sp.Matrix,
    eigenvalue: int,
    *,
    transpose: bool,
    name: str,
) -> sp.Matrix:
    """Return the unique exact crystallographic eigendirection/covector."""

    matrix = sp.Matrix(operation)
    operator = (matrix.T if transpose else matrix) - eigenvalue * sp.eye(3)
    nullspace = operator.nullspace()
    if len(nullspace) != 1:
        raise ValueError(
            f"{name} must be one-dimensional; exact nullity={len(nullspace)}"
        )
    vector = sp.simplify(nullspace[0])
    if vector == sp.zeros(3, 1):
        raise ValueError(f"Failed to extract {name}")
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
    """Cayron Type-I transformation twin, Eqs. (17)--(20)."""

    metric_a = _as_metric(M_a, name="M_a")
    metric_m = _as_metric(M_m, name="M_m")
    kind = classify_parent_order_two_isometry(reflection_a, metric_a, tol=tol)
    if kind != "reflection":
        raise ValueError(
            "Parent operation is not a metric-preserving mirror reflection "
            f"(classified as {kind!r})"
        )

    p_a_exact = _exact_eigenvector(
        reflection_a,
        -1,
        transpose=True,
        name="parent reflection-plane covector",
    )
    p_a = normalize_plane(np.asarray(p_a_exact, dtype=float).reshape(3), metric_a)

    p_m_exact = correspondence.map_plane_A_to_M(p_a_exact)
    p_m = normalize_plane(np.asarray(p_m_exact, dtype=float).reshape(3), metric_m)
    n_m = plane_to_unit_normal(p_m, metric_m)

    C_int_exact = correspondence.intercorrespondence_from_parent_symmetry(reflection_a)
    C_int = np.asarray(C_int_exact, dtype=float)
    C_inv = np.asarray(correspondence.C_A_from_M, dtype=float)

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
            "Cayron Type-I Eqs. (17)-(20): exact parent mirror and rational "
            "K1/intercorrespondence; shear and eta1 depend on martensite metric."
        ),
    )


def type_ii_from_parent_twofold(
    rotation_a: sp.Matrix,
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
    tol: float = 1.0e-10,
) -> CTTwin:
    """Cayron Type-II transformation twin, Eqs. (21)--(24)."""

    metric_a = _as_metric(M_a, name="M_a")
    metric_m = _as_metric(M_m, name="M_m")
    kind = classify_parent_order_two_isometry(rotation_a, metric_a, tol=tol)
    if kind != "twofold":
        raise ValueError(
            "Parent operation is not a metric-preserving proper twofold "
            f"(classified as {kind!r})"
        )

    axis_a_exact = _exact_eigenvector(
        rotation_a,
        1,
        transpose=False,
        name="parent twofold axis",
    )
    eta2_a = normalize_direct(
        np.asarray(axis_a_exact, dtype=float).reshape(3),
        metric_a,
    )

    eta2_m_exact = correspondence.map_direction_A_to_M(axis_a_exact)
    eta2_m = normalize_direct(
        np.asarray(eta2_m_exact, dtype=float).reshape(3),
        metric_m,
    )

    C_int_exact = correspondence.intercorrespondence_from_parent_symmetry(rotation_a)
    C_int = np.asarray(C_int_exact, dtype=float)
    C = np.asarray(correspondence.C_M_from_A, dtype=float)

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
    C_star = np.asarray(C_int_exact.inv().T, dtype=float)
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
            "Cayron Type-II Eqs. (21)-(24): exact parent twofold and rational "
            "eta2/intercorrespondence; shear and K2 depend on martensite metric."
        ),
    )


def _projective_metric_sine(
    lhs: np.ndarray,
    rhs: np.ndarray,
    metric: np.ndarray,
    *,
    reciprocal: bool,
) -> float:
    """Stable projective separation in a direct/reciprocal metric.

    The former sqrt(1-cos(theta)^2) expression is ill-conditioned near
    theta=0 and can flip an exact compound relation after an exact basis
    change. Here we compare normalized physical representatives using the
    appropriate direct/reciprocal Gram matrix.

    Opposite signs represent the same crystallographic line, so the residual
    is min(||u-v||, ||u+v||).
    """

    M = _as_metric(metric, name="metric")
    G = np.linalg.solve(M, np.eye(3)) if reciprocal else M
    G = 0.5 * (G + G.T)

    L = np.linalg.cholesky(G)

    def embedded_unit(value: np.ndarray) -> np.ndarray:
        crystal = np.asarray(value, dtype=float).reshape(3)
        physical = L.T @ crystal
        norm = float(np.linalg.norm(physical))
        if norm <= 1.0e-14:
            raise ValueError(
                "Cannot compare zero/invalid crystallographic elements"
            )
        return physical / norm

    u = embedded_unit(lhs)
    v = embedded_unit(rhs)
    return float(min(np.linalg.norm(u - v), np.linalg.norm(u + v)))


def _same_physical_twin(
    lhs: CTTwin,
    rhs: CTTwin,
    M_a: np.ndarray,
    M_m: np.ndarray,
    *,
    tol: float,
) -> bool:
    """Whether Type-I/II constructions describe the same physical twin.

    This is CT-native: it compares complete twinning elements in both phase
    coordinate systems plus shear, without using stretch/Mallard/Ball-James.
    """

    if lhs.kind == rhs.kind:
        return False

    metric_a = _as_metric(M_a, name="M_a")
    metric_m = _as_metric(M_m, name="M_m")
    geometry_tol = max(100.0 * tol, 1.0e-8)
    shear_tol = max(100.0 * tol, 1.0e-9)

    shear_scale = max(abs(lhs.shear), abs(rhs.shear), 1.0)
    if abs(lhs.shear - rhs.shear) > shear_tol * shear_scale:
        return False

    residuals = (
        _projective_metric_sine(lhs.plane_a, rhs.plane_a, metric_a, reciprocal=True),
        _projective_metric_sine(lhs.direction_a, rhs.direction_a, metric_a, reciprocal=False),
        _projective_metric_sine(lhs.plane_m, rhs.plane_m, metric_m, reciprocal=True),
        _projective_metric_sine(lhs.direction_m, rhs.direction_m, metric_m, reciprocal=False),
    )
    return max(residuals) <= geometry_tol


def classify_compound_twins(
    twins: list[CTTwin],
    M_a: np.ndarray,
    M_m: np.ndarray,
    *,
    tol: float = 1.0e-10,
) -> list[CTTwin]:
    """Add physical twin classification without altering CT solutions.

    A relation is compound only when the same full twin geometry and shear is
    independently generated by Type-I and Type-II CT constructions in the same
    correspondence operator. ``CTTwin.kind`` is retained as route provenance.
    """

    _as_metric(M_a, name="M_a")
    _as_metric(M_m, name="M_m")

    result: list[CTTwin] = []
    for index, twin in enumerate(twins):
        partner_indices = [
            other_index
            for other_index, other in enumerate(twins)
            if other_index != index
            and _same_physical_twin(twin, other, M_a, M_m, tol=tol)
        ]

        if partner_indices:
            partner_routes = {twins[i].kind for i in partner_indices}
            partner_routes.add(twin.kind)
            if partner_routes == {"I", "II"}:
                result.append(
                    replace(
                        twin,
                        classification="compound",
                        representations=("I", "II"),
                        classification_note=(
                            "CT degeneracy: the same full twin geometry and shear "
                            "is generated by both Type-I and Type-II parent-symmetry "
                            "constructions in this operator."
                        ),
                    )
                )
                continue

        result.append(
            replace(
                twin,
                classification=f"type_{twin.kind}",
                representations=(twin.kind,),
                classification_note=(
                    f"Only the Type-{twin.kind} CT construction generates this "
                    "complete twin geometry in the present operator."
                ),
            )
        )

    return result


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

    return classify_compound_twins(out, M_a, M_m)
