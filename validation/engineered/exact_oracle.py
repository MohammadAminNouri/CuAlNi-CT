from __future__ import annotations

"""Independent mathematical oracle.

CRITICAL: this module must never import ``cualni_cryst``.  It implements only
linear algebra/theorem statements used to judge production code.
"""

from dataclasses import dataclass
import numpy as np
import sympy as sp


def rational_rotation_from_quaternion(q: tuple[int, int, int, int]) -> sp.Matrix:
    """Exact SO(3) rotation from an integer quaternion."""
    w, x, y, z = map(sp.Integer, q)
    n = w*w + x*x + y*y + z*z
    if n == 0:
        raise ValueError("zero quaternion")
    return sp.Matrix([
        [w*w+x*x-y*y-z*z, 2*(x*y-w*z),     2*(x*z+w*y)],
        [2*(x*y+w*z),     w*w-x*x+y*y-z*z, 2*(y*z-w*x)],
        [2*(x*z-w*y),     2*(y*z+w*x),     w*w-x*x-y*y+z*z],
    ]) / n


@dataclass(frozen=True)
class ExactRationalCase:
    M_parent: sp.Matrix
    M_product: sp.Matrix
    C: sp.Matrix
    U: sp.Matrix
    F: sp.Matrix
    lambda_values: tuple[sp.Rational, sp.Rational, sp.Rational]


def exact_rational_case(
    lambda_values: tuple[sp.Rational, sp.Rational, sp.Rational],
    *,
    q_u: tuple[int, int, int, int] = (1, 2, 3, 4),
    q_r: tuple[int, int, int, int] = (2, -1, 1, 3),
) -> ExactRationalCase:
    """Construct a fully exact crystallographic encoding from physical F.

    B_M is obtained from the physical mapping F B_A = B_M C.  The oracle
    therefore starts from physical kinematics, not the production CMC formula.
    """
    Qu = rational_rotation_from_quaternion(q_u)
    R = rational_rotation_from_quaternion(q_r)
    D = sp.diag(*lambda_values)
    U = sp.simplify(Qu * D * Qu.T)
    F = sp.simplify(R * U)

    B_a = sp.Matrix([[2, 1, 0], [0, 3, 1], [0, 0, 2]])
    C = sp.Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    B_m = sp.simplify(F * B_a * C.inv())
    M_a = sp.simplify(B_a.T * B_a)
    M_m = sp.simplify(B_m.T * B_m)
    return ExactRationalCase(M_a, M_m, C, U, F, lambda_values)


def exact_generalized_characteristic(case: ExactRationalCase) -> sp.Expr:
    mu = sp.symbols("mu")
    G = sp.simplify(case.C.T * case.M_product * case.C)
    return sp.factor((G - mu * case.M_parent).det())


def expected_generalized_characteristic(case: ExactRationalCase) -> sp.Expr:
    mu = sp.symbols("mu")
    det_ma = sp.factor(case.M_parent.det())
    polynomial = det_ma
    for lam in case.lambda_values:
        polynomial *= (lam**2 - mu)
    return sp.factor(polynomial)


def exact_cmc_determinant(case: ExactRationalCase) -> sp.Expr:
    return sp.factor((case.C.T * case.M_product * case.C - case.M_parent).det())


def exact_singular_values_squared(case: ExactRationalCase) -> tuple[sp.Expr, ...]:
    return tuple(sp.factor(value**2) for value in sorted(case.lambda_values))


def physical_ball_james_habit_normals(U: np.ndarray) -> list[np.ndarray]:
    """Independent Ball-James normals for R U - I = a⊗n.

    Inputs are physical Cartesian matrices.  No production metric,
    correspondence, or CT routine is used.
    """
    U = 0.5 * (np.asarray(U, float) + np.asarray(U, float).T)
    lam, V = np.linalg.eigh(U)
    order = np.argsort(lam)
    lam = lam[order]
    V = V[:, order]
    l1, l2, l3 = map(float, lam)
    if not (l1 < 1.0 < l3) or abs(l2 - 1.0) > 1e-8:
        return []
    e1, e3 = V[:, 0], V[:, 2]
    L1, L3 = l1*l1, l3*l3
    den = np.sqrt(L3 - L1)
    pref = (l3 - l1) / den
    out = []
    for kappa in (-1.0, 1.0):
        n = pref * (
            -np.sqrt(1.0 - L1) * e1
            + kappa * np.sqrt(L3 - 1.0) * e3
        )
        n /= np.linalg.norm(n)
        out.append(n)
    return out


def independent_rank_one_residual(D: np.ndarray) -> float:
    s = np.linalg.svd(np.asarray(D, float), compute_uv=False)
    return float(np.hypot(s[1], s[2]) / max(s[0], np.finfo(float).tiny))


def independent_cofactor_matrix(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, float)
    return np.column_stack((
        np.cross(A[:, 1], A[:, 2]),
        np.cross(A[:, 2], A[:, 0]),
        np.cross(A[:, 0], A[:, 1]),
    ))


def independent_cofactor_values(
    U: np.ndarray,
    a: np.ndarray,
    n: np.ndarray,
) -> tuple[float, float, float]:
    """Return raw CC1, CC2, CC3 theorem expressions independently."""
    U = np.asarray(U, float)
    a = np.asarray(a, float).reshape(3)
    n = np.asarray(n, float).reshape(3)
    lam = np.sort(np.linalg.eigvalsh(0.5 * (U + U.T)))
    cc1 = float(lam[1] - 1.0)
    X = U @ U - np.eye(3)
    cc2 = float(a @ U @ independent_cofactor_matrix(X) @ n)
    U2 = U @ U
    cc3 = float(
        np.trace(U2)
        - np.linalg.det(U2)
        - 0.25 * float(a @ a) * float(n @ n)
        - 2.0
    )
    return cc1, cc2, cc3


def determinant_polynomial_samples(
    U: np.ndarray, a: np.ndarray, n: np.ndarray, xs: np.ndarray
) -> np.ndarray:
    """Independent PTMC determinant samples g(f)=det(Fbar.T Fbar-I)."""
    U = np.asarray(U, float)
    a = np.asarray(a, float)
    n = np.asarray(n, float)
    out = []
    for f in np.asarray(xs, float):
        F = U + float(f) * np.outer(a, n)
        out.append(np.linalg.det(F.T @ F - np.eye(3)))
    return np.asarray(out)
