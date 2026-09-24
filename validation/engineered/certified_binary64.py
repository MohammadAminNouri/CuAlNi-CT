from __future__ import annotations

"""Independent exact-binary64 oracle for engineered validation.

CRITICAL: this module deliberately does not import ``cualni_cryst``.
Production and oracle must not share the same implementation path.
"""

import mpmath as mp
import numpy as np
import sympy as sp


def _ratio(x: float) -> tuple[int, int]:
    return float(x).as_integer_ratio()


def canonical_metric_binary64(M: np.ndarray) -> np.ndarray:
    """Independent semantic canonicalization of a metric tensor.

    The production API accepts metric tensors, hence symmetry is part of the
    object definition.  Validation therefore exactifies the same canonical
    binary64 symmetric representation rather than treating tiny skew roundoff
    as physical input.
    """
    A = np.asarray(M, float).reshape(3, 3)
    if not np.all(np.isfinite(A)):
        raise ValueError("metric contains non-finite entries")
    return 0.5 * (A + A.T)


def _sp_matrix(A: np.ndarray) -> sp.Matrix:
    return sp.Matrix(
        [[sp.Rational(*_ratio(x)) for x in row] for row in np.asarray(A, float)]
    )


def _mpf(x: float) -> mp.mpf:
    n, d = _ratio(x)
    return mp.mpf(n) / mp.mpf(d)


def _mp_matrix(A: np.ndarray) -> mp.matrix:
    return mp.matrix([[_mpf(x) for x in row] for row in np.asarray(A, float)])


def _sp_metric(M: np.ndarray) -> sp.Matrix:
    return _sp_matrix(canonical_metric_binary64(M))


def _mp_metric(M: np.ndarray) -> mp.matrix:
    return _mp_matrix(canonical_metric_binary64(M))


def exact_canonical_metric_spd(M: np.ndarray) -> bool:
    """Exact Sylvester criterion on canonical IEEE-754 metric entries."""
    A = _sp_metric(M)
    d1 = A[:1, :1].det()
    d2 = A[:2, :2].det()
    d3 = A.det()
    return bool(d1 > 0 and d2 > 0 and d3 > 0)


def exact_binary64_nonsingular(A: np.ndarray) -> bool:
    return bool(_sp_matrix(np.asarray(A, float).reshape(3, 3)).det() != 0)


def cmc_literal_forward_error_bound(
    M_a: np.ndarray,
    M_m: np.ndarray,
    C: np.ndarray,
) -> float:
    r"""Frobenius forward-error bound for literal binary64 CMC assembly.

    For each length-3 dot product we use the standard
    ``gamma_3 = 3u/(1-3u)`` model with unit roundoff ``u=2^-53``.  The bound
    propagates the two matrix products in ``C.T @ M_m @ C`` and the final
    subtraction of ``M_a``.  It is a numerical-analysis bound, not a fitted
    tolerance.
    """
    A = canonical_metric_binary64(M_a)
    M = canonical_metric_binary64(M_m)
    X = np.asarray(C, float).reshape(3, 3)
    u = np.finfo(float).eps / 2.0
    gamma3 = (3.0 * u) / (1.0 - 3.0 * u)

    T = M @ X
    G = X.T @ T
    e1 = gamma3 * (np.abs(M) @ np.abs(X))
    e2 = np.abs(X).T @ e1 + gamma3 * (np.abs(X).T @ np.abs(T))
    esub = u * (np.abs(G) + np.abs(A))
    return float(np.linalg.norm(e2 + esub, ord="fro"))


def certified_mu(
    M_a: np.ndarray,
    M_m: np.ndarray,
    C: np.ndarray,
    *,
    decimal_digits: int = 35,
) -> np.ndarray:
    """Certified real roots of det(C.T M_m C - mu M_a)=0."""
    mu = sp.symbols("mu")
    A = _sp_metric(M_a)
    M = _sp_metric(M_m)
    X = _sp_matrix(C)
    poly = sp.Poly(sp.expand((X.T * M * X - mu * A).det()), mu)
    isolated = sp.polys.polytools.intervals(
        poly, eps=sp.Rational(1, 10 ** int(decimal_digits))
    )
    roots: list[float] = []
    for (lo, hi), multiplicity in isolated:
        roots.extend([float((lo + hi) / 2)] * int(multiplicity))
    roots.sort()
    if len(roots) != 3:
        raise AssertionError(f"expected three real generalized roots, got {isolated}")
    out = np.asarray(roots, float)
    if np.min(out) <= 0.0 or not np.all(np.isfinite(out)):
        raise AssertionError(f"invalid certified SPD spectrum: {out}")
    return out


def high_precision_eigensystem(
    M_a: np.ndarray,
    M_m: np.ndarray,
    C: np.ndarray,
    *,
    digits: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent arbitrary-precision generalized eigensystem."""
    with mp.workdps(int(digits)):
        A = _mp_metric(M_a)
        M = _mp_metric(M_m)
        X = _mp_matrix(C)
        G = X.T * M * X
        L = mp.cholesky(A)
        Li = L ** -1
        H = Li * G * Li.T
        H = (H + H.T) / 2
        values, Q = mp.eigsy(H)
        Vmp = Li.T * Q
        pairs = sorted(
            [
                (
                    mp.mpf(values[i]),
                    [mp.mpf(Vmp[j, i]) for j in range(3)],
                )
                for i in range(3)
            ],
            key=lambda item: item[0],
        )
        mu = np.asarray([float(item[0]) for item in pairs], float)
        V = np.column_stack(
            [np.asarray([float(x) for x in item[1]], float) for item in pairs]
        )
    for j in range(3):
        k = int(np.argmax(np.abs(V[:, j])))
        if V[k, j] < 0:
            V[:, j] *= -1
    return mu, V


def classify_mu(mu: np.ndarray, tol: float) -> tuple[bool, int]:
    q = np.sort(np.asarray(mu, float)) - 1.0
    zero = np.abs(q) <= tol
    nzero = int(np.sum(zero))
    if nzero == 3:
        return True, 3
    if nzero == 2:
        return True, 2
    if nzero == 1:
        other = q[~zero]
        ok = bool(other[0] * other[1] < 0.0)
        return ok, (1 if ok else 0)
    return False, 0


def _plane_norm(p: np.ndarray, M: np.ndarray) -> float:
    p = np.asarray(p, float).reshape(3)
    x = np.linalg.solve(np.asarray(M, float), p)
    q = float(p @ x)
    if q <= 0:
        raise ArithmeticError(f"oracle reciprocal norm is non-positive: {q}")
    return float(np.sqrt(q))


def _normalize_plane(p: np.ndarray, M: np.ndarray) -> np.ndarray:
    return np.asarray(p, float) / _plane_norm(p, M)


def oracle_habit_planes(
    M_a: np.ndarray,
    M_m: np.ndarray,
    C: np.ndarray,
    *,
    tol: float,
) -> list[np.ndarray]:
    """Independent habit-plane factorization of the certified metric pencil."""
    mu_cert = certified_mu(M_a, M_m, C)
    cls, order = classify_mu(mu_cert, tol)
    if not cls or order == 3:
        return []
    _, V = high_precision_eigensystem(M_a, M_m, C)
    eta = mu_cert - 1.0
    zero = np.abs(eta) <= tol
    M_a = np.asarray(M_a, float)
    if order == 2:
        k = int(np.where(~zero)[0][0])
        return [_normalize_plane(M_a @ V[:, k], M_a)]
    iz = int(np.where(zero)[0][0])
    idx = [i for i in range(3) if i != iz]
    i, j = idx
    if eta[i] * eta[j] >= 0:
        return []
    if eta[i] > 0:
        ip, im = i, j
    else:
        ip, im = j, i
    p1 = M_a @ (np.sqrt(eta[ip]) * V[:, ip] + np.sqrt(-eta[im]) * V[:, im])
    p2 = M_a @ (np.sqrt(eta[ip]) * V[:, ip] - np.sqrt(-eta[im]) * V[:, im])
    return [_normalize_plane(p1, M_a), _normalize_plane(p2, M_a)]


def high_precision_cmc(
    M_a: np.ndarray,
    M_m: np.ndarray,
    C: np.ndarray,
    *,
    digits: int = 100,
) -> np.ndarray:
    """Exact-equation CMC oracle on canonical binary64 metric entries."""
    with mp.workdps(int(digits)):
        A, M, X = _mp_metric(M_a), _mp_metric(M_m), _mp_matrix(C)
        D = X.T * M * X - A
        return np.asarray(
            [[float(D[i, j]) for j in range(3)] for i in range(3)], float
        )


def high_precision_smc(
    M_a: np.ndarray,
    M_m: np.ndarray,
    C: np.ndarray,
    *,
    digits: int = 100,
) -> np.ndarray:
    r"""Independent Eq.-41 SMC oracle using the *other* exact algebraic route.

    Production evaluates ``A^-1 - (C.T M C)^-1``.  The validator instead
    evaluates the defining expression

        ``A^-1 - C^-1 M^-1 C^-T``

    at arbitrary precision.  Agreement therefore checks the inverse identity
    rather than sharing production's implementation path.
    """
    with mp.workdps(int(digits)):
        A, M, X = _mp_metric(M_a), _mp_metric(M_m), _mp_matrix(C)
        Xi = X ** -1
        S = A ** -1 - Xi * (M ** -1) * Xi.T
        return np.asarray(
            [[float(S[i, j]) for j in range(3)] for i in range(3)], float
        )
