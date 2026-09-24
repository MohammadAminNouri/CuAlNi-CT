from __future__ import annotations

"""Classical crystallographic-theory (PTMC) laminate compatibility."""

from dataclasses import dataclass

import mpmath as mp
import numpy as np
import sympy as sp

from .ball_james import AnalyticalRankOneSolution, analytical_rank_one_connections
from .rank_one import RankOneSolution, solve_rank_one_connection


@dataclass(frozen=True)
class PTMCSolution:
    volume_fraction: float
    average_deformation: np.ndarray
    habit_connections: tuple[AnalyticalRankOneSolution, ...]
    middle_stretch_residual: float


def middle_singular_value(F: np.ndarray) -> float:
    return float(np.sort(np.linalg.svd(np.asarray(F, float), compute_uv=False))[1])


def _g(U: np.ndarray, a: np.ndarray, n: np.ndarray, f: float) -> float:
    """Direct binary64 evaluation of the unchanged PTMC determinant equation."""
    F = U + f * np.outer(a, n)
    return float(np.linalg.det(F.T @ F - np.eye(3)))


def _rat(x: float) -> sp.Rational:
    num, den = float(x).as_integer_ratio()
    return sp.Rational(num, den)


def _sp_matrix(A: np.ndarray) -> sp.Matrix:
    return sp.Matrix([[_rat(x) for x in row] for row in np.asarray(A, float)])


def _mpf_exact_binary64(x: float) -> mp.mpf:
    num, den = float(x).as_integer_ratio()
    return mp.mpf(num) / mp.mpf(den)


def _mp_matrix(A: np.ndarray) -> mp.matrix:
    A = np.asarray(A, float)
    return mp.matrix([[_mpf_exact_binary64(x) for x in row] for row in A])


def _exact_binary64_ptmc_polynomial(
    U: np.ndarray, a: np.ndarray, n: np.ndarray
) -> sp.Poly:
    """Exact rational polynomial of det((U+f a⊗n)^T(U+f a⊗n)-I)."""
    f = sp.symbols("f")
    Us = _sp_matrix(U)
    av = sp.Matrix([_rat(x) for x in np.asarray(a, float).reshape(3)])
    nv = sp.Matrix([_rat(x) for x in np.asarray(n, float).reshape(3)])
    F = Us + f * (av * nv.T)
    expr = sp.expand((F.T * F - sp.eye(3)).det())
    return sp.Poly(expr, f)


def _certified_real_roots(poly: sp.Poly, *, decimal_digits: int = 35) -> list[float]:
    if poly.is_zero:
        raise ValueError(
            "PTMC determinant polynomial is identically zero: compatibility "
            "holds for a continuum of volume fractions, which cannot be "
            "represented by this discrete-root API"
        )
    if poly.degree() <= 0:
        return []
    eps = sp.Rational(1, 10 ** int(decimal_digits))
    intervals = sp.polys.polytools.intervals(poly, eps=eps)
    roots: list[float] = []
    for (lo, hi), multiplicity in intervals:
        root = float((lo + hi) / 2)
        roots.extend([root] * int(multiplicity))
    return roots


def _high_precision_middle_stretch(
    U: np.ndarray, a: np.ndarray, n: np.ndarray, f: float, *, digits: int = 90
) -> mp.mpf:
    with mp.workdps(int(digits)):
        Um = _mp_matrix(U)
        am = mp.matrix([_mpf_exact_binary64(x) for x in np.asarray(a, float).reshape(3)])
        nm = mp.matrix([_mpf_exact_binary64(x) for x in np.asarray(n, float).reshape(3)])
        fm = _mpf_exact_binary64(float(f))
        F = Um + fm * (am * nm.T)
        H = F.T * F
        H = (H + H.T) / 2
        values, _ = mp.eigsy(H)
        vals = sorted(mp.mpf(v) for v in values)
        if vals[1] < 0:
            raise ArithmeticError("PTMC F.T@F produced a negative high-precision eigenvalue")
        return mp.sqrt(vals[1])


def ptmc_volume_fractions(
    U: np.ndarray, a: np.ndarray, n: np.ndarray, tol: float = 1e-9
) -> list[float]:
    r"""Solve the PTMC laminate volume-fraction condition without hidden tolerance widening.

    The scientific equation remains

    .. math::

        \det\!\left[(U+f\,a\otimes n)^T(U+f\,a\otimes n)-I\right]=0.

    V3 constructs this polynomial from the *exact IEEE-754 input values* as
    SymPy rationals and isolates all real roots.  This removes the old hidden
    assumptions/thresholds used to reconstruct a quadratic from three samples.
    Every candidate in ``[0,1]`` is then accepted only when the high-precision
    middle singular value satisfies the caller's actual ``tol``.  The hidden
    ``1e-6`` acceptance is removed.
    """
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    U = np.asarray(U, float)
    a = np.asarray(a, float).reshape(3)
    n = np.asarray(n, float).reshape(3)
    if U.shape != (3, 3) or not np.all(np.isfinite(U)):
        raise ValueError("U must be a finite 3x3 matrix")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(n)):
        raise ValueError("a and n must be finite")

    poly = _exact_binary64_ptmc_polynomial(U, a, n)
    candidates = _certified_real_roots(poly)
    out: list[float] = []
    merge = max(64.0 * np.finfo(float).eps, 0.1 * float(tol))

    for root in candidates:
        if root < -tol or root > 1.0 + tol:
            continue
        f = min(1.0, max(0.0, float(root)))
        residual = abs(float(_high_precision_middle_stretch(U, a, n, f)) - 1.0)
        if residual <= tol and not any(abs(f - r) <= merge for r in out):
            out.append(f)

    # Endpoint roots are checked explicitly against the same declared tolerance
    # so interval-to-float conversion cannot drop f=0 or f=1.
    for f in (0.0, 1.0):
        residual = abs(float(_high_precision_middle_stretch(U, a, n, f)) - 1.0)
        if residual <= tol and not any(abs(f - r) <= merge for r in out):
            out.append(f)

    return sorted(out)


def solve_ptmc_laminate(
    U: np.ndarray, twin_a: np.ndarray, twin_n: np.ndarray, tol: float = 1e-9
) -> list[PTMCSolution]:
    out = []
    for f in ptmc_volume_fractions(U, twin_a, twin_n, tol):
        Fbar = np.asarray(U, float) + f * np.outer(twin_a, twin_n)
        habits = tuple(analytical_rank_one_connections(Fbar, np.eye(3), tol=tol))
        out.append(
            PTMCSolution(
                f,
                Fbar,
                habits,
                abs(middle_singular_value(Fbar) - 1.0),
            )
        )
    return out


def solve_ptmc_laminate_numerical_crosscheck(
    U: np.ndarray, twin_a: np.ndarray, twin_n: np.ndarray, f: float
) -> RankOneSolution:
    """Independent optimizer-based rank-one check for a PTMC solution."""
    Fbar = np.asarray(U, float) + float(f) * np.outer(twin_a, twin_n)
    return solve_rank_one_connection(Fbar, np.eye(3))
