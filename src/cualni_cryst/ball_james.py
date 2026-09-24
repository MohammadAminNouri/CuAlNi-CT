from __future__ import annotations

"""Analytical rank-one compatibility and Mallard-law twin solutions.

These routines form an independent nonlinear-elasticity branch.  They do not
call the CT twin formulas, which makes CT-vs-Ball-James comparison meaningful.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnalyticalRankOneSolution:
    rotation: np.ndarray
    a: np.ndarray
    n: np.ndarray
    branch: int
    residual: float
    eigenvalues_C: np.ndarray


def _rank_one_relative_residual(D: np.ndarray) -> float:
    s = np.linalg.svd(D, compute_uv=False)
    return float(np.hypot(s[1], s[2]) / max(s[0], np.finfo(float).tiny))


def _validated_gradient(F: np.ndarray, name: str) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    if F.shape != (3, 3):
        raise ValueError(f"{name} must be 3x3")
    if not np.all(np.isfinite(F)):
        raise ValueError(f"{name} contains non-finite entries")
    s = np.linalg.svd(F, compute_uv=False)
    if s[0] == 0.0 or s[-1] <= 3.0 * np.finfo(float).eps * s[0]:
        raise ValueError(f"{name} is numerically singular; singular_values={s}")
    return F


def analytical_rank_one_connections(
    F1: np.ndarray, F2: np.ndarray, tol: float = 1e-9
) -> list[AnalyticalRankOneSolution]:
    r"""Solve ``R F1 - F2 = a⊗n`` using the Ball-James proposition.

    Put ``A=F1 F2^{-1}`` and ``C=A^T A``.  A rank-one connection exists when
    the ordered eigenvalues ``L1<=L2<=L3`` satisfy ``L1>0`` and ``L2=1``
    (excluding the trivial ``C=I`` case).  The formulas below choose the
    otherwise arbitrary Ball-James scaling parameter ``rho=1``.

    Explicit inverse multiplications are evaluated by algebraically equivalent
    linear solves; the theorem and formulas are unchanged.
    """
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    F1 = _validated_gradient(F1, "F1")
    F2 = _validated_gradient(F2, "F2")

    # A = F1 @ F2^{-1}, evaluated without explicitly forming F2^{-1}.
    A = np.linalg.solve(F2.T, F1.T).T
    C = A.T @ A
    L, V = np.linalg.eigh(0.5 * (C + C.T))
    order = np.argsort(L)
    L = L[order]
    V = V[:, order]
    L1, L2, L3 = map(float, L)

    if L1 <= 0.0 or abs(L2 - 1.0) > tol:
        return []
    if abs(L3 - L1) <= tol:
        return []
    # The explicit real formulas require 1-L1 >= 0 and L3-1 >= 0.
    # Do not hide a sign violation with sqrt(max(...,0)).
    if L1 > 1.0 or L3 < 1.0:
        return []

    e1, e3 = V[:, 0], V[:, 2]
    gap = L3 - L1
    den = np.sqrt(gap)
    r1 = 1.0 - L1
    r3 = L3 - 1.0
    if r1 < 0.0 or r3 < 0.0:
        return []

    out: list[AnalyticalRankOneSolution] = []
    for kappa in (-1, 1):
        b = (
            np.sqrt(L3 * r1) * e1
            + kappa * np.sqrt(L1 * r3) * e3
        ) / den
        m = (np.sqrt(L3) - np.sqrt(L1)) / den * (
            -np.sqrt(r1) * e1 + kappa * np.sqrt(r3) * e3
        )

        # R=(I+b⊗m)A^{-1}, evaluated by a right solve.
        K = np.eye(3) + np.outer(b, m)
        R = np.linalg.solve(A.T, K.T).T

        # Map R A-I=b⊗m back to R F1-F2=a⊗n.
        a = b
        n = F2.T @ m
        D = R @ F1 - F2
        rank_res = np.linalg.norm(D - np.outer(a, n), "fro") / max(
            np.linalg.norm(D, "fro"), np.finfo(float).tiny
        )
        ortho = np.linalg.norm(R.T @ R - np.eye(3), "fro")
        deterr = abs(np.linalg.det(R) - 1.0)
        resid = max(float(rank_res), float(ortho), float(deterr))
        out.append(AnalyticalRankOneSolution(R, a, n, kappa, resid, L.copy()))
    return out


def single_variant_austenite_habit_solutions(
    U: np.ndarray, tol: float = 1e-9
) -> list[AnalyticalRankOneSolution]:
    """Solve R U - I = b⊗m for a single martensite stretch variant."""
    return analytical_rank_one_connections(np.asarray(U, float), np.eye(3), tol=tol)


@dataclass(frozen=True)
class MallardTwinSolution:
    kind: str
    rotation: np.ndarray
    a: np.ndarray
    n: np.ndarray
    shear_magnitude: float
    residual: float


def mallard_law_twins(
    Ui: np.ndarray, Uj: np.ndarray, twofold_axis: np.ndarray, tol: float = 1e-8
) -> list[MallardTwinSolution]:
    r"""Type-I and Type-II Mallard-law solutions for symmetry-related stretches.

    Assumes ``Uj = Q Ui Q`` with Q the parent 180° rotation about unit axis e.
    Implements the formulas reproduced by James & Hane, Eqs. (16)-(17).
    The scaling of ``a`` and ``n`` is non-unique; Type-II is normalized here so
    ``|n|=1``.
    """
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    Ui = _validated_gradient(Ui, "Ui")
    Uj = _validated_gradient(Uj, "Uj")
    e = np.asarray(twofold_axis, float).reshape(3)
    en = float(np.linalg.norm(e))
    if not np.isfinite(en) or en == 0.0:
        raise ValueError("twofold_axis must be a finite nonzero vector")
    e = e / en
    Q = -np.eye(3) + 2 * np.outer(e, e)
    symres = np.linalg.norm(Uj - Q @ Ui @ Q, "fro") / max(
        np.linalg.norm(Uj, "fro"), np.finfo(float).tiny
    )
    if symres > tol:
        raise ValueError(
            f"Uj is not related to Ui by supplied 180° symmetry; relative residual={symres:g}"
        )

    # Type I
    x = np.linalg.solve(Uj, e)
    x2 = float(x @ x)
    if x2 <= 0.0:
        raise ArithmeticError("Mallard Type-I auxiliary norm is non-positive")
    aI = 2.0 * (x / x2 - Uj @ e)
    nI = e.copy()
    RI = (-np.eye(3) + 2.0 * np.outer(x, x) / x2) @ Q

    # Type II
    y = Uj @ e
    y2 = float(y @ y)
    if y2 <= 0.0:
        raise ArithmeticError("Mallard Type-II auxiliary norm is non-positive")
    n0 = e - (Uj @ Uj @ e) / y2
    n0norm = np.linalg.norm(n0)
    if n0norm < tol:
        typeII = None
    else:
        rho = 2.0 * n0norm
        nII = (2.0 / rho) * n0
        aII = rho * y
        RII = (-np.eye(3) + 2.0 * np.outer(y, y) / y2) @ Q
        typeII = (RII, aII, nII)

    out: list[MallardTwinSolution] = []
    for kind, R, a, n in [("I", RI, aI, nI)] + (
        [] if typeII is None else [("II", *typeII)]
    ):
        D = R @ Ui - Uj
        resid = np.linalg.norm(D - np.outer(a, n), "fro") / max(
            np.linalg.norm(D, "fro"), np.finfo(float).tiny
        )
        shear = float(np.linalg.norm(np.linalg.solve(Uj.T, n)) * np.linalg.norm(a))
        out.append(MallardTwinSolution(kind, R, a, n, shear, float(resid)))
    return out
