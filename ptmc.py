from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq

from .rank_one import solve_rank_one_connection, RankOneSolution


@dataclass(frozen=True)
class PTMCSolution:
    volume_fraction: float
    average_deformation: np.ndarray
    habit_connection: RankOneSolution


def middle_singular_value(F: np.ndarray) -> float:
    s = np.linalg.svd(np.asarray(F, dtype=float), compute_uv=False)
    s = np.sort(s)
    return float(s[1])


def find_ptmc_volume_fractions(
    U: np.ndarray,
    twin_a: np.ndarray,
    twin_n: np.ndarray,
    grid: int = 1001,
    tol: float = 1e-9,
) -> list[float]:
    """Solve the classical crystallographic-theory condition for a twin laminate.

    If the second variant in the common reference frame is U + a⊗n, then the
    laminate average is F(f)=U+f a⊗n. An A/M rank-one connection is possible when
    the middle singular value of F(f) is 1.
    """
    U = np.asarray(U, dtype=float)
    a = np.asarray(twin_a, dtype=float).reshape(3)
    n = np.asarray(twin_n, dtype=float).reshape(3)

    def phi(f: float) -> float:
        return middle_singular_value(U + f * np.outer(a, n)) - 1.0

    xs = np.linspace(0.0, 1.0, grid)
    ys = np.array([phi(x) for x in xs])
    roots: list[float] = []
    for x, y in zip(xs, ys):
        if abs(y) <= tol and not any(abs(x-r) < 1e-6 for r in roots):
            roots.append(float(x))
    for i in range(len(xs) - 1):
        if ys[i] * ys[i + 1] < 0:
            r = float(brentq(phi, xs[i], xs[i + 1], xtol=1e-13))
            if not any(abs(r-r0) < 1e-6 for r0 in roots):
                roots.append(r)
    return sorted(roots)


def solve_ptmc_laminate(
    U: np.ndarray,
    twin_a: np.ndarray,
    twin_n: np.ndarray,
    grid: int = 1001,
) -> list[PTMCSolution]:
    roots = find_ptmc_volume_fractions(U, twin_a, twin_n, grid=grid)
    out = []
    for f in roots:
        Fbar = U + f * np.outer(twin_a, twin_n)
        # Seek R Fbar - I = b⊗m.
        habit = solve_rank_one_connection(Fbar, np.eye(3))
        out.append(PTMCSolution(f, Fbar, habit))
    return out
