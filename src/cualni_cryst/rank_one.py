from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class RankOneSolution:
    rotation: np.ndarray
    a: np.ndarray
    n: np.ndarray
    residual: float
    singular_values: np.ndarray


def rank_one_residual(D: np.ndarray) -> float:
    """Scale-free residual: zero iff D has rank <=1 (up to numerical precision)."""
    D = np.asarray(D, dtype=float)
    s = np.linalg.svd(D, compute_uv=False)
    scale = max(s[0], 1e-15)
    return float(np.sqrt(s[1] ** 2 + s[2] ** 2) / scale)


def _extract_rank_one(D: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(D)
    a = s[0] * U[:, 0]
    n = Vt[0, :]
    return a, n, s


def solve_rank_one_connection(
    F1: np.ndarray,
    F2: np.ndarray,
    starts: int = 24,
    tol: float = 1e-8,
) -> RankOneSolution:
    """Numerically solve R F1 - F2 = a⊗n, R∈SO(3).

    This is a transparent numerical solver, not a replacement for an analytical theorem.
    It is useful as an independent cross-check of CT/PTMC results.
    """
    F1 = np.asarray(F1, dtype=float)
    F2 = np.asarray(F2, dtype=float)

    # Deterministic spread of rotation-vector starts.
    axes = np.array(
        [
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1],
            [1, -1, 0], [1, 0, -1], [0, 1, -1],
        ], dtype=float
    )
    axes = axes / np.linalg.norm(axes, axis=1)[:, None]
    angles = np.linspace(0, np.pi, max(3, starts // len(axes) + 1))
    seeds = [np.zeros(3)]
    for ax in axes:
        for ang in angles[1:]:
            seeds.append(ax * ang)
    seeds = seeds[: max(starts, 1)]

    def objective(rv: np.ndarray) -> float:
        R = Rotation.from_rotvec(rv).as_matrix()
        D = R @ F1 - F2
        r = rank_one_residual(D)
        return r * r

    best = None
    for seed in seeds:
        res = minimize(objective, seed, method="BFGS", options={"gtol": 1e-11, "maxiter": 1000})
        if best is None or res.fun < best.fun:
            best = res

    assert best is not None
    R = Rotation.from_rotvec(best.x).as_matrix()
    D = R @ F1 - F2
    a, n, s = _extract_rank_one(D)
    rr = rank_one_residual(D)
    # Keep the result even if approximate; caller sees residual.
    if rr < tol:
        rr = float(rr)
    return RankOneSolution(R, a, n, float(rr), s)
