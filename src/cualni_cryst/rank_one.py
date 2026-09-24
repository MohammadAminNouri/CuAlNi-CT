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
    if D.shape != (3, 3) or not np.all(np.isfinite(D)):
        raise ValueError("D must be a finite 3x3 matrix")
    s = np.linalg.svd(D, compute_uv=False)
    scale = max(s[0], np.finfo(float).tiny)
    return float(np.hypot(s[1], s[2]) / scale)


def _extract_rank_one(D: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(D)
    a = s[0] * U[:, 0]
    n = Vt[0, :]
    return a, n, s


def _validated_gradient(F: np.ndarray, name: str) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    if F.shape != (3, 3) or not np.all(np.isfinite(F)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    return F


def solve_rank_one_connection(
    F1: np.ndarray,
    F2: np.ndarray,
    starts: int = 24,
    tol: float = 1e-8,
) -> RankOneSolution:
    """Numerically solve R F1 - F2 = a⊗n, R∈SO(3).

    This optimizer is an independent cross-check, not a replacement for the
    analytical Ball-James theorem.  ``tol`` does not alter the returned
    residual; callers decide how to interpret it.
    """
    if starts < 1:
        raise ValueError("starts must be >= 1")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    F1 = _validated_gradient(F1, "F1")
    F2 = _validated_gradient(F2, "F2")

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
    seeds = seeds[:starts]

    def objective(rv: np.ndarray) -> float:
        R = Rotation.from_rotvec(rv).as_matrix()
        return rank_one_residual(R @ F1 - F2) ** 2

    best = None
    for seed in seeds:
        res = minimize(
            objective,
            seed,
            method="BFGS",
            options={"gtol": 1e-11, "maxiter": 1000},
        )
        if best is None or res.fun < best.fun:
            best = res

    if best is None:
        raise RuntimeError("rank-one optimizer produced no result")
    R = Rotation.from_rotvec(best.x).as_matrix()
    D = R @ F1 - F2
    a, n, s = _extract_rank_one(D)
    rr = rank_one_residual(D)
    return RankOneSolution(R, a, n, float(rr), s)
