from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar


def middle_sv(F: np.ndarray) -> float:
    return float(np.sort(np.linalg.svd(np.asarray(F, float), compute_uv=False))[1])


def independent_ptmc_roots(
    U: np.ndarray,
    a: np.ndarray,
    n: np.ndarray,
    *,
    tol: float = 1e-9,
) -> list[float]:
    """Independent scan+bracketing PTMC oracle using the same declared tolerance.

    The route is intentionally unlike production: it scans the middle singular
    value directly, brackets sign changes, and minimizes absolute residual to
    detect tangent/double roots.  It does not use the determinant polynomial.
    """
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    U = np.asarray(U, float)
    a = np.asarray(a, float)
    n = np.asarray(n, float)

    def h(f):
        return middle_sv(U + float(f) * np.outer(a, n)) - 1.0

    grid = np.linspace(0.0, 1.0, 4001)
    vals = np.array([h(x) for x in grid])
    roots = []
    for i in range(len(grid) - 1):
        x0, x1 = grid[i], grid[i + 1]
        y0, y1 = vals[i], vals[i + 1]
        if abs(y0) <= tol:
            roots.append(float(x0))
        if y0 * y1 < 0:
            roots.append(float(brentq(h, x0, x1, xtol=1e-13, rtol=1e-13)))

    # Tangent/double roots.  Acceptance uses the exact caller tolerance rather
    # than the former hidden 2e-8 relaxation.
    coarse = grid[::100]
    if coarse[-1] != 1.0:
        coarse = np.append(coarse, 1.0)
    for lo, hi in zip(coarse[:-1], coarse[1:]):
        res = minimize_scalar(
            lambda x: abs(h(x)),
            bounds=(float(lo), float(hi)),
            method="bounded",
            options={"xatol": 1e-13},
        )
        if res.fun <= tol:
            roots.append(float(res.x))

    roots = sorted(min(1.0, max(0.0, r)) for r in roots)
    unique = []
    merge = max(10.0 * tol, 1e-10)
    for r in roots:
        if not unique or abs(r - unique[-1]) > merge:
            unique.append(r)
    return unique
