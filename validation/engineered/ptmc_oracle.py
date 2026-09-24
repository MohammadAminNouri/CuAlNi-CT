from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar


def middle_sv(F: np.ndarray) -> float:
    return float(np.sort(np.linalg.svd(np.asarray(F, float), compute_uv=False))[1])


def independent_ptmc_roots(U: np.ndarray, a: np.ndarray, n: np.ndarray) -> list[float]:
    """Independent scan+bracketing oracle, intentionally unlike production.

    It also minimizes |sigma2-1| in each interval to detect tangent/double roots.
    """
    U = np.asarray(U, float)
    a = np.asarray(a, float)
    n = np.asarray(n, float)

    def h(f):
        return middle_sv(U + float(f)*np.outer(a,n)) - 1.0

    grid = np.linspace(0.0, 1.0, 2001)
    vals = np.array([h(x) for x in grid])
    roots = []
    for i in range(len(grid)-1):
        x0, x1 = grid[i], grid[i+1]
        y0, y1 = vals[i], vals[i+1]
        if abs(y0) < 1e-9:
            roots.append(float(x0))
        if y0*y1 < 0:
            roots.append(float(brentq(h, x0, x1, xtol=1e-13, rtol=1e-13)))

    # Tangent/double roots.
    for lo, hi in zip(grid[::100][:-1], grid[::100][1:]):
        res = minimize_scalar(lambda x: abs(h(x)), bounds=(lo, hi), method="bounded")
        if res.fun < 2e-8:
            roots.append(float(res.x))

    roots = sorted(min(1.0, max(0.0, r)) for r in roots)
    unique = []
    for r in roots:
        if not unique or abs(r-unique[-1]) > 2e-6:
            unique.append(r)
    return unique
