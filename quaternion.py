from __future__ import annotations

import numpy as np


def cross_tensor(M: np.ndarray) -> np.ndarray:
    """Cayron crystallographic cross tensor X = sqrt(det M) M^{-1}."""
    M = np.asarray(M, dtype=float)
    return np.sqrt(np.linalg.det(M)) * np.linalg.inv(M)


def crystallographic_quaternion_product(
    q1: tuple[float, np.ndarray], q2: tuple[float, np.ndarray], M: np.ndarray
) -> tuple[float, np.ndarray]:
    """Product of crystallographic quaternions q=(scalar, vector).

    Vectors are written directly in the crystallographic basis.
    """
    c1, u1 = q1
    c2, u2 = q2
    u1 = np.asarray(u1, dtype=float).reshape(3)
    u2 = np.asarray(u2, dtype=float).reshape(3)
    X = cross_tensor(M)
    c3 = float(c1 * c2 - u1 @ M @ u2)
    u3 = c2 * u1 + c1 * u2 + X @ np.cross(u1, u2)
    return c3, u3
