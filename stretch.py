from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm

from .correspondence import Correspondence
from .ct import normalized_correspondence_metric


def stretch_from_metrics(M_a: np.ndarray, M_m: np.ndarray, correspondence: Correspondence) -> np.ndarray:
    """Return positive-definite stretch U in the parent metric-whitened orthonormal basis.

    G_hat = M_A^{-1/2} C^T M_M C M_A^{-1/2} = U^2.
    """
    U2 = normalized_correspondence_metric(M_a, M_m, correspondence)
    U2 = 0.5 * (U2 + U2.T)
    U = np.real_if_close(sqrtm(U2))
    U = np.asarray(U, dtype=float)
    return 0.5 * (U + U.T)


def principal_stretches(U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(0.5 * (U + U.T))
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def generate_stretch_variants(U: np.ndarray, parent_proper_rotations: list[np.ndarray], tol: float = 1e-9) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for Q in parent_proper_rotations:
        Q = np.asarray(Q, dtype=float)
        V = Q @ U @ Q.T
        if not any(np.linalg.norm(V - W) <= tol for W in out):
            out.append(V)
    return out
