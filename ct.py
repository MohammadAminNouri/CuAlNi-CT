from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh

from .correspondence import Correspondence
from .lattice import (
    metric_inv_sqrt,
    metric_sqrt,
    normalize_plane,
    plane_to_unit_normal,
    metric_norm,
)


def cmc(M_a: np.ndarray, M_m: np.ndarray, correspondence: Correspondence) -> np.ndarray:
    """Cayron Compatibility by Metric Correspondence matrix.

    With u_M = C u_A:
        CMC = C^T M_M C - M_A
    """
    C = np.array(correspondence.C_m_from_a, dtype=float)
    return C.T @ M_m @ C - M_a


def normalized_correspondence_metric(
    M_a: np.ndarray, M_m: np.ndarray, correspondence: Correspondence
) -> np.ndarray:
    """Dimensionless metric pullback M_A^{-1/2} C^T M_M C M_A^{-1/2}.

    This normalization is a project-level comparison tool, not a replacement for Cayron's
    dimensional crystallographic formulation.
    """
    W = metric_inv_sqrt(M_a)
    C = np.array(correspondence.C_m_from_a, dtype=float)
    return W @ (C.T @ M_m @ C) @ W


def normalized_cmc(M_a: np.ndarray, M_m: np.ndarray, correspondence: Correspondence) -> np.ndarray:
    return normalized_correspondence_metric(M_a, M_m, correspondence) - np.eye(3)


@dataclass(frozen=True)
class CMCAnalysis:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    compatible: bool
    nullity: int
    reason: str


def analyze_cmc(
    M_a: np.ndarray, M_m: np.ndarray, correspondence: Correspondence, tol: float = 1e-8
) -> CMCAnalysis:
    """Analyze CMC in the dimensionless metric-whitened parent basis."""
    D = 0.5 * (normalized_cmc(M_a, M_m, correspondence) + normalized_cmc(M_a, M_m, correspondence).T)
    vals, vecs = eigh(D)
    zero = np.abs(vals) <= tol
    nullity = int(np.sum(zero))
    if nullity == 3:
        return CMCAnalysis(vals, vecs, True, 3, "third-order degeneracy: full space")
    if nullity == 2:
        return CMCAnalysis(vals, vecs, True, 2, "second-order degeneracy: one plane")
    if nullity == 1:
        nz = vals[~zero]
        ok = bool(nz[0] * nz[1] <= tol)
        return CMCAnalysis(vals, vecs, ok, 1, "first-order degeneracy" if ok else "zero eigenvalue but wrong signature")
    return CMCAnalysis(vals, vecs, False, 0, "no CMC degeneracy")


def habit_planes_from_cmc(
    M_a: np.ndarray, M_m: np.ndarray, correspondence: Correspondence, tol: float = 1e-8
) -> list[np.ndarray]:
    """Return compatible A/M habit-plane covectors in parent crystal coordinates.

    The factorization is done in the metric-whitened parent basis. The result is transformed
    back to crystallographic reciprocal coordinates and normalized with M_A^{-1}.
    """
    ana = analyze_cmc(M_a, M_m, correspondence, tol=tol)
    if not ana.compatible:
        return []
    vals, Q = ana.eigenvalues, ana.eigenvectors
    S = metric_sqrt(M_a)
    zero = np.abs(vals) <= tol

    if ana.nullity == 3:
        return []  # every direction is invariant; no unique habit plane

    if ana.nullity == 2:
        idx = int(np.where(~zero)[0][0])
        p_hat = Q[:, idx]
        p = S @ p_hat
        return [normalize_plane(p, M_a)]

    iz = int(np.where(zero)[0][0])
    nz_idx = [i for i in range(3) if i != iz]
    i, j = nz_idx
    qi, qj = vals[i], vals[j]
    if qi * qj > tol:
        return []
    # q_i y_i^2 + q_j y_j^2 = 0 -> sqrt(|q_i|) y_i +/- sqrt(|q_j|) y_j = 0
    # coefficients are arranged so the product reconstructs the quadratic form up to scale.
    if qi > 0 and qj < 0:
        pos_i, neg_i = i, j
    elif qj > 0 and qi < 0:
        pos_i, neg_i = j, i
    else:
        return []
    p1_hat = np.sqrt(vals[pos_i]) * Q[:, pos_i] + np.sqrt(-vals[neg_i]) * Q[:, neg_i]
    p2_hat = np.sqrt(vals[pos_i]) * Q[:, pos_i] - np.sqrt(-vals[neg_i]) * Q[:, neg_i]
    return [normalize_plane(S @ p1_hat, M_a), normalize_plane(S @ p2_hat, M_a)]


def smc(M_a: np.ndarray, M_m: np.ndarray, correspondence: Correspondence) -> np.ndarray:
    """Cayron Shear by Metric Correspondence in this package's mapping convention.

    u_M = C u_A, hence:
        SMC = M_A^{-1} - C^{-1} M_M^{-1} C^{-T}
    """
    C = np.array(correspondence.C_m_from_a, dtype=float)
    Ci = np.linalg.inv(C)
    return np.linalg.inv(M_a) - Ci @ np.linalg.inv(M_m) @ Ci.T


def ips_shear_from_habit_plane(
    p_a: np.ndarray, M_a: np.ndarray, M_m: np.ndarray, correspondence: Correspondence
) -> np.ndarray:
    p = normalize_plane(p_a, M_a)
    return smc(M_a, M_m, correspondence) @ p


def ct_supercompatibility_residual(
    habit_plane_a: np.ndarray,
    d_a: np.ndarray,
    twin_plane_a: np.ndarray,
    twin_direction_a: np.ndarray,
    twin_shear: float,
    M_a: np.ndarray,
) -> float:
    """Cayron shear/shear incompatibility epsilon.

    All plane covectors are normalized in reciprocal metric and all directions in direct metric.
    twin_direction_a is treated as a direction; it is rescaled to length twin_shear.
    """
    m = normalize_plane(habit_plane_a, M_a)
    n = plane_to_unit_normal(twin_plane_a, M_a)
    a_dir = np.asarray(twin_direction_a, dtype=float).reshape(3)
    a_dir = a_dir / metric_norm(a_dir, M_a)
    a = float(twin_shear) * a_dir
    lhs = 2.0 * float(m @ n) * np.asarray(d_a, dtype=float).reshape(3)
    denom = abs(float(twin_shear))
    if denom == 0:
        raise ValueError("Twin shear magnitude must be nonzero.")
    return metric_norm(lhs - a, M_a) / denom
