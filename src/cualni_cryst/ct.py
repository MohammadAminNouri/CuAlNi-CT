from __future__ import annotations

"""Cayron Correspondence-Theory metric compatibility (CMC/SMC) tools."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from .correspondence import Correspondence
from .lattice import (
    metric_inv_sqrt,
    metric_norm,
    metric_sqrt,
    normalize_plane,
    plane_to_unit_normal,
)


def _validated_metric(M: np.ndarray, name: str) -> np.ndarray:
    M = np.asarray(M, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(M)):
        raise ValueError(f"{name} contains non-finite entries")
    if np.linalg.norm(M - M.T, ord="fro") > 1e-10 * max(np.linalg.norm(M), 1.0):
        raise ValueError(f"{name} must be symmetric")
    M = 0.5 * (M + M.T)
    eig = np.linalg.eigvalsh(M)
    if np.min(eig) <= 0:
        raise ValueError(f"{name} must be positive definite; eigenvalues={eig}")
    return M


def _validated_inputs(
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    M_a = _validated_metric(M_a, "M_a")
    M_m = _validated_metric(M_m, "M_m")
    C = np.asarray(correspondence.C_m_from_a, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(C)):
        raise ValueError("Correspondence contains non-finite entries")
    if abs(float(np.linalg.det(C))) < 1e-14:
        raise ValueError("Correspondence must be invertible")
    return M_a, M_m, C


def cmc(
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
) -> np.ndarray:
    r"""Dimensional CMC = C^T M_M C - M_A (Cayron 2026 Eq. 32)."""
    M_a, M_m, C = _validated_inputs(M_a, M_m, correspondence)
    A = C.T @ M_m @ C - M_a
    return 0.5 * (A + A.T)


def normalized_correspondence_metric(
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
) -> np.ndarray:
    r"""Dimensionless parent-whitened pulled-back daughter metric.

    .. math::

        \widehat G=M_A^{-1/2} C^T M_M C M_A^{-1/2}.

    This normalization is a project comparison/representation utility, not a
    replacement for Cayron's dimensional crystallographic formulation.
    """
    M_a, M_m, C = _validated_inputs(M_a, M_m, correspondence)
    W = metric_inv_sqrt(M_a)
    G = W @ (C.T @ M_m @ C) @ W
    return 0.5 * (G + G.T)


def normalized_cmc(
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
) -> np.ndarray:
    return normalized_correspondence_metric(M_a, M_m, correspondence) - np.eye(3)


@dataclass(frozen=True)
class CMCAnalysis:
    eigenvalues: np.ndarray
    eigenvectors_whitened: np.ndarray
    exact_compatible: bool
    degeneracy_order: int
    reason: str
    zero_index: int | None
    nearest_zero_index: int
    nearest_zero_residual: float
    signature_after_nearest_zero: tuple[int, int]
    inertia: tuple[int, int, int]  # (negative, zero, positive)


def analyze_cmc(
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
    tol: float = 1e-8,
) -> CMCAnalysis:
    """Analyze Cayron's CMC degeneracy in a parent metric-orthonormal basis.

    The dimensionless matrix is used so that ``tol`` is an algebraic tolerance,
    not a length-squared tolerance.  Exact first-order compatibility requires
    one zero eigenvalue and opposite signs for the other two.  Two and three
    zero eigenvalues are the second- and third-order degeneracies described by
    Cayron.
    """
    if tol <= 0:
        raise ValueError("tol must be positive")
    D = normalized_cmc(M_a, M_m, correspondence)
    vals, Q = eigh(0.5 * (D + D.T))
    zero = np.abs(vals) <= tol
    neg = vals < -tol
    pos = vals > tol
    nullity = int(np.sum(zero))
    inertia = (int(np.sum(neg)), nullity, int(np.sum(pos)))
    nearest = int(np.argmin(np.abs(vals)))
    nz_near = [vals[i] for i in range(3) if i != nearest]
    sig = (
        sum(x > tol for x in nz_near),
        sum(x < -tol for x in nz_near),
    )

    if nullity == 3:
        return CMCAnalysis(
            vals,
            Q,
            True,
            3,
            "third-order degeneracy: pulled-back martensite metric equals parent metric",
            None,
            nearest,
            float(abs(vals[nearest])),
            sig,
            inertia,
        )

    if nullity == 2:
        return CMCAnalysis(
            vals,
            Q,
            True,
            2,
            "second-order degeneracy: one compatible habit plane",
            None,
            nearest,
            float(abs(vals[nearest])),
            sig,
            inertia,
        )

    if nullity == 1:
        iz = int(np.where(zero)[0][0])
        others = [vals[i] for i in range(3) if i != iz]
        # Cayron: q_i=0 and q_j q_k <= 0.  With exactly one zero eigenvalue,
        # strict opposite signs give first-order degeneracy.
        ok = bool(others[0] * others[1] < 0.0)
        return CMCAnalysis(
            vals,
            Q,
            ok,
            1 if ok else 0,
            (
                "first-order degeneracy: two compatible habit planes"
                if ok
                else "one zero eigenvalue but the remaining CMC eigenvalues have the same sign"
            ),
            iz,
            nearest,
            float(abs(vals[nearest])),
            sig,
            inertia,
        )

    return CMCAnalysis(
        vals,
        Q,
        False,
        0,
        "no exact CMC degeneracy",
        None,
        nearest,
        float(abs(vals[nearest])),
        sig,
        inertia,
    )


def _planes_from_eigensystem(
    vals: np.ndarray,
    Q: np.ndarray,
    zero_index: int,
    M_a: np.ndarray,
    tol: float,
) -> list[np.ndarray]:
    idx = [i for i in range(3) if i != zero_index]
    i, j = idx
    qi, qj = vals[i], vals[j]
    if qi * qj > tol:
        return []
    if abs(qi) <= tol or abs(qj) <= tol:
        # Second-order degeneracy is handled explicitly by the caller.
        return []
    if qi > 0:
        ip, ineg = i, j
    else:
        ip, ineg = j, i

    # In the whitened eigenbasis the cone factors as
    # sqrt(q+) X +/- sqrt(-q-) Z = 0.
    p1h = np.sqrt(vals[ip]) * Q[:, ip] + np.sqrt(-vals[ineg]) * Q[:, ineg]
    p2h = np.sqrt(vals[ip]) * Q[:, ip] - np.sqrt(-vals[ineg]) * Q[:, ineg]

    # u_hat = M_A^(1/2) u_A, hence p_A = M_A^(1/2) p_hat.
    S = metric_sqrt(M_a)
    return [
        normalize_plane(S @ p1h, M_a),
        normalize_plane(S @ p2h, M_a),
    ]


def habit_planes_from_cmc(
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
    tol: float = 1e-8,
) -> list[np.ndarray]:
    """Exact CT A/M habit-plane covectors.

    Returns an empty list if exact CMC degeneracy is absent.  Third-order
    degeneracy also returns an empty list because every direction preserves
    length and there is no unique finite set of habit planes to enumerate.
    """
    ana = analyze_cmc(M_a, M_m, correspondence, tol)
    if not ana.exact_compatible or ana.degeneracy_order == 3:
        return []
    vals, Q = ana.eigenvalues, ana.eigenvectors_whitened
    zero = np.abs(vals) <= tol
    S = metric_sqrt(M_a)
    if ana.degeneracy_order == 2:
        k = int(np.where(~zero)[0][0])
        return [normalize_plane(S @ Q[:, k], M_a)]
    assert ana.zero_index is not None
    return _planes_from_eigensystem(vals, Q, ana.zero_index, M_a, tol)


@dataclass(frozen=True)
class ApproximateCMCResult:
    residual: float
    candidate_planes: tuple[np.ndarray, ...]
    admissible_signature: bool
    explanation: str


def approximate_cmc_habit_planes(
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
) -> ApproximateCMCResult:
    """Nearest-zero diagnostic for measured alloys near exact compatibility.

    The nearest eigenvalue of normalized CMC is set to zero *only* to generate
    a diagnostic candidate plane.  This is not an exact CT prediction, and the
    returned nonzero residual must be reported with the candidate.
    """
    D = normalized_cmc(M_a, M_m, correspondence)
    vals, Q = eigh(D)
    iz = int(np.argmin(np.abs(vals)))
    other = [vals[i] for i in range(3) if i != iz]
    admiss = bool(other[0] * other[1] < 0)
    planes = tuple(_planes_from_eigensystem(vals, Q, iz, M_a, 0.0)) if admiss else ()
    return ApproximateCMCResult(
        float(abs(vals[iz])),
        planes,
        admiss,
        "diagnostic projection to nearest first-order CMC degeneracy; not an exact compatibility solution",
    )


def smc(
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
) -> np.ndarray:
    r"""Cayron SMC: shear by metric correspondence (2026 Eq. 41).

    With the package convention ``C=C_{M<-A}``,

    .. math::

        SMC=M_A^{-1}-C^{-1}M_M^{-1}C^{-T}.

    SMC is not the inverse of CMC.
    """
    M_a, M_m, C = _validated_inputs(M_a, M_m, correspondence)
    Ci = np.linalg.inv(C)
    S = np.linalg.inv(M_a) - Ci @ np.linalg.inv(M_m) @ Ci.T
    return 0.5 * (S + S.T)


def ips_shear_from_habit_plane(
    p_a: np.ndarray,
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
) -> np.ndarray:
    """Return the direct-space IPS shear vector d_A = SMC m_A."""
    return smc(M_a, M_m, correspondence) @ normalize_plane(p_a, M_a)


def ct_supercompatibility_vector(
    habit_plane_a: np.ndarray,
    d_a: np.ndarray,
    twin_plane_a: np.ndarray,
    twin_direction_a: np.ndarray,
    twin_shear: float,
    M_a: np.ndarray,
) -> np.ndarray:
    """Residual vector of Cayron's 2(m_A^T n)d_A = a condition."""
    m = normalize_plane(habit_plane_a, M_a)
    n = plane_to_unit_normal(twin_plane_a, M_a)
    ad = np.asarray(twin_direction_a, float).reshape(3)
    ad = ad / metric_norm(ad, M_a)
    a = float(twin_shear) * ad
    return 2.0 * float(m @ n) * np.asarray(d_a, float).reshape(3) - a


def ct_supercompatibility_residual(
    habit_plane_a: np.ndarray,
    d_a: np.ndarray,
    twin_plane_a: np.ndarray,
    twin_direction_a: np.ndarray,
    twin_shear: float,
    M_a: np.ndarray,
) -> float:
    """Dimensionless Cayron shear/shear mismatch, zero at exact compatibility."""
    if abs(float(twin_shear)) < 1e-15:
        raise ValueError("Twin shear must be nonzero")
    r = ct_supercompatibility_vector(
        habit_plane_a,
        d_a,
        twin_plane_a,
        twin_direction_a,
        twin_shear,
        M_a,
    )
    return metric_norm(r, M_a) / abs(float(twin_shear))


@dataclass(frozen=True)
class CTAMResult:
    cmc_dimensional: np.ndarray
    cmc_normalized: np.ndarray
    analysis: CMCAnalysis
    exact_habit_planes: tuple[np.ndarray, ...]
    approximate: ApproximateCMCResult


def analyze_austenite_martensite(
    M_a: np.ndarray,
    M_m: np.ndarray,
    C: Correspondence,
    tol: float = 1e-8,
) -> CTAMResult:
    return CTAMResult(
        cmc(M_a, M_m, C),
        normalized_cmc(M_a, M_m, C),
        analyze_cmc(M_a, M_m, C, tol),
        tuple(habit_planes_from_cmc(M_a, M_m, C, tol)),
        approximate_cmc_habit_planes(M_a, M_m, C),
    )
