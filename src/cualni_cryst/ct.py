from __future__ import annotations

"""Cayron Correspondence-Theory metric compatibility (CMC/SMC) tools."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from .adaptive_metric import (
    adaptive_metric_eigensystem,
    adaptive_smc,
    validate_metric_pencil_inputs,
)
from .correspondence import Correspondence
from .lattice import (
    metric_inv_sqrt,
    metric_norm,
    metric_sqrt,
    normalize_plane,
    plane_to_unit_normal,
)


def _validated_metric(M: np.ndarray, name: str) -> np.ndarray:
    try:
        A, _, _ = validate_metric_pencil_inputs(M, M, np.eye(3))
    except Exception as exc:
        raise ValueError(f"{name} is not a valid SPD metric: {exc}") from exc
    return A


def _validated_inputs(
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    C = np.asarray(correspondence.C_m_from_a, dtype=float).reshape(3, 3)
    return validate_metric_pencil_inputs(M_a, M_m, C)


def cmc(
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
) -> np.ndarray:
    r"""Dimensional CMC = C^T M_M C - M_A (Cayron 2026 Eq. 32)."""
    M_a, M_m, C = _validated_inputs(M_a, M_m, correspondence)
    A = C.T @ M_m @ C - M_a
    return A


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
    generalized_mu: np.ndarray | None = None
    eigenvectors_crystal: np.ndarray | None = None
    generalized_eigen_residual: float = float("nan")
    metric_orthonormality_residual: float = float("nan")
    solver_source: str = ""
    precision_escalated: bool = False
    route_disagreement: float = float("nan")


def analyze_cmc(
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
    tol: float = 1e-8,
) -> CMCAnalysis:
    r"""Analyze Cayron CMC degeneracy without changing the CMC equation.

    The normalized CMC spectrum is evaluated through the exactly equivalent
    generalized symmetric-definite pencil

    .. math::

        (C^T M_M C)v_i=\mu_i M_Av_i,\qquad \eta_i=\mu_i-1.

    In exact arithmetic, ``eta_i`` are exactly the eigenvalues of
    ``M_A^{-1/2}(C^T M_M C)M_A^{-1/2}-I``.  Numerical precision is escalated
    when required; ``tol`` itself is never enlarged.
    """
    if tol <= 0:
        raise ValueError("tol must be positive")
    M_a, M_m, C = _validated_inputs(M_a, M_m, correspondence)
    spec = adaptive_metric_eigensystem(M_a, M_m, C, decision_tol=tol)
    vals = np.asarray(spec.mu - 1.0, dtype=float)
    V = np.asarray(spec.eigenvectors_crystal, dtype=float)
    # Backward-compatible representation of the eigenvectors in the symmetric
    # M_A^(1/2) orthonormal basis.  Habit-plane construction below does not
    # depend on this extra transformation.
    Q = metric_sqrt(M_a) @ V

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

    common = dict(
        eigenvalues=vals,
        eigenvectors_whitened=Q,
        nearest_zero_index=nearest,
        nearest_zero_residual=float(abs(vals[nearest])),
        signature_after_nearest_zero=sig,
        inertia=inertia,
        generalized_mu=np.asarray(spec.mu, float),
        eigenvectors_crystal=V,
        generalized_eigen_residual=spec.eigen_equation_residual,
        metric_orthonormality_residual=spec.metric_orthonormality_residual,
        solver_source=spec.source,
        precision_escalated=spec.escalated,
        route_disagreement=spec.route_disagreement,
    )

    if nullity == 3:
        return CMCAnalysis(
            exact_compatible=True,
            degeneracy_order=3,
            reason="third-order degeneracy: pulled-back martensite metric equals parent metric",
            zero_index=None,
            **common,
        )

    if nullity == 2:
        return CMCAnalysis(
            exact_compatible=True,
            degeneracy_order=2,
            reason="second-order degeneracy: one compatible habit plane",
            zero_index=None,
            **common,
        )

    if nullity == 1:
        iz = int(np.where(zero)[0][0])
        others = [vals[i] for i in range(3) if i != iz]
        ok = bool(others[0] * others[1] < 0.0)
        return CMCAnalysis(
            exact_compatible=ok,
            degeneracy_order=1 if ok else 0,
            reason=(
                "first-order degeneracy: two compatible habit planes"
                if ok
                else "one zero eigenvalue but the remaining CMC eigenvalues have the same sign"
            ),
            zero_index=iz,
            **common,
        )

    return CMCAnalysis(
        exact_compatible=False,
        degeneracy_order=0,
        reason="no exact CMC degeneracy",
        zero_index=None,
        **common,
    )


def _planes_from_eigensystem(
    vals: np.ndarray,
    Q: np.ndarray,
    zero_index: int,
    M_a: np.ndarray,
    tol: float,
) -> list[np.ndarray]:
    """Legacy whitened-basis factorization, retained for equivalence tests."""
    idx = [i for i in range(3) if i != zero_index]
    i, j = idx
    qi, qj = float(vals[i]), float(vals[j])
    if qi * qj >= 0.0:
        return []
    if abs(qi) <= tol or abs(qj) <= tol:
        return []
    if qi > 0:
        ip, ineg = i, j
    else:
        ip, ineg = j, i

    p1h = np.sqrt(vals[ip]) * Q[:, ip] + np.sqrt(-vals[ineg]) * Q[:, ineg]
    p2h = np.sqrt(vals[ip]) * Q[:, ip] - np.sqrt(-vals[ineg]) * Q[:, ineg]
    S = metric_sqrt(M_a)
    return [
        normalize_plane(S @ p1h, M_a),
        normalize_plane(S @ p2h, M_a),
    ]


def _planes_from_generalized_eigensystem(
    vals: np.ndarray,
    V: np.ndarray,
    zero_index: int,
    M_a: np.ndarray,
    tol: float,
) -> list[np.ndarray]:
    r"""Factor the same CMC cone directly in crystal coordinates.

    If ``V.T M_A V = I`` and ``eta_i=mu_i-1``, then

    .. math::

        p_A=M_A\left(\sqrt{\eta_+}v_+\pm\sqrt{-\eta_-}v_-\right),

    which is algebraically identical to the former whitened-basis formula.
    """
    idx = [i for i in range(3) if i != zero_index]
    i, j = idx
    qi, qj = float(vals[i]), float(vals[j])
    if qi * qj >= 0.0:
        return []
    if abs(qi) <= tol or abs(qj) <= tol:
        return []
    if qi > 0.0:
        ip, ineg = i, j
    else:
        ip, ineg = j, i

    p1 = M_a @ (
        np.sqrt(vals[ip]) * V[:, ip]
        + np.sqrt(-vals[ineg]) * V[:, ineg]
    )
    p2 = M_a @ (
        np.sqrt(vals[ip]) * V[:, ip]
        - np.sqrt(-vals[ineg]) * V[:, ineg]
    )
    return [normalize_plane(p1, M_a), normalize_plane(p2, M_a)]


def habit_planes_from_analysis(
    M_a: np.ndarray,
    analysis: CMCAnalysis,
    tol: float,
) -> list[np.ndarray]:
    """Construct exact CT habit planes from an already-computed CMC analysis."""
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if not analysis.exact_compatible or analysis.degeneracy_order == 3:
        return []
    M_a = _validated_metric(M_a, "M_a")
    vals = np.asarray(analysis.eigenvalues, float)
    V = analysis.eigenvectors_crystal
    if V is None:
        raise RuntimeError("CMC analysis does not contain generalized eigenvectors")
    V = np.asarray(V, float)
    zero = np.abs(vals) <= tol

    if analysis.degeneracy_order == 2:
        k = int(np.where(~zero)[0][0])
        # eta_k (v_k^T M_A u)^2 = 0, hence p_A is proportional to M_A v_k.
        return [normalize_plane(M_a @ V[:, k], M_a)]

    if analysis.zero_index is None:
        raise RuntimeError("first-order CMC analysis lacks a zero eigenvalue index")
    return _planes_from_generalized_eigensystem(
        vals, V, analysis.zero_index, M_a, tol
    )


def habit_planes_from_cmc(
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
    tol: float = 1e-8,
) -> list[np.ndarray]:
    """Exact CT A/M habit-plane covectors."""
    ana = analyze_cmc(M_a, M_m, correspondence, tol)
    return habit_planes_from_analysis(M_a, ana, tol)


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

    The nearest generalized CMC eigenvalue is set to zero *only* to generate a
    diagnostic candidate plane.  This is not an exact CT prediction, and the
    returned nonzero residual must be reported with the candidate.
    """
    from .numerics import DEFAULT_NUMERICAL_POLICY

    M_a, M_m, C = _validated_inputs(M_a, M_m, correspondence)
    spec = adaptive_metric_eigensystem(
        M_a,
        M_m,
        C,
        decision_tol=DEFAULT_NUMERICAL_POLICY.exact_eigenvalue,
    )
    vals = np.asarray(spec.mu - 1.0, float)
    V = np.asarray(spec.eigenvectors_crystal, float)
    iz = int(np.argmin(np.abs(vals)))
    other = [vals[i] for i in range(3) if i != iz]
    admiss = bool(other[0] * other[1] < 0.0)
    planes = (
        tuple(_planes_from_generalized_eigensystem(vals, V, iz, M_a, 0.0))
        if admiss
        else ()
    )
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

    The exact identity

    .. math::

        (C^T M_M C)^{-1}=C^{-1}M_M^{-1}C^{-T}

    permits solve/high-precision evaluation without changing the equation.
    """
    M_a, M_m, C = _validated_inputs(M_a, M_m, correspondence)
    return adaptive_smc(M_a, M_m, C)


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
    analysis = analyze_cmc(M_a, M_m, C, tol)
    return CTAMResult(
        cmc(M_a, M_m, C),
        normalized_cmc(M_a, M_m, C),
        analysis,
        tuple(habit_planes_from_analysis(M_a, analysis, tol)),
        approximate_cmc_habit_planes(M_a, M_m, C),
    )
