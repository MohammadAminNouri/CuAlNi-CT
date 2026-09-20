from __future__ import annotations

"""Transformation-stretch calculations from the same crystallographic inputs used by CT."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

from .correspondence import Correspondence
from .ct import normalized_correspondence_metric


def positive_definite_sqrt(A: np.ndarray) -> np.ndarray:
    """Symmetric positive square root for a symmetric positive-definite matrix."""
    A = 0.5 * (np.asarray(A, float) + np.asarray(A, float).T)
    w, V = np.linalg.eigh(A)
    if np.min(w) <= 0:
        raise ValueError(f"Matrix is not positive definite; eigenvalues={w}")
    return (V * np.sqrt(w)) @ V.T


def stretch_from_metrics(
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
) -> np.ndarray:
    r"""Return U in the parent metric-whitened orthonormal basis.

    .. math::

        U^2=M_A^{-1/2}C^T M_M C M_A^{-1/2}.

    For a cubic parent ``M_A=a0^2 I`` this reduces to
    ``U^2 = C^T M_M C / a0^2``.

    This is a representation bridge, not a statement that correspondence,
    deformation and stretch are the same object.
    """
    return positive_definite_sqrt(
        normalized_correspondence_metric(M_a, M_m, correspondence)
    )


def principal_stretches(U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(0.5 * (np.asarray(U, float) + np.asarray(U, float).T))
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


@dataclass(frozen=True)
class MetricNativeStretchSpectrum:
    """Generalized-eigenvalue form of the stretch spectrum.

    The crystal-coordinate eigenvectors ``v_i`` solve

        C^T M_M C v_i = mu_i M_A v_i

    and are normalized such that ``V.T @ M_A @ V = I``.  The principal
    stretches are ``lambda_i = sqrt(mu_i)``.

    This is useful for arbitrary non-Cartesian parent cells because it works
    directly with the two metrics and the correspondence; no arbitrary
    Cartesian basis is required to define the spectrum.
    """

    mu: np.ndarray
    lambdas: np.ndarray
    eigenvectors_crystal: np.ndarray
    metric_orthonormality_residual: float
    eigen_equation_residual: float


def metric_native_stretch_spectrum(
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
) -> MetricNativeStretchSpectrum:
    r"""Solve the exact generalized metric eigenproblem.

    .. math::

        (C^T M_M C)v_i=\mu_i M_A v_i,
        \qquad \lambda_i=\sqrt{\mu_i}.

    For positive-definite parent/product metrics and an invertible
    correspondence, all ``mu_i`` must be strictly positive.
    """
    M_a = np.asarray(M_a, dtype=float).reshape(3, 3)
    M_m = np.asarray(M_m, dtype=float).reshape(3, 3)
    M_a = 0.5 * (M_a + M_a.T)
    M_m = 0.5 * (M_m + M_m.T)
    if np.min(np.linalg.eigvalsh(M_a)) <= 0:
        raise ValueError("Parent metric must be symmetric positive definite")
    if np.min(np.linalg.eigvalsh(M_m)) <= 0:
        raise ValueError("Martensite metric must be symmetric positive definite")

    C = np.asarray(correspondence.C_m_from_a, dtype=float).reshape(3, 3)
    if abs(float(np.linalg.det(C))) < 1e-14:
        raise ValueError("Correspondence must be invertible")

    G = 0.5 * (C.T @ M_m @ C + (C.T @ M_m @ C).T)
    mu, V = eigh(G, M_a)
    order = np.argsort(mu)
    mu = np.asarray(mu[order], dtype=float)
    V = np.asarray(V[:, order], dtype=float)
    if np.min(mu) <= 0:
        raise ValueError(f"Generalized stretch eigenvalues must be positive; mu={mu}")

    scale = max(float(np.linalg.norm(G)), 1.0)
    eigen_res = float(np.linalg.norm(G @ V - M_a @ V @ np.diag(mu)) / scale)
    metric_res = float(np.linalg.norm(V.T @ M_a @ V - np.eye(3)))

    return MetricNativeStretchSpectrum(
        mu=mu,
        lambdas=np.sqrt(mu),
        eigenvectors_crystal=V,
        metric_orthonormality_residual=metric_res,
        eigen_equation_residual=eigen_res,
    )


def correspondence_metric_identity_residual(
    U: np.ndarray,
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
) -> float:
    """Relative Frobenius residual of U^2 versus normalized pulled-back metric."""
    lhs = np.asarray(U, float).T @ np.asarray(U, float)
    rhs = normalized_correspondence_metric(M_a, M_m, correspondence)
    return float(np.linalg.norm(lhs - rhs) / max(np.linalg.norm(rhs), 1e-15))


def generate_stretch_variants(
    U: np.ndarray,
    parent_proper_rotations: list[np.ndarray],
    tol: float = 1e-10,
) -> list[np.ndarray]:
    """Generate distinct ``U_i = Q U Q^T`` using proper parent rotations."""
    out: list[np.ndarray] = []
    for Q in parent_proper_rotations:
        Q = np.asarray(Q, float)
        V = Q @ np.asarray(U, float) @ Q.T
        V = 0.5 * (V + V.T)
        if not any(np.linalg.norm(V - W, ord="fro") <= tol for W in out):
            out.append(V)
    return out


@dataclass(frozen=True)
class FamilyMatch:
    """Optimal one-to-one comparison of two unordered matrix families."""

    success: bool
    mapping: tuple[tuple[int, int], ...]
    residuals: np.ndarray
    maximum_residual: float
    rms_residual: float
    residual_matrix: np.ndarray
    tolerance: float


def optimal_match_matrix_family(
    derived: list[np.ndarray],
    reference: list[np.ndarray],
    tol: float = 1e-9,
) -> FamilyMatch:
    """Compare two matrix families with a globally optimal bijection."""
    if len(derived) != len(reference):
        R = np.array(
            [[np.linalg.norm(a - b, "fro") for b in reference] for a in derived],
            dtype=float,
        )
        return FamilyMatch(
            False,
            (),
            np.array([], dtype=float),
            float("inf"),
            float("inf"),
            R,
            float(tol),
        )

    R = np.array(
        [[np.linalg.norm(a - b, "fro") for b in reference] for a in derived],
        dtype=float,
    )
    row_ind, col_ind = linear_sum_assignment(R)
    residuals = R[row_ind, col_ind]
    mapping = tuple((int(i), int(j)) for i, j in zip(row_ind, col_ind, strict=True))
    maximum = float(np.max(residuals)) if len(residuals) else 0.0
    rms = float(np.sqrt(np.mean(residuals**2))) if len(residuals) else 0.0
    return FamilyMatch(
        bool(maximum <= tol),
        mapping,
        residuals,
        maximum,
        rms,
        R,
        float(tol),
    )


def match_matrix_family(
    derived: list[np.ndarray],
    reference: list[np.ndarray],
    tol: float = 1e-9,
) -> tuple[bool, np.ndarray]:
    """Backward-compatible family check.

    New scientific code should prefer :func:`optimal_match_matrix_family`,
    which additionally reports the one-to-one assignment and residuals.
    """
    result = optimal_match_matrix_family(derived, reference, tol=tol)
    return result.success, result.residual_matrix


@dataclass(frozen=True)
class StretchAnalysis:
    U: np.ndarray
    lambdas: np.ndarray
    eigenvectors: np.ndarray
    lambda2_residual: float


def analyze_stretch(
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
) -> StretchAnalysis:
    U = stretch_from_metrics(M_a, M_m, correspondence)
    lam, V = principal_stretches(U)
    return StretchAnalysis(U, lam, V, float(lam[1] - 1.0))
