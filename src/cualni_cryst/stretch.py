from __future__ import annotations

"""Transformation-stretch calculations from exactly the same inputs used by CT."""

from dataclasses import dataclass

import numpy as np

from .correspondence import Correspondence
from .ct import normalized_correspondence_metric


def positive_definite_sqrt(A: np.ndarray) -> np.ndarray:
    """Symmetric positive square root for a symmetric positive-definite matrix."""
    A = 0.5*(np.asarray(A,float)+np.asarray(A,float).T)
    w, V = np.linalg.eigh(A)
    if np.min(w) <= 0:
        raise ValueError(f"Matrix is not positive definite; eigenvalues={w}")
    return (V*np.sqrt(w)) @ V.T


def stretch_from_metrics(M_a: np.ndarray, M_m: np.ndarray, correspondence: Correspondence) -> np.ndarray:
    r"""Return U in the parent metric-whitened orthonormal basis.

        U^2 = M_A^{-1/2} C^T M_M C M_A^{-1/2}.

    For a cubic parent M_A=a0^2 I this is Cayron 2026 Eq. (50)/(51)
    specialized to ``U^2 = C^T M_M C / a0^2``.
    """
    return positive_definite_sqrt(normalized_correspondence_metric(M_a, M_m, correspondence))


def principal_stretches(U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(0.5*(np.asarray(U,float)+np.asarray(U,float).T))
    order = np.argsort(vals)
    return vals[order], vecs[:,order]


def correspondence_metric_identity_residual(
    U: np.ndarray, M_a: np.ndarray, M_m: np.ndarray, correspondence: Correspondence
) -> float:
    """Relative Frobenius residual of U^2 versus the normalized pulled-back metric."""
    lhs = np.asarray(U,float).T @ np.asarray(U,float)
    rhs = normalized_correspondence_metric(M_a,M_m,correspondence)
    return float(np.linalg.norm(lhs-rhs)/max(np.linalg.norm(rhs),1e-15))


def generate_stretch_variants(
    U: np.ndarray, parent_proper_rotations: list[np.ndarray], tol: float=1e-10
) -> list[np.ndarray]:
    """Generate distinct U_i = Q U Q^T using proper parent rotations."""
    out: list[np.ndarray] = []
    for Q in parent_proper_rotations:
        Q=np.asarray(Q,float)
        V=Q@np.asarray(U,float)@Q.T
        V=0.5*(V+V.T)
        if not any(np.linalg.norm(V-W,ord='fro') <= tol for W in out):
            out.append(V)
    return out


def match_matrix_family(
    derived: list[np.ndarray], reference: list[np.ndarray], tol: float=1e-9
) -> tuple[bool, np.ndarray]:
    """Bijective nearest-family comparison; returns success and residual matrix."""
    if len(derived)!=len(reference):
        return False, np.empty((len(derived),len(reference)))
    R=np.array([[np.linalg.norm(a-b,'fro') for b in reference] for a in derived])
    # small sets: greedy is enough only for report; verify each ref and each derived has a close match
    ok = bool(np.all(np.min(R,axis=1)<tol) and np.all(np.min(R,axis=0)<tol))
    return ok,R


@dataclass(frozen=True)
class StretchAnalysis:
    U: np.ndarray
    lambdas: np.ndarray
    eigenvectors: np.ndarray
    lambda2_residual: float


def analyze_stretch(M_a: np.ndarray,M_m: np.ndarray,correspondence: Correspondence)->StretchAnalysis:
    U=stretch_from_metrics(M_a,M_m,correspondence)
    lam,V=principal_stretches(U)
    return StretchAnalysis(U,lam,V,float(lam[1]-1.0))
