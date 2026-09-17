from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import sympy as sp

from .correspondence import Correspondence
from .lattice import normalize_direct, normalize_plane, plane_to_unit_normal, metric_norm


@dataclass(frozen=True)
class CTTwin:
    kind: str
    parent_symmetry: np.ndarray
    plane_m: np.ndarray
    direction_m: np.ndarray
    shear: float
    plane_a: np.ndarray
    direction_a: np.ndarray


def _real_eigenvector(G: np.ndarray, eigenvalue: float) -> np.ndarray:
    vals, vecs = np.linalg.eig(G)
    i = int(np.argmin(np.abs(vals - eigenvalue)))
    v = np.real(vecs[:, i])
    return v / np.linalg.norm(v)


def type_i_from_parent_reflection(
    reflection_a: sp.Matrix,
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
    tol: float = 1e-10,
) -> CTTwin:
    """Cayron Type-I transformation twin from a parent reflection.

    Implements Acta Materialia 316 (2026), Eqs. 18-20 in the package convention.
    """
    G = np.array(reflection_a, dtype=float)
    if round(np.linalg.det(G)) != -1 or not np.isclose(np.trace(G), 1.0):
        raise ValueError("Input is not a simple reflection matrix.")
    p_a = _real_eigenvector(G.T, -1.0)  # cubic parent: plane covector/normal coincide numerically
    p_a = normalize_plane(p_a, M_a)

    C = np.array(correspondence.C_m_from_a, dtype=float)
    Ci = np.linalg.inv(C)
    p_m = np.linalg.inv(C).T @ p_a
    p_m = normalize_plane(p_m, M_m)
    n_m = plane_to_unit_normal(p_m, M_m)
    Cint = C @ G @ Ci

    s2 = float(np.trace(Cint.T @ M_m @ Cint @ np.linalg.inv(M_m)) - 3.0)
    if s2 < -tol:
        raise ValueError(f"Negative Type-I shear^2={s2}; check conventions/data.")
    shear = float(np.sqrt(max(0.0, s2)))
    a_m_raw = -(Cint + np.eye(3)) @ n_m
    if metric_norm(a_m_raw, M_m) < tol:
        raise ValueError("Type-I shear direction collapsed to zero; check symmetry/correspondence.")
    a_m = normalize_direct(a_m_raw, M_m)
    a_a = normalize_direct(Ci @ a_m, M_a)
    return CTTwin("I", G, p_m, a_m, shear, p_a, a_a)


def type_ii_from_parent_twofold(
    rotation_a: sp.Matrix,
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
    tol: float = 1e-10,
) -> CTTwin:
    """Cayron Type-II transformation twin from a parent 180-degree rotation.

    Implements Acta Materialia 316 (2026), Eqs. 22-25. Reciprocal action is C* = C^{-T}.
    """
    G = np.array(rotation_a, dtype=float)
    if round(np.linalg.det(G)) != 1 or not np.isclose(np.trace(G), -1.0):
        raise ValueError("Input is not a 180-degree proper rotation.")
    a_a = normalize_direct(_real_eigenvector(G, 1.0), M_a)

    C = np.array(correspondence.C_m_from_a, dtype=float)
    Ci = np.linalg.inv(C)
    a_m = normalize_direct(C @ a_a, M_m)
    Cint = C @ G @ Ci

    s2 = float(np.trace(Cint @ np.linalg.inv(M_m) @ Cint.T @ M_m) - 3.0)
    if s2 < -tol:
        raise ValueError(f"Negative Type-II shear^2={s2}; check conventions/data.")
    shear = float(np.sqrt(max(0.0, s2)))

    p_m = M_m @ a_m  # plane covector normal to the rational eta2 direction
    Cint_star = np.linalg.inv(Cint).T
    jp_m = -(Cint_star - np.eye(3)) @ p_m
    if np.linalg.norm(jp_m) < tol:
        raise ValueError("Type-II K2 plane collapsed to zero; check conventions/data.")
    jp_m = normalize_plane(jp_m, M_m)
    jp_a = normalize_plane(C.T @ jp_m, M_a)
    return CTTwin("II", G, jp_m, a_m, shear, jp_a, a_a)
