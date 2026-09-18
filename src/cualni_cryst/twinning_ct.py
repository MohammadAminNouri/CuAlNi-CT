from __future__ import annotations

"""Cayron CT transformation-twin calculations, independent of stretch tensors."""

from dataclasses import dataclass

import numpy as np
import sympy as sp

from .correspondence import Correspondence
from .lattice import (
    metric_norm,
    normalize_direct,
    normalize_plane,
    plane_to_unit_normal,
)
from .symmetry import classify_symmetry


@dataclass(frozen=True)
class CTTwin:
    kind: str
    parent_symmetry: sp.Matrix
    plane_m: np.ndarray          # K1 for I, K2 for II
    direction_m: np.ndarray      # eta1 for I, eta2 for II
    shear: float
    plane_a: np.ndarray
    direction_a: np.ndarray
    intercorrespondence: np.ndarray
    rational_element: str
    equation_note: str


def _eigvec(G:np.ndarray,eigenvalue:float)->np.ndarray:
    vals,V=np.linalg.eig(G);i=int(np.argmin(np.abs(vals-eigenvalue)));v=np.real(V[:,i]);
    nz=np.linalg.norm(v)
    if nz<1e-14: raise ValueError("Failed to extract symmetry eigenvector")
    return v/nz


def type_i_from_parent_reflection(reflection_a:sp.Matrix,M_a:np.ndarray,M_m:np.ndarray,correspondence:Correspondence,tol:float=1e-10)->CTTwin:
    r"""Cayron 2026 Eqs. (18)-(21), parent reflection -> Type-I twin."""
    G=np.array(reflection_a,float); info=classify_symmetry(reflection_a)
    if info.kind!="reflection": raise ValueError("Parent operation is not a reflection")
    p_a=normalize_plane(_eigvec(G.T,-1.0),M_a)
    C=np.array(correspondence.C_m_from_a,float);Ci=np.linalg.inv(C)
    p_m=normalize_plane(np.linalg.inv(C).T@p_a,M_m)
    n_m=plane_to_unit_normal(p_m,M_m)
    Cint=C@G@Ci
    s2=float(np.trace(Cint.T@M_m@Cint@np.linalg.inv(M_m))-3.0)
    if s2 < -tol: raise ValueError(f"Negative CT Type-I shear^2 {s2}")
    shear=float(np.sqrt(max(0.0,s2)))
    avec=-(Cint+np.eye(3))@n_m
    if metric_norm(avec,M_m)<tol: raise ValueError("Collapsed CT Type-I shear vector")
    eta1=normalize_direct(avec,M_m)
    eta1_a=normalize_direct(Ci@eta1,M_a)
    return CTTwin("I",sp.Matrix(reflection_a),p_m,eta1,shear,p_a,eta1_a,Cint,
                  "K1 is rational/correspondence-derived",
                  "Cayron 2026 Eqs. 18-21; shear and eta1 depend on martensite metric")


def type_ii_from_parent_twofold(rotation_a:sp.Matrix,M_a:np.ndarray,M_m:np.ndarray,correspondence:Correspondence,tol:float=1e-10)->CTTwin:
    r"""Cayron 2026 Eqs. (22)-(25), parent 180° rotation -> Type-II twin."""
    G=np.array(rotation_a,float);info=classify_symmetry(rotation_a)
    if info.kind!="rotation" or info.order!=2: raise ValueError("Parent operation is not a proper twofold rotation")
    a_a=normalize_direct(_eigvec(G,1.0),M_a)
    C=np.array(correspondence.C_m_from_a,float);Ci=np.linalg.inv(C)
    eta2=normalize_direct(C@a_a,M_m); Cint=C@G@Ci
    s2=float(np.trace(Cint@np.linalg.inv(M_m)@Cint.T@M_m)-3.0)
    if s2 < -tol: raise ValueError(f"Negative CT Type-II shear^2 {s2}")
    shear=float(np.sqrt(max(0.0,s2)))
    p_m=M_m@eta2; Cstar=np.linalg.inv(Cint).T
    K2=-(Cstar-np.eye(3))@p_m
    if np.linalg.norm(K2)<tol: raise ValueError("Collapsed CT Type-II K2")
    K2=normalize_plane(K2,M_m); K2_a=normalize_plane(C.T@K2,M_a)
    return CTTwin("II",sp.Matrix(rotation_a),K2,eta2,shear,K2_a,a_a,Cint,
                  "eta2 is rational/correspondence-derived",
                  "Cayron 2026 Eqs. 22-25; shear and K2 depend on martensite metric")


def twins_from_operator(operator:list[sp.Matrix],M_a:np.ndarray,M_m:np.ndarray,correspondence:Correspondence)->list[CTTwin]:
    """Calculate one CT Type-I/II solution per relevant parent symmetry in a double coset.

    Symmetry-equivalent duplicates are intentionally retained here; downstream
    reporting can group them by plane/direction. This preserves the exact parent
    symmetry provenance of every prediction.
    """
    out=[]
    for g in operator:
        info=classify_symmetry(g)
        try:
            if info.kind=="reflection":
                tw=type_i_from_parent_reflection(g,M_a,M_m,correspondence)
            elif info.kind=="rotation" and info.order==2:
                tw=type_ii_from_parent_twofold(g,M_a,M_m,correspondence)
            else:
                continue
        except ValueError as exc:
            # Symmetries inside the correspondence subgroup may map the reference
            # variant onto itself.  Their intercorrespondence has zero shear, so
            # they are not M/M twin boundaries and must not be reported as twins.
            if "Collapsed CT" in str(exc):
                continue
            raise
        if tw.shear > 1e-12:
            out.append(tw)
    return out
