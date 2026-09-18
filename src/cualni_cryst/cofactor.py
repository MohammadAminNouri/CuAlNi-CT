from __future__ import annotations

"""Cofactor conditions of Chen, Srivastava, Dabade & James (2013)."""

from dataclasses import dataclass

import numpy as np

from .stretch import principal_stretches


def cofactor_matrix(A: np.ndarray) -> np.ndarray:
    """Classical 3x3 cofactor matrix, well-defined even when A is singular."""
    A=np.asarray(A,float)
    c1,c2,c3=A[:,0],A[:,1],A[:,2]
    return np.column_stack((np.cross(c2,c3),np.cross(c3,c1),np.cross(c1,c2)))


@dataclass(frozen=True)
class CofactorResult:
    lambdas: np.ndarray
    middle_eigenvector: np.ndarray
    cc1_residual: float
    cc2_residual: float
    cc2_simplified: float
    cc3_margin: float
    cc1_satisfied: bool
    cc2_satisfied: bool
    cc3_satisfied: bool
    satisfied: bool


def evaluate_cofactor_conditions(
    U: np.ndarray, a: np.ndarray, n: np.ndarray, tol: float=1e-7
) -> CofactorResult:
    r"""Evaluate CC1-CC3 for a specified twin system.

    Chen et al. 2013, Theorem 2:

      CC1: lambda_2 = 1
      CC2: a · U cof(U^2-I) n = 0
      CC3: tr(U^2) - det(U^2) - |a|^2 |n|^2 / 4 - 2 >= 0

    Note the MINUS sign in front of det(U^2).  A previous draft of this
    repository had that sign wrong; a regression test prevents recurrence.
    """
    U=np.asarray(U,float); a=np.asarray(a,float).reshape(3); n=np.asarray(n,float).reshape(3)
    lam,V=principal_stretches(U); v2=V[:,1]
    cc1=float(lam[1]-1.0)
    A=U@U-np.eye(3)
    cc2=float(a @ U @ cofactor_matrix(A) @ n)
    cc2p=float((a@v2)*(n@v2))
    U2=U@U
    cc3=float(np.trace(U2)-np.linalg.det(U2)-0.25*(a@a)*(n@n)-2.0)
    c1=abs(cc1)<=tol; c2=abs(cc2)<=tol; c3=cc3>=-tol
    return CofactorResult(lam,v2,cc1,cc2,cc2p,cc3,c1,c2,c3,bool(c1 and c2 and c3))


def laminate_middle_stretch_residual(U: np.ndarray,a:np.ndarray,n:np.ndarray,f:float)->float:
    """Middle singular-value residual of U+f a⊗n used to verify all-f compatibility."""
    F=np.asarray(U,float)+float(f)*np.outer(np.asarray(a,float),np.asarray(n,float))
    s=np.sort(np.linalg.svd(F,compute_uv=False))
    return float(s[1]-1.0)


def verify_all_volume_fractions(U:np.ndarray,a:np.ndarray,n:np.ndarray,samples:int=101)->float:
    """Maximum |middle singular value - 1| over f∈[0,1]."""
    return max(abs(laminate_middle_stretch_residual(U,a,n,f)) for f in np.linspace(0,1,samples))
