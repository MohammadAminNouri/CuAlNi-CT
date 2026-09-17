from __future__ import annotations

"""Analytical rank-one compatibility and Mallard-law twin solutions.

These routines form an independent nonlinear-elasticity branch.  They do not
call the CT twin formulas, which makes CT-vs-Ball-James comparison meaningful.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AnalyticalRankOneSolution:
    rotation: np.ndarray
    a: np.ndarray
    n: np.ndarray
    branch: int
    residual: float
    eigenvalues_C: np.ndarray


def _rank_one_relative_residual(D: np.ndarray) -> float:
    s=np.linalg.svd(D,compute_uv=False)
    return float(np.hypot(s[1],s[2])/max(s[0],1e-15))


def analytical_rank_one_connections(
    F1: np.ndarray, F2: np.ndarray, tol: float=1e-9
) -> list[AnalyticalRankOneSolution]:
    r"""Solve ``R F1 - F2 = a⊗n`` using the Ball-James compatibility proposition.

    Put A=F1 F2^{-1} and C=A^T A.  A rank-one connection exists iff the
    ordered eigenvalues L1<=L2<=L3 satisfy L1>0 and L2=1 (excluding the
    trivial C=I case).  The explicit formulas then give two branches.
    """
    F1=np.asarray(F1,float); F2=np.asarray(F2,float)
    if F1.shape!=(3,3) or F2.shape!=(3,3):
        raise ValueError("F1 and F2 must be 3x3")
    A=F1@np.linalg.inv(F2)
    C=A.T@A
    L,V=np.linalg.eigh(0.5*(C+C.T))
    order=np.argsort(L); L=L[order]; V=V[:,order]
    L1,L2,L3=map(float,L)
    if L1 <= 0 or abs(L2-1.0)>tol:
        return []
    if abs(L3-L1)<tol:
        return []
    e1,e3=V[:,0],V[:,2]
    out=[]
    den=np.sqrt(L3-L1)
    for kappa in (-1,1):
        # James-Hane Proposition 1 / Ball-James rank-one formula.
        b=(
            np.sqrt(L3*(1-L1))*e1
            + kappa*np.sqrt(max(0.0,L1*(L3-1)))*e3
        )/den
        m=(np.sqrt(L3)-np.sqrt(L1))/den * (
            -np.sqrt(max(0.0,1-L1))*e1
            + kappa*np.sqrt(max(0.0,L3-1))*e3
        )
        R=(np.eye(3)+np.outer(b,m))@np.linalg.inv(A)
        # Map from relative equation R A-I=b⊗m back to R F1-F2=a⊗n.
        a=b
        n=F2.T@m
        D=R@F1-F2
        resid=np.linalg.norm(D-np.outer(a,n),'fro')/max(np.linalg.norm(D,'fro'),1e-15)
        ortho=np.linalg.norm(R.T@R-np.eye(3),'fro')
        deterr=abs(np.linalg.det(R)-1.0)
        resid=max(float(resid),float(ortho),float(deterr))
        out.append(AnalyticalRankOneSolution(R,a,n,kappa,resid,L.copy()))
    return out


def single_variant_austenite_habit_solutions(U: np.ndarray,tol:float=1e-9)->list[AnalyticalRankOneSolution]:
    """Solve R U - I = b⊗m for a single martensite stretch variant."""
    return analytical_rank_one_connections(np.asarray(U,float),np.eye(3),tol=tol)


@dataclass(frozen=True)
class MallardTwinSolution:
    kind: str
    rotation: np.ndarray
    a: np.ndarray
    n: np.ndarray
    shear_magnitude: float
    residual: float


def mallard_law_twins(
    Ui: np.ndarray, Uj: np.ndarray, twofold_axis: np.ndarray, tol: float=1e-8
) -> list[MallardTwinSolution]:
    r"""Type-I and Type-II Mallard-law solutions for symmetry-related stretches.

    Assumes ``Uj = Q Ui Q`` with Q the parent 180° rotation about unit axis e.
    Implements the formulas reproduced by James & Hane, Eqs. (16)-(17).
    The scaling of ``a`` and ``n`` is non-unique; Type-II is normalized here so
    ``|n|=1``.
    """
    Ui=np.asarray(Ui,float); Uj=np.asarray(Uj,float)
    e=np.asarray(twofold_axis,float).reshape(3); e=e/np.linalg.norm(e)
    Q=-np.eye(3)+2*np.outer(e,e)
    symres=np.linalg.norm(Uj-Q@Ui@Q,'fro')
    if symres>tol:
        raise ValueError(f"Uj is not related to Ui by supplied 180° symmetry; residual={symres:g}")

    # Type I
    x=np.linalg.solve(Uj,e)
    x2=float(x@x)
    aI=2.0*(x/x2-Uj@e)
    nI=e.copy()
    RI=(-np.eye(3)+2.0*np.outer(x,x)/x2)@Q

    # Type II; first form a scale-free normal vector then normalize it.
    y=Uj@e; y2=float(y@y)
    n0=e-(Uj@Uj@e)/y2
    n0norm=np.linalg.norm(n0)
    if n0norm<tol:
        typeII=None
    else:
        rho=2.0*n0norm  # makes nII unit because nII=(2/rho)n0
        nII=(2.0/rho)*n0
        aII=rho*y
        RII=(-np.eye(3)+2.0*np.outer(y,y)/y2)@Q
        typeII=(RII,aII,nII)

    out=[]
    for kind,R,a,n in [("I",RI,aI,nI)] + ([] if typeII is None else [("II",*typeII)]):
        D=R@Ui-Uj
        resid=np.linalg.norm(D-np.outer(a,n),'fro')/max(np.linalg.norm(D,'fro'),1e-15)
        shear=float(np.linalg.norm(np.linalg.solve(Uj.T,n))*np.linalg.norm(a))
        out.append(MallardTwinSolution(kind,R,a,n,shear,float(resid)))
    return out
