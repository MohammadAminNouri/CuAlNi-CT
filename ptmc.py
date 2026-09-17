from __future__ import annotations

"""Classical crystallographic-theory (PTMC) laminate compatibility.

The module starts from an independently obtained martensite twin relation
``Rhat Uhat = U + a⊗n`` and asks for which volume fractions f the average
martensite deformation ``Fbar = U + f a⊗n`` can meet austenite by a rank-one
interface.  It therefore provides a theory branch independent of Cayron CMC.
"""

from dataclasses import dataclass
import numpy as np

from .ball_james import analytical_rank_one_connections, AnalyticalRankOneSolution
from .rank_one import solve_rank_one_connection, RankOneSolution


@dataclass(frozen=True)
class PTMCSolution:
    volume_fraction: float
    average_deformation: np.ndarray
    habit_connections: tuple[AnalyticalRankOneSolution,...]
    middle_stretch_residual: float


def middle_singular_value(F:np.ndarray)->float:
    return float(np.sort(np.linalg.svd(np.asarray(F,float),compute_uv=False))[1])


def _g(U:np.ndarray,a:np.ndarray,n:np.ndarray,f:float)->float:
    F=U+f*np.outer(a,n)
    return float(np.linalg.det(F.T@F-np.eye(3)))


def ptmc_volume_fractions(U:np.ndarray,a:np.ndarray,n:np.ndarray,tol:float=1e-9)->list[float]:
    r"""Solve the crystallographic-theory volume-fraction condition without scanning.

    Under a valid twin relation, ``g(f)=det(Fbar^T Fbar-I)`` is at most quadratic.
    We reconstruct that quadratic from three exact numerical evaluations, verify
    it at extra points, and retain roots in [0,1] for which 1 is the *middle*
    singular value (not merely any singular value).
    """
    U=np.asarray(U,float);a=np.asarray(a,float).reshape(3);n=np.asarray(n,float).reshape(3)
    # g(f)=c0+c1 f+c2 f^2. values at 0,1/2,1.
    y0=_g(U,a,n,0.0); yh=_g(U,a,n,0.5); y1=_g(U,a,n,1.0)
    c0=y0
    c2=2.0*(y1+y0-2.0*yh)
    c1=y1-c0-c2
    # Verify quadratic assumption. This catches misuse with a pair that is not a twin relation.
    for x in (0.25,0.75):
        pred=c0+c1*x+c2*x*x
        if abs(pred-_g(U,a,n,x))>1e-7*max(1.0,abs(pred),abs(_g(U,a,n,x))):
            raise ValueError("PTMC determinant is not quadratic: supplied a,n likely do not define a valid twin relation")
    coeff=[c2,c1,c0] if abs(c2)>1e-14 else [c1,c0]
    roots=np.roots(coeff) if len(coeff)>1 else np.array([])
    out=[]
    for z in roots:
        if abs(z.imag)>1e-8: continue
        f=float(z.real)
        if -tol<=f<=1+tol:
            f=min(1.0,max(0.0,f))
            if abs(middle_singular_value(U+f*np.outer(a,n))-1.0)<=1e-6:
                if not any(abs(f-r)<1e-7 for r in out): out.append(f)
    # Include degenerate exact endpoints if polynomial solver loses them.
    for f in (0.0,1.0):
        if abs(middle_singular_value(U+f*np.outer(a,n))-1.0)<=tol and not any(abs(f-r)<1e-7 for r in out): out.append(f)
    return sorted(out)


def solve_ptmc_laminate(U:np.ndarray,twin_a:np.ndarray,twin_n:np.ndarray,tol:float=1e-9)->list[PTMCSolution]:
    out=[]
    for f in ptmc_volume_fractions(U,twin_a,twin_n,tol):
        Fbar=np.asarray(U,float)+f*np.outer(twin_a,twin_n)
        habits=tuple(analytical_rank_one_connections(Fbar,np.eye(3),tol=max(tol,1e-8)))
        out.append(PTMCSolution(f,Fbar,habits,abs(middle_singular_value(Fbar)-1.0)))
    return out


def solve_ptmc_laminate_numerical_crosscheck(U:np.ndarray,twin_a:np.ndarray,twin_n:np.ndarray,f:float)->RankOneSolution:
    """Independent optimizer-based rank-one check for a PTMC solution."""
    Fbar=np.asarray(U,float)+float(f)*np.outer(twin_a,twin_n)
    return solve_rank_one_connection(Fbar,np.eye(3))
