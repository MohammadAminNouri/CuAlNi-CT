from __future__ import annotations

"""Cayron Correspondence-Theory metric compatibility (CMC/SMC) tools."""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh

from .correspondence import Correspondence
from .lattice import metric_inv_sqrt, metric_sqrt, normalize_plane, plane_to_unit_normal, metric_norm


def cmc(M_a:np.ndarray,M_m:np.ndarray,correspondence:Correspondence)->np.ndarray:
    r"""Dimensional CMC = C^T M_M C - M_A (Cayron 2026 Eq. 32)."""
    C=np.array(correspondence.C_m_from_a,dtype=float)
    A=C.T@np.asarray(M_m,float)@C-np.asarray(M_a,float)
    return 0.5*(A+A.T)


def normalized_correspondence_metric(M_a:np.ndarray,M_m:np.ndarray,correspondence:Correspondence)->np.ndarray:
    r"""Dimensionless parent-whitened pulled-back daughter metric.

    This normalization is a project comparison utility, not a replacement for
    Cayron's dimensional crystallographic formulation.
    """
    W=metric_inv_sqrt(np.asarray(M_a,float)); C=np.array(correspondence.C_m_from_a,float)
    G=W@(C.T@np.asarray(M_m,float)@C)@W
    return 0.5*(G+G.T)


def normalized_cmc(M_a:np.ndarray,M_m:np.ndarray,correspondence:Correspondence)->np.ndarray:
    return normalized_correspondence_metric(M_a,M_m,correspondence)-np.eye(3)


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
    signature_after_nearest_zero: tuple[int,int]


def analyze_cmc(M_a:np.ndarray,M_m:np.ndarray,correspondence:Correspondence,tol:float=1e-8)->CMCAnalysis:
    """Analyze Cayron's degeneracy condition in a metric-orthonormal parent basis."""
    D=normalized_cmc(M_a,M_m,correspondence); vals,Q=eigh(0.5*(D+D.T))
    zero=np.abs(vals)<=tol; nullity=int(np.sum(zero))
    nearest=int(np.argmin(np.abs(vals)))
    nz_near=[vals[i] for i in range(3) if i!=nearest]
    sig=(sum(x>tol for x in nz_near),sum(x<-tol for x in nz_near))
    if nullity==3:
        return CMCAnalysis(vals,Q,True,3,"third-order degeneracy: pulled metrics identical",None,nearest,float(abs(vals[nearest])),sig)
    if nullity==2:
        return CMCAnalysis(vals,Q,True,2,"second-order degeneracy: one compatible plane",None,nearest,float(abs(vals[nearest])),sig)
    if nullity==1:
        iz=int(np.where(zero)[0][0]); others=[vals[i] for i in range(3) if i!=iz]
        ok=others[0]*others[1] <= tol
        return CMCAnalysis(vals,Q,bool(ok),1 if ok else 0,
            "first-order degeneracy: two compatible planes" if ok else "one zero eigenvalue but incompatible signature",
            iz,nearest,float(abs(vals[nearest])),sig)
    return CMCAnalysis(vals,Q,False,0,"no exact CMC degeneracy",None,nearest,float(abs(vals[nearest])),sig)


def _planes_from_eigensystem(vals:np.ndarray,Q:np.ndarray,zero_index:int,M_a:np.ndarray,tol:float)->list[np.ndarray]:
    idx=[i for i in range(3) if i!=zero_index]; i,j=idx; qi,qj=vals[i],vals[j]
    if qi*qj>tol: return []
    if abs(qi)<=tol or abs(qj)<=tol:
        # second-order handled elsewhere
        return []
    if qi>0: ip,ineg=i,j
    else: ip,ineg=j,i
    # Plane factors of q_pos X^2 - |q_neg| Z^2.
    p1h=np.sqrt(vals[ip])*Q[:,ip]+np.sqrt(-vals[ineg])*Q[:,ineg]
    p2h=np.sqrt(vals[ip])*Q[:,ip]-np.sqrt(-vals[ineg])*Q[:,ineg]
    S=metric_sqrt(M_a)
    return [normalize_plane(S@p1h,M_a),normalize_plane(S@p2h,M_a)]


def habit_planes_from_cmc(M_a:np.ndarray,M_m:np.ndarray,correspondence:Correspondence,tol:float=1e-8)->list[np.ndarray]:
    """Exact CT A/M habit-plane covectors. Empty if exact degeneracy is absent."""
    ana=analyze_cmc(M_a,M_m,correspondence,tol)
    if not ana.exact_compatible or ana.degeneracy_order==3: return []
    vals,Q=ana.eigenvalues,ana.eigenvectors_whitened; zero=np.abs(vals)<=tol; S=metric_sqrt(M_a)
    if ana.degeneracy_order==2:
        # q_k X_k^2=0 -> plane X_k=0; its covector is eigenvector of nonzero q.
        k=int(np.where(~zero)[0][0])
        return [normalize_plane(S@Q[:,k],M_a)]
    assert ana.zero_index is not None
    return _planes_from_eigensystem(vals,Q,ana.zero_index,M_a,tol)


@dataclass(frozen=True)
class ApproximateCMCResult:
    residual: float
    candidate_planes: tuple[np.ndarray,...]
    admissible_signature: bool
    explanation: str


def approximate_cmc_habit_planes(M_a:np.ndarray,M_m:np.ndarray,correspondence:Correspondence)->ApproximateCMCResult:
    """Nearest-zero diagnostic for measured alloys that are close but not exactly compatible.

    We set the eigenvalue of normalized CMC having smallest absolute magnitude
    to zero *only to generate a diagnostic candidate plane*.  This is NOT an
    exact CT prediction and the returned residual |q_nearest| must always be
    reported with it.
    """
    D=normalized_cmc(M_a,M_m,correspondence); vals,Q=eigh(D)
    iz=int(np.argmin(np.abs(vals))); other=[vals[i] for i in range(3) if i!=iz]
    admiss=bool(other[0]*other[1]<0)
    planes=tuple(_planes_from_eigensystem(vals,Q,iz,M_a,0.0)) if admiss else ()
    return ApproximateCMCResult(float(abs(vals[iz])),planes,admiss,
        "diagnostic projection to nearest first-order CMC degeneracy; not an exact compatibility solution")


def smc(M_a:np.ndarray,M_m:np.ndarray,correspondence:Correspondence)->np.ndarray:
    r"""SMC = M_A^-1 - C^-1 M_M^-1 C^-T (Cayron 2026 Eq. 41)."""
    C=np.array(correspondence.C_m_from_a,float); Ci=np.linalg.inv(C)
    S=np.linalg.inv(M_a)-Ci@np.linalg.inv(M_m)@Ci.T
    return 0.5*(S+S.T)


def ips_shear_from_habit_plane(p_a:np.ndarray,M_a:np.ndarray,M_m:np.ndarray,correspondence:Correspondence)->np.ndarray:
    return smc(M_a,M_m,correspondence)@normalize_plane(p_a,M_a)


def ct_supercompatibility_vector(
    habit_plane_a:np.ndarray,d_a:np.ndarray,twin_plane_a:np.ndarray,twin_direction_a:np.ndarray,twin_shear:float,M_a:np.ndarray
)->np.ndarray:
    m=normalize_plane(habit_plane_a,M_a); n=plane_to_unit_normal(twin_plane_a,M_a)
    ad=np.asarray(twin_direction_a,float).reshape(3); ad=ad/metric_norm(ad,M_a)
    a=float(twin_shear)*ad
    return 2.0*float(m@n)*np.asarray(d_a,float).reshape(3)-a


def ct_supercompatibility_residual(
    habit_plane_a:np.ndarray,d_a:np.ndarray,twin_plane_a:np.ndarray,twin_direction_a:np.ndarray,twin_shear:float,M_a:np.ndarray
)->float:
    """Dimensionless Cayron shear/shear mismatch ε, zero at exact A/M/M compatibility."""
    if abs(float(twin_shear))<1e-15: raise ValueError("Twin shear must be nonzero")
    r=ct_supercompatibility_vector(habit_plane_a,d_a,twin_plane_a,twin_direction_a,twin_shear,M_a)
    return metric_norm(r,M_a)/abs(float(twin_shear))


@dataclass(frozen=True)
class CTAMResult:
    cmc_dimensional: np.ndarray
    cmc_normalized: np.ndarray
    analysis: CMCAnalysis
    exact_habit_planes: tuple[np.ndarray,...]
    approximate: ApproximateCMCResult


def analyze_austenite_martensite(M_a:np.ndarray,M_m:np.ndarray,C:Correspondence,tol:float=1e-8)->CTAMResult:
    return CTAMResult(cmc(M_a,M_m,C),normalized_cmc(M_a,M_m,C),analyze_cmc(M_a,M_m,C,tol),
                      tuple(habit_planes_from_cmc(M_a,M_m,C,tol)), approximate_cmc_habit_planes(M_a,M_m,C))
