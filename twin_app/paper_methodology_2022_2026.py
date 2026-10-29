from __future__ import annotations

"""Independent, read-only 2026 Cayron CMC/SMC mathematical evidence.

References: Acta Materialia 316 (2026) 122399, Eqs. 31–42 and Fig. 6.
The *input* is only the parent/product metrics and the declared correspondence;
no imported CT habit solutions, BJ twins, benchmark outputs or fitted habits.

Important: these are single-variant A/M checks. Full A/M/M additionally needs
a valid transformation twin and Eq.42 or Eq.43 on that SAME physical branch.
"""
from typing import Any
import numpy as np


def _array(m: Any) -> np.ndarray:
    a=np.asarray(m,dtype=np.float64)
    if a.shape!=(3,3) or not np.all(np.isfinite(a)):
        raise ValueError("Expected a finite 3x3 matrix")
    return a


def _symmetric_spd(m: Any, name: str) -> np.ndarray:
    a=_array(m)
    if not np.allclose(a,a.T,rtol=0,atol=1e-10*max(1.,float(np.linalg.norm(a)))):
        raise ValueError(f"{name} is not a symmetric metric")
    if np.linalg.eigvalsh(a)[0] <= 0:
        raise ValueError(f"{name} is not positive definite")
    return a


def _unit_reciprocal(plane: np.ndarray, inv_ma: np.ndarray) -> np.ndarray:
    n=float(np.sqrt(plane@inv_ma@plane))
    if not np.isfinite(n) or n<=1.e-15:
        raise ValueError("Degenerate CMC plane")
    return plane/n


def cmc_smc_paper_audit(
    parent_metric: Any, product_metric: Any, correspondence_M_from_A: Any,
    *, tolerance: float=2.e-8
) -> dict:
    """Calculate Eqs.32/41, exact CMC degeneracy, and Eq.40 for all A/M habits.

    Tolerances apply to dimensionless generalized-metric eigenvalues and
    relative CMC eigenvalues, not raw dimensional Angstrom^2 eigenvalues.
    For a generic rank-two indefinite CMC, the null quadratic factors into
    two habit planes. A rank-one CMC has one plane. Rank zero has infinitely
    many choices, and is not misreported as two.
    """
    if not np.isfinite(tolerance) or tolerance<=0 or tolerance>.001:
        raise ValueError("Invalid dimensionless CMC tolerance")
    MA=_symmetric_spd(parent_metric,"Parent")
    MM=_symmetric_spd(product_metric,"Product")
    C=_array(correspondence_M_from_A)
    if abs(float(np.linalg.det(C)))<1.e-12:
        raise ValueError("Correspondence is singular")
    G=C.T@MM@C
    K=.5*((G-MA)+(G-MA).T)
    mi=np.linalg.inv(MA)
    ci=np.linalg.inv(C)
    S=mi-ci@np.linalg.inv(MM)@ci.T   # Eq.41, NOT the inverse of CMC

    # Dimensionless generalized eigenproblem; avoids treating near-zero
    # dimensional metric values as compatible solely due to their units.
    l,V=np.linalg.eigh(MA)
    W=V@np.diag(1./np.sqrt(l))@V.T
    H=W@G@W
    H=.5*(H+H.T)
    lam=np.sqrt(np.maximum(0.,np.linalg.eigvalsh(H)))
    defect=lam-1.
    vK, VK=np.linalg.eigh(K)
    # Inertia is congruence-invariant; numerical zero gate is taken from
    # the generalized pencil so micrometre and Å inputs agree.
    zero_count=int(np.sum(np.abs(defect)<=tolerance))
    signatures=[-1 if x < -tolerance else 1 if x>tolerance else 0 for x in defect]
    neg=signatures.count(-1)
    pos=signatures.count(1)
    compatible=(zero_count==3 or zero_count==2 or (zero_count==1 and neg==1 and pos==1))
    # For rank-two CMC with null middle eigenvalue: two planes.
    # For rank one: one plane. For rank zero: infinitely many.
    if zero_count==3:
        label="THIRD_ORDER_IDENTITY_METRIC"
        expected="nonunique_continuum"
    elif zero_count==2:
        label="SECOND_ORDER_SINGLE_PLANE"
        expected=1
    elif zero_count==1 and neg==1 and pos==1:
        label="FIRST_ORDER_DOUBLE_PLANE"
        expected=2
    elif zero_count==1:
        label="DEGENERATE_SEMIDEFINITE_QUADRATIC"
        expected="special_case_requires_check"
    else:
        label="NONDEGENERATE_CMC"
        expected=0

    # Avoid constructing spurious habit normals when eigenvalues are
    # extremely near the declared tolerance, where modes can exchange.
    planes=[]
    invma=np.linalg.inv(MA)
    if expected==2:
        jneg=int(np.argmin(vK));jpos=int(np.argmax(vK))
        if vK[jneg]>=0 or vK[jpos]<=0:
            raise ArithmeticError("Inertia/CMC eigenpair contradiction")
        for sign in (1.,-1.):
            p=(np.sqrt(vK[jpos])*VK[:,jpos]
               +sign*np.sqrt(-vK[jneg])*VK[:,jneg])
            p=_unit_reciprocal(p,invma)
            d=S@p
            # Distortion F = I + d p^T, with p unit as a reciprocal
            # covector. This describes one A/M IPS solution only.
            planes.append({
                "habit_plane_parent_reciprocal":p.tolist(),
                "IPS_shear_parent_direct":d.tolist(),
                "IPS_shear_metric_norm":float(np.sqrt(max(0.,d@MA@d))),
                "volume_change_p_dot_d":float(p@d),
                "shape_strain_parent_basis":(np.outer(d,p)).tolist(),
                "status":"CMC_FACTOR_AND_SMC_SOLUTION",
            })
    elif expected==1:
        j=int(np.argmax(np.abs(vK)))
        p=_unit_reciprocal(VK[:,j],invma)
        d=S@p
        planes.append({
            "habit_plane_parent_reciprocal":p.tolist(),
            "IPS_shear_parent_direct":d.tolist(),
            "IPS_shear_metric_norm":float(np.sqrt(max(0.,d@MA@d))),
            "volume_change_p_dot_d":float(p@d),
            "shape_strain_parent_basis":(np.outer(d,p)).tolist(),
            "status":"SECOND_ORDER_CMC_PLANE",
        })
    return {
        "method":"INDEPENDENT_CMC_SMC_FROM_METRICS_NOT_THEORY_OUTPUT",
        "source_doi":"10.1016/j.actamat.2026.122399",
        "equations":{"CMC":"32-39","SMC":"40-41","shear_shear":"42-43"},
        "correspondence_M_from_A":C.tolist(),
        "parent_metric":MA.tolist(),"product_metric":MM.tolist(),
        "pulled_back_product_metric":G.tolist(),
        "CMC_matrix":K.tolist(),"SMC_matrix":S.tolist(),
        "principal_stretches":lam.tolist(),
        "lambda2_minus_one":float(lam[1]-1.),
        "dimensionless_CMC_defects":defect.tolist(),
        "cmc_inertia":{"negative":neg,"near_zero":zero_count,"positive":pos},
        "degeneracy_order":zero_count,
        "degeneracy_class":label,
        "theoretical_habit_plane_count":expected,
        "computed_habit_plane_count":len(planes),
        "individual_AM_compatible":bool(compatible),
        "habit_planes":planes,
        "tolerance":float(tolerance),
        "full_AMM_supercompatibility_certified":False,
        "scientific_limit":(
            "This checks the A/M CMC/SMC part only. A valid M/M "
            "transformation twin, its correct direct/reciprocal frames "
            "and Eq.42/43 shear/shear must be independently verified. "
            "No CT/Ball-James matching or alloy synthesis is inferred."
        ),
    }


def audit_project_paper_equations(project: Any, transformation_id: str,
                                  *, tolerance:float=2.e-8) -> dict:
    t=project.transformation(transformation_id)
    pa=project.phase(t.parent_phase_id).lattice.metric()
    pm=project.phase(t.product_phase_id).lattice.metric()
    c=t.correspondence.C_M_from_A
    return cmc_smc_paper_audit(pa,pm,c,tolerance=tolerance)
