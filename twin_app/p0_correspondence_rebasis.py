"""Exact passive crystal-basis transformations (NOT physical transformations).

C sends parent direct coordinates into daughter direct coordinates, per Cayron
(Acta Cryst A75, 2019, 411-437; corrigenda p777).
For daughter basis B'_m = B_m P, P ∈ GL(3,Z), det P = ±1:
    G'_m = P.T G_m P ; C' = P^-1 C ; H'_m = P^-1 H_m P.
Directions u'_m=P^-1 u_m; plane covectors k'_m=P.T k_m.
The physical transformation tensor C.T G_m C is unchanged exactly.

No published crystallographic correspondence or orientation is inferred.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping
import numpy as np
import sympy as sp


@dataclass(frozen=True)
class BasisRebaseAudit:
    metric_residual: float
    parent_frame_deformation_metric_residual: float
    plane_direction_duality_residual: float
    determinant: int


def _exact_P(P: Any) -> tuple[sp.Matrix, sp.Matrix]:
    matrix = sp.Matrix(P)
    if matrix.shape != (3,3):
        raise ValueError("Unimodular basis matrix must be 3x3")
    for value in matrix:
        if value.is_Integer is not True:
            raise ValueError("Passive lattice rebasing requires INTEGER P")
    det = matrix.det()
    if det not in (sp.Integer(1),sp.Integer(-1)):
        raise ValueError("P must be in GL(3,Z), det=±1")
    return matrix, matrix.inv()


def rebase_metric_correspondence(
    G_m: Any, C_m_from_a: Any, P: Any,
    product_symmetry: tuple[Any, ...],
) -> tuple[np.ndarray, sp.Matrix, tuple[sp.Matrix, ...], BasisRebaseAudit]:
    """Passive re-basing with exact C & H and verified physical metric identity."""
    p, pi = _exact_P(P)
    metric = np.asarray(G_m,dtype=float)
    if metric.shape!=(3,3) or not np.isfinite(metric).all():
        raise ValueError("Product metric must be a finite 3x3 matrix")
    if np.linalg.eigvalsh(metric)[0] <= 0 or not np.allclose(metric,metric.T,atol=1e-10):
        raise ValueError("Product metric must be positive definite and symmetric")
    c = sp.Matrix(C_m_from_a)
    if c.shape!=(3,3) or c.det()==0:
        raise ValueError("Correspondence must be a nonsingular 3x3 matrix")
    pf=np.asarray(p,dtype=float)
    gmnew=pf.T@metric@pf
    cnew=pi*c
    ops=tuple(pi*sp.Matrix(g)*p for g in product_symmetry)
    if not ops:
        raise ValueError("Product symmetry list cannot be empty")
    for k,g in enumerate(product_symmetry):
        gf=np.asarray(g,dtype=float)
        if not np.allclose(gf.T@metric@gf,metric,rtol=1e-9,atol=1e-9):
            raise ValueError(f"Original product symmetry #{k} violates its metric")
    for k,g in enumerate(ops):
        gf=np.asarray(g,dtype=float)
        if not np.allclose(gf.T@gmnew@gf,gmnew,rtol=1e-9,atol=1e-9):
            raise AssertionError(f"Rebased product symmetry #{k} violates its metric")
    orig=np.asarray(c,dtype=float).T@metric@np.asarray(c,dtype=float)
    new=np.asarray(cnew,dtype=float).T@gmnew@np.asarray(cnew,dtype=float)
    residual=float(np.linalg.norm(orig-new,'fro')/max(np.linalg.norm(orig,'fro'),1e-15))
    if residual>1e-10:
        raise AssertionError(f"Passive re-basing changed physical stretch metric: {residual:.3e}")
    u=sp.Matrix([2,3,5]); k=sp.Matrix([7,11,13])
    identity=sp.simplify((p.T*k).dot(pi*u)-k.dot(u))
    if identity!=0:raise AssertionError("Plane/direction duality violated")
    return gmnew,cnew,ops,BasisRebaseAudit(residual,residual,float(identity),int(p.det()))


def _cell_from_metric(G: np.ndarray, unit: str) -> dict[str, Any]:
    lengths=np.sqrt(np.diag(G))
    if np.min(lengths)<=0:raise ValueError("Invalid metric diagonal")
    a,b,c=[float(v) for v in lengths]
    def angle(i:int,j:int)->float:
        cosine=np.clip(G[i,j]/(lengths[i]*lengths[j]),-1.,1.)
        return float(np.degrees(np.arccos(cosine)))
    return {"a":a,"b":b,"c":c,"alpha_deg":angle(1,2),
            "beta_deg":angle(0,2),"gamma_deg":angle(0,1),"length_unit":unit}


def rebase_product_project_payload(
    payload:Mapping[str,Any], transformation_id:str, P:Any,
) -> tuple[dict[str,Any],BasisRebaseAudit]:
    """Rebase a standard ProjectState payload without inventing crystal symmetries.

    Existing explicit/point-group symmetry matrices are obtained from its
    VALIDATED source phase and conjugated exactly; the source remains unchanged.
    """
    from cualni_cryst.project_io import project_from_dict
    original=deepcopy(dict(payload))
    loaded=project_from_dict(original,source="passive_rebasing")
    loaded.project.validate().assert_passed()
    tr=loaded.project.transformation(transformation_id)
    product=loaded.project.phase(tr.product_phase_id)
    symmetry=tuple(sp.Matrix(g) for g in product.symmetry_matrices())
    gmnew,cnew,gnew,audit=rebase_metric_correspondence(
        product.lattice.metric(),tr.correspondence.C_M_from_A,P,symmetry,
    )
    # Fail closed if payload schema or phase IDs deviate from the actual project.
    phases=original.get('phases')
    transformations=original.get('transformations')
    if not isinstance(phases,list) or not isinstance(transformations,list):
        raise ValueError("Expected standard project payload phases/transformations lists")
    matches=[ph for ph in phases if ph.get('phase_id')==tr.product_phase_id]
    t_matches=[t for t in transformations if t.get('transformation_id')==transformation_id]
    if len(matches)!=1 or len(t_matches)!=1:
        raise ValueError("Expected exactly one product phase and transformation")
    product_record=matches[0]
    unit=str(product_record['cell'].get('length_unit','angstrom'))
    product_record['cell'].update(_cell_from_metric(gmnew,unit))
    product_record['symmetry_matrices']=[np.asarray(op,dtype=float).tolist() for op in gnew]
    t_matches[0]['correspondence_M_from_A']=[[str(cnew[i,j]) for j in range(3)] for i in range(3)]
    # CRUCIAL: symmetry_matrices are crystal-coordinate matrices in the NEW basis.
    project_from_dict(original,source='rebasing_selftest').project.validate().assert_passed()
    return original,audit
