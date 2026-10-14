"""P1 solution-orbits, measured EBSD/TEM comparison and evidence hierarchy.

Coordinate-frame correctness is mandatory: a Miller normal is reciprocal, not
an ordinary direction. A geometrical agreement is never certification that an
observation is authentic, representative or uniquely assigns a variant.
"""
from __future__ import annotations
from typing import Any,Mapping,Sequence
from dataclasses import dataclass
import numpy as np

class EvidenceError(ValueError):pass


def _unit(v):
    x=np.asarray(v,float)
    if x.shape!=(3,) or not np.isfinite(x).all() or np.linalg.norm(x)<1e-14:
        raise EvidenceError("Finite nonzero 3-vector required")
    return x/np.linalg.norm(x)


def _metric_root(G):
    G=np.asarray(G,float)
    if G.shape!=(3,3) or not np.isfinite(G).all() or np.linalg.norm(G-G.T,'fro')>1e-8*np.linalg.norm(G):
        raise EvidenceError("Invalid metric")
    eig,V=np.linalg.eigh(G)
    if eig[0]<=0:raise EvidenceError("Metric must be positive definite")
    return (V*np.sqrt(eig))@V.T


def direct_to_cartesian(v,G):return _metric_root(G)@np.asarray(v,float)

def reciprocal_normal_to_cartesian(h,G):return np.linalg.solve(_metric_root(G).T,np.asarray(h,float))


def proper_cartesian_group(parent_metric,lattice_operations,*,tol=1e-8):
    B=_metric_root(parent_metric);Bi=np.linalg.inv(B)
    group=[]
    for k,g in enumerate(lattice_operations):
        g=np.asarray(g,float)
        if g.shape!=(3,3) or not np.isfinite(g).all():raise EvidenceError(f"Invalid parent symmetry #{k}")
        if np.linalg.norm(g.T@parent_metric@g-parent_metric,'fro')>tol*max(1.,np.linalg.norm(parent_metric,'fro')):
            raise EvidenceError(f"Non-metric symmetry #{k}")
        q=B@g@Bi
        if np.linalg.norm(q.T@q-np.eye(3),'fro')>tol:raise EvidenceError("Nonorthogonal Cartesian group operator")
        if abs(np.linalg.det(q)-1)<=tol and not any(np.linalg.norm(q-p,'fro')<=tol for p in group):group.append(q)
    if not group or not any(np.linalg.norm(q-np.eye(3))<=tol for q in group):
        raise EvidenceError("Proper symmetry group lacks identity")
    for q in group:
        for p in group:
            if not any(np.linalg.norm(q@p-r,'fro')<=10*tol for r in group):
                raise EvidenceError("Parent proper symmetry group is not closed")
    return group


def _orbit_labels(tensors, group, *,rtol=1e-7):
    x=[np.asarray(t,float) for t in tensors]
    for t in x:
        if t.shape!=(3,3) or not np.isfinite(t).all():raise EvidenceError("Invalid rank-one tensor")
    parent=list(range(len(x)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            tol=rtol*max(1.,np.linalg.norm(x[i]),np.linalg.norm(x[j]))
            if any(np.linalg.norm(q@x[i]@q.T-x[j])<=tol for q in group):
                a,b=root(i),root(j)
                if a!=b:parent[max(a,b)]=min(a,b)
    groups={}
    for i in range(len(x)):groups.setdefault(root(i),[]).append(i)
    return list(groups.values())


def report_orbits(report,parent_metric,parent_symmetry_ops):
    """Groups macroscopic twin D=a⊗n and habit H=b⊗m separately.

    This does NOT claim that physically different internal twins are the same
    microstructure. Branch identity, λ and variant pair are always retained.
    """
    group=proper_cartesian_group(np.asarray(parent_metric,float),parent_symmetry_ops)
    twins=[];habits=[];tids=[];hids=[]
    for f in report.families:
        for p in f.pair_records:
            for c in p.constructions:
                twins.append(np.outer(c.a_parent_cartesian,c.n_parent_cartesian))
                tids.append((p.pair_id,c.branch,c.classification))
                for h in c.habit_solutions:
                    habits.append(np.outer(h.shape_vector_parent_cartesian,h.habit_normal_parent_cartesian))
                    hids.append((p.pair_id,c.branch,h.habit_branch,float(h.other_variant_volume_fraction)))
    tg=_orbit_labels(twins,group);hg=_orbit_labels(habits,group)
    return {"raw_twin_branches":len(twins),"macroscopic_twin_tensor_orbits":len(tg),
            "twin_orbits":[[tids[i] for i in orbit] for orbit in tg],
            "raw_habit_records":len(habits),"macroscopic_habit_tensor_orbits":len(hg),
            "habit_orbits":[[hids[i] for i in orbit] for orbit in hg],
            "physical_microstructure_orbits":"NOT_DETERMINED_FROM_MACROSCOPIC_TENSORS",
            "note":"Each orbit groups a⊗n or b⊗m up to parent proper symmetry; it does not identify internal laminate geometry or phase-volume fractions."}


def projective_angle_deg(a,b):
    aa=_unit(a);bb=_unit(b)
    return float(np.degrees(np.arccos(np.clip(abs(float(aa@bb)),0,1))))


def measured_plane_cartesian(measured:Mapping[str,Any],*,parent_metric,product_metric=None,OR_M_cart_to_A_cart=None):
    frame=measured.get('frame')
    coords=measured.get('coordinates')
    if frame=='parent_cartesian':return _unit(coords)
    if frame=='parent_reciprocal_miller':return _unit(reciprocal_normal_to_cartesian(coords,parent_metric))
    if frame=='product_cartesian':
        if OR_M_cart_to_A_cart is None:raise EvidenceError("Measured product orientation relationship required")
        return _unit(np.asarray(OR_M_cart_to_A_cart,float)@_unit(coords))
    if frame=='product_reciprocal_miller':
        if product_metric is None or OR_M_cart_to_A_cart is None:
            raise EvidenceError("Product metric AND measured OR are required for product Miller plane comparison")
        return _unit(np.asarray(OR_M_cart_to_A_cart,float)@
                     _unit(reciprocal_normal_to_cartesian(coords,product_metric)))
    raise EvidenceError("Frame must be one of parent_cartesian, parent_reciprocal_miller, product_cartesian, product_reciprocal_miller")


def validate_orientation_rotation(R):
    R=np.asarray(R,float)
    if R.shape!=(3,3) or not np.isfinite(R).all() or np.linalg.norm(R.T@R-np.eye(3),'fro')>1e-6 or abs(float(np.linalg.det(R))-1)>1e-6:
        raise EvidenceError("OR must be a proper Cartesian rotation")
    return R


def compare_observed_planes(calculated_normals,measured:Mapping[str,Any],*,parent_metric,product_metric=None,
                           OR_M_cart_to_A_cart=None,parent_symmetry_ops=None):
    """Geometric residual ONLY: never authenticates independent experiment."""
    unc=float(measured.get('uncertainty_deg',float('nan')))
    if not np.isfinite(unc) or not 0<unc<=45:
        raise EvidenceError("Measured plane must specify a defensible angular uncertainty (0,45] degrees")
    if OR_M_cart_to_A_cart is not None:OR_M_cart_to_A_cart=validate_orientation_rotation(OR_M_cart_to_A_cart)
    normal=measured_plane_cartesian(measured,parent_metric=parent_metric,
        product_metric=product_metric,OR_M_cart_to_A_cart=OR_M_cart_to_A_cart)
    group=[np.eye(3)] if parent_symmetry_ops is None else proper_cartesian_group(np.asarray(parent_metric,float),parent_symmetry_ops)
    trials=[]
    for i,calc in enumerate(calculated_normals):
        c=_unit(calc)
        angle=min(projective_angle_deg(q@c,normal) for q in group)
        trials.append((angle,i))
    if not trials:raise EvidenceError("No computed planes to compare")
    angle,idx=min(trials)
    return {"status":"GEOMETRIC_MATCH_ONLY_EXTERNAL_SOURCE_REVIEW_REQUIRED" if angle<=unc else
             "NO_GEOMETRIC_MATCH_AT_DECLARED_UNCERTAINTY",
            "nearest_predicted_index":idx,"angle_deg":float(angle),
            "declared_uncertainty_deg":unc,"standardized_residual":float(angle/unc),
            "compared_candidates":len(trials),
            "caution":"Matching a plane does NOT validate C, OR, phase identity, independent sample, or theoretical completeness."}


def branch_evidence_status(branch,*,lattice_source=None,correspondence_source=None,observations=None):
    label=str(branch.classification)
    classification=str(branch.classification_status).lower()
    if label in ('Type I','Type II','Compound'):
        quality='SYMMETRY_TRANSPORTED_TYPE' if 'transport' in classification else 'CROSS_LOCKED_CLASSICAL_TYPE' if 'cross-lock' in classification else 'CLASSICAL_TYPE_EVIDENCE_UNCLEAR'
    elif 'nonconventional' in label.lower():quality='NONCONVENTIONAL_RANK_ONE_PARENT_TWOFOLD_EXCLUDED'
    else:quality='UNRESOLVED_TYPE'
    residuals=[float(branch.rank_one_residual),float(branch.rotation_residual)]
    finite=all(np.isfinite(x) and x<=1e-6 for x in residuals)
    return {"equation_status":"RESIDUAL_PASS_DOUBLE_PRECISION" if finite else "NUMERICAL_FAILURE_OR_UNCERTIFIED",
            "type_status":quality,"material_status":"C_NOT_SOURCE_VERIFIED" if not correspondence_source else "C_SOURCE_PROVIDED_NOT_INDEPENDENTLY_CONFIRMED",
            "experimental_status":"NOT_EXPERIMENTALLY_VALIDATED" if not observations else "OBSERVATIONS_EXIST_COMPARISON_REQUIRED",
            "precision_warning":"Small tensor residual is not a statement about lattice measurement accuracy."}
