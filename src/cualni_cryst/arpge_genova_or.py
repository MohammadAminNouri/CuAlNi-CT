from __future__ import annotations

"""Independent numerical GenOVa OR groupoid, with complete finite-group checks.

Input Q is a proper Cartesian daughter->parent rotation. Never substitute
the non-orthogonal transformation/correspondence matrix from legacy GenOVa.
All returned discrete operator classes follow proper SO(3) parent symmetry.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .orientation_kernel import require_so3


def or_groupoid(
    q_parent_from_daughter:object,
    parent_proper_ops:object,
    daughter_proper_ops:object,
    *, intersection_tolerance_deg:float=1e-4,
)->dict:
    Q=require_so3(np.asarray(q_parent_from_daughter,float),
                  tolerance=2e-8,name="parent<-daughter Q")
    G=np.asarray(parent_proper_ops,float)
    M=np.asarray(daughter_proper_ops,float)
    if (G.ndim!=3 or M.ndim!=3 or G.shape[1:]!=(3,3)
        or M.shape[1:]!=(3,3) or not 1<=len(G)<=48
        or not 1<=len(M)<=48):
        raise ValueError("Proper parent and daughter point groups required")
    if not np.isfinite(intersection_tolerance_deg) or not 0<intersection_tolerance_deg<=.1:
        raise ValueError("Symmetry intersection tolerance must be precise")
    for x in list(G)+list(M):
        require_so3(x,tolerance=2e-7,name="proper symmetry operator")
    def radians_distance(A:np.ndarray,B:np.ndarray)->np.ndarray:
        # Frobenius distance to a proper rotation maps uniquely to angle.
        r=np.linalg.norm(A-B,axis=(-2,-1))/np.sqrt(8)
        return 2*np.arcsin(np.clip(r,0,1))
    def map_group(ops:np.ndarray)->tuple[np.ndarray,int,int]:
        n=len(ops)
        multiplication=np.full((n,n),-1,dtype=int)
        for i in range(n):
            lhs=ops[i]@ops
            dist=np.linalg.norm(lhs[:,None,:,:]-ops[None,:,:,:],axis=(2,3))
            indices=np.argmin(dist,axis=1)
            if np.max(dist[np.arange(n),indices])>1e-6:
                raise ValueError("Entered proper symmetry matrices are not group-closed")
            multiplication[i]=indices
        ident=np.linalg.norm(ops-np.eye(3),axis=(1,2))
        idx=int(np.argmin(ident))
        if ident[idx]>1e-6 or len(set(multiplication.ravel().tolist()))!=n:
            raise ValueError("Symmetry group lacks identity or completeness")
        return multiplication,idx,n
    table,e,n=map_group(G)
    map_group(M)
    if len(np.unique(np.round(G.reshape(n,9),8),axis=0))!=n:
        raise ValueError("Duplicate proper parent symmetry operators")
    aligned=(Q.T@G@Q)
    distances=np.linalg.norm(aligned[:,None]-M[None],axis=(2,3))
    # Exact proper-rotation disorientation threshold in Frobenius units.
    angular=np.rad2deg(2*np.arcsin(np.clip(np.min(distances,axis=1)/np.sqrt(8),0,1)))
    H=np.flatnonzero(angular<=intersection_tolerance_deg)
    if len(H)==0 or e not in H:
        raise ValueError("Parent/product proper symmetry intersection missing identity")
    subset=set(int(x) for x in H)
    if any(int(table[i,j]) not in subset for i in H for j in H):
        raise ValueError("Numerical symmetry intersection fails subgroup closure")
    inverse=np.asarray([int(np.flatnonzero(
        (table[i]==e)&(table[:,i]==e)
    )[0]) for i in range(n)],int)
    cosets=[];assigned={}
    for i in range(n):
        if i in assigned:continue
        entries=sorted(int(table[i,h]) for h in H)
        label=len(cosets)
        for k in entries:
            if k in assigned and assigned[k]!=label:
                raise AssertionError("Parent coset overlap")
            assigned[k]=label
        cosets.append(entries)
    dcosets=[];which={}
    for i in range(n):
        if i in which:continue
        elems=sorted({int(table[int(table[h,i]),k])
                      for h in H for k in H})
        label=len(dcosets)
        for k in elems:which[k]=label
        dcosets.append(elems)
    if len(assigned)!=n or len(which)!=n:
        raise AssertionError("Incomplete finite group partitions")
    reps=np.asarray([x[0] for x in cosets],int)
    adjacency=[]
    for i in reps:
        adjacency.append([which[int(table[inverse[i],j])] for j in reps])
    inverse_classes=[which[inverse[entries[0]]] for entries in dcosets]
    nfix=0
    for h in H:
        for k in H:
            for g in range(n):
                # a*g=b? fixed action h*g*k^-1=g
                if table[int(table[h,g]),inverse[k]]==g:
                    nfix+=1
    if nfix%(len(H)**2)!=0:
        raise AssertionError("Noninteger Burnside double-coset count")
    burnside=nfix//(len(H)**2)
    if burnside!=len(dcosets):
        raise AssertionError("Burnside disagrees with double-coset enumeration")
    N=len(cosets)
    composition=[[set() for _ in dcosets] for _ in dcosets]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                composition[adjacency[i][j]][adjacency[j][k]].add(adjacency[i][k])
    # Identity operator adjacency on each vertex
    identity_operator=which[e]
    if any(adjacency[i][i]!=identity_operator for i in range(N)):
        raise AssertionError("Groupoid identity invalid")
    if n%len(H)!=0 or sum(len(x) for x in dcosets)!=n:
        raise AssertionError("Invalid Lagrange or double-coset partition")
    return {
        "parent_symmetry_order":n,"daughter_symmetry_order":len(M),
        "intersection_order":len(H),"n_variants":N,
        "n_operator_classes":len(dcosets),
        "coset_member_parent_op_ids":cosets,
        "operator_class_member_parent_op_ids":dcosets,
        "adjacency_operator_class":adjacency,
        "inverse_operator_class":inverse_classes,
        "operator_composition":[[sorted(v) for v in row] for row in composition],
        "burnside_count":burnside,
        "max_matched_symmetry_angle_deg":float(np.max(angular[H])),
        "status":"numerically verified proper OR groupoid; original GenOVa saved table numerical parity NOT established",
    }


def audit_archived_or_groupoids(inventory:dict,geometry:dict)->dict:
    """Attempt every 2021 preset's proper-OR groupoid with source metric audit.

    Returns explicit per-case outcomes; missing crystal data, incompatible
    point groups or inconsistent signed ORs are never silently fabricated.
    """
    from collections import Counter
    from .ebsd_map import EBSDPhase
    from .lattice import Lattice
    phase={p["name"]:p for p in inventory["crystal_metrics"]}
    results=[]
    for t,g in zip(inventory["transitions"],geometry["results"]):
        name=t["name"]
        if g.get("status")!="geometric candidate":
            results.append({"name":name,"status":"blocked by geometric/phase evidence",
                            "cause":g.get("status")})
            continue
        try:
            def sym(pid,rec):
                p=phase[rec]
                L=Lattice(p["a_angstrom"],p["b_angstrom"],p["c_angstrom"],
                          p["alpha_deg"],p["beta_deg"],p["gamma_deg"])
                pg=p.get("point_group_legacy","").strip()
                if not pg:
                    raise ValueError("Archive crystal has no point group metadata")
                return tuple(EBSDPhase.from_point_group(
                    pid,rec,L,pg,
                ).proper_symmetry_cartesian)
            parent=sym(1,t["parent"])
            daughter=sym(2,t["product"])
            calc=or_groupoid(np.asarray(g["Q_parent_from_daughter"]),parent,daughter)
            results.append({
                "name":name,"status":"theoretical OR groupoid computed",
                "parent_order":calc["parent_symmetry_order"],
                "daughter_order":calc["daughter_symmetry_order"],
                "intersection_order":calc["intersection_order"],
                "variants":calc["n_variants"],
                "operator_classes":calc["n_operator_classes"],
                "burnside_count":calc["burnside_count"],
                "operator_composition_verified":True,
                "original_genova_pickle_table_parity":False,
            })
        except (ValueError,KeyError,TypeError,ArithmeticError,AssertionError) as exc:
            results.append({"name":name,"status":"rejected","reason":str(exc)})
    return {
        "n_presets":len(results),
        "status_counts":dict(Counter(r["status"] for r in results)),
        "results":results,
        "original_saved_operator_table_parity_verified":False,
    }
