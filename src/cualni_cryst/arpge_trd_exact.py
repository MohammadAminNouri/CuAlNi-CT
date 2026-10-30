from __future__ import annotations
"""Exact cubic Sigma3 composition and 2021 ARPGE longest twin chain semantics.

The original 2021 ARPGE 'calculateLLC' computes longest distance in a
prefix-tree of twin operator chains, not the greatest Sigma3 BFS depth.
This is independently implemented without copying the Python-2 source.
"""
from collections import deque
from itertools import product
from math import gcd

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .orientation_kernel import require_so3
from .parent_or_interop import proper_cubic_parent_operations


def legacy_llc_from_twin_chains(chains:object)->int:
    """Exact prefix-chain longest-length statistic, O(total tokens).

    Includes root as a node, so a sole chain of n operations has LLC=n.
    Distinct branches are joined at their last common prefix. Results match
    the 2021 loop definition, including duplicate/prefix chains, without O(n²).
    """
    if not isinstance(chains,(tuple,list)):
        raise ValueError("Twin operator chains must be a list")
    graph=[[]]
    children=[{}]
    for chain in chains:
        if not isinstance(chain,(list,tuple)):
            raise ValueError("Every twin chain must be an operator token list")
        if len(chain)>10000:
            raise ValueError("Twin chain exceeds supported depth")
        node=0
        for token in chain:
            if not isinstance(token,(int,str,tuple)):
                raise ValueError("Twin chain token is not an immutable operator label")
            try: hash(token)
            except TypeError as exc:
                raise ValueError("Unhashable operator token") from exc
            if token not in children[node]:
                child=len(graph)
                children[node][token]=child
                graph[node].append(child)
                graph.append([node])
                children.append({})
            node=children[node][token]
        if len(graph)>150000:raise ValueError("Twin chain graph too large")
    if len(graph)==1:return 0

    def farthest(start:int)->tuple[int,int]:
        queue=deque([start]);visited={start};end=start;distance={start:0}
        while queue:
            node=queue.popleft()
            if distance[node]>distance[end]:end=node
            for nxt in graph[node]:
                if nxt not in visited:
                    visited.add(nxt);distance[nxt]=distance[node]+1
                    queue.append(nxt)
        return end,distance[end]
    endpoint,_=farthest(0)
    _,diameter=farthest(endpoint)
    return int(diameter)


def _sigma3_numerator(axis:object)->np.ndarray:
    """Exactly generate 60° <111> cubic twin rotation as N/3, N integer."""
    a=np.asarray(axis)
    if a.shape!=(3,) or not np.all(np.abs(a)==1):
        raise ValueError("Sigma3 axis must be a signed integer <111> triple")
    x,y,z=map(int,a)
    skew=np.asarray([[0,-z,y],[z,0,-x],[-y,x,0]],int)
    numerator=3*np.eye(3,dtype=int)+np.outer(a,a)+3*skew
    if np.any(numerator%2):
        raise AssertionError("Exact Sigma3 rotation has odd numerator")
    return numerator//2


def exact_sigma3_chain(axes:object)->dict:
    """Compose exact 60° cubic <111> matrices and reduce rational denominator.

    The denominator of the reduced rational cubic rotation is its coincidence
    index; for these products it is a power of three. This distinguishes
    cubic-equivalent Sigma1 from Sigma3, Sigma9 and higher twins, irrespective
    of mere graph hop count.
    """
    if not isinstance(axes,(list,tuple)) or len(axes)>14:
        raise ValueError("Provide at most 14 exact Sigma3 operator steps")
    numerator=np.eye(3,dtype=np.int64)
    denominator=1
    labels=[]
    for axis in axes:
        T=_sigma3_numerator(axis)
        numerator=numerator@T
        denominator*=3
        while denominator>1 and np.all(numerator%3==0):
            numerator//=3
            denominator//=3
        labels.append(list(map(int,axis)))
    if not np.array_equal(numerator.T@numerator,
                          denominator**2*np.eye(3,dtype=np.int64)):
        raise AssertionError("Exact Sigma3 chain composition violates SO(3)")
    if abs(round(np.linalg.det(numerator))-denominator**3)>0:
        raise AssertionError("Improper exact Sigma3 product")
    return {"sigma_index":int(denominator),
            "numerator":numerator.astype(int).tolist(),
            "denominator":int(denominator),"operator_steps":labels,
            "chain_hops":len(labels),
            "status":"exact rational cubic CSL for selected ideal Sigma3 operators"}


def identify_sigma3_edges(
    grain_orientations:object, adjacency:object, *, tolerance_deg:float=2.,
    maximum_edges:int=50000,
)->dict:
    """Fit exact ideal Sigma3 operators under BOTH proper cubic symmetries.

    A source misorientation is conjugate/right-equivalent by cubic indexer
    representatives. Reject large residuals, and return exact rational 3x3
    operator numerators for downstream chain composition.
    """
    g=np.asarray(grain_orientations,float)
    edges=np.asarray(adjacency,int).reshape(-1,2)
    if g.ndim!=3 or g.shape[1:]!=(3,3) or len(g)>10000:
        raise ValueError("Cubic grain orientation stack invalid")
    if len(edges)>maximum_edges or not np.isfinite(tolerance_deg) or not 0<tolerance_deg<=10:
        raise ValueError("Sigma3 edge search bound/tolerance invalid")
    for a in g:
        require_so3(a,tolerance=2e-7,name="cubic grain")
    ops=proper_cubic_parent_operations().astype(int)
    numbers=[]
    for axis in product((-1,1),repeat=3):
        N=_sigma3_numerator(axis)
        for left in ops:
            for right in ops:
                numbers.append((left@N@right).reshape(9))
    unique=np.unique(np.asarray(numbers,dtype=np.int32),axis=0)
    mats=unique.reshape(-1,3,3)/3.
    qs=Rotation.from_matrix(mats).as_quat()
    tree=cKDTree(np.concatenate([qs,-qs],axis=0))
    output=[];residuals=[]
    for i,j in edges:
        if not (0<=i<len(g) and 0<=j<len(g)) or i==j:
            raise ValueError("Invalid spatial grain adjacency")
        measured=Rotation.from_matrix(g[i].T@g[j]).as_quat()
        _,index=tree.query(measured)
        idx=int(index)%len(unique)
        dot=abs(np.dot(measured,qs[idx]))
        error=float(np.rad2deg(2*np.arccos(np.clip(dot,0,1))))
        if error<=tolerance_deg:
            output.append((int(i),int(j),idx))
            residuals.append(error)
    return {
        "accepted_grain_edge_and_exact_operator_id":np.asarray(output,int).reshape(-1,3),
        "accepted_edge_residual_deg":np.asarray(residuals,float),
        "exact_rotation_numerators_over_3":unique.reshape(-1,3,3),
        "n_accepted_sigma3_edges":len(output),
        "n_all_edges":len(edges),
        "status":"approximate observed cubic boundary fitted to exact ideal Sigma3 operator, not certified twin plane",
    }


def sigma3_tree_csl_and_llc(
    n_grains:int, identification:dict
)->dict:
    """Compose ideal Sigma3 operators along a chosen BFS spanning tree.

    Verify closure on independent loops; conflicts are explicitly reported.
    LLC is the maximum *graph tree* diameter, distinct from the original
    operator-word LLC for arbitrary crystallographic assignment words.
    """
    if not isinstance(n_grains,int) or not 1<=n_grains<=10000:
        raise ValueError("Invalid cubic grain count")
    links=np.asarray(identification["accepted_grain_edge_and_exact_operator_id"],int).reshape(-1,3)
    ops=np.asarray(identification["exact_rotation_numerators_over_3"],int)
    adj=[[] for _ in range(n_grains)]
    for i,j,k in links:
        if not 0<=i<n_grains or not 0<=j<n_grains or not 0<=k<len(ops):
            raise ValueError("Invalid exact sigma3 edge")
        adj[i].append((j,ops[k]))
        adj[j].append((i,ops[k].T))
    group=np.full(n_grains,-1,int)
    numerator=[None]*n_grains
    denominators=np.ones(n_grains,dtype=np.int64)
    loop_conflicts=0
    domain_sizes=[]
    for root in range(n_grains):
        if group[root]>=0:continue
        dom=len(domain_sizes);group[root]=dom
        numerator[root]=np.eye(3,dtype=np.int64)
        q=deque([root]);size=0
        while q:
            i=q.popleft();size+=1
            for j,T in adj[i]:
                M=numerator[i]@T
                d=int(denominators[i])*3
                while d>1 and np.all(M%3==0):
                    M//=3;d//=3
                if group[j]<0:
                    group[j]=dom;numerator[j]=M;denominators[j]=d
                    q.append(j)
                else:
                    # Compare exact rational products, not noisy fitted Euler angles.
                    if not np.array_equal(M*denominators[j],numerator[j]*d):
                        loop_conflicts+=1
        domain_sizes.append(size)
    return {
        "domain_id":group,
        "composed_sigma_index_from_root":denominators,
        "n_domains":len(domain_sizes),"domain_sizes":domain_sizes,
        "inconsistent_closed_loop_edges":loop_conflicts//2,
        "status":"exact ideal-operator CSL along BFS tree; cycles flagged; NOT original 2021 operator-word LLC",
    }
