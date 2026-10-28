from __future__ import annotations

"""Vectorized EXACT same-phase EBSD neighbor disorientations and grain/KAM maps.

For a finite proper point group G, two-sided disorientation reduces to
min_{S in G} angle(g_a.T g_b S). This is the same SO(3) quotient used by the
native minimum_disorientation reference implementation, without nested Python
loops over every pixel edge. Native grain_statistics remains authoritative
for symmetry-aligned means, GOS and maximum grain spread.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from cualni_cryst.ebsd_analysis import GrainSegmentation


def edge_disorientations(data, phases, neighbor_graph, *, chunk=4096):
    """Return one angle per edge, NaN across phases or unindexed pixels."""
    edges=np.asarray(neighbor_graph.edges,dtype=int).reshape(-1,2)
    out=np.full(len(edges),np.nan,dtype=float)
    if len(edges)==0:
        return out
    if chunk < 128:
        raise ValueError("Disorientation batch must be >= 128")
    a,b=edges[:,0],edges[:,1]
    active=np.asarray(data.indexed[a]&data.indexed[b]
                      &(data.phase_id[a]==data.phase_id[b]),bool)
    for phase_id,phase in phases.items():
        keys=np.flatnonzero(active & (data.phase_id[a]==int(phase_id)))
        sym=np.asarray(phase.proper_symmetry_cartesian,dtype=float)
        if sym.ndim!=3 or sym.shape[1:]!=(3,3):
            raise ValueError("Phase symmetry must consist of proper 3x3 rotations")
        for start in range(0,len(keys),chunk):
            take=keys[start:start+chunk]
            m1=np.asarray(data.orientations[a[take]],dtype=float)
            m2=np.asarray(data.orientations[b[take]],dtype=float)
            rel=np.einsum("nji,njk->nik",m1,m2,optimize=True)
            traces=np.einsum("nij,kji->nk",rel,sym,optimize=True)
            out[take]=np.rad2deg(np.arccos(np.clip(
                (np.max(traces,axis=1)-1.)/2.,-1.,1.,
            )))
    return out


def accelerated_ebsd_maps(data, phases, neighbor_graph, *,
                          threshold_deg, minimum_points,
                          kam_max_deg):
    """Exact graph-components segmentation + edge-averaged KAM.

    Unindexed/missing rows are never made into grains; isolated pixels below
    minimum_points are -1. KAM excludes mixed-phase neighbors and
    disorientations greater than the explicit cutoff.
    """
    if threshold_deg<=0 or minimum_points<1 or kam_max_deg<=0:
        raise ValueError("Invalid EBSD threshold/minimum/cutoff")
    n=int(data.n_points)
    edges=np.asarray(neighbor_graph.edges,dtype=int).reshape(-1,2)
    angles=edge_disorientations(data,phases,neighbor_graph)
    allowed=np.isfinite(angles)&(angles<=threshold_deg)
    a,b=edges[:,0],edges[:,1]
    if np.any(allowed):
        selected_a,selected_b=a[allowed],b[allowed]
        graph=coo_matrix((
            np.ones(2*len(selected_a),dtype=np.int8),
            (np.r_[selected_a,selected_b],np.r_[selected_b,selected_a]),
        ),shape=(n,n)).tocsr()
        _,raw_labels=connected_components(graph,directed=False)
    else:
        raw_labels=np.arange(n,dtype=int)
    active=np.asarray(data.indexed,dtype=bool)
    observed=raw_labels[active]
    unique,counts=np.unique(observed,return_counts=True)
    retained=unique[counts>=minimum_points]
    # This canonical ordering is by component first appearance in pixel scan.
    # It matches the native segment_grains sorted-by-first-pixel convention.
    labels=np.full(n,-1,dtype=int)
    if len(retained):
        first=np.full(len(retained),n,dtype=int)
        for k,label in enumerate(retained):
            first[k]=int(np.flatnonzero(active&(raw_labels==label))[0])
        ordered=retained[np.argsort(first)]
        for gid,label in enumerate(ordered):
            labels[active&(raw_labels==label)]=gid
    segment=GrainSegmentation(
        grain_id=labels,n_grains=len(retained),
        threshold_deg=float(threshold_deg),
        neighbor_graph=neighbor_graph,
    )
    valid=np.isfinite(angles)&(angles<=kam_max_deg)
    total=np.zeros(n,float)
    samples=np.zeros(n,dtype=np.int32)
    if np.any(valid):
        va,vb=a[valid],b[valid]
        values=angles[valid]
        np.add.at(total,va,values)
        np.add.at(total,vb,values)
        np.add.at(samples,va,1)
        np.add.at(samples,vb,1)
    kam=np.full(n,np.nan,float)
    np.divide(total,samples,out=kam,where=samples>0)
    # Max and mean over all indexed same-phase neighbors, not just KAM cutoff.
    linked=np.isfinite(angles)
    maximum=np.full(n,np.nan,float)
    neighbor_count=np.zeros(n,dtype=np.int32)
    if np.any(linked):
        aa,bb=a[linked],b[linked]
        vv=angles[linked]
        maxima=np.full(n,-np.inf)
        np.maximum.at(maxima,aa,vv)
        np.maximum.at(maxima,bb,vv)
        neighbor_count+=np.bincount(
            np.r_[aa,bb],minlength=n,
        ).astype(np.int32)
        maximum[neighbor_count>0]=maxima[neighbor_count>0]
    boundary_mask=np.zeros(n,dtype=bool)
    inter=(labels[a]>=0)&(labels[b]>=0)&(labels[a]!=labels[b])
    if np.any(inter):
        boundary_mask[a[inter]]=True
        boundary_mask[b[inter]]=True
    return {
        "segmentation":segment,
        "kam_deg":kam,
        "neighbor_max_deg":maximum,
        "neighbor_count":neighbor_count,
        "grain_boundary_mask":boundary_mask,
        "edge_angles_deg":angles,
    }
