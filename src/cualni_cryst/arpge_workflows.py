from __future__ import annotations

"""Auditable deterministic reconstruction cycles, morphology and operators.

These are explicitly independent modern algorithms, not execution/parity of
the original 2021 GUI. Parent identity remains conditional on a supplied OR.
"""

from collections import Counter
import numpy as np
from scipy import ndimage

from .arpge_spatial import reconstruct_spatial_parents
from .parent_or_interop import frozen_or_boundary_audit


def reconstruction_tolerance_cycles(
    segmentation:dict,Q:np.ndarray,parent_ops:tuple,child_ops:tuple,
    *, cycles:tuple[tuple[float,float],...]=((1.0,1.5),(2.0,2.5),(3.0,3.5)),
    nucleus_size:int=3,
)->dict:
    """Sweep a stated, deterministic tolerance schedule without hidden tuning.

    Each cycle is independent; do NOT pretend that a changed candidate domain
    ID refers to the identical physical domain. Report pixel-label agreement
    after overlap matching only through counts and unordered partitions.
    """
    if not 1<=len(cycles)<=12:
        raise ValueError("Specify one to twelve explicit tolerance cycles")
    runs=[]
    for nuc,growth in cycles:
        if not np.isfinite(nuc) or not np.isfinite(growth) or nuc>growth:
            raise ValueError("Invalid stated nucleation/growth schedule")
        result=reconstruct_spatial_parents(
            segmentation,Q,parent_ops,child_ops,
            nucleation_tolerance_deg=float(nuc),
            growth_tolerance_deg=float(growth),
            nucleus_size=nucleus_size,
        )
        runs.append(result)
    reports=[]
    for i,(params,res) in enumerate(zip(cycles,runs)):
        px=np.asarray(res["pixel_parent_id"],int)
        assigned=px>=0
        reports.append({
            "cycle":i+1,"nucleation_deg":float(params[0]),
            "growth_deg":float(params[1]),
            "n_parent_domains":res["n_parent_domains"],
            "n_assigned_pixels":int(np.sum(assigned)),
            "n_ambiguous_nuclei_skipped":res["n_ambiguous_nuclei_skipped"],
            "n_search_limit_seeds":res["n_seeds_skipped_combinatorial_limit"],
            "physical_or_certified":False,
        })
    # Select the most conservative cycle with highest *supported* pixel
    # coverage; this is a declared heuristic, not a fitted scientific optimum.
    ranked=sorted(range(len(runs)),key=lambda i:(
        -reports[i]["n_assigned_pixels"],
        reports[i]["n_ambiguous_nuclei_skipped"],
        reports[i]["growth_deg"],i,
    ))
    winning=ranked[0]
    return {
        "chosen_cycle":winning+1,"cycle_reports":reports,
        "candidate":runs[winning],
        "selection_policy":"maximum assigned pixels, then minimum ambiguous nuclei, then tighter tolerance",
        "warning":"Tolerance sweep can overfit morphology; inspect all cycles and validate against held-out parent phase",
    }


def polish_indexed_parent_islands(
    parent_pixel_id:np.ndarray, indexed_mask:np.ndarray,shape:tuple[int,int],
    *, maximum_hole_pixels:int=4,
)->dict:
    """Fill only small *indexed but unassigned* enclosed islands.

    No unindexed pixel is ever touched. Input parent labels remain unchanged,
    and every correction is returned in a separate explicit delta mask.
    """
    if not 0<=maximum_hole_pixels<=100 or len(shape)!=2:
        raise ValueError("Invalid bounded morphology settings")
    ids=np.asarray(parent_pixel_id,int)
    indexed=np.asarray(indexed_mask,bool)
    if ids.shape!=indexed.shape or ids.size!=shape[0]*shape[1]:
        raise ValueError("Parent morphology and original map shape mismatch")
    original=ids.reshape(shape)
    valid=indexed.reshape(shape)
    result=original.copy()
    gap=(original<0)&valid
    components,n=ndimage.label(gap,np.asarray(
        [[0,1,0],[1,1,1],[0,1,0]],bool,
    ))
    filled=np.zeros(shape,bool)
    for i in range(1,n+1):
        island=components==i
        if np.count_nonzero(island)>maximum_hole_pixels:continue
        # Does not fill edges or unindexed masks. Require all 4-sided
        # surrounding neighbours to carry exactly one assigned parent ID.
        dilated=ndimage.binary_dilation(island,structure=np.asarray(
            [[0,1,0],[1,1,1],[0,1,0]],bool,
        ))
        border=dilated&~island
        if (not np.any(border) or np.any(~valid[border]) or
                np.any(original[border]<0) or
                np.any(island[0]) or np.any(island[-1]) or
                np.any(island[:,0]) or np.any(island[:,-1])):
            continue
        domain=np.unique(original[border])
        if len(domain)!=1:continue
        result[island]=domain[0]
        filled[island]=True
    return {
        "polished_parent_pixel_id":result.ravel(),
        "changed_pixel_mask":filled.ravel(),
        "n_pixels_changed":int(np.sum(filled)),
        "n_unindexed_pixels_changed":int(np.sum(filled&~valid)),
        "status":"separate morphological hypothesis, never overwritten original",
    }


def boundary_operator_histogram(
    segmentation:dict,Q:np.ndarray,parent_symmetry:tuple,
    child_symmetry:tuple,*, bin_width_deg:float=1.0,
)->dict:
    """Orientation-operator misfit histogram on ACTUAL daughter adjacency."""
    if not np.isfinite(bin_width_deg) or not 0.1<=bin_width_deg<=10:
        raise ValueError("Invalid histogram bin width")
    edges=np.asarray(segmentation["grain_adjacency"],int).reshape(-1,2)
    if not len(edges):
        return {"n_pairs":0,"histogram":[],"histogram_contacts":[]}
    means=np.asarray(segmentation["grain_orientations"])
    pairs=means[edges]
    report=frozen_or_boundary_audit(
        pairs,Q,parent_symmetry,child_symmetry,
    )
    angles=np.asarray(report["operator_misfit_deg"],float)
    contacts=np.asarray(segmentation["boundary_contacts"],float)
    stop=max(181,float(np.max(angles)+bin_width_deg))
    bins=np.arange(0,stop+bin_width_deg,bin_width_deg)
    count,_=np.histogram(angles,bins=bins)
    weighted,_=np.histogram(angles,bins=bins,weights=contacts)
    return {
        "n_pairs":len(edges),"bins_lower_deg":bins[:-1].tolist(),
        "histogram":count.tolist(),
        "histogram_contacts":weighted.tolist(),
        "median_operator_misfit_deg":float(np.median(angles)),
        "fraction_under_3deg":float(np.mean(angles<3)),
        "source":"spatial segmented daughter pairs, not proof of physical OR",
    }
