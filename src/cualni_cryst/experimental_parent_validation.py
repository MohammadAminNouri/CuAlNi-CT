from __future__ import annotations
"""Experimental retained-parent evidence for a FROZEN parent<-daughter OR.

Adapted from the Cu-Al-Ni 18R Site6/7/8 *logic*: prohibit inverse-fit data
leakage; preserve the crystallographic basis and specimen-frame contract;
generate the complete parent/product proper symmetry family; evaluate the
unchanged OR against independent EBSD measurements, including negative
controls. Daughter-only replay does NOT independently validate an absolute
parent orientation.

Here retained parent phase pixels really DO provide independent orientation
measurements for the evaluation, unlike the daughter-only Cu-Al-Ni scans.
No "nearest beta must be same prior grain" assumption is made: an adjacent
cross-phase boundary is evidence only of proximity, not common ancestry.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .orientation_kernel import require_so3


def cross_phase_contact_pairs(
    coords_xy: object, phase_ids: object, parent_phase_id: int,
    daughter_phase_id: int, *, distance:float, max_edges:int=1000000
) -> np.ndarray:
    """Every adjacent (measured parent, measured daughter) pixel pair.

    Guard against unintentionally linking nonadjacent islands across a gap.
    Caller must supply a measured square-grid step or a known calibrated
    local-neighbour radius. Outputs are deterministic sorted unique pairs.
    """
    xy=np.asarray(coords_xy,float)
    phase=np.asarray(phase_ids,int)
    if (xy.ndim!=2 or xy.shape[1]!=2 or phase.shape!=(len(xy),)
            or not np.isfinite(xy).all() or len(xy)>1000000
            or parent_phase_id==daughter_phase_id):
        raise ValueError("Invalid source EBSD coordinates or phase definitions")
    if not np.isfinite(distance) or not 0<distance<10000:
        raise ValueError("Explicit finite physical pixel-neighbour distance required")
    parent=np.flatnonzero(phase==parent_phase_id)
    daughter=np.flatnonzero(phase==daughter_phase_id)
    if not len(parent) or not len(daughter):
        raise ValueError("Source contains no indexed retained parent and daughter phases")
    kdt=cKDTree(xy[daughter])
    neighbor=kdt.query_ball_point(xy[parent],r=distance)
    count=sum(map(len,neighbor))
    if count>max_edges:
        raise ValueError("Cross-phase contact graph exceeds explicit memory limit")
    contacts=np.asarray([(int(parent[j]),int(daughter[k]))
                         for j,ks in enumerate(neighbor) for k in ks],int)
    return contacts.reshape(-1,2)


def complete_or_family(
    q_parent_from_daughter: object, parent_proper:object,
    daughter_proper:object
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Return every proper S_parent Q S_daughter crystal-frame relationship."""
    Q=require_so3(np.asarray(q_parent_from_daughter,float),
                  tolerance=2e-8,name="frozen parent<-daughter OR")
    gp=np.asarray(parent_proper,float)
    gm=np.asarray(daughter_proper,float)
    if gp.ndim!=3 or gm.ndim!=3 or gp.shape[1:]!=(3,3) or gm.shape[1:]!=(3,3):
        raise ValueError("Complete proper parent/daughter point groups required")
    if not 1<=len(gp)<=48 or not 1<=len(gm)<=48:
        raise ValueError("Point-group order outside crystallographic bounds")
    for v in list(gp)+list(gm):
        require_so3(v,tolerance=2e-8,name="proper point-group operation")
    family=(gp[:,None]@Q)@gm[None]
    # Proper symmetry pairs can yield duplicates; retaining them changes no
    # closest-match residual but preserves transparent source combinations.
    return family.reshape(-1,3,3),gp,gm


def frozen_or_contact_residuals(
    measured_parent:object, measured_daughter:object, frozen_q:object,
    parent_proper:object, daughter_proper:object, *,
    chunk:int=2500,
)->np.ndarray:
    """Vectorized complete-symmetry misfit on paired *observed* EBSD pixels.

    Both EBSD rotations must be crystal->specimen in ONE calibrated frame.
    R_observed=g_parent.T @ g_daughter; ideal R=S_parent @ Q @ S_child.
    This evaluates OR geometry, NOT reconstructed original parent-grain IDs.
    """
    p=np.asarray(measured_parent,float)
    m=np.asarray(measured_daughter,float)
    if p.shape!=m.shape or p.ndim!=3 or p.shape[1:]!=(3,3) or not 1<=len(p)<=1000000:
        raise ValueError("Expected equal length (N,3,3) measured parent/daughter pairs")
    if not 1<=chunk<=10000:raise ValueError("Invalid angular chunk")
    family,_,_=complete_or_family(frozen_q,parent_proper,daughter_proper)
    if not np.isfinite(p).all() or not np.isfinite(m).all():
        raise ValueError("Indexed parent/daughter rotations must be finite")
    for t in np.linspace(0,len(p)-1,min(len(p),50),dtype=int):
        require_so3(p[t],tolerance=2e-7,name="indexed parent rotation")
        require_so3(m[t],tolerance=2e-7,name="indexed daughter rotation")
    target=Rotation.from_matrix(family).as_quat()
    residual=np.empty(len(p),float)
    for start in range(0,len(p),chunk):
        cut=slice(start,min(start+chunk,len(p)))
        relative=np.swapaxes(p[cut],1,2)@m[cut]
        observed=Rotation.from_matrix(relative).as_quat()
        dot=np.max(abs(observed@target.T),axis=1)
        residual[cut]=np.rad2deg(2*np.arccos(np.clip(dot,0,1)))
    return residual


def frozen_or_boundary_audit(
    *, orientations:object, phase_ids:object, xy:object,
    parent_phase_id:int,daughter_phase_id:int, neighbor_distance:float,
    frozen_q:object,parent_proper:object,daughter_proper:object,
    negative_q:object,quality_mask:object|None=None,
    pixel_group:object|None=None,maximum_sample:int=20000,
    seed:int=20261030,acceptance_deg:float=3.0,
)->dict:
    """Strictly predeclared OR and negative control on the SAME spatial edges.

    Quality filters and random subsample are selected before examining any
    measured orientation/OR score; optional pixel-group tags support
    spatial group bootstrap downstream. Neither Q is fitted from these data.
    """
    if not 1<=maximum_sample<=100000 or not 0<acceptance_deg<=15:
        raise ValueError("Invalid pre-registered sample size or threshold")
    phases=np.asarray(phase_ids,int);positions=np.asarray(xy,float)
    measured=np.asarray(orientations,float)
    if measured.shape!=(len(phases),3,3):
        raise ValueError("Invalid source orientation stack")
    # Completeness of the supplied cubic/hcp groups is the caller's scientific
    # obligation; all entries are validated before any score is computed.
    complete_or_family(frozen_q,parent_proper,daughter_proper)
    complete_or_family(negative_q,parent_proper,daughter_proper)
    edges=cross_phase_contact_pairs(positions,phases,
             parent_phase_id,daughter_phase_id,distance=neighbor_distance)
    if quality_mask is not None:
        valid=np.asarray(quality_mask,bool)
        if valid.shape!=(len(phases),):
            raise ValueError("Quality-mask shape mismatch")
        edges=edges[valid[edges[:,0]]&valid[edges[:,1]]]
    candidates=len(edges)
    if not len(edges):
        return {"status":"insufficient matched-phase contacts after quality gate",
                "n_source_contacts":0,"physical_parent_grain_certified":False}
    rng=np.random.default_rng(seed)
    if len(edges)>maximum_sample:
        edges=edges[np.sort(rng.choice(len(edges),maximum_sample,replace=False))]
    a=measured[edges[:,0]]
    m=measured[edges[:,1]]
    correct=frozen_or_contact_residuals(a,m,frozen_q,parent_proper,daughter_proper)
    wrong=frozen_or_contact_residuals(a,m,negative_q,parent_proper,daughter_proper)
    # No Q is refitted, labels are not shuffled to make the target look better.
    out={
        "schema":"cualni.frozen-or-real-parent-interface/1",
        "n_source_contacts_quality_filtered":candidates,
        "n_tested_contacts":len(edges),
        "n_parent_pixels_in_sample":len(np.unique(edges[:,0])),
        "n_daughter_pixels_in_sample":len(np.unique(edges[:,1])),
        "correct_OR_median_deg":float(np.median(correct)),
        "negative_OR_median_deg":float(np.median(wrong)),
        "correct_OR_fraction_within_threshold":float(np.mean(correct<=acceptance_deg)),
        "negative_OR_fraction_within_threshold":float(np.mean(wrong<=acceptance_deg)),
        "acceptance_deg":float(acceptance_deg),
        "selection_random_seed":int(seed),
        "Q_optimized_from_holdout":False,
        "sample_frame_calibrated":False,
        "common_ancestry_of_adjacent_pixels_verified":False,
        "full_parent_grain_reconstruction_verified":False,
        "physical_parent_grain_certified":False,
        "status":"observed parent/daughter contact OR benchmark, NOT validated prior-parent topology",
    }
    if pixel_group is not None:
        groups=np.asarray(pixel_group)
        if groups.shape!=(len(phases),):
            raise ValueError("Group labels must be defined for all source pixels")
        out["contact_parent_group_ids"]=groups[edges[:,0]].tolist()
    return out


def group_disjoint_source_or_selection(
    *, orientations:object, phase_ids:object, xy:object, parent_groups:object,
    parent_phase_id:int, daughter_phase_id:int, neighbor_distance:float,
    source_or_candidates:dict[str,object], parent_proper:object,
    daughter_proper:object, quality_mask:object|None=None,
    train_fraction:float=.6, seed:int=20261030,
    max_pairs_per_split:int=2500, acceptance_deg:float=3.,
)->dict:
    """Train on one set of retained-parent islands, freeze the named OR,
    then score disjoint *parent island IDs* without retuning.

    Group IDs are spatial connected components of the ORIGINAL retained
    parent phase (island, not certified previous parent grain). Both train
    and test include measured parent orientations. No parent orientation
    from a test island enters the choice of which archived OR is used.
    For source-transfer 18R with no retained beta, use the separate frozen
    cross-scan daughter replay instead; this API must reject such data.
    """
    x=np.asarray(orientations,float)
    phase=np.asarray(phase_ids,int)
    groups=np.asarray(parent_groups,int)
    if (x.ndim!=3 or x.shape[1:]!=(3,3)
        or phase.shape!=(len(x),) or groups.shape!=(len(x),)
        or not 0<train_fraction<1 or not 10<=max_pairs_per_split<=30000):
        raise ValueError("Invalid parent-island group cross-validation contract")
    if not isinstance(source_or_candidates,dict) or not 2<=len(source_or_candidates)<=24:
        raise ValueError("Need at least two predeclared source OR candidates")
    if not 0<acceptance_deg<=15:
        raise ValueError("Invalid acceptance threshold")
    for name,Q in source_or_candidates.items():
        if not isinstance(name,str) or not name:
            raise ValueError("Source OR candidates must be named")
        complete_or_family(Q,parent_proper,daughter_proper)
    edge=cross_phase_contact_pairs(xy,phase,parent_phase_id,
                                   daughter_phase_id,distance=neighbor_distance)
    if quality_mask is not None:
        valid=np.asarray(quality_mask,bool)
        if valid.shape!=(len(x),):
            raise ValueError("Quality mask shape differs from original map")
        edge=edge[valid[edge[:,0]]&valid[edge[:,1]]]
    # A retained-pixel island is defined independently of Euler/OR values.
    gp=groups[edge[:,0]]
    unique=np.unique(gp[gp>0])
    if len(unique)<4:
        raise ValueError("Insufficient independent retained-phase islands for group holdout")
    rng=np.random.default_rng(seed)
    rng.shuffle(unique)
    n_training=max(2,min(len(unique)-2,int(round(train_fraction*len(unique)))))
    train=set(unique[:n_training].tolist())
    mask=np.isin(gp,list(train))
    train_pairs=edge[mask];test_pairs=edge[~mask]
    if min(len(train_pairs),len(test_pairs))<10:
        raise ValueError("Train/holdout contact coverage insufficient")
    def subset(pairs):
        if len(pairs)>max_pairs_per_split:
            ix=np.sort(rng.choice(len(pairs),max_pairs_per_split,replace=False))
            return pairs[ix]
        return pairs
    train_pairs=subset(train_pairs);test_pairs=subset(test_pairs)
    candidates=[]
    for name,Q in sorted(source_or_candidates.items()):
        residual=frozen_or_contact_residuals(
            x[train_pairs[:,0]],x[train_pairs[:,1]],
            Q,parent_proper,daughter_proper,
        )
        candidates.append({
            "name":name,
            "training_median_deg":float(np.median(residual)),
            "training_fraction_within_threshold":
                float(np.mean(residual<=acceptance_deg)),
        })
    chosen=sorted(candidates,key=lambda q:(
        -q["training_fraction_within_threshold"],
        q["training_median_deg"],q["name"],
    ))[0]
    Q=np.asarray(source_or_candidates[chosen["name"]],float)
    test=frozen_or_contact_residuals(
        x[test_pairs[:,0]],x[test_pairs[:,1]],Q,
        parent_proper,daughter_proper,
    )
    negative=frozen_or_contact_residuals(
        x[test_pairs[:,0]],x[test_pairs[:,1]],
        Q@Rotation.from_euler("x",15,degrees=True).as_matrix(),
        parent_proper,daughter_proper,
    )
    # Parent island group split is disjoint; other (daughter) specimen regions
    # may still correlate spatially, so this is NOT a fully independent grain.
    return {
        "chosen_archived_or_by_training_only":chosen["name"],
        "training_preset_comparison":candidates,
        "n_training_parent_islands":len(train),
        "n_holdout_parent_islands":len(unique)-len(train),
        "n_train_contact_pairs":len(train_pairs),
        "n_holdout_contact_pairs":len(test_pairs),
        "heldout_median_deg":float(np.median(test)),
        "heldout_fraction_within_threshold":float(np.mean(test<=acceptance_deg)),
        "wrong_15deg_heldout_median_deg":float(np.median(negative)),
        "wrong_15deg_heldout_fraction_within_threshold":
            float(np.mean(negative<=acceptance_deg)),
        "parent_island_labels_disjoint":True,
        "candidate_OR_refit_on_holdout":False,
        "source_crossphase_metadata_verified":False,
        "parent_grain_topology_certified":False,
        "experimental_parent_reconstruction_certified":False,
        "status":"held-out retained-phase island OR-selection screen, not same prior-parent reconstruction",
    }
