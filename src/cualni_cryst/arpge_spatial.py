from __future__ import annotations

"""Independent ARPGE-style spatial EBSD reconstruction.

No original Python-2 source is imported or executed. Crystal->specimen SO(3)
orientations, a specified parent<-daughter Cartesian OR and complete proper
groups are mandatory. This is a bounded, testable spatial baseline, NOT an
exact reproduction of Cayron's 2021 GUI or its original groupoid nucleation.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .orientation_kernel import require_so3
from .parent_variant_audit import daughter_variant_assignment


def _qmul(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Vectorized scalar-last Hamilton product q*s."""
    q=np.asarray(q,float)
    s=np.asarray(s,float)
    return np.column_stack((
        q[:,3,None]*s[:3] + s[3]*q[:,:3] + np.cross(q[:,:3],s[:3]),
        q[:,3]*s[3] - q[:,:3]@s[:3],
    ))


def disorientation_edges(q: np.ndarray, edges: np.ndarray,
                         symmetry: tuple[np.ndarray,...],
                         chunk: int = 40000) -> np.ndarray:
    """Proper point-group-reduced angular distance, degrees, without projections."""
    q=np.asarray(q,float)
    edges=np.asarray(edges,int)
    if edges.ndim!=2 or edges.shape[1]!=2:
        raise ValueError("Edges must be (E,2)")
    if q.ndim!=2 or q.shape[1]!=4 or not np.isfinite(q).all():
        raise ValueError("Invalid quaternion array")
    syms=[Rotation.from_matrix(require_so3(np.asarray(s,float),
          tolerance=2e-8,name="daughter symmetry")).as_quat() for s in symmetry]
    if not syms:
        raise ValueError("Daughter symmetry is empty")
    out=np.empty(len(edges),float)
    for start in range(0,len(edges),chunk):
        ij=edges[start:start+chunk]
        qa=q[ij[:,0]]; qb=q[ij[:,1]]
        best=np.zeros(len(ij),float)
        for s in syms:
            np.maximum(best,np.abs(np.sum(qa*_qmul(qb,s),axis=1)),out=best)
        out[start:start+chunk]=np.rad2deg(2*np.arccos(np.clip(best,0,1)))
    return out


def _validate_ebsd(ebsd, phase_id: int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    n=len(ebsd.phase_id)
    coords=np.column_stack((np.asarray(ebsd.x,float),np.asarray(ebsd.y,float)))
    good=np.asarray(ebsd.indexed,bool)&(np.asarray(ebsd.phase_id)==phase_id)
    if coords.shape!=(n,2) or not np.isfinite(coords).all():
        raise ValueError("All EBSD pixel coordinates must be finite 2D positions")
    if np.count_nonzero(good)<2:
        raise ValueError("Need at least two indexed daughter pixels")
    return coords,good,np.asarray(ebsd.orientations,float)


def segment_daughter_grains(ebsd, product_phase_id: int,
                            product_symmetry: tuple[np.ndarray,...],
                            *, tolerance_deg: float=3.,
                            minimum_pixels: int=5, edge_radius_factor: float=1.12,
                            max_pixels: int=850000, max_edges: int=3600000) -> dict:
    """Adjacent indexed pixels only; symmetry-aware union-find segmentation.

    The inferred neighbour distance is an auditable *geometric heuristic*.
    A phase boundary and unindexed pixel are NEVER bridged. Returns actual
    daughter regions and contact-weighted adjacency, not parent assignments.
    """
    if not np.isfinite(tolerance_deg) or not 0<tolerance_deg<=15:
        raise ValueError("Invalid grain disorientation threshold")
    if not 1<=minimum_pixels<=100000 or not 1<edge_radius_factor<1.4:
        raise ValueError("Invalid segmentation limits")
    coords,good,orientation=_validate_ebsd(ebsd,product_phase_id)
    n=len(coords)
    if n>max_pixels:
        raise ValueError("EBSD map exceeds bounded spatial reconstruction limit")
    tree=cKDTree(coords)
    distances,_=tree.query(coords,k=2)
    nearest=distances[:,1]
    if not np.isfinite(nearest).all() or np.any(nearest<=0):
        raise ValueError("Duplicate or invalid pixel coordinates; cannot infer topology")
    step=float(np.median(nearest))
    radius=step*edge_radius_factor
    raw_edges=tree.query_pairs(radius,output_type="ndarray")
    if len(raw_edges)>max_edges:
        raise ValueError("Too many spatial edges; tighten coordinate/step contract")
    edges=raw_edges[good[raw_edges[:,0]] & good[raw_edges[:,1]]]
    indexed_ids=np.flatnonzero(good)
    quats=np.zeros((n,4),float)
    quats[indexed_ids]=Rotation.from_matrix(orientation[indexed_ids]).as_quat()
    angular=disorientation_edges(quats,edges,product_symmetry)
    links=edges[angular<=tolerance_deg]
    parent=np.arange(n,dtype=int)
    ranks=np.zeros(n,np.uint8)

    def find(i:int)->int:
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=int(parent[i])
        return i

    for i,j in links:
        i=find(int(i)); j=find(int(j))
        if i==j: continue
        if ranks[i]<ranks[j]: i,j=j,i
        parent[j]=i
        if ranks[i]==ranks[j]: ranks[i]+=1
    roots=np.fromiter((find(int(i)) for i in indexed_ids),int,len(indexed_ids))
    _,labels=np.unique(roots,return_inverse=True)
    sizes=np.bincount(labels)
    keep=sizes>=minimum_pixels
    grain_ids=np.full(n,-1,dtype=np.int32)
    candidates=np.flatnonzero(keep)
    if not len(candidates):
        raise ValueError("No daughter regions pass the minimum size")
    remap=np.full(len(sizes),-1,dtype=int)
    remap[candidates]=np.arange(len(candidates))
    grain_ids[indexed_ids]=remap[labels]
    # Mean daughter orientation uses the proper daughter symmetry representative
    # closest to the first pixel, then a normalized quaternion chordal average.
    sym_q=[Rotation.from_matrix(require_so3(np.asarray(s,float),
           tolerance=2e-8,name="daughter symmetry")).as_quat() for s in product_symmetry]
    # Sort a single time: repeated full-map searches are O(pixels * grains)
    # and become unusable for a 720,896-pixel Oxford map.
    kept_pixels=np.flatnonzero(grain_ids>=0)
    ordered=kept_pixels[np.argsort(grain_ids[kept_pixels],kind="stable")]
    group_sizes=np.bincount(grain_ids[kept_pixels],minlength=len(candidates))
    offsets=np.r_[0,np.cumsum(group_sizes)]
    means=[]
    counts=[]
    for g in range(len(candidates)):
        ix=ordered[offsets[g]:offsets[g+1]]
        qs=quats[ix]
        ref=qs[0]
        copies=np.stack([_qmul(qs,s) for s in sym_q],axis=1)
        dots=np.abs(np.einsum("nsi,i->ns",copies,ref))
        aligned=copies[np.arange(len(ix)),np.argmax(dots,axis=1)]
        aligned*=np.where((aligned@ref)<0,-1.,1.)[:,None]
        mean=np.mean(aligned,axis=0)
        mean/=np.linalg.norm(mean)
        means.append(Rotation.from_quat(mean).as_matrix())
        counts.append(len(ix))
    contacts=defaultdict(int)
    for i,j in raw_edges:
        a=int(grain_ids[i]); b=int(grain_ids[j])
        if a>=0 and b>=0 and a!=b:
            contacts[tuple(sorted((a,b)))]+=1
    pair_array=np.asarray(sorted(contacts),dtype=int).reshape(-1,2)
    return {
        "pixel_daughter_grain_id":grain_ids,
        "grain_orientations":np.asarray(means),
        "grain_sizes":np.asarray(counts,int),
        "grain_adjacency":pair_array,
        "boundary_contacts":np.asarray([contacts[tuple(e)] for e in pair_array],int),
        "n_grains":len(means),
        "n_indexed_product_pixels":int(np.sum(good)),
        "n_small_region_pixels_rejected":int(np.sum(good & (grain_ids<0))),
        "topology_step":step,
        "topology_radius":radius,
        "neighbour_edges":len(raw_edges),
        "grain_boundary_edges":len(pair_array),
        "segmentation_scope":"geometric-neighbour connected regions; not indexed grains externally certified",
    }


def _min_parent_angle(a:np.ndarray,b:np.ndarray,syms:np.ndarray)->float:
    """Parent-parent symmetry-reduced geodesic distance (degrees)."""
    v=np.einsum("ij,sjk->sik",b,syms)
    delta=np.einsum("ij,sjk->sik",a.T,v)
    cos=np.clip((np.trace(delta,axis1=1,axis2=2)-1)/2,-1,1)
    return float(np.rad2deg(np.min(np.arccos(cos))))


def reconstruct_spatial_parents(
    segmentation:dict, q_parent_from_daughter:np.ndarray,
    parent_symmetry:tuple[np.ndarray,...], product_symmetry:tuple[np.ndarray,...],
    *, nucleation_tolerance_deg:float=2., growth_tolerance_deg:float=3.,
    nucleus_size:int=3, max_combinations_per_grain:int=20000,
    min_variant_diversity:int=2, ambiguity_margin_deg:float=.4,
) -> dict:
    """Bounded exhaustive triplet/quadruplet nucleation and competitive growth.

    Every seed tests all neighbour subsets until the stated MAX is reached.
    If a seed exceeds the explicit combinatorial limit it is SKIPPED, not
    described as exhausted. A candidate parent must explain each daughter
    through the physical OR and proper symmetries. Multi-solution seeds are
    recorded; no arbitrary first-hit assignment or invented unique parent.
    """
    if nucleus_size not in (3,4):
        raise ValueError("Nucleus must contain 3 or 4 adjacent daughter grains")
    if not all(np.isfinite(v) and 0<v<=20 for v in
               (nucleation_tolerance_deg,growth_tolerance_deg)):
        raise ValueError("Nonfinite or invalid nucleation/growth tolerance")
    if not 1<=max_combinations_per_grain<=200000 or not 1<=min_variant_diversity<=4:
        raise ValueError("Invalid bounded search policy")
    Q=require_so3(np.asarray(q_parent_from_daughter,float),tolerance=2e-8,
                  name="parent<-daughter OR")
    parent_ops=np.asarray([require_so3(np.asarray(s,float),tolerance=2e-8,
                 name="parent symmetry") for s in parent_symmetry])
    child_ops=np.asarray([require_so3(np.asarray(s,float),tolerance=2e-8,
                 name="product symmetry") for s in product_symmetry])
    if not len(parent_ops) or not len(child_ops):
        raise ValueError("Both proper symmetry groups are required")
    orientations=np.asarray(segmentation["grain_orientations"],float)
    sizes=np.asarray(segmentation["grain_sizes"],int)
    n=len(orientations)
    if n<3 or orientations.shape!=(n,3,3) or sizes.shape!=(n,):
        raise ValueError("Invalid daughter grain array")
    for rot in orientations:
        require_so3(rot,tolerance=2e-7,name="daughter grain orientation")
    adjacency=np.asarray(segmentation["grain_adjacency"],int).reshape(-1,2)
    neighbors=[set() for _ in range(n)]
    for i,j in adjacency:
        if not 0<=i<n or not 0<=j<n or i==j:
            raise ValueError("Invalid daughter adjacency")
        neighbors[i].add(int(j));neighbors[j].add(int(i))
    # A daughter's observed right-symmetry copies give *alternative* parent
    # orientations; cubic-parent symmetry is separately quotiented in fits.
    cands=np.einsum("nij,sjk,kl->nsil",orientations,
                    child_ops.transpose(0,2,1),Q.T)
    if cands.shape!=(n,len(child_ops),3,3):
        raise AssertionError("Improper inversion convention")
    def candidate_fit(g:int,ref:np.ndarray)->tuple[float,int]:
        scores=[_min_parent_angle(ref,x,parent_ops) for x in cands[g]]
        idx=int(np.argmin(scores))
        return float(scores[idx]),idx

    nuclei=[];skipped=[];ambiguous_seeds=0
    for seed in range(n):
        adj=sorted(neighbors[seed])
        if len(adj)<nucleus_size-1: continue
        attempts=comb(len(adj),nucleus_size-1)
        if attempts>max_combinations_per_grain:
            skipped.append({"seed":seed,"candidate_subsets":attempts})
            continue
        scored=[]
        for cidx,ref in enumerate(cands[seed]):
            matched=[]
            for other in adj:
                angle,which=candidate_fit(other,ref)
                if angle<=nucleation_tolerance_deg:
                    matched.append((other,angle,which))
            if len(matched)<nucleus_size-1: continue
            # Evaluate expensive crystallographic variant fits only ONCE
            # per seed/reference candidate, not for each graph subset.
            covered=(seed,)+tuple(x[0] for x in matched)
            bulk=daughter_variant_assignment(
                orientations[np.asarray(covered)],ref,Q,
                tuple(parent_ops),tuple(child_ops),
                acceptance_deg=nucleation_tolerance_deg,
                minimum_margin_deg=ambiguity_margin_deg,
            )
            id_by_grain={g:i for i,g in enumerate(covered)}
            all_variants=np.asarray(bulk["variant_index_1based"],int)
            all_accepted=np.asarray(bulk["accepted"],bool)
            all_residuals=np.asarray(bulk["residual_deg"],float)
            for subset in combinations(matched,nucleus_size-1):
                member=(seed,)+tuple(x[0] for x in subset)
                lookup=np.asarray([id_by_grain[g] for g in member],int)
                variants=all_variants[lookup]
                if (not np.all(all_accepted[lookup])
                        or len(set(variants.tolist()))<min_variant_diversity):
                    continue
                residuals=all_residuals[lookup]
                score=float(np.sum(residuals**2))
                scored.append((score,member,cidx,ref,{
                    "variant_index_1based":variants.tolist(),
                    "residual_deg":residuals.tolist(),
                }))
        if not scored:continue
        scored.sort(key=lambda x:(round(x[0],12),x[1],x[2]))
        best=scored[0]
        # Distinct SO3 parent hypotheses, not merely duplicate graph subsets.
        alternatives=[]
        for other in scored[1:]:
            if all(_min_parent_angle(other[3],r,parent_ops)>ambiguity_margin_deg
                   for r in [best[3]]+alternatives):
                alternatives.append(other[3])
        is_ambiguous=bool(alternatives)
        if is_ambiguous: ambiguous_seeds+=1
        nuclei.append({
            "seed":seed,"grain_ids":list(best[1]),"parent_orientation":best[3],
            "score":best[0],"alternative_parent_count":len(alternatives),
            "ambiguous":is_ambiguous,"variant_ids":best[4]["variant_index_1based"],
            "weight":int(np.sum(sizes[list(best[1])])),
        })
    nuclei.sort(key=lambda x:(x["ambiguous"],x["score"],-x["weight"],x["seed"]))
    owners=np.full(n,-1,int)
    parent_models=[]; grown=[];rejected_ambiguous=0
    for nucleus in nuclei:
        # A competing group that already owns the nucleus is not re-seeded.
        ids=nucleus["grain_ids"]
        if any(owners[i]!=-1 for i in ids):
            continue
        if nucleus["ambiguous"]:
            rejected_ambiguous+=1
            continue
        pid=len(parent_models)
        ref=nucleus["parent_orientation"]
        for i in ids:owners[i]=pid
        accepted=set(ids)
        queue=deque(ids)
        while queue:
            i=queue.popleft()
            for j in sorted(neighbors[i]):
                if owners[j]!=-1:continue
                residual,_=candidate_fit(j,ref)
                if residual>growth_tolerance_deg:continue
                # Competing previously accepted parent models may also explain
                # this boundary; leave ambiguous assignments unallocated.
                competitor=min(
                    (candidate_fit(j,model)[0] for model in parent_models),
                    default=float("inf"),
                )
                if competitor-residual<ambiguity_margin_deg:
                    continue
                owners[j]=pid;accepted.add(j);queue.append(j)
        parent_models.append(ref)
        grown.append(sorted(accepted))
    assignments=np.zeros(n,np.int32)
    residuals=np.full(n,np.nan,float)
    ambiguous=np.zeros(n,bool)
    for pid,ref in enumerate(parent_models):
        ix=np.flatnonzero(owners==pid)
        report=daughter_variant_assignment(
            orientations[ix],ref,Q,tuple(parent_ops),tuple(child_ops),
            acceptance_deg=growth_tolerance_deg,
            minimum_margin_deg=ambiguity_margin_deg,
        )
        assignments[ix]=np.asarray(report["variant_index_1based"],int)
        residuals[ix]=np.asarray(report["residual_deg"])
        ambiguous[ix]=np.asarray(report["ambiguous"])
    pixelids=np.asarray(segmentation["pixel_daughter_grain_id"],int)
    pixelowners=np.full(len(pixelids),-1,np.int32)
    valid=pixelids>=0
    pixelowners[valid]=owners[pixelids[valid]]
    return {
        "schema":"cualni.arpge-independent-spatial/0.1",
        "n_daughter_grains":n,
        "n_parent_domains":len(grown),
        "parent_crystal_to_sample":np.asarray(parent_models).reshape(-1,3,3),
        "parent_grain_daughter_ids":grown,
        "daughter_parent_id":owners,
        "pixel_parent_id":pixelowners,
        "daughter_variant_id_1based":assignments,
        "daughter_parent_residual_deg":residuals,
        "daughter_variant_ambiguous":ambiguous,
        "n_unassigned_daughter_grains":int(np.count_nonzero(owners<0)),
        "n_ambiguous_nuclei_skipped":rejected_ambiguous,
        "n_seeds_with_multiple_parent_hypotheses":ambiguous_seeds,
        "n_seeds_skipped_combinatorial_limit":len(skipped),
        "skipped_seed_details":skipped[:20],
        "n_valid_nuclei":len(nuclei),
        "nucleus_size":nucleus_size,
        "status":"independent bounded spatial hypothesis; no legacy parity or physical OR certification",
        "physical_or_certified":False,
    }


def load_h5oina_spatial_grid(filename:str, payload:bytes,
                            *, max_pixels:int=850000):
    """Load every indexed Oxford H5OINA pixel, not a sparse AR sample.

    Up to 850k pixels permits the original 1024x704 Oxford Site8 grid.
    Fails closed on wrong Euler units, missing metadata, or nonfinite
    indexed input; does not silently rotate Oxford CS1 to sample CS0.
    """
    from io import BytesIO
    import h5py
    from .ebsd_h5oina import load_h5oina_sample
    from .ebsd_map import EBSDMap,OrientationConvention
    _,report=load_h5oina_sample(filename,payload,maximum_sample=256)
    with h5py.File(BytesIO(payload),"r") as h:
        root=h["1/EBSD/Data"]
        phase=np.asarray(root["Phase"],int)
        n=len(phase)
        if n>max_pixels:raise ValueError("Full Oxford map exceeds bounded spatial limit")
        raw_euler=np.asarray(root["Euler"],float)
        x=np.asarray(root["X"],float)
        y=np.asarray(root["Y"],float)
        mad=np.asarray(root["Mean Angular Deviation"],float)
        if (raw_euler.shape!=(n,3) or x.shape!=(n,) or y.shape!=(n,)
                or mad.shape!=(n,) or
                str(root["Euler"].attrs.get("Unit"))!="rad"
                or str(root["Mean Angular Deviation"].attrs.get("Unit"))!="rad"):
            raise ValueError("Oxford full-map schema or radians units mismatch")
        good=phase>0
        if not np.isfinite(raw_euler[good]).all():
            raise ValueError("Nonfinite indexed Oxford orientations")
        # Nonindexed Euler placeholders are ignored and replaced by identity.
        raw_euler[~good]=0
        ebsd=EBSDMap.from_eulers(
            raw_euler,phase,x,y,z=np.zeros(n),
            quality={"MAD_deg":np.rad2deg(mad)},
            convention=OrientationConvention.bunge_crystal_to_sample_radians(
                label="Oxford CS1 raw crystal->sample; NO inferred scan/tilt correction"),
            indexed=good,
            metadata={"source_format":"h5oina", "phase_geometry":report["phase_geometry"],
                      "specimen_frame_calibrated":False},
        )
    return ebsd,report
