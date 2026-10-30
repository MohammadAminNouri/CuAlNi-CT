from __future__ import annotations

"""Spatial EBSD analytics and auditable open-format export, not original GUI parity.

These functions deliberately report missing topology / phase evidence rather
than fabricating a physical orientation relationship or assigning a parent.
"""

from collections import deque
from io import StringIO
import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .orientation_kernel import require_so3
from .arpge_spatial import disorientation_edges


def kam_gos(ebsd, segmentation:dict, product_symmetry:tuple[np.ndarray,...],
            *, kam_cutoff_deg:float=5., max_pixels:int=650000)->dict:
    """Per-pixel neighbour kernel average and symmetry-reduced grain spread.

    KAM excludes misorientations ABOVE the explicit cutoff and unindexed,
    phase-inconsistent and small-region pixels. Unknown values are NaNs.
    GOS measures each pixel against its segmented grain mean (not a texture
    spread automatically fitted to a physical parent).
    """
    n=len(ebsd.phase_id)
    labels=np.asarray(segmentation["pixel_daughter_grain_id"],int)
    if n!=len(labels) or n>max_pixels:
        raise ValueError("KAM pixel/segmentation shape mismatch or size limit")
    if not np.isfinite(kam_cutoff_deg) or not 0<kam_cutoff_deg<=30:
        raise ValueError("Invalid KAM cutoff")
    xy=np.column_stack((np.asarray(ebsd.x,float),np.asarray(ebsd.y,float)))
    tree=cKDTree(xy)
    step=segmentation["topology_step"]
    edges=tree.query_pairs(segmentation["topology_radius"],output_type="ndarray")
    accepted=(labels[edges[:,0]]>=0)&(labels[edges[:,1]]>=0)
    # KAM may cross a grain boundary if the angular threshold permits; this
    # follows the neighbour-angle rather than same-grain definition.
    edges=edges[accepted]
    quats=np.zeros((n,4),float)
    valid=np.flatnonzero(labels>=0)
    quats[valid]=Rotation.from_matrix(np.asarray(ebsd.orientations)[valid]).as_quat()
    angles=disorientation_edges(quats,edges,product_symmetry)
    chosen=edges[angles<=kam_cutoff_deg]
    angles=angles[angles<=kam_cutoff_deg]
    sums=np.zeros(n,float); hits=np.zeros(n,int)
    np.add.at(sums,chosen[:,0],angles);np.add.at(sums,chosen[:,1],angles)
    np.add.at(hits,chosen[:,0],1);np.add.at(hits,chosen[:,1],1)
    kam=np.full(n,np.nan,float)
    mask=hits>0
    kam[mask]=sums[mask]/hits[mask]
    means=np.asarray(segmentation["grain_orientations"],float)
    gos=np.full(n,np.nan,float)
    grain_gos=np.full(len(means),np.nan,float)
    for gid in range(len(means)):
        ids=np.flatnonzero(labels==gid)
        if not len(ids):continue
        ref=means[gid]
        qs=Rotation.from_matrix(np.r_[ref[None],np.asarray(ebsd.orientations)[ids]]).as_quat()
        pair=np.column_stack((np.zeros(len(ids),int),np.arange(1,len(ids)+1)))
        angles=disorientation_edges(qs,pair,product_symmetry)
        gos[ids]=angles
        grain_gos[gid]=np.mean(angles)
    return {
        "kam_deg":kam,"gos_pixel_deg":gos,"gos_grain_mean_deg":grain_gos,
        "n_kam_evaluated_pixels":int(np.sum(mask)),
        "kam_cutoff_deg":float(kam_cutoff_deg),
        "method":"2D spatial nearest-neighbour KAM; GOS proper daughter-symmetry reduced",
    }


def cubic_sigma3_domains(
    grain_orientations:np.ndarray, grain_adjacency:np.ndarray,
    coordinates_xy:np.ndarray, *, tolerance_deg:float=2.0, max_order:int=7,
)->dict:
    """Sigma3-related domains and finite-length Sigma3 graph-chain depth.

    Only applicable to PHYSICALLY INDEXED cubic phase and calibrated
    crystal orientations; this method cannot label non-cubic martensite as Cu.
    The order is shortest number of Sigma3 edges, NOT a certified CSL
    sigma value for the composed path.
    """
    g=np.asarray(grain_orientations,float)
    adj=np.asarray(grain_adjacency,int).reshape(-1,2)
    xy=np.asarray(coordinates_xy,float)
    n=len(g)
    if (g.shape!=(n,3,3) or xy.shape!=(n,2) or
            not np.isfinite(g).all() or not np.isfinite(xy).all()):
        raise ValueError("Invalid grain orientation / centroid geometry")
    if n>10000 or len(adj)>50000 or not 0<tolerance_deg<=15 or not 1<=max_order<=7:
        raise ValueError("Sigma3 graph parameters exceed validated limits")
    for rot in g:
        require_so3(rot,tolerance=2e-7,name="cubic grain orientation")
    # CUBIC double equivalence: construct all S1*T_sigma3*S2, unique up to ±q.
    from .parent_or_interop import proper_cubic_parent_operations
    cubic=proper_cubic_parent_operations()
    twinned=[]
    for sx in (-1.,1.):
        for sy in (-1.,1.):
            for sz in (-1.,1.):
                axis=np.asarray([sx,sy,sz])/np.sqrt(3)
                twin=Rotation.from_rotvec(axis*np.pi/3).as_matrix()
                twinned.extend((A@twin@B for A in cubic for B in cubic))
    target=Rotation.from_matrix(np.asarray(twinned)).as_quat()
    # Roundoff/sign equivalence; eliminate redundant crystal operators.
    target=np.unique(np.round(np.where(target[:,3,None]<0,-target,target),10),axis=0)
    observed=Rotation.from_matrix(g).as_quat()
    # relative orientation between two crystal->sample frames
    edges=[]
    for i,j in adj:
        if i<0 or j<0 or i>=n or j>=n or i==j:
            raise ValueError("Invalid Sigma3 grain adjacency")
        relative=g[i].T@g[j]
        q=Rotation.from_matrix(relative).as_quat()
        dots=np.abs(target@q)
        mis=np.rad2deg(2*np.arccos(np.clip(np.max(dots),0,1)))
        if mis<=tolerance_deg:edges.append((int(i),int(j),float(mis)))
    neighbours=[[] for _ in range(n)]
    for i,j,_ in edges:
        neighbours[i].append(j);neighbours[j].append(i)
    domains=np.full(n,-1,int)
    depth=np.full(n,-1,int)
    sizes=[]
    for root in range(n):
        if domains[root]>=0:continue
        domain=len(sizes)
        domains[root]=domain;depth[root]=0
        queue=deque([root]);area=0
        while queue:
            i=queue.popleft();area+=1
            for j in neighbours[i]:
                if domains[j]<0:
                    domains[j]=domain;depth[j]=depth[i]+1;queue.append(j)
        sizes.append(area)
    within=np.where(depth<=max_order,depth,-1)
    # Box count yields a descriptive apparent 2D spatial scaling slope only.
    box_dimensions=[]
    box_levels=[]
    span=np.ptp(xy,axis=0)
    for boxes in (2,4,8,16,32):
        if max(span)<=0:continue
        bins=np.floor((xy-xy.min(axis=0))/(max(span)/boxes+1e-12)).astype(int)
        occupied=len(np.unique(bins,axis=0))
        if 1<occupied<n:
            box_levels.append(np.log(float(boxes)))
            box_dimensions.append(np.log(float(occupied)))
    apparent_fractal=float(np.polyfit(box_levels,box_dimensions,1)[0]) if len(box_levels)>=3 else None
    return {
        "sigma3_edges":np.asarray(edges,float).reshape(-1,3),
        "sigma3_edge_count":len(edges),
        "trd_id":domains,"sigma3_chain_depth":depth,
        "sigma3_chain_depth_within_selected_order":within,
        "n_domains":len(sizes),"domain_grain_counts":sizes,
        "max_order":int(max_order),"apparent_planar_box_count_slope":apparent_fractal,
        "caveat":"Graph Σ3 edge connectivity, not verified Σ3^n disorientation classification, LLC, or diffraction-certified twins",
    }


def stereographic_poles(orientations:np.ndarray, crystal_directions:np.ndarray)->dict:
    """Projected direct crystal directions, preserving antipodal symmetry."""
    g=np.asarray(orientations,float)
    directions=np.asarray(crystal_directions,float)
    if g.ndim!=3 or g.shape[1:]!=(3,3) or directions.ndim!=2 or directions.shape[1]!=3:
        raise ValueError("Pole figure requires (N,3,3) orientations and (K,3) directions")
    if np.any(np.linalg.norm(directions,axis=1)<=0) or not np.isfinite(directions).all():
        raise ValueError("Invalid pole direction")
    d=directions/np.linalg.norm(directions,axis=1)[:,None]
    sample=np.einsum("nij,kj->nki",g,d)
    sample=np.where(sample[:,:,2,None]<0,-sample,sample)
    xy=sample[:,:,:2]/(1+sample[:,:,2,None])
    return {"stereographic_xy":xy,
            "convention":"direct crystal directions mapped to upper specimen hemisphere; no PF density inference"}


def habit_plane_trace(hkl:np.ndarray, lattice, crystal_to_sample:np.ndarray,
                      sample_surface_normal=(0.,0.,1.))->dict:
    """Physical reciprocal-plane normal intersected with calibrated sample surface."""
    hkl=np.asarray(hkl,float).reshape(3)
    g=require_so3(np.asarray(crystal_to_sample,float),tolerance=2e-8,
                  name="crystal-to-specimen orientation")
    normal=np.asarray(sample_surface_normal,float).reshape(3)
    if not np.isfinite(hkl).all() or not np.any(hkl):
        raise ValueError("Nonzero finite (hkl) required")
    if not np.isfinite(normal).all() or np.linalg.norm(normal)==0:
        raise ValueError("Nonzero specimen surface normal required")
    normal/=np.linalg.norm(normal)
    crystal_normal=np.linalg.solve(lattice.structure_matrix().T,hkl)
    crystal_normal/=np.linalg.norm(crystal_normal)
    specimen_normal=g@crystal_normal
    trace=np.cross(specimen_normal,normal)
    length=np.linalg.norm(trace)
    return {"sample_habit_normal":specimen_normal.tolist(),
            "sample_trace_direction":(trace/length).tolist() if length>1e-10 else None,
            "degenerate_parallel_surface":bool(length<=1e-10),
            "note":"Geometrical trace, not measured interface or diffraction proof"}


def export_parent_ebsd_text(ebsd, pixel_parent_id:np.ndarray,
                            parent_rotations:np.ndarray, *,
                            format:str="ctf", parent_phase_name:str="parent hypothetical",
                            parent_lattice=None)->str:
    """Write conservative research CTF or square-grid ANG with explicit phase.

    AROUND-TRIP TEST REQUIRED. Not an Oxford/EDAX instrument-native converter.
    All unassigned pixels receive Phase=0. Coordinates and raw specimen frame
    remain unchanged; non-square/hex grids require a dedicated validated adapter.
    """
    ids=np.asarray(pixel_parent_id,int)
    g=np.asarray(parent_rotations,float)
    n=len(ebsd.phase_id)
    if len(ids)!=n or g.ndim!=3 or g.shape[1:]!=(3,3):
        raise ValueError("Parent-pixel and parent-orientation dimensions mismatch")
    if parent_lattice is None:
        raise ValueError("Explicit parent lattice and phase metadata are required")
    if np.any(ids>=len(g)) or np.any(ids< -1):
        raise ValueError("Unrecognized parent-domain ID")
    for a in g:require_so3(a,tolerance=2e-8,name="reconstructed parent")
    x=np.asarray(ebsd.x,float);y=np.asarray(ebsd.y,float)
    ux=np.unique(x);uy=np.unique(y)
    if len(ux)*len(uy)!=n or len(ux)<2 or len(uy)<2:
        raise ValueError("Native text export currently requires a complete square grid")
    dx=np.diff(ux);dy=np.diff(uy)
    if (np.max(np.abs(dx-np.median(dx)))>1e-4 or
            np.max(np.abs(dy-np.median(dy)))>1e-4):
        raise ValueError("Grid step is irregular")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite EBSD coordinates")
    valid=ids>=0
    ori=np.repeat(np.eye(3)[None],n,axis=0)
    ori[valid]=g[ids[valid]]
    angles_deg=Rotation.from_matrix(ori).as_euler("ZXZ",degrees=True)
    from .lattice import Lattice
    if not isinstance(parent_lattice,Lattice):
        raise ValueError("Use a parent Lattice, not invented source phase data")
    result=StringIO()
    if format=="ctf":
        result.write("Channel Text File\n")
        result.write(f"Prj\tReconstructed computational hypothesis\n")
        result.write(f"Author\tCuAlNi-CT research export\n")
        result.write(f"JobMode\tGrid\nXCells\t{len(ux)}\nYCells\t{len(uy)}\n")
        result.write(f"XStep\t{np.median(dx):.10g}\nYStep\t{np.median(dy):.10g}\n")
        result.write("AcqE1\t0\nAcqE2\t0\nAcqE3\t0\n")
        result.write("Phase X Y Bands Error Euler1 Euler2 Euler3 MAD BC BS\n")
        for i in range(n):
            result.write(f"{int(valid[i])} {x[i]:.9g} {y[i]:.9g} 0 0 "+
                         " ".join(f"{a:.10f}" for a in angles_deg[i])+" 0 0 0\n")
    elif format=="ang":
        result.write("# GRID: SqrGrid\n")
        result.write(f"# XSTEP: {np.median(dx):.10g}\n# YSTEP: {np.median(dy):.10g}\n")
        result.write(f"# NCOLS_ODD: {len(ux)}\n# NCOLS_EVEN: {len(ux)}\n# NROWS: {len(uy)}\n")
        result.write("# SOURCE: Computed candidate, no vendor/experimental certification\n")
        for i in range(n):
            phi=np.deg2rad(angles_deg[i])
            result.write(" ".join(f"{v:.12g}" for v in phi)+
                f" {x[i]:.9g} {y[i]:.9g} 0 0 {int(valid[i])} 0 0\n")
    else:
        raise ValueError("Export supports only research .ctf or .ang")
    return result.getvalue()
