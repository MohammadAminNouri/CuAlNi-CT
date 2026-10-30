from __future__ import annotations

"""Native full-resolution Oxford 18R -> conditional beta orientation HDF5.

This is a fully specified, bounded per-pixel application of an INPUT reference
OR and an INPUT candidate beta parent. It does NOT call MTEX calcVariantGraph,
perform grain-boundary Markov clustering or identify an independent physical OR.

Proper orientation convention is crystal Cartesian -> raw Oxford CS1 sample.
"""

from io import BytesIO
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

from .ebsd_h5oina import load_h5oina_sample
from .orientation_kernel import require_so3
from .parent_variant_audit import parent_generated_variants


def reconstruct_conditional_parent_h5oina(
    filename: str, data: bytes,
    parent_crystal_to_raw_sample: object,
    q_parent_from_daughter: object,
    parent_proper_symmetries: tuple[np.ndarray, ...],
    daughter_proper_symmetries: tuple[np.ndarray, ...],
    *,
    max_mad_deg: float = 1.5,
    max_fit_deg: float = 3.0,
    min_variant_patch_pixels: int = 50,
    chunk_pixels: int = 8192,
) -> tuple[bytes, dict]:
    """Return (HDF5 bytes, diagnostics), retaining unassigned pixels as NaNs.

    A 'variant-consistent patch' is four-neighbour connected, has the same
    theoretical daughter variant ID and exceeds min pixels. It is NOT a
    MTEX grain, and passing a patch never certifies a physical beta parent.
    """
    if not 0 < max_mad_deg <= 10 or not 0 < max_fit_deg <= 20:
        raise ValueError("Invalid MAD or fit threshold")
    if not 1 <= min_variant_patch_pixels <= 10000:
        raise ValueError("Invalid variant-patch minimum")
    if not 64 <= chunk_pixels <= 65536:
        raise ValueError("Chunk pixels must be within 64..65536")
    # Audit the vendor format and phase metadata before touching the whole map.
    _, scan = load_h5oina_sample(filename, data, maximum_sample=256)
    if len(scan["phase_geometry"]) != 1:
        raise ValueError("Full-map reconstruction requires one indexed daughter phase")
    Q = require_so3(np.asarray(q_parent_from_daughter,float),
                    tolerance=2e-8, name="parent<-18R Q")
    gA = require_so3(np.asarray(parent_crystal_to_raw_sample,float),
                     tolerance=2e-8, name="candidate beta orientation")
    p = tuple(require_so3(np.asarray(x,float),tolerance=2e-8,name="beta symmetry")
              for x in parent_proper_symmetries)
    m = tuple(require_so3(np.asarray(x,float),tolerance=2e-8,name="18R symmetry")
              for x in daughter_proper_symmetries)
    if not p or not m:
        raise ValueError("Proper parent and daughter symmetry must be supplied")
    variants=np.asarray(parent_generated_variants(Q,p,m))
    predicted=np.einsum("ij,vjk->vik",gA,variants)
    predicted_copies=np.einsum("vij,sjk->vsik",predicted,np.asarray(m))
    target_quats=Rotation.from_matrix(predicted_copies.reshape(-1,3,3)).as_quat()
    output_quats=None
    parent_errors=None
    assigned_variant=None
    indexed_quality=None
    raw_mad=None
    with h5py.File(BytesIO(data),"r") as h:
        header=h["1/EBSD/Header"]
        n_x=int(np.asarray(header["X Cells"]).reshape(-1)[0])
        n_y=int(np.asarray(header["Y Cells"]).reshape(-1)[0])
        n=n_x*n_y
        if n > 1_500_000:
            raise ValueError("Interactive full-map limit is 1,500,000 pixels")
        d=h["1/EBSD/Data"]
        if str(d["Euler"].attrs.get("Unit")) != "rad" or str(
            d["Mean Angular Deviation"].attrs.get("Unit")
        ) != "rad":
            raise ValueError("H5OINA Euler and MAD must have Unit=rad")
        phase=np.asarray(d["Phase"],dtype=np.int32)
        mad=np.asarray(d["Mean Angular Deviation"],dtype=float)
        if phase.shape!=(n,) or mad.shape!=(n,):
            raise ValueError("Corrupt H5OINA pixel shapes")
        good=(phase>0) & np.isfinite(mad) & (mad<=np.deg2rad(max_mad_deg))
        indexed_quality=good.copy()
        raw_mad=mad.astype(np.float32)
        parent_errors=np.full(n,np.nan,dtype=np.float32)
        assigned_variant=np.zeros(n,dtype=np.uint16)
        # Accepted-only quaternions, with NaN everywhere else. xyzw order.
        output_quats=np.full((n,4),np.nan,dtype=np.float32)
        ids=np.flatnonzero(good)
        for start in range(0,len(ids),chunk_pixels):
            ix=ids[start:start+chunk_pixels]
            eulers=np.asarray(d["Euler"][ix],float)
            if not np.isfinite(eulers).all():
                raise ValueError("Indexed H5OINA Euler values are nonfinite")
            gM=Rotation.from_euler("ZXZ",eulers).as_matrix()
            quat=Rotation.from_matrix(gM).as_quat()
            dot=np.abs(quat@target_quats.T)
            choices=np.argmax(dot,axis=1)
            mis=np.rad2deg(2*np.arccos(np.clip(
                dot[np.arange(len(ix)),choices],0,1
            )))
            parent_errors[ix]=mis.astype(np.float32)
            passed=mis<=max_fit_deg
            if not np.any(passed):
                continue
            chosen_ix=ix[passed]
            choices=choices[passed]
            v=choices//len(m)
            sm=choices%len(m)
            assigned_variant[chosen_ix]=(v+1).astype(np.uint16)
            # Reverse the *assigned* proper daughter indexer copy.
            gM_ok=gM[passed]
            recovered=(gM_ok @ np.transpose(np.asarray(m)[sm],(0,2,1))
                       @ np.transpose(variants[v],(0,2,1)))
            output_quats[chosen_ix]=Rotation.from_matrix(recovered).as_quat().astype(np.float32)

    # Conservative FOUR-NEIGHBOR spatial islands. This is not an MTEX grain
    # segmentation: it only suppresses tiny local label islands.
    labels=assigned_variant.reshape(n_y,n_x)
    patch_mask=np.zeros((n_y,n_x),dtype=bool)
    structure=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)
    for variant_id in np.unique(labels[labels>0]):
        components,_=ndimage.label(labels==variant_id,structure=structure)
        sizes=np.bincount(components.reshape(-1))
        sizes[0]=0
        patch_mask |= sizes[components]>=min_variant_patch_pixels
    report={
        "schema":"cualni.h5oina-parent-conditional/1.0",
        "source_filename":Path(filename).name,
        "n_x":n_x,"n_y":n_y,"n_pixels":n,
        "n_indexed":int(scan["indexed_points"]),
        "n_quality":int(np.count_nonzero(indexed_quality)),
        "n_fitted_within_deg":int(np.count_nonzero(assigned_variant)),
        "n_variant_patch_supported":int(np.count_nonzero(patch_mask)),
        "fraction_quality_fitted":float(
            np.count_nonzero(assigned_variant)/max(np.count_nonzero(indexed_quality),1)),
        "fraction_quality_patch_supported":float(
            np.count_nonzero(patch_mask)/max(np.count_nonzero(indexed_quality),1)),
        "parent_variant_count":int(len(variants)),
        "proper_child_symmetry_count":int(len(m)),
        "indexed_phase_metadata":scan["phase_geometry"],
        "max_MAD_deg":float(max_mad_deg),
        "max_parent_fit_deg":float(max_fit_deg),
        "min_variant_patch_pixels":int(min_variant_patch_pixels),
        "orientation_convention":"crystal Cartesian->raw Oxford CS1; no implicit scan-frame correction",
        "physical_parent_certified":False,
        "MTEX_graph_or_MCL_performed":False,
        "warning":"Conditional fixed-Q fixed-parent per-pixel variants; four-neighbour patch filter is not MTEX grain clustering",
    }
    output_buffer=BytesIO()
    with h5py.File(output_buffer,"w") as out:
        out.attrs["report_json"]=json.dumps(report,allow_nan=False)
        out.attrs["status"]="unanchored conditional beta parent; NOT physical OR proof"
        inputs=out.create_group("inputs")
        inputs.create_dataset("Q_parent_from_18R",data=Q)
        inputs.create_dataset("g_parent_to_raw_oxford_CS1",data=gA)
        grp=out.create_group("map")
        shape=(n_y,n_x)
        grp.create_dataset("parent_quat_xyzw",data=output_quats.reshape(*shape,4),
                           compression="gzip",shuffle=True)
        grp.create_dataset("parent_fit_deg",data=parent_errors.reshape(shape),
                           compression="gzip",shuffle=True)
        grp.create_dataset("assigned_variant_id_1based",data=labels,
                           compression="gzip",shuffle=True)
        grp.create_dataset("indexed_MAD_deg",data=np.rad2deg(raw_mad).reshape(shape),
                           compression="gzip",shuffle=True)
        grp.create_dataset("quality_mask",data=indexed_quality.reshape(shape),
                           compression="gzip",shuffle=True)
        grp.create_dataset("variant_patch_support_mask",data=patch_mask,
                           compression="gzip",shuffle=True)
    return output_buffer.getvalue(),report
