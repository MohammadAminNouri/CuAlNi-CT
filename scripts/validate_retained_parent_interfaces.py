#!/usr/bin/env python3
"""Frozen-source-OR independent real EBSD interface validation.

Usage:
 python scripts/validate_retained_parent_interfaces.py ARPGE_Sept2021.zip \
   --output frozen_parent_interface.json

This is an INTERFACE test, not a reconstructed prior-parent map.
One cannot presume neighbouring retained beta/gamma and daughter grains
belong to the same original parent. Parent orientations are NEVER used to
fit, choose or adjust the frozen legacy OR or the predeclared quality gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from scipy.ndimage import label
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from cualni_cryst.arpge_archive import inspect_arpge_archive,audit_legacy_orientation_presets
from cualni_cryst.ebsd_io import load_ang,load_ctf
from cualni_cryst.ebsd_map import EBSDPhase
from cualni_cryst.lattice import Lattice
from cualni_cryst.experimental_parent_validation import frozen_or_boundary_audit

SOURCE_SHA256="7b1df4abf1ceac278e64d4cf2ab6dda4e5ab8efab3c80fbed4707be9b789e873"


def run(archive_path:Path,output_path:Path)->dict:
    if not archive_path.is_file():
        raise ValueError("Original ARPGE ZIP must be available locally")
    digest=hashlib.sha256()
    with archive_path.open("rb") as reader:
        for chunk in iter(lambda:reader.read(1<<20),b""):
            digest.update(chunk)
    if digest.hexdigest()!=SOURCE_SHA256:
        raise ValueError("Unknown archive bytes; refuse to reuse the frozen 2021 reference ORs")
    archive=inspect_arpge_archive(str(archive_path))
    geometry=audit_legacy_orientation_presets(archive)
    report={
        "schema":"cualni.arpge-2021-real-cross-phase/1",
        "source_archive_sha256":SOURCE_SHA256,
        "source_frozen_or_without_measured_parent_orientations":True,
        "tested":{},
        "absolute_specimen_frame_independently_calibrated":False,
        "parent_grain_topology_certified":False,
        "scientific_validation_status":"local measured cross-phase OR evidence, NOT successful prior-parent grain reconstruction",
    }
    with ZipFile(archive_path) as z,TemporaryDirectory() as folder:
        names=z.namelist()
        for name,suffix,kind,parent_id,daughter_id,step,pa_cell,ma_cell,pa_pg,ma_pg in (
            ("Ti64","Ti-6Al-V-AdditiveManuf.ang","Burgers",1,2,.201,
             Lattice.cubic(3.31),Lattice(2.95,2.95,4.68,90,90,120),
             "m-3m","6/mmm"),
            ("Bainitic steel","BainiticSteel.ctf","Fe-GT",2,1,.101,
             Lattice.cubic(3.6599),Lattice.cubic(2.866),
             "m-3m","m-3m"),
        ):
            member=next((x for x in names if x.endswith(suffix)),None)
            if member is None:
                raise ValueError(f"Missing original EBSD dataset {suffix}")
            outfile=Path(folder)/suffix
            outfile.write_bytes(z.read(member))
            ebsd=load_ang(outfile) if suffix.endswith(".ang") else load_ctf(outfile)
            candidate=next((x for x in geometry["results"] if x["name"]==kind),None)
            if candidate is None or candidate["status"]!="geometric candidate":
                raise ValueError(f"Frozen source OR {kind} failed geometry audit")
            Q=np.asarray(candidate["Q_parent_from_daughter"],float)
            gp=tuple(EBSDPhase.from_point_group(1,"measured parent",pa_cell,pa_pg).
                     proper_symmetry_cartesian)
            gm=tuple(EBSDPhase.from_point_group(2,"measured daughter",ma_cell,ma_pg).
                     proper_symmetry_cartesian)
            q=ebsd.quality
            if suffix.endswith(".ang"):
                good=(np.asarray(q["IQ"])>=55)&(np.asarray(q["Fit"])<=2.1)
                gate="IQ >= 55 and Fit <= 2.1, predeclared indexer-quality sensitivity"
                nx,ny=521,506
            else:
                good=(np.asarray(q["BANDS"])>=5)&(np.asarray(q["ERROR"])==0)&(np.asarray(q["MAD"])<=1.5)
                gate="Bands >= 5, Error == 0, MAD <= 1.5 degrees, predeclared quality sensitivity"
                nx,ny=398,278
            if len(ebsd.phase_id)!=nx*ny:
                raise ValueError("Unexpected source scan raster shape")
            # Retained parent phase labels may define local connected islands;
            # NOT a certified prior-parent grain or independent grain boundary.
            mask=(ebsd.phase_id==parent_id).reshape(ny,nx)
            islands,n=label(mask,np.asarray([[0,1,0],[1,1,1],[0,1,0]],bool))
            other=Q@Rotation.from_euler("x",15,degrees=True).as_matrix()
            trial={}
            for group,keep in (("all_indexed",None),("quality_gate",good)):
                args=dict(
                    orientations=ebsd.orientations,
                    phase_ids=ebsd.phase_id,
                    xy=np.column_stack((ebsd.x,ebsd.y)),
                    parent_phase_id=parent_id,daughter_phase_id=daughter_id,
                    neighbor_distance=step,
                    frozen_q=Q,parent_proper=gp,daughter_proper=gm,
                    quality_mask=keep,
                    pixel_group=islands.ravel(),
                    maximum_sample=6000,seed=20261030,acceptance_deg=3.,
                )
                correct=frozen_or_boundary_audit(negative_q=Q.T,**args)
                shifted=frozen_or_boundary_audit(negative_q=other,**args)
                trial[group]={
                    "source_contacts":correct["n_source_contacts_quality_filtered"],
                    "sampled_contacts":correct["n_tested_contacts"],
                    "frozen_or_median_deg":correct["correct_OR_median_deg"],
                    "frozen_or_within_3deg":correct["correct_OR_fraction_within_threshold"],
                    "transposed_or_median_deg":correct["negative_OR_median_deg"],
                    "transposed_or_within_3deg":correct["negative_OR_fraction_within_threshold"],
                    "offset_15deg_or_median_deg":shifted["negative_OR_median_deg"],
                    "offset_15deg_or_within_3deg":shifted["negative_OR_fraction_within_threshold"],
                    "independent_parent_grain_matching_confirmed":False,
                }
            report["tested"][name]={
                "source_map":member,"frozen_original_2021_transformation":kind,
                "n_source_pixels":len(ebsd.phase_id),
                "n_parent_phase_pixels":int(np.count_nonzero(mask)),
                "n_retained_parent_phase_islands":int(n),
                "quality_policy":gate,
                "results":trial,
                "physical_prior_parent_or_certified":False,
            }
    output_path.parent.mkdir(exist_ok=True,parents=True)
    output_path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive",type=Path)
    cli.add_argument("--output",type=Path,default=Path("arpge_real_frozen_interface.json"))
    a=cli.parse_args()
    report=run(a.archive,a.output)
    print(json.dumps(report,indent=2))
