from __future__ import annotations

"""Direct A/M OR candidate classes from *indexed* adjacent phase pairs.

This route is fundamentally different from child-only blind OR: when a
parent grain is indexed, R_A<-M = g_A.T g_M for a co-located A/M interface.
The physical OR is considered modulo G_A x G_M, and input phase-indexing
quality and the actual interfacial correspondence remain separate evidence.
"""
import numpy as np
from cualni_cryst.ebsd_map import EBSDPhase
from cualni_cryst.ebsd_theory_bridge import orientation_relationship_distance_deg
from cualni_cryst.lattice import Lattice
from cualni_cryst.orientation_kernel import require_so3


def observed_am_or_classes(
    measured: dict, declared_phases: tuple[dict,...], *,
    parent_phase_id: int, product_phase_id: int,
    tolerance_deg: float = 3.0,
    maximum_boundaries: int = 32,
) -> dict:
    if not 0<tolerance_deg<15:
        raise ValueError("OR clustering tolerance must be between 0 and 15 degrees")
    by_id={p["phase_id"]:p for p in declared_phases}
    if (parent_phase_id not in by_id or product_phase_id not in by_id
        or parent_phase_id==product_phase_id):
        raise ValueError("Different CTF-declared parent/product phases required")
    by_grain={int(g["grain_id"]):g for g in measured["grains"]}
    pairs=[]
    for e in measured["adjacent_grain_pairs"]:
        ga,gb=by_grain.get(int(e["grain_a"])),by_grain.get(int(e["grain_b"]))
        if ga is None or gb is None:continue
        roles={int(ga["phase_id"]),int(gb["phase_id"])}
        if roles!={int(parent_phase_id),int(product_phase_id)}:continue
        A=ga if int(ga["phase_id"])==parent_phase_id else gb
        M=gb if A is ga else ga
        pairs.append((A,M,int(e["pixel_neighbor_edges"])))
    pairs.sort(key=lambda t:-t[2])
    pairs=pairs[:maximum_boundaries]
    def phase(pid):
        p=by_id[pid]
        return EBSDPhase.from_point_group(
            pid,p["name"],
            Lattice(*p["lengths_angstrom"],*p["angles_deg"],
                    length_unit="angstrom"),
            p["declared_point_group"],
        )
    if len(pairs)<3:
        return {
            "status":"INSUFFICIENT_INDEXED_A_M_INTERFACES",
            "n_usable_A_M_adjacencies":len(pairs),
            "representative_classes":[],
            "unique_physical_rotation":False,
            "C":"NOT_INFERRED",
            "scope":"Requires at least three independently indexed A/M region adjacencies.",
        }
    parent=phase(parent_phase_id)
    product=phase(product_phase_id)
    clusters=[]
    for ga,gm,w in pairs:
        gA=require_so3(np.asarray(
            ga["mean_crystal_to_sample_rotation"],float
        ),tolerance=2e-8,name="indexed parent orientation")
        gM=require_so3(np.asarray(
            gm["mean_crystal_to_sample_rotation"],float
        ),tolerance=2e-8,name="indexed product orientation")
        R=require_so3(gA.T@gM,tolerance=2e-8,name="observed A/M OR")
        for cluster in clusters:
            d=orientation_relationship_distance_deg(
                R,cluster["R"],parent.proper_symmetry_cartesian,
                product.proper_symmetry_cartesian,
            )
            if d<=tolerance_deg:
                cluster["pairs"]+=1
                cluster["weights"]+=w
                cluster["residuals"].append(d)
                break
        else:
            clusters.append({"R":R,"pairs":1,"weights":w,"residuals":[0.]})
    clusters.sort(key=lambda c:(-c["pairs"],-c["weights"]))
    top=clusters[0]
    dominance=top["pairs"]/len(pairs)
    # A dominant crystallographic OR equivalence class is not a measured
    # unique representative rotation, and not the lattice correspondence C.
    single_class=bool(
        len(pairs)>=3 and top["pairs"]>=3 and dominance>=.75
    )
    return {
        "status":(
            "DOMINANT_OBSERVED_OR_EQUIVALENCE_CLASS"
            if single_class else "MULTIPLE_AMBIGUOUS_OR_CLASSES"
        ),
        "n_usable_A_M_adjacencies":len(pairs),
        "dominant_class_support_fraction":dominance,
        "unique_crystallographic_OR_equivalence_class_observed":single_class,
        "unique_physical_rotation":False,
        "C":"NOT_INFERRED",
        "representative_classes":[{
            "R_parent_from_product_representative":c["R"].tolist(),
            "support_count":c["pairs"],
            "boundary_pixel_edges":c["weights"],
            "median_symmetry_reduced_residual_deg":float(np.median(c["residuals"])),
        } for c in clusters],
        "scope":(
            "Measured indexed parent/product grain orientations at adjacent "
            "pixels, quotient by both proper point groups. Requires valid "
            "parent/product indexing and actual A/M interfaces. "
            "Symmetry-related OR representatives are not physically unique; "
            "no atom mapping or correspondence C is inferred."
        ),
    }
