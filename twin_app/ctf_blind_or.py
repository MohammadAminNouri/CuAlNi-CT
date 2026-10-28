from __future__ import annotations

"""Exploratory *observable* parent-operator embedding from martensite-only EBSD.

Cayron 2019 distinguishes orientation variants from correspondence variants.
Child/child EBSD observes R.T G_parent R, which is invariant under the SO(3)
normalizer of the parent symmetry. Even an identified observable embedding
is not a uniquely reconstructed physical parent↔product OR matrix. The
correspondence C is not identified by this optimization.
"""
import numpy as np


def explore_blind_or(measured: dict, declared: tuple[dict, ...], *,
                     parent_phase_id: int, product_phase_id: int,
                     maximum_adjacencies: int = 96,
                     global_samples: int = 256) -> dict:
    from cualni_cryst.ebsd_blind_discovery import (
        BlindORSettings,discover_orientation_relationship_blind,
    )
    from cualni_cryst.ebsd_map import EBSDPhase
    from cualni_cryst.lattice import Lattice
    by_id={p["phase_id"]:p for p in declared}
    if parent_phase_id not in by_id or product_phase_id not in by_id:
        raise ValueError("Both parent and product cells must be CTF-declared")
    if parent_phase_id==product_phase_id:
        raise ValueError("Parent and product must differ")
    if maximum_adjacencies<4:
        raise ValueError("Need at least four candidate boundary adjacencies")
    grains=sorted(
        (g for g in measured["grains"] if int(g["phase_id"])==product_phase_id),
        key=lambda g:int(g["grain_id"]),
    )
    mapping={int(g["grain_id"]):i for i,g in enumerate(grains)}
    pairs=sorted(
        (
            e for e in measured["adjacent_grain_pairs"]
            if int(e["grain_a"]) in mapping and int(e["grain_b"]) in mapping
        ),key=lambda e:(-int(e["pixel_neighbor_edges"]),
                         int(e["grain_a"]),int(e["grain_b"])),
    )[:maximum_adjacencies]
    if len(pairs)<4:
        return {
            "status":"insufficient_evidence",
            "n_product_regions":len(grains),
            "n_eligible_adjacencies":len(pairs),
            "unique_physical_OR":False,
            "correspondence_C":"NOT_INFERRED",
            "message":"At least four product-region adjacencies required.",
        }
    orientations=np.asarray([
        g["mean_crystal_to_sample_rotation"] for g in grains
    ],dtype=float)
    adj=tuple((mapping[int(e["grain_a"])],mapping[int(e["grain_b"])])
              for e in pairs)
    def phase(pid):
        p=by_id[pid]
        return EBSDPhase.from_point_group(
            pid,p["name"],
            Lattice(*p["lengths_angstrom"],*p["angles_deg"],
                    length_unit="angstrom"),
            p["declared_point_group"],
        )
    settings=BlindORSettings(
        global_samples=global_samples,
        coarse_keep=16, exact_keep=8,
        local_max_iterations=40,
    )
    result=discover_orientation_relationship_blind(
        orientations,adj,phase(parent_phase_id),
        phase(product_phase_id),settings=settings,
    )
    def record(c):
        if c is None:return None
        return {
            "R_parent_from_product_representative":
                c.R_parent_from_product.tolist(),
            "objective":float(c.objective),
            "median_residual_deg":float(c.median_residual_deg),
            "p90_residual_deg":float(c.p90_residual_deg),
            "support_fraction":float(c.support_fraction),
            "n_operator_classes":int(c.n_operator_classes),
            "accepted_boundary_fraction":float(c.accepted_boundary_fraction),
        }
    return {
        "status":result.status,
        "identification_target":result.identification_target,
        "unique_physical_OR":False,
        "correspondence_C":"NOT_INFERRED",
        "n_product_regions":len(grains),
        "n_eligible_adjacencies":len(pairs),
        "n_global_samples":result.n_global_samples,
        "best_observable_representative":record(result.best),
        "alternatives":[record(c) for c in result.alternatives],
        "message":result.evidence_note,
        "scientific_limit":(
            "Child-only variants constrain the conjugate parent symmetry "
            "subgroup in the product frame. Its normalizer gauge prevents "
            "a unique physical OR. No candidate is automatically accepted "
            "as a parent orientation or lattice correspondence C."
        ),
    }
