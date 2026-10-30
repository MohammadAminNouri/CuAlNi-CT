from __future__ import annotations

"""Two independent published NiTi benchmark layers; no uploaded NiTi EBSD.

Layer 1: published 2020 Cayron OR AQ geometry, synthetic noisy measured
  B19' variants, known synthetic B2 parent and TWO known sample-direction
  anchors. Test forward, inverse and blind parent consensus.
Layer 2: published 2026 NiTi B2/B19' correspondence C and metrics, exact
  groupoid class numbers and independently computed singular stretches.
These two source data sets must never be conflated as one experiment.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .lattice import Lattice
from .ebsd_map import EBSDPhase
from .legacy_suite_bridge import Phase
from .ar_gen_cry_pipeline import cry_stage, gen_stage
from .niti_literature_or import niti_published_or, direction_cartesian
from .parent_or_interop import proper_cubic_parent_operations
from .parent_replay_exchange import forward_predict_daughter, forward_inverse_roundtrip
from .parent_consensus_complete import reconstruct_parent_full_variant_consensus
from .parent_or_evidence import (
    parent_orientation_from_two_directions, compare_parent_to_anchor,
)


def evaluate_niti_benchmarks() -> dict:
    a0 = 3.015
    a, b, c, beta = 2.889, 4.120, 4.622, 96.8
    parent_cell = Lattice.cubic(a0)
    daughter_cell = Lattice.monoclinic_unique_b(a,b,c,beta)
    parent_ops = tuple(proper_cubic_parent_operations())
    daughter_ops = tuple(EBSDPhase.from_point_group(
        1,"B19-prime",daughter_cell,"2/m",
    ).proper_symmetry_cartesian)
    AQ=niti_published_or("AQ",parent=parent_cell,daughter=daughter_cell)
    C=niti_published_or("C",parent=parent_cell,daughter=daughter_cell)
    Q=np.asarray(AQ["Q_parent_from_daughter"])
    gA_true=Rotation.from_euler("xyz",[18,-24,35],degrees=True).as_matrix()
    variant=forward_predict_daughter(gA_true,Q,parent_ops,daughter_ops)
    rng=np.random.default_rng(20261030)
    reps=variant["daughter_symmetry_representatives"]
    chosen=np.array([reps[i,i%2] for i in range(variant["n_variants"])])
    measurement_noise=Rotation.from_rotvec(
        rng.normal(size=(len(chosen),3)) * np.deg2rad(.12)
    ).as_matrix()
    observed=measurement_noise@chosen
    # Known synthetic B2 axes as if independently observed in a common
    # specimen frame. They are generated and cannot certify actual specimens.
    uvw1,uvw2=(1,0,0),(0,1,1)
    sample1=gA_true@direction_cartesian(parent_cell,uvw1)
    sample2=gA_true@direction_cartesian(parent_cell,uvw2)
    gA_anchor=parent_orientation_from_two_directions(
        parent_cell,uvw1,uvw2,sample1,sample2,
    )
    anchor_error=compare_parent_to_anchor(gA_anchor,gA_true,parent_ops)
    with_anchor=forward_inverse_roundtrip(
        gA_anchor,Q,parent_ops,daughter_ops,
        observed_daughter_g=observed,accepted_deg=1,
    )
    wrong_Q=forward_inverse_roundtrip(
        gA_anchor,np.asarray(C["Q_parent_from_daughter"]),
        parent_ops,daughter_ops,
        observed_daughter_g=observed,accepted_deg=1,
    )
    # Crucial: "blind" consensus must NOT receive the generated parent.
    inferred=reconstruct_parent_full_variant_consensus(
        observed,Q,parent_ops,daughter_ops,
        tolerance_deg=1.5,max_seeds=512,max_iterations=20,
    )
    blind_error=compare_parent_to_anchor(
        np.asarray(inferred["parent_crystal_to_sample_rotation"]),gA_true,
        parent_ops,
    )
    wrong_anchor=Rotation.from_euler("x",17,degrees=True).as_matrix()@gA_anchor
    wrong_anchor_error=compare_parent_to_anchor(wrong_anchor,gA_true,parent_ops)

    p=Phase("NiTi B2","m-3m",3.010,3.010,3.010)
    m=Phase("NiTi B19-prime","2/m",2.898,4.108,4.646,90,97.78,90)
    model=cry_stage(p,m,(1,0,0),(0,1,0))
    C_direct="0 0 1\n1/2 1/2 0\n-1/2 1/2 0"
    groupoid=gen_stage(model,C_direct)
    F=m.lattice().structure_matrix()@np.array(
        [[0.,0.,1.],[.5,.5,0.],[-.5,.5,0.]]
    )/p.a
    stretches=np.sort(np.linalg.svd(F)[1])
    return {
        "schema":"cualni.niti-independent-literature-benchmarks/1.0",
        "source_OR_2020":"https://doi.org/10.3390/cryst10070562",
        "source_CT_2026":"https://doi.org/10.1016/j.actamat.2026.122399",
        "synthetic_generated_EBSD":True,
        "experimental_niti_parent_measured":False,
        "niti_OR_AQ_2020":{
            "Q_parent_from_daughter":AQ["Q_parent_from_daughter"],
            "plane_error_deg":AQ["plane_normal_alignment_error_deg"],
            "direction_error_deg":AQ["inplane_direction_alignment_error_deg"],
            "daughter_variants":variant["n_variants"],
            "synthetic_noisy_daughter_count":len(observed),
            "anchor_error_deg":anchor_error,
            "blind_parent_consensus_error_modulo_cubic_deg":blind_error,
            "blind_parent_inlier_fraction":inferred["weighted_inlier_fraction"],
            "anchored_forward_median_deg":with_anchor["median_forward_residual_deg"],
            "anchored_forward_within_1deg":with_anchor["weighted_fraction_within_threshold"],
            "wrong_published_C_within_1deg":wrong_Q["weighted_fraction_within_threshold"],
            "wrong_anchor_misorientation_deg":wrong_anchor_error,
        },
        "niti_CT_2026":{
            "parent_a_angstrom":p.a,
            "daughter_abc_angstrom":[m.a,m.b,m.c],
            "daughter_beta_deg":m.beta,
            "C_product_direct_from_parent_direct":
                [[0,0,1],[.5,.5,0],[-.5,.5,0]],
            "correspondence_variants":groupoid["variants"],
            "intercorrespondence_operator_classes":groupoid["operator_classes"],
            "principal_stretches":stretches.tolist(),
        },
        "physical_cualni_parent_certified":False,
        "note":"Published lattice and OR geometry are real; EBSD orientations and parent-axis measurements in this benchmark are generated, not experimental.",
    }
