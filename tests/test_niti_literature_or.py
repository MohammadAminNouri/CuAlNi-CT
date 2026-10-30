"""Independent numerical checks on published NiTi OR AQ/A/C, plus exact anchors.

NiTi B2/B19' cell is from the legacy ARPGE NiTi fixture and Cayron 2020.
The daughter orientations below are GENERATED, not measured EBSD from Cayron.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cualni_cryst.ebsd_map import EBSDPhase
from cualni_cryst.lattice import Lattice
from cualni_cryst.niti_literature_or import (
    niti_published_or, or_from_plane_and_direction, direction_cartesian,
    plane_normal_cartesian,
)
from cualni_cryst.parent_or_interop import proper_cubic_parent_operations
from cualni_cryst.parent_replay_exchange import (
    forward_predict_daughter, forward_inverse_roundtrip,
)
from cualni_cryst.parent_consensus_complete import reconstruct_parent_full_variant_consensus
from cualni_cryst.parent_or_evidence import (
    parent_orientation_from_two_directions, compare_parent_to_anchor,
)


def preset():
    beta = Lattice.cubic(3.015)
    b19 = Lattice.monoclinic_unique_b(2.889, 4.120, 4.622, 96.8)
    sp=tuple(proper_cubic_parent_operations())
    sm=tuple(EBSDPhase.from_point_group(
        1, "NiTi B19-prime", b19, "2/m",
    ).proper_symmetry_cartesian)
    return beta,b19,sp,sm


def test_published_AQ_geometry_and_recored_expected_rotation_independent():
    p,m,sp,sm=preset()
    ref=niti_published_or("AQ",parent=p,daughter=m)
    Q=np.array(ref["Q_parent_from_daughter"])
    # Independent Cartesian construction from lattice b axis plus [101].
    # Rounded reference coefficients avoid tautologically copying the solver.
    expected=np.array([
        [.10097, .70710678, -.69986],
        [-.10097, .70710678, .69986],
        [.98975, 0., .14280],
    ])
    assert np.max(np.abs(Q-expected))<5e-4
    assert ref["plane_normal_alignment_error_deg"]<1e-5
    assert ref["inplane_direction_alignment_error_deg"]<1e-5
    assert np.linalg.norm(Q.T@Q-np.eye(3))<1e-12
    assert np.linalg.det(Q)==pytest.approx(1)
    assert np.allclose(ref["p2c_daughter_from_parent"],Q.T)
    assert np.allclose(
        Q@plane_normal_cartesian(m,(0,1,0)),
        plane_normal_cartesian(p,(1,1,0)),atol=1e-12,
    )
    assert np.allclose(
        Q@direction_cartesian(m,(1,0,1)),
        direction_cartesian(p,(-1,1,1)),atol=1e-12,
    )


def test_recovers_twelve_distinct_niti_variants_not_generic_twenty_four():
    p,m,sp,sm=preset()
    Q=np.array(niti_published_or("AQ")["Q_parent_from_daughter"])
    gA=Rotation.from_euler("xyz",[18,-11,29],degrees=True).as_matrix()
    result=forward_predict_daughter(gA,Q,sp,sm)
    assert result["n_variants"]==12
    assert result["n_daughter_symmetry_representatives"]==2
    observed=result["daughter_symmetry_representatives"].reshape(-1,3,3)
    checked=forward_inverse_roundtrip(gA,Q,sp,sm,observed_daughter_g=observed,
                                     accepted_deg=.02)
    assert checked["n_observations"]==24
    assert checked["weighted_fraction_within_threshold"]==pytest.approx(1)
    assert checked["median_forward_residual_deg"]<1e-5
    assert checked["max_forward_inverse_disagreement_deg"]<1e-4


def test_published_A_C_AQ_distinct_not_same_OR():
    p,m,sp,sm=preset()
    rotations=[np.asarray(niti_published_or(name)["Q_parent_from_daughter"])
               for name in ("AQ","A","C")]
    for mat in rotations:
        assert np.linalg.det(mat)==pytest.approx(1)
    assert np.linalg.norm(rotations[0]-rotations[2])>0.05
    parent=Rotation.from_euler("zyx",[16,22,-11],degrees=True).as_matrix()
    daughter=forward_predict_daughter(parent,rotations[0],sp,sm)[
        "daughter_crystal_to_sample"
    ]
    actual=forward_inverse_roundtrip(parent,rotations[0],sp,sm,
                                      observed_daughter_g=daughter,accepted_deg=1)
    wrong=forward_inverse_roundtrip(parent,rotations[2],sp,sm,
                                     observed_daughter_g=daughter,accepted_deg=1)
    assert actual["weighted_fraction_within_threshold"]==pytest.approx(1)
    assert wrong["weighted_fraction_within_threshold"]<.5
    assert wrong["median_forward_residual_deg"]>3


def test_synthetic_niti_parent_from_two_independent_beta_axes():
    p,m,sp,sm=preset()
    original=Rotation.from_euler("xyz",[22,-35,17],degrees=True).as_matrix()
    a=original@direction_cartesian(p,(1,0,0))
    b=original@direction_cartesian(p,(0,1,1))
    computed=parent_orientation_from_two_directions(p,(1,0,0),(0,1,1),a,b)
    assert compare_parent_to_anchor(computed,original,sp)<1e-5
    Q=np.asarray(niti_published_or("AQ")["Q_parent_from_daughter"])
    measured=forward_predict_daughter(original,Q,sp,sm)["daughter_crystal_to_sample"]
    anchored=forward_inverse_roundtrip(computed,Q,sp,sm,
                                        observed_daughter_g=measured)
    assert anchored["weighted_fraction_within_threshold"]==pytest.approx(1)
    # An independent wrong anchor must be detected, not auto-corrected.
    wrong_anchor=Rotation.from_euler("y",18,degrees=True).as_matrix()@computed
    bad=forward_inverse_roundtrip(wrong_anchor,Q,sp,sm,
                                  observed_daughter_g=measured,accepted_deg=1)
    assert bad["weighted_fraction_within_threshold"]<1


def test_coplanarity_and_lattice_setting_fail_closed():
    p,m,sp,sm=preset()
    with pytest.raises(ValueError,match="not in"):
        or_from_plane_and_direction(p,m,(1,1,0),(0,1,0),
                                    (1,1,0),(1,0,1))
    with pytest.raises(ValueError,match="NiTi"):
        niti_published_or("AQ",daughter=Lattice.monoclinic_unique_b(
            13.8,5.28,4.40,113.6
        ))
    with pytest.raises(ValueError,match="one of"):
        niti_published_or("Kajiwara")


def test_published_niti_2026_correspondence_12_variants_7_operator_classes():
    # Independent CT benchmark from Cayron (2026) and archive calculations.
    # C is a direct-coordinate correspondence, NOT the AQ orientation matrix.
    from cualni_cryst.legacy_suite_bridge import Phase
    from cualni_cryst.ar_gen_cry_pipeline import cry_stage,gen_stage
    parent=Phase("NiTi B2","m-3m",3.010,3.010,3.010)
    product=Phase("NiTi B19-prime","2/m",
                  2.898,4.108,4.646,90,97.78,90)
    cry=cry_stage(parent,product,(1,0,0),(0,1,0))
    C="0 0 1\n1/2 1/2 0\n-1/2 1/2 0"
    gen=gen_stage(cry,C)
    assert gen["variants"]==12
    assert gen["operator_classes"]==7
    # Independently evaluate stretch eigenvalues from physical direct metrics.
    B=product.lattice().structure_matrix()
    F=B@np.array([[0.,0.,1.],[.5,.5,0.],[-.5,.5,0.]])/3.010
    stretches=np.linalg.svd(F)[1]
    assert np.sort(stretches)==pytest.approx(
        [0.930026279,0.965048059,1.119484928],rel=2e-8
    )
    # Correspondence and orientation are different physical matrices.
    AQ=niti_published_or("AQ")
    assert not np.allclose(np.asarray(AQ["Q_parent_from_daughter"]),
                           np.asarray([[0,0,1],[.5,.5,0],[-.5,.5,0]]))
