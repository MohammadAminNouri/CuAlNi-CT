from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from cualni_cryst.experimental_parent_validation import (
    complete_or_family,cross_phase_contact_pairs,
    frozen_or_contact_residuals,frozen_or_boundary_audit,
    group_disjoint_source_or_selection,
)
from cualni_cryst.ebsd_map import EBSDPhase
from cualni_cryst.lattice import Lattice
from cualni_cryst.parent_or_interop import proper_cubic_parent_operations
from cualni_cryst.niti_literature_or import niti_published_or


def model(relationship="AQ",*,n=60):
    G=tuple(proper_cubic_parent_operations())
    M=tuple(EBSDPhase.from_point_group(
        1,"NiTi B19prime",Lattice.monoclinic_unique_b(2.889,4.120,4.622,96.8),
        "2/m",
    ).proper_symmetry_cartesian)
    Q=np.asarray(niti_published_or(relationship)["Q_parent_from_daughter"])
    g=Rotation.from_euler("xyz",[17,-28,39],degrees=True).as_matrix()
    rng=np.random.default_rng(14)
    parents=np.asarray([g@G[i%len(G)] for i in range(n)])
    children=np.asarray([
        g@G[i%len(G)]@Q@M[i%len(M)] for i in range(n)
    ])
    # Cross-phase contacts on a 2-column stripe, at deliberately repeated
    # nearby parent/child sites but no same-phase short-circuit:
    xy=np.asarray([[i*3.0,0.] for i in range(n)]+
                  [[i*3.0,0.2] for i in range(n)])
    phase=np.asarray([1]*n+[2]*n)
    observed=np.concatenate([parents,children])
    return xy,phase,observed,Q,G,M


def test_frozen_niti_aq_real_measured_parent_contract_and_wrong_or():
    xy,phase,orientation,Q,G,M=model()
    contacts=cross_phase_contact_pairs(xy,phase,1,2,distance=0.25)
    assert contacts.shape==(60,2)
    assert np.all(contacts[:,0]<60) and np.all(contacts[:,1]>=60)
    correct=frozen_or_contact_residuals(
        orientation[contacts[:,0]],orientation[contacts[:,1]],Q,G,M,
    )
    assert max(correct)<2e-5
    # Wrong orientation-relationship direction is not an innocuous convention.
    wrong=frozen_or_contact_residuals(
        orientation[contacts[:,0]],orientation[contacts[:,1]],Q.T,G,M,
    )
    assert np.median(wrong)>3.
    report=frozen_or_boundary_audit(
        orientations=orientation,phase_ids=phase,xy=xy,
        parent_phase_id=1,daughter_phase_id=2,neighbor_distance=.25,
        frozen_q=Q,parent_proper=G,daughter_proper=M,
        negative_q=Q.T,acceptance_deg=2,
    )
    assert report["n_tested_contacts"]==60
    assert report["correct_OR_fraction_within_threshold"]==pytest.approx(1)
    assert report["negative_OR_fraction_within_threshold"]<.3
    assert report["full_parent_grain_reconstruction_verified"] is False


def test_specimen_frame_covariance_and_wrong_one_sided_calibration():
    _,_,g,Q,G,M=model(n=24)
    parents=g[:24];children=g[24:]
    H=Rotation.from_euler("xyz",[35,14,-23],degrees=True).as_matrix()
    base=frozen_or_contact_residuals(parents,children,Q,G,M)
    common=frozen_or_contact_residuals(H@parents,H@children,Q,G,M)
    assert common==pytest.approx(base,abs=2e-5)
    mismatched=frozen_or_contact_residuals(parents,H@children,Q,G,M)
    assert np.median(mismatched)>5


def test_quality_filter_and_grain_block_exclusion_remain_unambiguous():
    xy,phase,orientation,Q,G,M=model(n=24)
    filt=np.ones(len(phase),bool);filt[5]=False;filt[24+5]=False
    report=frozen_or_boundary_audit(
        orientations=orientation,phase_ids=phase,xy=xy,
        parent_phase_id=1,daughter_phase_id=2,neighbor_distance=.25,
        frozen_q=Q,parent_proper=G,daughter_proper=M,
        negative_q=Q.T,quality_mask=filt,maximum_sample=10,
        pixel_group=np.r_[np.arange(24),np.arange(24)],
    )
    assert report["n_source_contacts_quality_filtered"]==23
    assert report["n_tested_contacts"]==10
    assert len(report["contact_parent_group_ids"])==10
    assert report["Q_optimized_from_holdout"] is False
    with pytest.raises(ValueError,match="phase"):
        cross_phase_contact_pairs(xy,phase,1,1,distance=.25)
    with pytest.raises(ValueError,match="proper"):
        complete_or_family(np.diag([1,1,-1]),G,M)


def test_frozen_cualni_18r_morning_or_frames_and_complete_symmetry():
    # Frozen 30 Oct original Site8->Site6/7 source OR. We can verify its
    # algebra, but the original Oxford 18R H5OINA scan is not shipped in
    # the public repo; no independent measured beta anchor is invented.
    Q=np.asarray([[.677730,.087272,-.730113],
                  [.045760,-.996013,-.076579],
                  [-.733886,.018490,-.679021]],float)
    # Published display 6 decimals: replace with nearest SO3 ONLY for this
    # synthetic regression. The actual H5OINA report stores high precision.
    U,_,V=np.linalg.svd(Q);Q=U@V
    G=tuple(proper_cubic_parent_operations())
    M=tuple(EBSDPhase.from_point_group(
        1,"18R",Lattice.monoclinic_unique_b(13.8017,5.2856,4.3987,113.6),
        "2/m",
    ).proper_symmetry_cartesian)
    orientation=Rotation.from_euler("xyz",[22,-11,19],degrees=True).as_matrix()
    parents=np.asarray([orientation@G[i] for i in (0,3,6,9)])
    daughters=np.asarray([parents[i]@Q@M[i%2] for i in range(4)])
    residual=frozen_or_contact_residuals(parents,daughters,Q,G,M)
    assert max(residual)<2e-5
    assert len(complete_or_family(Q,G,M)[0])==len(G)*len(M)


def test_island_group_disjoint_preset_selection_never_uses_holdout_for_or_choice():
    xy,phase,ori,Q,G,M=model(n=60)
    groups=np.r_[np.arange(1,61),np.zeros(60,int)]
    cand={
        "wrong_transpose":Q.T,
        "source_AQ":Q,
        "wrong_shift":Q@Rotation.from_euler("x",15,degrees=True).as_matrix(),
    }
    first=group_disjoint_source_or_selection(
        orientations=ori,phase_ids=phase,xy=xy,parent_groups=groups,
        parent_phase_id=1,daughter_phase_id=2,neighbor_distance=.25,
        source_or_candidates=cand,parent_proper=G,daughter_proper=M,
        acceptance_deg=2.,seed=20261030,
    )
    assert first["chosen_archived_or_by_training_only"]=="source_AQ"
    assert first["parent_island_labels_disjoint"]
    assert first["candidate_OR_refit_on_holdout"] is False
    assert first["heldout_fraction_within_threshold"]==pytest.approx(1)
    assert first["wrong_15deg_heldout_fraction_within_threshold"]<.2
    assert first["experimental_parent_reconstruction_certified"] is False
    bad=groups.copy();bad[:]=1
    with pytest.raises(ValueError,match="Insufficient"):
        group_disjoint_source_or_selection(
            orientations=ori,phase_ids=phase,xy=xy,parent_groups=bad,
            parent_phase_id=1,daughter_phase_id=2,neighbor_distance=.25,
            source_or_candidates=cand,parent_proper=G,daughter_proper=M,
        )
