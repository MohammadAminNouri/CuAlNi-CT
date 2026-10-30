"""Synthetic beta->18R multi-variant parent consensus: no experimental OR claims."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cualni_cryst.lattice import Lattice
from cualni_cryst.ebsd_map import EBSDMap, EBSDPhase
from cualni_cryst.legacy_suite_bridge import Phase, reconstruct_ebsd_parent
from cualni_cryst.parent_variant_audit import parent_generated_variants
from cualni_cryst.parent_consensus_complete import reconstruct_parent_full_variant_consensus
from cualni_cryst.ebsd_analysis import minimum_disorientation


def fixture():
    p=Phase("beta", "m-3m", 5.8, 5.8, 5.8)
    m=Phase("18R model", "2/m", 10.0, 5.0, 4.0, 90, 108, 90)
    ps=EBSDPhase.from_point_group(1,p.name,p.lattice(),p.point_group).proper_symmetry_cartesian
    ms=EBSDPhase.from_point_group(2,m.name,m.lattice(),m.point_group).proper_symmetry_cartesian
    parent=Rotation.from_euler("xyz",[17.0,-11.0,43.0],degrees=True).as_matrix()
    R=Rotation.from_euler("xyz",[8,26,-17],degrees=True).as_matrix()
    variants=parent_generated_variants(R,tuple(ps),tuple(ms))
    daughters=np.asarray([
        parent @ variants[k] @ ms[k%2]
        for k in (0,3,6,9,12,15,18,21,1,4,7,10,13,16,19,22)
    ])
    return p,m,ps,ms,parent,R,variants,daughters


def test_complete_or_variant_vote_recovers_known_parent_modulo_cubic_symmetry():
    p,m,ps,ms,parent,R,variants,observed=fixture()
    assert len(variants) == 24
    result=reconstruct_parent_full_variant_consensus(
        observed,R,tuple(ps),tuple(ms),tolerance_deg=0.05,max_seeds=384,
    )
    assert result["n_parent_generated_variants"] == 24
    assert result["weighted_inlier_fraction"] >= .999999
    assert result["mean_inlier_residual_deg"] < 1e-4
    assert minimum_disorientation(
        np.asarray(result["parent_crystal_to_sample_rotation"]),parent,ps
    ).angle_deg < 1e-4
    assert all(x > 0 for x in result["assigned_variant_labels"])


def test_existing_ar_bridge_uses_full_family_in_parent_search():
    p,m,ps,ms,parent,R,variants,observed=fixture()
    n=len(observed)
    ebsd=EBSDMap(
        orientations=observed,phase_id=np.ones(n,int),indexed=np.ones(n,bool),
        x=np.arange(n,dtype=float),y=np.zeros(n),z=np.zeros(n),
    )
    result=reconstruct_ebsd_parent(ebsd,p,m,1,R[None,:,:],
                                    maximum_points=n,tolerance_deg=.05)
    assert result["weighted_inlier_fraction"] > .99999
    assert result["full_parent_generated_variant_assignment"]["accepted_fraction"]>.999
    assert result["n_parent_generated_variants"]==24
    assert minimum_disorientation(
        np.asarray(result["parent_crystal_to_sample_rotation"]),parent,ps
    ).angle_deg < 1e-4


def test_wrong_or_is_not_silently_certified_by_consensus():
    p,m,ps,ms,parent,R,variants,observed=fixture()
    wrong=Rotation.from_euler("xyz",[47,13,78],degrees=True).as_matrix()
    res=reconstruct_parent_full_variant_consensus(
        observed,wrong,tuple(ps),tuple(ms),tolerance_deg=1.0,max_seeds=384
    )
    assert res["experimental_OR_certified"] is False
    assert res["absolute_parent_frame_certified"] is False
    assert res["weighted_inlier_fraction"] < 0.9


def test_malformed_or_throws_not_repairs():
    p,m,ps,ms,parent,R,variants,observed=fixture()
    with pytest.raises(ValueError):
        reconstruct_parent_full_variant_consensus(
            observed,np.diag([1,1,-1]),tuple(ps),tuple(ms)
        )
