from __future__ import annotations
"""Synthetic two-parent spatial inverse problem: no invented experimental EBSD."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cualni_cryst.arpge_spatial import (
    segment_daughter_grains, reconstruct_spatial_parents, disorientation_edges,
)
from cualni_cryst.ebsd_map import EBSDMap, EBSDPhase, OrientationConvention
from cualni_cryst.lattice import Lattice
from cualni_cryst.parent_or_interop import proper_cubic_parent_operations
from cualni_cryst.parent_replay_exchange import forward_predict_daughter


def synthetic_six_grains():
    parent=tuple(proper_cubic_parent_operations())
    daughter=tuple(EBSDPhase.from_point_group(
        1,"B19prime",Lattice.monoclinic_unique_b(2.889,4.120,4.622,96.8),
        "2/m",
    ).proper_symmetry_cartesian)
    Q=Rotation.from_euler("xyz",[8,17,31],degrees=True).as_matrix()
    parents=[
        Rotation.from_euler("xyz",[12,-10,24],degrees=True).as_matrix(),
        Rotation.from_euler("xyz",[-38,28,-19],degrees=True).as_matrix(),
    ]
    variants=[forward_predict_daughter(g,Q,parent,daughter)[
        "daughter_crystal_to_sample"] for g in parents]
    nx,ny=12,8
    xx,yy=np.meshgrid(np.arange(nx),np.arange(ny))
    grain=xx.ravel()//2
    matrices=np.asarray([
        variants[int(g>=3)][[0,2,5][int(g)%3]]
        for g in grain
    ])
    euler=Rotation.from_matrix(matrices).as_euler("ZXZ")
    ebsd=EBSDMap.from_eulers(
        euler,np.ones(len(euler),int),xx.ravel(),yy.ravel(),
        convention=OrientationConvention.bunge_crystal_to_sample_radians(),
    )
    return ebsd,Q,parent,daughter,parents


def test_symmetry_aware_spatial_segmentation_has_six_independent_daughter_grains():
    ebsd,Q,p,m,parents=synthetic_six_grains()
    seg=segment_daughter_grains(ebsd,1,m,minimum_pixels=10,tolerance_deg=2)
    assert seg["n_grains"]==6
    assert sorted(seg["grain_sizes"].tolist())==[16]*6
    assert seg["grain_boundary_edges"]==5
    assert len(set(seg["pixel_daughter_grain_id"]))==6
    assert seg["n_small_region_pixels_rejected"]==0


def test_three_and_four_grain_nucleation_are_bounded_and_competitive():
    ebsd,Q,p,m,parents=synthetic_six_grains()
    seg=segment_daughter_grains(ebsd,1,m,minimum_pixels=10,tolerance_deg=2)
    tri=reconstruct_spatial_parents(seg,Q,p,m,
        nucleation_tolerance_deg=.75,growth_tolerance_deg=1,
        nucleus_size=3,ambiguity_margin_deg=.1)
    assert tri["n_parent_domains"]==2
    assert tri["n_unassigned_daughter_grains"]==0
    # Union-find root IDs do not imply left-to-right spatial grain ordering.
    # Verify parent labels on ACTUAL pixel coordinates, never array indices.
    x=np.asarray(ebsd.x)
    left=set(tri["pixel_parent_id"][x<6].tolist())
    right=set(tri["pixel_parent_id"][x>=6].tolist())
    assert len(left)==1 and len(right)==1
    assert left!=right
    assert -1 not in left and -1 not in right
    assert tri["physical_or_certified"] is False
    # Our simple stripe graph has no nodes with three neighbours.
    quad=reconstruct_spatial_parents(seg,Q,p,m,nucleus_size=4,
        nucleation_tolerance_deg=.75)
    assert quad["n_parent_domains"]==0


def test_bad_or_and_wrong_phase_fail_closed():
    ebsd,Q,p,m,parents=synthetic_six_grains()
    seg=segment_daughter_grains(ebsd,1,m,minimum_pixels=10)
    wrong=Q@Rotation.from_euler("xyz",[28,-21,17],degrees=True).as_matrix()
    bad=reconstruct_spatial_parents(seg,wrong,p,m,nucleus_size=3,
        nucleation_tolerance_deg=.35,growth_tolerance_deg=.5)
    assert bad["n_parent_domains"]==0
    with pytest.raises(ValueError,match="indexed daughter"):
        segment_daughter_grains(ebsd,2,m)
    with pytest.raises(ValueError):
        reconstruct_spatial_parents(seg,np.diag([1,1,-1]),p,m)


def test_duplicate_coordinate_and_truncated_search_are_explicit():
    ebsd,Q,p,m,parents=synthetic_six_grains()
    # Duplicate positions cannot be used to fabricate spatial neighbours.
    from dataclasses import replace
    x=np.array(ebsd.x,copy=True)
    x[1]=x[0]
    y=np.array(ebsd.y,copy=True)
    y[1]=y[0]
    damaged=replace(ebsd,x=x,y=y)
    with pytest.raises(ValueError,match="Duplicate"):
        segment_daughter_grains(damaged,1,m)
    seg=segment_daughter_grains(ebsd,1,m,minimum_pixels=10)
    with pytest.raises(ValueError,match="Nucleus"):
        reconstruct_spatial_parents(seg,Q,p,m,nucleus_size=5)


def test_disorientation_respects_right_product_symmetry():
    _,Q,p,m,parents=synthetic_six_grains()
    g=Rotation.from_euler("xyz",[14,29,-11],degrees=True).as_matrix()
    orientation=np.asarray([g,g@m[1]])
    q=Rotation.from_matrix(orientation).as_quat()
    assert disorientation_edges(q,np.asarray([[0,1]]),m)[0]<1e-5
    assert disorientation_edges(q,np.asarray([[0,1]]),(np.eye(3),))[0]>20
