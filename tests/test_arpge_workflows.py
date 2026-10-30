from __future__ import annotations

import numpy as np
import pytest
from cualni_cryst.arpge_workflows import (
    reconstruction_tolerance_cycles,polish_indexed_parent_islands,
    boundary_operator_histogram,
)
from cualni_cryst.arpge_spatial import segment_daughter_grains
from test_arpge_spatial import synthetic_six_grains


def test_stated_tolerance_cycles_have_provenance_and_never_certify_parent():
    ebsd,Q,p,m,_=synthetic_six_grains()
    seg=segment_daughter_grains(ebsd,1,m,minimum_pixels=8,tolerance_deg=2)
    result=reconstruction_tolerance_cycles(seg,Q,p,m,
        cycles=((.6,1.0),(1.0,1.5)))
    assert len(result["cycle_reports"])==2
    assert 1<=result["chosen_cycle"]<=2
    assert result["cycle_reports"][0]["physical_or_certified"] is False
    assert result["candidate"]["n_parent_domains"]>=1
    with pytest.raises(ValueError,match="schedule"):
        reconstruction_tolerance_cycles(seg,Q,p,m,cycles=((2.0,1.0),))


def test_morphology_never_fills_unindexed_hole_or_boundary():
    label=np.ones((5,5),int)
    label[2,2]=-1
    mask=np.ones((5,5),bool)
    result=polish_indexed_parent_islands(label.ravel(),mask.ravel(),(5,5))
    assert result["n_pixels_changed"]==1
    assert result["polished_parent_pixel_id"][12]==1
    label[2,2]=-1;mask[2,2]=False
    protected=polish_indexed_parent_islands(label.ravel(),mask.ravel(),(5,5))
    assert protected["n_pixels_changed"]==0
    assert protected["n_unindexed_pixels_changed"]==0
    label[2,2]=-1;mask[:]=True
    label[0,2]=-1
    out=polish_indexed_parent_islands(label.ravel(),mask.ravel(),(5,5))
    assert out["polished_parent_pixel_id"][2]==-1


def test_operator_histogram_is_contact_weighted_and_bounded():
    ebsd,Q,p,m,_=synthetic_six_grains()
    seg=segment_daughter_grains(ebsd,1,m,minimum_pixels=8,tolerance_deg=2)
    h=boundary_operator_histogram(seg,Q,p,m,bin_width_deg=1)
    assert h["n_pairs"]==5
    assert sum(h["histogram"])==5
    assert sum(h["histogram_contacts"])==pytest.approx(
        sum(seg["boundary_contacts"]),
    )
    with pytest.raises(ValueError):
        boundary_operator_histogram(seg,Q,p,m,bin_width_deg=float("nan"))
