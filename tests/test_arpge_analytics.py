from __future__ import annotations

"""Scientific acceptance for new spatial EBSD observables and open exports."""

import numpy as np
import pytest
from pathlib import Path
from scipy.spatial.transform import Rotation

from cualni_cryst.arpge_spatial import segment_daughter_grains
from cualni_cryst.arpge_analytics import (
    kam_gos,cubic_sigma3_domains,stereographic_poles,habit_plane_trace,
    export_parent_ebsd_text,
)
from cualni_cryst.ebsd_io import load_ang,load_ctf
from cualni_cryst.lattice import Lattice
from test_arpge_spatial import synthetic_six_grains


def test_kam_gos_identical_grain_core_and_bounded_boundary():
    ebsd,Q,p,m,parents=synthetic_six_grains()
    seg=segment_daughter_grains(ebsd,1,m,minimum_pixels=10,tolerance_deg=2)
    report=kam_gos(ebsd,seg,m,kam_cutoff_deg=4)
    assert len(report["kam_deg"])==len(ebsd.phase_id)
    assert np.nanmax(report["kam_deg"])<1e-5
    assert np.nanmax(report["gos_pixel_deg"])<1e-4
    assert np.nanmax(report["gos_grain_mean_deg"])<1e-4
    with pytest.raises(ValueError,match="KAM cutoff"):
        kam_gos(ebsd,seg,m,kam_cutoff_deg=float("nan"))


def test_cubic_sigma3_graph_detects_true_twin_but_not_generic_boundary():
    I=np.eye(3)
    twin=Rotation.from_rotvec(np.asarray([1.,1.,1.])/np.sqrt(3)*np.pi/3).as_matrix()
    other=Rotation.from_euler("xyz",[16,27,-11],degrees=True).as_matrix()
    g=np.asarray([I,twin,other])
    xy=np.asarray([[0.,0.],[1.,0.],[2.,0.]])
    result=cubic_sigma3_domains(
        g,np.asarray([[0,1],[1,2]]),xy,tolerance_deg=.05,
    )
    assert result["sigma3_edge_count"]==1
    assert result["n_domains"]==2
    assert sorted(result["domain_grain_counts"])==[1,2]
    assert np.max(result["sigma3_chain_depth"])==1
    with pytest.raises(ValueError):
        cubic_sigma3_domains(g,np.asarray([[0,0]]),xy)


def test_direct_pole_and_reciprocal_habit_trace_correct():
    I=np.eye(3)
    poles=stereographic_poles(
        np.asarray([I]),np.asarray([[0.,0.,1.],[1.,0.,0.]]),
    )
    assert poles["stereographic_xy"][0]==pytest.approx([[0,0],[1,0]])
    test=habit_plane_trace([1,0,0],Lattice.cubic(3.015),I)
    assert abs(np.dot(test["sample_trace_direction"],[0.,1.,0.]))==pytest.approx(1)
    assert not test["degenerate_parallel_surface"]
    bad=habit_plane_trace([0,0,1],Lattice.cubic(3.015),I)
    assert bad["sample_trace_direction"] is None
    assert bad["degenerate_parallel_surface"]


@pytest.mark.parametrize("extension,loader",[("ctf",load_ctf),("ang",load_ang)])
def test_native_text_research_export_preserves_pixel_parent_and_so3(tmp_path:Path,
                                                                    extension,loader):
    ebsd,Q,p,m,parents=synthetic_six_grains()
    ids=np.where(np.asarray(ebsd.x)<6,0,1).astype(int)
    # Enforce truly unassigned rather than silently fabricating orientations.
    ids[np.asarray(ebsd.y)==0]=-1
    payload=export_parent_ebsd_text(
        ebsd,ids,np.asarray(parents),format=extension,
        parent_lattice=Lattice.cubic(3.015),
    )
    path=tmp_path/("synthetic_reconstructed."+extension)
    path.write_text(payload,encoding="utf-8")
    restored=loader(path)
    assert len(restored.phase_id)==len(ids)
    assert np.array_equal(restored.phase_id>0,ids>=0)
    assert restored.indexed.sum()==(ids>=0).sum()
    assert np.allclose(restored.orientations[ids>=0],
                       np.asarray(parents)[ids[ids>=0]],atol=2e-6)
    with pytest.raises(ValueError,match="Explicit parent lattice"):
        export_parent_ebsd_text(ebsd,ids,np.asarray(parents),format=extension)


def test_sigma3_rejects_wrong_orientation_group_and_export_refuses_unsafe_grid():
    ebsd,Q,p,m,parents=synthetic_six_grains()
    with pytest.raises(ValueError,match="proper"):
        cubic_sigma3_domains(np.asarray([np.diag([1,1,-1])]),
                             np.empty((0,2),int),np.zeros((1,2)))
    from dataclasses import replace
    x=np.asarray(ebsd.x).copy(); x[1]+=.15
    with pytest.raises(ValueError,match="complete square grid|irregular"):
        export_parent_ebsd_text(replace(ebsd,x=x),
                                np.zeros(len(x),int),np.asarray([np.eye(3)]),
                                parent_lattice=Lattice.cubic(3.015))
