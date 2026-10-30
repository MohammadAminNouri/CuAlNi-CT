from __future__ import annotations

"""Small synthetic Oxford H5OINA full-grid parent map science regression."""

from io import BytesIO
import json
import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_h5oina_cualni import fixture_h5oina
from cualni_cryst.lattice import Lattice
from cualni_cryst.ebsd_map import EBSDPhase
from cualni_cryst.parent_fullmap_h5oina import reconstruct_conditional_parent_h5oina
from cualni_cryst.parent_or_interop import proper_cubic_parent_operations


def proper():
    beta=tuple(proper_cubic_parent_operations())
    mart=EBSDPhase.from_point_group(
        1,"test 18R",Lattice(10,5,4,90,108,90),"2/m",
    ).proper_symmetry_cartesian
    return beta,tuple(mart)


def test_full_h5oina_native_reconstruction_is_bounded_and_masked():
    parent,child=proper()
    raw=fixture_h5oina()
    out,report=reconstruct_conditional_parent_h5oina(
        "synthetic.h5oina",raw,np.eye(3),np.eye(3),parent,child,
        max_mad_deg=1.5,max_fit_deg=3,min_variant_patch_pixels=2
    )
    assert report["n_x"]==4 and report["n_y"]==2
    assert report["n_pixels"]==8 and report["n_indexed"]==6
    assert report["n_quality"]==6
    assert report["n_fitted_within_deg"]==6
    assert report["n_variant_patch_supported"]>=4
    assert report["physical_parent_certified"] is False
    assert report["MTEX_graph_or_MCL_performed"] is False
    with h5py.File(BytesIO(out),"r") as f:
        q=f["map/parent_quat_xyzw"][:]
        mask=f["map/quality_mask"][:]
        fit=f["map/parent_fit_deg"][:]
        variants=f["map/assigned_variant_id_1based"][:]
        assert q.shape==(2,4,4)
        assert int(np.sum(mask))==6
        assert int(np.sum(variants>0))==6
        assert np.isnan(q[~mask]).all()
        assert np.allclose(np.linalg.norm(q[mask],axis=1),1,atol=1e-5)
        assert np.nanmax(fit)<=1e-5
        stored=json.loads(f.attrs["report_json"])
        assert stored["n_pixels"]==8


def test_wrong_fixed_parent_or_rejects_most_daughters():
    parent,child=proper()
    Qwrong=Rotation.from_euler("xyz",[28,53,11],degrees=True).as_matrix()
    _,report=reconstruct_conditional_parent_h5oina(
        "test.h5oina",fixture_h5oina(),
        Rotation.from_euler("xyz",[18,27,55],degrees=True).as_matrix(),
        Qwrong,parent,child,max_fit_deg=0.1,
        min_variant_patch_pixels=2,
    )
    assert report["n_fitted_within_deg"]<6


def test_source_schema_and_improper_or_rejected_before_export():
    p,m=proper()
    with pytest.raises(ValueError):
        reconstruct_conditional_parent_h5oina(
            "broken.h5oina",b"not h5",np.eye(3),np.eye(3),p,m
        )
    with pytest.raises(ValueError):
        reconstruct_conditional_parent_h5oina(
            "synthetic.h5oina",fixture_h5oina(),
            np.eye(3),np.diag([1,1,-1]),p,m
        )
    with pytest.raises(ValueError):
        reconstruct_conditional_parent_h5oina(
            "synthetic.h5oina",fixture_h5oina(),
            np.eye(3),np.eye(3),p,m,chunk_pixels=1
        )
