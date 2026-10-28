"""Measured A/M OR is a point-group equivalence class; no inferred C."""
import numpy as np
from scipy.spatial.transform import Rotation

from twin_app.ctf_observed_am_or import observed_am_or_classes


def phases():
    return (
        dict(phase_id=1,name="parent",lengths_angstrom=(3.54,3.54,3.54),
             angles_deg=(90.,90.,90.),declared_point_group="m-3m"),
        dict(phase_id=3,name="18R",lengths_angstrom=(13.8,5.28,4.4),
             angles_deg=(90.,113.6,90.),declared_point_group="2/m"),
    )


def test_no_indexed_parent_cannot_produce_measured_am_or():
    data={
        "grains":[
            dict(grain_id=0,phase_id=3,
                 mean_crystal_to_sample_rotation=np.eye(3).tolist()),
        ],
        "adjacent_grain_pairs":[],
    }
    result=observed_am_or_classes(
        data,phases(),parent_phase_id=1,product_phase_id=3,
    )
    assert result["status"]=="INSUFFICIENT_INDEXED_A_M_INTERFACES"
    assert result["unique_physical_rotation"] is False
    assert result["C"]=="NOT_INFERRED"


def test_three_measured_parent_child_pairs_recover_or_class_only():
    R=Rotation.from_euler("xyz",[7,13,29],degrees=True).as_matrix()
    grains=[]
    boundaries=[]
    for k in range(3):
        grains.append(dict(
            grain_id=2*k,phase_id=1,
            mean_crystal_to_sample_rotation=np.eye(3).tolist(),
        ))
        grains.append(dict(
            grain_id=2*k+1,phase_id=3,
            mean_crystal_to_sample_rotation=R.tolist(),
        ))
        boundaries.append(dict(
            grain_a=2*k,grain_b=2*k+1,pixel_neighbor_edges=9,
        ))
    result=observed_am_or_classes(
        {"grains":grains,"adjacent_grain_pairs":boundaries},
        phases(),parent_phase_id=1,product_phase_id=3,
    )
    assert result["status"]=="DOMINANT_OBSERVED_OR_EQUIVALENCE_CLASS"
    assert result["dominant_class_support_fraction"]==1
    assert result["unique_physical_rotation"] is False
    assert result["C"]=="NOT_INFERRED"
    np.testing.assert_allclose(
        result["representative_classes"][0][
            "R_parent_from_product_representative"
        ],R,atol=1e-9,
    )
