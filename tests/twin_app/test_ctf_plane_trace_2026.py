"""A 3D experimental plane needs two independent registered section traces."""
import numpy as np
import pytest

from cualni_cryst.lattice import Lattice
from twin_app.ctf_plane_trace import reconstruct_registered_interface_plane


def test_two_registered_nonparallel_sections_recover_known_plane():
    # Same physical plane z=0, two orthogonal specimen cuts.
    result=reconstruct_registered_interface_plane(
        [1,0,0],[0,1,0],
        [0,1,0],[1,0,0],
        g_crystal_to_sample=np.eye(3),
        lattice=Lattice.cubic(3.544),
        trace_uncertainty_deg=1.,
    )
    assert result["status"]=="CONDITIONAL_TWO_SECTION_INTERFACE_PLANE"
    np.testing.assert_allclose(
        np.abs(result["plane_normal_sample"]),[0,0,1],atol=1e-10,
    )
    np.testing.assert_allclose(
        np.abs(result["hkl_reciprocal_covector_crystal_unrounded"]),
        [0,0,1],atol=1e-10,
    )
    assert result["normal_uncertainty_first_order_bound_deg"]==pytest.approx(2.)
    assert result["classification"].startswith("UNASSIGNED_")


def test_one_section_or_parallel_traces_cannot_determine_3d_plane():
    same_section=dict(
        trace_1=[1,0,0],section_normal_1=[0,0,1],
        trace_2=[0,1,0],section_normal_2=[0,0,1],
        g_crystal_to_sample=np.eye(3),lattice=Lattice.cubic(3.5),
    )
    with pytest.raises(ValueError,match="section normals are nearly parallel"):
        reconstruct_registered_interface_plane(**same_section)
    near_parallel=dict(
        trace_1=[1,0,0],section_normal_1=[0,1,0],
        trace_2=[1,.001,0],section_normal_2=[0,0,1],
        g_crystal_to_sample=np.eye(3),lattice=Lattice.cubic(3.5),
    )
    with pytest.raises(ValueError,match="traces are nearly parallel"):
        reconstruct_registered_interface_plane(**near_parallel)


def test_trace_must_lie_in_its_section_and_orientation_must_be_SO3():
    with pytest.raises(ValueError,match="trace must lie"):
        reconstruct_registered_interface_plane(
            [0,1,0],[0,1,0],[0,1,0],[1,0,0],
            g_crystal_to_sample=np.eye(3),lattice=Lattice.cubic(3.5),
        )
    with pytest.raises(ValueError):
        reconstruct_registered_interface_plane(
            [1,0,0],[0,1,0],[0,1,0],[1,0,0],
            g_crystal_to_sample=np.diag([-1,1,1]),
            lattice=Lattice.cubic(3.5),
        )
