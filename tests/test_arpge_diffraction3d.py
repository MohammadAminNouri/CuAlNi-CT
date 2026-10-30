from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cualni_cryst.arpge_diffraction3d import (
    reciprocal_geometry,kinematic_structure_factors,
    simulate_kinematic_diffraction,electron_wavelength_angstrom,
    cell_and_reciprocal_3d,
)
from cualni_cryst.lattice import Lattice


def _bcc():
    return [{"fractional":[0,0,0],"amplitude":1},
            {"fractional":[.5,.5,.5],"amplitude":1}]


def _fcc():
    return [{"fractional":p,"amplitude":1} for p in (
        [0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0],
    )]


def test_independent_atomic_bcc_fcc_systematic_absences():
    hkl=[[1,0,0],[1,1,0],[1,1,1],[2,0,0]]
    bcc=kinematic_structure_factors(hkl,_bcc())["relative_intensity"]
    fcc=kinematic_structure_factors(hkl,_fcc())["relative_intensity"]
    assert bcc==pytest.approx([0.,4.,0.,4.],abs=1e-12)
    assert fcc==pytest.approx([0.,0.,16.,16.],abs=1e-12)


def test_reciprocal_monoclinic_duality_and_3d_cell_edges():
    lattice=Lattice.monoclinic_unique_b(2.889,4.120,4.622,96.8)
    expected=lattice.metric()
    d=reciprocal_geometry(lattice)
    B=d["direct_basis"];G=d["reciprocal_basis_cycles_per_angstrom"]
    assert np.allclose(B.T@B,expected,atol=1e-11)
    assert np.allclose(G.T@G,np.linalg.inv(expected),atol=1e-11)
    assert np.allclose(B.T@G,np.eye(3),atol=1e-12)
    g=Rotation.from_euler("xyz",[19,-7,42],degrees=True).as_matrix()
    geo=cell_and_reciprocal_3d(lattice,g)
    assert geo["edges_12"].shape==(12,2)
    assert geo["vertices_sample_angstrom"].shape==(8,3)
    assert np.allclose(geo["direct_axes_sample_angstrom"].T.T@
                       geo["reciprocal_axes_sample_cycles_per_angstrom"],np.eye(3),
                       atol=1e-11)


def test_relativistic_electron_wavelength_and_diffraction_negative_controls():
    assert electron_wavelength_angstrom(200)==pytest.approx(.02508,abs=.00003)
    report=simulate_kinematic_diffraction(
        Lattice.cubic(3.0),np.eye(3),_bcc(),maximum_index=2,
        excitation_tolerance_cycles_per_angstrom=.02,
    )
    data={tuple(row):p for row,p in zip(report["hkl"],report["relative_intensity"])}
    assert (1,0,0) in data and (1,1,0) in data
    assert data[(1,0,0)]<1e-11
    assert data[(1,1,0)]==pytest.approx(4.,abs=1e-11)
    assert np.all(np.isfinite(report["detector_xy_mm"]))
    assert report["physical_structure_validated"] is False
    with pytest.raises(ValueError,match="atomic"):
        simulate_kinematic_diffraction(Lattice.cubic(3),np.eye(3),[])
    with pytest.raises(ValueError,match="proper"):
        cell_and_reciprocal_3d(Lattice.cubic(3),np.diag([1,1,-1]))


def test_reflection_intensity_requires_explicit_atomic_information():
    with pytest.raises(ValueError,match="atomic"):
        kinematic_structure_factors([[1,0,0]],[])
    with pytest.raises(ValueError):
        kinematic_structure_factors([[.5,0,0]],_bcc())
    with pytest.raises(ValueError):
        kinematic_structure_factors([[0,0,1]],[{"fractional":[0,0,0],
                                                  "amplitude":float("nan")}])
