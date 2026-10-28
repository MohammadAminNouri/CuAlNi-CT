"""Fast and native EBSD segmentation and KAM must agree on the same CTF."""
from pathlib import Path
import tempfile

import numpy as np
from numpy.testing import assert_allclose

from cualni_cryst.ebsd_analysis import (
    build_neighbor_graph, kernel_average_misorientation,
    segment_grains, minimum_disorientation, grain_statistics,
)
from cualni_cryst.ebsd_io import load_ctf
from cualni_cryst.ebsd_map import EBSDPhase, AngleUnit
from cualni_cryst.lattice import Lattice
from twin_app.ctf_fast_grains import (
    accelerated_ebsd_maps, edge_disorientations,
    accelerated_grain_statistics,
)
from twin_app.ctf_independent import independent_ctf_cartography


def make_ctf():
    lines=[
        "Channel Text File","JobMode Grid",
        "XCells 6","YCells 4","XStep 1","YStep 1",
        "Phases 1",
        "5.342;4.224;4.390\t90;90;90\t2H\t3\t59\t\t",
        "Phase X Y Bands Error Euler1 Euler2 Euler3 MAD BC BS",
    ]
    for y in range(4):
        for x in range(6):
            phase=0 if x==5 and y==3 else 1
            # An inhomogeneous intra-grain orientation plus a separate domain.
            phi1=15.0+0.3*x if x<3 else 77.0+0.4*y
            lines.append(
                f"{phase} {x} {y} 8 0 {phi1} 32 25 1.2 90 70"
            )
    return ("\n".join(lines)+"\n").encode()


def test_exact_disorientation_and_segmentation_vs_native_reference():
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/"sample.ctf"
        path.write_bytes(make_ctf())
        eb=load_ctf(path)
    p=EBSDPhase.from_point_group(
        1,"2H",Lattice(5.342,4.224,4.390),"mmm"
    )
    ph={1:p}
    nb=build_neighbor_graph(eb)
    angles=edge_disorientations(eb,ph,nb,chunk=128)
    for i,(a,b) in enumerate(nb.edges):
        if eb.indexed[a] and eb.indexed[b]:
            exact=minimum_disorientation(
                eb.orientations[a],eb.orientations[b],
                p.proper_symmetry_cartesian,
            ).angle_deg
            assert abs(angles[i]-exact)<1e-5
        else:
            assert np.isnan(angles[i])
    fast=accelerated_ebsd_maps(
        eb,ph,nb,threshold_deg=3.,minimum_points=2,
        kam_max_deg=5.,
    )
    native=segment_grains(
        eb,ph,threshold_deg=3.,neighbor_graph=nb,minimum_points=2,
    )
    assert np.array_equal(
        fast["segmentation"].grain_id,native.grain_id
    )
    kam=kernel_average_misorientation(
        eb,ph,neighbor_graph=nb,
        maximum_neighbor_misorientation_deg=5.,
    )
    assert_allclose(fast["kam_deg"],kam,atol=1e-5,equal_nan=True)
    assert np.count_nonzero(fast["grain_boundary_mask"])>0
    measured=grain_statistics(eb,native,ph)
    accelerated=accelerated_grain_statistics(eb,fast["segmentation"],ph)
    assert len(measured)==len(accelerated)
    for original,computed in zip(measured,accelerated):
        assert original.size==computed.size
        assert_allclose(original.centroid,computed.centroid,atol=1e-9)
        assert minimum_disorientation(
            original.mean_orientation,computed.mean_orientation,
            p.proper_symmetry_cartesian,
        ).angle_deg < 1e-4
        assert abs(original.gos_deg-computed.gos_deg) < 1e-4
        assert abs(original.maximum_spread_deg-computed.maximum_spread_deg) < 1e-4



def test_independent_output_preserves_quality_and_spread_fields():
    data=independent_ctf_cartography(
        make_ctf(),"sample.ctf",
        misorientation_threshold_deg=3.,
        minimum_grain_pixels=2,
    )
    assert data["n_indexed"]==23
    assert data["n_grains"]>=2
    assert {"gos_deg","neighbor_max_deg","grain_boundary_mask"}.issubset(
        data["maps"]
    )
    assert np.isfinite(data["maps"]["gos_deg"][data["maps"]["grain_id"]>=0]).all()


def test_3d_angle_units_are_required_and_explicit():
    three_d=(
        "Channel Text File\nXCells 2\nYCells 1\nZCells 2\n"
        "Phases 1\n"
        "5.342;4.224;4.390\t90;90;90\t2H\t3\t59\t\t\n"
        "Phase X Y Z Bands Error Euler1 Euler2 Euler3 MAD BC BS\n"
        "1 0 0 0 8 0 12 20 36 .9 90 80\n"
        "1 1 0 0 8 0 12 20 36 .9 90 80\n"
        "1 0 0 1 8 0 14 20 36 .9 90 80\n"
        "1 1 0 1 8 0 14 20 36 .9 90 80\n"
    ).encode()
    import pytest
    with pytest.raises(ValueError,match="angle unit"):
        independent_ctf_cartography(three_d,"a.ctf")
    o=independent_ctf_cartography(
        three_d,"a.ctf",minimum_grain_pixels=1,
        three_dimensional_angle_unit="degree",
    )
    assert o["n_indexed"]==4
    assert o["n_grains"]>=1
    assert len(np.unique(o["maps"]["z"]))==2
