import json

import numpy as np

from cualni_cryst.crystal_objects import (
    CrystalBasisRef,
    Direction,
    ObjectProvenance,
    Plane,
    axis_angle_deg,
    direction_angle_deg,
    direction_plane_angle_deg,
    equivalent_directions,
    equivalent_planes,
    incidence_residual,
    interplanar_angle_deg,
    lies_in_plane,
)
from cualni_cryst.lattice import Lattice
from cualni_cryst.provenance import DataStatus
from cualni_cryst.representation import CartesianConvention
from cualni_cryst.symmetry import cubic_full_m3m


def monoclinic():
    return Lattice.monoclinic_unique_b(4.43, 5.33, 12.79, 95.68)


def basis():
    return CrystalBasisRef(
        phase_id="martensite_6M",
        basis_id="reference_6M_unique_b",
        cell_representation="6M",
    )


def test_direction_and_plane_are_distinct_types_and_reject_zero_indices():
    d = Direction((1, 0, 1), basis())
    p = Plane((1, 0, 1), basis())

    assert type(d) is not type(p)

    for cls in (Direction, Plane):
        try:
            cls((0, 0, 0), basis())
        except ValueError:
            pass
        else:
            raise AssertionError("zero crystallographic object must be rejected")


def test_metric_normalization_does_not_mutate_stored_indices():
    lattice = monoclinic()
    d = Direction((2, -1, 3), basis())
    p = Plane((1, 0, 2), basis())

    original_d = d.indices
    original_p = p.indices

    du = d.unit_coordinates(lattice)
    pu = p.unit_covector(lattice)

    assert d.indices == original_d
    assert p.indices == original_p
    assert np.isclose(du @ lattice.metric() @ du, 1.0)
    assert np.isclose(
        pu @ np.linalg.inv(lattice.metric()) @ pu,
        1.0,
    )


def test_metric_and_all_cartesian_conventions_give_same_lengths():
    lattice = monoclinic()
    d = Direction((1, 2, -1), basis())
    p = Plane((2, -1, 1), basis())

    metric_direct = d.length(lattice)
    metric_reciprocal = p.reciprocal_length(lattice)

    for convention in CartesianConvention:
        assert np.isclose(
            np.linalg.norm(d.cartesian(lattice, convention)),
            metric_direct,
            atol=1e-10,
            rtol=1e-10,
        )
        assert np.isclose(
            np.linalg.norm(p.cartesian_normal(lattice, convention)),
            metric_reciprocal,
            atol=1e-10,
            rtol=1e-10,
        )


def test_incidence_is_metric_and_cartesian_invariant():
    lattice = monoclinic()
    d = Direction((1, 0, 0), basis())
    p = Plane((0, 1, 0), basis())

    assert lies_in_plane(d, p, lattice)
    assert incidence_residual(d, p, lattice) == 0.0
    assert direction_plane_angle_deg(d, p, lattice) == 0.0

    for convention in CartesianConvention:
        u = d.cartesian(lattice, convention)
        g = p.cartesian_normal(lattice, convention)
        assert np.isclose(np.dot(g, u), p.array @ d.array, atol=1e-12)


def test_nontrivial_direction_plane_angle_matches_cartesian_geometry():
    lattice = monoclinic()
    d = Direction((1, 1, 0), basis())
    p = Plane((1, 0, 1), basis())

    metric_angle = direction_plane_angle_deg(d, p, lattice)

    for convention in CartesianConvention:
        u = d.cartesian(lattice, convention, normalize=True)
        n = p.cartesian_normal(lattice, convention, normalize=True)
        cart_angle = np.rad2deg(np.arcsin(np.clip(abs(np.dot(u, n)), 0.0, 1.0)))
        assert np.isclose(metric_angle, cart_angle, atol=1e-10)


def test_oriented_direction_angle_and_projective_axis_angle_are_not_conflated():
    lattice = monoclinic()
    d1 = Direction((1, 0, 0), basis())
    d2 = Direction((-1, 0, 0), basis())

    assert np.isclose(direction_angle_deg(d1, d2, lattice), 180.0)
    assert np.isclose(axis_angle_deg(d1, d2, lattice), 0.0)


def test_interplanar_angle_identifies_opposite_plane_normals():
    lattice = monoclinic()
    p1 = Plane((1, 0, 1), basis())
    p2 = Plane((-1, 0, -1), basis())

    assert np.isclose(interplanar_angle_deg(p1, p2, lattice), 0.0)


def test_symmetry_uses_direct_action_for_directions_and_inverse_transpose_for_planes():
    lattice = monoclinic()
    d = Direction((1, 2, 0), basis())
    p = Plane((2, -1, 0), basis())

    # A simple invertible direct-space basis operator chosen so the dual action
    # is not numerically identical to the direct action.
    g = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    d2 = d.transformed(g)
    p2 = p.transformed(g)

    # Duality pairing must be invariant under u'=g u, p'=g^-T p.
    assert np.isclose(p2.array @ d2.array, p.array @ d.array)

    # This test is algebraic and independent of whether g is a symmetry of M.
    assert np.isfinite(d2.length(lattice))
    assert np.isfinite(p2.reciprocal_length(lattice))


def test_different_crystallographic_bases_cannot_be_compared_silently():
    lattice = monoclinic()
    d1 = Direction((1, 0, 0), basis())
    d2 = Direction(
        (1, 0, 0),
        CrystalBasisRef("austenite", "cubic_DO3"),
    )

    try:
        direction_angle_deg(d1, d2, lattice)
    except ValueError as exc:
        assert "explicit orientation/correspondence" in str(exc)
    else:
        raise AssertionError("cross-basis comparison must require an explicit map")


def test_cubic_symmetry_equivalent_sets_have_explicit_sense_policy():
    ops = cubic_full_m3m()
    cubic_basis = CrystalBasisRef("austenite", "cubic_DO3")
    d = Direction((1, 0, 0), cubic_basis)
    p = Plane((1, 0, 0), cubic_basis)

    d_oriented = equivalent_directions(d, ops, projective=False)
    d_axes = equivalent_directions(d, ops, projective=True)
    p_family = equivalent_planes(p, ops)

    assert len(d_oriented) == 6
    assert len(d_axes) == 3
    assert len(p_family) == 3


def test_provenance_and_object_payload_are_json_serializable():
    d = Direction(
        (1, 1, 0),
        basis(),
        label="observed trace direction",
        provenance=ObjectProvenance(
            status=DataStatus.USER_MEASURED,
            source_key="ebsd_map_001",
            uncertainty="±0.5 deg",
        ),
    )

    payload = d.to_dict()
    encoded = json.dumps(payload)

    assert payload["kind"] == "direction"
    assert payload["status"] == "USER_MEASURED"
    assert "ebsd_map_001" in encoded
