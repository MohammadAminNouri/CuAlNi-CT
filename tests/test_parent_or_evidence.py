from __future__ import annotations

"""SO3/reference frame, exact basis, and independent anchor regression tests."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cualni_cryst.lattice import Lattice
from cualni_cryst.ebsd_map import EBSDPhase
from cualni_cryst.parent_or_evidence import (
    audit_direct_basis, rebase_orientation_relationship,
    lattice_direction_to_cart, direction_anchor_residual,
    parent_orientation_from_two_directions, compare_parent_to_anchor,
    audit_anchor_status, parent_fit_in_raw_oxford_frame,
)


def test_nontrivial_reindexing_preserves_metric_and_physical_or():
    old = Lattice(3.0, 4.0, 5.0, 90, 90, 90)
    # B_new = B_old @ P: swap a/b with sign flip to keep det(P)>0
    P = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    new = Lattice(4, 3, 5, 90, 90, 90)
    old_parent = Lattice(6, 6, 6, 90, 90, 90)
    new_parent = Lattice(6, 6, 6, 90, 90, 90)
    p = audit_direct_basis(old_parent, new_parent, np.eye(3, dtype=int))
    m = audit_direct_basis(old, new, P)
    assert m.absolute_volume_factor == 1
    assert m.metric_relative_residual < 1e-12
    R_old = Rotation.from_euler("xyz", [12, 20, -18], degrees=True).as_matrix()
    R_new = rebase_orientation_relationship(R_old, p, m)
    for u_new in ((1, 0, 0), (0, 1, 0), (1, 1, 2)):
        d = lattice_direction_to_cart(new, u_new)
        old_d = lattice_direction_to_cart(old, P @ np.asarray(u_new))
        assert np.allclose(R_new @ d, R_old @ old_d, atol=1e-12)


@pytest.mark.parametrize("P", [
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
    [[1.001, 0, 0], [0, 1, 0], [0, 0, 1]],
])
def test_illegal_basis_rejected(P):
    a = Lattice(3, 3, 3, 90, 90, 90)
    with pytest.raises(ValueError):
        audit_direct_basis(a, a, P)


def test_wrong_metric_basis_is_rejected_even_with_integer_P():
    old = Lattice(3, 4, 5, 90, 90, 90)
    new = Lattice(3, 4.2, 5, 90, 90, 90)
    with pytest.raises(ValueError, match="metric equality"):
        audit_direct_basis(old, new, np.eye(3, dtype=int))


def test_parent_orientation_two_independent_directions_recovers_known_parent():
    cubic = Lattice(5.8, 5.8, 5.8, 90, 90, 90)
    g = Rotation.from_euler("zyx", [18, -22, 13], degrees=True).as_matrix()
    d1, d2 = (1, 0, 0), (0, 1, 1)
    a = g @ lattice_direction_to_cart(cubic, d1)
    b = g @ lattice_direction_to_cart(cubic, d2)
    recovered = parent_orientation_from_two_directions(cubic, d1, d2, a, b)
    assert Rotation.from_matrix(recovered.T @ g).magnitude() < 1e-9
    phase = EBSDPhase.from_point_group(1, "beta", cubic, "m-3m")
    assert compare_parent_to_anchor(recovered, g, phase.proper_symmetry_cartesian) < 1e-5
    assert direction_anchor_residual(
        recovered, phase.proper_symmetry_cartesian, cubic, d1, a
    ) < 1e-5


def test_one_direction_does_not_establish_absolute_parent_orientation():
    result = audit_anchor_status(
        measured_parent_orientation=False, independent_direction_count=1,
        indexed_parent_phase_present=False, verified_change_of_basis=True,
        scan_frame_verified=True,
    )
    assert result["physical_parent_or_certified"] is False
    assert result["blocking_requirements"]


def test_independent_parent_anchor_is_still_insufficient_without_basis_and_frame():
    result = audit_anchor_status(
        measured_parent_orientation=True, independent_direction_count=0,
        indexed_parent_phase_present=True, verified_change_of_basis=False,
        scan_frame_verified=False,
    )
    assert not result["physical_parent_or_certified"]
    assert len(result["blocking_requirements"]) == 2


def test_noncollinear_pair_internal_angle_must_match():
    cubic = Lattice(3, 3, 3, 90, 90, 90)
    with pytest.raises(ValueError, match="angle inconsistent"):
        parent_orientation_from_two_directions(
            cubic, (1, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)
        )
    with pytest.raises(ValueError, match="collinear"):
        parent_orientation_from_two_directions(
            cubic, (1, 0, 0), (1, 0, 0), (1, 0, 0), (0, 1, 0)
        )


def test_fully_declared_evidence_still_requires_external_scientific_verification():
    result = audit_anchor_status(
        measured_parent_orientation=True, independent_direction_count=2,
        indexed_parent_phase_present=True, verified_change_of_basis=True,
        scan_frame_verified=True,
    )
    assert result["declared_prerequisites_present"]
    assert result["physical_parent_or_certified"] is False


def test_rational_one_third_subcell_metric_is_supported_but_not_certified():
    conventional = Lattice(3, 3, 9, 90, 90, 90)
    reduced = Lattice(3, 3, 3, 90, 90, 90)
    P = [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1/3"]]
    audit = audit_direct_basis(conventional, reduced, P)
    assert audit.volume_ratio_new_over_old == pytest.approx(1/3)
    assert audit.absolute_volume_factor is None
    assert audit.integral_basis is False
    assert audit.metric_relative_residual < 1e-12
    assert "unverified" in audit.status.lower()


def test_relaxed_literature_metric_screen_does_not_turn_into_proof():
    conventional = Lattice(3, 3, 9, 90, 90, 90)
    measured = Lattice(3.01, 3.00, 3.0, 90, 90, 90)
    P = [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1/3"]]
    with pytest.raises(ValueError, match="metric equality"):
        audit_direct_basis(conventional, measured, P)
    guess = audit_direct_basis(conventional, measured, P, relative_tolerance=0.03)
    assert 0 < guess.metric_relative_residual < 0.03
    assert guess.integral_basis is False
    assert "unverified" in guess.status.lower()

def test_noisy_two_axis_anchor_uses_unbiased_least_squares_not_first_axis_only():
    beta = Lattice.cubic(5.8)
    d1 = lattice_direction_to_cart(beta,(1,0,0))
    d2 = lattice_direction_to_cart(beta,(0,1,0))
    a = Rotation.from_euler('y',1.0,degrees=True).apply(d1)
    b = Rotation.from_euler('x',-1.0,degrees=True).apply(d2)
    result = parent_orientation_from_two_directions(
        beta,(1,0,0),(0,1,0),a,b,
    )
    errors = np.rad2deg(np.arccos(np.clip(
        [np.dot(result @ d1,a),np.dot(result @ d2,b)],-1,1,
    )))
    assert 0.005 < errors[0] < 0.02
    assert 0.005 < errors[1] < 0.02
    # First-axis-exact triangulation has twice the least-squares error.
    first = a / np.linalg.norm(a)
    second = b - np.dot(b,first)*first
    second /= np.linalg.norm(second)
    old = np.column_stack((first,second,np.cross(first,second)))
    old_errors = np.rad2deg(np.arccos(np.clip(
        [np.dot(old @ d1,a),np.dot(old @ d2,b)],-1,1,
    )))
    assert np.sum(errors**2) < 0.75*np.sum(old_errors**2)


@pytest.mark.parametrize('tolerance',[float('nan'),float('inf'),-1.0])
def test_two_axis_anchor_rejects_invalid_angular_tolerance(tolerance):
    beta=Lattice.cubic(5.8)
    with pytest.raises(ValueError,match='tolerance'):
        parent_orientation_from_two_directions(
            beta,(1,0,0),(0,1,0),(1,0,0),(0,1,0),
            max_pair_angular_mismatch_deg=tolerance,
        )


def test_corrected_parent_fit_is_returned_to_raw_oxford_sample_frame():
    raw=Rotation.from_euler('xyz',[14,-18,31],degrees=True).as_matrix()
    H=Rotation.from_euler('y',180,degrees=True).as_matrix()
    recovered=parent_fit_in_raw_oxford_frame({
        "parent_crystal_to_sample_rotation":(H @ raw).tolist(),
        "explicit_frame_correction_matrix":H.tolist(),
    })
    assert np.allclose(recovered,raw,atol=1e-12)
    unchanged=parent_fit_in_raw_oxford_frame({
        "parent_crystal_to_sample_rotation":raw.tolist(),
        "explicit_frame_correction_matrix":None,
    })
    assert np.allclose(unchanged,raw,atol=1e-12)
    with pytest.raises(ValueError):
        parent_fit_in_raw_oxford_frame({
            "parent_crystal_to_sample_rotation":raw.tolist(),
            "explicit_frame_correction_matrix":np.diag([1,1,-1]).tolist(),
        })
