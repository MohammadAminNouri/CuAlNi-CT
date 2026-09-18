import json
from dataclasses import replace

import numpy as np
import pytest

from cualni_cryst.calculation_service import CalculationService
from cualni_cryst.crystal_objects import Direction, Plane
from cualni_cryst.orientation import (
    EulerConvention,
    OrientationService,
    axis_angle_from_matrix,
    closest_proper_rotation,
    euler_zxz_from_matrix,
    matrix_from_axis_angle,
    matrix_from_euler_zxz,
    matrix_from_quaternion_wxyz,
    misorientation_angle_deg,
    quaternion_wxyz_from_matrix,
    rotation_audit,
)
from cualni_cryst.project_state import (
    OrientationDefinition,
    OrientationTheoryOrigin,
    james_hane_6m_reference_project,
)
from cualni_cryst.representation import CartesianConvention


@pytest.fixture
def service():
    return OrientationService(james_hane_6m_reference_project())


def test_axis_angle_round_trip_is_rotation_exact():
    R = matrix_from_axis_angle(np.array([1.0, 2.0, -3.0]), 73.2)
    result = axis_angle_from_matrix(R)
    rebuilt = matrix_from_axis_angle(np.asarray(result.axis), result.angle_deg)
    assert rotation_audit(R).maximum_residual < 1e-12
    assert np.allclose(rebuilt, R, atol=1e-12)


@pytest.mark.parametrize(
    "angles",
    [
        (10.0, 20.0, 30.0),
        (200.0, 90.0, 123.0),
        (40.0, 0.0, 50.0),
        (40.0, 180.0, 50.0),
    ],
)
def test_explicit_zxz_euler_round_trip(angles):
    for convention in EulerConvention:
        R = matrix_from_euler_zxz(*angles, convention=convention)
        recovered = euler_zxz_from_matrix(R, convention)
        rebuilt = matrix_from_euler_zxz(
            recovered.phi1_deg,
            recovered.Phi_deg,
            recovered.phi2_deg,
            convention,
        )
        assert np.allclose(rebuilt, R, atol=1e-12)


def test_standard_hamilton_quaternion_round_trip():
    R = matrix_from_axis_angle(np.array([1.0, -1.0, 2.0]), 128.0)
    q = quaternion_wxyz_from_matrix(R)
    rebuilt = matrix_from_quaternion_wxyz(np.asarray(q))
    assert np.allclose(rebuilt, R, atol=1e-12)
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-12)


def test_improper_or_nonorthogonal_matrix_is_not_silently_accepted(service):
    with pytest.raises(ValueError, match="proper rotation"):
        service.state_from_matrix("do3", "6m", np.diag([1.0, 1.0, -1.0]))

    noisy = np.eye(3)
    noisy[0, 1] = 1.0e-4
    with pytest.raises(ValueError, match="repair=True"):
        service.state_from_matrix("do3", "6m", noisy)


def test_explicit_so3_repair_is_bounded_and_reported(service):
    noisy = np.eye(3)
    noisy[0, 1] = 1.0e-5
    state = service.state_from_matrix(
        "do3",
        "6m",
        noisy,
        repair=True,
        maximum_repair_residual=1.0e-3,
    )
    assert (
        rotation_audit(np.asarray(state.R_reference_from_moving)).maximum_residual
        < 1e-12
    )
    assert "SO(3) projection" in state.notes


def test_closest_rotation_rejects_large_repair_by_service(service):
    matrix = np.diag([2.0, 1.0, 1.0])
    projected, correction = closest_proper_rotation(matrix)
    assert rotation_audit(projected).maximum_residual < 1e-12
    assert correction > 0.1
    with pytest.raises(ValueError, match="repair is too large"):
        service.state_from_matrix(
            "do3",
            "6m",
            matrix,
            repair=True,
            maximum_repair_residual=1.0e-3,
        )


def test_orientation_state_is_not_correspondence_and_serializes(service):
    state = service.state_from_matrix("do3", "6m", np.eye(3))
    payload = state.to_dict()
    assert state.definition_method is OrientationDefinition.USER_MATRIX
    assert state.theory_origin is OrientationTheoryOrigin.USER_DEFINED
    assert "R_reference_from_moving" in payload
    assert "correspondence" not in json.dumps(payload).lower()


def test_representation_reexpression_preserves_physical_mapping(service):
    state = service.state_from_axis_angle(
        "do3",
        "6m",
        np.array([1.0, 2.0, 1.0]),
        37.0,
    )
    assert service.representation_parity(state).maximum_residual < 1e-10


def test_all_nine_cartesian_representation_pairs_are_proper_rotations(service):
    state = service.state_from_axis_angle(
        "do3",
        "6m",
        np.array([0.0, 0.0, 1.0]),
        22.0,
    )
    report = service.report(state)
    assert len(report.representations) == 9
    assert all(item.audit.maximum_residual < 1e-10 for item in report.representations)


def test_parallelism_solver_exposes_sign_ambiguity_and_exact_residuals(service):
    report = service.state_from_parallelisms(
        "do3",
        "6m",
        "(0 1 0)",
        "(0 1 0)",
        "[1 0 0]",
        "[1 0 0]",
    )
    assert len(report.candidates) >= 2
    assert all(
        candidate.first_parallelism_residual_deg < 1e-7
        and candidate.second_parallelism_residual_deg < 1e-7
        for candidate in report.candidates
    )


def test_parallelism_solver_rejects_incompatible_internal_geometry(service):
    with pytest.raises(ValueError, match="No exact proper rotation"):
        service.state_from_parallelisms(
            "do3",
            "6m",
            "(1 0 0)",
            "(1 0 0)",
            "[1 1 0]",
            "[0 1 0]",
            unoriented_directions=False,
        )


def test_orientation_mapping_round_trip_is_independent_of_correspondence(service):
    project = service.project
    state = service.state_from_axis_angle(
        "do3",
        "6m",
        np.array([0.0, 1.0, 0.0]),
        17.0,
    )
    moving = project.phase("martensite_long_period")
    direction = Direction((1.0, 2.0, -1.0), moving.basis)
    plane = Plane((2.0, -1.0, 1.0), moving.basis)

    mapped_direction = service.map_direction(state, direction)
    mapped_plane = service.map_plane(state, plane)
    back_direction = service.map_direction(state, mapped_direction)
    back_plane = service.map_plane(state, mapped_plane)

    assert np.allclose(back_direction.array, direction.array, atol=1e-10)
    assert np.allclose(back_plane.array, plane.array, atol=1e-10)
    assert "not correspondence mapping" in mapped_direction.provenance.notes


def test_polar_candidate_is_transpose_of_bridge_polar_rotation(service):
    state = service.polar_orientation("do3_to_6m_reference")
    bridge_rotation = service.project.bridge(
        "do3_to_6m_reference"
    ).polar_rotation_cartesian()
    assert state.theory_origin is OrientationTheoryOrigin.POLAR_CORRESPONDENCE
    assert np.allclose(
        np.asarray(state.R_reference_from_moving),
        bridge_rotation.T,
        atol=1e-12,
    )


def test_polar_report_refuses_cayron_t_equivalence(service):
    report = service.report(service.polar_orientation("do3_to_6m_reference"))
    joined = " ".join(report.warnings)
    assert "not automatically Cayron" in joined
    assert "not Cayron's crystallographic metric quaternion" in joined


def test_orientation_variants_use_only_proper_point_group_operations(service):
    state = service.state_from_matrix("do3", "6m", np.eye(3))
    report = service.report(state)
    variants = service.variants(state)
    assert report.proper_reference_symmetry_order == 24
    assert report.proper_moving_symmetry_order == 2
    assert 1 <= len(variants) <= 48


def test_symmetry_equivalent_or_has_zero_disorientation(service):
    state = service.state_from_axis_angle(
        "do3",
        "6m",
        np.array([1.0, 1.0, 1.0]),
        31.0,
    )
    variant = service.variants(state)[-1]
    equivalent = replace(
        state,
        orientation_id="equivalent",
        R_reference_from_moving=variant.matrix_reference_from_moving,
    )
    comparison = service.compare_orientations(state, equivalent)
    assert comparison.symmetry_reduced_disorientation_deg < 1e-7


def test_two_phase_angle_identity_example_is_interpretable(service):
    state = service.state_from_matrix("do3", "6m", np.eye(3))
    report = service.compare_two_phase_objects(
        state,
        "[1 0 0]",
        "[1 0 0]",
    )
    assert report.relation == "direction-direction"
    assert np.isclose(report.angle_deg, 0.0, atol=1e-12)


def test_geometric_misfit_has_explicit_reference_denominator(service):
    state = service.state_from_matrix("do3", "6m", np.eye(3))
    report = service.geometric_misfit(
        state,
        "[1 0 0]",
        "[1 0 0]",
    )
    project = service.project
    a_ref = project.phase("austenite_do3").lattice.a
    a_mov = project.phase("martensite_long_period").lattice.a
    assert np.isclose(
        report.signed_relative_percent,
        100.0 * (a_mov - a_ref) / a_ref,
    )


def test_project_orientation_does_not_invalidate_ct_metric_nodes(service):
    state = service.state_from_matrix("do3", "6m", np.eye(3))
    proposed = replace(service.project, orientations=(state,))
    proposed.validate().assert_passed()
    impact = CalculationService(service.project).impact(
        proposed,
        "do3_to_6m_reference",
    )
    assert not impact.changed


def test_project_payload_includes_registered_orientation(service):
    state = service.state_from_matrix("do3", "6m", np.eye(3))
    payload = replace(service.project, orientations=(state,)).to_dict()
    assert payload["orientations"][0]["orientation_id"] == state.orientation_id
    json.dumps(payload)


def test_misorientation_identity_is_zero():
    R = matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]), 50.0)
    assert np.isclose(misorientation_angle_deg(R, R), 0.0, atol=1e-12)


def test_ptclab_reexpression_exists_in_report(service):
    report = service.report(service.polar_orientation("do3_to_6m_reference"))
    pairs = {
        (item.reference_convention, item.moving_convention)
        for item in report.representations
    }
    assert (
        CartesianConvention.PTCLAB_A_X_C_XZ.value,
        CartesianConvention.PTCLAB_A_X_C_XZ.value,
    ) in pairs
