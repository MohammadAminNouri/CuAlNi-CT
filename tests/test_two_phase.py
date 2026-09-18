import json

import numpy as np
import pytest

from cualni_cryst.crystal_objects import Direction, Plane
from cualni_cryst.project_state import james_hane_6m_reference_project
from cualni_cryst.representation import CartesianConvention
from cualni_cryst.two_phase import (
    TwoPhaseWorkbench,
    WorkbenchSide,
)


@pytest.fixture
def project():
    return james_hane_6m_reference_project()


@pytest.fixture
def workbench(project):
    return TwoPhaseWorkbench(project)


@pytest.fixture
def polar(workbench):
    return workbench.resolve_orientation("polar:do3_to_6m_reference")


def test_summary_is_general_and_ptclab_compatible(workbench, polar):
    report = workbench.summary(polar)
    assert report.reference_phase_id == "austenite_do3"
    assert report.moving_phase_id == "martensite_long_period"
    assert report.variant_count == 12
    assert np.all(np.isfinite(report.ptclab_matrix_reference_from_moving))
    assert "General two-phase crystallography" in report.note
    json.dumps(report.to_dict())


def test_coordinate_matrices_are_mutual_inverses(workbench, polar):
    report = workbench.coordinate_matrices(polar)
    assert report.direct_inverse_residual < 1e-12
    assert report.plane_inverse_residual < 1e-12

    A = np.asarray(report.direct_reference_from_moving)
    P = np.asarray(report.plane_reference_from_moving)

    # Duality contract for one linear direct-coordinate transform:
    # p_ref = A^-T p_mov.
    assert np.allclose(P, np.linalg.inv(A).T, atol=1e-12)


def test_coordinate_matrix_matches_direction_service_mapping(
    workbench,
    polar,
    project,
):
    moving = project.phase("martensite_long_period")
    direction = Direction((1.0, -2.0, 3.0), moving.basis)
    mapped = workbench.orientations.map_direction(polar, direction)

    A = np.asarray(workbench.coordinate_matrices(polar).direct_reference_from_moving)
    assert np.allclose(A @ direction.array, mapped.array, atol=1e-12)


def test_coordinate_matrix_matches_plane_service_mapping(
    workbench,
    polar,
    project,
):
    moving = project.phase("martensite_long_period")
    plane = Plane((2.0, 1.0, -1.0), moving.basis)
    mapped = workbench.orientations.map_plane(polar, plane)

    P = np.asarray(workbench.coordinate_matrices(polar).plane_reference_from_moving)
    assert np.allclose(P @ plane.array, mapped.array, atol=1e-12)


def test_mapping_roundtrip_is_machine_precision(workbench, polar):
    for side, text in (
        (WorkbenchSide.MOVING, "[1 2 -1]"),
        (WorkbenchSide.MOVING, "(2 -1 1)"),
        (WorkbenchSide.REFERENCE, "[1 -1 2]"),
        (WorkbenchSide.REFERENCE, "(1 2 1)"),
    ):
        report = workbench.map_object(
            polar,
            side,
            text,
            max_index=8,
        )
        assert report.roundtrip_projective_residual < 1e-12
        assert np.all(np.isfinite(report.exact_target_coefficients))


def test_pair_matches_existing_orientation_service_angle(workbench, polar):
    report = workbench.pair(
        polar,
        "[1 1 0]",
        "(1 0 1)",
        max_index=6,
    )
    old = workbench.orientations.compare_two_phase_objects(
        polar,
        "[1 1 0]",
        "(1 0 1)",
    )
    assert report.relation == old.relation
    assert np.isclose(report.angle_deg, old.angle_deg, atol=1e-12)


def test_same_kind_pair_reports_length_misfit(workbench, polar):
    report = workbench.pair(
        polar,
        "[1 0 0]",
        "[1 0 0]",
        max_index=4,
    )
    old = workbench.orientations.geometric_misfit(
        polar,
        "[1 0 0]",
        "[1 0 0]",
    )
    assert np.isclose(
        report.signed_relative_misfit_percent,
        old.signed_relative_percent,
        atol=1e-12,
    )
    assert report.reference_quantity_name == "direct_length"
    assert report.moving_quantity_name == "direct_length"


def test_plane_pair_reports_spacing_misfit(workbench, polar):
    report = workbench.pair(
        polar,
        "(1 0 0)",
        "(1 0 0)",
        max_index=4,
    )
    assert report.reference_quantity_name == "interplanar_spacing"
    assert report.moving_quantity_name == "interplanar_spacing"
    assert report.absolute_relative_misfit_percent is not None


def test_mixed_pair_has_no_invalid_length_misfit(workbench, polar):
    report = workbench.pair(
        polar,
        "[1 0 0]",
        "(1 0 0)",
        max_index=4,
    )
    assert report.relation == "direction-plane"
    assert report.signed_relative_misfit_percent is None
    assert report.absolute_relative_misfit_percent is None


def test_identity_or_maps_a_axis_exactly_between_ptclab_frames(workbench):
    state = workbench.resolve_orientation("identity")
    mapped = workbench.map_object(
        state,
        WorkbenchSide.MOVING,
        "[1 0 0]",
        max_index=3,
    )
    assert mapped.nearest_low_index.notation == "[1 0 0]"
    assert mapped.nearest_low_index.exact_within_tolerance
    assert mapped.nearest_low_index.angular_mismatch_deg < 1e-10


def test_variant_angle_table_has_one_row_per_crystallographic_variant(
    workbench,
    polar,
):
    table = workbench.variant_angles(
        polar,
        "[1 0 0]",
        "[1 0 0]",
        max_index=4,
    )
    assert len(table.rows) == 12
    assert [row.angle_deg for row in table.rows] == sorted(
        row.angle_deg for row in table.rows
    )
    assert {row.variant_index for row in table.rows} == set(range(1, 13))


def test_equivalent_angle_table_is_sorted_and_bounded(workbench, polar):
    table = workbench.equivalent_angles(
        polar,
        "[1 0 0]",
        "[1 0 0]",
        limit=7,
    )
    assert 1 <= len(table.rows) <= 7
    assert table.total_pairs >= len(table.rows)
    assert [row.angle_deg for row in table.rows] == sorted(
        row.angle_deg for row in table.rows
    )


def test_equivalent_angle_proper_only_path(workbench, polar):
    full = workbench.equivalent_angles(
        polar,
        "[1 2 3]",
        "[1 0 1]",
        limit=1000,
        proper_only=False,
    )
    proper = workbench.equivalent_angles(
        polar,
        "[1 2 3]",
        "[1 0 1]",
        limit=1000,
        proper_only=True,
    )
    assert proper.total_pairs <= full.total_pairs


@pytest.mark.parametrize(
    "spec",
    [
        "identity",
        "matrix:1 0 0; 0 1 0; 0 0 1",
        "euler:0 0 0",
        "quat:1 0 0 0",
        "axis:0 0 1; 0",
        "axis-crystal:[1 0 0]; 0",
        "parallel:(0 1 0)|(0 1 0)|[1 0 0]|[1 0 0]",
    ],
)
def test_supported_orientation_inputs_resolve(workbench, spec):
    state = workbench.resolve_orientation(spec)
    matrix = np.asarray(state.R_reference_from_moving)
    assert np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(matrix), 1.0, atol=1e-10)


def test_explicit_binding_is_preserved_for_user_or(workbench):
    state = workbench.resolve_orientation(
        "identity",
        bind_transformation="do3_to_6m_reference",
    )
    assert state.transformation_id == "do3_to_6m_reference"


def test_invalid_parallel_candidate_fails_loudly(workbench):
    with pytest.raises(ValueError, match="parallel_candidate"):
        workbench.resolve_orientation(
            "parallel:(0 1 0)|(0 1 0)|[1 0 0]|[1 0 0]",
            parallel_candidate=999,
        )


def test_invalid_low_index_bound_fails_loudly(workbench, polar):
    with pytest.raises(ValueError, match="max_index"):
        workbench.map_object(
            polar,
            WorkbenchSide.MOVING,
            "[1 0 0]",
            max_index=0,
        )


def test_pair_json_is_serializable(workbench, polar):
    payload = workbench.pair(
        polar,
        "[1 0 0]",
        "[1 0 0]",
        max_index=3,
    ).to_dict()
    json.dumps(payload)


def test_pair_angle_is_invariant_under_cartesian_reexpression(
    workbench,
    polar,
):
    baseline = workbench.pair(
        polar,
        "[1 2 -1]",
        "(2 0 1)",
        max_index=3,
    ).angle_deg
    for ref_convention in workbench.orientations.report(polar).representations:
        # The report contains every pair, so use its exact named conventions.
        state = workbench.orientations.reexpress(
            polar,
            CartesianConvention(ref_convention.reference_convention),
            CartesianConvention(ref_convention.moving_convention),
        )
        current = workbench.pair(
            state,
            "[1 2 -1]",
            "(2 0 1)",
            max_index=3,
        ).angle_deg
        assert np.isclose(current, baseline, atol=1e-10)
