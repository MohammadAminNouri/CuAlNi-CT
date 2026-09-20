from dataclasses import replace

import numpy as np
import pytest
import sympy as sp

from cualni_cryst.group_theory import correspondence_groupoid
from cualni_cryst.operator_crosslock import OperatorCrosslockService
from cualni_cryst.orientation import OrientationService
from cualni_cryst.project_state import james_hane_6m_reference_project

EXPECTED_H = (
    (-1, 0, 0, 0, -1, 0, 0, 0, -1),
    (-1, 0, 0, 0, 1, 0, 0, 0, 1),
    (1, 0, 0, 0, -1, 0, 0, 0, -1),
    (1, 0, 0, 0, 1, 0, 0, 0, 1),
)

EXPECTED_DOUBLE_COSETS = (
    (
        (-1, 0, 0, 0, -1, 0, 0, 0, -1),
        (-1, 0, 0, 0, 1, 0, 0, 0, 1),
        (1, 0, 0, 0, -1, 0, 0, 0, -1),
        (1, 0, 0, 0, 1, 0, 0, 0, 1),
    ),
    (
        (-1, 0, 0, 0, -1, 0, 0, 0, 1),
        (-1, 0, 0, 0, 1, 0, 0, 0, -1),
        (1, 0, 0, 0, -1, 0, 0, 0, 1),
        (1, 0, 0, 0, 1, 0, 0, 0, -1),
    ),
    (
        (-1, 0, 0, 0, 0, -1, 0, -1, 0),
        (-1, 0, 0, 0, 0, 1, 0, 1, 0),
        (1, 0, 0, 0, 0, -1, 0, -1, 0),
        (1, 0, 0, 0, 0, 1, 0, 1, 0),
    ),
    (
        (-1, 0, 0, 0, 0, -1, 0, 1, 0),
        (-1, 0, 0, 0, 0, 1, 0, -1, 0),
        (1, 0, 0, 0, 0, -1, 0, 1, 0),
        (1, 0, 0, 0, 0, 1, 0, -1, 0),
    ),
    (
        (0, -1, 0, -1, 0, 0, 0, 0, -1),
        (0, -1, 0, -1, 0, 0, 0, 0, 1),
        (0, -1, 0, 1, 0, 0, 0, 0, -1),
        (0, -1, 0, 1, 0, 0, 0, 0, 1),
        (0, 1, 0, -1, 0, 0, 0, 0, -1),
        (0, 1, 0, -1, 0, 0, 0, 0, 1),
        (0, 1, 0, 1, 0, 0, 0, 0, -1),
        (0, 1, 0, 1, 0, 0, 0, 0, 1),
    ),
    (
        (0, -1, 0, 0, 0, -1, -1, 0, 0),
        (0, -1, 0, 0, 0, -1, 1, 0, 0),
        (0, -1, 0, 0, 0, 1, -1, 0, 0),
        (0, -1, 0, 0, 0, 1, 1, 0, 0),
        (0, 1, 0, 0, 0, -1, -1, 0, 0),
        (0, 1, 0, 0, 0, -1, 1, 0, 0),
        (0, 1, 0, 0, 0, 1, -1, 0, 0),
        (0, 1, 0, 0, 0, 1, 1, 0, 0),
    ),
    (
        (0, 0, -1, -1, 0, 0, 0, -1, 0),
        (0, 0, -1, -1, 0, 0, 0, 1, 0),
        (0, 0, -1, 1, 0, 0, 0, -1, 0),
        (0, 0, -1, 1, 0, 0, 0, 1, 0),
        (0, 0, 1, -1, 0, 0, 0, -1, 0),
        (0, 0, 1, -1, 0, 0, 0, 1, 0),
        (0, 0, 1, 1, 0, 0, 0, -1, 0),
        (0, 0, 1, 1, 0, 0, 0, 1, 0),
    ),
    (
        (0, 0, -1, 0, -1, 0, -1, 0, 0),
        (0, 0, -1, 0, -1, 0, 1, 0, 0),
        (0, 0, -1, 0, 1, 0, -1, 0, 0),
        (0, 0, -1, 0, 1, 0, 1, 0, 0),
        (0, 0, 1, 0, -1, 0, -1, 0, 0),
        (0, 0, 1, 0, -1, 0, 1, 0, 0),
        (0, 0, 1, 0, 1, 0, -1, 0, 0),
        (0, 0, 1, 0, 1, 0, 1, 0, 0),
    ),
)


def _flat_int(matrix) -> tuple[int, ...]:
    return tuple(round(float(value)) for value in list(sp.Matrix(matrix)))


def _project():
    return james_hane_6m_reference_project()


def _polar():
    project = _project()
    return project, OrientationService(project).polar_orientation("do3_to_6m_reference")


def test_polar_orientation_is_explicitly_bound_to_transformation():
    project, state = _polar()
    assert state.transformation_id == "do3_to_6m_reference"
    assert state.to_dict()["transformation_id"] == "do3_to_6m_reference"
    transformation = project.transformation(state.transformation_id)
    assert transformation.parent_phase_id == state.reference_phase_id
    assert transformation.product_phase_id == state.moving_phase_id


def test_unbound_or_is_never_matched_to_correspondence_by_phase_names():
    project = _project()
    service = OrientationService(project)
    state = service.state_from_axis_angle("do3", "6m", np.array([1.0, 2.0, 3.0]), 17.0)
    assert state.transformation_id == ""
    with pytest.raises(ValueError, match="not bound"):
        OperatorCrosslockService(project).build(state)


def test_reference_h_t_membership_is_exactly_truth_locked():
    project, state = _polar()
    topology = OrientationService(project).topology(state)
    h_t = tuple(
        sorted(
            _flat_int(item.reference_crystallographic_matrix)
            for item in topology.full_intersection
        )
    )
    assert h_t == EXPECTED_H


def test_reference_h_c_and_every_double_coset_membership_are_exactly_truth_locked():
    project, state = _polar()
    transformation = project.transformation(state.transformation_id)
    parent = [
        sp.Matrix(np.rint(matrix).astype(int))
        for matrix in project.phase("austenite_do3").symmetry_matrices()
    ]
    product = [
        sp.Matrix(np.rint(matrix).astype(int))
        for matrix in project.phase("martensite_long_period").symmetry_matrices()
    ]
    groupoid = correspondence_groupoid(parent, product, transformation.correspondence)

    h_c = tuple(sorted(_flat_int(matrix) for matrix in groupoid.subgroup))
    assert h_c == EXPECTED_H

    actual = tuple(
        tuple(_flat_int(matrix) for matrix in operator)
        for operator in groupoid.operators
    )
    assert actual == EXPECTED_DOUBLE_COSETS
    assert tuple(index + 1 for index in groupoid.inverse_operators) == (
        1,
        2,
        3,
        4,
        5,
        7,
        6,
        8,
    )


def test_formal_ambivalence_uses_inverse_double_coset_not_angle_or_generator_presence():
    project, state = _polar()
    operators = OrientationService(project).operators(state)
    by_index = {operator.index: operator for operator in operators}

    # Formal groupoid result: O6 and O7 are the complementary polar pair.
    assert by_index[6].inverse_operator_index == 7
    assert by_index[7].inverse_operator_index == 6
    assert by_index[6].cayron_class == "polar"
    assert by_index[7].cayron_class == "polar"

    # O4 is self-inverse even though it has no standard reflection/twofold generator.
    assert by_index[4].inverse_operator_index == 4
    assert by_index[4].ambivalent is True
    assert by_index[4].cayron_class == "ambivalent"
    assert by_index[4].contains_parent_reflection is False
    assert by_index[4].contains_parent_180_rotation is False


def test_crosslock_matches_all_eight_operators_by_exact_membership_not_by_angle():
    project, state = _polar()
    report = OperatorCrosslockService(project).build(state)
    assert report.exact_subgroups_equal is True
    assert report.one_to_one_operator_crosslock is True
    assert report.orientation_operator_count == 8
    assert report.correspondence_operator_count == 8

    assert {row.orientation_operator for row in report.rows} == set(range(1, 9))
    assert {row.correspondence_operator for row in report.rows} == set(range(1, 9))

    for row in report.rows:
        assert row.correspondence_operator is not None
        c_expected = set(EXPECTED_DOUBLE_COSETS[row.correspondence_operator - 1])
        row_members = {
            tuple(int(value) for value in member) for member in row.exact_parent_members
        }
        assert row_members == c_expected


def test_crosslock_emits_actual_ct_twin_elements_and_independent_bj_checks():
    project, state = _polar()
    report = OperatorCrosslockService(project).build(state)

    rows_with_twins = [row for row in report.rows if row.ct_twins]
    assert rows_with_twins
    assert sum(len(row.ct_twins) for row in rows_with_twins) > 0

    for row in rows_with_twins:
        kinds = {twin.kind for twin in row.ct_twins}
        assert kinds <= {"I", "II"}
        for twin in row.ct_twins:
            assert twin.shear > 0.0
            assert np.all(np.isfinite(twin.plane_m))
            assert np.all(np.isfinite(twin.direction_m))

    verified = [
        row.independent_check
        for row in report.rows
        if row.independent_check.relation_count
    ]
    assert verified
    assert all(check.status == "verified" for check in verified)
    assert max(check.max_plane_or_direction_angle_deg for check in verified) < 1e-8
    assert max(check.max_relative_shear_residual for check in verified) < 1e-11
    assert max(check.max_rank_one_residual for check in verified) < 1e-11


def test_generic_explicitly_bound_or_reports_topology_difference_without_fake_mapping():
    project = _project()
    service = OrientationService(project)
    state = service.state_from_axis_angle(
        "do3",
        "6m",
        np.array([1.0, 2.0, 3.0]),
        17.0,
        transformation_id="do3_to_6m_reference",
    )
    report = OperatorCrosslockService(project).build(state)

    assert report.exact_subgroups_equal is False
    assert report.one_to_one_operator_crosslock is False
    assert all(row.correspondence_operator is None for row in report.rows)


def test_project_state_rejects_orientation_bound_to_unknown_transformation():
    project, state = _polar()
    bad = replace(state, transformation_id="does_not_exist")
    with pytest.raises(ValueError, match="unknown transformation"):
        replace(project, orientations=(bad,))
