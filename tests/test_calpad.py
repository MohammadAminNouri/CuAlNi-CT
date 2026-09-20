import json

import numpy as np
import pytest

from cualni_cryst.calpad import (
    AngleSense,
    CalPadRenderer,
    CalPadService,
    CandidateKind,
    primitive_low_index_triplets,
)
from cualni_cryst.crystal_objects import Direction, Plane, interplanar_angle_deg
from cualni_cryst.project_state import james_hane_6m_reference_project


@pytest.fixture
def service():
    return CalPadService(james_hane_6m_reference_project())


def test_primitive_low_index_generation_removes_multiples_and_sign_duplicates():
    triplets = primitive_low_index_triplets(2, projective=True)

    assert (1, 0, 1) in triplets
    assert (2, 0, 2) not in triplets
    assert (-1, 0, -1) not in triplets
    assert (0, 0, 0) not in triplets


def test_phase_cell_reports_direct_and_reciprocal_metrics_consistently(service):
    report = service.phase_cell("6m")

    assert report.length_unit == "angstrom"
    assert report.symmetry_order == 4
    assert report.point_group_symbol == "2/m (unique b)"
    assert np.allclose(
        report.direct_metric @ report.reciprocal_metric,
        np.eye(3),
        atol=1e-12,
    )
    assert report.metric_inverse_residual < 1e-12
    assert np.isclose(
        report.volume * report.reciprocal_cell.volume_star,
        1.0,
        atol=1e-12,
    )


def test_cubic_plane_normal_conversion_recovers_same_low_index_axis(service):
    report = service.normal_conversion("do3", "(1 2 3)", max_index=4)

    assert report.target_kind is CandidateKind.DIRECTION
    assert report.nearest_low_index.notation == "<1 2 3>"
    assert report.nearest_low_index.angular_mismatch_deg < 1e-10
    assert report.nearest_low_index.exact_within_numerical_tolerance
    assert report.roundtrip_projective_residual < 1e-12


def test_cubic_direction_to_normal_plane_recovers_same_indices(service):
    report = service.normal_conversion("do3", "[1 2 3]", max_index=4)

    assert report.target_kind is CandidateKind.PLANE
    assert report.nearest_low_index.notation == "{1 2 3}"
    assert report.nearest_low_index.angular_mismatch_deg < 1e-10
    assert report.nearest_low_index.exact_within_numerical_tolerance


def test_monoclinic_plane_normal_is_not_falsely_identified_with_same_triplet(service):
    report = service.normal_conversion("6m", "(1 0 1)", max_index=12)

    projective = np.asarray(report.projective_target_coefficients)
    assert not np.allclose(projective, np.array([1.0, 0.0, 1.0]))
    assert report.nearest_low_index.angular_mismatch_deg > 0.0
    assert not report.nearest_low_index.exact_within_numerical_tolerance
    assert report.roundtrip_projective_residual < 1e-12


def test_monoclinic_direction_to_plane_uses_metric_not_cubic_shortcut(service):
    report = service.normal_conversion("6m", "[1 0 1]", max_index=12)

    projective = np.asarray(report.projective_target_coefficients)
    assert not np.allclose(projective, np.array([1.0, 0.0, 1.0]))
    assert report.roundtrip_projective_residual < 1e-12


def test_low_index_direction_table_cubic_has_zero_angle_target_axis(service):
    report = service.low_index_table(
        "do3",
        "[1 0 0]",
        candidate_kind="direction",
        max_index=2,
        limit=10,
    )

    assert report.candidate_kind is CandidateKind.DIRECTION
    assert report.angle_sense is AngleSense.PROJECTIVE
    assert report.rows[0].notation == "<1 0 0>"
    assert np.isclose(report.rows[0].angle_deg, 0.0)


def test_oriented_direction_table_distinguishes_opposite_sense(service):
    report = service.low_index_table(
        "do3",
        "[1 0 0]",
        candidate_kind="direction",
        max_index=1,
        limit=30,
        angle_sense="oriented",
    )

    angles = {row.notation: row.angle_deg for row in report.rows}
    assert np.isclose(angles["[1 0 0]"], 0.0)
    assert np.isclose(angles["[-1 0 0]"], 180.0)


def test_low_index_plane_table_uses_reciprocal_metric(service):
    report = service.low_index_table(
        "6m",
        "(1 0 1)",
        candidate_kind="plane",
        max_index=2,
        limit=20,
    )

    phase = service.console.resolve_phase("6m")
    target = Plane((1, 0, 1), phase.basis)
    first = Plane(report.rows[0].indices, phase.basis)
    expected = interplanar_angle_deg(target, first, phase.lattice)

    assert np.isclose(report.rows[0].angle_deg, expected, atol=1e-12)


def test_direction_plane_low_index_table_reports_incidence(service):
    report = service.low_index_table(
        "do3",
        "[1 0 0]",
        candidate_kind="plane",
        max_index=1,
        limit=20,
    )

    zero_angle_rows = [
        row for row in report.rows if np.isclose(row.angle_deg, 0.0, atol=1e-12)
    ]
    assert zero_angle_rows
    assert all(
        row.incidence_residual_value is not None
        and row.incidence_residual_value < 1e-12
        for row in zero_angle_rows
    )


def test_normal_conversion_cartesian_views_represent_same_physical_axis(service):
    report = service.normal_conversion("6m", "(1 0 1)")

    assert len(report.cartesian_unit_views) == 3
    for view in report.cartesian_unit_views:
        assert np.isclose(np.linalg.norm(view.coordinates), 1.0, atol=1e-12)


def test_renderer_is_explicit_about_exact_metric_and_approximate_low_index(service):
    report = service.normal_conversion("6m", "(1 0 1)", max_index=12)
    text = CalPadRenderer().normal_conversion(report, show_derivation=True)

    assert "EXACT METRIC RESULT" in text
    assert "NEAREST LOW-INDEX LATTICE OBJECT" in text
    assert "angular mismatch" in text
    assert "approximation only" in text
    assert "DERIVATION" in text


def test_all_reports_are_json_serializable(service):
    payload = {
        "cell": service.phase_cell("6m").to_dict(),
        "normal": service.normal_conversion("6m", "(1 0 1)").to_dict(),
        "low": service.low_index_table(
            "6m",
            "[1 0 1]",
            candidate_kind="plane",
            max_index=2,
            limit=10,
        ).to_dict(),
    }

    encoded = json.dumps(payload)
    assert '"reciprocal_cell"' in encoded
    assert '"nearest_low_index"' in encoded
    assert '"rows"' in encoded


def test_low_index_invalid_arguments_fail_loudly(service):
    with pytest.raises(ValueError, match="max_index"):
        primitive_low_index_triplets(0)

    with pytest.raises(ValueError, match="limit"):
        service.low_index_table("6m", "[1 0 1]", limit=0)


def test_metric_normal_conversion_matches_direct_manual_formula(service):
    phase = service.console.resolve_phase("6m")
    report = service.normal_conversion("6m", "(1 0 1)")
    p = np.array([1.0, 0.0, 1.0])
    expected = np.linalg.solve(phase.lattice.metric(), p)
    expected /= np.max(np.abs(expected))
    if expected[np.flatnonzero(np.abs(expected) > 1e-14)[0]] < 0:
        expected = -expected

    assert np.allclose(
        report.projective_target_coefficients,
        expected,
        atol=1e-12,
    )


def test_cubic_direction_length_and_plane_spacing_remain_independent(service):
    phase = service.console.resolve_phase("do3")
    direction = Direction((1, 1, 0), phase.basis)
    plane = Plane((1, 1, 0), phase.basis)

    assert direction.length(phase.lattice) > 1.0
    assert plane.spacing(phase.lattice) > 1.0
    assert not np.isclose(
        direction.length(phase.lattice),
        plane.spacing(phase.lattice),
    )
