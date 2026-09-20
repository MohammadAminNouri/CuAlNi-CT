import json

import numpy as np
import pytest

from cualni_cryst.crystal_objects import Direction, Plane
from cualni_cryst.crystallography_console import (
    ComparisonKind,
    ConsoleInputKind,
    ConsoleRenderer,
    CrystallographyConsole,
    parse_crystal_input,
)
from cualni_cryst.project_state import james_hane_6m_reference_project
from cualni_cryst.representation import CartesianConvention


@pytest.fixture
def console():
    return CrystallographyConsole(james_hane_6m_reference_project())


@pytest.mark.parametrize(
    ("text", "kind", "family", "canonical"),
    [
        ("[1 0 -1]", ConsoleInputKind.DIRECTION, False, "[1 0 -1]"),
        ("[1,0,−1]", ConsoleInputKind.DIRECTION, False, "[1 0 -1]"),
        ("(1 1 0)", ConsoleInputKind.PLANE, False, "(1 1 0)"),
        ("<1 0 0>", ConsoleInputKind.DIRECTION, True, "<1 0 0>"),
        ("⟨1 0 0⟩", ConsoleInputKind.DIRECTION, True, "<1 0 0>"),
        ("{1 1 0}", ConsoleInputKind.PLANE, True, "{1 1 0}"),
        ("[1/2 0 -1]", ConsoleInputKind.DIRECTION, False, "[1/2 0 -1]"),
    ],
)
def test_parser_accepts_human_friendly_safe_notation(
    text,
    kind,
    family,
    canonical,
):
    parsed = parse_crystal_input(text)

    assert parsed.kind is kind
    assert parsed.family is family
    assert parsed.canonical_text == canonical


def test_parser_refuses_ambiguous_or_conflicting_input():
    with pytest.raises(ValueError, match="Bare indices are ambiguous"):
        parse_crystal_input("1 0 1")

    with pytest.raises(ValueError, match="Refusing to guess"):
        parse_crystal_input("[1 0 1]", kind_hint="plane")

    with pytest.raises(ValueError, match="Exactly three"):
        parse_crystal_input("[10-1]")


def test_bare_input_is_allowed_only_with_explicit_kind():
    parsed = parse_crystal_input("1 0 -1", kind_hint="direction")

    assert parsed.kind is ConsoleInputKind.DIRECTION
    assert parsed.canonical_text == "[1 0 -1]"


def test_phase_resolution_is_easy_but_ambiguity_safe(console):
    assert console.resolve_phase("6m").phase_id == "martensite_long_period"
    assert console.resolve_phase("DO3").phase_id == "austenite_do3"

    with pytest.raises(ValueError, match="Unknown phase"):
        console.resolve_phase("not-a-phase")


def test_direction_inspection_matches_typed_object_and_all_cartesian_frames(console):
    report = console.inspect("6m", "[1 0 1]")
    phase = console.resolve_phase("6m")
    direction = Direction((1, 0, 1), phase.basis)

    assert report.parsed.kind is ConsoleInputKind.DIRECTION
    assert np.isclose(
        report.dimensional_quantity,
        direction.length(phase.lattice),
    )
    assert len(report.cartesian_views) == len(CartesianConvention)
    assert report.maximum_representation_residual < 1.0e-10
    assert report.secondary_quantity is None


def test_plane_inspection_reports_reciprocal_norm_and_d_spacing(console):
    report = console.inspect("6m", "(0 1 1)")
    phase = console.resolve_phase("6m")
    plane = Plane((0, 1, 1), phase.basis)

    assert report.parsed.kind is ConsoleInputKind.PLANE
    assert np.isclose(
        report.dimensional_quantity,
        plane.reciprocal_length(phase.lattice),
    )
    assert np.isclose(report.secondary_quantity, plane.spacing(phase.lattice))
    assert report.maximum_representation_residual < 1.0e-10


def test_cubic_family_expansion_respects_direction_and_plane_semantics(console):
    direction = console.inspect("do3", "<1 0 0>")
    plane = console.inspect("do3", "{1 0 0}")

    assert direction.symmetry_equivalent_count == 3
    assert plane.symmetry_equivalent_count == 3


def test_direction_direction_comparison_reports_oriented_and_axis_angles(console):
    report = console.compare("6m", "[1 0 0]", "[-1 0 0]")

    assert report.comparison_kind is ComparisonKind.DIRECTION_DIRECTION
    assert np.isclose(report.primary_value, 180.0)
    assert np.isclose(report.secondary_value, 0.0)
    assert report.maximum_representation_residual < 1.0e-10


def test_plane_plane_comparison_is_projective_for_primary_angle(console):
    report = console.compare("6m", "(1 0 1)", "(-1 0 -1)")

    assert report.comparison_kind is ComparisonKind.PLANE_PLANE
    assert np.isclose(report.primary_value, 0.0)
    assert np.isclose(report.secondary_value, 180.0)


def test_direction_plane_comparison_reports_incidence_and_angle(console):
    report = console.compare("6m", "[1 0 0]", "(0 1 0)")

    assert report.comparison_kind is ComparisonKind.DIRECTION_PLANE
    assert np.isclose(report.primary_value, 0.0)
    assert np.isclose(report.incidence_residual_value, 0.0)
    assert report.lies_in_plane_value is True
    assert report.maximum_representation_residual < 1.0e-10


def test_cross_phase_mapping_uses_correspondence_not_raw_index_comparison(console):
    report = console.map(
        "do3_to_6m_reference",
        "do3",
        "[1 0 0]",
    )

    assert report.source_phase_id == "austenite_do3"
    assert report.target_phase_id == "martensite_long_period"
    assert report.mapped_object_payload["kind"] == "direction"
    assert report.mapped_canonical_text.startswith("[")


def test_plane_mapping_preserves_plane_type(console):
    report = console.map(
        "do3_to_6m_reference",
        "do3",
        "(0 1 0)",
    )

    assert report.mapped_object_payload["kind"] == "plane"
    assert report.mapped_canonical_text.startswith("(")


def test_renderer_is_clear_and_contains_reliability_and_derivation(console):
    report = console.inspect("6m", "[1 0 1]")
    text = ConsoleRenderer().object_report(report, show_derivation=True)

    assert "METRIC" in text
    assert "CARTESIAN VIEWS" in text
    assert "RELIABILITY" in text
    assert "DERIVATION" in text
    assert "representation residual" in text


def test_structured_reports_are_json_serializable(console):
    object_payload = console.inspect("6m", "[1 0 1]").to_dict()
    comparison_payload = console.compare(
        "6m",
        "[1 0 1]",
        "(0 1 1)",
    ).to_dict()
    mapping_payload = console.map(
        "do3_to_6m_reference",
        "do3",
        "[1 0 0]",
    ).to_dict()

    encoded = json.dumps(
        {
            "object": object_payload,
            "comparison": comparison_payload,
            "mapping": mapping_payload,
        }
    )

    assert '"cartesian_views"' in encoded
    assert '"direction_plane"' in encoded
    assert '"mapped_canonical_text"' in encoded
