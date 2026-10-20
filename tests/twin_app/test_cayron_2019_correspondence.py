"""Cayron (2019) supercell correspondence and exact rational UI contracts."""
from __future__ import annotations

from pathlib import Path
from fractions import Fraction

import pytest
import sympy as sp

from twin_app.cayron_correspondence import (
    calculate_from_paired_vectors, parse_triplet, plane_image, seed_paired_vectors,
)
from twin_app.input_logic import canonical_correspondence_rows, exact_matrix
from twin_app.input_explainer import preview_correspondence

NITI = (
    ("0", "0", "1"),
    ("1/2", "1/2", "0"),
    ("-1/2", "1/2", "0"),
)


def test_cayron_three_pairs_generate_niti_exactly_no_floating_rounding():
    source, destination = seed_paired_vectors(NITI, direction="A_TO_M")
    value = calculate_from_paired_vectors(
        [parse_triplet(t) for t in source],
        [parse_triplet(t) for t in destination],
        direction="A_TO_M",
    )
    assert value.canonical_M_from_A == NITI
    assert value.source_columns == (("1", "0", "0"), ("0", "1", "0"), ("0", "0", "1"))
    assert preview_correspondence(value.canonical_M_from_A, direction="A_TO_M").determinant == "1/2"


def test_arbitrary_unimodular_supercells_preserve_exact_correspondence():
    C = exact_matrix(NITI)
    # Nontrivial source supercell with determinant 1.
    P = sp.Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    Q = C * P
    source = [[str(P[i, j]) for i in range(3)] for j in range(3)]
    destination = [[str(Q[i, j]) for i in range(3)] for j in range(3)]
    result = calculate_from_paired_vectors(source, destination, direction="A_TO_M")
    assert result.canonical_M_from_A == NITI
    assert exact_matrix(result.source_to_destination) * P == Q


def test_reverse_paired_vectors_are_exact_inverse_to_canonical():
    source, destination = seed_paired_vectors(NITI, direction="M_TO_A")
    result = calculate_from_paired_vectors(
        [parse_triplet(v) for v in source],
        [parse_triplet(v) for v in destination],
        direction="M_TO_A",
    )
    assert result.canonical_M_from_A == NITI
    assert exact_matrix(result.source_to_destination) == exact_matrix(NITI).inv()


def test_exact_rational_decimal_is_not_guessed_to_be_one_third():
    vector = parse_triplet("0.3333, 1/3, -2/6")
    assert vector == ("3333/10000", "1/3", "-1/3")
    assert Fraction(vector[0]) != Fraction(1, 3)


def test_exact_reciprocal_plane_transformation_preserves_incidence():
    C = exact_matrix(NITI)
    parent_plane = sp.Matrix([1, 1, 0])
    mapped_h = sp.Matrix([sp.Rational(x) for x in plane_image(NITI, ["1", "1", "0"])])
    assert mapped_h == C.inv().T * parent_plane
    for vector in (sp.Matrix([1, -1, 0]), sp.Matrix([0, 0, 1])):
        assert (parent_plane.T * vector)[0] == 0
        assert (mapped_h.T * (C * vector))[0] == 0


@pytest.mark.parametrize("vectors,other,direction", [
    (("1,0,0", "2,0,0", "0,0,1"), ("1,0,0", "0,1,0", "0,0,1"), "A_TO_M"),
    (("1,0,0", "0,1,0", "0,0,1"), ("1,0,0", "2,0,0", "0,0,1"), "M_TO_A"),
])
def test_degenerate_supercells_fail_closed(vectors, other, direction):
    with pytest.raises(ValueError, match="linearly dependent"):
        calculate_from_paired_vectors(
            [parse_triplet(x) for x in vectors],
            [parse_triplet(x) for x in other], direction=direction
        )


def test_no_unverified_correspondence_inferred_from_lattice_metrics():
    root = Path(__file__).resolve().parents[2]
    ui = (root / "twin_app" / "input_ui.py").read_text()
    assert "Three paired lattice vectors" in ui
    assert "Exact rational 3 × 3 matrix" in ui
    assert "calculate_from_paired_vectors(" in ui
    assert "C_{A\\to M}" in ui
    assert "calculate_action(entered_matrix, parent_vector)" in ui
    assert "plane_image(entered_matrix" in ui
    assert "source_name, destination_name" in ui
    backend = (root / "twin_app" / "cayron_correspondence.py").read_text()
    assert "Q * P.inv()" in backend
    assert "C.inv().T" in backend


def test_invalid_editor_switch_returns_to_original_editable_method():
    source = (Path(__file__).resolve().parents[2] / "twin_app" / "input_ui.py").read_text()
    assert "st.session_state[method_key] = previous_method" in source
    assert "st.session_state[dir_key] = (" in source
    assert "st.warning(st.session_state.pop(warning_key))" in source
