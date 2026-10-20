"""Independent exact parity contracts for Cayron paired-supercell and matrix entry."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import sympy as sp

from app.application import TransformationInput
from twin_app.cayron_correspondence import (
    calculate_from_paired_vectors, direction_labels, entered_matrix_from_canonical,
    parse_triplet, plane_image, seed_paired_vectors,
)
from twin_app.input_logic import exact_matrix, canonical_correspondence_rows
from twin_app.correspondence_explorer import calculate_action

ROOT = Path(__file__).resolve().parents[2]


def _vecs(matrix):
    """The three COLUMN lattice vectors, for paired-supercell input."""
    return [[str(matrix[i, j]) for i in range(3)] for j in range(3)]


def _matrix_rows(matrix):
    """Actual matrix ROWS, never column vectors, for direct entry."""
    return [[str(matrix[i, j]) for j in range(3)] for i in range(3)]


@pytest.mark.parametrize("benchmark", [
    "bhattacharya_niti_twin_habit_v1.json",
    "bhattacharya_cualni_orthorhombic_habit_v1.json",
])
@pytest.mark.parametrize("direction", ["A_TO_M", "M_TO_A"])
def test_real_input_benchmarks_are_exactly_equivalent_in_both_editors(benchmark, direction):
    inputs = json.loads((ROOT / "data" / "benchmarks" / benchmark).read_text())["input"]
    C = exact_matrix(inputs["C_M_from_A"])
    D = C if direction == "A_TO_M" else C.inv()
    # Nontrivial independent source vectors -- no identity-shortcut used in the test.
    P = sp.Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    Q = D * P
    paired = calculate_from_paired_vectors(_vecs(P), _vecs(Q), direction=direction)
    direct = canonical_correspondence_rows(
        [[str(D[i,j]) for j in range(3)] for i in range(3)], direction=direction
    )
    assert paired.canonical_M_from_A == direct
    assert exact_matrix(direct) == C
    # The real adapter feeding the science engine receives identical exact inputs.
    from_pairs = TransformationInput.from_rows("T", "A", "M", paired.canonical_M_from_A).to_payload()
    from_matrix = TransformationInput.from_rows("T", "A", "M", direct).to_payload()
    assert from_pairs == from_matrix
    assert from_matrix["correspondence_M_from_A"] == [[str(C[i,j]) for j in range(3)] for i in range(3)]
    assert exact_matrix(entered_matrix_from_canonical(direct, direction=direction)) == D
    for v in (sp.Matrix([1,0,0]), sp.Matrix([0,1,0]), sp.Matrix([1,1,-2])):
        action = calculate_action(entered_matrix_from_canonical(direct, direction=direction), v)
        assert sp.Matrix(action.daughter) == D * v
        assert D.inv() * sp.Matrix(action.daughter) == v
    # Reciprocal-space incidence must be invariant in both directions.
    h = sp.Matrix([1, -2, 3])
    h_dst = sp.Matrix([sp.Rational(z) for z in plane_image(
        entered_matrix_from_canonical(direct, direction=direction), tuple(h)
    )])
    assert h_dst == D.inv().T * h
    assert (h_dst.T * (D * sp.Matrix([2, 0, -1])))[0] == (h.T * sp.Matrix([2,0,-1]))[0]


def test_random_mixed_rational_matrix_and_supercells_are_algebraically_identical():
    rng = random.Random(2019)
    accepted = 0
    while accepted < 36:
        C = sp.Matrix([
            [sp.Rational(rng.randint(-6,6), rng.randint(1,7)) for _ in range(3)]
            for _ in range(3)
        ])
        P = sp.Matrix([[rng.randint(-3,3) for _ in range(3)] for _ in range(3)])
        if not C.det() or not P.det():
            continue
        for direction in ("A_TO_M","M_TO_A"):
            D = C if direction == "A_TO_M" else C.inv()
            Q = D * P
            result = calculate_from_paired_vectors(_vecs(P), _vecs(Q), direction=direction)
            direct = canonical_correspondence_rows(_matrix_rows(D), direction=direction)
            assert result.canonical_M_from_A == direct
            assert exact_matrix(result.canonical_M_from_A) == C
        accepted += 1


def test_direction_labels_always_read_from_left_to_right():
    assert direction_labels("A_TO_M") == ("Parent crystal", "Product crystal")
    assert direction_labels("M_TO_A") == ("Product crystal", "Parent crystal")
    with pytest.raises(ValueError):
        direction_labels("wrong")
    with pytest.raises(ValueError):
        entered_matrix_from_canonical(((1,0,0),(0,1,0),(0,0,1)), direction="wrong")


def test_reverse_seeded_triplets_return_original_exact_C():
    C = (( "1/3","-2/5","0"), ("2","0","1/7"), ("0","1","1"))
    for direction in ("A_TO_M", "M_TO_A"):
        a,b = seed_paired_vectors(C, direction=direction)
        result = calculate_from_paired_vectors(
            [parse_triplet(t) for t in a],
            [parse_triplet(t) for t in b], direction=direction
        )
        assert exact_matrix(result.canonical_M_from_A) == exact_matrix(C)


def test_production_layout_uses_source_to_destination_preview_for_both_methods():
    code = (ROOT / "twin_app" / "input_ui.py").read_text()
    assert "guides[2].markdown(\"**→**\")" in code
    assert "cols[2].markdown(\"**→**\")" in code
    assert 'st.markdown(f"**Exact 3 × 3 correspondence · {source_name} → {destination_name}**")' in code
    assert "calculate_action(entered_matrix, parent_vector)" in code
    assert "images_of_parent_basis(entered_matrix)" in code
    assert "plane_image(entered_matrix," in code
    assert "arrow, equation = displayed_relation(direction)" in code
    assert "source_name, destination_name = direction_labels(direction)" in code
    assert "←" not in code[code.index("def render_correspondence_input("):code.index("def _standard_basis(")]


def test_asymmetric_matrix_regression_catches_column_row_transpose():
    C = sp.Matrix([[-sp.Rational(4,7), -sp.Rational(3,4), -2],
                   [2, sp.Rational(4,3), sp.Rational(6,7)],
                   [sp.Rational(4,7), sp.Rational(1,2), -sp.Rational(1,2)]])
    P = sp.Matrix([[1,1,0],[0,1,1],[0,0,1]])
    paired = calculate_from_paired_vectors(_vecs(P), _vecs(C*P), direction="A_TO_M")
    direct = canonical_correspondence_rows(_matrix_rows(C), direction="A_TO_M")
    assert paired.canonical_M_from_A == direct
    assert exact_matrix(direct) == C
    assert exact_matrix(direct) != C.T
