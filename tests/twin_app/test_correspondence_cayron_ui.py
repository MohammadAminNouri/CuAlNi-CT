"""Protect exact rational parent→daughter correspondence without hardcoding alloy inputs."""
from pathlib import Path

import sympy as sp
from twin_app.input_logic import canonical_correspondence_rows, exact_matrix

ROOT = Path(__file__).resolve().parents[2]


def test_exact_rational_correspondence_and_basis_column_action():
    rows = (("0", "0", "1"), ("1/2", "1/2", "0"), ("-1/2", "1/2", "0"))
    C = exact_matrix(canonical_correspondence_rows(rows, direction="A_TO_M"))
    assert C[1, 0] == sp.Rational(1, 2)
    assert C[2, 0] == -sp.Rational(1, 2)
    assert C*sp.Matrix([1,0,0]) == sp.Matrix([0,sp.Rational(1,2),-sp.Rational(1,2)])
    assert C.det() != 0


def test_noninvertible_correspondence_rejected():
    import pytest
    with pytest.raises(ValueError, match="invertible"):
        exact_matrix((("1", "0", "0"),("0", "1", "0"),("0", "0", "0")))


def test_exact_cayron_ui_has_two_routes_and_left_to_right_arrows():
    from twin_app.input_ui import render_correspondence_input
    import inspect
    source = inspect.getsource(render_correspondence_input)
    assert "Three paired lattice vectors" in source
    assert "Exact rational 3 × 3 matrix" in source
    assert "A → M" in source and "M → A" in source
    assert "calculate_from_paired_vectors" in source
    assert "calculate_action(entered_matrix, parent_vector)" in source
    assert "entered_matrix_from_canonical(canonical, direction=direction)" in source
    assert "arrow, equation = displayed_relation(direction)" in source
