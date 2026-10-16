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


def test_public_correspondence_input_has_only_explicit_parent_to_daughter_mapping():
    from twin_app.input_ui import render_correspondence_input
    import inspect
    source = inspect.getsource(render_correspondence_input)
    assert 'direction="A_TO_M"' in source
    assert 'st.latex(' in source
    assert 'C_c' in source
    assert 'p/q' in source
    assert 'st.radio(' not in source
