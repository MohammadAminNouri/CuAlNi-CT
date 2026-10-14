"""Physics-safe tree and exact-coordinate interaction regression checks."""
from __future__ import annotations

from pathlib import Path
import ast
import sympy as sp
import pytest

from twin_app.correspondence_explorer import calculate_action, images_of_parent_basis, vector_text
from twin_app.mode_view_model import alphabetic_index

ROOT = Path(__file__).resolve().parents[2]


def test_parent_basis_images_are_exact_and_matrix_columns():
    ni_ti = (("0", "0", "1"), ("1/2", "1/2", "0"), ("-1/2", "1/2", "0"))
    images = images_of_parent_basis(ni_ti)
    assert [e.daughter for e in images] == [
        (sp.Integer(0), sp.Rational(1, 2), sp.Rational(-1, 2)),
        (sp.Integer(0), sp.Rational(1, 2), sp.Rational(1, 2)),
        (sp.Integer(1), sp.Integer(0), sp.Integer(0)),
    ]
    assert images[0].determinant != 0


def test_arbitrary_rational_probe_is_exact_and_invertible():
    C = (("0", "0", "1"), ("1/2", "1/2", "0"), ("-1/2", "1/2", "0"))
    output = calculate_action(C, ("3/2", "-1/2", "1/3"))
    assert output.parent == (sp.Rational(3, 2), sp.Rational(-1, 2), sp.Rational(1, 3))
    assert output.daughter == (sp.Rational(1, 3), sp.Rational(1, 2), sp.Rational(-1))
    assert output.matrix.inv() * sp.Matrix(output.daughter) == sp.Matrix(output.parent)
    assert vector_text(output.daughter) == "[1/3, 1/2, -1]"


def test_any_nonsingular_correspondence_not_just_niti():
    arbitrary = (("1", "1/3", "0"), ("0", "2", "0"), ("0", "-1/7", "3"))
    r = calculate_action(arbitrary, ("2", "3", "7/2"))
    assert r.daughter == (sp.Integer(3), sp.Integer(6), sp.Rational(141, 14))


def test_invalid_inputs_fail_instead_of_fabricated_output():
    with pytest.raises(ValueError):
        calculate_action((("1", "0", "0"), ("0", "1", "0"), ("0", "0", "0")), (1, 0, 0))
    with pytest.raises(ValueError, match="3 entries"):
        calculate_action((("1", "0", "0"), ("0", "1", "0"), ("0", "0", "1")), (1, 0))


def test_mode_letters_scale_beyond_four_or_26():
    assert alphabetic_index(0) == "A"
    assert alphabetic_index(3) == "D"
    assert alphabetic_index(4) == "E"
    assert alphabetic_index(25) == "Z"
    assert alphabetic_index(26) == "AA"
    assert alphabetic_index(53) == "BB"


def test_interactive_mode_tree_renders_solver_modes_and_verified_types():
    root = ROOT / "twin_app"
    renderer = (root / "mode_renderer.py").read_text(encoding="utf-8")
    component = (root / "mode_tree_component.py").read_text(encoding="utf-8")
    frontend = (root / "mode_tree_frontend" / "index.html").read_text(encoding="utf-8")
    ast.parse(renderer)
    assert "build_mode_view(catalogue)" in renderer
    assert "build_tree_model(view.visible_modes)" in renderer
    assert "render_mode_tree(graph_model" in renderer
    assert "validate_selection(value, model)" in component
    assert "mode-col" in frontend and "type-fork" in frontend
    assert "streamlit:setComponentValue" in frontend
    assert "_show_audit(view, catalogue, report)" in renderer
    assert "K₁" in renderer and "η₁" in renderer
    for forbidden in ("0.2385", "0.2804", "0.3096", "0.1423"):
        assert forbidden not in frontend + component + renderer


def test_coordinate_input_has_interactive_probe_without_reversing_mapping():
    ui = (ROOT / "twin_app" / "input_ui.py").read_text(encoding="utf-8")
    src = ui[ui.index("def render_correspondence_input("):ui.index("\ndef _standard_basis(")]
    assert "calculate_action(canonical, parent_vector)" in src
    assert "images_of_parent_basis(canonical)" in src
    assert "st.plotly_chart(" in src
    assert 'direction="A_TO_M"' in src
    assert "separate coordinate axes" in src
    assert "C_c" in src
    assert "C.inv()" in src
    assert "st.radio(" not in src
