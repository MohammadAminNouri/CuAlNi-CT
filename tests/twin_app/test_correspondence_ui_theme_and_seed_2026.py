"""Protect accessible UI labels, theme semantics, and integer paired-vector seeds."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import sympy as sp

from twin_app.cayron_correspondence import (
    calculate_from_paired_vectors, parse_triplet, seed_paired_vectors,
)
from twin_app.input_logic import canonical_correspondence_rows, exact_matrix

ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / "twin_app" / "input_ui.py"
APP = ROOT / "twin_app" / "streamlit_app.py"


@pytest.mark.parametrize("C", [
    (("0","0","1"),("1/2","1/2","0"),("-1/2","1/2","0")),
    (("1/3","-2/5","0"),("2","0","1/7"),("0","1","1")),
    (("0","1","1"),("1","0","0"),("0","1","-1")),
    (("1","0","0"),("0","1","0"),("0","0","1")),
])
@pytest.mark.parametrize("direction", ["A_TO_M", "M_TO_A"])
def test_generated_seed_pairs_are_integral_exact_and_bidirectionally_consistent(C, direction):
    """The UI should never seed fractional *lattice translations* by default."""
    matrix = exact_matrix(C)
    expected = matrix if direction == "A_TO_M" else matrix.inv()
    src, dst = seed_paired_vectors(C, direction=direction)
    P = sp.Matrix.hstack(*(sp.Matrix(parse_triplet(x)) for x in src))
    Q = sp.Matrix.hstack(*(sp.Matrix(parse_triplet(x)) for x in dst))
    assert P.det() != 0 and Q.det() != 0
    assert all(x.is_Integer for x in P)
    assert all(x.is_Integer for x in Q)
    assert Q * P.inv() == expected
    recovered = calculate_from_paired_vectors(
        [parse_triplet(x) for x in src],
        [parse_triplet(x) for x in dst], direction=direction,
    )
    assert exact_matrix(recovered.canonical_M_from_A) == matrix
    assert exact_matrix(canonical_correspondence_rows(
        recovered.source_to_destination, direction=direction,
    )) == matrix


def _luminance(hex_color: str) -> float:
    linear = []
    for index in (1, 3, 5):
        value = int(hex_color[index:index + 2], 16) / 255.0
        linear.append(value / 12.92 if value <= 0.04045
                      else ((value + 0.055) / 1.055) ** 2.4)
    return sum(a * b for a, b in zip((0.2126, 0.7152, 0.0722), linear))


def _contrast(first: str, second: str) -> float:
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("foreground,background", [
    ("#18313f", "#ffffff"), ("#344d5c", "#ffffff"),
    ("#f2f5f8", "#101922"), ("#c2cdd6", "#101922"),
    ("#ffffff", "#050505"),
])
def test_theme_palette_text_exceeds_wcag_aa_contrast(foreground, background):
    assert _contrast(foreground, background) >= 4.5


def test_theme_styles_actual_streamlit_widgets_not_only_outer_background():
    source = APP.read_text(encoding="utf-8")
    ast.parse(source)
    for required in (
        '--tf-fg:', '--tf-muted:', '--tf-field:', '--tf-border:',
        '[data-testid="stSidebar"]',
        '[data-testid="stCaptionContainer"]',
        '[data-baseweb="input"]', '[data-baseweb="select"]',
        '[role="listbox"]', 'input:focus-visible',
        'heading="Parent crystal (initial phase)"',
        'heading="Product crystal (transformed phase)"',
    ):
        assert required in source, required


def test_correspondence_controls_explain_crystal_role_and_do_not_name_people():
    text = UI.read_text(encoding="utf-8")
    code = ast.parse(text)
    fn = next(node for node in code.body
              if isinstance(node, ast.FunctionDef)
              and node.name == "render_correspondence_input")
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr not in {"subheader", "caption", "expander",
                                      "radio", "markdown", "help", "write"}:
                continue
            for arg in node.args:
                words = " ".join(
                    child.value for child in ast.walk(arg)
                    if isinstance(child, ast.Constant)
                    and isinstance(child.value, str)
                )
                assert "Cayron" not in words
    for label in (
        'st.subheader("Crystallographic correspondence")',
        '"Parent crystal → Product crystal"',
        '"Product crystal → Parent crystal"',
        '"Pair 1", "Pair 2", "Pair 3"',
        "Check vector mapping and technical details",
        "Selected direction:",
        "D * C != sp.eye(3)",
        r"C_{\mathrm{parent}\to\mathrm{product}}",
        '"A → M": "Parent crystal → Product crystal"',
        '"M → A": "Product crystal → Parent crystal"',
        "entered_matrix_from_canonical(canonical, direction=direction)",
        "calculate_action(entered_matrix, parent_vector)",
        "plane_image(entered_matrix",
    ):
        assert label in text, label


def test_single_primary_correspondence_matrix_and_collapsed_advanced_checks():
    """Paired input displays exactly one C; the matrix editor itself is C."""
    import inspect
    from twin_app.input_ui import render_correspondence_input
    ui = inspect.getsource(render_correspondence_input)
    visible, technical = ui.split(
        'with st.expander("Check vector mapping and technical details", expanded=False):',
        1,
    )
    assert 'if method == methods[0]:\n        st.latex(label + "=" + sp.latex(D))' in visible
    assert 'The editable 3 × 3 matrix above is the active correspondence.' in visible
    assert 'C_{\\mathrm{entered}}' not in visible
    assert 'st.latex(sp.latex(D) +' not in ui
    assert 'Show source basis mappings and optional drawings' in technical
    assert 'Show solver matrix, inverse and Miller-plane mapping' in technical
    assert 'if direction == "M_TO_A":' in technical
    assert 'calculate_action(entered_matrix, parent_vector)' in technical
    assert 'plane_image(entered_matrix, ("1", "0", "0"))' in technical
    assert 'D * C != sp.eye(3)' in ui
    assert 'C * D != sp.eye(3)' in ui


def test_consolidating_display_never_changes_forward_or_reverse_math():
    from twin_app.correspondence_explorer import calculate_action
    from twin_app.cayron_correspondence import (
        entered_matrix_from_canonical, plane_image, seed_paired_vectors,
        calculate_from_paired_vectors, parse_triplet,
    )
    canonical = (("1", "1/3", "0"), ("0", "1", "1/5"), ("0", "0", "1"))
    C = exact_matrix(canonical)
    for direction, expected in (("A_TO_M", C), ("M_TO_A", C.inv())):
        pair_a, pair_b = seed_paired_vectors(canonical, direction=direction)
        recovered = calculate_from_paired_vectors(
            [parse_triplet(v) for v in pair_a],
            [parse_triplet(v) for v in pair_b],
            direction=direction,
        )
        assert exact_matrix(recovered.canonical_M_from_A) == C
        rows = entered_matrix_from_canonical(canonical, direction=direction)
        actual = exact_matrix(rows)
        assert actual == expected
        u = sp.Matrix([3, 5, 7])
        assert sp.Matrix(calculate_action(rows, ("3", "5", "7")).daughter) == actual * u
        transformed_plane = plane_image(rows, ("1", "2", "3"))
        assert sp.Matrix([sp.Rational(x) for x in transformed_plane]) == actual.inv().T * sp.Matrix([1, 2, 3])
