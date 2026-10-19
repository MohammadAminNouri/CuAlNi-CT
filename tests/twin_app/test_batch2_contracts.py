from __future__ import annotations

from pathlib import Path

import pytest

from twin_app.input_logic import canonical_correspondence_rows


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT / "twin_app"


def test_reverse_correspondence_is_exact_not_floating():
    reverse = (("0", "1", "-1"), ("0", "1", "1"), ("1", "0", "0"))
    assert canonical_correspondence_rows(reverse, direction="M_TO_A") == (
        ("0", "0", "1"),
        ("1/2", "1/2", "0"),
        ("-1/2", "1/2", "0"),
    )


def test_new_app_contains_no_material_answer_lookup_and_no_named_theorist():
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in PRODUCTION.rglob("*.py")
    )
    # Published output numbers must remain in test data only.
    for forbidden in ("0.2385", "0.2804", "0.3096", "0.1423", "0.29202", "0.27102"):
        assert forbidden not in text
    # Scientific attribution in exported method metadata is welcome; hard-coded
    # published answer values or benchmark lookups in production are not.
    assert 'json.loads((ROOT / "data" / "benchmarks"' not in text


def test_streamlit_ui_is_linear_and_user_triggered():
    text = (PRODUCTION / "streamlit_app.py").read_text(encoding="utf-8")
    # Require a real user-triggered calculation, independent of button wording.
    import ast
    module = ast.parse(text)
    triggers = [
        node for node in ast.walk(module)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Attribute)
        and isinstance(node.test.func.value, ast.Name)
        and node.test.func.value.id == "st"
        and node.test.func.attr == "button"
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "build_twin_family_report"
            for statement in node.body
            for child in ast.walk(statement)
        )
    ]
    assert len(triggers) == 1, "Expected one manual calculation button"
    label = triggers[0].test.args[0]
    assert isinstance(label, ast.Constant)
    assert isinstance(label.value, str)
    assert "calculat" in label.value.lower()
    assert "st.tabs(" not in text
    assert "animation-duration: 0s" in text
    assert "transition-duration: 0s" in text
    assert "previous calculation is hidden" in text


def test_tree_renders_book_style_twin_and_habit_quantities_without_answers():
    text = (PRODUCTION / "tree_renderer.py").read_text(encoding="utf-8")
    for token in (
        "a = {_vec(a)}", "K₁", "η₁",
        "other_variant_volume_fraction", "shape_vector_parent_cartesian",
        "habit_normal_parent_cartesian", "render_navigator",
    ):
        assert token in text


def test_engine_keeps_correspondence_and_stretch_variants_separate():
    models = (PRODUCTION / "scientific_models.py").read_text(encoding="utf-8")
    engine = (PRODUCTION / "scientific_engine.py").read_text(encoding="utf-8")
    assert "CorrespondenceVariantRecord" in models
    assert "stretch_variant_index" in models
    assert "metric collapse" in engine
    assert "correspondence_to_stretch" in engine
    assert "C_i_exact = sp.simplify(base_correspondence.C_M_from_A * representative.inv())" in engine
