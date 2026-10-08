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
    assert "Cayron" not in text
    assert "cayron" not in text.lower()


def test_streamlit_ui_is_linear_and_user_triggered():
    text = (PRODUCTION / "streamlit_app.py").read_text(encoding="utf-8")
    assert "Calculate twin family and habit planes" in text
    assert "st.tabs(" not in text
    assert "animation-duration: 0s" in text
    assert "transition-duration: 0s" in text
    assert "previous calculation is hidden" in text


def test_tree_renders_book_style_twin_and_habit_quantities_without_answers():
    text = (PRODUCTION / "tree_renderer.py").read_text(encoding="utf-8")
    for token in (
        "a · parent Cartesian", "n̂ · parent Cartesian", "K₁", "η₁",
        "other_variant_volume_fraction", "shape_vector_parent_cartesian",
        "habit_normal_parent_cartesian", "st.plotly_chart(",
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
