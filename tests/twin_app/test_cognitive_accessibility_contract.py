from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT / "twin_app"


def test_result_tree_is_semantically_linear_and_progressively_disclosed():
    source = (PRODUCTION / "tree_renderer.py").read_text(encoding="utf-8")
    assert "Root -> family -> variant pair -> twin branch -> A/M habit branch" in source
    assert 'st.markdown(f"### Branch ' in source
    assert 'st.markdown(f"#### Variant pair ' in source
    assert 'f"##### Twin branch ' in source
    assert "Representative pair calculation" in source
    assert "additional symmetry-equivalent pair calculation(s)" in source
    assert "st.columns(" not in source
    assert "st.dataframe(" not in source


def test_definitive_no_solution_is_not_hidden_as_low_priority_caption():
    source = (PRODUCTION / "tree_renderer.py").read_text(encoding="utf-8")
    assert 'st.info("Calculated result: no exact A/M habit-plane solution' in source
    assert "continuous exact compatible-fraction family exists" in source


def test_point_group_contents_are_available_without_nested_dataframe_navigation():
    source = (PRODUCTION / "input_ui.py").read_text(encoding="utf-8")
    assert "Show every exact operation in this point group" in source
    assert "st.dataframe(" not in source
    assert "rotation axis [uvw]" in source
    assert "mirror-plane covector (hkl)" in source


def test_main_page_is_user_triggered_predictable_and_has_explicit_reading_order():
    source = (PRODUCTION / "streamlit_app.py").read_text(encoding="utf-8")
    assert "Calculate twin family and habit planes" in source
    assert "Root → family → pair → twin branch → habit plane" in source
    assert "st.tabs(" not in source
    assert "animation-duration: 0s" in source
    assert "transition-duration: 0s" in source
    assert "previous calculation is hidden" in source


def test_habit_display_names_coordinate_roles_and_frame_crosschecks():
    renderer = (PRODUCTION / "tree_renderer.py").read_text(encoding="utf-8")
    models = (PRODUCTION / "scientific_models.py").read_text(encoding="utf-8")
    engine = (PRODUCTION / "scientific_engine.py").read_text(encoding="utf-8")
    for token in (
        "parent reciprocal/projective coefficients",
        "parent direct coefficients",
        "frame_plane_residual",
        "frame_shape_vector_residual",
    ):
        assert token in renderer or token in models
    assert "Habit-plane laminate fractions do not sum to one" in engine
    assert "Habit-plane parent reciprocal-coordinate conversion failed" in engine
