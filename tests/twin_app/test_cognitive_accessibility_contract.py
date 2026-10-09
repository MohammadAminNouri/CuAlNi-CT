from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "twin_app"


def test_all_couples_share_one_tree_and_one_results_pane():
    renderer = (APP / "tree_renderer.py").read_text(encoding="utf-8")
    graph = (APP / "family_tree_graph.py").read_text(encoding="utf-8")
    assert "st.plotly_chart(" in renderer
    assert "on_select=_select_from_chart" in renderer
    assert 'selection_mode="points"' in renderer
    assert "Twin couple (keyboard-friendly alternative" in renderer
    assert renderer.index("st.plotly_chart(") < renderer.index("_selected_details(lookup[")
    assert "for family in report.families:" in graph
    assert "for i, j in all_pairs:" in graph
    assert "y=[0.] * len(shown_couples)" in graph
    assert "st.columns(" not in renderer
    assert "st.table(" not in renderer


def test_book_notation_exists_without_wide_numeric_result_tables():
    renderer = (APP / "tree_renderer.py").read_text(encoding="utf-8")
    for needed in (
        "Twin plane K₁", "Shear direction η₁", "Twin shear s",
        "**a**", "**n̂**", "other_variant_volume_fraction",
        "shape_vector_parent_cartesian", "habit_normal_parent_cartesian",
        "Habit alternative", "Research details", "No exact austenite–martensite habit plane",
        "representative_habit_solutions",
    ):
        assert needed in renderer
    assert renderer.index("_book_twin_elements(twin)") < renderer.index("Research details")
    assert "if extra:" in renderer
    assert "complementary fractions" in renderer


def test_general_entry_is_one_workflow_without_material_mode_switch():
    source = (APP / "streamlit_app.py").read_text(encoding="utf-8")
    assert "Crystal twins & habit planes" in source
    assert "Calculate twins and habit planes" in source
    assert "Load example inputs" in source
    assert "NiTi book example · inputs only" not in source
    assert "Custom transformation" not in source
    assert "st.tabs(" not in source
    assert "animation-duration: 0s" in source
    assert "transition-duration: 0s" in source
    assert "previous calculation is hidden" in source
    assert 'st.session_state["twin_family_report"] = report' in source


def test_point_group_is_named_and_explained_before_matrix_inspector():
    source = (APP / "input_ui.py").read_text(encoding="utf-8")
    atlas = (APP / "point_group_explainer.py").read_text(encoding="utf-8")
    assert "simple_group_label" in source
    assert "POINT_GROUP_SHORT_NAMES" in atlas
    assert "Why does this point group matter?" in source
    assert "Explore one symmetry operation (optional)" in source
    assert "Show exact 3×3 transformation matrix" in source
    assert source.index("Why does this point group matter?") < source.index("Show exact 3×3")
    assert "Candidate is not confirmation" in source
    assert "[uvw] = direct crystal direction; (hkl) = reciprocal plane normal" in source


def test_colour_system_uses_only_three_meaningful_accents_with_text_labels():
    palette = (APP / "ux_language.py").read_text(encoding="utf-8")
    app = (APP / "streamlit_app.py").read_text(encoding="utf-8")
    assert "COLOR_SELECTED" in palette
    assert "COLOR_HABIT" in palette
    assert "COLOR_UNRESOLVED" in palette
    assert "Blue = selected twin" in app
    assert "Teal = exact habit found" in app
    assert "Amber = scientific verification pending" in app
    assert "Colours are always accompanied by written labels" in app


def test_unresolved_classification_is_never_silently_type_i_or_ii():
    from twin_app.ux_language import habit_status, twin_name

    assert "not verified" in twin_name("Exact rank-one relation — classification cross-lock unresolved")
    assert "Type I" in twin_name("Type I")
    assert "Type II" in twin_name("Type II")
    assert "Compound" in twin_name("Compound")
    assert "No exact habit" in habit_status(0)
    assert "Exact habit" in habit_status(1)
