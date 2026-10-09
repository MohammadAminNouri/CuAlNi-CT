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
    assert "Find and select a couple (keyboard or screen-reader accessible)" in renderer
    assert renderer.index("st.plotly_chart(") < renderer.index("_selected_details(lookup[")
    assert "for family in report.families:" in graph
    assert "for i, j in all_pairs:" in graph
    assert "y=[0.] * len(shown_couples)" in graph
    assert "st.columns(" not in renderer
    assert "st.table(" not in renderer


def test_book_notation_exists_without_wide_numeric_result_tables():
    renderer = (APP / "tree_renderer.py").read_text(encoding="utf-8")
    for needed in (
        "Twin interface plane:", "Direction of shear:", "Amount of shear:",
        "**a · shear vector:**", "**n̂ · unit normal", "other_variant_volume_fraction",
        "shape_vector_parent_cartesian", "habit_normal_parent_cartesian",
        "Habit alternative", "Research checks", "No exact austenite–martensite interface",
        "representative_habit_solutions",
    ):
        assert needed in renderer
    assert renderer.index("#### Martensite–martensite twin") < renderer.index("Research checks")
    assert "if extra:" in renderer
    assert "Complementary variant fractions" in renderer


def test_general_entry_is_one_workflow_without_material_mode_switch():
    source = (APP / "streamlit_app.py").read_text(encoding="utf-8")
    assert "Martensitic Crystallography" in source
    assert "Calculate twin families and habit planes" in source
    assert "Restore published NiTi crystal inputs" in source
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
    assert "Verified contents:" in source
    assert "Inspect one symmetry operation (optional)" in source
    assert "Show exact coordinate matrix and numerical check" in source
    assert source.index("Verified contents:") < source.index("Show exact coordinate matrix")
    assert "To predict actual twin families" in source
    assert "[uvw] is a direct-space rotation axis; (hkl) is a reciprocal plane covector" in source


def test_colour_system_uses_only_three_meaningful_accents_with_text_labels():
    palette = (APP / "ux_language.py").read_text(encoding="utf-8")
    app = (APP / "streamlit_app.py").read_text(encoding="utf-8")
    assert "COLOR_SELECTED" in palette
    assert "COLOR_HABIT" in palette
    assert "COLOR_UNRESOLVED" in palette
    renderer = (APP / "tree_renderer.py").read_text(encoding="utf-8")
    assert "Blue circle = selected couple" in renderer
    assert "Teal diamond = a calculated exact habit plane" in renderer
    assert "Classification not confirmed" in renderer
    assert "Colour is never the only indicator" in renderer


def test_unresolved_classification_is_never_silently_type_i_or_ii():
    from twin_app.ux_language import habit_status, twin_name

    assert "not verified" in twin_name("Exact rank-one relation — classification cross-lock unresolved")
    assert "Type I" in twin_name("Type I")
    assert "Type II" in twin_name("Type II")
    assert "Compound" in twin_name("Compound")
    assert "No exact habit" in habit_status(0)
    assert "Exact habit" in habit_status(1)
