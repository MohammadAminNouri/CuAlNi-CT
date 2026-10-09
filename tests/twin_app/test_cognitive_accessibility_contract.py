from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "twin_app"


def test_tree_has_single_fixed_lower_panel_with_family_focus_and_keyboard_path():
    renderer = (APP / "tree_renderer.py").read_text(encoding="utf-8")
    graph = (APP / "family_tree_graph.py").read_text(encoding="utf-8")
    assert "st.plotly_chart(" in renderer
    assert "on_select=_sync_chart_selection" in renderer
    assert 'selection_mode="points"' in renderer
    assert "Family to display" in renderer
    assert "Twin couple (keyboard-accessible selection)" in renderer
    assert renderer.index("st.plotly_chart(") < renderer.index("_details(choices[")
    assert "for family in report.families:" in graph
    assert "for i, j in all_pairs:" in graph
    assert "y=[0.] * len(shown_couples)" in graph
    assert 'st.columns(' not in renderer
    assert 'st.dataframe(' not in renderer


def test_selected_couple_shows_exact_twin_and_habit_results_before_audit():
    renderer = (APP / "tree_renderer.py").read_text(encoding="utf-8")
    for token in (
        "K₁", "η₁", "a · parent Cartesian", "n̂ · parent Cartesian",
        "shape_vector_parent_cartesian", "habit_normal_parent_cartesian",
        "other_variant_volume_fraction", "No exact A/M habit solution",
    ):
        assert token in renderer
    assert "_physical_twin_table(twin)" in renderer
    assert "_habit_table(twin)" in renderer
    assert renderer.index("_physical_twin_table(twin)") < renderer.index('st.expander("More: crystal indices')
    assert "st.table(rows)" in renderer
    assert "if c.habit_solutions" in renderer


def test_every_point_group_has_its_own_explanation_and_metric_validated_inventory():
    source = (APP / "input_ui.py").read_text(encoding="utf-8")
    atlas = (APP / "point_group_explainer.py").read_text(encoding="utf-8")
    assert "Point-group contents: operations, axes and planes" in source
    assert "rotation axis [uvw]" in source
    assert "mirror-plane covector (hkl)" in source
    assert "These are not the same kind of object" in source
    assert "Inspect one exact symmetry operation" in source
    assert "explain_group" in source
    assert "POINT_GROUP_SIGNATURES" in atlas
    assert "does not independently prove a twin" in source


def test_main_page_is_predictable_and_visible_summary_is_not_hidden():
    source = (APP / "streamlit_app.py").read_text(encoding="utf-8")
    assert "animation-duration: 0s" in source
    assert "transition-duration: 0s" in source
    assert "st.tabs(" not in source
    assert "Calculate twin family and habit planes" in source
    assert "NiTi book example · inputs only" in source
    assert "Custom transformation" in source
    assert "the previous calculation is hidden" in source
    assert "Symmetry of the selected transformation" in source
    assert 'st.session_state["twin_family_report"] = report' in source
