from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "twin_app"


def test_one_persistent_results_pane_and_not_a_plotly_selection_chart():
    renderer = (APP / "tree_renderer.py").read_text(encoding="utf-8")
    navigator = (APP / "navigator_frontend" / "index.html").read_text(encoding="utf-8")
    assert "render_navigator" in renderer
    assert renderer.index("render_navigator(model") < renderer.index("_selected_details(lookup[")
    assert "st.plotly_chart(" not in renderer
    assert "st.table(" not in renderer and "st.dataframe(" not in renderer
    assert "streamlit:setComponentValue" in navigator
    assert "aria-pressed" in navigator
    assert "aria-label" in navigator
    assert "aria-live" in navigator
    assert "focus-visible" in navigator
    assert "Show all couples" in navigator
    assert "Show family overview" in navigator


def test_results_are_bhattacharya_compatible_without_hide_and_seek():
    renderer = (APP / "tree_renderer.py").read_text(encoding="utf-8")
    for expected in (
        "K₁", "η₁", "Twin shear", "Second-variant fraction", "Shape-strain vector",
        "Habit-plane normal", "other_variant_volume_fraction", "shape_vector_parent_cartesian",
        "habit_normal_parent_cartesian", "representative_habit_solutions",
        "No exact A/M habit plane for this twin branch", "Twin type not verified",
        "Scientific verification and full coordinates",
    ):
        assert expected in renderer
    assert renderer.index("Martensite–martensite twinning elements") < renderer.index("Scientific verification and full coordinates")


def test_ui_uses_one_workflow_and_independent_appearance_choices():
    app = (APP / "streamlit_app.py").read_text(encoding="utf-8")
    assert "Calculate twin families and habit planes" in app
    assert "tf_appearance_v8" in app and "tf_font_scale_v8" in app
    assert "tf_density_v8" in app
    assert "animation-duration: 0s" in app
    assert "transition-duration: 0s" in app
    assert "st.tabs(" not in app
    assert 'st.session_state["twin_family_report"] = report' in app
    assert "make_export" in app


def test_point_groups_explain_exact_operators_not_just_names():
    source = (APP / "input_ui.py").read_text(encoding="utf-8")
    atlas = (APP / "point_group_explainer.py").read_text(encoding="utf-8")
    assert "POINT_GROUP_SHORT_NAMES" in atlas
    assert "Verified contents:" in source
    assert "Inspect one symmetry operation (optional)" in source
    assert "_coordinate_rule(op.matrix)" in source
    assert "make_operation_scene" in source
    assert "Show exact coordinate matrix and numerical check" in source
    assert "build_symmetry_inventory(point_group, lattice.metric())" in source


def test_three_accents_have_visible_textual_labels():
    graph = (APP / "navigator_frontend" / "index.html").read_text(encoding="utf-8")
    for item in (
        "Blue outline: selected", "Teal line: exact habit", "Amber line: twin type unresolved",
        "Grey: no exact habit",
    ):
        assert item in graph
    assert "prefers-reduced-motion" in graph


def test_unresolved_is_never_auto_converted_to_type_i():
    from twin_app.ux_language import habit_status, twin_name
    assert "not verified" in twin_name("Exact rank-one relation — classification cross-lock unresolved")
    assert "Type I" not in twin_name("Exact rank-one relation — classification cross-lock unresolved")
    assert "Type I" in twin_name("Type I")
    assert "No exact habit" in habit_status(0)
