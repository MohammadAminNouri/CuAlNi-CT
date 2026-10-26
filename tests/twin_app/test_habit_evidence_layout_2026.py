"""Evidence-preserving habit-plane UI, with scientific scope kept visible."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_habit_summary_is_scoped_and_not_hidden():
    source = (ROOT / "twin_app" / "mode_renderer.py").read_text(encoding="utf-8")
    start = source.index("def _show_habits(")
    end = source.index("def _show_audit(", start)
    summary = source[start:end]
    assert "No exact habit-plane solution found for this branch" in summary
    assert "M/M twin interface calculated above remains" in summary
    assert "does not exclude the calculated M/M twin" in summary
    assert "Equivalent sources and other modes may give different results" in summary
    assert "Thesis-ready statement" in summary
    assert "continuum of compatible laminate fractions" in summary
    assert "habit_normal_parent_cartesian" in summary
    assert "rank_one_residual" in summary


def test_three_clear_verification_tabs_keep_all_detailed_evidence():
    source = (ROOT / "twin_app" / "mode_renderer.py").read_text(encoding="utf-8")
    assert '"Numerical checks", "Variant pairs and sources", "Unresolved records"' in source
    assert "render_p1_source_evidence(entry, report, inline=True)" in source
    assert "_show_audit(view, catalogue, report, inline=True)" in source
    for value in (
        "independent_outer_product_residual", "independent_shear_relative_residual",
        "classical_system_ids", "twin.rank_one_residual", "twin.rotation_residual",
        "source.construction.habit_solutions",
    ):
        assert value in source
    p1 = (ROOT / "twin_app" / "p1_ui.py").read_text(encoding="utf-8")
    for value in (
        "branch_evidence_status", "habit_fractions", "habit_normals_parent_cartesian",
        "Export selected source numerical evidence (JSON)",
        "report_orbits(report,parent.lattice.metric(),parent.symmetry_matrices())",
    ):
        assert value in p1


def test_research_tools_preserve_every_export_field_and_action():
    main = (ROOT / "twin_app" / "streamlit_app.py").read_text(encoding="utf-8")
    helper = (ROOT / "twin_app" / "research_tools_ui.py").read_text(encoding="utf-8")
    assert "render_research_tools(catalogue, report, payload, signature)" in main
    assert "render_p1_symmetry_orbit_panel(report, payload, signature, inline=True)" in helper
    for value in (
        "make_export", "physical_mode_catalogue", "incompatible_pair_orbits",
        "symmetry_equivalent_stretch_pairs", "source_records",
        "Download complete analysis as JSON", "tf_export_v8",
    ):
        assert value in helper
    assert "st.tabs(" not in main  # Preserve initial keyboard-linear navigation.

def test_calculated_habit_panel_renders_updated_section():
    """Actually calculate before checking the habit-panel UI, not just startup."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(ROOT / "twin_app" / "streamlit_app.py"), default_timeout=170
    )
    app.run()
    assert not app.exception, [str(x.message) for x in app.exception]

    calculate = next(
        item for item in app.button
        if item.label == "Calculate physical twinning modes and habit planes"
    )
    calculate.click()
    app.run()
    assert not app.exception, [str(x.message) for x in app.exception]

    assert any(
        "Austenite–martensite habit-plane compatibility" in getattr(x, "value", "")
        for x in app.markdown
    )
    labels = [x.label for x in app.get("expander")]
    assert "Verify this result and inspect its scientific evidence" in labels
    assert "Research tools and complete scientific export" in labels
    assert "Scientific validity · numerical checks, twins, habits and evidence" not in labels
    assert "P1 · macroscopic symmetry orbits and solution-count interpretation" not in labels
