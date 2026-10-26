"""Regression checks for the standalone theory comparison and CTF input placement."""
from pathlib import Path
from types import SimpleNamespace

from twin_app.theory_comparison_ui import matched_rows, overview_rows


ROOT = Path(__file__).resolve().parents[2]


def test_explanation_and_comparison_are_separate_from_physics():
    source = (ROOT / "twin_app" / "theory_comparison_ui.py").read_text(encoding="utf-8")
    assert "TheoryComparisonAdapter" in source
    assert "build_equivalence_report" in source
    assert "include_independent_mm_audit=False" in source
    assert "No missing theory prediction" in source or "No missing theory prediction was filled in" in source
    assert "Calculated prediction rows" in source


def test_matched_rows_keep_missing_observables_missing():
    fake = SimpleNamespace(
        matches=(SimpleNamespace(
            family=SimpleNamespace(value="martensite_martensite_twin"),
            left_theory=SimpleNamespace(value="cayron_ct"),
            right_theory=SimpleNamespace(value="ball_james"),
            left_branch="one", right_branch="two",
            residuals=SimpleNamespace(to_dict=lambda: {
                "habit_plane_angle_deg": None,
                "twin_plane_angle_deg": 0.25,
                "shear_relative": 0.001,
                "rank_one_tensor_relative": None,
                "or_disorientation_deg": None,
            }),
            completeness=SimpleNamespace(value="partial"),
            all_required_components_within_tolerance=None,
        ),)
    )
    row = matched_rows(fake)[0]
    assert row["Habit plane gap (deg)"] is None
    assert row["Twin plane gap (deg)"] == 0.25
    assert row["All required quantities within tolerance"] is None


def test_ctf_upload_lives_only_in_setup_before_results():
    source = (ROOT / "twin_app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert source.count("render_ctf_panel(") == 1
    assert source.index("render_ctf_panel(") < source.index(
        'st.subheader("02  /  Calculated twinning modes")'
    )
    assert source.count("render_theory_comparison(") == 1
    assert source.index("render_theory_comparison(") > source.index(
        "render_mode_report(catalogue, report)"
    )
