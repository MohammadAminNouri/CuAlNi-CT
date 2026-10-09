from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as Record

from twin_app.family_tree_graph import plot_family_tree, tree_layout
from twin_app.point_group_explainer import POINT_GROUP_SHORT_NAMES, POINT_GROUP_SIGNATURES
from twin_app.ux_language import twin_name, representative_habit_solutions

ROOT = Path(__file__).resolve().parents[2]


def _report():
    def family(fid: str, start: int, count: int):
        pairs = tuple((i, i + 1) for i in range(start, start + count))
        return Record(
            family_id=fid,
            route="classical_exact",
            equivalent_pairs=pairs,
            pair_records=(),
            representative_pair=pairs[0],
            classical_systems=(),
            weak_candidates=(),
        )
    return Record(families=(family("F1", 1, 6), family("F2", 8, 7), family("F3", 16, 5)),
                  parent_phase_id="A", product_phase_id="M")


def test_initial_tree_contains_every_family_and_every_couple_without_crop():
    report = _report()
    layout = tree_layout(report)
    fig = plot_family_tree(report, selected_key=layout.couples[0].key)
    assert len(layout.families) == 3
    assert len(layout.couples) == 18
    assert len(fig.data[1].x) == 3
    assert len(fig.data[2].x) == 18
    assert set(fig.data[2].y) == {0.0}
    xmin, xmax = fig.layout.xaxis.range
    assert xmin < min(fig.data[2].x)
    assert xmax > max(fig.data[2].x)
    assert xmin < min(fig.data[1].x)
    assert xmax > max(fig.data[1].x)


def test_app_does_not_default_to_one_family():
    source = (ROOT / "twin_app" / "tree_renderer.py").read_text(encoding="utf-8")
    assert "All families remain visible" in source
    assert "focus_family=None" in source
    assert "twin_tree_family_view" not in source
    assert "st.plotly_chart(" in source
    assert "Find and select a couple" in source
    assert "The M numbers are calculated correspondence-variant IDs" in source


def test_point_group_meaning_immediately_below_selection_is_metric_validated():
    source = (ROOT / "twin_app" / "input_ui.py").read_text(encoding="utf-8")
    assert "selected_group_explanation = st.container()" in source
    assert "with selected_group_explanation:" in source
    assert "build_symmetry_inventory(point_group, lattice.metric())" in source
    assert "Verified contents:" in source
    assert "_coordinate_rule(op.matrix)" in source
    assert len(POINT_GROUP_SIGNATURES) == len(POINT_GROUP_SHORT_NAMES) == 32


def test_standard_notation_is_explained_and_unresolved_stays_unverified():
    src = (ROOT / "twin_app" / "tree_renderer.py").read_text(encoding="utf-8")
    for text in ("**K₁**", "**η₁**", "**s**", "fraction λ", "shape-strain", "unit normal",
                 "No exact austenite–martensite interface", "Full correspondence inventory"):
        assert text.lower() in src.lower()
    assert "Classification not confirmed" in src
    assert "Type I" not in twin_name("Exact rank-one relation — classification cross-lock unresolved")
    assert "not verified" in twin_name("Exact rank-one relation — classification cross-lock unresolved")


def test_plus_minus_and_complementary_habits_are_never_discarded():
    def habit(branch: int, frac: float):
        return Record(habit_branch=branch, other_variant_volume_fraction=frac)
    results = (habit(1, .29), habit(-1, .29), habit(1, .71), habit(-1, .71))
    shown, extra = representative_habit_solutions(results)
    assert {r.habit_branch for r in shown} == {1, -1}
    assert len(shown) + len(extra) == 4
    assert set(map(id, shown + extra)) == set(map(id, results))


def test_accessibility_has_single_decision_action_and_large_text_defaults():
    src = (ROOT / "twin_app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "Calculate twin families and habit planes" in src
    assert "st.tabs(" not in src
    assert "font-size: 1.06rem" in src
    assert "animation-duration: 0s" in src
    assert "Restore published NiTi crystal inputs" in src
    assert 'st.session_state["twin_family_report"] = report' in src
