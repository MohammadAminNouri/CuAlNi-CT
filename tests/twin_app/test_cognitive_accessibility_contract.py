from __future__ import annotations

from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
APP=ROOT/'twin_app'


def test_real_rooted_tree_and_single_stable_lower_results_panel():
    source=(APP/'tree_renderer.py').read_text(encoding='utf-8')
    graph=(APP/'family_tree_graph.py').read_text(encoding='utf-8')
    assert 'st.plotly_chart(' in source
    assert 'on_select=_sync_chart_selection' in source
    assert 'selection_mode="points"' in source
    assert 'Twin couple (keyboard-accessible selection)' in source
    assert source.index('st.plotly_chart(') < source.index('_details(choices[')
    assert 'for family in report.families:' in graph
    assert 'for i, j in pairs:' in graph
    assert 'y=[0.0] * len(layout.couples)' in graph
    assert 'st.columns(' not in source
    assert 'st.dataframe(' not in source


def test_book_twin_and_habit_are_immediately_visible_in_one_panel():
    source=(APP/'tree_renderer.py').read_text(encoding='utf-8')
    for token in ('K₁', 'η₁', 'a · parent Cartesian', 'n̂ · parent Cartesian',
                  'shape_vector_parent_cartesian','habit_normal_parent_cartesian',
                  'other_variant_volume_fraction','No exact A/M habit solution'):
        assert token in source
    assert '_physical_twin_table(pair.constructions)' in source
    assert '_habit_table(pair.constructions)' in source
    assert 'st.table(rows)' in source
    assert 'st.expander("More: crystal indices' in source


def test_point_group_description_is_explicit_but_technical_list_optional():
    source=(APP/'input_ui.py').read_text(encoding='utf-8')
    assert 'Point-group contents: operations, axes and planes' in source
    assert 'rotation axis [uvw]' in source
    assert 'mirror-plane covector (hkl)' in source
    assert 'These are not the same kind of object' in source
    assert 'st.table(rows)' in source
    assert 'Inspect one exact symmetry operation' in source


def test_interactions_minimize_overload_without_autonomous_recalculation():
    source=(APP/'streamlit_app.py').read_text(encoding='utf-8')
    assert 'animation-duration: 0s' in source
    assert 'transition-duration: 0s' in source
    assert 'st.tabs(' not in source
    assert 'Calculate twin family and habit planes' in source
    assert 'NiTi book example · inputs only' in source
    assert 'Custom transformation' in source
    assert 'the previous calculation is hidden' in source
    assert 'st.session_state["twin_family_report"] = report' in source
