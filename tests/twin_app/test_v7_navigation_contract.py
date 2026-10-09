from __future__ import annotations

from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def test_no_habit_message_is_scoped_to_branch_and_global_habits_are_discoverable():
    source=(ROOT/'twin_app'/'tree_renderer.py').read_text(encoding='utf-8')
    assert 'No exact A/M habit plane for this twin branch' in source
    assert 'Show a couple with an exact habit plane' in source
    assert 'first_interpretable_couple(model)' in source
    assert 'build_navigator_model(report)' in source


def test_recalculation_clears_stale_couple_selection():
    source=(ROOT/'twin_app'/'streamlit_app.py').read_text(encoding='utf-8')
    assert 'st.session_state.pop("tf_selected_couple_v6", None)' in source
    assert 'st.session_state.pop("twin_selected_couple", None)' in source


def test_backend_certifies_raw_twin_and_habit_equations_before_output():
    source=(ROOT/'twin_app'/'scientific_engine.py').read_text(encoding='utf-8')
    assert 'certify_ptmc_twinning(ptmc)' in source
    assert source.index('certify_ptmc_twinning(ptmc)') < source.index('    variant_records, correspondence_to_stretch')
