"""Accessible tree and coordinate-entry contracts without alloy-specific scientific answers."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[2]


def test_real_tree_is_connected_interactive_and_matches_original_goal():
    root = ROOT/'twin_app'
    renderer = (root/'mode_renderer.py').read_text()
    component = (root/'mode_tree_component.py').read_text()
    frontend = (root/'mode_tree_frontend/index.html').read_text()
    ast.parse(renderer)
    assert 'build_tree_model(view.visible_modes)' in renderer
    assert 'render_mode_tree(graph_model' in renderer
    assert 'mode-col' in frontend and 'type-fork' in frontend
    assert 'root-stem' in frontend and 'mode-col::before' in frontend
    assert 'streamlit:setComponentValue' in frontend
    assert 'aria-pressed' in frontend and 'prefers-reduced-motion' in frontend
    assert '_show_habits(entry' in renderer
    assert 'twin.habit_solutions' in renderer
    for value in ('0.2385', '0.2804', '0.3096', '0.1423'):
        assert value not in renderer + component + frontend


def test_only_verified_types_and_partial_coverage_remain_explicit():
    src = (ROOT/'twin_app'/'mode_renderer.py').read_text()
    widget = (ROOT/'twin_app'/'mode_tree_component.py').read_text()
    frontend = (ROOT/'twin_app'/'mode_tree_frontend/index.html').read_text()
    assert 'Incomplete symmetry-orbit classification' in src
    assert 'twin_type.stretch_couples' in src
    assert 'ct_tree_science_digest' in src
    assert 'twin_type.label not in _ALLOWED_TYPES' in widget
    assert 't.verifiedPairs' in frontend and 't.totalPairs' in frontend


def test_restorable_correspondence_preserves_canonical_solver_input():
    """Restore the baseline without confusing solver C with reverse UI probes."""
    src = (ROOT / "twin_app" / "input_ui.py").read_text(encoding="utf-8")
    ast.parse(src)
    start = src.index("def render_correspondence_input(")
    end = src.index("def _standard_basis(", start)
    fn = src[start:end]
    assert "def restore_starting_correspondence():" in fn
    assert "set_editor(method, direction, baseline)" in fn
    assert "Correspondence differs from the starting editable example" in fn
    assert "canonical_correspondence_rows(typed, direction=direction)" in fn
    assert "entered_matrix_from_canonical(canonical, direction=direction)" in fn
    assert "calculate_action(entered_matrix, parent_vector)" in fn
    assert "images_of_parent_basis(entered_matrix)" in fn
    assert "canonical_rows=canonical" in fn
    assert "value=False" in fn  # Optional 3D sketches remain opt-in.
    assert "C.inv()" in fn
    assert "st.stop()" in fn
    # The format and input-direction radio controls are deliberate UI features.
    assert "on_change=synchronize_editors" in fn
