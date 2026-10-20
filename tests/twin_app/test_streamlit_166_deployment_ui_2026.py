"""Guards for hosted Streamlit 1.66 and its accessibility contract."""
from __future__ import annotations
import ast
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / "twin_app" / "input_ui.py"
APP = ROOT / "twin_app" / "streamlit_app.py"

def test_locked_lattice_fields_do_not_set_widget_state_or_return_stale_values():
    source = UI.read_text(encoding="utf-8")
    code = ast.parse(source)
    fn = next(n for n in code.body if isinstance(n, ast.FunctionDef) and n.name == "_locked_number")
    sub = ast.get_source_segment(source, fn)
    assert sub is not None
    assert "st.session_state[key]" not in sub
    assert '"key": f"{key}_{float(value).hex()}"' in sub
    assert "st.number_input(**kwargs)" in sub
    assert "return float(value)" in sub

def test_app_uses_current_streamlit_width_convention():
    for path in (ROOT / "twin_app").rglob("*.py"):
        assert "use_container_width=" not in path.read_text(encoding="utf-8"), str(path)
    assert 'width="stretch"' in APP.read_text(encoding="utf-8")

def test_accessible_linear_sections_and_evidence_without_complete_inputs():
    source = APP.read_text(encoding="utf-8")
    ast.parse(source)
    assert "st.tabs(" not in source
    assert "Show optional material-specific references" in source
    assert "if show_material_references:" in source
    assert "Cu–Al–Ni independent evidence and source references (optional)" not in source
    assert "01  /  Crystal setup" in source
    assert "02  /  Calculated twinning modes" in source
    assert source.index("if show_material_references:") < source.index("01  /  Crystal setup")
    assert source.index("01  /  Crystal setup") < source.index("02  /  Calculated twinning modes")
    assert "build_twin_family_report(" in source
    assert "render_mode_report(catalogue, report)" in source
