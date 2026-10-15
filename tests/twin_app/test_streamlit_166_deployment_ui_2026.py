"""Contract checks for the deployed Streamlit 1.66/3.14 user-interface route."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / "twin_app" / "input_ui.py"
APP = ROOT / "twin_app" / "streamlit_app.py"


def test_locked_lattice_fields_do_not_set_widget_state_or_return_stale_values():
    source = UI.read_text(encoding="utf-8")
    code = ast.parse(source)
    function = next(n for n in code.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_locked_number")
    sub = ast.get_source_segment(source, function)
    assert sub is not None
    assert "st.session_state[key]" not in sub
    assert '"key": f"{key}_{float(value).hex()}"' in sub
    assert "st.number_input(**kwargs)" in sub
    assert "return float(value)" in sub


def test_app_uses_current_streamlit_width_convention():
    # Do not revive the obsolete width prop in any production Python module.
    for path in (ROOT / "twin_app").rglob("*.py"):
        assert "use_container_width=" not in path.read_text(encoding="utf-8"), str(path)
    source = APP.read_text(encoding="utf-8")
    assert 'width="stretch"' in source


def test_three_workspace_tabs_and_evidence_survive_incomplete_inputs():
    source = APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    tabs = [n for n in ast.walk(main)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == "tabs"]
    assert len(tabs) == 1
    assert all(name in source for name in
               ("01  ·  Crystal setup", "02  ·  Twin family & modes",
                "03  ·  Evidence & Cu–Al–Ni"))
    assert source.index("    with evidence_tab:") < source.index("    with setup_tab:")
    assert "build_twin_family_report(" in source
    assert "render_mode_report(catalogue, report)" in source
