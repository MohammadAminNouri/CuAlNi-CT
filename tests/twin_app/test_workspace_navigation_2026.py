"""Guard the scientist-facing workspace navigation without importing Streamlit."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "twin_app" / "streamlit_app.py"


def _main():
    source = APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    return source, main


def test_three_real_workspaces_exist_and_run_inside_main():
    source, main = _main()
    tabs = [
        node for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "tabs"
    ]
    assert len(tabs) == 1
    assert "01  ·  Crystal setup" in source
    assert "02  ·  Twin family & modes" in source
    assert "03  ·  Evidence & Cu–Al–Ni" in source
    assert "with setup_tab:" in source
    assert "with results_tab:" in source
    assert "with evidence_tab:" in source


def test_data_flow_and_integrity_not_replaced_by_decorative_ui():
    source, main = _main()
    assert "build_twin_family_report(" in source
    assert "validate_report_integrity(report)" in source
    assert "build_mode_catalogue(report, payload)" in source
    assert 'st.session_state.get("twin_family_result_signature") == signature' in source
    assert "render_mode_report(catalogue, report)" in source
    assert "render_cualni_reference_guide()" in source
    assert "summary_a.metric" in source
    assert "summary_b.metric" in source
    assert "summary_c.metric" in source
    assert "Numerical compatibility is not independent experimental validation" in source
    assert 'elif source == "NiTi input example":' in source
    assert "with st.expander(\"Crystal parameters and correspondence\"" in source


def test_evidence_is_not_inserted_as_solver_inputs():
    source, main = _main()
    evidence_tab = next(
        node for node in ast.walk(main)
        if isinstance(node, ast.With)
        and any(isinstance(item.context_expr, ast.Name)
                and item.context_expr.id == "evidence_tab"
                for item in node.items)
    )
    names = {n.id for n in ast.walk(evidence_tab) if isinstance(n, ast.Name)}
    assert "payload" not in names
    assert "build_twin_family_report" not in names
    assert "render_cualni_reference_guide" in names
