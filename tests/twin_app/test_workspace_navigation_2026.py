"""Accessibility and scientific contract for the single-path Twin App workspace."""
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

def test_single_linear_workflow_and_optional_references():
    source, main = _main()
    assert "st.tabs(" not in source
    assert "01  /  Crystal setup" in source
    assert "02  /  Calculated twin-family genealogy" in source
    assert "Cu–Al–Ni independent evidence and source references (optional)" in source
    assert source.index("01  /  Crystal setup") < source.index("02  /  Calculated twin-family genealogy")
    assert "Twin Family Explorer" in source
    assert "Restore published NiTi crystal inputs" in source
    assert "st.button(\"Calculate physical twinning modes and habit planes\"" in source
    assert "render_mode_report(catalogue, report)" in source

def test_scientific_data_flow_and_result_freshness():
    source, _ = _main()
    assert "build_twin_family_report(" in source
    assert "validate_report_integrity(report)" in source
    assert "build_mode_catalogue(report, payload)" in source
    assert 'st.session_state.get("twin_family_result_signature") == signature' in source
    assert "summary_a.metric" in source and "summary_b.metric" in source and "summary_c.metric" in source
    assert "Numerical compatibility is not independent experimental validation" in source
    assert "with st.expander(\"Crystal parameters and correspondence\"" in source

def test_evidence_available_before_input_validation_without_solver_contamination():
    source, main = _main()
    evidences = [node for node in ast.walk(main)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "render_cualni_reference_guide"]
    assert len(evidences) == 1
    assert source.index("render_cualni_reference_guide()") < source.index("01  /  Crystal setup")
    assert source.index("render_cualni_reference_guide()") < source.index("build_twin_family_report(")
    # The guide itself manages its own expanders; never nest another one around it.
    assert 'with st.expander("Cu–Al–Ni experimental references (optional)"' not in source
