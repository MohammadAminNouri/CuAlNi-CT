from __future__ import annotations

import ast
from pathlib import Path


def test_v11_is_additive_and_solver_free():
    root = Path(__file__).resolve().parents[2]
    v11 = (root / "app" / "research_workspaces_v11.py").read_text(encoding="utf-8")
    geometry = (root / "app" / "cayron_geometry.py").read_text(encoding="utf-8")
    page = (root / "app" / "pages" / "2_CT_Equivalence_Lab.py").read_text(encoding="utf-8")

    ast.parse(v11)
    ast.parse(geometry)

    forbidden_solver_calls = (
        "analyze_austenite_martensite(",
        "TheoryComparisonAdapter(",
        "twins_from_operator(",
        "CayronOrientationAdapter(",
        "single_variant_austenite_habit_solutions(",
        "ct_supercompatibility_residual(",
        "ips_shear_from_habit_plane(",
    )
    for token in forbidden_solver_calls:
        assert token not in v11
        assert token not in geometry

    assert "v10.render_research_extension()" in v11
    assert "v10._render_step = _render_step_v11" in v11
    assert "research_workspaces_v11" in page
    assert "research_workspaces_v10" in page


def test_v11_guides_each_unresolved_cayron_question_to_real_app_sections():
    root = Path(__file__).resolve().parents[2]
    source = (root / "app" / "research_workspaces_v11.py").read_text(encoding="utf-8")

    for step_id in (
        '"state"',
        '"topology"',
        '"mm_twins"',
        '"am_exact"',
        '"nearest"',
        '"habit"',
        '"smc"',
        '"closing_gap"',
        '"supercompatibility"',
        '"overall"',
    ):
        assert step_id in source

    assert "Workbench → State definition" in source
    assert "Workbench → Compatibility" in source
    assert "Workbench → Twins & PTMC" in source
    assert "Workbench → Orientation & reconstruction" in source
    assert "Theory comparison → Compatibility map" in source
    assert "Theory comparison → EBSD ↔ theory" in source
    assert "CT closing-gap ORs" in source
    assert "CT supercompatibility" in source
    assert "Native theory inventory and provenance" in source


def test_cayron_geometry_uses_authoritative_outputs_and_keeps_diagnostics_nonexact():
    root = Path(__file__).resolve().parents[2]
    source = (root / "app" / "cayron_geometry.py").read_text(encoding="utf-8")

    assert 'metric.get("cmc_normalized")' in source
    assert 'ct.get("exact_compatible", False)' in source
    assert 'ct.get("exact_habit_planes_parent_covectors"' in source
    assert 'approximate.get("candidate_planes_parent_covectors"' in source
    assert "diagnostic projected plane" in source
    assert "NOT exact CT habit planes" in source
    assert "parent metric-orthonormal frame" in source
    assert '"ct_mm_twin"' in source
    assert '"ct_am_habit"' in source
    assert '"ct_supercompatibility"' in source
    assert "2(m_A^T n)d_A=a" in source


def test_plotly_is_an_app_dependency_not_a_core_science_dependency():
    root = Path(__file__).resolve().parents[2]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'app = ["streamlit>=1.40,<2", "plotly>=5.24,<7"]' in pyproject
