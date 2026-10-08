from __future__ import annotations

import ast
from pathlib import Path


def _versioned_overlays(root: Path) -> list[tuple[int, Path]]:
    overlays: list[tuple[int, Path]] = []
    for path in (root / "app").glob("research_workspaces_v*.py"):
        suffix = path.stem.rsplit("_v", 1)[-1]
        if suffix.isdigit():
            overlays.append((int(suffix), path))
    return sorted(overlays)


def test_live_research_page_uses_latest_additive_presentation_head():
    root = Path(__file__).resolve().parents[2]
    page = (
        root / "app" / "pages" / "2_CT_Equivalence_Lab.py"
    ).read_text(encoding="utf-8")

    overlays = _versioned_overlays(root)
    assert overlays
    latest = overlays[-1][0]
    assert latest >= 12
    assert f"from app.research_workspaces_v{latest} import render_research_extension" in page

    # V10 is the established assessment base. Every later presentation overlay
    # must remain a solver-free additive layer delegating to its predecessor.
    forbidden_solver_calls = (
        "analyze_austenite_martensite(",
        "TheoryComparisonAdapter(",
        "twins_from_operator(",
        "CayronOrientationAdapter(",
        "single_variant_austenite_habit_solutions(",
        "ct_supercompatibility_residual(",
        "ips_shear_from_habit_plane(",
    )

    by_version = {version: path for version, path in overlays}
    for version in range(11, latest + 1):
        assert version in by_version, f"missing additive presentation layer V{version}"
        source = by_version[version].read_text(encoding="utf-8")
        ast.parse(source)
        for token in forbidden_solver_calls:
            assert token not in source
        assert f"research_workspaces_v{version - 1}" in source
        assert f"v{version - 1}.render_research_extension()" in source
