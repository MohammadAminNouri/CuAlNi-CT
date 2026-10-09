"""Safeguards for the consolidated, non-solver-changing research UI release."""
from __future__ import annotations

from pathlib import Path
import hashlib
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def test_manifest_tracks_every_v9_file_with_matching_hash():
    entries = (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8")
    for rel in (
        "twin_app/branch_presentation.py",
        "twin_app/input_explainer.py",
        "twin_app/structure_validation.py",
        "twin_app/input_ui.py",
        "twin_app/streamlit_app.py",
        "twin_app/tree_renderer.py",
        "tests/twin_app/test_v9_scientific_inputs_and_branches.py",
        "tests/twin_app/test_v9_property_based_science.py",
        "tests/twin_app/test_v9_release_guard.py",
        "tools/rebuild_twin_manifest.py",
        "pyproject.toml",
    ):
        path = ROOT / rel
        assert path.is_file(), rel
        assert hashlib.sha256(path.read_bytes()).hexdigest() + "  " + rel in entries, rel


def test_scientific_solvers_unchanged_and_benchmark_answers_not_loaded_at_runtime():
    for rel in (
        "twin_app/branch_presentation.py",
        "twin_app/input_explainer.py",
        "twin_app/structure_validation.py",
        "twin_app/tree_renderer.py",
    ):
        source = (ROOT / rel).read_text(encoding="utf-8")
        assert "data/benchmarks/" not in source
    renderer = (ROOT / "twin_app/tree_renderer.py").read_text(encoding="utf-8")
    assert "classification evidence" in renderer
    assert "Interface A" in renderer
    assert "not the same as Type I/Type II" in renderer


def test_environment_installs_validation_tools_without_changing_base_solver_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    assert any(x.startswith("spglib") for x in extras["app"])
    assert any(x.startswith("gemmi") for x in extras["app"])
    assert any(x.startswith("hypothesis") for x in extras["dev"])
    assert "spglib" not in " ".join(project["project"]["dependencies"])
    assert "hypothesis" not in " ".join(project["project"]["dependencies"])


def test_tracked_files_have_no_unresolved_git_merge_markers():
    """Prevent github.dev's previous silent conflict-marker commits.

    Search each manifest-tracked source and document as a *whole line*;
    ordinary comparison operators in Python do not count as conflicts.
    """
    tracked = (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    paths = ["MANIFEST.sha256"] + [line.partition("  ")[2] for line in tracked]
    markers = ("<<<<<<< ", "=======", ">>>>>>> ", "||||||| ")
    for name in paths:
        assert name and (ROOT / name).is_file(), name
        for number, line in enumerate((ROOT / name).read_text(encoding="utf-8").splitlines(), 1):
            assert not line.startswith(markers), f"Unresolved merge marker in {name}:{number}"
