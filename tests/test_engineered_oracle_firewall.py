from __future__ import annotations

from pathlib import Path
import ast


INDEPENDENT = (
    Path("validation/engineered/exact_oracle.py"),
    Path("validation/engineered/physical.py"),
)

FORBIDDEN_TEXT = (
    "cayron",
    "122399",
    "james_hane",
    "data/benchmarks",
    "validation/independent",
)


def test_oracle_modules_do_not_import_production_package():
    for path in INDEPENDENT:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("cualni_cryst") for alias in node.names), path
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("cualni_cryst"), path


def test_engineered_framework_contains_no_literature_answer_oracle():
    # Do not scan this firewall's own literal forbidden-token table.
    this_file = Path(__file__).resolve()
    paths = list(Path("validation/engineered").glob("*.py"))
    paths += [
        p for p in Path("tests").glob("test_engineered_*.py")
        if p.resolve() != this_file
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for token in FORBIDDEN_TEXT:
            assert token.lower() not in text, f"{path}: forbidden oracle token {token!r}"
