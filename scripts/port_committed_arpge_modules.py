#!/usr/bin/env python3
"""Create syntax-only Python 3 copies of the original 2021 ARPGE suite modules.

Designed for Python 3.11/3.12's lib2to3. NOT a runtime or scientific migration:
GUI imports, Windows paths, old data formats, and numerics still require work.
Never import the original GUI scripts from the Streamlit server process.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "third_party" / "arpge_2021_original"
PORT = ROOT / "third_party" / "arpge_2021_py3_syntax"
EXPECTED = {
    "arpge/EBSD/ARPGE.py":
        "93892600f30f2ea2ce817018d5c97ac57a9e50fe9db26594a753e30505d3b077",
    "arpge/Crystallography/CC_Crystals.py":
        "7c8b28e1d49e2493012e646865c1e787fca0dff1d9b169c89f0b5f8036e0d829",
    "arpge/Crystallography/CC_GenOVa.py":
        "9f738fb3451eeb09aafec64753565d0be8731f31ef69784d4f4e77944792c139",
}


def main() -> None:
    try:
        from lib2to3.refactor import RefactoringTool, get_fixers_from_package
    except ImportError:
        raise SystemExit("Run syntax converter with Python 3.11 or 3.12, not 3.13+")
    transformer = RefactoringTool(get_fixers_from_package("lib2to3.fixes"))
    PORT.mkdir(parents=True, exist_ok=True)
    report = []
    for relative, expected_sha in EXPECTED.items():
        source = ORIGINAL / relative
        raw = source.read_bytes()
        actual_sha = sha256(raw).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"Original input altered or truncated: {relative}")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        try:
            converted = str(transformer.refactor_string(
                text if text.endswith("\n") else text + "\n", relative))
            compile(converted, str(source), "exec")
        except Exception as exc:
            report.append({
                "path": relative, "original_sha256": actual_sha,
                "syntax_status": "failed", "error": str(exc)
            })
            continue
        target = PORT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(converted, encoding="utf-8", newline="\n")
        report.append({
            "path": relative, "original_sha256": actual_sha,
            "syntax_status": "compiled",
            "python3_sha256": sha256(target.read_bytes()).hexdigest(),
        })
    (PORT / "PORT_STATUS.json").write_text(
        json.dumps({"stage": "syntax only, not runtime or scientific parity",
                    "modules": report}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if any(r["syntax_status"] == "failed" for r in report):
        sys.exit(2)


if __name__ == "__main__":
    main()
