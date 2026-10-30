#!/usr/bin/env python3
"""Preserve and optionally convert every Python source in user-supplied 2021 ARPGE ZIP.

Usage (run this on Python 3.11 where stdlib lib2to3 is available):
    python scripts/prepare_arpge_legacy.py /path/to/ARPGE_Sept2021.zip --output third_party/arpge_suite

Two trees are created:
  original/ : unmodified archived files and their original relative paths
  py3_syntax/ : automated lib2to3 conversion of every .py file, plus data

A successful syntax conversion does NOT imply a working numerical code or UI.
Never execute legacy pickles from untrusted sources or import old modules
without auditing import-time Windows path and GUI side effects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
from zipfile import ZipFile


def convert(archive: Path, output: Path) -> dict:
    from lib2to3.refactor import RefactoringTool, get_fixers_from_package

    tool = RefactoringTool(get_fixers_from_package("lib2to3.fixes"))
    results: list[dict] = []
    output.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as z:
        for info in z.infolist():
            rel = PurePosixPath(info.filename)
            if info.is_dir():
                continue
            if rel.is_absolute() or ".." in rel.parts or not rel.parts:
                raise ValueError(f"Unsafe ZIP entry: {info.filename}")
            # Keep source, native phase input and test data; exclude obsolete Windows installers.
            if any(part == "Python_Download_27_win32" for part in rel.parts):
                continue
            if rel.parts[0] == "ARPGE_Sept2021":
                rel = PurePosixPath(*rel.parts[1:])
            if not rel.parts:
                continue
            raw = z.read(info)
            source = output / "original" / str(rel)
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(raw)
            entry = {
                "path": str(rel), "sha256_original": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw), "python": rel.suffix == ".py",
            }
            target = output / "py3_syntax" / str(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            if rel.suffix == ".py":
                try:
                    try:
                        text = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        text = raw.decode("latin-1")
                    transformed = str(tool.refactor_string(text + ("\n" if not text.endswith("\n") else ""), str(rel)))
                    compile(transformed, str(rel), "exec")
                    target.write_text(transformed, encoding="utf-8")
                    entry["syntax"] = "passed"
                except Exception as exc:
                    target.write_bytes(raw)
                    entry["syntax"] = "failed"
                    entry["error"] = f"{type(exc).__name__}: {exc}"
            else:
                target.write_bytes(raw)
            results.append(entry)
    manifest = {
        "input_archive": archive.name,
        "files": len(results),
        "python_files": sum(item["python"] for item in results),
        "python_syntax_passed": sum(item.get("syntax") == "passed" for item in results),
        "python_syntax_failed": sum(item.get("syntax") == "failed" for item in results),
        "results": results,
        "warning": "Syntax port only. No runtime, scientific parity, or redistribution license is asserted.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("third_party/arpge_suite"))
    args = parser.parse_args()
    result = convert(args.archive, args.output)
    print(json.dumps({key: result[key] for key in ("files", "python_files",
          "python_syntax_passed", "python_syntax_failed")}, indent=2))
    if result["python_syntax_failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
