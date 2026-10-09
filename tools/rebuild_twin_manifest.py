from __future__ import annotations

"""Rehash ONLY the deliberately tracked research-workbench files.

Avoid the previous unsafe advice to hash every file under a repository:
that would include generated caches, untracked local files and changing Git
artifacts. This tool validates the tracked path list before rewriting it.
"""

from hashlib import sha256
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"


def tracked_paths() -> tuple[str, ...]:
    entries = []
    for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, separator, name = raw.partition("  ")
        if not separator or len(digest) != 64 or not name:
            raise ValueError(f"Invalid manifest entry: {raw[:100]!r}")
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or name in entries or name == MANIFEST.name:
            raise ValueError(f"Unsafe or repeated manifest path: {name}")
        entries.append(name)
    return tuple(sorted(entries))


def build_lines() -> list[str]:
    lines = []
    for name in tracked_paths():
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing manifest-tracked file: {name}")
        lines.append(f"{sha256(path.read_bytes()).hexdigest()}  {name}")
    return lines


def main() -> None:
    expected = "\n".join(build_lines()) + "\n"
    if "--check" in sys.argv:
        if MANIFEST.read_text(encoding="utf-8") != expected:
            raise SystemExit("Manifest does not match repository files; regenerate after editing")
        print("Manifest verified")
    else:
        MANIFEST.write_text(expected, encoding="utf-8", newline="\n")
        print(f"Manifest regenerated with {len(tracked_paths())} tracked paths")


if __name__ == "__main__":
    main()
