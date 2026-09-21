#!/usr/bin/env python3
from __future__ import annotations

"""Enforce the scope of the experimental-validation branch.

The scientific core is frozen at `ct-ebsd-reliability-gate-v1`.
This branch may add validation harnesses, evidence, input adapters, examples,
and validation-specific tests/docs. It may not modify the frozen mathematical
backend.

This is a scope guard, not a scientific test.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


FROZEN_TAG = "ct-ebsd-reliability-gate-v1"
FROZEN_COMMIT = "f7c616379108c0ec06d804e4692cd437c35ec4e0"

ALLOWED_EXACT = {
    ".github/workflows/experimental-validation-guard.yml",
    "tools/check_experimental_validation_scope.py",
}

ALLOWED_PREFIXES = (
    "validation/",
    "evidence/",
    "docs/validation/",
    "examples/validation/",
    "tests/validation/",
    "src/cualni_cryst/io_adapters/",
)

# Raw experimental data and opaque/binary artifacts should not be committed.
# Store them outside git and commit only hashes/manifests/compact evidence.
BLOCKED_SUFFIXES = {
    ".ang",
    ".ctf",
    ".h5",
    ".hdf5",
    ".h5oina",
    ".h5ebsd",
    ".npz",
    ".npy",
    ".zip",
    ".7z",
    ".rar",
    ".tif",
    ".tiff",
    ".bmp",
    ".raw",
}

MAX_TRACKED_FILE_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class Change:
    status: str
    path: str


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def output(*args: str) -> str:
    return run(*args).stdout.strip()


def allowed_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lstrip("./")
    return (
        normalized in ALLOWED_EXACT
        or any(normalized.startswith(prefix) for prefix in ALLOWED_PREFIXES)
    )


def blocked_artifact(path: str) -> bool:
    return Path(path).suffix.lower() in BLOCKED_SUFFIXES


def parse_name_status(text: str) -> list[Change]:
    changes: list[Change] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        fields = raw.split("\t")
        status = fields[0]
        # Rename/copy statuses have old + new path; validate the destination.
        if status.startswith(("R", "C")) and len(fields) >= 3:
            path = fields[-1]
        elif len(fields) >= 2:
            path = fields[-1]
        else:
            raise ValueError(f"cannot parse git name-status line: {raw!r}")
        changes.append(Change(status=status, path=path))
    return changes


def committed_changes(base_tag: str, head: str) -> list[Change]:
    text = output(
        "git",
        "diff",
        "--name-status",
        f"{base_tag}...{head}",
    )
    return parse_name_status(text)


def working_tree_changes() -> list[Change]:
    result = run(
        "git",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    changes: list[Change] = []
    for raw in result.stdout.splitlines():
        if not raw:
            continue
        status = raw[:2].strip() or "?"
        path = raw[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        changes.append(Change(status=status, path=path))
    return changes


def tracked_file_size_at_head(path: str, head: str) -> int | None:
    result = run("git", "cat-file", "-s", f"{head}:{path}", check=False)
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def working_file_size(path: str) -> int | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    return candidate.stat().st_size


def validate_changes(
    changes: list[Change],
    *,
    head: str,
    working_tree: bool,
) -> list[str]:
    problems: list[str] = []

    for change in changes:
        path = change.path.replace("\\", "/")

        if not allowed_path(path):
            problems.append(
                f"{path}: outside experimental-validation scope "
                f"(status {change.status})"
            )
            continue

        if change.status.startswith("D"):
            # Deleting validation-only files is okay. The core cannot be here
            # because it is outside the allowlist.
            continue

        if blocked_artifact(path):
            problems.append(
                f"{path}: raw/binary experimental artifact is blocked; "
                "store it outside git and commit a hash/manifest instead"
            )

        size = (
            working_file_size(path)
            if working_tree
            else tracked_file_size_at_head(path, head)
        )
        if size is not None and size > MAX_TRACKED_FILE_BYTES:
            problems.append(
                f"{path}: {size} bytes exceeds the 5 MiB validation-branch "
                "tracked-file limit"
            )

    return problems


def verify_frozen_anchor(base_tag: str) -> list[str]:
    problems: list[str] = []
    tag_commit = output("git", "rev-list", "-n", "1", base_tag)
    if base_tag == FROZEN_TAG and tag_commit != FROZEN_COMMIT:
        problems.append(
            f"{FROZEN_TAG} resolves to {tag_commit}, expected {FROZEN_COMMIT}"
        )

    head = output("git", "rev-parse", "HEAD")
    ancestor = run(
        "git",
        "merge-base",
        "--is-ancestor",
        tag_commit,
        head,
        check=False,
    )
    if ancestor.returncode != 0:
        problems.append(
            f"HEAD {head} does not descend from frozen anchor "
            f"{base_tag} ({tag_commit})"
        )
    return problems


def self_test() -> None:
    assert allowed_path("validation/cases/a.json")
    assert allowed_path("src/cualni_cryst/io_adapters/foo.py")
    assert allowed_path("tests/validation/test_x.py")
    assert not allowed_path("src/cualni_cryst/twinning_ct.py")
    assert not allowed_path("src/cualni_cryst/ebsd_pipeline.py")
    assert not allowed_path("pyproject.toml")
    assert blocked_artifact("validation/raw/map.ctf")
    assert blocked_artifact("evidence/results.npz")
    assert not blocked_artifact("evidence/result.json")
    print("scope guard self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-tag", default=FROZEN_TAG)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument(
        "--working-tree",
        action="store_true",
        help="also validate tracked/untracked working-tree changes",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    if not Path(".git").exists():
        print("ERROR: run from repository root", file=sys.stderr)
        return 2

    problems = verify_frozen_anchor(args.base_tag)

    committed = committed_changes(args.base_tag, args.head)
    problems.extend(
        validate_changes(
            committed,
            head=args.head,
            working_tree=False,
        )
    )

    if args.working_tree:
        working = working_tree_changes()
        problems.extend(
            validate_changes(
                working,
                head=args.head,
                working_tree=True,
            )
        )

    print(f"frozen anchor: {args.base_tag}")
    print(f"head:          {output('git', 'rev-parse', args.head)}")
    print("changed paths:")
    for item in committed:
        print(f"  {item.status:>3}  {item.path}")
    if not committed:
        print("  (none)")

    if problems:
        print("\nEXPERIMENTAL-VALIDATION SCOPE: FAIL", file=sys.stderr)
        for problem in problems:
            print(f" - {problem}", file=sys.stderr)
        print(
            "\nCore mathematics is frozen. If a genuine counterexample has "
            "been found, preserve the reproducer and open a separate "
            "investigation branch from ct-ebsd-reliability-gate-v1.",
            file=sys.stderr,
        )
        return 1

    print("\nEXPERIMENTAL-VALIDATION SCOPE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
