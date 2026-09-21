#!/usr/bin/env python3
from __future__ import annotations

"""Enforce experimental-validation scope against the frozen scientific core.

The branch is allowed to add validation harnesses, evidence, validation-only
tests/docs, and format/convention adapters.  It is not allowed to modify the
frozen scientific backend.

Important implementation detail:
Git diffs are evaluated with ``--no-renames``.  A rename therefore appears as
DELETE(old_path) + ADD(new_path), which prevents a core file from being moved
into an allowed directory to bypass the path guard.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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


def run_text(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def run_bytes(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def output(*args: str) -> str:
    return run_text(*args).stdout.strip()


def normalize_repo_path(path: str) -> str | None:
    """Normalize only explicit './' prefixes; never use character lstrip().

    ``str.lstrip("./")`` is incorrect here because it strips *characters*, so
    `.github/...` becomes `github/...`.  That bug caused the first scope-lock
    attempt to reject its own workflow file.
    """

    value = path.replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]

    if not value or value.startswith("/"):
        return None

    pure = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in pure.parts):
        return None

    return pure.as_posix()


def allowed_path(path: str) -> bool:
    normalized = normalize_repo_path(path)
    if normalized is None:
        return False
    return (
        normalized in ALLOWED_EXACT
        or any(normalized.startswith(prefix) for prefix in ALLOWED_PREFIXES)
    )


def blocked_artifact(path: str) -> bool:
    normalized = normalize_repo_path(path)
    if normalized is None:
        return True
    return Path(normalized).suffix.lower() in BLOCKED_SUFFIXES


def parse_name_status_z(data: bytes) -> list[Change]:
    """Parse `git diff --name-status --no-renames -z`.

    With --no-renames every record is exactly STATUS NUL PATH NUL.  This avoids
    quoted-path parsing bugs and, critically, makes renames visible as D+A.
    """

    if not data:
        return []
    fields = data.split(b"\0")
    if fields[-1] == b"":
        fields.pop()
    if len(fields) % 2:
        raise ValueError(
            "unexpected NUL-delimited git name-status record count"
        )

    changes: list[Change] = []
    for index in range(0, len(fields), 2):
        status = fields[index].decode("utf-8", errors="strict")
        path = fields[index + 1].decode("utf-8", errors="strict")
        changes.append(Change(status=status, path=path))
    return changes


def committed_changes(base_tag: str, head: str) -> list[Change]:
    result = run_bytes(
        "git",
        "diff",
        "--no-renames",
        "--name-status",
        "-z",
        f"{base_tag}...{head}",
    )
    return parse_name_status_z(result.stdout)


def working_tree_changes() -> list[Change]:
    tracked = parse_name_status_z(
        run_bytes(
            "git",
            "diff",
            "--no-renames",
            "--name-status",
            "-z",
            "HEAD",
        ).stdout
    )
    untracked_raw = run_bytes(
        "git",
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    ).stdout
    untracked = []
    for raw in untracked_raw.split(b"\0"):
        if not raw:
            continue
        untracked.append(
            Change(
                status="??",
                path=raw.decode("utf-8", errors="strict"),
            )
        )
    return tracked + untracked


def tracked_file_size_at_head(path: str, head: str) -> int | None:
    normalized = normalize_repo_path(path)
    if normalized is None:
        return None
    result = run_text(
        "git",
        "cat-file",
        "-s",
        f"{head}:{normalized}",
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def tracked_file_mode_at_head(path: str, head: str) -> str | None:
    normalized = normalize_repo_path(path)
    if normalized is None:
        return None
    result = run_text(
        "git",
        "ls-tree",
        head,
        "--",
        normalized,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    # "<mode> <type> <object>\t<path>"
    return result.stdout.split(None, 1)[0]


def working_file_size(path: str) -> int | None:
    normalized = normalize_repo_path(path)
    if normalized is None:
        return None
    candidate = Path(normalized)
    if not candidate.is_file() or candidate.is_symlink():
        return None
    return candidate.stat().st_size


def working_file_is_symlink(path: str) -> bool:
    normalized = normalize_repo_path(path)
    return normalized is not None and Path(normalized).is_symlink()


def validate_changes(
    changes: list[Change],
    *,
    head: str,
    working_tree: bool,
) -> list[str]:
    problems: list[str] = []

    for change in changes:
        normalized = normalize_repo_path(change.path)
        display = change.path if normalized is None else normalized

        if normalized is None or not allowed_path(normalized):
            problems.append(
                f"{display}: outside experimental-validation scope "
                f"(status {change.status})"
            )
            continue

        if change.status.startswith("D"):
            continue

        if blocked_artifact(normalized):
            problems.append(
                f"{normalized}: raw/binary experimental artifact is blocked; "
                "store it outside git and commit hashes/manifests instead"
            )

        if working_tree:
            if working_file_is_symlink(normalized):
                problems.append(
                    f"{normalized}: symlinks are not allowed in validation scope"
                )
            size = working_file_size(normalized)
        else:
            mode = tracked_file_mode_at_head(normalized, head)
            if mode in {"120000", "160000"}:
                problems.append(
                    f"{normalized}: symlinks/submodules are not allowed "
                    "in validation scope"
                )
            size = tracked_file_size_at_head(normalized, head)

        if size is not None and size > MAX_TRACKED_FILE_BYTES:
            problems.append(
                f"{normalized}: {size} bytes exceeds the 5 MiB "
                "validation-branch tracked-file limit"
            )

    return problems


def verify_frozen_anchor(base_tag: str, head: str) -> list[str]:
    problems: list[str] = []

    tag_result = run_text(
        "git", "rev-list", "-n", "1", base_tag, check=False
    )
    if tag_result.returncode != 0 or not tag_result.stdout.strip():
        return [f"frozen base tag/ref cannot be resolved: {base_tag}"]
    tag_commit = tag_result.stdout.strip()

    if base_tag == FROZEN_TAG and tag_commit != FROZEN_COMMIT:
        problems.append(
            f"{FROZEN_TAG} resolves to {tag_commit}, "
            f"expected {FROZEN_COMMIT}"
        )

    head_commit = output("git", "rev-parse", head)
    ancestor = run_text(
        "git",
        "merge-base",
        "--is-ancestor",
        tag_commit,
        head_commit,
        check=False,
    )
    if ancestor.returncode != 0:
        problems.append(
            f"{head_commit} does not descend from frozen anchor "
            f"{base_tag} ({tag_commit})"
        )

    return problems


def self_test() -> None:
    # Exact/prefix allowlist behavior.
    assert allowed_path(
        ".github/workflows/experimental-validation-guard.yml"
    )
    assert allowed_path(
        "./.github/workflows/experimental-validation-guard.yml"
    )
    assert allowed_path("validation/cases/a.json")
    assert allowed_path("src/cualni_cryst/io_adapters/foo.py")
    assert allowed_path("tests/validation/test_x.py")

    # Core paths remain forbidden.
    assert not allowed_path("src/cualni_cryst/twinning_ct.py")
    assert not allowed_path("src/cualni_cryst/ebsd_pipeline.py")
    assert not allowed_path("pyproject.toml")

    # No traversal/absolute-path tricks.
    assert not allowed_path("../validation/a.json")
    assert not allowed_path("/validation/a.json")

    # Artifact policy.
    assert blocked_artifact("validation/raw/map.ctf")
    assert blocked_artifact("evidence/results.npz")
    assert not blocked_artifact("evidence/result.json")

    # NUL parser + no-renames contract.
    parsed = parse_name_status_z(
        b"D\0src/cualni_cryst/ebsd_pipeline.py\0"
        b"A\0validation/ebsd_pipeline.py\0"
    )
    assert len(parsed) == 2
    assert parsed[0].status == "D"
    assert parsed[0].path == "src/cualni_cryst/ebsd_pipeline.py"
    assert parsed[1].status == "A"
    assert parsed[1].path == "validation/ebsd_pipeline.py"

    print("scope guard self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-tag", default=FROZEN_TAG)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument(
        "--working-tree",
        action="store_true",
        help="also validate current tracked/untracked working-tree changes",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    if not Path(".git").exists():
        print("ERROR: run from repository root", file=sys.stderr)
        return 2

    problems = verify_frozen_anchor(args.base_tag, args.head)

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
            "\nCore mathematics is frozen. Preserve any genuine "
            "counterexample and investigate it on a separate branch from "
            "ct-ebsd-reliability-gate-v1.",
            file=sys.stderr,
        )
        return 1

    print("\nEXPERIMENTAL-VALIDATION SCOPE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
