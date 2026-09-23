#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


NEW_TESTS = [
    "tests/test_ct_basis_covariance_second_layer.py",
    "tests/test_groupoid_hypergroup_exact.py",
    "tests/test_cayron_2026_full_supercompatibility.py",
    "tests/test_cayron_2026_o2_type_i_source_audit.py",
    "tests/test_blind_oracle_firewall_v2.py",
]


def run_with_log(
    cmd: list[str],
    log_path: Path,
) -> int:
    """Run pytest, stream output to terminal, and persist the same output."""

    print("+", " ".join(cmd), flush=True)
    print("log:", log_path, flush=True)

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return int(process.wait())


def git_status() -> str:
    p = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return p.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("quick", "full"),
        default="quick",
        help="quick skips the exhaustive all-32 point-group property test",
    )
    args = parser.parse_args()

    before = git_status()
    start = time.perf_counter()

    if args.profile == "quick":
        tests = [
            "tests/test_ct_basis_covariance_second_layer.py",
            "tests/test_cayron_2026_full_supercompatibility.py",
            "tests/test_blind_oracle_firewall_v2.py",
            "tests/test_groupoid_hypergroup_exact.py::test_representative_crystal_families_obey_exact_double_coset_hypergroup_laws",
        ]
    else:
        tests = NEW_TESTS

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--maxfail=1",
        "--durations=15",
        *tests,
    ]
    log_path = Path(
        f"/tmp/science_novel_hardening_{args.profile}.log"
    )
    returncode = run_with_log(cmd, log_path)
    elapsed = time.perf_counter() - start
    after = git_status()

    report = {
        "schema_version": 2,
        "profile": args.profile,
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "passed": returncode == 0,
        "repository_mutated_by_runner": before != after,
        "tests": tests,
        "log": str(log_path),
    }
    out = Path("/tmp/science_novel_hardening_report.json")
    out.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== SCIENCE NOVEL HARDENING V1 ===")
    print(f"profile: {args.profile}")
    print(f"elapsed_seconds: {elapsed:.3f}")
    print(f"status: {'PASS' if returncode == 0 else 'FAIL'}")
    print(f"repository_mutated_by_runner: {before != after}")
    print(f"report: {out}")

    if before != after:
        print("ERROR: test runner changed the working tree.")
        return 97
    return int(returncode)


if __name__ == "__main__":
    raise SystemExit(main())
