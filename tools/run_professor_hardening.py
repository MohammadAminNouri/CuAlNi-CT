#!/usr/bin/env python3
from __future__ import annotations

"""Professor-readiness hardening runner.

No repository mutation, no network access, no oracle files.
The quick profile runs only new additive facade/adversarial tests.
Use --full exactly once before freezing a milestone.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


QUICK = [
    "tests/test_user_workflow.py",
    "tests/test_user_input_adversarial.py",
    "tests/test_representation_adversarial.py",
    "tests/test_weak_operator_adversarial_v2.py",
]

CORE = QUICK + [
    "tests/test_representation_bridge.py",
    "tests/test_physical_equivalence_validation.py",
    "tests/test_cayron_exact_robustness.py",
    "tests/test_ct_generality_campaign.py",
    "tests/test_weak_operator_engine.py",
    "tests/test_weak_twins.py",
    "tests/test_operator_crosslock.py",
    "tests/test_units_and_console_clarity.py",
    "tests/test_scientific_contracts.py",
    "tests/test_project_io.py",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true")
    group.add_argument("--core", action="store_true")
    group.add_argument("--full", action="store_true")
    ap.add_argument("--repo", default=".")
    ap.add_argument(
        "--report",
        default="/tmp/professor_hardening_report.json",
    )
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if args.full:
        targets = ["tests"]
        profile = "full"
    elif args.core:
        targets = CORE
        profile = "core"
    else:
        targets = QUICK
        profile = "quick"

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--durations=15",
        *targets,
    ]
    start = time.monotonic()
    cp = subprocess.run(command, cwd=repo, text=True)
    elapsed = time.monotonic() - start

    report = {
        "profile": profile,
        "command": command,
        "elapsed_seconds": elapsed,
        "returncode": cp.returncode,
        "passed": cp.returncode == 0,
        "repository_mutated_by_runner": False,
        "oracle_loaded": False,
    }
    Path(args.report).write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print()
    print("=== PROFESSOR HARDENING ===")
    print("profile:", profile)
    print("elapsed_seconds:", f"{elapsed:.3f}")
    print("status:", "PASS" if cp.returncode == 0 else "FAIL")
    print("report:", args.report)
    return cp.returncode


if __name__ == "__main__":
    raise SystemExit(main())
