#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, subprocess

EXPECTED_BRANCH = "science/universal-generative-validation-v1"
EXPECTED_HEAD = "84e4338ac7b112e9f0fd1844748176a2388ddc8d"
TARGET = "tools/run_engineered_falsification_v2.py"
CONTENT = '#!/usr/bin/env python3\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport os\nfrom pathlib import Path\nimport subprocess\nimport sys\nimport time\n\n# When this file is executed as\n#   python tools/run_engineered_falsification_v2.py\n# Python places ``tools/`` rather than the repository root on sys.path.\n# Insert the repo root explicitly before importing the validation package.\nREPO_ROOT = Path(__file__).resolve().parents[1]\nrepo_text = str(REPO_ROOT)\nif repo_text not in sys.path:\n    sys.path.insert(0, repo_text)\n\nfrom validation.engineered.harness import run\n\n\nSTRUCTURAL_TESTS = (\n    "tests/test_engineered_exact_oracle.py",\n    "tests/test_engineered_degeneracy_boundaries.py",\n    "tests/test_engineered_mutation_score.py",\n    "tests/test_engineered_metamorphic.py",\n    "tests/test_engineered_cofactor_ptmc_rankone.py",\n    "tests/test_engineered_high_precision_conditioning.py",\n    "tests/test_engineered_end_to_end_project.py",\n    "tests/test_engineered_oracle_firewall.py",\n)\n\n\ndef git(repo, *args):\n    return subprocess.run(\n        ["git", "-C", str(repo), *args],\n        text=True,\n        stdout=subprocess.PIPE,\n        stderr=subprocess.PIPE,\n    )\n\n\ndef fingerprint(repo):\n    return (\n        git(repo, "rev-parse", "HEAD").stdout.strip(),\n        git(repo, "diff", "--binary", "HEAD").stdout,\n    )\n\n\ndef main():\n    ap = argparse.ArgumentParser()\n    ap.add_argument("--profile", choices=("quick", "full", "torture"), default="quick")\n    ap.add_argument("--report", default="/tmp/engineered_falsification_v2_report.json")\n    args = ap.parse_args()\n\n    repo = REPO_ROOT\n    before = fingerprint(repo)\n    started = time.time()\n\n    harness_report = run(args.profile)\n\n    env = os.environ.copy()\n    env["PYTHONHASHSEED"] = "0"\n    pytest = subprocess.run(\n        [\n            sys.executable,\n            "-m",\n            "pytest",\n            "-q",\n            "-vv",\n            "--durations=30",\n            *STRUCTURAL_TESTS,\n        ],\n        cwd=repo,\n        env=env,\n    )\n\n    after = fingerprint(repo)\n    report = {\n        "campaign": "ENGINEERED_FALSIFICATION_V2",\n        "profile": args.profile,\n        "head": before[0],\n        "harness": harness_report,\n        "pytest_returncode": pytest.returncode,\n        "repository_mutated_by_runner": before != after,\n        "elapsed_seconds": time.time() - started,\n    }\n    report["status"] = (\n        "PASS"\n        if harness_report["passed"] and pytest.returncode == 0 and before == after\n        else "FAIL"\n    )\n    Path(args.report).write_text(\n        json.dumps(report, indent=2, sort_keys=True) + "\\n",\n        encoding="utf-8",\n    )\n\n    print("\\n=== ENGINEERED FALSIFICATION V2 ===")\n    print("profile:", args.profile)\n    print("status:", report["status"])\n    print("mutation score:", harness_report["mutation_score"])\n    print("physical failures:", harness_report["physical_failures"])\n    print("metamorphic failures:", harness_report["metamorphic_failures"])\n    print("repository_mutated_by_runner:", report["repository_mutated_by_runner"])\n    print("report:", args.report)\n    print("failure ledgers: /tmp/cualni_engineered_failures/")\n\n    return 0 if report["status"] == "PASS" else 1\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'

def git(repo, *args, check=True):
    p = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if check and p.returncode:
        raise RuntimeError("git " + " ".join(args) + "\n" + p.stdout + p.stderr)
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()

    branch = git(repo, "branch", "--show-current").stdout.strip()
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    print("branch:", branch)
    print("HEAD:  ", head)
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"wrong branch: {branch}")
    if head != EXPECTED_HEAD:
        raise RuntimeError(f"wrong HEAD: {head}")

    target = repo / TARGET
    if not target.exists():
        raise RuntimeError("missing installed runner: " + TARGET)

    old = target.read_text(encoding="utf-8")
    if "from validation.engineered.harness import run" not in old:
        raise RuntimeError("unexpected runner content; refusing to patch")

    target.write_text(CONTENT, encoding="utf-8")
    compile(CONTENT, TARGET, "exec")

    print("patched:", TARGET)
    print("production solver files changed: NO")
    print("cause fixed: repo root is inserted into sys.path before validation import")
    print("sha256:", hashlib.sha256(target.read_bytes()).hexdigest())
    print("\nNEXT:")
    print("python tools/run_engineered_falsification_v2.py --profile full")

if __name__ == "__main__":
    main()
