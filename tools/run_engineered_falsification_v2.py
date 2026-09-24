#!/usr/bin/env python3
from __future__ import annotations

import argparse,hashlib,json,os
from pathlib import Path
import subprocess,sys,time

REPO_ROOT=Path(__file__).resolve().parents[1]
repo_text=str(REPO_ROOT)
if repo_text not in sys.path:
    sys.path.insert(0,repo_text)

from validation.engineered.harness import run
from validation.engineered.jsonutil import to_jsonable

STRUCTURAL_TESTS=(
    "tests/test_engineered_framework_integrity.py",
    "tests/test_engineered_exact_oracle.py",
    "tests/test_engineered_degeneracy_boundaries.py",
    "tests/test_engineered_mutation_score.py",
    "tests/test_engineered_metamorphic.py",
    "tests/test_engineered_representation_error_ladder.py",
    "tests/test_engineered_ct_generalized_vs_whitened.py",
    "tests/test_engineered_metric_sqrt_stability.py",
    "tests/test_engineered_cofactor_ptmc_rankone.py",
    "tests/test_engineered_high_precision_conditioning.py",
    "tests/test_engineered_end_to_end_project.py",
    "tests/test_engineered_oracle_firewall.py",
)
EXCLUDED_DIRS={".git",".pytest_cache","__pycache__",".mypy_cache",".ruff_cache",".venv"}
EXCLUDED_SUFFIXES={".pyc",".pyo"}

def git(repo,*args):
    return subprocess.run(["git","-C",str(repo),*args],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

def source_tree_fingerprint(repo):
    h=hashlib.sha256()
    for path in sorted(repo.rglob("*")):
        if not path.is_file(): continue
        rel=path.relative_to(repo)
        if any(part in EXCLUDED_DIRS for part in rel.parts): continue
        if path.suffix in EXCLUDED_SUFFIXES or path.name==".DS_Store": continue
        h.update(str(rel).encode()+b"\0");h.update(path.read_bytes());h.update(b"\0")
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--profile",choices=("quick","full","torture"),default="quick")
    ap.add_argument("--report",default="/tmp/engineered_falsification_v2_report.json")
    args=ap.parse_args()
    repo=REPO_ROOT
    before=source_tree_fingerprint(repo)
    head=git(repo,"rev-parse","HEAD").stdout.strip()
    start=time.time()
    harness=run(args.profile)
    env=os.environ.copy();env["PYTHONHASHSEED"]="0"
    pytest=subprocess.run(
        [sys.executable,"-m","pytest","-q","-vv","--durations=30",*STRUCTURAL_TESTS],
        cwd=repo,env=env,
    )
    after=source_tree_fingerprint(repo)
    report=to_jsonable({
        "campaign":"ENGINEERED_FALSIFICATION_V2_3",
        "profile":args.profile,"head":head,"harness":harness,
        "pytest_returncode":pytest.returncode,
        "repository_mutated_by_runner":before!=after,
        "elapsed_seconds":time.time()-start,
    })
    report["status"]="PASS" if harness["passed"] and pytest.returncode==0 and before==after else "FAIL"
    Path(args.report).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("\n=== ENGINEERED FALSIFICATION V2.3 ===")
    print("profile:",args.profile)
    print("status:",report["status"])
    print("counts:",harness["counts"])
    ms=harness["mutation_score"]
    print(f"mutation score: {ms['killed']}/{ms['total']} killed; survivors={ms['survivors']}")
    pf=harness["physical_failure_examples"]
    mf=harness["metamorphic_failure_examples"]
    if pf:
        print("first physical failure:",{k:pf[0].get(k) for k in ("case_id","contract","failure_class","residual","allowed")})
    if mf:
        first=mf[0]
        names=[x.get("name") for x in first.get("failures",[])[:3]]
        print("first metamorphic failure:",first.get("case"),names)
    print("repository_mutated_by_runner:",report["repository_mutated_by_runner"])
    print("report:",args.report)
    print("representation ladder: /tmp/engineered_representation_error_ladder.json")
    print("conditioning sweep: /tmp/engineered_conditioning_sweep.json")
    print("failure ledgers: /tmp/cualni_engineered_failures/")
    return 0 if report["status"]=="PASS" else 1

if __name__=="__main__":
    raise SystemExit(main())
