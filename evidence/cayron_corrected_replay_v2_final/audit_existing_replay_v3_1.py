#!/usr/bin/env python3
from __future__ import annotations

"""
Read-only forensic audit v3.1 for the existing corrected replay.

NO scientific solver calls.
NO prediction regeneration.
NO tolerance fitting.
NO repository writes.

Purpose:
  1) verify stored replay predictions against their native seals;
  2) correct the NW/KS source mapping in the previous reveal harness;
  3) strengthen Cu-Al-Ni twin checks to include BOTH plane and direction;
  4) keep literature checks separate from internal-regression checks.

Exit:
  0 = all externally scored checks pass
  1 = one or more scientific checks fail
  2 = integrity failure
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np


EXPECTED_BACKBONE_BLOB = "881f63cd0074daea5b9672b76ea2e90905ef951a"
EXPECTED_CASES = ["case_001", "case_002", "case_003", "case_004", "case_005"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def git_blob(repo: Path, relpath: str) -> str:
    cp = subprocess.run(
        ["git", "hash-object", relpath],
        cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True,
    )
    return cp.stdout.strip()


def load_backbone_verifier(repo: Path):
    rel = "validation/cayron_first_backbone.py"
    actual = git_blob(repo, rel)
    if actual != EXPECTED_BACKBONE_BLOB:
        raise RuntimeError(
            f"backbone blob mismatch: expected {EXPECTED_BACKBONE_BLOB}, got {actual}"
        )
    path = repo / rel
    spec = importlib.util.spec_from_file_location("_audit_backbone", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen backbone verifier")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "verify_prediction_payload", None)
    if fn is None:
        raise RuntimeError("backbone verify_prediction_payload missing")
    return fn


def verify_orientation(payload: Mapping[str, Any], manifest_case: Mapping[str, Any]) -> str:
    if payload.get("mode") != "orientation_topology_blind_prediction":
        raise RuntimeError(
            f"{manifest_case['case_id']}: unexpected orientation payload mode "
            f"{payload.get('mode')!r}"
        )
    if payload.get("case_id") != manifest_case["case_id"]:
        raise RuntimeError(f"{manifest_case['case_id']}: case_id mismatch")
    embedded = payload.get("prediction_sha256")
    recomputed = canonical_sha256(
        {"case_id": payload["case_id"], "prediction": payload.get("prediction")}
    )
    if embedded != recomputed:
        raise RuntimeError(f"{manifest_case['case_id']}: orientation content seal mismatch")
    if embedded != manifest_case.get("prediction_sha256"):
        raise RuntimeError(f"{manifest_case['case_id']}: orientation manifest seal mismatch")
    return embedded


def load_predictions(root: Path, repo: Path):
    manifest_path = root / "campaign_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing {manifest_path}")
    man = json.loads(manifest_path.read_text())

    if man.get("campaign") != "cayron_corrected_replay_v2":
        raise RuntimeError(f"wrong campaign {man.get('campaign')!r}")
    if man.get("oracle_loaded") is not False:
        raise RuntimeError("oracle_loaded is not false in replay manifest")
    if man.get("literature_outputs_loaded") is not False:
        raise RuntimeError("literature_outputs_loaded is not false in replay manifest")

    cases = man.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError("manifest cases missing")
    ids = [c.get("case_id") for c in cases]
    if ids != EXPECTED_CASES:
        raise RuntimeError(f"case set/order mismatch: {ids!r}")

    verify_transform = load_backbone_verifier(repo)
    out = {}

    for c in cases:
        cid = c["case_id"]
        if c.get("status") != "PASS":
            raise RuntimeError(f"{cid}: replay internal status is not PASS")

        path = root / cid / "prediction.json"
        if not path.is_file():
            raise RuntimeError(f"{cid}: prediction missing")
        actual_file = sha256_file(path)
        if actual_file != c.get("prediction_file_sha256"):
            raise RuntimeError(f"{cid}: prediction file SHA mismatch")

        payload = json.loads(path.read_text())
        if c.get("mode") == "orientation_topology":
            native_sha = verify_orientation(payload, c)
        elif c.get("mode") == "transformation":
            native_sha = verify_transform(payload)
            if payload.get("case_id") != cid:
                raise RuntimeError(f"{cid}: transformation case_id mismatch")
            if native_sha != c.get("prediction_sha256"):
                raise RuntimeError(f"{cid}: transformation manifest seal mismatch")
        else:
            raise RuntimeError(f"{cid}: unknown manifest mode {c.get('mode')!r}")

        out[cid] = payload
        print(f"INTEGRITY {cid}: PASS  {native_sha}")

    return man, out


def projective_residual(a: Any, b: Any) -> float:
    x = np.asarray(a, dtype=float).reshape(3)
    y = np.asarray(b, dtype=float).reshape(3)
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    if nx <= 1e-15 or ny <= 1e-15:
        return math.inf
    x /= nx
    y /= ny
    return float(np.linalg.norm(np.cross(x, y)))


def equivalents(v: Any, symmetry: str):
    x = np.asarray(v, dtype=float).reshape(3)
    if symmetry == "orthorhombic_mmm":
        mats = [
            np.diag([sx, sy, sz])
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
    elif symmetry == "monoclinic_2m_b":
        # global sign handled by projective_residual
        mats = [np.eye(3), np.diag([-1.0, 1.0, -1.0])]
    else:
        raise RuntimeError(f"unknown symmetry {symmetry}")
    return [M @ x for M in mats]


def family_residual(actual: Any, expected: Any, symmetry: str) -> float:
    return min(projective_residual(actual, e) for e in equivalents(expected, symmetry))


def topology(pred: Mapping[str, Any]):
    return pred["prediction"]["cayron"]["topology"]


def twin_rows(pred: Mapping[str, Any]):
    return [
        r for r in pred["prediction"]["cayron"]["rows"]
        if r.get("prediction_kind") == "ct_mm_twin"
    ]


def geom(row: Mapping[str, Any], kind: str):
    meta = row.get("metadata", {})
    if kind == "plane":
        return meta.get("plane_product_crystal")
    if kind == "direction":
        return meta.get("direction_product_crystal")
    raise RuntimeError(kind)


def check(cid: str, passed: bool, actual: Any, expected: Any, source_class="external"):
    return {
        "check_id": cid,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
        "source_class": source_class,
    }


def match_twin(
    pred: Mapping[str, Any],
    *,
    check_id: str,
    classification: str | None,
    symmetry: str,
    primary_kind: str,
    primary_expected: Any,
    primary_tol: float,
    shear_expected: float | None = None,
    shear_tol: float | None = None,
    secondary_kind: str | None = None,
    secondary_expected: Any = None,
    secondary_tol: float | None = None,
    source_class: str = "external",
):
    best = None
    for row in twin_rows(pred):
        cls = row.get("metadata", {}).get("twin_classification")
        if classification is not None and cls != classification:
            continue

        v1 = geom(row, primary_kind)
        if v1 is None:
            continue
        r1 = family_residual(v1, primary_expected, symmetry)

        r2 = None
        if secondary_kind is not None:
            v2 = geom(row, secondary_kind)
            if v2 is None:
                continue
            r2 = family_residual(v2, secondary_expected, symmetry)

        sd = None
        if shear_expected is not None:
            if row.get("shear_magnitude") is None:
                continue
            sd = abs(float(row["shear_magnitude"]) - float(shear_expected))

        # selection score only; pass/fail below remains in native tolerances
        score = r1 / max(primary_tol, 1e-30)
        if r2 is not None:
            score = max(score, r2 / max(float(secondary_tol), 1e-30))
        if sd is not None:
            score = max(score, sd / max(float(shear_tol), 1e-30))

        item = {
            "row_id": row.get("row_id"),
            "classification": cls,
            "plane": row.get("metadata", {}).get("plane_product_crystal"),
            "direction": row.get("metadata", {}).get("direction_product_crystal"),
            "shear": row.get("shear_magnitude"),
            "primary_residual": r1,
            "secondary_residual": r2,
            "shear_abs_diff": sd,
        }
        if best is None or score < best[0]:
            best = (score, item)

    passed = best is not None and best[1]["primary_residual"] <= primary_tol
    if secondary_kind is not None:
        passed = passed and best[1]["secondary_residual"] <= float(secondary_tol)
    if shear_expected is not None:
        passed = passed and best[1]["shear_abs_diff"] <= float(shear_tol)

    return {
        "check_id": check_id,
        "passed": bool(passed),
        "actual": None if best is None else best[1],
        "expected": {
            "classification": classification,
            "primary_kind": primary_kind,
            "primary": primary_expected,
            "primary_tol": primary_tol,
            "secondary_kind": secondary_kind,
            "secondary": secondary_expected,
            "secondary_tol": secondary_tol,
            "shear": shear_expected,
            "shear_tol": shear_tol,
        },
        "source_class": source_class,
    }


def orientation_audit(pred: Mapping[str, Any], cid: str, expected):
    report = pred["prediction"]["parallelism_report"]
    candidates = pred["prediction"]["candidates"]

    # source identity is audited from the STORED solver input/result, not from a label
    actual_input = {
        "reference_first": report["reference_first"],
        "moving_first": report["moving_first"],
        "reference_second": report["reference_second"],
        "moving_second": report["moving_second"],
    }

    checks = [
        check(
            f"{cid}.input_relation",
            actual_input == expected["input"],
            actual_input,
            expected["input"],
        )
    ]

    for name, key in [
        ("intersection_order", "full_orientation_intersection_order"),
        ("variant_count", "full_orientation_variant_count"),
        ("operator_count", "full_orientation_operator_count"),
    ]:
        actual = [int(c["topology"]["audit"][key]) for c in candidates]
        unique_actual = sorted(set(actual))
        target = expected[name]
        checks.append(
            check(
                f"{cid}.{name}",
                bool(actual) and unique_actual == [target],
                {
                    "per_candidate": actual,
                    "unique_values": unique_actual,
                    "candidate_count": len(actual),
                },
                target,
            )
        )
    return checks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/workspaces/CuAlNi-CT")
    ap.add_argument("--replay-root", default="/tmp/cayron_corrected_replay_v2-output")
    ap.add_argument("--output", default="/tmp/cayron_forensic_audit_v3.json")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    root = Path(args.replay_root).resolve()

    try:
        _, preds = load_predictions(root, repo)
    except Exception as exc:
        print(f"INTEGRITY FAILURE: {type(exc).__name__}: {exc}")
        return 2

    result = {
        "mode": "READ_ONLY_FORENSIC_AUDIT_V3_1",
        "solver_rerun": False,
        "tolerances_changed_after_result": False,
        "orientation_benchmark_binding": (
            "KS/NW identity is bound to stored plane+direction parallelism, "
            "never inferred from solver topology counts"
        ),
        "cases": [],
    }

    # AUTHORITATIVE OR mapping for FCC gamma -> BCC alpha:
    #
    # Kurdjumov-Sachs (KS):
    #   (110)_alpha || (111)_gamma
    #   [1 -1 1]_alpha || [1 -1 0]_gamma
    #   Cayron groupoid benchmark: |H|=2, N=24, N_O=24
    #
    # Nishiyama-Wassermann (NW):
    #   (110)_alpha || (111)_gamma
    #   [001]_alpha || [1 -1 0]_gamma
    #   Cayron groupoid benchmark: |H|=4, N=12, N_O=7
    #
    # IMPORTANT: identity is bound to the ACTUAL stored parallelism signature.
    # We never infer NW/KS from variant/operator counts, so the benchmark cannot
    # "choose the answer" after seeing solver output.
    orientation_expected = {
        "case_001": {
            "name": "KS",
            "input": {
                "reference_first": "(1 1 1)",
                "moving_first": "(1 1 0)",
                "reference_second": "[1 -1 0]",
                "moving_second": "[1 -1 1]",
            },
            "intersection_order": 2,
            "variant_count": 24,
            "operator_count": 24,
        },
        "case_002": {
            "name": "NW",
            "input": {
                "reference_first": "(1 1 1)",
                "moving_first": "(1 1 0)",
                "reference_second": "[1 -1 0]",
                "moving_second": "[0 0 1]",
            },
            "intersection_order": 4,
            "variant_count": 12,
            "operator_count": 7,
        },
    }

    for cid in ("case_001", "case_002"):
        checks = orientation_audit(preds[cid], cid, orientation_expected[cid])
        result["cases"].append({
            "case_id": cid,
            "identity": orientation_expected[cid]["name"],
            "passed": all(x["passed"] for x in checks),
            "checks": checks,
        })

    # Cayron 2022 B2 -> B19'
    p = preds["case_003"]
    t = topology(p)
    checks = [
        check("003.Hc", t["correspondence_subgroup_order"] == 4,
              t["correspondence_subgroup_order"], 4),
        check("003.variants", t["variant_count"] == 12, t["variant_count"], 12),
        check("003.operators", t["operator_count"] == 7, t["operator_count"], 7),
        match_twin(p, check_id="003.O2.compound.100", classification="compound",
                   symmetry="monoclinic_2m_b", primary_kind="plane",
                   primary_expected=[1,0,0], primary_tol=1e-6,
                   shear_expected=0.2385, shear_tol=5e-4),
        match_twin(p, check_id="003.O2.compound.001", classification="compound",
                   symmetry="monoclinic_2m_b", primary_kind="plane",
                   primary_expected=[0,0,1], primary_tol=1e-6,
                   shear_expected=0.2389, shear_tol=5e-4),
        match_twin(p, check_id="003.O4.typeI", classification="type_I",
                   symmetry="monoclinic_2m_b", primary_kind="plane",
                   primary_expected=[-1,1,1], primary_tol=1e-6,
                   shear_expected=0.3096, shear_tol=5e-4),
        match_twin(p, check_id="003.O4.typeII", classification="type_II",
                   symmetry="monoclinic_2m_b", primary_kind="direction",
                   primary_expected=[-2,1,1], primary_tol=1e-6,
                   shear_expected=0.3096, shear_tol=5e-4),
        match_twin(p, check_id="003.O5.typeI", classification="type_I",
                   symmetry="monoclinic_2m_b", primary_kind="plane",
                   primary_expected=[1,1,1], primary_tol=1e-6,
                   shear_expected=0.1422, shear_tol=5e-4),
        match_twin(p, check_id="003.O5.typeII", classification="type_II",
                   symmetry="monoclinic_2m_b", primary_kind="direction",
                   primary_expected=[2,1,1], primary_tol=1e-6,
                   shear_expected=0.1422, shear_tol=5e-4),
        match_twin(p, check_id="003.O6.typeI", classification="type_I",
                   symmetry="monoclinic_2m_b", primary_kind="plane",
                   primary_expected=[0,1,1], primary_tol=1e-6,
                   shear_expected=0.2804, shear_tol=5e-4),
        match_twin(p, check_id="003.O6.typeII", classification="type_II",
                   symmetry="monoclinic_2m_b", primary_kind="direction",
                   primary_expected=[0,1,1], primary_tol=1e-6,
                   shear_expected=0.2804, shear_tol=5e-4),
    ]
    result["cases"].append({
        "case_id": "case_003",
        "identity": "Cayron 2022 NiTi",
        "passed": all(x["passed"] for x in checks),
        "checks": checks,
    })

    # Cayron 2026 measured Kudoh metric.
    p = preds["case_004"]
    t = topology(p)
    rows = p["prediction"]["cayron"]["rows"]
    exact_habits = [
        r for r in rows
        if r.get("prediction_kind") == "ct_am_habit" and r.get("exact") is True
    ]
    superrows = [
        r for r in rows if r.get("prediction_kind") == "ct_supercompatibility"
    ]
    actual_l2m1 = float(
        p["prediction"]["shared_metric_diagnostics"]["lambda2_minus_one"]
    )
    expected_l2m1 = 4.108 / (math.sqrt(2.0) * 3.01) - 1.0
    checks = [
        check("004.Hc", t["correspondence_subgroup_order"] == 4,
              t["correspondence_subgroup_order"], 4),
        check("004.variants", t["variant_count"] == 12, t["variant_count"], 12),
        check("004.operators", t["operator_count"] == 7, t["operator_count"], 7),
        check("004.no_exact_AM_compatibility", len(exact_habits) == 0,
              len(exact_habits), 0),
        check("004.no_exact_supercompatibility", len(superrows) == 0,
              len(superrows), 0),
        check("004.lambda2_minus_one",
              abs(actual_l2m1 - expected_l2m1) <= 2e-10,
              actual_l2m1, expected_l2m1),
    ]
    result["cases"].append({
        "case_id": "case_004",
        "identity": "Cayron 2026 measured binary NiTi",
        "passed": all(x["passed"] for x in checks),
        "checks": checks,
    })

    # Cu-Al-Ni: external Otsuka/Shimizu full {121} twin elements + {101} family.
    p = preds["case_005"]
    t = topology(p)
    external_checks = [
        match_twin(
            p, check_id="005.external.121.TypeI.full_elements",
            classification="type_I", symmetry="orthorhombic_mmm",
            primary_kind="plane", primary_expected=[1,2,1], primary_tol=8e-4,
            secondary_kind="direction",
            secondary_expected=[1,0.7954,0.5907], secondary_tol=8e-4,
            shear_expected=0.261, shear_tol=5e-4,
        ),
        match_twin(
            p, check_id="005.external.121.TypeII.full_elements",
            classification="type_II", symmetry="orthorhombic_mmm",
            primary_kind="plane",
            primary_expected=[1,1.5036,0.5036], primary_tol=8e-4,
            secondary_kind="direction", secondary_expected=[1,1,1],
            secondary_tol=1e-6,
            shear_expected=0.261, shear_tol=5e-4,
        ),
        # Observed {101} family: external source establishes plane family;
        # it does not supply our CT classification label.
        match_twin(
            p, check_id="005.external.101.family",
            classification=None, symmetry="orthorhombic_mmm",
            primary_kind="plane", primary_expected=[1,0,1], primary_tol=1e-6,
        ),
    ]
    internal_checks = [
        check("005.internal.Hc", t["correspondence_subgroup_order"] == 8,
              t["correspondence_subgroup_order"], 8, "internal_regression"),
        check("005.internal.variants", t["variant_count"] == 6,
              t["variant_count"], 6, "internal_regression"),
        check("005.internal.operators", t["operator_count"] == 3,
              t["operator_count"], 3, "internal_regression"),
        match_twin(
            p, check_id="005.internal.101.compound",
            classification="compound", symmetry="orthorhombic_mmm",
            primary_kind="plane", primary_expected=[1,0,1], primary_tol=1e-6,
            source_class="internal_regression",
        ),
    ]
    checks = external_checks + internal_checks
    result["cases"].append({
        "case_id": "case_005",
        "identity": "Cu-Al-Ni beta1(D03) -> gamma-prime",
        "passed": all(x["passed"] for x in checks),
        "external_passed": all(x["passed"] for x in external_checks),
        "internal_regression_passed": all(x["passed"] for x in internal_checks),
        "checks": checks,
    })

    result["all_cases_passed"] = all(c["passed"] for c in result["cases"])
    result["all_external_checks_passed"] = all(
        x["passed"]
        for c in result["cases"]
        for x in c["checks"]
        if x.get("source_class") == "external"
    )

    out = Path(args.output).resolve()
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("\n=== FORENSIC AUDIT V3.1 — READ ONLY ===")
    print("solver rerun: NO")
    for c in result["cases"]:
        print(f"\n{c['case_id']} {c['identity']}: {'PASS' if c['passed'] else 'FAIL'}")
        for x in c["checks"]:
            print(
                " ",
                "PASS" if x["passed"] else "FAIL",
                x["check_id"],
                f"[{x.get('source_class','external')}]",
            )
            if not x["passed"]:
                print("    actual  =", x["actual"])
                print("    expected=", x["expected"])
    print("\nALL EXTERNAL CHECKS:", "PASS" if result["all_external_checks_passed"] else "FAIL")
    print("ALL CASES:", "PASS" if result["all_cases_passed"] else "FAIL")
    print("report:", out)
    print("report sha256:", sha256_file(out))
    return 0 if result["all_cases_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
