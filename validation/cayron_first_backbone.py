from __future__ import annotations

"""Validation-only Cayron-first multi-theory backbone.

This file does not implement CT, Ball--James or PTMC equations. It calls the
frozen CuAlNi-CT scientific core, seals blind predictions, and only then allows
a separately held literature/experimental oracle to be compared.
"""

from dataclasses import asdict, is_dataclass
from enum import Enum
import argparse, hashlib, json, math, os, platform, subprocess, sys, tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_FROM_FILE = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "validation" else None
if REPO_FROM_FILE is not None and (REPO_FROM_FILE / "src").is_dir():
    sys.path.insert(0, str(REPO_FROM_FILE / "src"))

from cualni_cryst.group_theory import correspondence_groupoid, representatives
from cualni_cryst.project_io import load_project
from cualni_cryst.stretch import metric_native_stretch_spectrum
from cualni_cryst.theory_unified import (
    PTMCMode,
    TheoryComparisonAdapter,
    TheoryKind,
    exact_symmetry_group_for_ct,
)

SCHEMA_VERSION = 1
ORACLE_SCHEMA_VERSION = 1
FROZEN_CORE_TAG = "ct-ebsd-reliability-gate-v1"
FROZEN_CORE_COMMIT = "f7c616379108c0ec06d804e4692cd437c35ec4e0"
FORBIDDEN_BLIND_KEYS = frozenset({
    "answer", "benchmark_expected", "expected", "expected_output",
    "ground_truth", "literature_answer", "oracle", "target_result",
    "validation_answer",
})

CAYRON_AUDIT_TESTS = (
    "tests/test_ct_permanent_benchmarks.py",
    "tests/test_cayron_exact_robustness.py",
    "tests/test_ct_generality_campaign.py",
    "tests/test_ct_twinning.py",
    "tests/test_ct_weak_orientation.py",
    "tests/test_weak_operator_engine.py",
    "tests/test_weak_twins.py",
    "tests/test_theory_unified.py",
    "tests/test_ball_james_adapter.py",
    "tests/test_ptmc_adapter.py",
)

class CampaignError(RuntimeError):
    pass


def _run(*args: str, cwd: Path | None = None, check: bool = True):
    return subprocess.run(
        list(args), cwd=str(cwd) if cwd else None, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check,
    )


def _git(repo: Path, *args: str) -> str:
    try:
        return _run("git", *args, cwd=repo).stdout.strip()
    except Exception as exc:
        raise CampaignError(f"git command failed: git {' '.join(args)}") from exc


def _repo_root() -> Path:
    try:
        return Path(_git(Path.cwd(), "rev-parse", "--show-toplevel")).resolve()
    except CampaignError as exc:
        raise CampaignError("Run from inside the CuAlNi-CT checkout") from exc


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Enum): return value.value
    if isinstance(value, Path): return str(value)
    if isinstance(value, set): return sorted(_jsonable(x) for x in value)
    if isinstance(value, tuple): return [_jsonable(x) for x in value]
    if isinstance(value, list): return [_jsonable(x) for x in value]
    if isinstance(value, Mapping): return {str(k): _jsonable(v) for k, v in value.items()}
    if is_dataclass(value): return _jsonable(asdict(value))
    if isinstance(value, float) and not math.isfinite(value):
        raise CampaignError("Refusing non-finite float in sealed output")
    if isinstance(value, (str, int, float, bool)) or value is None: return value
    if hasattr(value, "tolist"):
        try: return _jsonable(value.tolist())
        except Exception: pass
    return str(value)


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _scan_forbidden(value: Any, path: str = "$") -> list[str]:
    out: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            k = str(key)
            if k.casefold() in FORBIDDEN_BLIND_KEYS: out.append(f"{path}.{k}")
            out.extend(_scan_forbidden(child, f"{path}.{k}"))
    elif isinstance(value, list):
        for i, child in enumerate(value): out.extend(_scan_forbidden(child, f"{path}[{i}]"))
    return out


def _source_audit(source: str) -> dict[str, Any]:
    if source.startswith("preset:"):
        return {"source_kind": "preset", "source": source, "source_sha256": None}
    path = Path(source).expanduser().resolve()
    if not path.is_file(): raise CampaignError(f"Project file not found: {path}")
    try: raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc: raise CampaignError("Invalid project JSON") from exc
    leaks = _scan_forbidden(raw)
    if leaks:
        raise CampaignError("Blind prediction refused; answer-bearing keys found: " + ", ".join(leaks))
    return {"source_kind": "json_file", "source": str(path), "source_sha256": _file_sha256(path)}


def validate_frozen_core(repo: Path, require_clean: bool = True) -> dict[str, Any]:
    resolved = _git(repo, "rev-list", "-n", "1", FROZEN_CORE_TAG)
    if resolved != FROZEN_CORE_COMMIT:
        raise CampaignError(f"Frozen core tag mismatch: {resolved}")
    head = _git(repo, "rev-parse", "HEAD")
    if _run("git", "merge-base", "--is-ancestor", FROZEN_CORE_COMMIT, head,
            cwd=repo, check=False).returncode != 0:
        raise CampaignError("HEAD does not descend from frozen core")
    changed = _git(repo, "diff", "--name-only", FROZEN_CORE_TAG, "--", "src/cualni_cryst").splitlines()
    illegal = [p for p in changed if not p.startswith("src/cualni_cryst/io_adapters/")]
    if illegal: raise CampaignError("Scientific core drift detected: " + ", ".join(illegal))
    if require_clean:
        dirty = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
        if dirty: raise CampaignError("Prediction requires a clean working tree:\n" + dirty)
    return {"frozen_core_tag": FROZEN_CORE_TAG, "frozen_core_commit": FROZEN_CORE_COMMIT,
            "validation_head": head, "branch": _git(repo, "branch", "--show-current")}


def _exact_strings(matrix: Any) -> list[list[str]]:
    return [[str(matrix[i, j]) for j in range(int(matrix.cols))] for i in range(int(matrix.rows))]


def _scientific_input(project: Any, transformation_id: str) -> dict[str, Any]:
    tr = project.transformation(transformation_id)
    parent = project.phase(tr.parent_phase_id)
    product = project.phase(tr.product_phase_id)
    return {
        "project_id": getattr(project, "project_id", ""),
        "project_title": getattr(project, "title", ""),
        "parent": _jsonable(parent.to_dict()),
        "product": _jsonable(product.to_dict()),
        "transformation": _jsonable(tr.to_dict()),
        "numerical_policy": _jsonable(project.numerical_policy),
    }


def _topology(project: Any, transformation_id: str) -> dict[str, Any]:
    tr = project.transformation(transformation_id)
    parent = project.phase(tr.parent_phase_id)
    product = project.phase(tr.product_phase_id)
    tol = max(float(project.numerical_policy.representation), float(project.numerical_policy.algebraic))
    ga = exact_symmetry_group_for_ct(list(parent.symmetry_matrices()), parent.lattice.metric(),
                                     tolerance=tol, label="parent")
    gm = exact_symmetry_group_for_ct(list(product.symmetry_matrices()), product.lattice.metric(),
                                     tolerance=tol, label="product")
    g = correspondence_groupoid(ga, gm, tr.correspondence)
    return {
        "parent_group_order": len(ga), "product_group_order": len(gm),
        "correspondence_subgroup_order": len(g.subgroup),
        "variant_count": g.n_variants, "operator_count": g.n_operators,
        "burnside_operator_count": g.burnside_count,
        "lagrange_identity": len(g.subgroup) * g.n_variants == len(ga),
        "subgroup_exact_matrices": [_exact_strings(x) for x in g.subgroup],
        "variant_representatives": [_exact_strings(x) for x in representatives(g.variants)],
        "operator_representatives": [_exact_strings(x) for x in representatives(g.operators)],
        "operator_summaries": _jsonable(g.summaries),
        "operator_adjacency": _jsonable(g.adjacency),
        "inverse_operator_map": _jsonable(g.inverse_operators),
        "operator_composition": [[sorted(int(x) for x in cell) for cell in row] for row in g.composition],
    }


def _metric_diagnostics(project: Any, transformation_id: str) -> dict[str, Any]:
    tr = project.transformation(transformation_id)
    a = project.phase(tr.parent_phase_id)
    m = project.phase(tr.product_phase_id)
    s = metric_native_stretch_spectrum(a.lattice.metric(), m.lattice.metric(), tr.correspondence)
    return {
        "mu": _jsonable(s.mu), "lambda": _jsonable(s.lambdas),
        "lambda2_minus_one": float(s.lambdas[1] - 1.0),
        "metric_orthonormality_residual": float(s.metric_orthonormality_residual),
        "generalized_eigen_equation_residual": float(s.eigen_equation_residual),
        "note": "Shared metric diagnostic; not an OR and not a substitute for any theory output.",
    }


def build_prediction(*, project_source: str, transformation_id: str, case_id: str,
                     ptmc_mode: str = "all_twinning", ptmc_base_variant_index: int | None = None,
                     ptmc_dilatational_factor: float = 1.0, ball_james_fraction_samples: int = 101,
                     include_ct_closing_gap: bool = True,
                     include_ct_supercompatibility: bool = True,
                     repository_provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not case_id.strip() or not transformation_id.strip(): raise CampaignError("case_id/transformation_id required")
    if ball_james_fraction_samples < 2: raise CampaignError("ball_james_fraction_samples must be >=2")
    source_audit = _source_audit(project_source)
    loaded = load_project(project_source)
    loaded.assert_representation_links()
    project = loaded.project
    project.validate().assert_passed()
    inputs = _scientific_input(project, transformation_id)
    options = {
        "ptmc_mode": PTMCMode(ptmc_mode).value,
        "ptmc_base_variant_index": ptmc_base_variant_index,
        "ptmc_dilatational_factor": float(ptmc_dilatational_factor),
        "ball_james_fraction_samples": int(ball_james_fraction_samples),
        "include_ct_closing_gap": bool(include_ct_closing_gap),
        "include_ct_supercompatibility": bool(include_ct_supercompatibility),
        "natural_orientation_supplied": False,
        "experimental_rows_supplied": False,
    }
    report = TheoryComparisonAdapter(project, transformation_id).compare(
        ptmc_mode=PTMCMode(ptmc_mode), ptmc_request=None,
        ptmc_base_variant_index=ptmc_base_variant_index,
        ptmc_dilatational_factor=ptmc_dilatational_factor,
        natural_orientation=None, include_ct_closing_gap=include_ct_closing_gap,
        include_ct_supercompatibility=include_ct_supercompatibility,
        ball_james_fraction_samples=ball_james_fraction_samples, experiments=(),
    )
    unified = report.to_dict()
    rows = unified["rows"]
    by_theory: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for row in rows:
        by_theory[row["theory"]] = by_theory.get(row["theory"], 0) + 1
        by_kind[row["prediction_kind"]] = by_kind.get(row["prediction_kind"], 0) + 1
    prediction = {
        "cayron": {
            "topology": _topology(project, transformation_id),
            "austenite_martensite": unified["raw_reports"]["ct"],
            "rows": [r for r in rows if r["theory"] == TheoryKind.CAYRON_CT.value],
        },
        "ball_james_cofactor": {
            "raw_report": unified["raw_reports"]["ball_james"],
            "rows": [r for r in rows if r["theory"] == TheoryKind.BALL_JAMES.value],
        },
        "ptmc": {
            "raw_report": unified["raw_reports"]["ptmc"],
            "rows": [r for r in rows if r["theory"] == TheoryKind.PTMC.value],
        },
        "shared_metric_diagnostics": _metric_diagnostics(project, transformation_id),
        "all_rows": rows,
        "row_counts": {"by_theory": dict(sorted(by_theory.items())),
                       "by_prediction_kind": dict(sorted(by_kind.items())), "total_rows": len(rows)},
        "warnings": unified["warnings"], "notes": unified["notes"],
    }
    basis = {"scientific_input": inputs, "options": options, "prediction": prediction}
    return _jsonable({
        "schema_version": SCHEMA_VERSION, "mode": "blind_prediction", "case_id": case_id.strip(),
        "scientific_scope": "Crystallographic validation conditional on parent/product/correspondence; not thermodynamic phase-stability prediction.",
        "repository": dict(repository_provenance or {}), "source_audit": source_audit,
        "scientific_input": inputs, "options": options,
        "blind_contract": {
            "oracle_argument_available_to_predictor": False,
            "expected_result_keys_allowed_in_project_json": False,
            "stored_orientation_used_as_cayron_natural_or": False,
            "experimental_rows_used_to_fit_theories": False,
            "one_common_project_state_for_all_theories": True,
            "theory_backends_called_independently": True,
            "all_discrete_branches_preserved": True,
            "scalar_winner_score_computed": False,
        },
        "prediction": prediction,
        "seals": {"scientific_input_sha256": _canonical_sha256(inputs),
                  "prediction_sha256": _canonical_sha256(basis)},
    })


def verify_prediction_payload(payload: Mapping[str, Any]) -> str:
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("mode") != "blind_prediction":
        raise CampaignError("Unsupported/not-blind prediction document")
    seals = payload.get("seals")
    if not isinstance(seals, Mapping): raise CampaignError("Missing seals")
    if _canonical_sha256(payload.get("scientific_input")) != seals.get("scientific_input_sha256"):
        raise CampaignError("Scientific-input seal mismatch")
    expected = _canonical_sha256({"scientific_input": payload.get("scientific_input"),
                                  "options": payload.get("options"),
                                  "prediction": payload.get("prediction")})
    if expected != seals.get("prediction_sha256"): raise CampaignError("Prediction seal mismatch")
    return expected


def _atomic_json(path: Path, payload: Any) -> None:
    path = path.resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise


def _pointer(doc: Any, pointer: str) -> Any:
    if pointer == "": return doc
    if not pointer.startswith("/"): raise CampaignError("JSON pointer must start with /")
    cur = doc
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list): cur = cur[int(token)]
        elif isinstance(cur, Mapping): cur = cur[token]
        else: raise CampaignError(f"Pointer traverses scalar: {pointer}")
    return cur


def _dotted(doc: Any, path: str) -> Any:
    cur = doc
    for token in path.split("."):
        if not isinstance(cur, Mapping) or token not in cur: raise CampaignError(f"Missing row field {path}")
        cur = cur[token]
    return cur


def _projective(actual: Any, expected: Any) -> float:
    a = np.asarray(actual, float).reshape(3); b = np.asarray(expected, float).reshape(3)
    if np.linalg.norm(a) < 1e-15 or np.linalg.norm(b) < 1e-15: raise CampaignError("zero projective vector")
    return float(np.linalg.norm(np.cross(a/np.linalg.norm(a), b/np.linalg.norm(b))))


def _check(pred: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    cid = str(spec.get("id", "")).strip(); op = str(spec.get("op", "")).strip()
    if not cid: raise CampaignError("check id required")
    if op == "exact":
        actual = _pointer(pred, str(spec["path"])); passed = actual == spec.get("expected")
        return {"id": cid, "op": op, "passed": passed, "actual": actual, "expected": spec.get("expected")}
    if op == "numeric":
        actual = float(_pointer(pred, str(spec["path"]))); expected = float(spec["expected"])
        atol = float(spec.get("atol", 0)); rtol = float(spec.get("rtol", 0))
        diff = abs(actual-expected); allowed = atol + rtol*abs(expected)
        return {"id": cid, "op": op, "passed": diff <= allowed, "actual": actual,
                "expected": expected, "absolute_difference": diff, "allowed_difference": allowed}
    if op == "row_exists":
        rows = _pointer(pred, str(spec.get("collection_path", "/prediction/all_rows")))
        where = spec.get("where", {}); numeric = spec.get("numeric", {}); projective = spec.get("projective", {})
        matches = []
        for i, row in enumerate(rows if isinstance(rows, list) else []):
            try:
                if any(_dotted(row, str(k)) != v for k, v in where.items()): continue
                ok = True; details = {}
                for field, s in numeric.items():
                    a=float(_dotted(row, field)); e=float(s["expected"]); d=abs(a-e); t=float(s.get("atol",0))+float(s.get("rtol",0))*abs(e)
                    details[field] = {"actual":a,"expected":e,"difference":d,"allowed":t}; ok &= d<=t
                for field, s in projective.items():
                    r=_projective(_dotted(row, field), s["expected"]); t=float(s.get("tol",1e-8)); details[field]={"projective_residual":r,"allowed":t}; ok &= r<=t
                if ok: matches.append({"index": i, "row_id": row.get("row_id"), "details": details})
            except Exception:
                continue
        return {"id": cid, "op": op, "passed": bool(matches), "match_count": len(matches), "matches": matches}
    raise CampaignError(f"Unsupported oracle op: {op}")


def reveal_and_compare(prediction_path: Path, oracle_path: Path) -> dict[str, Any]:
    pred = json.loads(prediction_path.read_text())
    pred_hash = verify_prediction_payload(pred)
    oracle_bytes = oracle_path.read_bytes()  # intentionally only after seal verification
    oracle = json.loads(oracle_bytes)
    if oracle.get("schema_version") != ORACLE_SCHEMA_VERSION: raise CampaignError("Unsupported oracle schema")
    bound = oracle.get("prediction_sha256")
    if bound not in (None, "") and bound != pred_hash: raise CampaignError("Oracle bound to different prediction")
    checks = oracle.get("checks")
    if not isinstance(checks, list) or not checks: raise CampaignError("Oracle checks required")
    results = [_check(pred, c) for c in checks]
    return {"schema_version":1,"mode":"post_freeze_reveal","case_id":pred.get("case_id"),
            "prediction_sha256":pred_hash,"prediction_file_sha256":_file_sha256(prediction_path),
            "oracle_file_sha256":hashlib.sha256(oracle_bytes).hexdigest(),
            "all_checks_passed":all(x["passed"] for x in results),
            "passed_count":sum(x["passed"] for x in results),"failed_count":sum(not x["passed"] for x in results),
            "checks":results,"methodology":{"oracle_opened_after_prediction_seal_verified":True,
            "predictor_refit_after_reveal":False,"scalar_theory_winner_score_computed":False}}


def run_audit(repo: Path) -> int:
    validate_frozen_core(repo, require_clean=True)
    cmd=[sys.executable,"-m","pytest","-q",*CAYRON_AUDIT_TESTS]
    print("Running frozen Cayron-first audit:\n  "+" ".join(cmd))
    return subprocess.run(cmd,cwd=repo).returncode


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Blind Cayron-first CT/Ball-James/PTMC campaign")
    s=p.add_subparsers(dest="cmd",required=True)
    a=s.add_parser("predict"); a.add_argument("--project",required=True); a.add_argument("--transformation",required=True); a.add_argument("--case-id",required=True); a.add_argument("--output",required=True)
    a.add_argument("--ptmc-mode",choices=("none","all_twinning"),default="all_twinning"); a.add_argument("--ptmc-base-variant-index",type=int); a.add_argument("--ptmc-dilatational-factor",type=float,default=1.0); a.add_argument("--ball-james-fraction-samples",type=int,default=101); a.add_argument("--no-ct-closing-gap",action="store_true"); a.add_argument("--no-ct-supercompatibility",action="store_true")
    v=s.add_parser("verify"); v.add_argument("prediction")
    r=s.add_parser("reveal"); r.add_argument("--prediction",required=True); r.add_argument("--oracle",required=True); r.add_argument("--output",required=True)
    s.add_parser("audit")
    return p


def main(argv: Sequence[str] | None=None) -> int:
    args=parser().parse_args(argv)
    if args.cmd=="audit": return run_audit(_repo_root())
    if args.cmd=="verify":
        path=Path(args.prediction).resolve(); payload=json.loads(path.read_text()); h=verify_prediction_payload(payload)
        print("PREDICTION SEAL: PASS\nprediction_sha256:",h,"\nfile_sha256:",_file_sha256(path)); return 0
    if args.cmd=="reveal":
        result=reveal_and_compare(Path(args.prediction).resolve(),Path(args.oracle).resolve()); _atomic_json(Path(args.output),result)
        print("POST-FREEZE REVEAL:","PASS" if result["all_checks_passed"] else "FAIL",result["passed_count"],"passed",result["failed_count"],"failed")
        return 0 if result["all_checks_passed"] else 1
    repo=_repo_root(); provenance=validate_frozen_core(repo,require_clean=True)
    payload=build_prediction(project_source=args.project,transformation_id=args.transformation,case_id=args.case_id,ptmc_mode=args.ptmc_mode,ptmc_base_variant_index=args.ptmc_base_variant_index,ptmc_dilatational_factor=args.ptmc_dilatational_factor,ball_james_fraction_samples=args.ball_james_fraction_samples,include_ct_closing_gap=not args.no_ct_closing_gap,include_ct_supercompatibility=not args.no_ct_supercompatibility,repository_provenance={**provenance,"python":platform.python_version(),"platform":platform.platform()})
    verify_prediction_payload(payload); out=Path(args.output).resolve(); _atomic_json(out,payload)
    t=payload["prediction"]["cayron"]["topology"]
    print("=== BLIND THEORY PREDICTION FROZEN ===\ncase:",payload["case_id"],"\nprediction:",payload["seals"]["prediction_sha256"],"\nfile sha256:",_file_sha256(out))
    print(f"Cayron topology: |GA|={t['parent_group_order']} |GM|={t['product_group_order']} |HC|={t['correspondence_subgroup_order']} variants={t['variant_count']} operators={t['operator_count']}")
    print("theory rows:",payload["prediction"]["row_counts"]["by_theory"],"\noutput:",out)
    return 0

if __name__ == "__main__": raise SystemExit(main())
