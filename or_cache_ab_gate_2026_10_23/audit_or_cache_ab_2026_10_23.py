#!/usr/bin/env python3
"""Read-only A/B gate for installed OR-frame cache; no theory source edits.

Loads the exact backed-up original method into a separate matcher instance while
using the installed method in another. Both share the same project and actual
orientation kernel, but use isolated per-matcher state. Uses deterministic
synthetic proper rotations, NOT measured/published Cu-Al-Ni orientations.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import time
import types
from pathlib import Path

ROOT = Path.cwd().resolve()
SOURCE = ROOT / "src/cualni_cryst/theory_equivalence.py"
BACKUP = ROOT / "audit_findings/or_frame_cache_patch_v1/theory_equivalence.before.py"
OUTPUT = ROOT / "audit_findings/or_cache_ab_2026_10_23.json"
EXPECTED_ORIGINAL_SHA = "3b6f433789097cef98fddebbc4caded59a50184c096cc4284de1c93f6a7ea317"
EXPECTED_CACHED_SHA = "44ef938c8d34d623861cefd2b18779162d45a9468f2ab880aaf917dbf5a3c1ca"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_method(text: str, namespace: dict):
    tree = ast.parse(text)
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PhysicalBranchMatcher"), None)
    if cls is None:
        raise RuntimeError("Original matcher class not found; refuse guessed comparison")
    method = next((n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == "_or_in_symmetric_metric_frame"), None)
    if not isinstance(method, ast.FunctionDef):
        raise RuntimeError("Original OR method unavailable")
    standalone = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(standalone)
    scope = dict(namespace)
    exec(compile(standalone, str(BACKUP), "exec"), scope)
    return scope[method.name]


def r_z(angle: float):
    import math
    c, s = math.cos(angle), math.sin(angle)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def timed(fn):
    start = time.perf_counter_ns()
    result = fn()
    return result, (time.perf_counter_ns() - start) / 1e9


def run():
    try:
        if not SOURCE.is_file() or not BACKUP.is_file():
            raise RuntimeError("Missing installed source or preserved original backup")
        old_hash, current_hash = sha(BACKUP), sha(SOURCE)
        if old_hash != EXPECTED_ORIGINAL_SHA or current_hash != EXPECTED_CACHED_SHA:
            raise RuntimeError("SOURCE_HASH_MISMATCH: original/cached source differs from the audited pair. Do not force.")

        sys.path.insert(0, str(ROOT / "src"))
        import numpy as np
        import cualni_cryst.theory_equivalence as eq
        from cualni_cryst.project_state import james_hane_6m_reference_project
        from cualni_cryst.theory_unified import (ComparisonRow, PredictionKind, TheoryKind, UnifiedTheoryReport)

        old_fn = original_method(BACKUP.read_text(encoding="utf-8"), vars(eq))
        project = james_hane_6m_reference_project()
        transformation_id = "do3_to_6m_reference"
        old = eq.PhysicalBranchMatcher(project, transformation_id)
        new = eq.PhysicalBranchMatcher(project, transformation_id)
        old._or_in_symmetric_metric_frame = types.MethodType(old_fn, old)

        left = [ComparisonRow(row_id=f"L{i}", theory=TheoryKind.CAYRON_CT,
                 prediction_kind=PredictionKind.CT_CLOSING_GAP_OR, branch_label=f"L{i}",
                 or_parent_from_product=r_z(0.009*i + 0.0002*i*i))
                for i in range(6)]
        right = [ComparisonRow(row_id=f"R{j}", theory=TheoryKind.PTMC,
                  prediction_kind=PredictionKind.PTMC_HABIT, branch_label=f"R{j}",
                  exact=True, or_parent_from_product=r_z(0.008*j + 0.0003*j*j),
                  metadata={"true_invariant_plane": True, "dilatational_factor": 1.0})
                 for j in range(7)]
        experimental = ComparisonRow(row_id="experiment_test_or", theory=TheoryKind.EXPERIMENT,
                       prediction_kind=PredictionKind.EXPERIMENT, branch_label="synthetic observation",
                       or_parent_from_product=r_z(0.009),
                       residuals={"uncertainty_orientation_deg": 0.1})
        transformation = project.transformation(transformation_id)
        report = UnifiedTheoryReport(
            transformation_id=transformation_id,
            parent_phase_id=transformation.parent_phase_id,
            product_phase_id=transformation.product_phase_id,
            rows=tuple([experimental, *left, *right]),
            ct_report=None, ball_james_report=None, ptmc_report=None,
            warnings=(), notes=(),
        )

        # Full physical residual + assignment comparisons, using identical rows.
        def results(m):
            matches, unmatched = m.match_pair(left, right, family=eq.MatchFamily.ORIENTATION,
                                              scales=eq.AssignmentScales())
            return {"matches": [a.to_dict() for a in matches],
                    "unmatched": [a.to_dict() for a in unmatched]}

        original, t_original = timed(lambda: results(old))
        cached_cold, t_cached_cold = timed(lambda: results(new))
        cached_warm, t_cached_warm = timed(lambda: results(new))
        if original != cached_cold or original != cached_warm:
            raise AssertionError("MATCH_OUTPUT_DIFFERS: assignment and/or residual values changed")

        # Comparison of experiment/prediction orientation residual paths.
        old_exp = [x.to_dict() for x in old.experiment_residuals(report)]
        new_exp = [x.to_dict() for x in new.experiment_residuals(report)]
        if old_exp != new_exp:
            raise AssertionError("EXPERIMENT_RESIDUALS_DIFFER")

        # Source-specific near-degenerate OR inputs must never alias in the cache.
        e0 = right[0]
        from dataclasses import replace
        e1 = replace(e0, or_parent_from_product=r_z(1e-10))
        v0 = new._or_in_symmetric_metric_frame(e0)
        v1 = new._or_in_symmetric_metric_frame(e1)
        if np.array_equal(v0, v1):
            raise AssertionError("DISTINCT_OR_ALIASING")
        for entry in (e0, e1):
            np.testing.assert_array_equal(old._or_in_symmetric_metric_frame(entry),
                                          new._or_in_symmetric_metric_frame(entry))

        report_json = {
            "status": "PASS_CONDITIONAL",
            "source_sha256": current_hash,
            "backed_up_source_sha256": old_hash,
            "case": "deterministic synthetic OR matrices on actual James-Hane 6M project",
            "comparison_count": len(left)*len(right),
            "matched_branch_assignments": len(original["matches"]),
            "experimental_comparison_records": len(old_exp),
            "bit_exact_assignment_and_residual_equality": True,
            "bit_exact_experiment_residual_equality": True,
            "no_near_orientation_aliasing": True,
            "seconds": {
                "original_first": round(t_original, 6),
                "cached_first": round(t_cached_cold, 6),
                "cached_repeat": round(t_cached_warm, 6),
            },
            "notes": [
                "This is a read-only A/B integration check, not a published-input theory equivalence audit.",
                "Synthetic proper rotations do not certify numerical science for measured material inputs.",
                "Timing is indicative only; a speedup is not required for numerical PASS.",
                "The 6M full comparison, EBSD Powell optimizer and the complete pytest suite remain unverified after caching.",
            ],
        }
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(report_json, indent=2) + "\n", encoding="utf-8")
        print("PASS_CONDITIONAL: original versus cached assignment/physical residuals EXACTLY match")
        print(f"MATCHING: {len(left)*len(right)} pair evaluations, {len(original['matches'])} matches; experimental records={len(old_exp)}")
        print(f"TIME seconds: original={t_original:.3f}, cached cold={t_cached_cold:.3f}, cached warm={t_cached_warm:.3f}")
        print("EVIDENCE:", OUTPUT.relative_to(ROOT))
    except Exception as exc:
        error = {"status": "REVIEW_REQUIRED", "error_type": type(exc).__name__, "error": str(exc)}
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(error, indent=2) + "\n", encoding="utf-8")
        print("REVIEW_REQUIRED:", type(exc).__name__, str(exc), file=sys.stderr)
        print("EVIDENCE:", OUTPUT.relative_to(ROOT), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    run()
