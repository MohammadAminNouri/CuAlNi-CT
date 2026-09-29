from __future__ import annotations

"""Scientific status helpers for research-workspace orchestration."""

from dataclasses import replace
from math import acos, degrees
from typing import Any, Iterable

import numpy as np

from app.phase1_contracts import physical_closing_gap_label

from cualni_cryst.supercompatibility_atlas import AtlasClass, SupercompatibilityAtlasReport


def normalize_atlas_semantics(report: SupercompatibilityAtlasReport) -> SupercompatibilityAtlasReport:
    """Enforce exact-seed gating for CT supercompatibility.

    If exact CT A/M compatibility is absent, exact CT supercompatibility is not
    evaluable.  Cofactor quantities remain available because that theory is
    evaluated independently.
    """

    states = []
    for state in report.states:
        if state.ct_am_exact_compatible is False:
            state = replace(
                state,
                classification=AtlasClass.NOT_EVALUABLE,
                ct_supercompatible=None,
                ct_best_supercompatibility_residual=None,
                ct_supercompatible_branch_count=0,
                matched_relation_count=0,
                pair_agreement_count=0,
                pair_disagreement_count=0,
                pair_results=(),
            )
        states.append(state)
    return replace(
        report,
        states=tuple(states),
        notes=tuple(report.notes)
        + (
            "Exact CT supercompatibility is gated by an exact CT A/M seed; states without one are classified NOT_EVALUABLE rather than NEITHER.",
        ),
    )


def minimum_projective_separation_deg(rows: Iterable[dict[str, Any]]) -> tuple[float | None, tuple[str, str] | None]:
    """Minimum angular separation between different plotted pole series.

    Plane/direction poles are treated projectively: n and -n are equivalent.
    """

    data = list(rows)
    best: float | None = None
    pair: tuple[str, str] | None = None
    for i, left in enumerate(data):
        a = np.asarray(left.get("cartesian_reference"), dtype=float).reshape(3)
        na = float(np.linalg.norm(a))
        if na <= 0:
            continue
        a = a / na
        for right in data[i + 1 :]:
            if left.get("series") == right.get("series"):
                continue
            b = np.asarray(right.get("cartesian_reference"), dtype=float).reshape(3)
            nb = float(np.linalg.norm(b))
            if nb <= 0:
                continue
            b = b / nb
            angle = degrees(acos(float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))))
            if best is None or angle < best:
                best = angle
                pair = (str(left.get("series", "")), str(right.get("series", "")))
    return best, pair


def human_branch_label(row: Any) -> str:
    theory = str(getattr(getattr(row, "theory", None), "value", getattr(row, "theory", "")))
    kind = str(getattr(getattr(row, "prediction_kind", None), "value", getattr(row, "prediction_kind", "")))
    branch = str(getattr(row, "branch_label", "")).replace("_", " ")
    names = {
        "cayron_ct": "Cayron CT",
        "ball_james": "Ball–James",
        "ptmc": "PTMC",
        "experiment": "Experiment",
    }
    prefix = names.get(theory, theory or "Prediction")
    if kind == "ct_am_habit" and getattr(row, "exact", None) is False:
        return f"{prefix} · Approximate CT diagnostic plane — not an exact A/M solution · {branch}"
    if kind == "ct_closing_gap_or":
        return f"{prefix} · {physical_closing_gap_label(row)}"
    return f"{prefix} · {branch}".strip(" ·")
