from __future__ import annotations

"""Phase-2 presentation/provenance helpers.

These helpers deliberately do not alter scientific solver values.  They only
normalize professor-facing status language, projective notation, display
precision and pole-series bookkeeping.
"""

from dataclasses import replace
import json
import math
import re
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


_STATUS_TRUE = "satisfied"
_STATUS_FALSE = "not satisfied"
_STATUS_NONE = "not evaluable"


def scientific_status(value: bool | None) -> str:
    if value is True:
        return _STATUS_TRUE
    if value is False:
        return _STATUS_FALSE
    return _STATUS_NONE


def sanitize_scientific_text(text: str) -> str:
    """Remove raw boolean implementation wording from professor-facing prose."""

    out = str(text)
    out = re.sub(
        r"CT\s+exact-compatible\s*=\s*True",
        "CT exact A/M compatibility: satisfied",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"CT\s+exact-compatible\s*=\s*False",
        "CT exact A/M compatibility: not satisfied",
        out,
        flags=re.IGNORECASE,
    )
    for cc in ("CC1", "CC2", "CC3"):
        out = re.sub(
            rf"{cc}\s*=\s*True",
            f"{cc}: satisfied",
            out,
            flags=re.IGNORECASE,
        )
        out = re.sub(
            rf"{cc}\s*=\s*False",
            f"{cc}: not satisfied",
            out,
            flags=re.IGNORECASE,
        )
    return out


def sanitize_finding(finding: Any) -> Any:
    """Return a Finding-like dataclass with raw boolean prose humanized."""

    updates: dict[str, Any] = {}
    for name in ("conclusion", "rationale", "how", "physical_meaning", "limitation", "status"):
        value = getattr(finding, name, None)
        if isinstance(value, str):
            updates[name] = sanitize_scientific_text(value)
    try:
        return replace(finding, **updates)
    except Exception:
        return finding


def projective_representative_notation(notation: str, *, kind: str) -> str:
    """Render ± projective equivalence without misusing family braces/brackets."""

    text = str(notation).strip()
    if len(text) >= 2 and text[0] in "{<[(" and text[-1] in "}>])":
        body = text[1:-1].strip()
    else:
        body = text
    if str(kind).lower().startswith("plane"):
        return f"±({body})"
    return f"±[{body}]"


def _compact_float(value: float, *, residual: bool = False) -> str:
    if not math.isfinite(float(value)):
        return "not evaluable"
    value = float(value)
    if residual or (value != 0.0 and (abs(value) < 1.0e-4 or abs(value) >= 1.0e6)):
        return f"{value:.3e}"
    return f"{value:.8g}"


def _compact_nested(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return float(f"{float(value):.8g}")
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Mapping):
        return {str(k): _compact_nested(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_compact_nested(v) for v in list(value)]
    return value


def compact_scalar_table(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Compact professor-facing scalars while preserving raw data elsewhere."""

    safe_rows: list[dict[str, Any]] = []
    for row in rows:
        safe: dict[str, Any] = {}
        for key, value in row.items():
            name = str(key)
            if isinstance(value, (Mapping, list, tuple, np.ndarray)):
                safe[name] = json.dumps(_compact_nested(value), sort_keys=True)
            elif isinstance(value, (float, np.floating)):
                safe[name] = _compact_float(float(value), residual="residual" in name.lower())
            elif value is True:
                safe[name] = "yes"
            elif value is False:
                safe[name] = "no"
            elif value is None:
                safe[name] = "not evaluable"
            else:
                safe[name] = value
        safe_rows.append(safe)
    return pd.DataFrame(safe_rows)


def humanize_scientific_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove implementation tokens from professor-facing dataframes."""

    out = frame.copy()
    if "classification" in out.columns:
        out["classification"] = out["classification"].map(
            lambda v: str(v).replace("_", " ") if v is not None else "not evaluable"
        )
    if "CT supercompatibility residual" in out.columns:
        out["CT supercompatibility residual"] = out["CT supercompatibility residual"].map(
            lambda v: "not evaluable"
            if v is None or (isinstance(v, float) and math.isnan(v)) or str(v) in {"—", "None", "nan"}
            else v
        )
    if "current state" in out.columns:
        out["current state"] = out["current state"].map(
            lambda v: "yes" if v is True else "no" if v is False else v
        )
    return out


def _canonical_projective_vector(value: Any, *, digits: int = 10) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1.0e-15:
        raise ValueError("zero pole cannot define a projective key")
    arr = arr / norm
    nz = np.flatnonzero(np.abs(arr) > 10.0 ** (-digits))
    if len(nz) and arr[int(nz[0])] < 0.0:
        arr = -arr
    arr[np.abs(arr) < 10.0 ** (-digits)] = 0.0
    return tuple(float(x) for x in np.round(arr, digits))  # type: ignore[return-value]


def pole_family_signature(family: Any, *, digits: int = 10) -> tuple[tuple[float, float, float], ...]:
    """Geometry-only signature for a plotted projective pole family."""

    vectors: set[tuple[float, float, float]] = set()
    for point in getattr(family, "points", ()):
        data = point.to_dict() if hasattr(point, "to_dict") else point
        if isinstance(data, Mapping):
            vector = data.get("cartesian_reference")
        else:
            vector = getattr(point, "cartesian_reference", None)
        if vector is None:
            continue
        vectors.add(_canonical_projective_vector(vector, digits=digits))
    return tuple(sorted(vectors))


def _clean_native_branch(text: str) -> str:
    value = str(text).replace("_", " ")
    value = re.sub(r"\s*[·,]?\s*sign branch\s+[A-Za-z0-9+-]+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*[·,]?\s*type\s+[ivx]+\s*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" ·,")
    return value


def physical_pole_label(row: Any) -> str:
    """Compact physical/geometric label; native generator labels remain provenance."""

    theory = str(getattr(getattr(row, "theory", None), "value", getattr(row, "theory", "")))
    kind = str(getattr(getattr(row, "prediction_kind", None), "value", getattr(row, "prediction_kind", "")))
    exact = getattr(row, "exact", None)
    metadata = getattr(row, "metadata", {}) or {}
    names = {
        "cayron_ct": "Cayron CT",
        "ball_james": "Ball–James",
        "ptmc": "PTMC",
        "experiment": "Experiment",
    }
    prefix = names.get(theory, theory or "Prediction")
    if kind == "ct_am_habit" and exact is False:
        return f"{prefix} · approximate A/M diagnostic — not exact"
    if kind == "ct_mm_twin":
        op = metadata.get("operator_index")
        twin = metadata.get("twin_index")
        twin_kind = metadata.get("twin_kind")
        parts = [prefix]
        if op is not None:
            parts.append(f"operator {op}")
        if twin is not None:
            parts.append(f"twin {twin}")
        if twin_kind:
            parts.append(f"Type-{twin_kind}")
        return " · ".join(parts)
    branch = _clean_native_branch(str(getattr(row, "branch_label", "")))
    return f"{prefix} · {branch}".strip(" ·")


def deduplicate_pole_entries(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group native rows that produce exactly the same plotted projective family.

    This is geometric-series deduplication, not a claim that the source theories
    or generator derivations are identical.  All native branch labels are kept
    in provenance.
    """

    groups: dict[tuple[tuple[float, float, float], ...], dict[str, Any]] = {}
    for entry in entries:
        family = entry["family"]
        signature = pole_family_signature(family)
        label = str(entry.get("native_label", ""))
        if signature not in groups:
            groups[signature] = dict(entry)
            groups[signature]["native_labels"] = [label]
        else:
            groups[signature]["native_labels"].append(label)
    return list(groups.values())


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def ct_ptmc_or_availability(unified: Any) -> dict[str, Any]:
    """Describe whether a physical CT↔PTMC OR residual can actually be formed.

    This is a gating audit only. It does not compute or alter any orientation
    relationship; it counts exact native rows already present in the unified
    report and checks whether they carry an explicit physical OR matrix.
    """

    rows = tuple(getattr(unified, "rows", ()) or ()) if unified is not None else ()
    ct_or = []
    ptmc_exact = []
    ptmc_or = []
    for row in rows:
        theory = _enum_value(getattr(row, "theory", ""))
        kind = _enum_value(getattr(row, "prediction_kind", ""))
        exact = getattr(row, "exact", None)
        has_or = getattr(row, "or_parent_from_product", None) is not None
        if theory == "cayron_ct" and kind == "ct_closing_gap_or" and exact is True and has_or:
            ct_or.append(row)
        if theory == "ptmc" and kind == "ptmc_habit" and exact is True:
            ptmc_exact.append(row)
            if has_or:
                ptmc_or.append(row)

    if not ct_or and not ptmc_or:
        reason = "neither_source_has_or"
    elif not ct_or:
        reason = "ct_or_unavailable"
    elif not ptmc_or:
        reason = "ptmc_or_unavailable"
    else:
        reason = "evaluable"
    return {
        "evaluable": reason == "evaluable",
        "reason": reason,
        "ct_exact_or_rows": len(ct_or),
        "ptmc_exact_habit_rows": len(ptmc_exact),
        "ptmc_exact_or_rows": len(ptmc_or),
    }


def atlas_professor_audit_rows(states: Iterable[Any], axes: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Human-readable atlas audit with no raw booleans/None/status tokens."""

    output: list[dict[str, Any]] = []
    axis_names = tuple(str(name) for name in axes)
    for state in states:
        metadata = getattr(state, "metadata", {}) or {}
        classification = _enum_value(getattr(state, "classification", "not_evaluable")).replace("_", " ")
        row: dict[str, Any] = {
            "state": str(getattr(state, "state_id", "")),
            **{name: metadata.get(name) for name in axis_names},
            "classification": classification,
            "CT exact A/M": scientific_status(getattr(state, "ct_am_exact_compatible", None)),
            "CT supercompatibility": scientific_status(getattr(state, "ct_supercompatible", None)),
            "cofactor": scientific_status(getattr(state, "cofactor_compatible", None)),
            "CT supercompatibility residual": (
                "not evaluable"
                if getattr(state, "ct_best_supercompatibility_residual", None) is None
                else getattr(state, "ct_best_supercompatibility_residual")
            ),
            "exact CT seed branches": int(getattr(state, "ct_compatible_branch_count", 0) or 0),
            "CT supercompatibility branches": int(getattr(state, "ct_supercompatible_branch_count", 0) or 0),
            "cofactor-compatible branches": int(getattr(state, "cofactor_compatible_branch_count", 0) or 0),
            "matched physical relations": int(getattr(state, "matched_relation_count", 0) or 0),
            "pair agreements": int(getattr(state, "pair_agreement_count", 0) or 0),
            "pair disagreements": int(getattr(state, "pair_disagreement_count", 0) or 0),
        }
        output.append(row)
    return output
