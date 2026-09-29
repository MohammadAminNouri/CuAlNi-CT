from __future__ import annotations

"""State/provenance guards for the Streamlit workstation.

This module contains no crystallographic equations.  It preserves the user's
editable draft, keeps the last successful calculated state independently, and
binds every derived result to the exact signature that produced it.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, MutableMapping


TRANSFORMATION_DEPENDENT_KEYS: tuple[str, ...] = (
    "orientation_result",
    "parallelism_solve_result",
    "correspondence_map_result",
    "orientation_map_result",
    "reconstruction_result",
    "martensite_pair_result",
    "manual_ptmc_result",
    "calpad_cell_result",
    "calpad_normal_result",
    "calpad_low_result",
    "ebsd_result",
    "research_atlas_report",
    "research_pole_report",
    "research_invariant_report",
    "research_double_shear_report",
    "research_ebsd_result",
)

OR_DEPENDENT_KEYS: tuple[str, ...] = (
    "correspondence_map_result",
    "orientation_map_result",
    "reconstruction_result",
    "research_pole_report",
)


def _canonical(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        # Keep the exact Python binary64 value rather than display rounding.
        return {"__float__": repr(value)}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def fingerprint(*parts: object) -> str:
    payload = json.dumps(
        _canonical(parts),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class BoundResult:
    """A value bound to the exact input signature that produced it."""

    signature: str
    value: object


@dataclass(frozen=True)
class CalculatedSnapshot:
    """The last successful scientific state, kept separately from the draft."""

    signature: str
    project_payload: object
    response: object


def put_bound(
    state: MutableMapping[str, Any], key: str, signature: str, value: object
) -> None:
    state[key] = BoundResult(signature=signature, value=value)


def get_bound(state: Mapping[str, Any], key: str, signature: str) -> object | None:
    item = state.get(key)
    if isinstance(item, BoundResult) and item.signature == signature:
        return item.value
    return None


def get_bound_envelope(state: Mapping[str, Any], key: str) -> BoundResult | None:
    """Return a bound result even when stale, so the UI can label it as stale."""

    item = state.get(key)
    return item if isinstance(item, BoundResult) else None


def bound_payload(state: Mapping[str, Any], key: str) -> dict[str, object] | None:
    """Return a serializable result envelope for export."""

    item = state.get(key)
    if not isinstance(item, BoundResult):
        return None
    return {"input_signature": item.signature, "result": item.value}


def clear_keys(state: MutableMapping[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        state.pop(key, None)


def invalidate_transformation_dependents(state: MutableMapping[str, Any]) -> None:
    """Clear derived results only after a *new successful* base calculation."""

    clear_keys(state, TRANSFORMATION_DEPENDENT_KEYS)


def invalidate_or_dependents(state: MutableMapping[str, Any]) -> None:
    """Clear OR-derived results only when a new OR is successfully accepted."""

    clear_keys(state, OR_DEPENDENT_KEYS)


def record_draft(
    state: MutableMapping[str, Any], snapshot: Mapping[str, Any], draft_signature: str
) -> None:
    """Persist the editable draft independently of the calculated state."""

    state["draft_snapshot"] = deepcopy(dict(snapshot))
    state["draft_signature"] = str(draft_signature)


def mark_calculation_success(
    state: MutableMapping[str, Any], draft_signature: str
) -> None:
    """Freeze the successful state without destroying the editable draft."""

    # Derived results from the previous calculated base are no longer current
    # once a new base state succeeds.  Clearing happens here -- never merely
    # because the user edited a widget or because a calculation failed.
    invalidate_transformation_dependents(state)

    state["calculated_draft_signature"] = str(draft_signature)
    state["requires_recalculation"] = False
    state.pop("last_calculation_error", None)
    state["calculated_snapshot"] = CalculatedSnapshot(
        signature=str(draft_signature),
        project_payload=deepcopy(state.get("current_project_payload")),
        response=state.get("current_response"),
    )


def mark_calculation_failure(state: MutableMapping[str, Any], error: object) -> None:
    """Record failure non-destructively.

    The last successful calculated state and every result bound to it are kept.
    They are simply not presented as current while the editable draft is stale.
    """

    state["requires_recalculation"] = True
    state["last_calculation_error"] = error


def observe_draft(state: MutableMapping[str, Any], draft_signature: str) -> bool:
    """Mark the draft stale when it differs from the last successful revision.

    Crucially, this function never deletes the last successful calculation or
    downstream results.  A stale result can therefore be preserved for audit
    while the UI refuses to present it as current.
    """

    state["draft_signature"] = str(draft_signature)
    calculated = state.get("calculated_draft_signature")
    if calculated is not None and calculated != draft_signature:
        state["requires_recalculation"] = True
    # Deliberately sticky: once a calculated state has been made stale by an
    # edit, merely editing widgets back to the old numerical values must not
    # silently revive cached analyses.  Only mark_calculation_success() clears
    # the stale flag after the user explicitly recalculates.
    elif calculated == draft_signature and "requires_recalculation" not in state:
        state["requires_recalculation"] = False
    return bool(state.get("requires_recalculation", False))


def calculated_snapshot(state: Mapping[str, Any]) -> CalculatedSnapshot | None:
    item = state.get("calculated_snapshot")
    return item if isinstance(item, CalculatedSnapshot) else None
