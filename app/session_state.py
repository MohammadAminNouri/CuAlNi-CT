from __future__ import annotations

"""State/provenance guards for the Streamlit workstation.

No crystallographic equations live here.  The only purpose of this module is
preventing a result calculated from one UI state from being shown as though it
belonged to a different state.
"""

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
)

OR_DEPENDENT_KEYS: tuple[str, ...] = (
    "correspondence_map_result",
    "orientation_map_result",
    "reconstruction_result",
)


def _canonical(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        # Keep the exact Python binary64 value rather than a display rounding.
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


def put_bound(
    state: MutableMapping[str, Any], key: str, signature: str, value: object
) -> None:
    state[key] = BoundResult(signature=signature, value=value)


def get_bound(state: Mapping[str, Any], key: str, signature: str) -> object | None:
    item = state.get(key)
    if isinstance(item, BoundResult) and item.signature == signature:
        return item.value
    return None


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
    clear_keys(state, TRANSFORMATION_DEPENDENT_KEYS)


def invalidate_or_dependents(state: MutableMapping[str, Any]) -> None:
    clear_keys(state, OR_DEPENDENT_KEYS)


def mark_calculation_success(
    state: MutableMapping[str, Any], draft_signature: str
) -> None:
    state["calculated_draft_signature"] = draft_signature
    state["requires_recalculation"] = False
    state.pop("last_calculation_error", None)
    invalidate_transformation_dependents(state)


def mark_calculation_failure(state: MutableMapping[str, Any], error: object) -> None:
    # A failed attempt invalidates all downstream analysis until a valid
    # transformation is calculated again.
    state["requires_recalculation"] = True
    state["last_calculation_error"] = error
    invalidate_transformation_dependents(state)


def observe_draft(state: MutableMapping[str, Any], draft_signature: str) -> bool:
    """Mark the project stale when Setup differs from the calculated revision.

    Once stale, it stays stale until a successful calculation.  Editing fields
    back to earlier values does not silently revive cached analyses.
    """

    calculated = state.get("calculated_draft_signature")
    if calculated is not None and calculated != draft_signature:
        if not state.get("requires_recalculation", False):
            invalidate_transformation_dependents(state)
        state["requires_recalculation"] = True
    return bool(state.get("requires_recalculation", False))
