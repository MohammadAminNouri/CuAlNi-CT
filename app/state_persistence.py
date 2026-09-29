from __future__ import annotations

"""Cross-page persistence for Streamlit widget drafts.

Streamlit may delete widget-owned session keys when a widget disappears on a
different page.  The workbench therefore mirrors scientific input controls into
one detached ordinary session-state object.  Returning to a workspace restores
those values before any widget is instantiated.

This module stores UI input only; calculated scientific outputs remain governed
by app.session_state signatures.
"""

from copy import deepcopy
from typing import Any, Mapping, MutableMapping

MIRROR_KEY = "_persistent_scientific_widget_draft_v1"

_EXACT_KEYS = {
    "project_title",
    "project_id",
    "length_unit",
    "transformation_id",
    "transformation_label",
    "workbench_v4_workspace",
}

_PREFIXES = (
    "parent_",
    "product_",
    "C_",
    "or_",
    "euler_",
    "parallelism_",
    "reconstruction_",
    "martensite_",
    "manual_ptmc_",
    "calpad_",
    "ebsd_",
    "research_",
)

# These are actions or file handles, not durable inputs.  Replaying them can
# trigger invalid Streamlit widget state or repeat an action unintentionally.
_EPHEMERAL_PREFIXES = (
    "research_run",
    "research_v4_run",
    "research_download",
    "research_v4_download",
    "workbench_v4_open",
    "workbench_v4_calculate",
    "workbench_v5_calculate",
    "workbench_v4_project_upload",
    "research_pole_generate",
    "research_v5_pole_generate",
    "research_v5_atlas_run",
    "research_v5_pole_csv",
    "research_v6_atlas_run",
    "research_v6_pole_generate",
    "research_v6_pole_csv",
    "workbench_v6_calculate",
)


def _is_persistent_key(key: str) -> bool:
    if key == MIRROR_KEY:
        return False
    if key in _EXACT_KEYS:
        return True
    if any(key.startswith(prefix) for prefix in _EPHEMERAL_PREFIXES):
        return False
    return any(key.startswith(prefix) for prefix in _PREFIXES)


def _safe_copy(value: Any) -> Any:
    """Copy ordinary control values while rejecting opaque uploaded/runtime objects."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return tuple(_safe_copy(item) for item in value)
    if isinstance(value, list):
        return [_safe_copy(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _safe_copy(v) for k, v in value.items()}
    # Streamlit selections can occasionally expose enum-like values.
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    raise TypeError(type(value).__name__)


def restore_persistent_widget_state(state: MutableMapping[str, Any]) -> None:
    """Restore mirrored controls before page widgets are instantiated."""

    mirror = state.get(MIRROR_KEY)
    if not isinstance(mirror, Mapping):
        return
    for key, value in mirror.items():
        if not _is_persistent_key(str(key)):
            continue
        # Do not overwrite a key already supplied by the current page/session.
        if key in state:
            continue
        try:
            state[str(key)] = deepcopy(value)
        except Exception:
            continue


def snapshot_persistent_widget_state(state: MutableMapping[str, Any]) -> None:
    """Refresh the detached mirror after the page has rendered."""

    previous = state.get(MIRROR_KEY)
    mirror: dict[str, Any] = (
        deepcopy(dict(previous)) if isinstance(previous, Mapping) else {}
    )
    for key in list(state.keys()):
        name = str(key)
        if not _is_persistent_key(name):
            continue
        try:
            mirror[name] = _safe_copy(state[key])
        except Exception:
            # Runtime-only objects are deliberately not mirrored.
            continue
    state[MIRROR_KEY] = mirror


def mirrored_values(state: Mapping[str, Any]) -> dict[str, Any]:
    """Small test/audit helper."""

    value = state.get(MIRROR_KEY)
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}
