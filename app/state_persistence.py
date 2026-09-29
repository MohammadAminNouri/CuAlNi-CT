from __future__ import annotations

"""Cross-page persistence for Streamlit widget drafts.

Streamlit may delete widget-owned session keys when a widget disappears on a
different page. The workbench therefore mirrors scientific input controls into
one detached ordinary session-state object. Returning to a workspace restores
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
    "mart_vi",
    "mart_vj",
    "manual_load_system",
    "map_object",
    "map_source_phase",
    "parallel_candidate",
    "recon_observed_phase",
    "v4_twin_system",
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
    "manual_a_",
    "manual_n_",
    "sample_g_",
    "calpad_",
    "ebsd_",
    "research_",
)


# File-upload widgets are owned by Streamlit.
# Their session-state values must never be replayed manually.
#
# In particular:
#   research_ebsd_pipeline_upload
# was causing StreamlitValueAssignmentNotAllowedError because the persistence
# layer restored it before st.file_uploader() was instantiated.
_STREAMLIT_OWNED_EXACT_KEYS = {
    "research_ebsd_pipeline_upload",
    "workbench_v4_project_upload",
    "ebsd_upload",
}


_STREAMLIT_OWNED_SUFFIXES = (
    "_upload",
    "_uploader",
)


# Actions, downloads and upload handles are not persistent scientific inputs.
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
    "research_v7_pole_generate",
    "research_v7_pole_csv",
    "workbench_v6_calculate",
)


def _is_streamlit_owned_key(key: str) -> bool:
    """Return True for widget keys that must not be restored manually."""

    if key in _STREAMLIT_OWNED_EXACT_KEYS:
        return True

    return key.endswith(_STREAMLIT_OWNED_SUFFIXES)


def _is_persistent_key(key: str) -> bool:
    if key == MIRROR_KEY:
        return False

    if _is_streamlit_owned_key(key):
        return False

    if key in _EXACT_KEYS:
        return True

    if any(key.startswith(prefix) for prefix in _EPHEMERAL_PREFIXES):
        return False

    return any(key.startswith(prefix) for prefix in _PREFIXES)


def _safe_copy(value: Any) -> Any:
    """Copy ordinary control values while rejecting opaque runtime objects."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, tuple):
        return tuple(_safe_copy(item) for item in value)

    if isinstance(value, list):
        return [_safe_copy(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _safe_copy(item)
            for key, item in value.items()
        }

    # Streamlit selections can occasionally expose enum-like values.
    enum_value = getattr(value, "value", None)

    if isinstance(enum_value, (str, int, float, bool)):
        return enum_value

    raise TypeError(type(value).__name__)


def restore_persistent_widget_state(
    state: MutableMapping[str, Any],
) -> None:
    """Restore mirrored controls before page widgets are instantiated."""

    mirror = state.get(MIRROR_KEY)

    if not isinstance(mirror, Mapping):
        return

    for key, value in mirror.items():
        name = str(key)

        # CRITICAL:
        # Never restore uploader-owned widget values.
        # This also protects sessions whose mirror was created by an older
        # application build and still contains uploader=None.
        if not _is_persistent_key(name):
            continue

        # Do not overwrite state already supplied in the current session.
        if name in state:
            continue

        try:
            state[name] = deepcopy(value)
        except Exception:
            continue


def snapshot_persistent_widget_state(
    state: MutableMapping[str, Any],
) -> None:
    """Refresh the detached mirror after the page has rendered."""

    previous = state.get(MIRROR_KEY)

    # Rebuild the mirror instead of blindly copying the previous one.
    # This removes stale uploader entries left by older deployments.
    mirror: dict[str, Any] = {}

    if isinstance(previous, Mapping):
        for key, value in previous.items():
            name = str(key)

            if not _is_persistent_key(name):
                continue

            try:
                mirror[name] = deepcopy(value)
            except Exception:
                continue

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


def mirrored_values(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a copy of the detached persistent widget mirror."""

    value = state.get(MIRROR_KEY)

    if not isinstance(value, Mapping):
        return {}

    return deepcopy(dict(value))