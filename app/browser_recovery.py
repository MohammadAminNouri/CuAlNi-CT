from __future__ import annotations

"""Browser-URL recovery for editable Streamlit scientific controls.

Calculated scientific outputs are intentionally NOT serialized here. Only the
already-whitelisted detached widget draft from ``state_persistence`` is stored.
After a process restart/redeployment the draft can therefore be reconstructed,
but a new explicit crystallographic calculation is still required.
"""

import base64
import hashlib
import json
import zlib
from typing import Any, Mapping, MutableMapping

from app.state_persistence import (
    MIRROR_KEY,
    mirrored_values,
    restore_persistent_widget_state,
)

RECOVERY_PARAM = "cualni_draft_v1"
_RECOVERY_VERSION = 1
_MAX_TOKEN_CHARS = 24_000
_MAX_RAW_BYTES = 160_000


def _canonical_bytes(values: Mapping[str, Any]) -> bytes:
    payload = {"version": _RECOVERY_VERSION, "state": dict(values)}
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def encode_recovery_token(values: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(values)
    if len(raw) > _MAX_RAW_BYTES:
        raise ValueError("Persistent draft is too large for browser-URL recovery")
    digest = hashlib.sha256(raw).hexdigest()[:20]
    packed = zlib.compress(raw, level=9)
    encoded = base64.urlsafe_b64encode(packed).decode("ascii").rstrip("=")
    token = f"{digest}.{encoded}"
    if len(token) > _MAX_TOKEN_CHARS:
        raise ValueError("Persistent draft exceeds the browser-URL recovery budget")
    return token


def decode_recovery_token(token: str) -> dict[str, Any]:
    text = str(token).strip()
    if not text or len(text) > _MAX_TOKEN_CHARS or "." not in text:
        raise ValueError("Invalid browser recovery token")
    digest, encoded = text.split(".", 1)
    padding = "=" * (-len(encoded) % 4)
    try:
        packed = base64.urlsafe_b64decode((encoded + padding).encode("ascii"))
        raw = zlib.decompress(packed)
    except Exception as exc:
        raise ValueError("Corrupt browser recovery token") from exc
    if len(raw) > _MAX_RAW_BYTES:
        raise ValueError("Browser recovery payload is too large")
    if hashlib.sha256(raw).hexdigest()[:20] != digest:
        raise ValueError("Browser recovery checksum mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError("Browser recovery payload is not valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("version") != _RECOVERY_VERSION:
        raise ValueError("Unsupported browser recovery version")
    values = payload.get("state")
    if not isinstance(values, dict):
        raise ValueError("Browser recovery state is missing")
    return {str(key): value for key, value in values.items()}


def restore_browser_draft(
    state: MutableMapping[str, Any],
    query_params: Mapping[str, Any],
) -> bool:
    """Restore only when the live session has no detached draft of its own."""

    if isinstance(state.get(MIRROR_KEY), Mapping) and state.get(MIRROR_KEY):
        return False
    # A URL token is only a cold-start recovery mechanism. Never let it
    # overwrite any live draft or calculated state in an active session.
    live_markers = (
        "draft_snapshot",
        "draft_signature",
        "current_response",
        "current_project_payload",
        "calculated_draft_signature",
        "calculated_snapshot",
    )
    if any(key in state for key in live_markers):
        return False
    token = query_params.get(RECOVERY_PARAM)
    if isinstance(token, (list, tuple)):
        token = token[0] if token else None
    if token is None:
        return False
    try:
        values = decode_recovery_token(str(token))
    except ValueError:
        state["_browser_recovery_invalid"] = True
        return False

    # Reuse the existing whitelist/filtering logic instead of duplicating it.
    state[MIRROR_KEY] = values
    restore_persistent_widget_state(state)
    state["_browser_draft_recovered"] = True
    state["requires_recalculation"] = True
    # Never revive a calculated state from a browser token.
    state.pop("current_response", None)
    state.pop("current_project_payload", None)
    state.pop("calculated_draft_signature", None)
    return True


def persist_browser_draft(
    state: Mapping[str, Any],
    query_params: MutableMapping[str, Any],
) -> bool:
    """Write the current detached input draft if it changed."""

    values = mirrored_values(state)
    if not values:
        return False
    try:
        token = encode_recovery_token(values)
    except ValueError:
        return False
    current = query_params.get(RECOVERY_PARAM)
    if isinstance(current, (list, tuple)):
        current = current[0] if current else None
    if str(current or "") == token:
        return False
    query_params[RECOVERY_PARAM] = token
    return True
