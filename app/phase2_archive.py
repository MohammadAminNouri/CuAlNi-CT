from __future__ import annotations

"""Deterministic, integrity-checked research archive for the Phase-2 workstation.

This module is intentionally serialization-only.  Loading an archive never
activates a calculated result, never mutates a Streamlit session and never
recomputes crystallography.  Exact string-valued scientific inputs (for example
rational correspondence entries such as ``"1/3"``) are preserved verbatim.
"""

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import hmac
import json
from typing import Any, Mapping

ARCHIVE_SCHEMA = "cualni-ct-research-archive/v1"
ARCHIVE_VERSION = 1


class ResearchArchiveError(ValueError):
    """Raised when an archive is malformed or fails its integrity contract."""


def _json_ready(value: Any) -> Any:
    """Convert scientific/runtime objects to strict JSON without changing strings."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ResearchArchiveError("Research archive cannot contain NaN or infinity.")
        return value
    if isinstance(value, Enum):
        return _json_ready(value.value)
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]

    # NumPy arrays/scalars and similar scientific containers.
    if hasattr(value, "tolist"):
        try:
            return _json_ready(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _json_ready(value.item())
        except Exception:
            pass

    # pandas.DataFrame / Series: records preserve row structure more usefully
    # than the default column-oriented DataFrame.to_dict().
    module = type(value).__module__.split(".", 1)[0]
    if module == "pandas" and hasattr(value, "to_dict"):
        try:
            return _json_ready(value.to_dict(orient="records"))
        except TypeError:
            return _json_ready(value.to_dict())

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_ready(value.to_dict())

    raise ResearchArchiveError(
        f"Unsupported archive value type {type(value).__name__}; serialize it explicitly."
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_sha256(content: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(content).encode("utf-8")).hexdigest()


def build_research_archive(
    *,
    project_payload: Mapping[str, Any],
    calculated_draft_signature: str | None,
    transformation_result: Any = None,
    workbench_analyses: Any = None,
    unified_theory: Any = None,
    physical_equivalence: Any = None,
    ebsd_result: Any = None,
) -> dict[str, Any]:
    """Build one deterministic archive around the exact calculated ProjectState.

    No current-time field is included in the signed content so exporting the
    same scientific state twice produces byte-identical canonical content.
    """

    if not isinstance(project_payload, Mapping):
        raise ResearchArchiveError("project_payload must be a mapping.")
    signature = None if calculated_draft_signature is None else str(calculated_draft_signature)
    content = _json_ready(
        {
            "project_payload": deepcopy(dict(project_payload)),
            "calculated_draft_signature": signature,
            "transformation_result": transformation_result,
            "workbench_analyses": workbench_analyses or {},
            "research_outputs": {
                "unified_theory": unified_theory,
                "physical_equivalence": physical_equivalence,
                "ebsd": ebsd_result,
            },
            "scientific_contract": {
                "archive_restores_calculated_state_automatically": False,
                "exact_string_inputs_preserved": True,
                "machine_exact_approximate_status_preserved": True,
                "provenance_preserved": True,
            },
        }
    )
    digest = content_sha256(content)
    return {
        "schema": ARCHIVE_SCHEMA,
        "version": ARCHIVE_VERSION,
        "content": content,
        "integrity": {
            "algorithm": "sha256",
            "canonical_content_sha256": digest,
        },
    }


def serialize_research_archive(archive: Mapping[str, Any]) -> str:
    """Serialize an already-built archive deterministically and strictly."""

    # Validation before serialization catches accidentally edited envelopes.
    validated = load_research_archive(archive)
    envelope = {
        "schema": ARCHIVE_SCHEMA,
        "version": ARCHIVE_VERSION,
        "content": validated,
        "integrity": {
            "algorithm": "sha256",
            "canonical_content_sha256": content_sha256(validated),
        },
    }
    return json.dumps(
        envelope,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )


def load_research_archive(source: str | bytes | Mapping[str, Any]) -> dict[str, Any]:
    """Verify and return archive content without activating it."""

    if isinstance(source, bytes):
        try:
            source = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ResearchArchiveError("Research archive is not valid UTF-8.") from exc
    if isinstance(source, str):
        try:
            envelope = json.loads(source)
        except json.JSONDecodeError as exc:
            raise ResearchArchiveError(f"Research archive is not valid JSON: {exc}") from exc
    elif isinstance(source, Mapping):
        envelope = deepcopy(dict(source))
    else:
        raise ResearchArchiveError("Archive source must be JSON text, bytes or a mapping.")

    if not isinstance(envelope, Mapping):
        raise ResearchArchiveError("Research archive top level must be an object.")
    if envelope.get("schema") != ARCHIVE_SCHEMA:
        raise ResearchArchiveError(
            f"Unsupported research archive schema {envelope.get('schema')!r}."
        )
    if envelope.get("version") != ARCHIVE_VERSION:
        raise ResearchArchiveError(
            f"Unsupported research archive version {envelope.get('version')!r}."
        )
    content = envelope.get("content")
    integrity = envelope.get("integrity")
    if not isinstance(content, Mapping) or not isinstance(integrity, Mapping):
        raise ResearchArchiveError("Research archive is missing content or integrity metadata.")
    if integrity.get("algorithm") != "sha256":
        raise ResearchArchiveError("Unsupported research archive integrity algorithm.")
    expected = integrity.get("canonical_content_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ResearchArchiveError("Research archive SHA-256 field is malformed.")
    actual = content_sha256(content)
    if not hmac.compare_digest(expected.lower(), actual.lower()):
        raise ResearchArchiveError(
            "Research archive integrity check failed; the signed scientific content was modified."
        )

    project = content.get("project_payload")
    if not isinstance(project, Mapping):
        raise ResearchArchiveError("Research archive does not contain a project_payload object.")
    research = content.get("research_outputs")
    if not isinstance(research, Mapping):
        raise ResearchArchiveError("Research archive research_outputs field is malformed.")
    return deepcopy(dict(content))


def exact_project_payload(source: str | bytes | Mapping[str, Any]) -> dict[str, Any]:
    """Recover the exact archived project JSON, still without calculating it."""

    content = load_research_archive(source)
    return deepcopy(dict(content["project_payload"]))
