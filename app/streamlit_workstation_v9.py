from __future__ import annotations

"""Final Phase-2C workstation: deterministic research archive over cumulative 2B."""

import json
from typing import Any, Mapping

import streamlit as st

import app.research_workspaces as rw
import app.streamlit_workstation as legacy
import app.streamlit_workstation_v4 as v4
import app.streamlit_workstation_v8 as v8
from app.phase2_archive import (
    ResearchArchiveError,
    build_research_archive,
    exact_project_payload,
    load_research_archive,
    serialize_research_archive,
)


_original_audit = getattr(v4._render_audit_workspace, "_phase2c_original_audit", v4._render_audit_workspace)


def _bound_value(key: str) -> Any:
    value = st.session_state.get(key)
    return getattr(value, "value", value)


def _to_json_object(text: str) -> Mapping[str, Any] | None:
    try:
        value = json.loads(text)
    except Exception:
        return None
    return value if isinstance(value, Mapping) else None


def _render_final_archive() -> None:
    try:
        _, response = legacy._require_project()
    except Exception:
        return

    project = _to_json_object(response.project_json())  # type: ignore[attr-defined]
    result = _to_json_object(response.to_json())  # type: ignore[attr-defined]
    if project is None:
        st.error("Final archive suppressed: the active calculated ProjectState could not be serialized as a JSON object.")
        return

    unified = rw._current_unified_report()
    equivalence = rw._current_equivalence_report()
    ebsd = _bound_value("research_ebsd_result")
    try:
        archive = build_research_archive(
            project_payload=project,
            calculated_draft_signature=st.session_state.get("calculated_draft_signature"),
            transformation_result=result,
            workbench_analyses=v4._collect_workbench_exports(),
            unified_theory=unified,
            physical_equivalence=equivalence,
            ebsd_result=ebsd,
        )
        archive_text = serialize_research_archive(archive)
    except ResearchArchiveError as exc:
        st.error(f"Final archive suppressed because strict serialization failed: {exc}")
        return

    st.markdown("### Final reproducibility archive")
    st.caption(
        "One integrity-checked archive binds the exact calculated project payload, calculated-draft signature, "
        "transformation result, workbench analyses and any currently available theory/EBSD outputs. "
        "Exact rational strings and exact/approximate/provenance fields are preserved. Import never activates a calculated result automatically."
    )
    st.code(archive["integrity"]["canonical_content_sha256"], language=None)
    st.download_button(
        "Complete reproducibility archive",
        data=archive_text.encode("utf-8"),
        file_name="cualni_ct_research_archive.json",
        mime="application/json",
        use_container_width=True,
        key="phase2c_archive_download",
    )

    uploaded = st.file_uploader(
        "Validate an existing research archive",
        type=["json"],
        key="phase2c_archive_validate_upload",
        help="Validation checks schema and SHA-256 only. It does not load or calculate the project.",
    )
    if uploaded is not None:
        try:
            content = load_research_archive(uploaded.getvalue())
            recovered = exact_project_payload(uploaded.getvalue())
        except ResearchArchiveError as exc:
            st.error(f"Archive rejected: {exc}")
        else:
            st.success(
                "Archive integrity verified. Its project is still inactive; no calculation or scientific result was restored implicitly."
            )
            st.download_button(
                "Recovered exact project JSON",
                data=json.dumps(recovered, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8"),
                file_name="cualni_ct_recovered_project.json",
                mime="application/json",
                use_container_width=True,
                key="phase2c_recovered_project_download",
            )
            signature = content.get("calculated_draft_signature")
            st.caption(f"Archived calculated-draft signature: {signature if signature else 'not supplied'}")


def _render_audit_phase2c() -> None:
    _original_audit()
    _render_final_archive()


_render_audit_phase2c._phase2c_original_audit = _original_audit  # type: ignore[attr-defined]
v4._render_audit_workspace = _render_audit_phase2c


def main() -> None:
    v8.main()


if __name__ == "__main__":
    main()
