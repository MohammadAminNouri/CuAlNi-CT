from __future__ import annotations

"""Optional Oxford/AZtec CTF input for the Twin Family Explorer.

The native CTF reader can inspect an EBSD map without a transformation OR.
Operator matching requires a proper parent-from-martensite OR supplied from
independent crystallographic evidence and a current theory calculation.
Neither import route establishes 2H/18R indexing or physical twin identity.
"""

import hashlib
import json

import streamlit as st

from .ebsd_integration import (
    MAX_CTF_BYTES, ctf_analysis_signature, inspect_ctf_upload,
    parse_explicit_or, run_ctf_observation, summarize_twin_candidates,
)


@st.cache_data(show_spinner=False, max_entries=4)
def _ctf_preview(file_bytes: bytes, filename: str) -> dict:
    # The cache is keyed by complete bytes and filename, never only by name.
    return inspect_ctf_upload(file_bytes, filename=filename)


def render_ctf_panel(report, project_payload, transformation_id, signature):
    """One compact Step-01 input, with preview independent of model results."""
    upload = st.file_uploader(
        "Oxford / AZtec indexed CTF map", type=["ctf"], key="tf_ctf_v1_file",
        help="Native CTF (Bunge ZXZ, 2-D degrees). Analysis limit: 35 MB.",
    )
    if upload is None:
        return

    content = upload.getvalue()
    preview = None
    if len(content) > MAX_CTF_BYTES:
        st.error("CTF exceeds the 35 MB analysis limit.")
        return
    try:
        preview = _ctf_preview(content, upload.name)
    except (ValueError, RuntimeError, UnicodeError, OSError) as exc:
        st.error(f"Invalid CTF map: {exc}")
        return

    counts = preview["phase_counts"]
    display = ", ".join(f"{k}: {v:,} pixels" for k, v in counts.items()) or "none"
    st.caption(
        f"Indexed {preview['indexed_pixels']:,}/{preview['total_pixels']:,} pixels"
        f" · CTF phases {display}"
    )
    st.caption("CTF phase IDs are labels, not independent proof of 2H or 18R.")

    left, right, third = st.columns([1, 1, 1])
    phase_a = int(left.number_input(
        "Parent phase ID", min_value=1, value=1, step=1,
        key="tf_ctf_v1_parent",
        help="ID used by AZtec. The parent may be absent from a martensite-only map.",
    ))
    phase_m = int(right.number_input(
        "Martensite phase ID", min_value=1, value=2, step=1,
        key="tf_ctf_v1_product",
        help="Must be present among the indexed CTF pixels.",
    ))
    nmin = int(third.number_input(
        "Min. pixels per grain", min_value=1, max_value=1000, value=3,
        step=1, key="tf_ctf_v1_minpix",
    ))
    invalid = False
    if phase_a == phase_m:
        st.error("Parent and martensite must have different phase IDs.")
        invalid = True
    if phase_m not in counts:
        st.error(f"Martensite phase {phase_m} has no indexed pixels in this CTF.")
        invalid = True
    extras = sorted(set(counts).difference({phase_a, phase_m}))
    if extras:
        st.error(
            f"This two-phase analysis cannot include additional indexed "
            f"phase IDs {extras}. Use a map containing only these two phases."
        )
        invalid = True

    st.markdown("**Orientation relationship**")
    has_or = st.checkbox(
        "I have an independently established parent-from-martensite OR",
        value=False, key="tf_ctf_v1_has_or",
        help="CIF cells and indexed CTF Euler angles alone do not establish a unique parent↔martensite OR.",
    )
    if not has_or:
        st.caption(
            "CTF inspection is available now. Supply a verified OR to compare "
            "grain-boundary operators with the crystallographic model."
        )
        return

    matrix_txt = st.text_area(
        "OR rotation matrix (parent ← martensite)",
        value="",
        placeholder="r11 r12 r13\nr21 r22 r23\nr31 r32 r33",
        key="tf_ctf_v1_or",
        help=(
            "Orthogonal, determinant +1; columns/rows in corresponding "
            "orthonormal Cartesian crystal frames. x_parent = R x_martensite. "
            "This is NOT the lattice correspondence C."
        ),
    )
    selected = ctf_analysis_signature(
        content, signature, matrix_txt, phase_a, phase_m, nmin,
    )
    confirm = st.checkbox(
        "I checked the phase IDs, CTF indexing, crystal-axis settings and OR frames.",
        value=False,
        key="tf_ctf_v1_confirm_" + selected[:20],
    )
    if report is None:
        st.caption("Calculate crystallographic modes below, then compare EBSD.")
    elif invalid:
        st.caption("Correct the CTF phase IDs before comparison.")

    can_compare = bool(
        report is not None and not invalid and matrix_txt.strip() and confirm
    )
    if st.button(
        "Compare EBSD with theory",
        type="secondary", key="tf_ctf_v1_analyze", disabled=not can_compare,
    ):
        try:
            R = parse_explicit_or(matrix_txt)
            with st.spinner("Analyzing indexed grains and boundary operators…"):
                result = run_ctf_observation(
                    content, project_payload, transformation_id,
                    filename=upload.name, parent_phase_id=phase_a,
                    product_phase_id=phase_m, verified_or=R,
                    minimum_grain_points=nmin,
                )
            st.session_state["tf_ctf_v1_result"] = (selected, result)
        except (ValueError, RuntimeError, ArithmeticError, OSError) as exc:
            st.error(f"CTF analysis failed: {exc}")
        except Exception as exc:
            # Failure must not be reinterpreted as a zero-match result.
            st.error(f"CTF analysis failed ({type(exc).__name__}): {exc}")

    saved = st.session_state.get("tf_ctf_v1_result")
    if saved is None or report is None or not confirm or invalid:
        return
    if saved[0] != selected:
        st.caption("CTF/OR/model input changed. Repeat the comparison.")
        return

    result = saved[1]
    audit = summarize_twin_candidates(report, result)
    a, b, c = st.columns(3)
    a.metric("Boundaries", audit["observed_boundaries"])
    b.metric("Operator matches", audit["accepted_operator_fits"])
    c.metric("Ambiguous", audit["ambiguous_operator_fits"])
    st.warning(
        "NOT CERTIFIED: operator matches do not establish Type I/II/Compound "
        "twins or a 3-D habit plane. Confirm 2H/18R indexing independently."
    )
    with st.expander("EBSD analysis details", expanded=False):
        st.json(audit)
        st.json(result.summary)
        if result.boundaries:
            st.dataframe(result.boundaries[:50], hide_index=True, width="stretch")
    st.download_button(
        "Export EBSD evidence (.json)",
        data=json.dumps({
            "file_name": upload.name,
            "source_sha256": result.sha256,
            "model_signature": signature,
            "parent_phase_id": phase_a,
            "martensite_phase_id": phase_m,
            "minimum_grain_points": nmin,
            "or_direction": "parent_from_product",
            "or_matrix": parse_explicit_or(matrix_txt).tolist(),
            "audit": audit,
            "summary": result.summary,
            "boundaries": result.boundaries,
            "assignments": result.grain_assignments,
        }, default=str, indent=2),
        file_name="ctf_orientation_evidence.json",
        mime="application/json", key="tf_ctf_v1_export",
    )
