from __future__ import annotations

"""Standalone Twin App extension; never alters either scientific engine."""
import hashlib
import json
import streamlit as st
from .ebsd_integration import parse_explicit_or, run_ctf_observation, summarize_twin_candidates


def render_ctf_panel(report, project_payload, transformation_id, signature):
    st.subheader("03 / Experimental EBSD (CTF)")
    st.caption("This is an analysis of already indexed CTF orientations. It does not reindex diffraction patterns or certify twin planes from phase labels.")
    with st.expander("Import an experimental CTF map", expanded=False):
        upload = st.file_uploader("Oxford / AZtec CTF map", type=["ctf"], key="tf_ctf_v1_file")
        left, right = st.columns(2)
        phase_a = int(left.number_input("Parent phase ID in AZtec", min_value=1, value=1, step=1, key="tf_ctf_v1_parent"))
        phase_m = int(right.number_input("Martensite phase ID in AZtec", min_value=1, value=2, step=1, key="tf_ctf_v1_product"))
        nmin = int(st.number_input("Minimum indexed pixels per grain", min_value=1, max_value=1000, value=3, step=1, key="tf_ctf_v1_minpix"))
        matrix_txt = st.text_area("Independently established parent ← martensite OR (3 × 3 proper rotation)", value="", placeholder="r11 r12 r13\nr21 r22 r23\nr31 r32 r33", key="tf_ctf_v1_or", help="No default identity OR and no inference of crystallographic correspondence from CIF or CTF")
        confirm = st.checkbox("I have checked AZtec indexing, phase IDs, crystal settings, reference frames and this OR against independent evidence.", value=False, key="tf_ctf_v1_confirm")
        if st.button("Analyze experimental CTF", type="primary", key="tf_ctf_v1_analyze"):
            if upload is None or not matrix_txt.strip() or not confirm:
                st.error("Supply a CTF, a verified OR, and explicit scientific input confirmation.")
            else:
                try:
                    R = parse_explicit_or(matrix_txt)
                    content = upload.getvalue()
                    key = hashlib.sha256(content + signature.encode() + matrix_txt.encode() + f"{phase_a}:{phase_m}:{nmin}".encode()).hexdigest()
                    with st.spinner("Parsing CTF, segmenting grains and testing orientation-operator hypotheses…"):
                        result = run_ctf_observation(content, project_payload, transformation_id, filename=upload.name, parent_phase_id=phase_a, product_phase_id=phase_m, verified_or=R, minimum_grain_points=nmin)
                    st.session_state["tf_ctf_v1_result"] = (key, result)
                    st.success("CTF orientation analysis finished. Physical twin certification remains separate.")
                except Exception as exc:
                    st.error(f"CTF analysis not accepted: {type(exc).__name__}: {exc}")
        saved = st.session_state.get("tf_ctf_v1_result")
        if saved is not None and upload is not None:
            digest = hashlib.sha256(upload.getvalue() + signature.encode() + matrix_txt.encode() + f"{phase_a}:{phase_m}:{nmin}".encode()).hexdigest()
            if saved[0] != digest:
                st.info("Input changed. Re-run CTF analysis; stale output is hidden.")
            else:
                result = saved[1]
                audit = summarize_twin_candidates(report, result)
                c1, c2, c3 = st.columns(3)
                c1.metric("Experimental boundaries", audit["observed_boundaries"])
                c2.metric("Operator fits", audit["accepted_operator_fits"])
                c3.metric("Ambiguous fits", audit["ambiguous_operator_fits"])
                st.warning("NOT CERTIFIED: a matched OR/boundary operator is not a measured Type I/II/Compound twin or A/M habit plane. 2H/18R indexing must be independently checked.")
                st.json(audit)
                st.download_button("Export CTF analysis evidence", data=json.dumps({"source_sha256":result.sha256,"audit":audit,"summary":result.summary,"boundaries":result.boundaries,"assignments":result.grain_assignments},default=str,indent=2), file_name="ctf_orientation_evidence.json", mime="application/json", key="tf_ctf_v1_export")
