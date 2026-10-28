from __future__ import annotations

"""One CTF input, two scientific routes: measured EBSD and conditional OR fits.

OR-free maps NEVER assert phase identity, transformation OR, crystallographic
correspondence or a physical twin type. OR-dependent outputs require
independent input consistency and the separately calculated theory report.
"""

import hashlib
import io
import json

import numpy as np
import pandas as pd
import streamlit as st

from .ctf_independent import (
    compare_ctf_model_settings, declared_ctf_phases,
    independent_ctf_cartography, plot_ctf_field,
)
from .ebsd_integration import (
    MAX_CTF_BYTES, ctf_analysis_signature, inspect_ctf_upload,
    parse_explicit_or, run_ctf_observation, summarize_twin_candidates,
)


@st.cache_data(show_spinner=False, max_entries=4)
def _ctf_preview(file_bytes: bytes, filename: str) -> dict:
    return inspect_ctf_upload(file_bytes, filename=filename)


@st.cache_data(show_spinner=False, max_entries=2, ttl=3600)
def _independent_cartography(file_bytes, filename, threshold, nmin):
    return independent_ctf_cartography(
        file_bytes, filename,
        misorientation_threshold_deg=threshold,
        minimum_grain_pixels=nmin,
    )


def _map_download(result) -> bytes:
    payload = io.BytesIO()
    np.savez_compressed(
        payload,
        **{name: np.asarray(values) for name, values in result["maps"].items()},
    )
    return payload.getvalue()


def _show_independent_cartography(result):
    st.markdown("**Measured EBSD — no OR supplied**")
    a, b, c = st.columns(3)
    a.metric("Indexed pixels", f"{result['n_indexed']:,}")
    b.metric("Retained grains", f"{result['n_grains']:,}")
    c.metric("Grain adjacencies", f"{result['n_distinct_grain_adjacencies']:,}")
    st.caption(
        f"Grain threshold {result['threshold_deg']}° · KAM neighbor cutoff "
        f"{result['kam_cutoff_deg']}° · CTF-declared phase symmetries only"
    )
    field_labels = {
        "Phase ID (AZtec indexing)": "phase",
        "Segmented grain ID": "grain_id",
        "Kernel average misorientation, °": "kam_deg",
        "Mean angular deviation (MAD, AZtec)": "MAD",
        "Band contrast (BC)": "BC",
        "Band slope (BS)": "BS",
    }
    selected = st.selectbox(
        "EBSD map", list(field_labels),
        key="tf_ctf_map_channel",
    )
    st.plotly_chart(
        plot_ctf_field(result["maps"], field_labels[selected]),
        width="stretch",
    )
    with st.expander("Measured grains and boundaries", expanded=False):
        st.dataframe(
            pd.DataFrame(result["grains"]).drop(
                columns=["mean_crystal_to_sample_rotation"],
                errors="ignore",
            ),
            hide_index=True, width="stretch",
        )
        st.caption(
            "These are orientation-segmented grains, not Cayron CT variants. "
            "A 2-D trace does not determine a unique 3-D twin plane."
        )
    st.download_button(
        "Export all pixel maps (.npz)",
        data=_map_download(result),
        file_name="measured_ebsd_maps.npz",
        mime="application/octet-stream",
        key="tf_ctf_independent_npz",
    )
    summary = {k: v for k, v in result.items() if k != "maps"}
    st.download_button(
        "Export measured EBSD audit (.json)",
        data=json.dumps(summary, default=str, indent=2),
        file_name="measured_ebsd_independent.json",
        mime="application/json", key="tf_ctf_independent_json",
    )
    st.warning(
        "NOT CERTIFIED: CTF phase labels and reconstructed orientation grains "
        "are not independent 2H/18R diffraction indexing, a parent OR, "
        "or a Type I/II/Compound twin assignment."
    )


def render_ctf_panel(report, project_payload, transformation_id, signature):
    """Compact CTF controls in Step 01: upload -> measure -> optional compare."""
    upload = st.file_uploader(
        "Oxford / AZtec indexed CTF map", type=["ctf"],
        key="tf_ctf_v1_file", max_upload_size=35,
        help="Indexed CTF (2-D Bunge ZXZ Euler degrees), maximum 35 MB.",
    )
    if upload is None:
        return
    content = upload.getvalue()
    if len(content) > MAX_CTF_BYTES:
        st.error("CTF exceeds the 35 MB analysis limit.")
        return
    try:
        preview = _ctf_preview(content, upload.name)
        phases_declared = declared_ctf_phases(content)
    except (ValueError, RuntimeError, UnicodeError, OSError) as exc:
        st.error(f"CTF not accepted: {exc}")
        return

    indexed_counts = preview["phase_counts"]
    st.caption(
        f"{preview['indexed_pixels']:,}/{preview['total_pixels']:,} indexed pixels · "
        + ", ".join(f"phase {k}: {v:,}" for k,v in indexed_counts.items())
    )
    if phases_declared:
        st.dataframe(pd.DataFrame([
            {
                "CTF ID": p["phase_id"], "Header label": p["name"],
                "Space group (declared)": p["space_group"],
                "Point group": p["declared_point_group"],
                "Indexed pixels": indexed_counts.get(p["phase_id"], 0),
            } for p in phases_declared
        ]), hide_index=True, width="stretch")
    else:
        st.warning("The CTF has no complete phase definitions. No symmetry-based independent analysis is possible.")
    st.caption("AZtec phase names are indexing assumptions, not independent diffraction results.")

    a, b, c = st.columns(3)
    phase_a = int(a.number_input(
        "Parent phase ID", min_value=1, value=1, step=1,
        key="tf_ctf_v1_parent",
        help="The parent may be absent among indexed pixels.",
    ))
    phase_m = int(b.number_input(
        "Martensite phase ID", min_value=1, value=2, step=1,
        key="tf_ctf_v1_product",
    ))
    nmin = int(c.number_input(
        "Minimum pixels per grain", min_value=1, max_value=1000,
        value=3, step=1, key="tf_ctf_v1_minpix",
    ))
    threshold = float(st.number_input(
        "Orientation segmentation threshold (degrees)", min_value=0.1,
        max_value=30.0, value=3.0, step=0.5, key="tf_ctf_independent_threshold",
    ))
    invalid = False
    if phase_a == phase_m:
        st.error("Parent and martensite must use different phase IDs.")
        invalid = True
    if phase_m not in indexed_counts:
        st.error(f"No indexed martensite pixels for phase {phase_m}.")
        invalid = True
    extras = sorted(set(indexed_counts).difference({phase_a,phase_m}))
    if extras:
        st.error(
            f"Indexed phases {extras} are outside the selected two-phase "
            "model. Split or explicitly remap the phase data first."
        )
        invalid = True

    # This branch computes directly from the supplied CTF and its
    # declared point groups; no OR, CIF or completed CT solver is required.
    independent_key = hashlib.sha256(
        content + json.dumps(
            {"threshold":threshold,"nmin":nmin},sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    if st.button(
        "Analyze indexed EBSD independently",
        key="tf_ctf_v1_measure", disabled=not bool(phases_declared),
    ):
        try:
            with st.spinner("Segmenting measured orientations and computing KAM…"):
                result = _independent_cartography(
                    content, upload.name, threshold, nmin,
                )
            st.session_state["tf_ctf_independent"] = (independent_key,result)
        except (ValueError, RuntimeError, ArithmeticError, OSError) as exc:
            st.error(f"EBSD measurement failed: {exc}")
        except Exception as exc:
            st.error(f"EBSD measurement failed ({type(exc).__name__}): {exc}")
    saved_independent = st.session_state.get("tf_ctf_independent")
    if saved_independent and saved_independent[0] == independent_key:
        _show_independent_cartography(saved_independent[1])

    st.divider()
    st.markdown("**Compare with crystallographic theory (optional)**")
    has_or = st.checkbox(
        "I have an independently established parent-from-martensite OR",
        value=False, key="tf_ctf_v1_has_or",
        help="A CTF or CIF alone does not determine a unique transformation OR.",
    )
    if not has_or:
        st.caption("No OR is needed for measured EBSD maps and grain statistics.")
        return

    if not phases_declared:
        st.error("CTF lacks phase definitions needed to check the theory model.")
        return
    try:
        model_audit = compare_ctf_model_settings(
            phases_declared, project_payload, transformation_id,
            phase_a, phase_m,
        )
    except (ValueError, RuntimeError, ArithmeticError) as exc:
        st.error(f"Crystal-model check failed: {exc}")
        return
    if not model_audit["pass"]:
        st.error("CTF cells/point groups do not match the selected model's conventional settings.")
        with st.expander("Resolve phase and basis mismatches",expanded=False):
            st.dataframe(pd.DataFrame(model_audit["comparisons"]),
                         hide_index=True, width="stretch")
            st.caption(model_audit["scope"])

    matrix_txt = st.text_area(
        "Parent ← martensite OR (3×3 proper rotation)",
        value="", placeholder="r11 r12 r13\nr21 r22 r23\nr31 r32 r33",
        key="tf_ctf_v1_or",
        help=(
            "Matrix in corresponding orthonormal crystal Cartesian frames: "
            "x_parent=R x_martensite. Not the lattice correspondence C."
        ),
    )
    selected = ctf_analysis_signature(
        content, signature, matrix_txt, phase_a, phase_m, nmin,
        segmentation_threshold_deg=threshold,
    )
    confirm = st.checkbox(
        "I verified the phase IDs, indexing, crystal axes and OR frames.",
        value=False, key="tf_ctf_v1_confirm_" + selected[:20],
    )
    if report is None:
        st.caption("Calculate crystallographic modes to enable theory comparison.")
    ready = bool(
        report is not None and not invalid and model_audit["pass"]
        and matrix_txt.strip() and confirm
    )
    if st.button(
        "Compare EBSD with theory",
        key="tf_ctf_v1_analyze", disabled=not ready,
    ):
        try:
            R = parse_explicit_or(matrix_txt)
            with st.spinner("Reconstructing candidate parent domains and boundary operators…"):
                result = run_ctf_observation(
                    content,project_payload,transformation_id,
                    parent_phase_id=phase_a,product_phase_id=phase_m,
                    verified_or=R,filename=upload.name,
                    minimum_grain_points=nmin,
                    segmentation_threshold_deg=threshold,
                )
            st.session_state["tf_ctf_v1_result"]=(selected,result)
        except (ValueError, RuntimeError, ArithmeticError, OSError) as exc:
            st.error(f"Experimental operator comparison failed: {exc}")
        except Exception as exc:
            st.error(f"Experimental operator comparison failed ({type(exc).__name__}): {exc}")

    saved = st.session_state.get("tf_ctf_v1_result")
    if (not saved or saved[0]!=selected or not ready):
        return
    result = saved[1]
    audit = summarize_twin_candidates(report,result)
    c1,c2,c3 = st.columns(3)
    c1.metric("Observed boundaries",audit["observed_boundaries"])
    c2.metric("Operator fits",audit["accepted_operator_fits"])
    c3.metric("Ambiguous",audit["ambiguous_operator_fits"])
    st.warning(
        "NOT CERTIFIED: operator fits alone do not identify physical Type I, "
        "Type II or Compound twins, nor a 3D habit plane."
    )
    with st.expander("EBSD comparison details",expanded=False):
        st.json(audit)
        st.json(result.summary)
        if result.boundaries:
            st.dataframe(pd.DataFrame(result.boundaries[:100]),
                         hide_index=True, width="stretch")
    st.download_button(
        "Download complete EBSD pipeline (.zip)",
        data=result.evidence_archive,
        file_name="ctf_ebsd_pipeline_all_outputs.zip",
        mime="application/zip",key="tf_ctf_v1_full_zip",
    )
    st.download_button(
        "Export EBSD/theory comparison (.json)",
        data=json.dumps({
            "source_sha256":result.sha256,"model_signature":signature,
            "CTF_parent_phase_id":phase_a,
            "CTF_martensite_phase_id":phase_m,
            "grain_segmentation_threshold_deg": threshold,
            "verified_R_parent_from_product":parse_explicit_or(matrix_txt).tolist(),
            "model_consistency":model_audit,
            "audit":audit,"summary":result.summary,
            "boundaries":result.boundaries,
            "assignments":result.grain_assignments,
        },default=str,indent=2),
        file_name="ctf_orientation_evidence.json",
        mime="application/json",key="tf_ctf_v1_export",
    )
