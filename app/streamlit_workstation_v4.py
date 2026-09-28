from __future__ import annotations

"""Professor-facing Streamlit workstation.

The scientific backends are unchanged. This file reorganizes the workstation so
that every section leads with the physical question and conclusion, while raw
branch/matrix detail remains available as an audit layer.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
import streamlit as st

import app.streamlit_workstation as legacy
from app.errors import ApplicationError, input_error
from app.scientific_interpretation import (
    ebsd_audit_finding,
    martensite_pair_finding,
    orientation_finding,
    topology_finding,
    transformation_findings,
)
from app.scientific_types import Evidence, Finding
from app.scientific_ui import (
    audit_expander,
    compact_metrics,
    custom_sidebar_navigation,
    format_dataframe_scientific,
    install_styles,
    page_header,
    render_finding,
    sci,
    section_header,
    state_strip,
)
from app.session_state import (
    fingerprint,
    get_bound,
    invalidate_or_dependents,
    mark_calculation_failure,
    mark_calculation_success,
    observe_draft,
    put_bound,
)
from app.ui_components import PhaseDefaults, correspondence_editor, matrix_editor, matrix_frame, smart_phase_editor, vector_editor
from app.workbench import (
    calpad_low_index_table,
    calpad_normal_conversion,
    calpad_phase_cell,
    manual_ptmc_cofactor_analysis,
    map_correspondence_object,
    map_orientation_object,
    martensite_variant_pair_analysis,
    orientation_from_euler,
    orientation_from_matrix,
    orientation_from_parallelisms,
    orientation_from_polar_correspondence,
    reconstruct_sample_orientations,
)
from cualni_cryst.ebsd_io import load_ang, load_ctf
from cualni_cryst.ebsd_map import AngleUnit, audit_map
from cualni_cryst.orientation import EulerConvention
from cualni_cryst.representation import CartesianConvention


def _state_snapshot(*, project_id: str, title: str, length_unit: str, parent: Any, product: Any, transformation_id: str, transformation_label: str, correspondence: Any) -> dict[str, Any]:
    return legacy._draft_snapshot(
        project_id=project_id,
        title=title,
        length_unit=length_unit,
        parent=parent,
        product=product,
        transformation_id=transformation_id,
        transformation_label=transformation_label,
        correspondence=correspondence,
    )


def _project_status() -> None:
    payload = st.session_state.get("current_project_payload")
    if not isinstance(payload, dict):
        return
    parent, product, _, _ = legacy._phase_names(payload)
    if st.session_state.get("requires_recalculation", False):
        state_strip(f"{payload.get('title', 'Project')} · {parent} → {product} · draft changed — recalculate before analysis")
    else:
        state_strip(f"{payload.get('title', 'Project')} · {parent} → {product} · calculated state active")


def _sidebar() -> tuple[str, str, str]:
    with st.sidebar:
        custom_sidebar_navigation(current="workbench")
        st.markdown("### Project")
        title = st.text_input("Title", value="Untitled transformation", key="project_title")
        length_unit = st.selectbox("Length unit", ["angstrom", "nm", "pm"], index=0, key="length_unit")
        with st.expander("Project metadata", expanded=False):
            project_id = st.text_input("Project ID", value="workbench_project", key="project_id")

        st.markdown("### Workspace")
        workspace = st.radio(
            "Workspace",
            [
                "State definition",
                "Compatibility",
                "Orientation & reconstruction",
                "Twins & PTMC",
                "Crystal geometry",
                "EBSD audit",
                "Numerical audit & export",
            ],
            label_visibility="collapsed",
            key="workbench_v4_workspace",
        )

        st.divider()
        st.markdown("### Open project")
        uploaded = st.file_uploader("Project JSON", type=["json"], label_visibility="collapsed", key="workbench_v4_project_upload")
        if st.button("Open and calculate", disabled=uploaded is None, use_container_width=True, key="workbench_v4_open"):
            try:
                payload = json.loads(uploaded.getvalue().decode("utf-8"))  # type: ignore[union-attr]
                if not isinstance(payload, dict):
                    raise ValueError("Top-level JSON must be an object.")
                response = legacy.calculate_payload(payload, source=uploaded.name)  # type: ignore[union-attr]
                st.session_state["pending_loaded_response"] = response
                st.session_state["pending_loaded_payload"] = response.project_payload
                st.session_state.pop("app_error", None)
                st.rerun()
            except ApplicationError as exc:
                st.session_state["app_error"] = exc
            except Exception as exc:
                st.session_state["app_error"] = input_error(f"Uploaded project could not be opened: {exc}")

        if st.session_state.get("current_response") is not None:
            if st.session_state.get("requires_recalculation", False):
                st.warning("Draft changed — analyses are blocked")
            else:
                st.success("Calculated state active")
    return title, length_unit, project_id


def _render_setup(title: str, length_unit: str, project_id: str) -> None:
    section_header(
        "State definition",
        "What parent lattice, product lattice and lattice correspondence define the transformation?",
        answer_hint="Nothing is inferred from alloy names. The state is defined explicitly by the two cells, point groups and C.",
    )
    point_groups = legacy.point_group_options()
    left, right = st.columns(2, gap="large")
    with left:
        parent = smart_phase_editor(
            prefix="parent",
            heading="Parent phase (A)",
            defaults=PhaseDefaults("phase_A", "Parent phase", "m-3m", 5.8, 5.8, 5.8, 90, 90, 90, "conventional cubic"),
            point_groups=point_groups,
            length_unit=length_unit,
        )
    with right:
        product = smart_phase_editor(
            prefix="product",
            heading="Product / daughter phase (M)",
            defaults=PhaseDefaults("phase_M", "Product phase", "2/m", 4.4, 5.3, 13.8, 90, 100, 90, "conventional unique-b"),
            point_groups=point_groups,
            length_unit=length_unit,
        )

    with st.expander("Lattice correspondence C and transformation metadata", expanded=True):
        c1, c2 = st.columns([1.1, 1.0], gap="large")
        with c1:
            correspondence = correspondence_editor()
        with c2:
            st.markdown("**Meaning**")
            st.write("C maps parent lattice directions to product lattice directions. It is not a physical orientation matrix R.")
            transformation_id = st.text_input("Transformation ID", value="A_to_M", key="transformation_id")
            transformation_label = st.text_input("Display name", value="Parent → Product", key="transformation_label")

    snapshot = _state_snapshot(
        project_id=project_id,
        title=title,
        length_unit=length_unit,
        parent=parent,
        product=product,
        transformation_id=transformation_id,
        transformation_label=transformation_label,
        correspondence=correspondence,
    )
    signature = fingerprint("setup", snapshot)
    if st.session_state.pop("pending_accept_loaded_draft", False):
        st.session_state["calculated_draft_signature"] = signature
        st.session_state["requires_recalculation"] = False
    stale = observe_draft(st.session_state, signature)
    if stale and st.session_state.get("current_response") is not None:
        st.warning("The draft differs from the calculated state. Downstream analysis is intentionally blocked until this exact draft is recalculated.")

    if st.button("Calculate this crystallographic state", type="primary", use_container_width=True, key="workbench_v4_calculate"):
        try:
            request = legacy.CalculationRequest(
                project_id=project_id,
                title=title,
                parent=parent,
                product=product,
                transformation=legacy.TransformationInput.from_rows(
                    transformation_id,
                    parent.phase_id,
                    product.phase_id,
                    correspondence,
                    label=transformation_label,
                ),
                notes="Created with the CuAlNi-CT workbench.",
            )
            response = legacy.calculate_request(request)
            st.session_state["current_response"] = response
            st.session_state["current_project_payload"] = response.project_payload
            st.session_state.pop("app_error", None)
            mark_calculation_success(st.session_state, signature)
            # A clean rerun prevents stale and success messages from coexisting.
            st.rerun()
        except ApplicationError as exc:
            mark_calculation_failure(st.session_state, exc)
            legacy.render_application_error(exc)

    response = st.session_state.get("current_response")
    if response is not None and not st.session_state.get("requires_recalculation", False):
        summary = response.result["summary"]  # type: ignore[attr-defined]
        compact_metrics(
            [
                ("λ₂", legacy._fmt(summary["lambda2"], 9)),
                ("|λ₂−1|", sci(summary["lambda2_residual"], 3)),
                ("Stretch variants", summary["variant_count"]),
                ("Operators", summary["operator_count"]),
            ],
            columns=4,
        )
        st.caption("The calculated state is frozen to the exact draft signature shown in the project state; changing any phase/correspondence input invalidates downstream analyses.")


def _render_compatibility() -> None:
    section_header(
        "Compatibility",
        "Can one martensite variant meet austenite exactly, and do CT and Ball–James independently give the same answer?",
    )
    try:
        _, response = legacy._require_project()
    except ApplicationError as exc:
        legacy.render_application_error(exc)
        return
    result = response.result  # type: ignore[attr-defined]
    ct_finding, bj_finding, agreement = transformation_findings(result)
    render_finding(agreement)

    summary = result["summary"]
    compact_metrics(
        [
            ("λ₁", legacy._fmt(summary["lambda1"], 9)),
            ("λ₂", legacy._fmt(summary["lambda2"], 9)),
            ("λ₃", legacy._fmt(summary["lambda3"], 9)),
            ("|λ₂−1|", sci(summary["lambda2_residual"], 3)),
        ],
        columns=4,
    )

    with st.expander("Independent derivations", expanded=False):
        render_finding(ct_finding)
        render_finding(bj_finding)

    render_finding(topology_finding(result))

    with audit_expander("Full transformation audit", expanded=False):
        legacy._render_transformation(response)


def _solve_orientation(payload: Mapping[str, Any], project_sig: str) -> tuple[Mapping[str, Any] | None, str]:
    parent_name, product_name, _, _ = legacy._phase_names(payload)
    source_mode = st.radio(
        "OR source",
        ["Polar rotation candidate", "Rotation matrix", "Euler ZXZ", "Two crystallographic parallelisms"],
        horizontal=True,
        key="or_source_mode",
    )
    analysis: Mapping[str, Any] | None = None
    active_or_sig = ""

    if source_mode == "Polar rotation candidate":
        st.caption("Uses the polar rotation of the correspondence deformation. This is a calculated candidate, not experimental OR data.")
        sig = fingerprint(project_sig, source_mode)
        if st.button("Analyze polar rotation", type="primary", key="v4_or_polar"):
            try:
                invalidate_or_dependents(st.session_state)
                result = orientation_from_polar_correspondence(payload)
                put_bound(st.session_state, "orientation_result", sig, result)
            except ApplicationError as exc:
                legacy.render_application_error(exc)
        value = get_bound(st.session_state, "orientation_result", sig)
        analysis = value if isinstance(value, Mapping) else None
        active_or_sig = sig

    elif source_mode == "Rotation matrix":
        matrix = matrix_editor("R(A ← M): product Cartesian → parent Cartesian", key_prefix="or_matrix")
        cols = st.columns(2)
        parent_conv = cols[0].selectbox("Parent Cartesian frame", [item.value for item in CartesianConvention], index=1, key="or_parent_frame")
        product_conv = cols[1].selectbox("Product Cartesian frame", [item.value for item in CartesianConvention], index=1, key="or_product_frame")
        repair = st.checkbox("Explicitly project a near-rotation to SO(3)", value=False, key="or_repair", help="Never done silently.")
        sig = fingerprint(project_sig, source_mode, matrix, parent_conv, product_conv, repair)
        if st.button("Analyze OR matrix", type="primary", key="v4_or_matrix"):
            try:
                invalidate_or_dependents(st.session_state)
                result = orientation_from_matrix(payload, matrix, parent_convention=parent_conv, product_convention=product_conv, repair=repair)
                if repair:
                    result = dict(result)
                    repaired = np.asarray(result["state"]["R_reference_from_moving"], dtype=float)
                    entered = np.asarray(matrix, dtype=float)
                    result["origin_note"] = f"User matrix explicitly projected to SO(3); correction residual = {legacy._relative_matrix_residual(entered, repaired):.3e}."
                put_bound(st.session_state, "orientation_result", sig, result)
            except ApplicationError as exc:
                legacy.render_application_error(exc)
        value = get_bound(st.session_state, "orientation_result", sig)
        analysis = value if isinstance(value, Mapping) else None
        active_or_sig = sig

    elif source_mode == "Euler ZXZ":
        cols = st.columns(3)
        phi1 = cols[0].number_input("φ₁ (deg)", value=0.0, key="or_phi1")
        Phi = cols[1].number_input("Φ (deg)", value=0.0, key="or_Phi")
        phi2 = cols[2].number_input("φ₂ (deg)", value=0.0, key="or_phi2")
        cols2 = st.columns(3)
        euler_conv = cols2[0].selectbox("Euler convention", [item.value for item in EulerConvention], key="or_euler_convention")
        parent_conv = cols2[1].selectbox("Parent Cartesian frame", [item.value for item in CartesianConvention], index=1, key="euler_parent_frame")
        product_conv = cols2[2].selectbox("Product Cartesian frame", [item.value for item in CartesianConvention], index=1, key="euler_product_frame")
        sig = fingerprint(project_sig, source_mode, phi1, Phi, phi2, euler_conv, parent_conv, product_conv)
        if st.button("Analyze Euler OR", type="primary", key="v4_or_euler"):
            try:
                invalidate_or_dependents(st.session_state)
                result = orientation_from_euler(payload, phi1, Phi, phi2, euler_convention=euler_conv, parent_convention=parent_conv, product_convention=product_conv)
                put_bound(st.session_state, "orientation_result", sig, result)
            except ApplicationError as exc:
                legacy.render_application_error(exc)
        value = get_bound(st.session_state, "orientation_result", sig)
        analysis = value if isinstance(value, Mapping) else None
        active_or_sig = sig

    else:
        st.caption("Two independent parallelisms are solved with the phase metrics; only proper-rotation candidates satisfying both are retained.")
        c1, c2 = st.columns(2)
        p1 = c1.text_input(f"{parent_name}: first object", value="(1 1 1)", key="or_p1")
        m1 = c2.text_input(f"{product_name}: first object", value="(0 1 1)", key="or_m1")
        p2 = c1.text_input(f"{parent_name}: second object", value="[1 0 -1]", key="or_p2")
        m2 = c2.text_input(f"{product_name}: second object", value="[1 1 -1]", key="or_m2")
        p1n, m1n = legacy._normalize_compact_crystal_text(p1), legacy._normalize_compact_crystal_text(m1)
        p2n, m2n = legacy._normalize_compact_crystal_text(p2), legacy._normalize_compact_crystal_text(m2)
        sig = fingerprint(project_sig, source_mode, p1n, m1n, p2n, m2n)
        if st.button("Solve exact OR from parallelisms", type="primary", key="v4_or_parallel"):
            try:
                invalidate_or_dependents(st.session_state)
                solved = orientation_from_parallelisms(payload, p1n, m1n, p2n, m2n)
                put_bound(st.session_state, "parallelism_solve_result", sig, solved)
            except ApplicationError as exc:
                legacy.render_application_error(exc)
        solved = get_bound(st.session_state, "parallelism_solve_result", sig)
        if isinstance(solved, Mapping):
            candidates = solved.get("candidates", [])
            if isinstance(candidates, list) and candidates:
                selected = st.selectbox("Exact OR candidate", range(len(candidates)), format_func=lambda i: f"Candidate {i + 1}", key="parallel_candidate")
                analysis = candidates[selected]
                active_or_sig = fingerprint(sig, selected)
                st.caption(f"{len(candidates)} exact candidate(s) satisfy both parallelisms.")
    return analysis, active_or_sig


def _render_orientation_workspace() -> None:
    section_header(
        "Orientation & reconstruction",
        "How is the product lattice physically oriented relative to the parent, and what can be reconstructed from that OR?",
        answer_hint="R is a physical Cartesian rotation. C is a lattice correspondence. They are never substituted for one another.",
    )
    try:
        payload, _ = legacy._require_project()
    except ApplicationError as exc:
        legacy.render_application_error(exc)
        return
    project_sig = str(st.session_state.get("calculated_draft_signature", ""))

    with st.expander("Define or derive the orientation relationship", expanded=True):
        analysis, active_or_sig = _solve_orientation(payload, project_sig)

    if analysis is None:
        st.info("No OR result is active for the current inputs. Analyze the selected OR source first.")
        return

    render_finding(orientation_finding(analysis))
    st.caption("Forward mapping: Product → Parent uses R(A←M). Inverse mapping: Parent → Product uses R(M←A)=R(A←M)ᵀ.")

    map_tab, recon_tab = st.tabs(["Map an object", "Reconstruct the other phase"])
    with map_tab:
        object_text_raw = st.text_input("Direction [uvw] or plane (hkl)", value="[1 0 0]", key="map_object")
        object_text = legacy._normalize_compact_crystal_text(object_text_raw)
        source_phase = st.radio("Object belongs to", ["parent", "product"], horizontal=True, key="map_source_phase")
        c_sig = fingerprint(project_sig, "C", object_text, source_phase)
        r_sig = fingerprint(project_sig, active_or_sig, "R", object_text, source_phase)
        left, right = st.columns(2)
        if left.button("Map by lattice correspondence C", use_container_width=True):
            try:
                put_bound(st.session_state, "correspondence_map_result", c_sig, map_correspondence_object(payload, object_text, source_phase=source_phase))
            except ApplicationError as exc:
                legacy.render_application_error(exc)
        if right.button("Map by physical OR R", use_container_width=True):
            try:
                base_R = analysis["state"]["R_reference_from_moving"]
                put_bound(st.session_state, "orientation_map_result", r_sig, map_orientation_object(payload, base_R, object_text, source_phase=source_phase))
            except ApplicationError as exc:
                legacy.render_application_error(exc)
        c_value = get_bound(st.session_state, "correspondence_map_result", c_sig)
        r_value = get_bound(st.session_state, "orientation_map_result", r_sig)
        a, b = st.columns(2)
        if isinstance(c_value, Mapping):
            with a:
                legacy._render_mapping_result("Lattice correspondence result", c_value)
        if isinstance(r_value, Mapping):
            with b:
                legacy._render_mapping_result("Physical OR result", r_value)

    with recon_tab:
        st.caption("Convention: x_sample = g(sample←crystal) · x_crystal. No EBSD-vendor convention is inferred here.")
        observed_phase = st.radio("Observed phase", ["parent", "product"], horizontal=True, key="recon_observed_phase")
        g = matrix_editor("Observed g(sample ← crystal)", key_prefix="sample_g")
        recon_sig = fingerprint(project_sig, active_or_sig, observed_phase, g)
        if st.button("Reconstruct symmetry-distinct candidates", type="primary"):
            try:
                put_bound(st.session_state, "reconstruction_result", recon_sig, reconstruct_sample_orientations(payload, analysis, g, observed_phase=observed_phase))
            except ApplicationError as exc:
                legacy.render_application_error(exc)
        recon = get_bound(st.session_state, "reconstruction_result", recon_sig)
        if isinstance(recon, Mapping):
            candidates = recon.get("candidates", [])
            st.write(f"**{len(candidates)} symmetry-distinct reconstruction candidate(s)**")
            if candidates:
                frame = pd.DataFrame([
                    {
                        "variant": item["variant_index"],
                        "reconstructed phase": item["reconstructed_phase_id"],
                        "φ₁ active": item["euler_zxz_active"]["phi1_deg"],
                        "Φ active": item["euler_zxz_active"]["Phi_deg"],
                        "φ₂ active": item["euler_zxz_active"]["phi2_deg"],
                        "rotation residual": item["rotation_audit"]["maximum_residual"],
                    }
                    for item in candidates
                ])
                st.dataframe(format_dataframe_scientific(frame), hide_index=True, use_container_width=True)

    with audit_expander("Full OR topology / matrix audit", expanded=False):
        legacy._render_orientation(analysis)


def _render_martensite_workspace() -> None:
    section_header(
        "Twins & PTMC",
        "For a selected pair of martensite variants, which physical twin systems exist, do cofactor conditions hold, and can classical PTMC produce an invariant-plane laminate?",
    )
    try:
        payload, response = legacy._require_project()
    except ApplicationError as exc:
        legacy.render_application_error(exc)
        return
    project_sig = str(st.session_state.get("calculated_draft_signature", ""))
    variants = response.result["stretch_variants"]["variants"]  # type: ignore[attr-defined]
    if len(variants) < 2:
        st.info("Fewer than two stretch variants exist; pair-twin analysis is unavailable.")
        return

    cols = st.columns(2)
    vi = cols[0].selectbox("First stretch variant", range(len(variants)), format_func=lambda i: f"U{i + 1}", key="mart_vi")
    vj_options = [i for i in range(len(variants)) if i != vi]
    vj = cols[1].selectbox("Second stretch variant", vj_options, format_func=lambda i: f"U{i + 1}", key="mart_vj")
    pair_sig = fingerprint(project_sig, vi, vj)
    if st.button("Analyze this variant pair", type="primary", key="v4_mart_pair"):
        try:
            put_bound(st.session_state, "martensite_pair_result", pair_sig, martensite_variant_pair_analysis(payload, vi, vj))
        except ApplicationError as exc:
            legacy.render_application_error(exc)
    pair = get_bound(st.session_state, "martensite_pair_result", pair_sig)
    if not isinstance(pair, Mapping):
        st.info("Select a pair and run the analysis.")
        return

    relations = pair.get("relations", [])
    grouped = legacy._twin_groups(relations) if isinstance(relations, list) else []
    render_finding(martensite_pair_finding(pair, unique_systems=len(grouped)))
    if not grouped:
        return

    selected = st.selectbox("Inspect physical twin system", range(len(grouped)), format_func=lambda i: f"Twin system {i + 1}", key="v4_twin_system")
    group = grouped[selected]
    relation = group[0]
    compact_metrics([
        ("Twin shear", legacy._fmt(relation["twin_shear_magnitude"], 8)),
        ("Mallard residual", sci(relation["mallard_residual"], 3)),
        ("PTMC branches", len(relation["ptmc"])),
        ("Full cofactor", "satisfied" if relation["cofactor"]["satisfied"] else "not satisfied"),
    ], columns=4)

    twin_tab, cofactor_tab, ptmc_tab = st.tabs(["Twin geometry", "Cofactor conditions", "Classical PTMC"])
    with twin_tab:
        st.caption("Twin plane/direction are the physical rank-one system for the selected relation; equivalent Mallard generators are symmetry construction routes, not extra physical twins.")
        st.dataframe(pd.DataFrame([
            {"quantity": "twin direction a", "components": relation["twin_a"]},
            {"quantity": "twin plane normal n", "components": relation["twin_n"]},
        ]), hide_index=True, use_container_width=True)
        with st.expander("Equivalent Mallard generators", expanded=False):
            st.dataframe(pd.DataFrame([
                {
                    "parent symmetry index": item["parent_symmetry_index"],
                    "Mallard type": item["mallard_kind"],
                    "twofold axis": item["twofold_axis_parent_symmetric_cartesian"],
                    "Mallard residual": sci(item["mallard_residual"], 3),
                }
                for item in group
            ]), hide_index=True, use_container_width=True)
    with cofactor_tab:
        co = relation["cofactor"]
        render_finding(Finding(
            title="Cofactor conditions for this twin system",
            conclusion="CC1, CC2 and CC3 are all satisfied." if co["satisfied"] else "The full cofactor criterion is not satisfied for this twin system.",
            rationale=f"CC1={co['cc1_satisfied']}, CC2={co['cc2_satisfied']}, CC3={co['cc3_satisfied']}.",
            how="Each condition is evaluated independently by the cofactor backend; the residuals remain separate and are not collapsed into one score.",
            physical_meaning="Full cofactor satisfaction is the stronger compatibility condition for this selected twin system." if co["satisfied"] else "At least one required cofactor condition fails for this selected twin system.",
            limitation="This conclusion applies to the selected twin system, not automatically to every M/M relation.",
            evidence=(
                Evidence("CC1 residual", "0", sci(co["cc1_residual"], 3), "pass" if co["cc1_satisfied"] else "fail"),
                Evidence("CC2 residual", "0", sci(co["cc2_residual"], 3), "pass" if co["cc2_satisfied"] else "fail"),
                Evidence("CC3 margin", "native inequality criterion", sci(co["cc3_margin"], 3), "pass" if co["cc3_satisfied"] else "fail"),
            ),
            tone="good" if co["satisfied"] else "warn",
        ))
    with ptmc_tab:
        if relation["ptmc"]:
            st.success(f"{len(relation['ptmc'])} admissible classical single-shear PTMC branch(es) exist for this twin system.")
            st.dataframe(format_dataframe_scientific(pd.DataFrame(relation["ptmc"])), hide_index=True, use_container_width=True)
        else:
            st.info(relation.get("ptmc_note") or "No admissible classical single-shear PTMC solution exists for this twin system.")

    with st.expander("Advanced: manual lattice-invariant shear a ⊗ n", expanded=False):
        variant_index = st.selectbox("Stretch variant", range(len(variants)), format_func=lambda i: f"U{i + 1}", key="manual_ptmc_variant")
        if grouped:
            load_index = st.selectbox("Load computed twin system", range(len(grouped)), format_func=lambda i: f"Twin system {i + 1}", key="manual_load_system")
            if st.button("Load selected Mallard a, n", key="v4_load_mallard"):
                source = grouped[load_index][0]
                for i, value in enumerate(source["twin_a"]):
                    st.session_state[f"manual_a_{i}"] = float(value)
                for i, value in enumerate(source["twin_n"]):
                    st.session_state[f"manual_n_{i}"] = float(value)
                st.rerun()
        a = vector_editor("a", default=(0.0, 0.0, 0.0), key_prefix="manual_a")
        n = vector_editor("n", default=(0.0, 0.0, 0.0), key_prefix="manual_n")
        ready = float(np.linalg.norm(a)) > 0.0 and float(np.linalg.norm(n)) > 0.0
        manual_sig = fingerprint(project_sig, variant_index, a, n)
        if st.button("Evaluate manual PTMC / cofactor", disabled=not ready, key="v4_manual_ptmc"):
            try:
                put_bound(st.session_state, "manual_ptmc_result", manual_sig, manual_ptmc_cofactor_analysis(payload, variant_index, a, n))
            except ApplicationError as exc:
                legacy.render_application_error(exc)
        manual = get_bound(st.session_state, "manual_ptmc_result", manual_sig)
        if isinstance(manual, Mapping):
            legacy._render_manual_ptmc(manual)


def _render_geometry_workspace() -> None:
    section_header(
        "Crystal geometry",
        "How do direct directions, reciprocal planes and physical normals relate under the actual non-cubic metric?",
        answer_hint="Direct and reciprocal spaces remain distinct; Euclidean cubic shortcuts are not used for a non-cubic phase.",
    )
    try:
        payload, _ = legacy._require_project()
    except ApplicationError as exc:
        legacy.render_application_error(exc)
        return
    project_sig = str(st.session_state.get("calculated_draft_signature", ""))
    parent_name, product_name, parent_id, product_id = legacy._phase_names(payload)
    phase = st.selectbox("Phase", [parent_id, product_id], format_func=lambda x: parent_name if x == parent_id else product_name, key="calpad_phase")
    cell_tab, normal_tab, low_tab = st.tabs(["Cell & reciprocal cell", "Plane ↔ physical normal", "Low-index ranking"])

    with cell_tab:
        st.caption("Question: what metric tensors define lengths and angles in this phase?")
        sig = fingerprint(project_sig, phase, "cell")
        if st.button("Inspect metric", key="v4_cell"):
            try:
                put_bound(st.session_state, "calpad_cell_result", sig, calpad_phase_cell(payload, phase))
            except ApplicationError as exc:
                legacy.render_application_error(exc)
        cell = get_bound(st.session_state, "calpad_cell_result", sig)
        if isinstance(cell, Mapping):
            compact_metrics([
                ("Point group", cell["point_group_symbol"]),
                ("Symmetry order", cell["symmetry_order"]),
                ("M·M⁻¹ residual", sci(cell["metric_inverse_residual"], 3)),
            ], columns=3)
            with audit_expander("Metric tensors", expanded=False):
                legacy._render_matrix("Direct metric", cell["direct_metric"], expanded=True)
                legacy._render_matrix("Reciprocal metric", cell["reciprocal_metric"])
                st.write("Direct cell:", cell["direct_cell"])
                st.write("Reciprocal cell:", cell["reciprocal_cell"])

    with normal_tab:
        st.caption("Question: what direct-space direction is physically normal to a reciprocal plane, or vice versa?")
        obj_raw = st.text_input("Plane (hkl) or direction [uvw]", value="(1 0 1)", key="calpad_normal_object")
        obj = legacy._normalize_compact_crystal_text(obj_raw)
        max_index = st.number_input("Nearest low-index search bound", min_value=1, max_value=30, value=12, key="calpad_normal_bound")
        sig = fingerprint(project_sig, phase, obj, int(max_index))
        if st.button("Convert with the lattice metric", key="v4_normal"):
            try:
                put_bound(st.session_state, "calpad_normal_result", sig, calpad_normal_conversion(payload, phase, obj, max_index=int(max_index)))
            except ApplicationError as exc:
                legacy.render_application_error(exc)
        report = get_bound(st.session_state, "calpad_normal_result", sig)
        if isinstance(report, Mapping):
            render_finding(Finding(
                title="Metric normal conversion",
                conclusion=f"Nearest low-index representation: {report['nearest_low_index']['notation']}.",
                rationale=f"Angular mismatch = {float(report['nearest_low_index']['angular_mismatch_deg']):.6g}°; round-trip residual = {sci(report['roundtrip_projective_residual'], 3)}.",
                how="Plane covectors are converted with the inverse/direct metric as required; direct and reciprocal coefficients are never identified by cubic shorthand unless the metric itself makes that valid.",
                physical_meaning=str(report["relation"]),
                limitation="The nearest low-index result is an approximation to the metric-derived physical direction/plane when the exact coefficients are irrational or high-index.",
                evidence=(
                    Evidence("Raw target coefficients", "metric-derived", str(report["raw_target_coefficients"]), ""),
                    Evidence("Nearest low-index", "search bound", report["nearest_low_index"]["notation"], ""),
                    Evidence("Angular mismatch", "0° for exact parallelism", f"{float(report['nearest_low_index']['angular_mismatch_deg']):.9g}°", ""),
                ),
                tone="good",
            ))

    with low_tab:
        st.caption("Question: which low-index planes or directions are physically closest to the target under the phase metric?")
        target_raw = st.text_input("Target object", value="[1 0 1]", key="calpad_low_target")
        target = legacy._normalize_compact_crystal_text(target_raw)
        source_kind = legacy._target_kind(target)
        cols = st.columns(4)
        kind = cols[0].selectbox("Candidate type", ["direction", "plane"], key="calpad_candidate_kind")
        cross_type = source_kind is not None and source_kind != kind
        if cross_type:
            comparison = cols[1].selectbox("Cross-type comparison", ["direction–plane incidence", "normal-parallel"], key="calpad_cross_relation")
            sense = "projective"
        else:
            comparison = "same-type physical angle"
            sense = cols[1].selectbox("Angle sense", ["projective", "oriented"], key="calpad_angle_sense")
        bound = cols[2].number_input("Max |index|", min_value=1, max_value=12, value=3, key="calpad_low_bound")
        limit = cols[3].number_input("Rows", min_value=1, max_value=200, value=20, key="calpad_low_limit")
        sig = fingerprint(project_sig, phase, target, kind, comparison, sense, int(bound), int(limit))
        if st.button("Rank low-index objects", key="v4_low"):
            try:
                if comparison == "normal-parallel":
                    report = legacy._normal_parallel_ranking(payload, phase, target, kind, max_index=int(bound), limit=int(limit))
                else:
                    report = calpad_low_index_table(payload, phase, target, candidate_kind=kind, max_index=int(bound), limit=int(limit), angle_sense=sense)
                put_bound(st.session_state, "calpad_low_result", sig, report)
            except ApplicationError as exc:
                legacy.render_application_error(exc)
        report = get_bound(st.session_state, "calpad_low_result", sig)
        if isinstance(report, Mapping):
            rows = [dict(item) for item in report["rows"]]
            for row in rows:
                row["notation"] = legacy._clean_projective_notation(row.get("notation", ""), kind, sense)
            st.dataframe(format_dataframe_scientific(pd.DataFrame(rows)), hide_index=True, use_container_width=True)
            with st.expander("Why these angles are valid", expanded=False):
                if comparison == "normal-parallel":
                    st.write("The target is first converted to its metric-correct normal counterpart, then the physical angle is evaluated in the corresponding direct/reciprocal space.")
                elif cross_type:
                    st.latex(r"\sin\theta=\frac{|p^T u|}{\sqrt{u^T M u}\sqrt{p^T M^{-1}p}}")
                    st.write("θ=0° means the direction lies in the plane; θ=90° means it is parallel to the plane normal.")
                elif kind == "plane":
                    st.latex(r"\cos\theta=\frac{p^T M^{-1}q}{\sqrt{p^TM^{-1}p}\sqrt{q^TM^{-1}q}}")
                else:
                    st.latex(r"\cos\theta=\frac{u^T M v}{\sqrt{u^TMu}\sqrt{v^TMv}}")


def _render_ebsd_workspace() -> None:
    section_header(
        "EBSD audit",
        "Can this orientation map be trusted as an input before segmentation, OR fitting and parent reconstruction?",
        answer_hint="This first layer checks format, orientation consistency and coordinate/indexing integrity. It does not guess frame corrections.",
    )
    ebsd_file = st.file_uploader("EBSD file", type=["ang", "ctf"], key="ebsd_upload")
    unit_choice = st.selectbox("CTF 3-D Euler-angle unit", ["auto / standard 2-D", "degrees", "radians"], key="ebsd_ctf_unit")
    if ebsd_file is None:
        st.info("Upload an ANG or CTF map for the quick import audit. The research page contains the deeper explicit-schema EBSD reconstruction pipeline.")
        return
    raw = ebsd_file.getvalue()
    file_hash = hashlib.sha256(raw).hexdigest()
    sig = fingerprint(file_hash, ebsd_file.name, unit_choice)
    if st.button("Load and audit map", type="primary", key="v4_ebsd_load"):
        suffix = Path(ebsd_file.name).suffix.lower()
        tmp_name = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(raw)
                tmp_name = handle.name
            if suffix == ".ang":
                data = load_ang(tmp_name)
            elif suffix == ".ctf":
                unit = None if unit_choice == "auto / standard 2-D" else AngleUnit.DEGREE if unit_choice == "degrees" else AngleUnit.RADIAN
                data = load_ctf(tmp_name, three_dimensional_angle_unit=unit)
            else:
                raise input_error("Only .ang and .ctf are enabled in the quick audit.")
            audit = audit_map(data)
            stride = max(1, data.n_points // 8000)
            idx = np.arange(0, data.n_points, stride, dtype=int)
            preview = pd.DataFrame({"x": data.x[idx], "y": data.y[idx], "phase": data.phase_id[idx].astype(str), "indexed": data.indexed[idx]})
            result = {
                "file_name": ebsd_file.name,
                "sha256": file_hash,
                "audit": asdict(audit),
                "metadata": legacy._jsonable(dict(data.metadata)),
                "quality_fields": sorted(map(str, data.quality.keys())),
                "preview": preview.to_dict(orient="records"),
            }
            put_bound(st.session_state, "ebsd_result", sig, result)
        except ApplicationError as exc:
            legacy.render_application_error(exc)
        except (ValueError, OSError, UnicodeError) as exc:
            legacy.render_application_error(input_error(str(exc), hint="Check the EBSD format and explicit angle-unit choice. No frame correction is inferred."))
        finally:
            if tmp_name:
                try:
                    Path(tmp_name).unlink(missing_ok=True)
                except OSError:
                    pass
    result = get_bound(st.session_state, "ebsd_result", sig)
    if isinstance(result, Mapping):
        render_finding(ebsd_audit_finding(result["audit"]))
        preview = pd.DataFrame(result["preview"])
        if not preview.empty:
            st.scatter_chart(preview, x="x", y="y", color="phase")
        with audit_expander("Import metadata and provenance", expanded=False):
            st.write("SHA-256:", result["sha256"])
            st.write("Quality fields:", result["quality_fields"])
            st.json(legacy._jsonable(result["metadata"]))
    st.caption("For segmentation, KAM/GOS, parent reconstruction, bounded OR refinement and trace validation, use Theory comparison → EBSD pipeline, where every convention and threshold is explicit.")


def _collect_workbench_exports() -> dict[str, Any]:
    output: dict[str, Any] = {}
    keys = (
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
    for key in keys:
        value = st.session_state.get(key)
        if value is None:
            continue
        # BoundResult is intentionally serialized only through its public fields.
        if hasattr(value, "value"):
            output[key] = legacy._jsonable(getattr(value, "value"))
        elif isinstance(value, Mapping):
            output[key] = legacy._jsonable(value)
    return output


def _render_audit_workspace() -> None:
    section_header(
        "Numerical audit & export",
        "Are the numerical routes self-consistent, and can the exact calculated state and analyses be reproduced?",
    )
    try:
        _, response = legacy._require_project()
    except ApplicationError as exc:
        legacy.render_application_error(exc)
        return
    diagnostics = response.result["diagnostics"]  # type: ignore[attr-defined]
    parity = float(diagnostics["representation_parity_residual"])
    route = float(diagnostics["route_disagreement"])
    ge = float(diagnostics["generalized_eigen_residual"])
    ortho = float(diagnostics["metric_orthonormality_residual"])
    agreement = bool(diagnostics["exact_classification_agreement_ct_vs_ball_james"])
    render_finding(Finding(
        title="Numerical integrity",
        conclusion="The independent numerical routes are internally consistent." if agreement else "The CT/Ball–James classification audit does not agree and requires investigation.",
        rationale=f"Representation parity = {sci(parity,3)}, route disagreement = {sci(route,3)}, generalized-eigen residual = {sci(ge,3)}.",
        how="The solver audits the generalized eigenproblem, metric orthonormality, independent numerical routes and representation parity. Precision escalation is used only when the binary64 decision is ambiguous or fails its guards.",
        physical_meaning="Small audit residuals support the numerical implementation; they do not by themselves validate a physical model against experiment.",
        limitation="Numerical self-consistency is not a substitute for theory-to-experiment validation.",
        evidence=(
            Evidence("Generalized eigen residual", "≈ 0", sci(ge,6), ""),
            Evidence("Metric orthonormality", "≈ 0", sci(ortho,6), ""),
            Evidence("Route disagreement", "≈ 0", sci(route,6), ""),
            Evidence("Representation parity", "≈ 0", sci(parity,6), ""),
            Evidence("CT/BJ existence classification", "independent agreement", "agree" if agreement else "differ", ""),
        ),
        tone="good" if agreement else "bad",
    ))

    compact_metrics([
        ("Metric solver", diagnostics["solver_source"] or "—"),
        ("Precision escalated", "yes" if diagnostics["precision_escalated"] else "no"),
    ], columns=2)

    with audit_expander("Theory-consistency contracts", expanded=False):
        st.json(legacy._jsonable(response.result["contracts"]))  # type: ignore[attr-defined]

    st.markdown("### Export")
    project_json = response.project_json()  # type: ignore[attr-defined]
    result_json = response.to_json()  # type: ignore[attr-defined]
    workstation_json = json.dumps({
        "calculated_draft_signature": st.session_state.get("calculated_draft_signature"),
        "analyses": _collect_workbench_exports(),
    }, indent=2, ensure_ascii=False, allow_nan=False, default=str)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Project state", project_json, "cualni_ct_project.json", "application/json", use_container_width=True)
    c2.download_button("Transformation results", result_json, "cualni_ct_results.json", "application/json", use_container_width=True)
    c3.download_button("Workbench analyses", workstation_json, "cualni_ct_workbench_analyses.json", "application/json", use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="CuAlNi-CT Workbench", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    install_styles()
    legacy._consume_pending_project_load()

    title, length_unit, project_id = _sidebar()
    page_header(
        "CuAlNi-CT",
        "A question-first crystallography workstation: every conclusion is accompanied by the criterion, calculation route, numerical evidence and scope of the claim.",
    )
    _project_status()

    if "app_error" in st.session_state:
        legacy.render_application_error(st.session_state["app_error"])

    workspace = st.session_state.get("workbench_v4_workspace", "State definition")
    if workspace == "State definition":
        _render_setup(title, length_unit, project_id)
    elif workspace == "Compatibility":
        _render_compatibility()
    elif workspace == "Orientation & reconstruction":
        _render_orientation_workspace()
    elif workspace == "Twins & PTMC":
        _render_martensite_workspace()
    elif workspace == "Crystal geometry":
        _render_geometry_workspace()
    elif workspace == "EBSD audit":
        _render_ebsd_workspace()
    elif workspace == "Numerical audit & export":
        _render_audit_workspace()
    else:
        st.error(f"Unknown workspace: {workspace}")


if __name__ == "__main__":
    main()
