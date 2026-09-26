from __future__ import annotations

"""CuAlNi-CT public phase-transformation crystallography workstation."""

import json
from pathlib import Path
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.application import (
    CalculationRequest,
    TransformationInput,
    calculate_payload,
    calculate_request,
    point_group_options,
)
from app.errors import ApplicationError, input_error
from app.ui_components import (
    PhaseDefaults,
    compact_key_value,
    correspondence_editor,
    matrix_editor,
    matrix_frame,
    render_application_error,
    smart_phase_editor,
    variant_table,
    vector_editor,
)
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
from cualni_cryst.orientation import EulerConvention
from cualni_cryst.representation import CartesianConvention


st.set_page_config(
    page_title="CuAlNi-CT Workbench",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.0rem; padding-bottom: 4rem; max-width: 1540px;}
[data-testid="stSidebar"] .block-container {padding-top: 1.0rem;}
[data-testid="stMetricValue"] {font-size: 1.32rem;}
.app-kicker {font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; opacity:.63;}
.app-title {font-size:2.15rem; font-weight:760; line-height:1.08; margin:.12rem 0 .18rem 0;}
.app-subtitle {font-size:1rem; opacity:.78; max-width:980px; margin-bottom:.7rem;}
.project-strip {border:1px solid rgba(128,128,128,.25); border-radius:.6rem; padding:.55rem .8rem; margin:.25rem 0 .8rem 0;}
.small-note {font-size:.86rem; opacity:.74;}
.matrix-caption {font-size:.82rem; opacity:.72; margin-top:-.25rem;}
</style>
""",
    unsafe_allow_html=True,
)


def _fmt(value: object, digits: int = 10) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _require_project() -> tuple[dict[str, object], object]:
    payload = st.session_state.get("current_project_payload")
    response = st.session_state.get("current_response")
    if not isinstance(payload, dict) or response is None:
        raise input_error("Create or open a project in Setup and calculate it first.")
    return payload, response


def _phase_names(payload: Mapping[str, object]) -> tuple[str, str, str, str]:
    phases = payload.get("phases", [])
    transformations = payload.get("transformations", [])
    if not isinstance(phases, list) or len(phases) < 2 or not isinstance(transformations, list) or not transformations:
        return "Parent", "Product", "phase_A", "phase_M"
    t = transformations[0]
    if not isinstance(t, Mapping):
        return "Parent", "Product", "phase_A", "phase_M"
    parent_id = str(t.get("parent_phase_id", "phase_A"))
    product_id = str(t.get("product_phase_id", "phase_M"))
    by_id = {str(item.get("phase_id")): item for item in phases if isinstance(item, Mapping)}
    parent = by_id.get(parent_id, {})
    product = by_id.get(product_id, {})
    return (
        str(parent.get("label", parent_id)),
        str(product.get("label", product_id)),
        parent_id,
        product_id,
    )


def _render_matrix(title: str, matrix: object, *, expanded: bool = False) -> None:
    with st.expander(title, expanded=expanded):
        st.dataframe(matrix_frame(matrix), use_container_width=True)


def _render_transformation(response: object) -> None:
    result = response.result  # type: ignore[attr-defined]
    summary = result["summary"]
    code = summary["classification"]
    if code == "EXACT_CT_COMPATIBLE":
        st.success(f"Exact CT compatibility · {summary['ct_reason']}")
    elif code == "NOT_EXACT_NEAREST_DEGENERACY_DIAGNOSTIC_AVAILABLE":
        st.warning("No exact CT compatibility. A nearest-degeneracy diagnostic exists and is kept explicitly separate from an exact solution.")
    else:
        st.info(f"No exact CT compatibility · {summary['ct_reason']}")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("λ₁", _fmt(summary["lambda1"], 12))
    c2.metric("λ₂", _fmt(summary["lambda2"], 12))
    c3.metric("λ₃", _fmt(summary["lambda3"], 12))
    c4.metric("|λ₂−1|", _fmt(summary["lambda2_residual"], 6))
    c5.metric("Variants", summary["topological_variant_count"])
    c6.metric("Operators", summary["operator_count"])

    overview, metric_tab, ct_tab, bj_tab, variants_tab = st.tabs(
        ["Summary", "Metric / stretch", "Correspondence Theory", "Ball–James", "Variants / topology"]
    )

    with overview:
        compact_key_value(
            [
                ("CT exact compatible", summary["ct_exact_compatible"]),
                ("CT degeneracy order", summary["degeneracy_order"]),
                ("Exact CT habit planes", summary["ct_exact_habit_plane_count"]),
                ("Ball–James λ₂ condition", summary["ball_james_lambda2_exact"]),
                ("Ball–James rank-one solutions", summary["ball_james_solution_count"]),
                ("Stretch variants", summary["variant_count"]),
            ]
        )
        st.markdown("#### Principal stretches")
        stretch_df = pd.DataFrame(
            {"stretch": ["λ₁", "λ₂", "λ₃"], "value": [summary["lambda1"], summary["lambda2"], summary["lambda3"]]}
        ).set_index("stretch")
        st.bar_chart(stretch_df)

    with metric_tab:
        metric = result["metric"]
        _render_matrix("Parent metric M_A", metric["parent_metric"])
        _render_matrix("Product metric M_M", metric["product_metric"])
        _render_matrix("Pulled product metric Cᵀ M_M C", metric["pulled_product_metric"])
        _render_matrix("Dimensional CMC", metric["cmc_dimensional"], expanded=True)
        _render_matrix("Normalized CMC", metric["cmc_normalized"])
        _render_matrix("Dimensional SMC", metric["smc_dimensional"])
        _render_matrix("Right stretch U", metric["stretch"], expanded=True)
        _render_matrix("Principal axes", metric["principal_axes"])

    with ct_tab:
        ct = result["ct_detail"]
        compact_key_value(
            [
                ("Classification", ct["classification"]),
                ("Degeneracy order", ct["degeneracy_order"]),
                ("Reason", ct["reason"]),
                ("Nearest zero residual", ct["nearest_zero_residual"]),
                ("Inertia (−,0,+)", ct["inertia"]),
            ]
        )
        st.markdown("#### Generalized metric spectrum")
        st.dataframe(
            pd.DataFrame({"η = μ−1": ct["eta_eigenvalues"], "μ": ct["generalized_mu"]}, index=[1, 2, 3]),
            use_container_width=True,
        )
        planes = ct["exact_habit_planes_parent_covectors"]
        st.markdown("#### Exact habit-plane covectors in parent coordinates")
        if planes:
            st.dataframe(pd.DataFrame(planes, columns=["p₁", "p₂", "p₃"]), use_container_width=True)
        else:
            st.caption("No exact CT habit plane is returned for this state.")
        approx = ct["approximate_diagnostic"]
        if not ct["exact_compatible"] and approx["candidate_planes_parent_covectors"]:
            with st.expander("Nearest-degeneracy diagnostic — not an exact CT solution"):
                st.warning(approx["explanation"])
                st.write("Residual:", _fmt(approx["residual"], 9))
                st.dataframe(pd.DataFrame(approx["candidate_planes_parent_covectors"], columns=["p₁", "p₂", "p₃"]), use_container_width=True)

    with bj_tab:
        bj = result["ball_james_detail"]
        st.write("λ₂ criterion satisfied:", bj["lambda2_exact"])
        if not bj["solutions"]:
            st.caption("No single-variant Ball–James rank-one solution was returned.")
        for index, solution in enumerate(bj["solutions"], 1):
            with st.expander(f"Solution {index} · branch {solution['branch']}", expanded=index == 1):
                left, right = st.columns(2)
                with left:
                    compact_key_value([("Residual", solution["residual"]), ("Eigenvalues C", solution["eigenvalues_C"])])
                    st.write("Shape vector a")
                    st.dataframe(pd.DataFrame([solution["shape_vector_a"]], columns=["a₁", "a₂", "a₃"]), hide_index=True, use_container_width=True)
                    st.write("Habit normal n")
                    st.dataframe(pd.DataFrame([solution["habit_normal_n"]], columns=["n₁", "n₂", "n₃"]), hide_index=True, use_container_width=True)
                with right:
                    st.write("Rotation R")
                    st.dataframe(matrix_frame(solution["rotation"]), use_container_width=True)

    with variants_tab:
        topology = result["topology"]
        stretch = result["stretch_variants"]
        cols = st.columns(4)
        cols[0].metric("Parent group", topology["parent_group_order"])
        cols[1].metric("Product group", topology["product_group_order"])
        cols[2].metric("Variants", topology["n_variants"])
        cols[3].metric("Operators", topology["n_operators"])
        if topology.get("operator_summaries"):
            st.dataframe(pd.DataFrame(topology["operator_summaries"]), hide_index=True, use_container_width=True)
        mats = stretch.get("variants", [])
        if mats:
            selected = st.selectbox("Stretch variant", range(len(mats)), format_func=lambda i: f"U{i + 1}", key="transformation_variant_view")
            st.dataframe(matrix_frame(mats[selected]), use_container_width=True)


def _render_orientation(analysis: Mapping[str, object]) -> None:
    report = analysis["report"]
    state = analysis["state"]
    st.info(str(analysis.get("origin_note", "")))
    cols = st.columns(5)
    cols[0].metric("OR variants", report["orientation_variant_count"])
    cols[1].metric("OR operators", report["orientation_operator_count"])
    cols[2].metric("Hᵀ proper order", report["proper_orientation_intersection_order"])
    cols[3].metric("Rotation residual", _fmt(report["audit"]["maximum_residual"], 4))
    cols[4].metric("Parity residual", _fmt(report["parity"]["maximum_residual"], 4))

    r1, r2 = st.columns([1.2, 1.0])
    with r1:
        st.markdown("#### Base orientation relationship")
        st.caption("Convention: x_parent = R(parent←product) · x_product")
        st.dataframe(matrix_frame(state["R_reference_from_moving"]), use_container_width=True)
    with r2:
        compact_key_value(
            [
                ("Axis", report["axis_angle"]["axis"]),
                ("Angle (deg)", report["axis_angle"]["angle_deg"]),
                ("Euler ZXZ active", [report["euler_zxz_active"]["phi1_deg"], report["euler_zxz_active"]["Phi_deg"], report["euler_zxz_active"]["phi2_deg"]]),
                ("Euler ZXZ passive", [report["euler_zxz_passive"]["phi1_deg"], report["euler_zxz_passive"]["Phi_deg"], report["euler_zxz_passive"]["phi2_deg"]]),
            ]
        )

    variants = analysis.get("variants", [])
    if isinstance(variants, list) and variants:
        st.markdown("#### Orientation variants")
        st.dataframe(variant_table(variants), hide_index=True, use_container_width=True)
        selected = st.selectbox("Inspect OR variant", range(len(variants)), format_func=lambda i: f"Variant {i + 1}", key="or_variant_inspect")
        item = variants[selected]
        left, right = st.columns(2)
        with left:
            st.caption("Parent ← Product")
            st.dataframe(matrix_frame(item["R_parent_from_product"]), use_container_width=True)
            st.write(item["forward_axis_angle"])
        with right:
            st.caption("Product ← Parent")
            st.dataframe(matrix_frame(item["R_product_from_parent"]), use_container_width=True)
            st.write(item["reverse_axis_angle"])

    operators = analysis.get("operators", [])
    if isinstance(operators, list) and operators:
        with st.expander("Orientation operator classes"):
            st.dataframe(pd.DataFrame(operators), hide_index=True, use_container_width=True)
    with st.expander("Topology / convention audit"):
        st.json(report["cayron_topology_audit"])
        for warning in report.get("warnings", []):
            st.caption(str(warning))


def _current_orientation() -> Mapping[str, object] | None:
    value = st.session_state.get("current_orientation_analysis")
    return value if isinstance(value, Mapping) else None


# Header
st.markdown('<div class="app-kicker">Phase-transformation crystallography workstation</div>', unsafe_allow_html=True)
st.markdown('<div class="app-title">CuAlNi-CT</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Build a two-phase crystallographic state, analyze transformation compatibility, generate orientation variants, reconstruct parent/product orientations, and inspect metric-correct crystallographic geometry.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Project")
    project_title = st.text_input("Title", value="Untitled transformation")
    length_unit = st.selectbox("Length unit", ["angstrom", "nm", "pm"], index=0)
    with st.expander("Advanced project metadata"):
        project_id = st.text_input("Project ID", value="workbench_project")
    st.divider()
    st.markdown("### Open project")
    uploaded = st.file_uploader("Project JSON", type=["json"], label_visibility="collapsed")
    if st.button("Open and calculate", disabled=uploaded is None, use_container_width=True):
        try:
            payload = json.loads(uploaded.getvalue().decode("utf-8"))  # type: ignore[union-attr]
            if not isinstance(payload, dict):
                raise ValueError("Top-level JSON must be an object.")
            response = calculate_payload(payload, source=uploaded.name)  # type: ignore[union-attr]
            st.session_state["current_response"] = response
            st.session_state["current_project_payload"] = response.project_payload
            st.session_state.pop("app_error", None)
        except ApplicationError as exc:
            st.session_state["app_error"] = exc
        except Exception as exc:
            st.session_state["app_error"] = input_error(f"Uploaded project could not be opened: {exc}")
    if st.session_state.get("current_response") is not None:
        st.success("Project ready")

if "app_error" in st.session_state:
    render_application_error(st.session_state["app_error"])

current_payload = st.session_state.get("current_project_payload")
if isinstance(current_payload, dict):
    parent_name, product_name, _, _ = _phase_names(current_payload)
    st.markdown(
        f'<div class="project-strip"><b>{current_payload.get("title", "Project")}</b> &nbsp; · &nbsp; {parent_name} → {product_name}</div>',
        unsafe_allow_html=True,
    )

setup_tab, transformation_tab, orientation_tab, martensite_tab, calpad_tab, diagnostics_tab = st.tabs(
    ["Setup", "Transformation", "OR & reconstruction", "Martensite", "CalPad", "Diagnostics / export"]
)

point_groups = point_group_options()

with setup_tab:
    st.markdown("## Define the two phases")
    st.caption("The editor exposes only independent conventional-cell parameters for the selected crystal family. The backend still validates the full metric and point-group consistency.")
    with st.container():
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

        st.divider()
        mcol, info = st.columns([1.0, 1.15], gap="large")
        with mcol:
            correspondence = correspondence_editor()
        with info:
            st.markdown("### Transformation")
            st.write("Parent → product crystallographic mapping is defined by the exact correspondence matrix at left.")
            st.caption("The default is a generic demonstration, not a stored Cu–Al–Ni material preset.")
            with st.expander("Advanced transformation metadata"):
                transformation_id = st.text_input("Transformation ID", value="A_to_M")
                transformation_label = st.text_input("Display name", value="Parent → Product")
        calculate = st.button("Calculate transformation", type="primary", use_container_width=True)

    if calculate:
        try:
            request = CalculationRequest(
                project_id=project_id,
                title=project_title,
                parent=parent,
                product=product,
                transformation=TransformationInput.from_rows(
                    transformation_id,
                    parent.phase_id,
                    product.phase_id,
                    correspondence,
                    label=transformation_label,
                ),
                notes="Created with the CuAlNi-CT public workstation.",
            )
            response = calculate_request(request)
            st.session_state["current_response"] = response
            st.session_state["current_project_payload"] = response.project_payload
            st.session_state.pop("app_error", None)
            st.session_state.pop("current_orientation_analysis", None)
            st.success("Transformation calculated. Continue with the analysis tabs.")
        except ApplicationError as exc:
            render_application_error(exc)

with transformation_tab:
    st.markdown("## Transformation analysis")
    try:
        _, response = _require_project()
        _render_transformation(response)
    except ApplicationError as exc:
        render_application_error(exc)

with orientation_tab:
    st.markdown("## Orientation relationship & bidirectional reconstruction")
    st.caption("Orientation R and lattice correspondence C are kept as different mathematical objects. Choose how the physical OR is defined; the app never substitutes one for the other silently.")
    try:
        payload, _ = _require_project()
        parent_name, product_name, parent_id, product_id = _phase_names(payload)
        source_mode = st.radio(
            "OR source",
            ["Polar rotation candidate", "Rotation matrix", "Euler ZXZ", "Two crystallographic parallelisms"],
            horizontal=True,
        )
        result = None
        if source_mode == "Polar rotation candidate":
            st.caption("Finite-strain polar rotation derived from the supplied correspondence deformation. It is labelled as a candidate, not as an experimentally established OR.")
            if st.button("Analyze polar-rotation candidate", type="primary"):
                result = orientation_from_polar_correspondence(payload)
        elif source_mode == "Rotation matrix":
            matrix = matrix_editor(
                "R(parent ← product)",
                key_prefix="or_matrix",
                help_text="Convention: x_parent = R · x_product. The matrix must be a proper rotation.",
            )
            cols = st.columns(2)
            parent_conv = cols[0].selectbox("Parent Cartesian frame", [item.value for item in CartesianConvention], index=1)
            product_conv = cols[1].selectbox("Product Cartesian frame", [item.value for item in CartesianConvention], index=1)
            repair = st.checkbox("Explicitly project a near-rotation to SO(3)", value=False, help="Never applied silently.")
            if st.button("Analyze OR matrix", type="primary"):
                result = orientation_from_matrix(payload, matrix, parent_convention=parent_conv, product_convention=product_conv, repair=repair)
        elif source_mode == "Euler ZXZ":
            cols = st.columns(3)
            phi1 = cols[0].number_input("φ₁ (deg)", value=0.0)
            Phi = cols[1].number_input("Φ (deg)", value=0.0)
            phi2 = cols[2].number_input("φ₂ (deg)", value=0.0)
            cols2 = st.columns(3)
            euler_conv = cols2[0].selectbox("Euler convention", [item.value for item in EulerConvention])
            parent_conv = cols2[1].selectbox("Parent Cartesian frame", [item.value for item in CartesianConvention], index=1, key="euler_parent_frame")
            product_conv = cols2[2].selectbox("Product Cartesian frame", [item.value for item in CartesianConvention], index=1, key="euler_product_frame")
            if st.button("Analyze Euler OR", type="primary"):
                result = orientation_from_euler(payload, phi1, Phi, phi2, euler_convention=euler_conv, parent_convention=parent_conv, product_convention=product_conv)
        else:
            st.caption("Use two non-collinear direction/plane parallelisms, for example (1 1 1) // (0 1 1) and [1 0 -1] // [1 1 -1].")
            c1, c2 = st.columns(2)
            p1 = c1.text_input(f"{parent_name}: first object", value="(1 1 1)")
            m1 = c2.text_input(f"{product_name}: first object", value="(0 1 1)")
            p2 = c1.text_input(f"{parent_name}: second object", value="[1 0 -1]")
            m2 = c2.text_input(f"{product_name}: second object", value="[1 1 -1]")
            if st.button("Solve exact OR from parallelisms", type="primary"):
                solved = orientation_from_parallelisms(payload, p1, m1, p2, m2)
                st.session_state["parallelism_solve"] = solved
                candidates = solved.get("candidates", [])
                if candidates:
                    result = candidates[0]

        if result is not None:
            st.session_state["current_orientation_analysis"] = result

        solved = st.session_state.get("parallelism_solve")
        if source_mode == "Two crystallographic parallelisms" and isinstance(solved, Mapping):
            candidates = solved.get("candidates", [])
            if isinstance(candidates, list) and len(candidates) > 1:
                selected = st.selectbox("Exact OR candidate", range(len(candidates)), format_func=lambda i: f"Candidate {i + 1}")
                st.session_state["current_orientation_analysis"] = candidates[selected]

        analysis = _current_orientation()
        if analysis is not None:
            st.divider()
            _render_orientation(analysis)

            map_tab, recon_tab = st.tabs(["Map crystallographic objects", "Parent ↔ product reconstruction"])
            with map_tab:
                st.markdown("### Correspondence mapping versus physical OR mapping")
                object_text = st.text_input("Direction or plane", value="[1 0 0]", help="Use [uvw] for direct directions and (hkl) for reciprocal planes.")
                source_phase = st.radio("Object belongs to", ["parent", "product"], horizontal=True)
                left, right = st.columns(2)
                if left.button("Map through correspondence C", use_container_width=True):
                    st.session_state["correspondence_map"] = map_correspondence_object(payload, object_text, source_phase=source_phase)
                if right.button("Map through physical OR R", use_container_width=True):
                    base_R = analysis["state"]["R_reference_from_moving"]
                    st.session_state["orientation_map"] = map_orientation_object(payload, base_R, object_text, source_phase=source_phase)
                lres, rres = st.columns(2)
                with lres:
                    if "correspondence_map" in st.session_state:
                        st.caption("Crystallographic correspondence")
                        st.json(st.session_state["correspondence_map"])
                with rres:
                    if "orientation_map" in st.session_state:
                        st.caption("Physical orientation relationship")
                        st.json(st.session_state["orientation_map"])

            with recon_tab:
                st.markdown("### Reconstruct the other phase from an observed orientation")
                st.caption("Matrix convention: x_sample = g(sample←crystal) · x_crystal. No EBSD-vendor Euler convention is guessed.")
                observed_phase = st.radio("Observed phase", ["parent", "product"], horizontal=True, key="recon_observed_phase")
                g = matrix_editor("Observed g(sample ← crystal)", key_prefix="sample_g")
                if st.button("Reconstruct all symmetry-distinct candidates", type="primary"):
                    st.session_state["reconstruction"] = reconstruct_sample_orientations(payload, analysis, g, observed_phase=observed_phase)
                recon = st.session_state.get("reconstruction")
                if isinstance(recon, Mapping):
                    candidates = recon.get("candidates", [])
                    if isinstance(candidates, list) and candidates:
                        table = pd.DataFrame(
                            [
                                {
                                    "variant": item["variant_index"] + 1,
                                    "reconstructed phase": item["reconstructed_phase_id"],
                                    "φ₁ active": item["euler_zxz_active"]["phi1_deg"],
                                    "Φ active": item["euler_zxz_active"]["Phi_deg"],
                                    "φ₂ active": item["euler_zxz_active"]["phi2_deg"],
                                    "rotation residual": item["rotation_audit"]["maximum_residual"],
                                }
                                for item in candidates
                            ]
                        )
                        st.dataframe(table, hide_index=True, use_container_width=True)
                        sel = st.selectbox("Inspect reconstructed candidate", range(len(candidates)), format_func=lambda i: f"Candidate {i + 1}")
                        st.dataframe(matrix_frame(candidates[sel]["g_sample_from_reconstructed"]), use_container_width=True)
                        st.caption(str(recon.get("note", "")))
    except ApplicationError as exc:
        render_application_error(exc)

with martensite_tab:
    st.markdown("## Martensite: twins, classical PTMC & cofactor conditions")
    st.caption("The workstation composes existing general solvers: selected stretch variants → Mallard Type-I/II twin candidates → classical single-shear PTMC → cofactor conditions. No twin system or solution is fabricated when the mathematical prerequisites are absent.")
    try:
        payload, response = _require_project()
        variants = response.result["stretch_variants"]["variants"]  # type: ignore[attr-defined]
        if len(variants) < 2:
            st.info("This transformation produced fewer than two distinct stretch variants, so a variant-pair twin analysis is not available.")
        else:
            cols = st.columns(2)
            vi = cols[0].selectbox("Variant i", range(len(variants)), format_func=lambda i: f"U{i + 1}")
            vj_options = [i for i in range(len(variants)) if i != vi]
            vj = cols[1].selectbox("Variant j", vj_options, format_func=lambda i: f"U{i + 1}")
            if st.button("Find Mallard twins and evaluate PTMC / cofactor", type="primary"):
                st.session_state["martensite_pair"] = martensite_variant_pair_analysis(payload, vi, vj)
            pair = st.session_state.get("martensite_pair")
            if isinstance(pair, Mapping):
                relations = pair.get("relations", [])
                if not relations:
                    st.info("No registered parent order-two symmetry produced a Mallard twin for this selected pair.")
                for k, relation in enumerate(relations, 1):
                    with st.expander(f"Relation {k} · Mallard Type {relation['mallard_kind']}", expanded=k == 1):
                        top = st.columns(4)
                        top[0].metric("Twin shear", _fmt(relation["twin_shear_magnitude"], 8))
                        top[1].metric("Mallard residual", _fmt(relation["mallard_residual"], 4))
                        top[2].metric("PTMC solutions", len(relation["ptmc"]))
                        top[3].metric("Cofactor CC1–3", "Satisfied" if relation["cofactor"]["satisfied"] else "Not all satisfied")
                        a_col, n_col = st.columns(2)
                        a_col.dataframe(pd.DataFrame([relation["twin_a"]], columns=["a₁", "a₂", "a₃"]), hide_index=True, use_container_width=True)
                        n_col.dataframe(pd.DataFrame([relation["twin_n"]], columns=["n₁", "n₂", "n₃"]), hide_index=True, use_container_width=True)
                        st.markdown("##### Cofactor conditions")
                        st.dataframe(pd.DataFrame([relation["cofactor"]]), hide_index=True, use_container_width=True)
                        st.markdown("##### Classical PTMC")
                        if relation["ptmc"]:
                            pt = pd.DataFrame(
                                [
                                    {
                                        "f": item["volume_fraction"],
                                        "middle-stretch residual": item["middle_stretch_residual"],
                                        "habit connections": len(item["habit_connections"]),
                                        "independent rank-one residual": item["independent_rank_one_crosscheck_residual"],
                                    }
                                    for item in relation["ptmc"]
                                ]
                            )
                            st.dataframe(pt, hide_index=True, use_container_width=True)
                        else:
                            st.caption(relation.get("ptmc_note") or "No classical PTMC volume-fraction solution was returned.")

        with st.expander("Advanced: user-specified lattice-invariant shear a ⊗ n"):
            variant_index = st.selectbox("Stretch variant", range(len(variants)), format_func=lambda i: f"U{i + 1}", key="manual_ptmc_variant") if variants else 0
            a = vector_editor("a", default=(0.1, 0.0, 0.0), key_prefix="manual_a")
            n = vector_editor("n", default=(0.0, 1.0, 0.0), key_prefix="manual_n")
            if st.button("Evaluate manual PTMC / cofactor"):
                st.session_state["manual_ptmc"] = manual_ptmc_cofactor_analysis(payload, variant_index, a, n)
            if "manual_ptmc" in st.session_state:
                st.json(st.session_state["manual_ptmc"])
        st.caption("Scope note: this panel exposes the repository's classical single-shear PTMC implementation. It does not label double-shear PTMC as available when no validated double-shear solver is present.")
    except ApplicationError as exc:
        render_application_error(exc)

with calpad_tab:
    st.markdown("## Metric-correct crystallography calculator")
    st.caption("Quick single-phase crystallographic geometry, retaining the direct/reciprocal distinction for non-cubic crystals.")
    try:
        payload, _ = _require_project()
        parent_name, product_name, parent_id, product_id = _phase_names(payload)
        phase = st.selectbox("Phase", [parent_id, product_id], format_func=lambda x: parent_name if x == parent_id else product_name)
        cell_tab, normal_tab, low_tab = st.tabs(["Cell / reciprocal cell", "Plane ↔ physical normal", "Low-index ranking"])
        with cell_tab:
            if st.button("Inspect phase cell"):
                st.session_state["calpad_cell"] = calpad_phase_cell(payload, phase)
            cell = st.session_state.get("calpad_cell")
            if isinstance(cell, Mapping):
                compact_key_value(
                    [
                        ("Point group", cell["point_group_symbol"]),
                        ("Symmetry order", cell["symmetry_order"]),
                        ("Direct cell", cell["direct_cell"]),
                        ("Reciprocal cell", cell["reciprocal_cell"]),
                        ("M·M⁻¹ residual", cell["metric_inverse_residual"]),
                    ]
                )
                _render_matrix("Direct metric", cell["direct_metric"], expanded=True)
                _render_matrix("Reciprocal metric", cell["reciprocal_metric"])
        with normal_tab:
            obj = st.text_input("Plane or direction", value="(1 0 1)", key="calpad_normal_object")
            max_index = st.number_input("Nearest low-index search bound", min_value=1, max_value=30, value=12)
            if st.button("Convert using the lattice metric"):
                st.session_state["calpad_normal"] = calpad_normal_conversion(payload, phase, obj, max_index=int(max_index))
            report = st.session_state.get("calpad_normal")
            if isinstance(report, Mapping):
                compact_key_value(
                    [
                        ("Relation", report["relation"]),
                        ("Raw coefficients", report["raw_target_coefficients"]),
                        ("Projective coefficients", report["projective_target_coefficients"]),
                        ("Nearest low-index", report["nearest_low_index"]["notation"]),
                        ("Angular mismatch (deg)", report["nearest_low_index"]["angular_mismatch_deg"]),
                        ("Round-trip residual", report["roundtrip_projective_residual"]),
                    ]
                )
                for warning in report.get("warnings", []):
                    st.warning(str(warning))
        with low_tab:
            target = st.text_input("Target object", value="[1 0 1]", key="calpad_low_target")
            cols = st.columns(4)
            kind = cols[0].selectbox("Candidates", ["direction", "plane"])
            sense = cols[1].selectbox("Angle sense", ["projective", "oriented"])
            bound = cols[2].number_input("Max |index|", min_value=1, max_value=12, value=3)
            limit = cols[3].number_input("Rows", min_value=1, max_value=200, value=20)
            if st.button("Rank low-index objects"):
                st.session_state["calpad_low"] = calpad_low_index_table(payload, phase, target, candidate_kind=kind, max_index=int(bound), limit=int(limit), angle_sense=sense)
            low = st.session_state.get("calpad_low")
            if isinstance(low, Mapping):
                st.dataframe(pd.DataFrame(low["rows"]), hide_index=True, use_container_width=True)
                with st.expander("Derivation / search definition"):
                    st.json(low["derivation"])
    except ApplicationError as exc:
        render_application_error(exc)

with diagnostics_tab:
    st.markdown("## Diagnostics, provenance & export")
    try:
        payload, response = _require_project()
        diagnostics = response.result["diagnostics"]  # type: ignore[attr-defined]
        cols = st.columns(4)
        cols[0].metric("Metric solver", diagnostics["solver_source"] or "—")
        cols[1].metric("Precision escalated", "Yes" if diagnostics["precision_escalated"] else "No")
        cols[2].metric("Representation parity", _fmt(diagnostics["representation_parity_residual"], 4))
        cols[3].metric("CT / BJ agreement", "Yes" if diagnostics["exact_classification_agreement_ct_vs_ball_james"] else "No")
        st.dataframe(
            pd.DataFrame(
                [
                    ("Generalized eigen equation", diagnostics["generalized_eigen_residual"]),
                    ("Metric orthonormality", diagnostics["metric_orthonormality_residual"]),
                    ("Route disagreement", diagnostics["route_disagreement"]),
                    ("Representation parity", diagnostics["representation_parity_residual"]),
                ],
                columns=["diagnostic", "residual"],
            ),
            hide_index=True,
            use_container_width=True,
        )
        with st.expander("Theory-consistency audit"):
            st.json(response.result["contracts"])  # type: ignore[attr-defined]

        st.markdown("### Export")
        project_json = response.project_json()  # type: ignore[attr-defined]
        result_json = response.to_json()  # type: ignore[attr-defined]
        extras = {
            "orientation_analysis": st.session_state.get("current_orientation_analysis"),
            "reconstruction": st.session_state.get("reconstruction"),
            "martensite_variant_pair": st.session_state.get("martensite_pair"),
        }
        workstation_json = json.dumps(extras, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        c1, c2, c3 = st.columns(3)
        c1.download_button("Project JSON", project_json, "cualni_ct_project.json", "application/json", use_container_width=True)
        c2.download_button("Transformation results", result_json, "cualni_ct_results.json", "application/json", use_container_width=True)
        c3.download_button("Workbench analyses", workstation_json, "cualni_ct_workbench_analyses.json", "application/json", use_container_width=True)

        with st.expander("Method scope / scientific honesty"):
            st.dataframe(
                pd.DataFrame(
                    [
                        ("Metric / CT / SMC / stretch", "Available", "General backend solver"),
                        ("Ball–James single-variant", "Available", "General backend solver"),
                        ("Orientation variants / operators", "Available", "Cayron-style topology"),
                        ("Parent ↔ product reconstruction", "Available", "Explicit OR + symmetry variants"),
                        ("Mallard twins", "Available", "For symmetry-related selected stretch pairs"),
                        ("Classical single-shear PTMC", "Available", "Exact-binary64 polynomial/root workflow"),
                        ("Cofactor CC1–CC3", "Available", "User/Mallard twin system"),
                        ("Metric CalPad", "Available", "Direct/reciprocal geometry"),
                        ("Double-shear PTMC", "Not claimed", "No validated solver exposed"),
                        ("E2E / 3-D NCS / diffraction simulation", "Not claimed", "Not represented by a validated backend workflow in this release"),
                    ],
                    columns=["Method", "Status", "Scope"],
                ),
                hide_index=True,
                use_container_width=True,
            )
    except ApplicationError as exc:
        render_application_error(exc)
