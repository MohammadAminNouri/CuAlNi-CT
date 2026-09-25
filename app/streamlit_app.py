from __future__ import annotations

"""CuAlNi-CT public crystallography workbench.

Normal users only enter crystallographic input and inspect/export results.  All
science is delegated to the existing verified ``CalculationService``.
"""

import json
from pathlib import Path
import sys

# Streamlit executes this file as a script. Keep the repository root importable
# without requiring normal users to manipulate PYTHONPATH.
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
from app.errors import ApplicationError, internal_error
from app.ui_components import (
    PhaseDefaults,
    correspondence_editor,
    matrix_frame,
    phase_editor,
    render_application_error,
)


st.set_page_config(
    page_title="CuAlNi-CT Workbench",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 4rem; max-width: 1480px;}
[data-testid="stMetricValue"] {font-size: 1.45rem;}
[data-testid="stSidebar"] .block-container {padding-top: 1.25rem;}
.app-kicker {font-size: 0.82rem; letter-spacing: .08em; text-transform: uppercase; opacity: .72;}
.app-title {font-size: 2.2rem; font-weight: 700; margin: .05rem 0 .15rem 0;}
.app-subtitle {font-size: 1.02rem; opacity: .78; margin-bottom: 1rem;}
.science-note {border-left: 3px solid rgba(130,130,130,.45); padding-left: .8rem; opacity: .86;}
</style>
""",
    unsafe_allow_html=True,
)


def _format_scalar(value: object, digits: int = 10) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}g}"
    return str(value)


def _classification_message(summary: dict[str, object]) -> None:
    code = summary["classification"]
    reason = str(summary.get("ct_reason", ""))
    if code == "EXACT_CT_COMPATIBLE":
        st.success(f"Exact CT compatibility: {reason}")
    elif code == "NOT_EXACT_NEAREST_DEGENERACY_DIAGNOSTIC_AVAILABLE":
        st.warning(
            "No exact CT compatibility. A nearest-degeneracy diagnostic plane can be "
            "constructed, but it is not an exact prediction."
        )
        if reason:
            st.caption(reason)
    else:
        st.info("No exact CT compatibility for the supplied state.")
        if reason:
            st.caption(reason)


def _render_results(response: object) -> None:
    result = response.result  # type: ignore[attr-defined]
    summary = result["summary"]

    st.divider()
    st.markdown("## Results")
    _classification_message(summary)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("λ₁", _format_scalar(summary["lambda1"], 12))
    m2.metric("λ₂", _format_scalar(summary["lambda2"], 12))
    m3.metric("λ₃", _format_scalar(summary["lambda3"], 12))
    m4.metric("|λ₂ − 1|", _format_scalar(summary["lambda2_residual"], 6))
    m5.metric("Stretch variants", str(summary["variant_count"]))

    tabs = st.tabs(
        [
            "Overview",
            "Metric / CT",
            "Ball–James",
            "Topology & variants",
            "Diagnostics",
            "Export",
        ]
    )

    with tabs[0]:
        left, right = st.columns([1.1, 1.0])
        with left:
            st.markdown("### Compatibility summary")
            overview = pd.DataFrame(
                [
                    ("CT exact compatible", summary["ct_exact_compatible"]),
                    ("CT degeneracy order", summary["degeneracy_order"]),
                    ("Exact CT habit planes", summary["ct_exact_habit_plane_count"]),
                    ("Ball–James λ₂ criterion", summary["ball_james_lambda2_exact"]),
                    ("Ball–James rank-one solutions", summary["ball_james_solution_count"]),
                    ("Topological variants", summary["topological_variant_count"]),
                    ("Operators", summary["operator_count"]),
                ],
                columns=["quantity", "value"],
            )
            st.dataframe(overview, hide_index=True, use_container_width=True)
        with right:
            st.markdown("### Principal stretches")
            stretch_df = pd.DataFrame(
                {
                    "principal stretch": ["λ₁", "λ₂", "λ₃"],
                    "value": [summary["lambda1"], summary["lambda2"], summary["lambda3"]],
                }
            )
            st.dataframe(stretch_df, hide_index=True, use_container_width=True)
            st.caption(
                "The compatibility threshold is controlled by the verified backend's "
                "numerical policy; this interface does not widen it."
            )

        ct_detail = result["ct_detail"]
        planes = ct_detail["exact_habit_planes_parent_covectors"]
        st.markdown("### CT habit-plane covectors")
        if planes:
            st.dataframe(
                pd.DataFrame(planes, columns=["p₁", "p₂", "p₃"]).rename_axis("solution"),
                use_container_width=True,
            )
            st.caption("Coordinates are parent-crystal covector components returned by the CT backend.")
        else:
            st.caption("No exact CT habit-plane covector is returned for this state.")

        approx = ct_detail["approximate_diagnostic"]
        if not ct_detail["exact_compatible"] and approx["candidate_planes_parent_covectors"]:
            with st.expander("Nearest-degeneracy diagnostic (not an exact solution)"):
                st.warning(approx["explanation"])
                st.write("Residual:", _format_scalar(approx["residual"], 8))
                st.dataframe(
                    pd.DataFrame(
                        approx["candidate_planes_parent_covectors"],
                        columns=["p₁", "p₂", "p₃"],
                    ),
                    use_container_width=True,
                )

    with tabs[1]:
        metric = result["metric"]
        st.markdown("### Metrics and correspondence compatibility")
        matrix_specs = [
            ("Parent metric  M_A", "parent_metric"),
            ("Product metric  M_M", "product_metric"),
            ("Pulled product metric  Cᵀ M_M C", "pulled_product_metric"),
            ("Dimensional CMC", "cmc_dimensional"),
            ("Normalized CMC", "cmc_normalized"),
            ("Dimensional SMC", "smc_dimensional"),
            ("Right stretch  U", "stretch"),
        ]
        for title, key in matrix_specs:
            with st.expander(title, expanded=key in {"cmc_dimensional", "stretch"}):
                st.dataframe(matrix_frame(metric[key]), use_container_width=True)
        st.markdown("### CT generalized spectrum")
        ct = result["ct_detail"]
        spectrum = pd.DataFrame(
            {
                "index": [1, 2, 3],
                "η = μ − 1": ct["eta_eigenvalues"],
                "μ": ct["generalized_mu"],
            }
        )
        st.dataframe(spectrum, hide_index=True, use_container_width=True)

    with tabs[2]:
        bj = result["ball_james_detail"]
        st.markdown("### Single-variant austenite / martensite rank-one solutions")
        st.write("λ₂ criterion satisfied:", bj["lambda2_exact"])
        if not bj["solutions"]:
            st.caption("No Ball–James single-variant rank-one solution was returned.")
        for index, solution in enumerate(bj["solutions"], start=1):
            with st.expander(f"Branch {solution['branch']} · solution {index}", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.write("Residual:", _format_scalar(solution["residual"], 8))
                    st.write("Shape vector a")
                    st.dataframe(pd.DataFrame([solution["shape_vector_a"]], columns=["a₁", "a₂", "a₃"]), hide_index=True, use_container_width=True)
                    st.write("Habit normal n")
                    st.dataframe(pd.DataFrame([solution["habit_normal_n"]], columns=["n₁", "n₂", "n₃"]), hide_index=True, use_container_width=True)
                with c2:
                    st.write("Rotation R")
                    st.dataframe(matrix_frame(solution["rotation"]), use_container_width=True)
                    st.write("Eigenvalues of C")
                    st.dataframe(pd.DataFrame([solution["eigenvalues_C"]], columns=["L₁", "L₂", "L₃"]), hide_index=True, use_container_width=True)

    with tabs[3]:
        topology = result["topology"]
        variants = result["stretch_variants"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Parent group order", topology["parent_group_order"])
        c2.metric("Product group order", topology["product_group_order"])
        c3.metric("Variants", topology["n_variants"])
        c4.metric("Operators", topology["n_operators"])

        summaries = topology.get("operator_summaries", [])
        if summaries:
            st.markdown("### Operator classes")
            st.dataframe(pd.DataFrame(summaries), hide_index=True, use_container_width=True)

        stretch_variants = variants.get("variants", [])
        if stretch_variants:
            st.markdown("### Stretch variants")
            selected = st.selectbox(
                "Variant",
                options=list(range(len(stretch_variants))),
                format_func=lambda idx: f"U{idx + 1}",
            )
            st.dataframe(matrix_frame(stretch_variants[selected]), use_container_width=True)

    with tabs[4]:
        diagnostics = result["diagnostics"]
        ct = result["ct_detail"]
        d1, d2, d3 = st.columns(3)
        d1.metric("CMC solver", diagnostics["solver_source"] or "—")
        d2.metric("Precision escalated", "Yes" if diagnostics["precision_escalated"] else "No")
        d3.metric("CT/BJ classification agreement", "Yes" if diagnostics["exact_classification_agreement_ct_vs_ball_james"] else "No")
        residual_table = pd.DataFrame(
            [
                ("Nearest CMC zero", ct["nearest_zero_residual"]),
                ("Generalized eigen equation", diagnostics["generalized_eigen_residual"]),
                ("Metric orthonormality", diagnostics["metric_orthonormality_residual"]),
                ("Representation parity", diagnostics["representation_parity_residual"]),
                ("Route disagreement", diagnostics["route_disagreement"]),
            ],
            columns=["diagnostic", "residual"],
        )
        st.dataframe(residual_table, hide_index=True, use_container_width=True)
        with st.expander("Theory-consistency audit"):
            st.json(result["contracts"])
        with st.expander("Full diagnostics JSON"):
            st.json(diagnostics)

    with tabs[5]:
        st.markdown("### Reproducible project")
        project_json = response.project_json()  # type: ignore[attr-defined]
        result_json = response.to_json()  # type: ignore[attr-defined]
        c1, c2 = st.columns(2)
        c1.download_button(
            "Download project JSON",
            data=project_json,
            file_name="cualni_ct_project.json",
            mime="application/json",
            use_container_width=True,
        )
        c2.download_button(
            "Download project + results JSON",
            data=result_json,
            file_name="cualni_ct_results.json",
            mime="application/json",
            use_container_width=True,
        )
        with st.expander("Project JSON"):
            st.code(project_json, language="json")
        with st.expander("Result JSON"):
            st.code(result_json, language="json")


st.markdown('<div class="app-kicker">Scientific crystallography workbench</div>', unsafe_allow_html=True)
st.markdown('<div class="app-title">CuAlNi-CT</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Enter the crystallographic state; the verified Python backend performs the calculation.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="science-note">No equation is evaluated in the browser/UI layer. Exact fractional correspondence entries are passed to the existing project loader and CalculationService.</div>',
    unsafe_allow_html=True,
)

point_groups = point_group_options()

with st.sidebar:
    st.markdown("## Project")
    project_id = st.text_input("Project ID", value="workbench_project")
    project_title = st.text_input("Title", value="Two-phase crystallography calculation")
    length_unit = st.selectbox("Length unit", ["angstrom", "nm", "pm"], index=0)
    st.caption("Use one length unit consistently for both phases.")

    st.divider()
    st.markdown("### Open project JSON")
    uploaded = st.file_uploader("Upload a CuAlNi-CT project", type=["json"], label_visibility="collapsed")
    run_uploaded = st.button("Calculate uploaded project", disabled=uploaded is None, use_container_width=True)
    if run_uploaded and uploaded is not None:
        try:
            payload = json.loads(uploaded.getvalue().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Top-level JSON must be an object.")
            st.session_state["last_response"] = calculate_payload(payload, source=uploaded.name)
            st.session_state.pop("last_error", None)
        except ApplicationError as exc:
            st.session_state["last_error"] = exc
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            st.session_state["last_error"] = internal_error(
                "The uploaded file is not a valid CuAlNi-CT project JSON.",
                detail=f"{type(exc).__name__}: {exc}",
            )

st.markdown("## Input")
with st.form("crystallography_input", clear_on_submit=False):
    parent_col, product_col = st.columns(2, gap="large")
    with parent_col:
        parent = phase_editor(
            prefix="parent",
            heading="Parent phase A",
            defaults=PhaseDefaults("phase_A", "Parent phase", "1", 5.0, 5.5, 6.0, 80.0, 90.0, 100.0),
            point_groups=point_groups,
            length_unit=length_unit,
        )
    with product_col:
        product = phase_editor(
            prefix="product",
            heading="Product phase M",
            defaults=PhaseDefaults("phase_M", "Product phase", "1", 5.0, 5.5, 6.0, 80.0, 90.0, 100.0),
            point_groups=point_groups,
            length_unit=length_unit,
        )

    st.divider()
    matrix_col, relation_col = st.columns([1.0, 1.1], gap="large")
    with matrix_col:
        correspondence = correspondence_editor()
    with relation_col:
        st.markdown("### Transformation")
        transformation_id = st.text_input("Transformation ID", value="A_to_M")
        transformation_label = st.text_input("Display name", value="Parent → Product")
        st.markdown("#### Interpretation")
        st.caption(
            "C(M←A) is entered explicitly and kept exact where possible. The app does "
            "not guess an orientation relationship, correspondence, or missing material data."
        )

    calculate = st.form_submit_button("Calculate transformation", type="primary", use_container_width=True)

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
            notes="Created interactively in the public CuAlNi-CT workbench.",
        )
        st.session_state["last_response"] = calculate_request(request)
        st.session_state.pop("last_error", None)
    except ApplicationError as exc:
        st.session_state["last_error"] = exc

if "last_error" in st.session_state:
    render_application_error(st.session_state["last_error"])

if "last_response" in st.session_state:
    _render_results(st.session_state["last_response"])
else:
    st.info(
        "Enter the two cells and the correspondence matrix, then choose **Calculate transformation**. "
        "The default values are a neutral identity demonstration, not a Cu–Al–Ni material preset."
    )
