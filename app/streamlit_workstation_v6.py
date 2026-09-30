from __future__ import annotations

"""Frozen-contract V6 professor-facing workstation layered on the verified V4 UI."""

import copy
import math

import numpy as np
import pandas as pd
import streamlit as st

import app.scientific_interpretation_v6 as interp
from app.phase1_contracts import (
    axis_angle_display,
    classify_scientific_error,
    metric_geometry_audit,
    orientation_provenance_label,
    proper_rotation_order,
)
import app.streamlit_workstation as legacy
import app.streamlit_workstation_v4 as v4
from app.session_state import (
    fingerprint,
    mark_calculation_failure,
    mark_calculation_success,
    observe_draft,
    record_draft,
)
from app.state_persistence import (
    restore_persistent_widget_state,
    snapshot_persistent_widget_state,
)


# Patch only interpretation/presentation hooks.  Scientific solvers remain frozen.
v4.transformation_findings = interp.transformation_findings
v4.topology_finding = interp.topology_finding
v4.orientation_finding = interp.orientation_finding
v4.martensite_pair_finding = interp.martensite_pair_finding
v4.ebsd_audit_finding = interp.ebsd_audit_finding


# Preserve the verified renderers and harden only their professor-facing semantics.
_legacy_render_application_error = legacy.render_application_error
_legacy_render_orientation = legacy._render_orientation
_v4_render_geometry_workspace = v4._render_geometry_workspace


def _render_application_error_phase1(exc: BaseException) -> None:
    classified = classify_scientific_error(exc)
    st.error(f"{classified['title']}: {classified['message']}")
    st.caption(classified["hint"])
    if classified.get("technical_detail"):
        with st.expander("Technical error audit", expanded=False):
            st.code(classified["technical_detail"])
            st.caption(f"Error class: {classified['code']}. The draft and last successful calculated state are preserved.")


def _render_orientation_phase1(analysis: object) -> None:
    if not isinstance(analysis, dict):
        _legacy_render_orientation(analysis)
        return
    safe = copy.deepcopy(analysis)
    report = safe.get("report", {})
    state = safe.get("state", {})
    if isinstance(report, dict):
        aa = report.get("axis_angle", {})
        if isinstance(aa, dict):
            shown = axis_angle_display(aa.get("axis", (1.0, 0.0, 0.0)), float(aa.get("angle_deg", 0.0)))
            aa["axis"] = shown["axis_text"]
            aa["angle_deg"] = shown["angle_deg"]
    variants = safe.get("variants", [])
    if isinstance(variants, list):
        for item in variants:
            if not isinstance(item, dict):
                continue
            for key in ("forward_axis_angle", "reverse_axis_angle"):
                value = item.get(key)
                if isinstance(value, dict):
                    shown = axis_angle_display(value.get("axis", (1.0, 0.0, 0.0)), float(value.get("angle_deg", 0.0)))
                    value["axis"] = shown["axis_text"]
                    value["angle_deg"] = shown["angle_deg"]
    provenance = orientation_provenance_label(state if isinstance(state, dict) else {}, safe)
    existing = str(safe.get("origin_note", "")).strip()
    safe["origin_note"] = f"Provenance: {provenance}." + (f" {existing}" if existing else "")
    _legacy_render_orientation(safe)
    with st.expander("How R is interpreted", expanded=False):
        st.latex(r"x_A=R_{A\leftarrow M}x_M,\qquad R_{M\leftarrow A}=R_{A\leftarrow M}^{T}")
        st.latex(r"u_M=C_{M\leftarrow A}u_A,\qquad p_M=C_{M\leftarrow A}^{-T}p_A,\qquad C\neq R")
        st.write("R is a proper Cartesian rotation; C is a lattice-coordinate correspondence. A zero rotation has no unique physical axis, so its axis is reported as undefined/conventional. A near-rotation is projected to SO(3) only when the explicit repair option is selected; that correction residual remains in the provenance note.")


def _render_geometry_workspace_phase1() -> None:
    _v4_render_geometry_workspace()
    try:
        payload, _ = legacy._require_project()
        parent_name, product_name, parent_id, product_id = legacy._phase_names(payload)
        phase = str(st.session_state.get("calpad_phase", parent_id))
        if phase not in {parent_id, product_id}:
            return
        cell = v4.calpad_phase_cell(payload, phase)
    except Exception:
        return

    symbol = str(cell.get("point_group_symbol", ""))
    full_order = int(cell.get("symmetry_order", 0))
    try:
        proper_order = proper_rotation_order(symbol)
    except Exception:
        proper_order = 0
    st.markdown("### Metric / symmetry interpretation")
    v4.compact_metrics(
        [
            ("Full point-group order", full_order),
            ("Proper rotational subgroup order", proper_order),
            ("M·M⁻¹ residual", v4.sci(cell.get("metric_inverse_residual"), 3)),
        ],
        columns=3,
    )
    if symbol == "m-3m":
        st.caption("For cubic m-3m this means 48 full crystallographic operations and 24 proper rotations. CT topology may need the full group; physical OR/disorientation minimization uses the proper subgroup.")

    M = np.asarray(cell["direct_metric"], dtype=float)
    check = metric_geometry_audit(M, (1, 0, 1), (1, 0, 1))
    r = np.asarray(check["direction_cartesian"], dtype=float)
    n = np.asarray(check["plane_normal_cartesian"], dtype=float)
    angle = math.degrees(math.acos(float(np.clip(abs(np.dot(r / np.linalg.norm(r), n / np.linalg.norm(n))), -1.0, 1.0))))
    with st.expander("How direct and reciprocal geometry is reconstructed", expanded=False):
        st.latex(r"\|u\|^2=u^TMu,\qquad \|p\|_*^2=p^TM^{-1}p")
        st.latex(r"r=B u,\qquad n=B^{-T}p,\qquad B^TB=M")
        st.write("u=[uvw] is a direct direction and p=(hkl) is a reciprocal plane covector. The physical normal to (hkl) is B⁻ᵀp, not generally Bp. Therefore [101] and (101) are different geometric objects in a monoclinic cell even though they share the same integer triplet.")
        st.write("**Current-phase [101] direction (Cartesian):**", [f"{x:.8e}" for x in r])
        st.write("**Current-phase (101) physical normal (Cartesian):**", [f"{x:.8e}" for x in n])
        st.write(f"Projective angle between them: **{angle:.9g}°**. In a cubic metric this can collapse to a familiar shortcut; in a non-cubic metric the code never assumes that shortcut.")
        st.write("**Backend mapping:** CalPad/typed crystal objects use the registered direct metric and reciprocal metric for conversion, angles, incidence and low-index ranking. Projective ± equivalence is an object convention, not a {hkl} family declaration.")
        st.write("**Verbal version:** directions live in the direct metric, plane normals live in the reciprocal metric, and the metric—not matching index text—decides physical parallelism.")


legacy.render_application_error = _render_application_error_phase1
legacy._render_orientation = _render_orientation_phase1
v4._render_geometry_workspace = _render_geometry_workspace_phase1

def _render_setup(title: str, length_unit: str, project_id: str) -> None:
    v4.section_header(
        "State definition",
        "What parent lattice, product lattice and lattice correspondence define the transformation?",
        answer_hint="Nothing is inferred from alloy names. The editable draft is preserved independently of the last successful calculation.",
    )
    point_groups = legacy.point_group_options()
    left, right = st.columns(2, gap="large")
    with left:
        parent = v4.smart_phase_editor(
            prefix="parent",
            heading="Parent phase (A)",
            defaults=v4.PhaseDefaults("phase_A", "Parent phase", "m-3m", 5.8, 5.8, 5.8, 90, 90, 90, "conventional cubic"),
            point_groups=point_groups,
            length_unit=length_unit,
        )
    with right:
        product = v4.smart_phase_editor(
            prefix="product",
            heading="Product / daughter phase (M)",
            defaults=v4.PhaseDefaults("phase_M", "Product phase", "2/m", 4.4, 5.3, 13.8, 90, 100, 90, "conventional unique-b"),
            point_groups=point_groups,
            length_unit=length_unit,
        )

    with st.expander("Lattice correspondence C and transformation metadata", expanded=True):
        c1, c2 = st.columns([1.1, 1.0], gap="large")
        with c1:
            correspondence = v4.correspondence_editor()
        with c2:
            st.markdown("**Meaning**")
            st.write("C maps parent lattice directions to product lattice directions. It is not a physical orientation matrix R.")
            transformation_id = st.text_input("Transformation ID", value="A_to_M", key="transformation_id")
            transformation_label = st.text_input("Display name", value="Parent → Product", key="transformation_label")

    snapshot = v4._state_snapshot(
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
    record_draft(st.session_state, snapshot, signature)

    if st.session_state.pop("pending_accept_loaded_draft", False):
        st.session_state["calculated_draft_signature"] = signature
        st.session_state["requires_recalculation"] = False

    stale = observe_draft(st.session_state, signature)
    has_calculated = st.session_state.get("current_response") is not None
    if stale and has_calculated:
        st.warning("Draft changed — recalculation required. The last successful calculated state is preserved for provenance, but it is not presented as current.")

    label = "Recalculate crystallographic state" if has_calculated else "Calculate crystallographic state"
    if st.button(label, type="primary", use_container_width=True, key="workbench_v6_calculate"):
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
            st.rerun()
        except legacy.ApplicationError as exc:
            # The editable draft and previous successful state are intentionally kept.
            mark_calculation_failure(st.session_state, exc)
            legacy.render_application_error(exc)

    response = st.session_state.get("current_response")
    if response is not None and not st.session_state.get("requires_recalculation", False):
        summary = response.result["summary"]
        v4.compact_metrics(
            [
                ("λ₂", legacy._fmt(summary["lambda2"], 9)),
                ("|λ₂−1|", v4.sci(summary["lambda2_residual"], 3)),
                ("Stretch variants", summary["variant_count"]),
                ("Operator classes", summary["operator_count"]),
            ],
            columns=4,
        )
        st.caption("Calculated and editable states are separate. Navigation does not recreate this draft from defaults.")


def _show_approximate_ct_planes(ct: object) -> bool:
    """Return True only when a genuinely non-exact CT diagnostic should be shown.

    Exact CMC states can still carry a backend diagnostic payload for audit/provenance.
    That payload must not be rendered as an approximate competing solution when exact
    compatibility has already been established.
    """

    if not isinstance(ct, dict) or bool(ct.get("exact_compatible", False)):
        return False
    approx = ct.get("approximate_diagnostic", {})
    return isinstance(approx, dict) and bool(
        approx.get("candidate_planes_parent_covectors")
    )


def _safe_transformation_audit(response: object) -> None:
    """Professor-facing audit without raw booleans or approximate/exact conflation."""

    result = response.result  # type: ignore[attr-defined]
    summary = result["summary"]
    ct = result["ct_detail"]
    bj = result["ball_james_detail"]
    st.markdown("#### Status")
    v4.compact_metrics(
        [
            ("CT exact A/M", "satisfied" if summary["ct_exact_compatible"] else "not satisfied"),
            ("Ball–James exact A/M", "satisfied" if bj["lambda2_exact"] else "not satisfied"),
            ("Stretch variants", summary["variant_count"]),
            ("Operator classes", summary["operator_count"]),
        ],
        columns=4,
    )
    st.markdown("#### Native residuals")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "theory": "Cayron CT",
                    "native quantity": "nearest normalized-CMC degeneracy residual",
                    "value": v4.sci(ct["nearest_zero_residual"], 6),
                    "status": "exact" if ct["exact_compatible"] else "outside exact degeneracy",
                },
                {
                    "theory": "Ball–James",
                    "native quantity": "|λ₂−1|",
                    "value": v4.sci(summary["lambda2_residual"], 6),
                    "status": "exact" if bj["lambda2_exact"] else "outside exact criterion",
                },
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption("These residuals answer related compatibility questions but are native quantities of different formulations; they are not equated numerically.")

    exact_planes = ct.get("exact_habit_planes_parent_covectors", [])
    approx = ct.get("approximate_diagnostic", {})
    st.markdown("#### Exact CT A/M habit-plane covectors m_A")
    if exact_planes:
        st.dataframe(
            pd.DataFrame(exact_planes, columns=["m₁", "m₂", "m₃"]),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Habit-plane covectors are projective: m_A and −m_A represent the same physical plane."
        )
    else:
        st.write("Exact CT A/M habit planes: none.")

    if _show_approximate_ct_planes(ct):
        with st.expander("Approximate CT diagnostic planes — not exact A/M solutions", expanded=False):
            st.warning(
                "Nearest-degeneracy diagnostic only. These planes are never used as exact "
                "A/M solutions or as exact supercompatibility seeds."
            )
            st.write("Diagnostic residual:", v4.sci(approx.get("residual"), 6))
            st.dataframe(
                pd.DataFrame(
                    approx["candidate_planes_parent_covectors"],
                    columns=["m₁ (diag)", "m₂ (diag)", "m₃ (diag)"],
                ),
                hide_index=True,
                use_container_width=True,
            )

    with st.expander("Raw matrices / solver audit", expanded=False):
        metric = result["metric"]
        for name, key in (
            ("Parent metric M_A", "parent_metric"),
            ("Product metric M_M", "product_metric"),
            ("Normalized CMC", "cmc_normalized"),
            ("Right stretch U", "stretch"),
        ):
            st.markdown(f"**{name}**")
            st.dataframe(legacy.matrix_frame(metric[key]), use_container_width=True)


# Install v5 orchestration hooks into v4.
v4._render_setup = _render_setup
legacy._render_transformation = _safe_transformation_audit
legacy._render_orientation = _render_orientation_phase1


def main() -> None:
    # Accessing session_state here is safe and happens before any widget is created.
    restore_persistent_widget_state(st.session_state)
    try:
        v4.main()
    finally:
        snapshot_persistent_widget_state(st.session_state)


if __name__ == "__main__":
    main()
