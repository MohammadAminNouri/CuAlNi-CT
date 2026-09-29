from __future__ import annotations

"""State-safe research workspace layered on the verified scientific backends."""

import json
from typing import Any, Mapping

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import app.research_workspaces as rw
import app.research_workspaces_v4 as v4
import app.scientific_interpretation_v6 as interp
from app.phase1_contracts import classify_scientific_error, orientation_provenance_label, pole_series_metadata
from app.research_contracts import (
    human_branch_label,
    minimum_projective_separation_deg,
    normalize_atlas_semantics,
)
from app.session_state import get_bound, get_bound_envelope, fingerprint, put_bound
from app.state_persistence import restore_persistent_widget_state, snapshot_persistent_widget_state
from cualni_cryst.calculation_service import CalculationKind, CalculationService
from cualni_cryst.representation import CartesianConvention



# Replace only professor-facing interpretation hooks.
v4.am_existence_finding = interp.am_existence_finding
v4.mm_audit_finding = interp.mm_audit_finding
v4.ptmc_finding = interp.ptmc_finding
v4.or_comparison_finding = interp.or_comparison_finding
v4.cofactor_finding = interp.cofactor_finding
v4.supercompatibility_finding = interp.supercompatibility_finding
v4.experiment_finding = interp.experiment_finding


_LABELS = {
    "a_scale": "a scale",
    "b_scale": "b scale",
    "c_scale": "c scale",
    "alpha_offset_deg": "α offset (deg)",
    "beta_offset_deg": "β offset (deg)",
    "gamma_offset_deg": "γ offset (deg)",
}
_NEUTRAL = {
    "a_scale": 1.0,
    "b_scale": 1.0,
    "c_scale": 1.0,
    "alpha_offset_deg": 0.0,
    "beta_offset_deg": 0.0,
    "gamma_offset_deg": 0.0,
}


def _axis_controls(name: str, *, prefix: str, samples: int) -> tuple[float, float, int, np.ndarray]:
    is_scale = name.endswith("_scale")
    cols = st.columns(3)
    low = float(cols[0].number_input(
        f"{_LABELS[name]} minimum",
        value=0.98 if is_scale else -2.0,
        step=0.001 if is_scale else 0.1,
        format="%.8g",
        key=f"{prefix}_low_{name}",
    ))
    high = float(cols[1].number_input(
        f"{_LABELS[name]} maximum",
        value=1.02 if is_scale else 2.0,
        step=0.001 if is_scale else 0.1,
        format="%.8g",
        key=f"{prefix}_high_{name}",
    ))
    n = int(cols[2].number_input(
        f"{_LABELS[name]} samples",
        min_value=2,
        max_value=101,
        value=samples,
        step=1,
        key=f"{prefix}_n_{name}",
    ))
    if high <= low:
        raise ValueError(f"{_LABELS[name]} maximum must exceed minimum")
    if is_scale and low <= 0.0:
        raise ValueError(f"{_LABELS[name]} must remain positive")
    return low, high, n, np.linspace(low, high, n)


def _atlas_extra_metrics(states: tuple[tuple[str, Any, dict[str, Any]], ...], transformation_id: str) -> dict[str, dict[str, float]]:
    """Collect continuous native A/M proximity measures for the already-requested grid."""

    output: dict[str, dict[str, float]] = {}
    for state_id, project, _ in states:
        service = CalculationService(project)
        metric = service.compute(CalculationKind.METRIC_CORE, transformation_id)
        am = service.compute(CalculationKind.AM_COMPATIBILITY, transformation_id)
        output[state_id] = {
            "lambda2_residual": float(metric.lambda2_residual),
            "ct_nearest_degeneracy_residual": float(am.ct.analysis.nearest_zero_residual),
        }
    return output


def _render_atlas(project: Any, transformation_id: str, base_signature: str) -> None:
    v4.section_header(
        "Compatibility map",
        "If the product lattice changes slightly, where do exact CT compatibility, CT supercompatibility and the cofactor conditions approach or leave their criteria?",
        answer_hint="Controls remain visible. A result is current only when its full dimensionality/axes/ranges/samples/base-state signature matches the controls below.",
    )
    independent = rw.independent_product_lattice_parameters(project, transformation_id)
    mode = st.radio(
        "Sweep dimensionality",
        ["1D", "2D"],
        horizontal=True,
        key="research_atlas_dimension",
        disabled=len(independent) < 2,
    )
    if len(independent) < 2:
        mode = "1D"

    # Separate keys preserve both modes' settings independently.
    if mode == "1D":
        primary = st.selectbox("Sweep parameter", independent, format_func=lambda x: _LABELS[x], key="research_atlas_1d_axis1")
        low1, high1, n1, axis1 = _axis_controls(primary, prefix="research_atlas_1d_axis1", samples=21)
        secondary = None
        axes = {primary: axis1}
        controls = {"mode": mode, "axis1": primary, "low1": low1, "high1": high1, "n1": n1}
    else:
        primary = st.selectbox("First parameter", independent, format_func=lambda x: _LABELS[x], key="research_atlas_2d_axis1")
        choices = [name for name in independent if name != primary]
        secondary = st.selectbox("Second parameter", choices, format_func=lambda x: _LABELS[x], key="research_atlas_2d_axis2")
        low1, high1, n1, axis1 = _axis_controls(primary, prefix="research_atlas_2d_axis1", samples=11)
        low2, high2, n2, axis2 = _axis_controls(secondary, prefix="research_atlas_2d_axis2", samples=11)
        axes = {primary: axis1, secondary: axis2}
        controls = {
            "mode": mode, "axis1": primary, "low1": low1, "high1": high1, "n1": n1,
            "axis2": secondary, "low2": low2, "high2": high2, "n2": n2,
        }

    signature = fingerprint(base_signature, "compatibility-atlas-v6", controls)
    envelope = get_bound_envelope(st.session_state, "research_atlas_report_v6")
    if envelope is not None and envelope.signature != signature:
        st.warning("Settings changed — recalculate map. The previous map is retained for provenance but is not displayed as current.")

    count = int(np.prod([len(v) for v in axes.values()]))
    st.caption(f"{count} lattice state(s) will be evaluated; unselected independent parameters remain at the calculated base-state value.")
    if st.button("Calculate / update compatibility map", type="primary", use_container_width=True, key="research_v6_atlas_run"):
        try:
            with st.spinner("Evaluating the requested lattice-state map…"):
                states = rw.symmetry_preserving_product_grid(project, transformation_id, axes, max_states=2000)
                report = rw.evaluate_supercompatibility_atlas(states, transformation_id)
                report = normalize_atlas_semantics(report)
                continuous = _atlas_extra_metrics(states, transformation_id)
                put_bound(
                    st.session_state,
                    "research_atlas_report_v6",
                    signature,
                    {"report": report, "continuous": continuous, "controls": controls},
                )
        except Exception as exc:
            rw._render_failure("Compatibility map failed", exc)

    payload = get_bound(st.session_state, "research_atlas_report_v6", signature)
    if not isinstance(payload, Mapping):
        return
    report = payload["report"]
    continuous = payload["continuous"]

    counts = report.counts
    cols = st.columns(5)
    for col, name in zip(cols, ("both", "ct_only", "cofactor_only", "neither", "not_evaluable"), strict=True):
        col.metric(name.replace("_", " ").title(), int(counts.get(name, 0)))
    st.caption("NOT EVALUABLE includes states without an exact CT A/M seed; those states are not relabelled as failed CT supercompatibility.")

    rows: list[dict[str, Any]] = []
    for state in report.states:
        extra = continuous.get(state.state_id, {})
        row = {
            **{k: state.metadata.get(k) for k in axes},
            "classification": state.classification.value,
            "|λ₂−1|": extra.get("lambda2_residual"),
            "CT nearest-degeneracy residual": extra.get("ct_nearest_degeneracy_residual"),
            "CT supercompatibility residual": state.ct_best_supercompatibility_residual,
            "CC1 |residual|": state.cofactor_best_cc1_abs,
            "CC2 |residual|": state.cofactor_best_cc2_abs,
            "CC3 margin": state.cofactor_best_cc3_margin,
            "current state": all(abs(float(state.metadata.get(k, np.nan)) - _NEUTRAL[k]) <= 1e-12 for k in axes),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    st.markdown("### Continuous proximity measures")
    st.caption("For |λ₂−1|, CT nearest-degeneracy, CT supercompatibility, CC1 and CC2, zero is the exact target. CC3 is satisfied on the non-negative side of its native margin.")
    st.dataframe(v4.format_dataframe_scientific(frame), hide_index=True, use_container_width=True)

    if mode == "1D" and not frame.empty:
        metrics = ["|λ₂−1|", "CT nearest-degeneracy residual", "CC1 |residual|", "CC2 |residual|"]
        long = frame[[primary, *metrics]].melt(id_vars=[primary], var_name="metric", value_name="value").dropna()
        lines = alt.Chart(long).mark_line(point=True).encode(
            x=alt.X(f"{primary}:Q", title=_LABELS[primary]),
            y=alt.Y("value:Q", title="native residual / margin quantity"),
            color=alt.Color("metric:N", title="native quantity"),
            tooltip=[alt.Tooltip(f"{primary}:Q", format=".8g"), "metric:N", alt.Tooltip("value:Q", format=".6e")],
        )
        current_value = float(_NEUTRAL[primary])
        marker_data = pd.DataFrame({primary: [current_value]})
        marker = alt.Chart(marker_data).mark_rule(strokeDash=[6, 4], strokeWidth=2).encode(
            x=alt.X(f"{primary}:Q")
        )
        st.altair_chart((lines + marker).properties(height=360), use_container_width=True)
        st.caption(f"Dashed vertical marker = calculated current state ({_LABELS[primary]} = {current_value:.8g}); it is shown even when the exact current value falls between sampled nodes.")
    elif mode == "2D" and secondary is not None and not frame.empty:
        plot = frame.rename(columns={primary: "x", secondary: "y"})
        base = alt.Chart(plot).mark_circle(size=85).encode(
            x=alt.X("x:Q", title=_LABELS[primary]),
            y=alt.Y("y:Q", title=_LABELS[secondary]),
            color=alt.Color("classification:N", title="classification"),
            tooltip=[alt.Tooltip("x:Q", format=".8g"), alt.Tooltip("y:Q", format=".8g"), "classification:N"],
        )
        current = pd.DataFrame({"x": [float(_NEUTRAL[primary])], "y": [float(_NEUTRAL[secondary])], "label": ["calculated current state"]})
        marker = alt.Chart(current).mark_point(shape="diamond", size=240, filled=False, strokeWidth=3).encode(
            x="x:Q", y="y:Q", tooltip=["label:N", alt.Tooltip("x:Q", format=".8g"), alt.Tooltip("y:Q", format=".8g")]
        )
        st.altair_chart((base + marker).properties(height=420), use_container_width=True)
        st.caption("Diamond = calculated current state; it remains explicit even when the sweep grid does not sample that exact point.")

    with v4.audit_expander("Full atlas audit", expanded=False):
        raw = []
        for state in report.states:
            item = state.to_dict()
            raw.append(item)
        st.json(raw)
        for note in report.notes:
            st.caption(str(note))


def _pole_rows(
    families: list[Any],
    *,
    reference_phase_id: str,
    metadata_by_series: Mapping[str, Mapping[str, str]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata_by_series = metadata_by_series or {}
    for family in families:
        series = f"{family.source} · {family.label}"
        base_meta = pole_series_metadata(
            label=series,
            phase_id=str(family.phase_id),
            object_kind=str(family.kind),
            reference_phase_id=reference_phase_id,
            provenance=str(metadata_by_series.get(series, {}).get("OR provenance", "not required / same reference frame")),
            source=str(metadata_by_series.get(series, {}).get("theory source", family.source)),
            exact_status=str(metadata_by_series.get(series, {}).get("scientific status", "see source branch")),
        )
        for point in family.points:
            item = point.to_dict()
            item.update(base_meta)
            rows.append(item)
    return rows


def _render_poles(project: Any, transformation_id: str) -> None:
    v4.section_header(
        "Pole figures",
        "Do selected predicted planes/directions overlap geometrically in one explicit reference frame?",
        answer_hint="Changing a pole control makes the previous figure stale. Nothing regenerates until Generate / update is pressed.",
    )
    unified = rw._current_unified_report()
    if unified is None:
        st.info("Calculate the unified theory comparison first.")
        return

    transformation = project.transformation(transformation_id)
    parent_id = transformation.parent_phase_id
    product_id = transformation.product_phase_id
    mode = st.radio("Pole analysis", ["Predicted interfaces / twins", "OR pole overlay"], horizontal=True, key="research_v6_pole_mode")
    families: list[Any] = []
    series_metadata: list[dict[str, str]] = []
    signature_parts: list[Any] = [st.session_state.get("research_current_unified_signature"), transformation_id, mode]

    if mode == "Predicted interfaces / twins":
        object_kind = st.selectbox("Physical object", ["A/M habit plane", "Twin plane", "Twin direction"], key="research_v6_pole_kind")
        candidates = []
        for row in unified.rows:
            coeffs, kind = rw._row_object(row, object_kind)
            if coeffs is not None:
                candidates.append((row, coeffs, kind))
        labels = {human_branch_label(row): (row, coeffs, kind) for row, coeffs, kind in candidates}
        selected = st.multiselect(
            "Branches",
            options=list(labels),
            default=[],
            max_selections=12,
            key="research_v6_pole_branches",
            help="No candidate is preselected: CT closing-gap candidates are dynamic and are not ranked as a preferred natural OR.",
        )
        signature_parts += [object_kind, selected]
        for label in selected:
            row, coeffs, kind = labels[label]
            exact = getattr(row, "exact", None)
            family_label = label
            if str(getattr(getattr(row, "prediction_kind", None), "value", "")) == "ct_am_habit" and exact is False:
                family_label = "Approximate CT diagnostic plane — not an exact A/M solution"
            source = str(getattr(getattr(row, "theory", None), "value", getattr(row, "theory", "")))
            family = rw.pole_family(
                project,
                parent_id,
                kind=kind,
                coefficients=coeffs,
                reference_phase_id=parent_id,
                projective=True,
                label=family_label,
                source=source,
            )
            families.append(family)
            series_metadata.append(pole_series_metadata(
                label=f"{source} · {family_label}",
                phase_id=parent_id,
                object_kind=str(kind),
                reference_phase_id=parent_id,
                provenance="no OR applied; object is expressed natively in the parent crystal/reference frame",
                source=str(getattr(row, "provenance", "") or source),
                exact_status="exact" if exact is True else "approximate diagnostic" if exact is False else "status not encoded",
            ))
    else:
        c1, c2 = st.columns(2)
        pole_kind = c1.radio("Product object type", ["direction", "plane"], horizontal=True, key="research_v6_or_pole_kind")
        seed_text = c2.text_input("Product indices", value="0 0 1", key="research_v6_or_pole_seed")
        try:
            seed = rw._parse_vector(seed_text, name="product pole")
        except ValueError as exc:
            st.error(str(exc))
            return
        or_rows = [row for row in unified.rows if row.or_parent_from_product is not None]
        labels = {human_branch_label(row): row for row in or_rows}
        selected = st.multiselect(
            "Orientation relationships",
            options=list(labels),
            default=[],
            max_selections=12,
            key="research_v6_or_pole_branches",
            help="No OR is preselected or presented as preferred.",
        )
        signature_parts += [pole_kind, seed_text, selected]
        for label in selected:
            row = labels[label]
            R = rw._reexpress_or(
                project,
                transformation_id,
                np.asarray(row.or_parent_from_product, dtype=float),
                source=CartesianConvention.PTCLAB_A_X_C_XZ,
                target=CartesianConvention.SYMMETRIC_METRIC,
            )
            source = str(getattr(getattr(row, "theory", None), "value", getattr(row, "theory", "")))
            family = rw.pole_family(
                project,
                product_id,
                kind=pole_kind,
                coefficients=seed,
                reference_phase_id=parent_id,
                R_reference_from_phase=R,
                projective=True,
                label=label,
                source=source,
            )
            families.append(family)
            series_metadata.append(pole_series_metadata(
                label=f"{source} · {label}",
                phase_id=product_id,
                object_kind=pole_kind,
                reference_phase_id=parent_id,
                provenance=orientation_provenance_label({
                    "theory_origin": getattr(row, "provenance", ""),
                    "definition": getattr(row, "rotation_role", ""),
                    "orientation_id": getattr(row, "row_id", ""),
                }),
                source=str(getattr(row, "provenance", "") or source),
                exact_status="exact" if getattr(row, "exact", None) is True else "approximate diagnostic" if getattr(row, "exact", None) is False else "status not encoded",
            ))

    signature = fingerprint("pole-figure-v6", *signature_parts)
    envelope = get_bound_envelope(st.session_state, "research_pole_report_v6")
    if envelope is not None and envelope.signature != signature:
        st.warning("Pole settings changed — regenerate the figure. The previous figure is retained but is not displayed as current.")

    if st.button("Generate / update pole figure", type="primary", disabled=not families, use_container_width=True, key="research_v6_pole_generate"):
        meta_map = {item["series"]: item for item in series_metadata}
        rows = _pole_rows(families, reference_phase_id=parent_id, metadata_by_series=meta_map)
        put_bound(st.session_state, "research_pole_report_v6", signature, {"rows": rows, "mode": mode, "series_metadata": series_metadata, "reference_phase_id": parent_id})

    payload = get_bound(st.session_state, "research_pole_report_v6", signature)
    if not isinstance(payload, Mapping):
        return
    rows = list(payload["rows"])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    st.altair_chart(rw._stereographic_chart(frame), use_container_width=False)
    best, pair = minimum_projective_separation_deg(rows)
    if best is not None and pair is not None:
        st.markdown("### Geometric overlap check")
        st.write(f"Closest projective pole separation: **{best:.6g}°** between **{pair[0]}** and **{pair[1]}**.")
        st.caption("A small angular separation shows geometric overlap in this reference frame. It does not prove identical derivation, mechanism, branch identity or theory equivalence.")
    metadata = list(payload.get("series_metadata", []))
    if metadata:
        st.markdown("### Pole-series provenance")
        st.dataframe(pd.DataFrame(metadata), hide_index=True, use_container_width=True)
    with st.expander("How this pole figure is constructed", expanded=False):
        st.markdown("**Theory / construction used — metric-tensor crystallography + projective stereographic projection.**")
        st.latex(r"B^TB=M,\qquad r=B\,u,\qquad n=B^{-T}p")
        st.write("u=[uvw] is a direct-space direction; p=(hkl) is a reciprocal plane covector; B is the phase structure/metric factor. Directions and plane normals are therefore not identified by a cubic shortcut in a non-cubic phase.")
        st.latex(r"(x,y)=\left(\frac{v_x}{1+v_z},\frac{v_y}{1+v_z}\right),\qquad v\sim -v")
        st.write("After mapping into the declared reference Cartesian frame, the backend selects the upper-hemisphere representative and applies the projective convention v ~ -v before the stereographic map.")
        st.write("**Backend mapping:** `cualni_cryst.pole_figure.pole_family` constructs metric-correct Cartesian poles; `app.research_workspaces._stereographic_chart` renders their stored stereographic coordinates.")
        st.write("**Verbal version:** first convert the crystal indices to the correct physical direction or plane normal under the real metric, then express it in one reference frame, identify antipodes, and project the upper hemisphere onto the plane.")
    with v4.audit_expander("Pole coordinates / provenance", expanded=False):
        st.dataframe(frame, hide_index=True, use_container_width=True)
        st.caption("Stereographic projection uses the upper-hemisphere projective convention n ~ −n for unoriented crystallographic poles.")
    st.download_button("Pole coordinates CSV", data=frame.to_csv(index=False).encode("utf-8"), file_name="cualni_ct_poles.csv", mime="text/csv", key="research_v6_pole_csv")


def _render_independent(project: Any, transformation_id: str, base_signature: str) -> None:
    v4.section_header(
        "Invariant-line & double-shear checks",
        "What additional compatibility is obtained by an independent invariant-line condition or by two lattice-invariant shears?",
        answer_hint="Results are signature-bound inside each solver; stale cached summaries are never surfaced as current.",
    )
    with st.expander("How the independent invariant-line and double-shear solvers work", expanded=False):
        st.markdown("**Invariant-line construction.**")
        st.latex(r"\det(R(\phi)D-I)=0")
        st.write("D is the explicit two-dimensional correspondence deformation built from the two correlated vector pairs. R(φ) is the remaining proper rotation about the declared plane normal (direct-space mode) or zone axis (reciprocal-space mode). Every real branch is retained and independently checked. The PTCLab manual motivates the input/output structure; the repository does not claim unpublished PTCLab source-equation identity.")
        st.markdown("**Double-shear / two-LIS construction.**")
        st.latex(r"F_{pre}=U+p_1 a_1\otimes n_1+p_2 a_2\otimes n_2")
        st.write("This is the backend's `additive_laminate` composition.")
        st.latex(r"A_i=a_i\otimes n_i,\quad K_i=A_iU^{-1},\quad F_{pre}=(I+f_2K_2)(I+f_1K_1)U")
        st.write("This is the alternative `sequential_simple_shear` composition. The second parameter is fixed and the first is solved from")
        st.latex(r"\det(F_{pre}^{T}F_{pre}-\delta^2 I)=0")
        st.write("Real roots are checked against the middle singular value, and habit branches are then obtained independently from the Ball–James rank-one solver. Parameters are called three-variant fractions only when both rank-one increments carry explicit common-base variant provenance; otherwise they remain generic coefficients.")
        st.write("**Verbal version:** the invariant-line solver closes one planar eigenvalue at unity after the allowed rotation; the double-shear solver composes two explicitly declared LIS increments and then asks whether the resulting macroscopic deformation has an invariant-plane connection.")
    rw._render_invariant_and_double(project, transformation_id, base_signature)


def _render_ebsd(project: Any, transformation_id: str, base_signature: str) -> None:
    v4.section_header(
        "EBSD ↔ theory",
        "Which theory branches approach the measured orientations, boundaries and traces after every convention and threshold is explicit?",
        answer_hint="No convention, angle unit, phase mapping or trace surface normal is guessed.",
    )
    rw._render_ebsd(project, transformation_id, base_signature)


def _phase1_failure(title: str, exc: Exception) -> None:
    info = classify_scientific_error(exc)
    st.error(f"{info['title']}: {info['message']}")
    st.caption(info["hint"])
    if info.get("technical_detail"):
        with st.expander("Technical detail", expanded=False):
            st.code(info["technical_detail"])


# Install specific, non-destructive research error semantics.
rw._render_failure = _phase1_failure


# Install orchestration patches.
v4._render_atlas = _render_atlas
v4._render_poles = _render_poles
v4._render_independent = _render_independent
v4._render_ebsd = _render_ebsd


def render_research_extension() -> None:
    restore_persistent_widget_state(st.session_state)
    try:
        v4.render_research_extension()
    finally:
        snapshot_persistent_widget_state(st.session_state)
