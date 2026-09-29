
from __future__ import annotations

"""Research workspaces layered on the stable Streamlit workstation.

The existing UI remains responsible for ProjectState editing and ordinary
calculations.  This module exposes the research question that motivated the
repository: branch-resolved CT/Ball--James/PTMC/experiment comparison, plus the
few supporting tools needed to interrogate that comparison.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app.session_state import fingerprint, get_bound, put_bound

from cualni_cryst.ebsd_pipeline import run_pipeline
from cualni_cryst.invariant_line import (
    solve_direct_invariant_line,
    solve_reciprocal_invariant_line,
)
from cualni_cryst.orientation import OrientationService
from cualni_cryst.pole_figure import pole_family
from cualni_cryst.project_io import project_from_dict
from cualni_cryst.representation import CartesianConvention
from cualni_cryst.ptmc_double_shear import (
    DoubleShearComposition,
    DoubleShearParameterSemantics,
    DoubleShearSystem,
    DoubleShearSystemSource,
    solve_double_shear_ptmc,
)
from cualni_cryst.supercompatibility_atlas import (
    evaluate_supercompatibility_atlas,
    independent_product_lattice_parameters,
    symmetry_preserving_product_grid,
    uniform_product_scale_sweep,
)
from cualni_cryst.theory_equivalence import (
    AssignmentScales,
    EquivalenceTolerances,
    MatchFamily,
    PhysicalBranchMatcher,
    build_equivalence_report,
    double_shear_comparison_rows,
)
from cualni_cryst.theory_unified import (
    ExperimentalObservation,
    PTMCMode,
    PredictionKind,
    TheoryComparisonAdapter,
    TheoryKind,
)


def _render_failure(title: str, exc: Exception) -> None:
    """Show a concise failure without dumping a traceback into the workstation."""

    st.error(f"{title}: {exc}")
    with st.expander("Technical detail", expanded=False):
        st.code(f"{type(exc).__name__}: {exc}")


def _base_signature(payload: Mapping[str, Any], transformation_id: str) -> str:
    return fingerprint(
        "ct-equivalence-laboratory-v2",
        st.session_state.get("calculated_draft_signature"),
        payload,
        transformation_id,
    )


def _bound_get(key: str, signature: str) -> Any | None:
    return get_bound(st.session_state, key, signature)


def _bound_put(key: str, signature: str, value: Any) -> None:
    put_bound(st.session_state, key, signature, value)


def _agreement_text(value: bool | None) -> str:
    if value is True:
        return "within tolerance"
    if value is False:
        return "outside tolerance"
    return "not comparable"


def _current_unified_report() -> Any | None:
    signature = st.session_state.get("research_current_unified_signature")
    if not isinstance(signature, str):
        return None
    return _bound_get("research_unified_report", signature)


def _current_equivalence_report() -> Any | None:
    signature = st.session_state.get("research_current_unified_signature")
    if not isinstance(signature, str):
        return None
    return _bound_get("research_equivalence_report", signature)


def _stereographic_chart(frame: pd.DataFrame) -> alt.Chart:
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    circle = pd.DataFrame({"x": np.cos(theta), "y": np.sin(theta)})
    horizontal = pd.DataFrame({"x": [-1.0, 1.0], "y": [0.0, 0.0]})
    vertical = pd.DataFrame({"x": [0.0, 0.0], "y": [-1.0, 1.0]})

    base = alt.Chart(circle).mark_line().encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[-1.05, 1.05]), axis=None),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[-1.05, 1.05]), axis=None),
    )
    axes = alt.layer(
        alt.Chart(horizontal).mark_line(opacity=0.25).encode(x="x:Q", y="y:Q"),
        alt.Chart(vertical).mark_line(opacity=0.25).encode(x="x:Q", y="y:Q"),
    )
    points = (
        alt.Chart(frame)
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-1.05, 1.05]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-1.05, 1.05]), axis=None),
            color=alt.Color("series:N", title="Series"),
            tooltip=[
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("phase:N", title="Phase"),
                alt.Tooltip("source:N", title="Source"),
                alt.Tooltip("x:Q", format=".6f"),
                alt.Tooltip("y:Q", format=".6f"),
            ],
        )
    )
    return (base + axes + points).properties(width=620, height=620)


def _parse_vector(text: str, *, name: str) -> tuple[float, float, float]:
    body = text.strip().replace("[", " ").replace("]", " ").replace("(", " ").replace(")", " ")
    tokens = body.replace(",", " ").split()
    if len(tokens) != 3:
        raise ValueError(f"{name} requires exactly three numbers")
    values = tuple(float(token) for token in tokens)
    if not all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    if float(np.linalg.norm(values)) <= 1.0e-15:
        raise ValueError(f"{name} must be nonzero")
    return values  # type: ignore[return-value]


def _parse_optional_vector(text: str, *, name: str):
    return None if not text.strip() else _parse_vector(text, name=name)


def _parse_matrix(text: str, *, name: str) -> tuple[tuple[float, float, float], ...]:
    rows = [row.strip() for row in text.strip().splitlines() if row.strip()]
    if len(rows) != 3:
        raise ValueError(f"{name} requires exactly three rows")
    matrix = []
    for index, row in enumerate(rows):
        matrix.append(_parse_vector(row, name=f"{name} row {index + 1}"))
    return tuple(matrix)


def _scalar_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    safe_rows = []
    for row in rows:
        safe = {}
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple)):
                safe[key] = json.dumps(value, default=str, sort_keys=True)
            else:
                safe[key] = value
        safe_rows.append(safe)
    return pd.DataFrame(safe_rows)


def _match_rows(report) -> list[dict[str, Any]]:
    rows = []
    for item in report.matches:
        r = item.residuals
        rows.append(
            {
                "family": item.family.value,
                "left theory": item.left_theory.value,
                "left branch": item.left_branch,
                "right theory": item.right_theory.value,
                "right branch": item.right_branch,
                "habit Δ (deg)": r.habit_plane_angle_deg,
                "shape Δ projective (deg)": r.shape_direction_projective_deg,
                "shape Δ oriented (deg)": r.shape_direction_oriented_deg,
                "shape magnitude rel.": r.shape_magnitude_relative,
                "rank-one tensor rel.": r.rank_one_tensor_relative,
                "twin-plane Δ (deg)": r.twin_plane_angle_deg,
                "twin-direction Δ (deg)": r.twin_direction_angle_deg,
                "shear rel.": r.shear_relative,
                "OR disorientation (deg)": r.or_disorientation_deg,
                "comparison": item.completeness.value,
                "agreement": _agreement_text(
                    item.all_required_components_within_tolerance
                ),
                "missing required": ", ".join(item.missing_required_components),
                "components within tolerance": json.dumps(
                    item.within_tolerance, sort_keys=True
                ),
            }
        )
    return rows


def _experiment_rows(report) -> list[dict[str, Any]]:
    rows = []
    for item in report.experiment_residuals:
        r = item.residuals
        rows.append(
            {
                "observation": item.observation_row_id,
                "theory": item.theory.value,
                "branch": item.theory_branch,
                "habit Δ (deg)": r.habit_plane_angle_deg,
                "twin-plane Δ (deg)": r.twin_plane_angle_deg,
                "twin-direction Δ (deg)": r.twin_direction_angle_deg,
                "shape Δ oriented (deg)": r.shape_direction_oriented_deg,
                "shear rel.": r.shear_relative,
                "shape magnitude rel.": r.shape_magnitude_relative,
                "OR disorientation (deg)": r.or_disorientation_deg,
                "uncertainty-normalized": (
                    json.dumps(item.uncertainty_normalized, sort_keys=True)
                    if item.uncertainty_normalized
                    else ""
                ),
            }
        )
    return rows


def _reexpress_or(
    project,
    transformation_id: str,
    matrix: np.ndarray,
    *,
    source: CartesianConvention,
    target: CartesianConvention,
) -> np.ndarray:
    """Re-express one physical OR without changing the physical mapping."""
    transformation = project.transformation(transformation_id)
    service = OrientationService(project)
    state = service.state_from_matrix(
        transformation.parent_phase_id,
        transformation.product_phase_id,
        np.asarray(matrix, dtype=float),
        orientation_id="research_frame_bridge",
        reference_convention=source,
        moving_convention=source,
        transformation_id=transformation_id,
    )
    converted = service.reexpress(state, target, target)
    return np.asarray(converted.R_reference_from_moving, dtype=float)


def _build_experiments(project, transformation_id: str, base_signature: str) -> tuple[ExperimentalObservation, ...]:
    observations = []
    ebsd_completed = _bound_get("research_ebsd_experiment_or", base_signature)
    include_ebsd = False
    if isinstance(ebsd_completed, Mapping):
        file_name = str(ebsd_completed.get("file_name", "EBSD map"))
        file_hash = str(ebsd_completed.get("file_sha256", ""))
        short_hash = file_hash[:12] if file_hash else "unhashed"
        include_ebsd = st.checkbox(
            f"Include completed EBSD OR — {file_name} · {short_hash}",
            value=True,
            key="research_include_ebsd_or",
            help=(
                "This is the final OR from a completed EBSD run. Its file hash and run "
                "signature are retained in provenance; editing EBSD controls does not "
                "silently relabel that completed result."
            ),
        )
    if include_ebsd and isinstance(ebsd_completed, Mapping):
        ebsd_matrix = np.asarray(ebsd_completed["matrix"], dtype=float)
        ebsd_ptclab = _reexpress_or(
            project,
            transformation_id,
            ebsd_matrix,
            source=CartesianConvention.SYMMETRIC_METRIC,
            target=CartesianConvention.PTCLAB_A_X_C_XZ,
        )
        observations.append(
            ExperimentalObservation(
                observation_id="ebsd_final_or",
                label=f"EBSD final OR — {ebsd_completed.get('file_name', 'map')}",
                or_parent_from_product=tuple(
                    tuple(float(x) for x in row)
                    for row in ebsd_ptclab
                ),
                provenance=(
                    "ebsd_pipeline_final_or; explicitly re-expressed from "
                    "symmetric-metric Cartesian frames to PTCLab frames; "
                    f"file_sha256={ebsd_completed.get('file_sha256', '')}; "
                    f"run_signature={ebsd_completed.get('run_signature', '')}"
                ),
            )
        )

    with st.expander("Optional manual experimental observation", expanded=False):
        enabled = st.checkbox("Add manual observation", key="research_manual_obs_enabled")
        if enabled:
            or_text = st.text_area(
                "Observed R(parent ← product), optional — PTCLab Cartesian frames",
                value="",
                height=96,
                placeholder="1 0 0\n0 1 0\n0 0 1",
                key="research_manual_or",
            )
            habit = st.text_input(
                "Observed A/M habit plane in parent crystal, optional",
                key="research_manual_habit",
            )
            twin_plane = st.text_input(
                "Observed twin plane in parent crystal, optional",
                key="research_manual_twin_plane",
            )
            twin_direction = st.text_input(
                "Observed twin direction in parent crystal, optional",
                key="research_manual_twin_dir",
            )
            shear_text = st.text_input(
                "Observed twin shear magnitude, optional",
                key="research_manual_shear",
            )
            shape_vector = st.text_input(
                "Observed shape-strain vector in parent crystal coordinates, optional",
                key="research_manual_shape_vector",
            )
            shape_magnitude_text = st.text_input(
                "Observed shape-strain magnitude, optional",
                key="research_manual_shape_magnitude",
            )
            uncertainty_text = st.text_input(
                "Uncertainties JSON, optional",
                placeholder='{"orientation_deg":0.2,"habit_plane_deg":1.0}',
                key="research_manual_uncertainty",
            )

            R = None if not or_text.strip() else _parse_matrix(or_text, name="observed OR")
            shear = None if not shear_text.strip() else float(shear_text)
            shape_magnitude = (
                None if not shape_magnitude_text.strip() else float(shape_magnitude_text)
            )
            uncertainty = (
                {}
                if not uncertainty_text.strip()
                else {
                    str(key): float(value)
                    for key, value in json.loads(uncertainty_text).items()
                }
            )
            observations.append(
                ExperimentalObservation(
                    observation_id="manual_001",
                    label="Manual experimental observation",
                    or_parent_from_product=R,
                    habit_plane_parent_crystal=_parse_optional_vector(
                        habit, name="habit plane"
                    ),
                    twin_plane_parent_crystal=_parse_optional_vector(
                        twin_plane, name="twin plane"
                    ),
                    twin_direction_parent_crystal=_parse_optional_vector(
                        twin_direction, name="twin direction"
                    ),
                    shear_magnitude=shear,
                    shape_vector_parent_crystal=_parse_optional_vector(
                        shape_vector, name="shape-strain vector"
                    ),
                    shape_vector_magnitude=shape_magnitude,
                    uncertainty=uncertainty,
                    provenance="user_supplied_experiment",
                )
            )
    return tuple(observations)


def _render_unified(project, transformation_id: str, base_signature: str) -> None:
    st.subheader("Unified physical comparison")
    st.caption(
        "CT, Ball–James/cofactor and PTMC are solved independently from the same "
        "crystallographic state. Physical branches are paired only after all predictions exist."
    )
    controls = st.columns(3)
    include_closing = controls[0].checkbox(
        "CT closing-gap ORs", value=True, key="research_ct_closing"
    )
    include_super = controls[1].checkbox(
        "CT supercompatibility", value=True, key="research_ct_super"
    )
    ptmc_mode = controls[2].selectbox(
        "PTMC branch set",
        options=[PTMCMode.ALL_TWINNING.value, PTMCMode.NONE.value],
        format_func=lambda value: "All twinning LIS" if value == "all_twinning" else "Off",
        key="research_ptmc_mode",
    )

    with st.expander("Numerical agreement tolerances", expanded=False):
        st.caption(
            "These thresholds classify whether matched physical observables agree numerically. "
            "They do not alter any theory calculation or branch assignment."
        )
        tcols = st.columns(4)
        plane_tol = tcols[0].number_input(
            "Plane-angle tolerance (deg)",
            min_value=1.0e-10,
            value=1.0e-4,
            format="%.8g",
            key="research_tol_plane",
        )
        direction_tol = tcols[1].number_input(
            "Direction-angle tolerance (deg)",
            min_value=1.0e-10,
            value=1.0e-4,
            format="%.8g",
            key="research_tol_direction",
        )
        magnitude_tol = tcols[2].number_input(
            "Relative-magnitude tolerance",
            min_value=1.0e-12,
            value=1.0e-6,
            format="%.8g",
            key="research_tol_magnitude",
        )
        or_tol = tcols[3].number_input(
            "OR disorientation tolerance (deg)",
            min_value=1.0e-10,
            value=1.0e-4,
            format="%.8g",
            key="research_tol_or",
        )
    tolerances = EquivalenceTolerances(
        plane_angle_deg=float(plane_tol),
        direction_angle_deg=float(direction_tol),
        relative_magnitude=float(magnitude_tol),
        or_disorientation_deg=float(or_tol),
    )

    try:
        experiments = _build_experiments(project, transformation_id, base_signature)
    except Exception as exc:
        _render_failure("Experimental observation is invalid", exc)
        experiments = tuple()

    experiment_payload = [asdict(item) for item in experiments]
    run_signature = fingerprint(
        base_signature,
        "unified-comparison-v2",
        include_closing,
        include_super,
        ptmc_mode,
        tolerances.to_dict(),
        experiment_payload,
    )
    st.session_state["research_current_unified_signature"] = run_signature

    if st.button("Calculate theory comparison", type="primary", key="research_run_unified"):
        with st.spinner("Calculating CT, Ball–James/cofactor and PTMC branches…"):
            try:
                unified = TheoryComparisonAdapter(project, transformation_id).compare(
                    ptmc_mode=ptmc_mode,
                    include_ct_closing_gap=include_closing,
                    include_ct_supercompatibility=include_super,
                    experiments=experiments,
                )
                equivalence = build_equivalence_report(
                    project,
                    transformation_id,
                    unified,
                    tolerances=tolerances,
                    include_independent_mm_audit=True,
                )
            except Exception as exc:
                _render_failure("Theory comparison failed", exc)
            else:
                _bound_put("research_unified_report", run_signature, unified)
                _bound_put("research_equivalence_report", run_signature, equivalence)

    unified = _bound_get("research_unified_report", run_signature)
    equivalence = _bound_get("research_equivalence_report", run_signature)
    if unified is None or equivalence is None:
        st.info("Calculate the comparison for the current state and settings.")
        return

    counts = {
        theory.value: len(unified.rows_for(theory))
        for theory in (TheoryKind.CAYRON_CT, TheoryKind.BALL_JAMES, TheoryKind.PTMC)
    }
    within = sum(
        item.all_required_components_within_tolerance is True
        for item in equivalence.matches
    )
    cols = st.columns(5)
    cols[0].metric("CT branches", counts["cayron_ct"])
    cols[1].metric("Ball–James branches", counts["ball_james"])
    cols[2].metric("PTMC branches", counts["ptmc"])
    cols[3].metric("Assigned physical pairs", len(equivalence.matches))
    cols[4].metric("Pairs within tolerance", within)

    if equivalence.warnings:
        for warning in equivalence.warnings:
            st.warning(warning)
    with st.expander("Definitions and frame conventions", expanded=False):
        st.json(
            {
                "agreement_tolerances": equivalence.tolerances.to_dict(),
                "classification_checks": equivalence.classification_checks,
            }
        )
        for note in equivalence.notes:
            st.markdown(f"- {note}")

    checks = equivalence.classification_checks
    if checks:
        st.markdown("#### Independent A/M existence check")
        check_cols = st.columns(4)
        check_cols[0].metric(
            "CT exact A/M",
            "yes" if checks.get("ct_am_exact_compatible") else "no",
        )
        check_cols[1].metric(
            "CT habit branches", checks.get("ct_am_habit_branch_count", 0)
        )
        check_cols[2].metric(
            "Ball–James A/M branches", checks.get("ball_james_am_branch_count", 0)
        )
        check_cols[3].metric(
            "CT / Ball–James existence",
            "agree"
            if checks.get("ct_vs_ball_james_am_existence_agreement")
            else "differ",
        )
        st.caption(str(checks.get("note", "")))

    match_table = _match_rows(equivalence)
    if match_table:
        st.markdown("#### Branch-resolved physical residuals")
        st.caption(
            "The one-to-one assignment prevents two branches from claiming the same counterpart. "
            "Each residual remains a separate physical quantity."
        )
        st.dataframe(_scalar_table(match_table), use_container_width=True, hide_index=True)
    else:
        st.warning("No branch pair shared enough physical observables for assignment.")

    if equivalence.mm_independent_audit is not None:
        with st.expander("Independent CT ↔ Mallard ↔ Ball–James M/M audit"):
            st.json(equivalence.mm_independent_audit)

    experiment_table = _experiment_rows(equivalence)
    if experiment_table:
        st.markdown("#### Theory ↔ experiment residuals")
        st.caption(
            "Experimental residuals and uncertainty-normalized residuals are shown component by component."
        )
        st.dataframe(
            _scalar_table(experiment_table),
            use_container_width=True,
            hide_index=True,
        )

    if equivalence.unmatched:
        with st.expander(f"Unassigned branches ({len(equivalence.unmatched)})"):
            st.dataframe(
                _scalar_table([item.to_dict() for item in equivalence.unmatched]),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("All native prediction branches"):
        st.dataframe(
            _scalar_table(unified.table_rows()),
            use_container_width=True,
            hide_index=True,
        )

    export_cols = st.columns(2)
    export_cols[0].download_button(
        "Unified theory report",
        data=json.dumps(unified.to_dict(), indent=2, default=str),
        file_name="cualni_ct_unified_theory_report.json",
        mime="application/json",
        key="research_download_unified",
        use_container_width=True,
    )
    export_cols[1].download_button(
        "Physical equivalence report",
        data=json.dumps(equivalence.to_dict(), indent=2, default=str),
        file_name="cualni_ct_physical_equivalence_report.json",
        mime="application/json",
        key="research_download_equivalence",
        use_container_width=True,
    )

def _render_atlas(project, transformation_id: str, base_signature: str) -> None:
    st.subheader("CT supercompatibility ↔ cofactor atlas")
    st.caption(
        "The product cell is varied only through independent parameters of its registered "
        "crystal setting. CT shear–shear closure and cofactor conditions are then solved independently."
    )

    independent = independent_product_lattice_parameters(project, transformation_id)
    label_map = {
        "a_scale": "a scale",
        "b_scale": "b scale",
        "c_scale": "c scale",
        "alpha_offset_deg": "α offset (deg)",
        "beta_offset_deg": "β offset (deg)",
        "gamma_offset_deg": "γ offset (deg)",
    }
    mode = st.radio(
        "Sweep dimensionality",
        ["1D", "2D"],
        horizontal=True,
        key="research_atlas_dimension",
        disabled=len(independent) < 2,
    )
    if len(independent) < 2:
        mode = "1D"

    primary = st.selectbox(
        "First parameter",
        independent,
        format_func=lambda name: label_map[name],
        key="research_atlas_axis1",
    )
    secondary = None
    if mode == "2D":
        choices = [name for name in independent if name != primary]
        secondary = st.selectbox(
            "Second parameter",
            choices,
            format_func=lambda name: label_map[name],
            key="research_atlas_axis2",
        )

    def axis_controls(name: str, prefix: str, default_samples: int) -> np.ndarray:
        is_scale = name.endswith("_scale")
        cols = st.columns(3)
        low = cols[0].number_input(
            f"{label_map[name]} minimum",
            value=0.98 if is_scale else -2.0,
            step=0.001 if is_scale else 0.1,
            format="%.8g",
            key=f"{prefix}_low_{name}",
        )
        high = cols[1].number_input(
            f"{label_map[name]} maximum",
            value=1.02 if is_scale else 2.0,
            step=0.001 if is_scale else 0.1,
            format="%.8g",
            key=f"{prefix}_high_{name}",
        )
        samples = cols[2].number_input(
            f"{label_map[name]} samples",
            min_value=2,
            max_value=101,
            value=default_samples,
            step=1,
            key=f"{prefix}_n_{name}",
        )
        if float(high) <= float(low):
            raise ValueError(f"{label_map[name]} maximum must exceed minimum")
        if is_scale and float(low) <= 0.0:
            raise ValueError(f"{label_map[name]} must remain positive")
        return np.linspace(float(low), float(high), int(samples))

    try:
        axis1 = axis_controls(primary, "research_atlas_axis1", 21 if mode == "1D" else 11)
        axes: dict[str, np.ndarray] = {primary: axis1}
        if secondary is not None:
            axes[secondary] = axis_controls(secondary, "research_atlas_axis2", 11)
    except ValueError as exc:
        st.error(str(exc))
        return

    requested_states = int(np.prod([len(values) for values in axes.values()]))
    st.caption(
        f"{requested_states} states will be evaluated. Parameters not selected remain at their current values."
    )
    run_signature = fingerprint(
        base_signature,
        "supercompatibility-atlas-v2",
        {name: [float(x) for x in values] for name, values in axes.items()},
    )

    if st.button("Evaluate compatibility atlas", key="research_run_atlas"):
        with st.spinner("Evaluating CT and cofactor conditions across the cell family…"):
            try:
                states = symmetry_preserving_product_grid(
                    project,
                    transformation_id,
                    axes,
                    max_states=2000,
                )
                report = evaluate_supercompatibility_atlas(states, transformation_id)
            except Exception as exc:
                _render_failure("Compatibility atlas failed", exc)
            else:
                _bound_put("research_atlas_report", run_signature, report)

    report = _bound_get("research_atlas_report", run_signature)
    if report is None:
        return

    count_cols = st.columns(5)
    for column, name in zip(
        count_cols,
        ("both", "ct_only", "cofactor_only", "neither", "not_evaluable"),
        strict=True,
    ):
        column.metric(name.replace("_", " ").title(), report.counts[name])

    rows = []
    for state in report.states:
        row = state.to_dict()
        metadata = row.pop("metadata")
        pair_counts = row.pop("matched_pair_counts")
        row.update(metadata)
        row.update({f"matched_{key}": value for key, value in pair_counts.items()})
        rows.append(row)
    frame = _scalar_table(rows)
    st.dataframe(frame, use_container_width=True, hide_index=True)

    if secondary is not None and rows:
        plot = pd.DataFrame(
            {
                "x": [float(state.metadata[primary]) for state in report.states],
                "y": [float(state.metadata[secondary]) for state in report.states],
                "classification": [state.classification.value for state in report.states],
                "pair disagreements": [state.pair_disagreement_count for state in report.states],
            }
        )
        st.markdown("#### State map")
        st.scatter_chart(plot, x="x", y="y", color="classification", size="pair disagreements")

    pair_rows = []
    for state in report.states:
        for pair in state.pair_results:
            pair_rows.append({"state_id": state.state_id, **pair.to_dict()})
    if pair_rows:
        with st.expander("Matched M/M relations"):
            st.caption(
                "Each row asks whether CT shear–shear closure and cofactor conditions agree on the same matched physical relation."
            )
            st.dataframe(_scalar_table(pair_rows), use_container_width=True, hide_index=True)

    st.download_button(
        "Compatibility atlas JSON",
        data=json.dumps(report.to_dict(), indent=2, default=str),
        file_name="cualni_ct_supercompatibility_atlas.json",
        mime="application/json",
        key="research_download_atlas",
    )

def _row_object(row, object_kind: str):
    if object_kind == "A/M habit plane":
        return row.habit_plane_parent_crystal, "plane"
    if object_kind == "Twin plane":
        return row.twin_plane_parent_crystal, "plane"
    if object_kind == "Twin direction":
        return row.twin_direction_parent_crystal, "direction"
    raise AssertionError(object_kind)


def _render_poles(project, transformation_id: str) -> None:
    st.subheader("Stereographic comparison")
    unified = _current_unified_report()
    if unified is None:
        st.info("Calculate the unified comparison first.")
        return

    transformation = project.transformation(transformation_id)
    parent_id = transformation.parent_phase_id
    product_id = transformation.product_phase_id
    prediction_tab, or_tab = st.tabs(["Predicted interfaces / twins", "OR pole overlay"])

    with prediction_tab:
        object_kind = st.selectbox(
            "Physical object",
            ["A/M habit plane", "Twin plane", "Twin direction"],
            key="research_pole_kind",
        )
        candidates = []
        for row in unified.rows:
            coeffs, kind = _row_object(row, object_kind)
            if coeffs is not None:
                candidates.append((row, coeffs, kind))

        labels = {
            f"{row.theory.value} · {row.branch_label}": (row, coeffs, kind)
            for row, coeffs, kind in candidates
        }
        selected = st.multiselect(
            "Branches",
            options=list(labels),
            default=list(labels)[: min(4, len(labels))],
            max_selections=12,
            key="research_pole_branches",
        )
        if selected:
            families = []
            for label in selected:
                row, coeffs, kind = labels[label]
                families.append(
                    pole_family(
                        project,
                        parent_id,
                        kind=kind,
                        coefficients=coeffs,
                        reference_phase_id=parent_id,
                        projective=True,
                        label=row.branch_label,
                        source=row.theory.value,
                    )
                )

            points = []
            for family in families:
                for point in family.points:
                    points.append(
                        {
                            "x": point.x,
                            "y": point.y,
                            "series": f"{family.source} · {family.label}",
                            "phase": family.phase_id,
                            "source": family.source,
                        }
                    )
            frame = pd.DataFrame(points)
            if not frame.empty:
                st.altair_chart(_stereographic_chart(frame), use_container_width=False)
                with st.expander("Pole coordinates"):
                    st.dataframe(frame, use_container_width=True, hide_index=True)

    with or_tab:
        st.caption(
            "A product direction or plane family is generated with product symmetry and then rotated into the parent frame by each physical OR."
        )
        c1, c2 = st.columns(2)
        pole_kind = c1.radio(
            "Product object type",
            ["direction", "plane"],
            horizontal=True,
            key="research_or_pole_kind",
        )
        seed_text = c2.text_input(
            "Product indices",
            value="0 0 1",
            key="research_or_pole_seed",
        )
        try:
            seed = _parse_vector(seed_text, name="product pole")
        except ValueError as exc:
            st.error(str(exc))
            return

        or_rows = [row for row in unified.rows if row.or_parent_from_product is not None]
        or_labels = {
            f"{row.theory.value} · {row.branch_label}": row for row in or_rows
        }
        selected_ors = st.multiselect(
            "Orientation relationships",
            options=list(or_labels),
            default=list(or_labels)[: min(5, len(or_labels))],
            max_selections=12,
            key="research_or_pole_branches",
        )
        add_parent = st.checkbox(
            "Add a parent reference family",
            value=False,
            key="research_or_pole_add_parent",
        )
        parent_seed = None
        if add_parent:
            parent_text = st.text_input(
                "Parent reference indices",
                value="0 0 1",
                key="research_or_parent_seed",
            )
            try:
                parent_seed = _parse_vector(parent_text, name="parent reference pole")
            except ValueError as exc:
                st.error(str(exc))
                return

        families = []
        for label in selected_ors:
            row = or_labels[label]
            R_symmetric = _reexpress_or(
                project,
                transformation_id,
                np.asarray(row.or_parent_from_product, dtype=float),
                source=CartesianConvention.PTCLAB_A_X_C_XZ,
                target=CartesianConvention.SYMMETRIC_METRIC,
            )
            families.append(
                pole_family(
                    project,
                    product_id,
                    kind=pole_kind,
                    coefficients=seed,
                    reference_phase_id=parent_id,
                    R_reference_from_phase=R_symmetric,
                    projective=True,
                    label=row.branch_label,
                    source=row.theory.value,
                )
            )
        if parent_seed is not None:
            families.append(
                pole_family(
                    project,
                    parent_id,
                    kind=pole_kind,
                    coefficients=parent_seed,
                    reference_phase_id=parent_id,
                    projective=True,
                    label="parent reference",
                    source="parent",
                )
            )

        points = []
        for family in families:
            for point in family.points:
                points.append(
                    {
                        "x": point.x,
                        "y": point.y,
                        "series": f"{family.source} · {family.label}",
                        "phase": family.phase_id,
                        "source": family.source,
                    }
                )
        frame = pd.DataFrame(points)
        if not frame.empty:
            st.altair_chart(_stereographic_chart(frame), use_container_width=False)
            st.download_button(
                "Pole coordinates CSV",
                data=frame.to_csv(index=False).encode("utf-8"),
                file_name="ct_equivalence_or_poles.csv",
                mime="text/csv",
                key="research_download_or_poles",
            )

def _render_invariant_and_double(
    project,
    transformation_id: str,
    base_signature: str,
) -> None:
    st.subheader("Independent geometric comparators")

    invariant_tab, double_tab = st.tabs(["Invariant line", "Double-shear PTMC"])

    with invariant_tab:
        mode = st.radio(
            "Space",
            ["direct", "reciprocal"],
            horizontal=True,
            key="research_invariant_space",
        )
        st.caption(
            "Specify the constraining plane or zone axis and two correlated vector pairs. "
            "The solver does not use the CT correspondence or a preselected OR."
        )
        c1, c2 = st.columns(2)
        parent_constraint = c1.text_input(
            "Parent plane" if mode == "direct" else "Parent zone axis",
            value="",
            key="research_inv_parent_constraint",
        )
        product_constraint = c2.text_input(
            "Product plane" if mode == "direct" else "Product zone axis",
            value="",
            key="research_inv_product_constraint",
        )
        p1, m1 = st.columns(2)
        parent_v1 = p1.text_input("Parent correlated vector 1", value="", key="research_inv_pa1")
        product_v1 = m1.text_input("Product correlated vector 1", value="", key="research_inv_pm1")
        p2, m2 = st.columns(2)
        parent_v2 = p2.text_input("Parent correlated vector 2", value="", key="research_inv_pa2")
        product_v2 = m2.text_input("Product correlated vector 2", value="", key="research_inv_pm2")

        inv_signature = fingerprint(
            base_signature,
            "invariant-line-v2",
            mode,
            parent_constraint,
            product_constraint,
            parent_v1,
            product_v1,
            parent_v2,
            product_v2,
        )
        if st.button("Solve invariant-line branches", key="research_run_invariant"):
            try:
                transformation = project.transformation(transformation_id)
                parent = project.phase(transformation.parent_phase_id)
                product = project.phase(transformation.product_phase_id)
                parent_vectors = [
                    _parse_vector(parent_v1, name="parent vector 1"),
                    _parse_vector(parent_v2, name="parent vector 2"),
                ]
                product_vectors = [
                    _parse_vector(product_v1, name="product vector 1"),
                    _parse_vector(product_v2, name="product vector 2"),
                ]
                if mode == "direct":
                    report = solve_direct_invariant_line(
                        parent.lattice,
                        product.lattice,
                        parent_plane=_parse_vector(parent_constraint, name="parent plane"),
                        product_plane=_parse_vector(product_constraint, name="product plane"),
                        parent_vectors=parent_vectors,
                        product_vectors=product_vectors,
                    )
                else:
                    report = solve_reciprocal_invariant_line(
                        parent.lattice,
                        product.lattice,
                        parent_zone_axis=_parse_vector(parent_constraint, name="parent zone axis"),
                        product_zone_axis=_parse_vector(product_constraint, name="product zone axis"),
                        parent_g_vectors=parent_vectors,
                        product_g_vectors=product_vectors,
                    )
            except Exception as exc:
                _render_failure("Invariant-line calculation failed", exc)
            else:
                _bound_put("research_invariant_report", inv_signature, report)

        report = _bound_get("research_invariant_report", inv_signature)
        if report is not None:
            rows = []
            for item in report.solutions:
                rows.append(
                    {
                        "branch": item.branch,
                        "rotation about constraint (deg)": item.rotation_about_constraint_deg,
                        "parent invariant index": item.invariant_parent_crystal.tolist(),
                        "product invariant index": item.invariant_product_crystal.tolist(),
                        "planar stretches": item.planar_principal_stretches,
                        "invariant-vector residual": item.invariant_residual,
                        "det(RD-I) residual": item.determinant_residual,
                        "rotation residual": item.rotation_residual,
                    }
                )
            st.dataframe(_scalar_table(rows), use_container_width=True, hide_index=True)
            if not rows:
                st.warning("No real invariant-line branch exists for these correlated pairs.")

            unified = _current_unified_report()
            if unified is not None and report.solutions:
                matcher = PhysicalBranchMatcher(project, transformation_id)
                candidates = [row for row in unified.rows if row.or_parent_from_product is not None]
                comparisons = []
                for solution in report.solutions:
                    for row in candidates:
                        R_other = _reexpress_or(
                            project,
                            transformation_id,
                            np.asarray(row.or_parent_from_product, dtype=float),
                            source=CartesianConvention.PTCLAB_A_X_C_XZ,
                            target=CartesianConvention.SYMMETRIC_METRIC,
                        )
                        comparisons.append(
                            {
                                "invariant branch": solution.branch,
                                "theory": row.theory.value,
                                "theory branch": row.branch_label,
                                "symmetry-reduced OR disorientation (deg)": (
                                    matcher.kernel.disorientation(
                                        solution.R_parent_from_product,
                                        R_other,
                                    ).angle_deg
                                ),
                            }
                        )
                with st.expander("Invariant-line OR ↔ CT/PTMC OR comparison"):
                    if comparisons:
                        st.dataframe(
                            _scalar_table(comparisons),
                            use_container_width=True,
                            hide_index=True,
                        )
                        available_theories = sorted({str(item["theory"]) for item in comparisons})
                        if "cayron_ct" in available_theories and "ptmc" not in available_theories:
                            st.info("Only Cayron CT OR predictions are available for this invariant-line comparison; PTMC produced no OR-bearing branch in the active unified run.")
                        elif "ptmc" in available_theories and "cayron_ct" not in available_theories:
                            st.info("Only PTMC OR predictions are available for this invariant-line comparison; no Cayron CT closing-gap OR is available in the active unified run.")
                    else:
                        st.info("No CT or PTMC OR-bearing branch is available for comparison with the invariant-line solution in the active unified run.")

    with double_tab:
        unified = _current_unified_report()
        if unified is None or unified.ptmc_report is None:
            st.info("Calculate the unified comparison with twinning PTMC enabled first.")
            return
        relations = unified.ptmc_report.twin_relations
        if len(relations) < 2:
            st.info("The current PTMC state has fewer than two twinning LIS relations.")
            return

        relation_labels = {
            (
                f"{index}: base {rel.base_variant_index} → {rel.other_variant_index}, "
                f"branch {rel.branch:+d}"
            ): index
            for index, rel in enumerate(relations)
        }
        left, right, fixed_col = st.columns(3)
        first_label = left.selectbox("First LIS", list(relation_labels), key="research_ds_first")
        second_label = right.selectbox(
            "Second LIS",
            list(relation_labels),
            index=min(1, len(relation_labels) - 1),
            key="research_ds_second",
        )
        fixed = fixed_col.number_input(
            "Fixed second parameter",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.01,
            key="research_ds_fixed",
        )
        composition = st.selectbox(
            "Two-LIS composition",
            [
                DoubleShearComposition.ADDITIVE_LAMINATE.value,
                DoubleShearComposition.SEQUENTIAL_SIMPLE_SHEAR.value,
            ],
            format_func=lambda value: (
                "Additive three-variant laminate"
                if value == DoubleShearComposition.ADDITIVE_LAMINATE.value
                else "Sequential finite simple shears"
            ),
            key="research_ds_composition",
        )
        if composition == DoubleShearComposition.ADDITIVE_LAMINATE.value:
            st.caption("F = U + f₁ a₁⊗n₁ + f₂ a₂⊗n₂, with f₁+f₂≤1.")
        else:
            st.caption(
                "Aᵢ=aᵢ⊗nᵢ is converted to Kᵢ=AᵢU⁻¹, then "
                "F=(I+f₂K₂)(I+f₁K₁)U. Parameters are shear multipliers, not phase fractions."
            )

        first_index = relation_labels[first_label]
        second_index = relation_labels[second_label]
        ds_signature = fingerprint(
            base_signature,
            "double-shear-v2",
            first_index,
            second_index,
            float(fixed),
            composition,
        )

        if st.button("Solve double-shear PTMC", key="research_run_ds"):
            first = relations[first_index]
            second = relations[second_index]
            if first.base_variant_index != second.base_variant_index:
                st.error("The two LIS relations must use the same base variant.")
            else:
                base = unified.ptmc_report.variants[first.base_variant_index]
                try:
                    report = solve_double_shear_ptmc(
                        base.U,
                        DoubleShearSystem(
                            tuple(float(x) for x in first.a),
                            tuple(float(x) for x in first.n_reference),
                            label=first_label,
                            source=DoubleShearSystemSource.VARIANT_RANK_ONE_INCREMENT,
                            base_variant_index=int(first.base_variant_index),
                            other_variant_index=int(first.other_variant_index),
                            source_rank_one_residual=float(first.rank_one_residual),
                        ),
                        DoubleShearSystem(
                            tuple(float(x) for x in second.a),
                            tuple(float(x) for x in second.n_reference),
                            label=second_label,
                            source=DoubleShearSystemSource.VARIANT_RANK_ONE_INCREMENT,
                            base_variant_index=int(second.base_variant_index),
                            other_variant_index=int(second.other_variant_index),
                            source_rank_one_residual=float(second.rank_one_residual),
                        ),
                        fixed_second_parameter=float(fixed),
                        composition=composition,
                        parameter_semantics=(
                            DoubleShearParameterSemantics.THREE_VARIANT_FRACTIONS
                            if composition == DoubleShearComposition.ADDITIVE_LAMINATE.value
                            else DoubleShearParameterSemantics.FINITE_SHEAR_MULTIPLIERS
                        ),
                        tolerance=1.0e-8,
                    )
                except Exception as exc:
                    _render_failure("Double-shear PTMC failed", exc)
                else:
                    _bound_put("research_double_shear_report", ds_signature, report)

        report = _bound_get("research_double_shear_report", ds_signature)
        if report is not None:
            rows = []
            for item in report.solutions:
                for habit in item.habit_connections:
                    rows.append(
                        {
                            "parameter 1": item.first_parameter,
                            "parameter 2": item.second_parameter,
                            "parameter semantics": report.parameter_semantics.value,
                            "habit branch": habit.branch,
                            "habit normal": habit.habit_normal_parent_cartesian.tolist(),
                            "shape vector": habit.shape_vector_parent_cartesian.tolist(),
                            "shape magnitude": habit.shape_vector_magnitude,
                            "middle-stretch residual": item.middle_stretch_residual,
                            "determinant residual": item.determinant_residual,
                            "rank-one residual": habit.residual,
                        }
                    )
            if rows:
                st.dataframe(_scalar_table(rows), use_container_width=True, hide_index=True)
            if report.continuum is not None:
                st.info(report.continuum.note)
            elif not rows:
                st.warning("Evaluated — no admissible double-shear root exists at the selected fixed second parameter.")

            ds_rows = list(double_shear_comparison_rows(report))
            ct_rows = [
                row
                for row in unified.rows
                if row.prediction_kind is PredictionKind.CT_AM_HABIT
            ]
            if ds_rows and ct_rows:
                matcher = PhysicalBranchMatcher(project, transformation_id)
                equivalence = _current_equivalence_report()
                tolerances = (
                    EquivalenceTolerances()
                    if equivalence is None
                    else equivalence.tolerances
                )
                matches, unmatched = matcher.match_pair(
                    ct_rows,
                    ds_rows,
                    family=MatchFamily.AM_INTERFACE,
                    scales=AssignmentScales(),
                    tolerances=tolerances,
                )
                with st.expander("CT ↔ double-shear PTMC physical comparison", expanded=True):
                    st.caption(
                        "Habit-plane and shape-strain residuals are compared in the parent symmetric-metric frame. "
                        "The generic two-LIS solver does not invent a physical OR when none is defined by its input contract."
                    )
                    st.dataframe(
                        _scalar_table(
                            [
                                {
                                    "CT branch": item.left_branch,
                                    "double-shear branch": item.right_branch,
                                    "habit Δ (deg)": item.residuals.habit_plane_angle_deg,
                                    "shape Δ projective (deg)": item.residuals.shape_direction_projective_deg,
                                    "shape magnitude rel.": item.residuals.shape_magnitude_relative,
                                    "rank-one tensor rel.": item.residuals.rank_one_tensor_relative,
                                    "comparison": item.completeness.value,
                                    "agreement": _agreement_text(item.all_required_components_within_tolerance),
                                }
                                for item in matches
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    if unmatched:
                        st.caption(f"Unassigned branches: {len(unmatched)}")

def _phase_config(phase, numeric_id: int) -> dict[str, Any]:
    lattice = phase.lattice
    return {
        "id": int(numeric_id),
        "name": phase.label or phase.phase_id,
        "point_group": phase.point_group_symbol,
        "lattice": {
            "a": lattice.a,
            "b": lattice.b,
            "c": lattice.c,
            "alpha_deg": lattice.alpha_deg,
            "beta_deg": lattice.beta_deg,
            "gamma_deg": lattice.gamma_deg,
            "length_unit": lattice.length_unit,
        },
    }


def _candidate_ors(unified) -> dict[str, np.ndarray]:
    output = {}
    if unified is None:
        return output
    for row in unified.rows:
        if row.or_parent_from_product is None:
            continue
        if row.prediction_kind not in {
            PredictionKind.CT_CLOSING_GAP_OR,
            PredictionKind.PTMC_HABIT,
        }:
            continue
        output[f"{row.theory.value} · {row.branch_label}"] = np.asarray(
            row.or_parent_from_product,
            dtype=float,
        )
    return output


def _render_ebsd(project, transformation_id: str, base_signature: str) -> None:
    st.subheader("EBSD analysis")
    st.caption(
        "Import an orientation map, segment grains, test the supplied OR against "
        "product/product boundaries, reconstruct parent candidates and retain "
        "ambiguity. No phase identity, reference-frame correction or acceptance "
        "threshold is inferred from the data."
    )

    upload = st.file_uploader(
        "Orientation map",
        type=["ang", "ctf", "csv", "tsv", "h5", "hdf5", "h5ebsd", "h5oina"],
        key="research_ebsd_pipeline_upload",
    )
    if upload is None:
        st.caption("Accepted formats: ANG, CTF, explicit-schema delimited text and HDF5.")
        return

    raw_bytes = upload.getvalue()
    file_hash = hashlib.sha256(raw_bytes).hexdigest()
    suffix = Path(upload.name).suffix.lower().lstrip(".")
    if suffix == "h5oina" or suffix == "h5ebsd":
        format_name = suffix
    elif suffix in {"h5", "hdf5"}:
        format_name = suffix
    elif suffix in {"ang", "ctf", "csv", "tsv"}:
        format_name = suffix
    else:
        st.error(f"Unsupported file extension: .{suffix}")
        return

    transformation = project.transformation(transformation_id)
    parent = project.phase(transformation.parent_phase_id)
    product = project.phase(transformation.product_phase_id)

    st.markdown("#### Phase IDs in this EBSD file")
    phase_cols = st.columns(2)
    parent_numeric = int(
        phase_cols[0].number_input(
            f"{parent.label or parent.phase_id} — EBSD phase ID",
            min_value=1,
            value=1,
            step=1,
            key="research_ebsd_parent_id",
        )
    )
    product_numeric = int(
        phase_cols[1].number_input(
            f"{product.label or product.phase_id} — EBSD phase ID",
            min_value=1,
            value=2,
            step=1,
            key="research_ebsd_product_id",
        )
    )
    if parent_numeric == product_numeric:
        st.error("Parent and product must have distinct EBSD phase IDs.")
        return

    st.markdown("#### Initial orientation relationship")
    unified = _current_unified_report()
    candidates = _candidate_ors(unified)
    or_choice_options = ["Manual matrix"] + list(candidates)
    selected_or = st.selectbox(
        "Source",
        or_choice_options,
        key="research_ebsd_or_source_v2",
    )
    if selected_or == "Manual matrix":
        matrix_text = st.text_area(
            "R(parent ← product) in symmetric-metric Cartesian frames",
            value="1 0 0\n0 1 0\n0 0 1",
            height=96,
            key="research_ebsd_or_manual_v2",
            help=(
                "This matrix maps product Cartesian vectors into the parent Cartesian "
                "frame used by the EBSD orientation kernel. It must be a proper rotation."
            ),
        )
        try:
            initial_R = np.asarray(_parse_matrix(matrix_text, name="initial OR"), dtype=float)
        except Exception as exc:
            _render_failure("Invalid OR matrix", exc)
            return
    else:
        # Unified CT/PTMC OR rows are expressed in PTCLab Cartesian frames.
        # The EBSD kernel uses the symmetric-metric embedding, so the physical
        # rotation is re-expressed explicitly rather than copied numerically.
        initial_R = _reexpress_or(
            project,
            transformation_id,
            candidates[selected_or],
            source=CartesianConvention.PTCLAB_A_X_C_XZ,
            target=CartesianConvention.SYMMETRIC_METRIC,
        )
        st.caption(
            "The selected theory OR is re-expressed into the EBSD symmetric-metric "
            "Cartesian frames before analysis; the physical OR is unchanged."
        )

    # Input-format contract -------------------------------------------------
    input_spec: dict[str, Any] = {"path": upload.name, "format": format_name}
    if suffix == "ctf":
        ctf_unit = st.selectbox(
            "3-D CTF Euler-angle unit",
            ["standard 2-D / not applicable", "degree", "radian"],
            key="research_ebsd_ctf_unit_v2",
            help=(
                "Standard 2-D CTF is interpreted by the native reader. For a 3-D CTF "
                "dataset, select its actual stored angle unit explicitly."
            ),
        )
        if ctf_unit != "standard 2-D / not applicable":
            input_spec["three_dimensional_angle_unit"] = ctf_unit

    advanced_schema_required = suffix in {"csv", "tsv", "h5", "hdf5", "h5ebsd", "h5oina"}
    if advanced_schema_required:
        st.warning(
            "This format has no universal orientation/schema convention. Supply the "
            "column or dataset mapping and orientation convention explicitly."
        )
        with st.expander("Input schema and orientation convention", expanded=True):
            convention_default = {
                "scipy_sequence": "ZXZ",
                "angle_unit": "degree",
                "raw_matrix_direction": "crystal_to_sample",
                "sample_correction": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "crystal_correction": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "label": "explicit user convention",
            }
            convention_text = st.text_area(
                "Orientation convention JSON",
                value=json.dumps(convention_default, indent=2),
                height=220,
                key="research_ebsd_input_convention_v2",
            )
            if suffix in {"csv", "tsv"}:
                schema_default = {
                    "phase": "phase",
                    "x": "x",
                    "y": "y",
                    "z": None,
                    "indexed": "indexed",
                    "euler": ["phi1", "Phi", "phi2"],
                    "quaternion": None,
                    "matrix": None,
                    "quaternion_order": "wxyz",
                    "quality": {},
                }
            else:
                schema_default = {
                    "phase": "/phase",
                    "x": "/x",
                    "y": "/y",
                    "z": None,
                    "indexed": None,
                    "euler": ["/phi1", "/Phi", "/phi2"],
                    "quaternion": None,
                    "matrix": None,
                    "quaternion_order": "wxyz",
                    "quality": {},
                }
            schema_text = st.text_area(
                "Column / dataset schema JSON",
                value=json.dumps(schema_default, indent=2),
                height=260,
                key="research_ebsd_input_schema_v2",
            )
            try:
                convention = json.loads(convention_text)
                schema = json.loads(schema_text)
                if not isinstance(convention, dict) or not isinstance(schema, dict):
                    raise TypeError("Convention and schema must each be JSON objects")
                input_spec["convention"] = convention
                input_spec["schema"] = schema
                if suffix == "tsv":
                    input_spec["separator"] = "\t"
            except Exception as exc:
                _render_failure("Input schema is not valid JSON", exc)
                return

    # Analysis settings -----------------------------------------------------
    st.markdown("#### Grain and boundary analysis")
    s1, s2, s3, s4 = st.columns(4)
    main_threshold = float(
        s1.number_input(
            "Segmentation threshold (deg)",
            min_value=1.0e-6,
            value=5.0,
            step=0.25,
            key="research_ebsd_seg_threshold_v2",
        )
    )
    minimum_points = int(
        s2.number_input(
            "Minimum points / grain",
            min_value=1,
            value=5,
            step=1,
            key="research_ebsd_min_points_v2",
        )
    )
    kam_cutoff = float(
        s3.number_input(
            "KAM neighbour cutoff (deg)",
            min_value=1.0e-6,
            value=5.0,
            step=0.25,
            key="research_ebsd_kam_v2",
        )
    )
    radius_factor = float(
        s4.number_input(
            "Neighbour radius factor",
            min_value=1.0e-6,
            value=1.15,
            step=0.05,
            key="research_ebsd_radius_factor_v2",
        )
    )
    sweep_text = st.text_input(
        "Segmentation sensitivity sweep (degrees)",
        value="2, 3, 4, 5, 6, 8",
        key="research_ebsd_sweep_v2",
        help="Strictly increasing thresholds. Every value is recorded in the run provenance.",
    )
    try:
        sweep_values = [float(x.strip()) for x in sweep_text.split(",") if x.strip()]
        if not sweep_values or any(not np.isfinite(x) or x <= 0 for x in sweep_values):
            raise ValueError("Sweep thresholds must be finite and positive")
        if any(b <= a for a, b in zip(sweep_values, sweep_values[1:])):
            raise ValueError("Sweep thresholds must be strictly increasing")
    except Exception as exc:
        _render_failure("Invalid segmentation sweep", exc)
        return

    with st.expander("Quality filters", expanded=False):
        st.caption(
            "Optional rules are applied only to quality fields actually present in the file. "
            "Example: [{\"field\":\"confidence_index\",\"op\":\">=\",\"value\":0.1}]"
        )
        quality_text = st.text_area(
            "Rules JSON",
            value="[]",
            height=110,
            key="research_ebsd_quality_v2",
        )
        try:
            quality_filters = json.loads(quality_text)
            if not isinstance(quality_filters, list):
                raise TypeError("Quality filters must be a JSON array")
        except Exception as exc:
            _render_failure("Quality-filter JSON is invalid", exc)
            return

    st.markdown("#### OR test and reconstruction")
    r1, r2, r3 = st.columns(3)
    refine_enabled = r1.checkbox(
        "Fit OR from product boundaries",
        value=True,
        key="research_ebsd_refine_enabled_v2",
    )
    use_refined = r1.checkbox(
        "Use fitted OR if accepted",
        value=False,
        key="research_ebsd_use_refined_v2",
        help="The supplied OR remains separate in provenance even when an accepted fit is used downstream.",
    )
    max_correction = float(
        r2.number_input(
            "Maximum OR correction (deg)",
            min_value=1.0e-6,
            value=5.0,
            step=0.25,
            key="research_ebsd_max_or_correction_v2",
        )
    )
    trim_fraction = float(
        r3.number_input(
            "Robust fit retained fraction",
            min_value=0.01,
            max_value=1.0,
            value=0.65,
            step=0.05,
            key="research_ebsd_trim_v2",
        )
    )

    p1, p2, p3 = st.columns(3)
    reconstruct_parent_enabled = p1.checkbox(
        "Reconstruct parent domains",
        value=True,
        key="research_ebsd_parent_recon_v2",
    )
    link_tolerance = float(
        p2.number_input(
            "Parent-domain link tolerance (deg)",
            min_value=1.0e-6,
            value=3.0,
            step=0.25,
            key="research_ebsd_link_tol_v2",
        )
    )
    recon_tolerance = float(
        p3.number_input(
            "Parent reconstruction tolerance (deg)",
            min_value=1.0e-6,
            value=3.0,
            step=0.25,
            key="research_ebsd_recon_tol_v2",
        )
    )

    b1, b2 = st.columns(2)
    boundary_residual = float(
        b1.number_input(
            "Boundary operator acceptance (deg)",
            min_value=1.0e-6,
            value=3.0,
            step=0.25,
            key="research_ebsd_boundary_resid_v2",
        )
    )
    boundary_margin = float(
        b2.number_input(
            "Minimum best/second-best margin (deg)",
            min_value=0.0,
            value=0.5,
            step=0.1,
            key="research_ebsd_boundary_margin_v2",
        )
    )

    with st.expander("Boundary-trace validation", expanded=False):
        trace_enabled = st.checkbox(
            "Evaluate configured plane hypotheses against map traces",
            value=False,
            key="research_ebsd_trace_enabled_v2",
        )
        surface_text = st.text_input(
            "Specimen surface normal in sample coordinates",
            value="0 0 1",
            disabled=not trace_enabled,
            key="research_ebsd_surface_normal_v2",
        )
        minimum_linearity = float(
            st.number_input(
                "Minimum boundary-line linearity",
                min_value=0.0,
                max_value=1.0,
                value=0.90,
                step=0.01,
                disabled=not trace_enabled,
                key="research_ebsd_linearity_v2",
            )
        )
        hypotheses_text = st.text_area(
            "Twin-plane hypotheses JSON",
            value="[]",
            height=140,
            disabled=not trace_enabled,
            key="research_ebsd_trace_hypotheses_v2",
            help=(
                "Each entry requires label, plane_side1_crystal and plane_side2_crystal; "
                "operator_index and source are optional."
            ),
        )
        try:
            trace_hypotheses = json.loads(hypotheses_text) if trace_enabled else []
            if trace_enabled and (not isinstance(trace_hypotheses, list) or not trace_hypotheses):
                raise ValueError("Trace validation requires at least one explicit plane hypothesis")
            surface_normal = _parse_vector(surface_text, name="surface normal") if trace_enabled else None
        except Exception as exc:
            _render_failure("Trace-validation input is invalid", exc)
            return

    config: dict[str, Any] = {
        "schema_version": 1,
        "input": input_spec,
        "phases": [
            _phase_config(parent, parent_numeric),
            _phase_config(product, product_numeric),
        ],
        "parent_phase_id": parent_numeric,
        "product_phase_id": product_numeric,
        "orientation_relationship": {
            "mode": "matrix",
            "matrix": initial_R.tolist(),
            "matrix_direction": "parent_from_product",
        },
        "quality_filters": quality_filters,
        "segmentation": {
            "main_threshold_deg": main_threshold,
            "sweep_thresholds_deg": sweep_values,
            "minimum_grain_points": minimum_points,
            "neighbor_radius_factor": radius_factor,
            "kam_max_neighbor_misorientation_deg": kam_cutoff,
        },
        "theory": {
            "quotient_tolerance_deg": 2.0e-7,
            "operator_equivalence_tolerance_deg": 2.0e-6,
            "crosscheck_topology": True,
        },
        "or_refinement": {
            "enabled": refine_enabled,
            "use_if_accepted": use_refined,
            "maximum_correction_deg": max_correction,
            "trim_fraction": trim_fraction,
            "huber_delta_deg": 2.0,
            "minimum_improvement_deg2": 0.02,
            "multi_start_step_deg": 0.75,
            "maximum_iterations": 120,
        },
        "parent_reconstruction": {
            "enabled": reconstruct_parent_enabled,
            "link_tolerance_deg": link_tolerance,
            "reconstruction_tolerance_deg": recon_tolerance,
            "minimum_grains": 2,
        },
        "boundary_classification": {
            "maximum_residual_deg": boundary_residual,
            "minimum_margin_deg": boundary_margin,
        },
        "trace_validation": (
            {
                "enabled": True,
                "surface_normal_sample": list(surface_normal),
                "minimum_linearity": minimum_linearity,
                "hypotheses": trace_hypotheses,
            }
            if trace_enabled
            else {"enabled": False}
        ),
        "output": {
            "directory": "results",
            "run_name": "streamlit-ebsd",
            "overwrite": False,
        },
    }

    with st.expander("Run configuration", expanded=False):
        st.json(config)
        st.caption(f"Input SHA-256: {file_hash}")

    run_signature = fingerprint(base_signature, file_hash, config)
    if st.button("Run EBSD analysis", type="primary", key="research_run_ebsd_v2"):
        with st.spinner("Segmenting grains and evaluating crystallographic hypotheses…"):
            try:
                with tempfile.TemporaryDirectory(prefix="cualni-ebsd-ui-") as td:
                    root = Path(td)
                    input_path = root / Path(upload.name).name
                    input_path.write_bytes(raw_bytes)
                    runtime_config = json.loads(json.dumps(config))
                    runtime_config["input"]["path"] = input_path.name
                    runtime_config["output"]["directory"] = "results"
                    runtime_config["output"]["run_name"] = "streamlit-ebsd"
                    config_path = root / "pipeline.json"
                    config_path.write_text(json.dumps(runtime_config, indent=2), encoding="utf-8")
                    result = run_pipeline(config_path)
                    tables: dict[str, pd.DataFrame] = {}
                    for filename in (
                        "phases.csv",
                        "grains.csv",
                        "boundaries.csv",
                        "parent_domains.csv",
                        "grain_variant_assignments.csv",
                        "trace_validation.csv",
                        "segmentation_sweep.csv",
                    ):
                        path = result.run_directory / filename
                        if path.is_file():
                            tables[filename] = pd.read_csv(path)
                    summary = dict(result.summary)
                    map_preview = pd.DataFrame()
                    map_path = result.run_directory / "map_fields.npz"
                    if map_path.is_file():
                        with np.load(map_path) as map_data:
                            n_points = len(map_data["x"])
                            stride = max(1, int(np.ceil(n_points / 30000)))
                            take = np.arange(0, n_points, stride, dtype=int)
                            map_preview = pd.DataFrame(
                                {
                                    "x": map_data["x"][take],
                                    "y": map_data["y"][take],
                                    "indexed": map_data["indexed"][take],
                                    "phase_id": map_data["phase_id"][take],
                                    "grain_id": map_data["grain_id"][take],
                                    "kam_deg": map_data["kam_deg"][take],
                                    "parent_domain_id": map_data["parent_domain_id"][take],
                                    "variant_id": map_data["variant_id"][take],
                                    "variant_residual_deg": map_data["variant_residual_deg"][take],
                                }
                            )
                _bound_put(
                    "research_ebsd_result",
                    run_signature,
                    {
                        "summary": summary,
                        "tables": tables,
                        "map_preview": map_preview,
                        "file_sha256": file_hash,
                    },
                )
                final_R = summary.get("orientation_relationship", {}).get(
                    "final_R_parent_from_product"
                )
                if final_R is not None:
                    # Bind the experimental OR to the calculated ProjectState,
                    # not to transient EBSD controls.  A different ProjectState
                    # can therefore never inherit an old experimental OR.
                    _bound_put(
                        "research_ebsd_experiment_or",
                        base_signature,
                        {
                            "matrix": final_R,
                            "run_signature": run_signature,
                            "file_sha256": file_hash,
                            "file_name": upload.name,
                        },
                    )
            except Exception as exc:
                _render_failure("EBSD analysis failed", exc)

    result = _bound_get("research_ebsd_result", run_signature)
    if not isinstance(result, Mapping):
        return
    summary = result["summary"]
    tables = result["tables"]
    map_preview = result.get("map_preview")
    if not isinstance(summary, Mapping) or not isinstance(tables, Mapping):
        return

    segmentation = summary.get("segmentation", {})
    boundaries = summary.get("boundaries", {})
    parent_recon = summary.get("parent_reconstruction", {})
    or_summary = summary.get("orientation_relationship", {})
    cols = st.columns(5)
    cols[0].metric("Grains", segmentation.get("n_grains", "—"))
    cols[1].metric("Product boundaries", boundaries.get("n_product_product_boundaries", "—"))
    cols[2].metric("Accepted operators", boundaries.get("n_operator_accepted", "—"))
    cols[3].metric("Ambiguous operators", boundaries.get("n_operator_ambiguous", "—"))
    cols[4].metric("Parent candidates", parent_recon.get("n_domain_candidates", "—"))

    st.markdown("#### Orientation-relationship check")
    or_rows = []
    for key, value in or_summary.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            or_rows.append({"quantity": key, "value": value})
    if or_rows:
        st.dataframe(pd.DataFrame(or_rows), hide_index=True, use_container_width=True)
    with st.expander("Complete OR provenance"):
        st.json(or_summary)

    if isinstance(map_preview, pd.DataFrame) and not map_preview.empty:
        st.markdown("#### Map fields")
        field = st.selectbox(
            "Display",
            [
                "phase_id",
                "grain_id",
                "kam_deg",
                "parent_domain_id",
                "variant_id",
                "variant_residual_deg",
            ],
            key="research_ebsd_map_field_v2",
        )
        plot = map_preview.loc[map_preview["indexed"].astype(bool)].copy()
        if field in {"phase_id", "grain_id", "parent_domain_id", "variant_id"}:
            plot["display_value"] = plot[field].astype(str)
            color = alt.Color("display_value:N", title=field)
            value_tooltip = alt.Tooltip("display_value:N", title=field)
        else:
            plot["display_value"] = pd.to_numeric(plot[field], errors="coerce")
            plot = plot[np.isfinite(plot["display_value"])]
            color = alt.Color("display_value:Q", title=field)
            value_tooltip = alt.Tooltip("display_value:Q", title=field, format=".6g")
        if not plot.empty:
            chart = (
                alt.Chart(plot)
                .mark_point(filled=True, size=18)
                .encode(
                    x=alt.X("x:Q", title="x"),
                    y=alt.Y("y:Q", title="y"),
                    color=color,
                    tooltip=[
                        alt.Tooltip("x:Q", format=".6g"),
                        alt.Tooltip("y:Q", format=".6g"),
                        value_tooltip,
                    ],
                )
                .properties(height=520)
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption(
                "Large maps are deterministically downsampled to at most ~30,000 points for display; calculations use every retained point."
            )

    important_tables = [
        "segmentation_sweep.csv",
        "grains.csv",
        "boundaries.csv",
        "parent_domains.csv",
        "grain_variant_assignments.csv",
        "trace_validation.csv",
    ]
    for name in important_tables:
        frame = tables.get(name)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            with st.expander(name.replace(".csv", "").replace("_", " ").title()):
                st.dataframe(frame, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download CSV",
                    data=frame.to_csv(index=False).encode("utf-8"),
                    file_name=name,
                    mime="text/csv",
                    key=f"research_download_{name}_v2",
                )

    st.download_button(
        "Download EBSD summary",
        data=json.dumps(summary, indent=2, default=str),
        file_name="ebsd_pipeline_summary.json",
        mime="application/json",
        key="research_download_ebsd_summary_v2",
    )


def render_research_extension() -> None:
    st.header("CT equivalence laboratory")
    payload = st.session_state.get("current_project_payload")
    if payload is None:
        st.info(
            "No calculated ProjectState is active in this session. Open the main "
            "workstation, define or load the transformation, and calculate it first."
        )
        return

    st.caption(
        "Branch-resolved comparison of correspondence theory, nonlinear-elastic "
        "compatibility, PTMC and experiment."
    )

    if st.session_state.get("requires_recalculation", False):
        st.warning(
            "Setup has changed since the last calculation. Recalculate the "
            "transformation before running research comparisons."
        )
        return

    try:
        loaded = project_from_dict(
            dict(payload),
            source="streamlit:ct-equivalence-laboratory",
        )
        project = loaded.project
    except Exception as exc:
        st.error(f"The active ProjectState cannot be loaded: {exc}")
        return

    ids = [item.transformation_id for item in project.transformations]
    if not ids:
        st.info("The active project contains no transformation.")
        return
    transformation_id = (
        ids[0]
        if len(ids) == 1
        else st.selectbox(
            "Transformation",
            ids,
            key="research_transformation_id",
        )
    )

    (
        comparison_tab,
        atlas_tab,
        pole_tab,
        independent_tab,
        ebsd_tab,
    ) = st.tabs(
        [
            "Unified comparison",
            "Compatibility atlas",
            "Pole figures",
            "Invariant line / double shear",
            "EBSD pipeline",
        ]
    )

    base_signature = _base_signature(dict(payload), transformation_id)

    with comparison_tab:
        _render_unified(project, transformation_id, base_signature)
    with atlas_tab:
        _render_atlas(project, transformation_id, base_signature)
    with pole_tab:
        _render_poles(project, transformation_id)
    with independent_tab:
        _render_invariant_and_double(project, transformation_id, base_signature)
    with ebsd_tab:
        _render_ebsd(project, transformation_id, base_signature)
