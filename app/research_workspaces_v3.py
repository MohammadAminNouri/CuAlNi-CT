from __future__ import annotations

"""Optimized CT-equivalence research workstation.

The original research workspaces remain available for the atlas, pole figures,
invariant-line/double-shear tools, and EBSD pipeline.  This module replaces only
the unified comparison orchestration and result semantics:

* CT/Ball--James/PTMC components are cached independently by calculated state;
* changing a display tolerance never reruns the theory solvers;
* CT approximate A/M habits are shown only as nearest-degeneracy diagnostics;
* the authoritative CT↔Mallard↔Ball--James M/M audit is cached separately;
* headline counts distinguish candidate assignments from actual agreement;
* unmatched counts are unique native branches rather than repeated comparison
  instances.
"""

from collections import Counter, defaultdict
from dataclasses import asdict
import json
from typing import Any, Mapping

import pandas as pd
import streamlit as st

import app.research_workspaces as legacy
from app.session_state import fingerprint
from cualni_cryst.project_io import project_from_dict
from cualni_cryst.theory_equivalence import (
    AssignmentScales,
    EquivalenceTolerances,
    MatchCompleteness,
    MatchFamily,
)
from cualni_cryst.theory_equivalence_optimized import (
    CachedPhysicalBranchMatcher,
    attach_independent_mm_audit,
    build_equivalence_report_optimized,
    independent_mm_audit,
)
from cualni_cryst.theory_unified import (
    PTMCMode,
    PredictionKind,
    TheoryKind,
    UnifiedTheoryReport,
)
from cualni_cryst.theory_unified_components import StagedTheoryComparisonAdapter


def _component_cache_key(key: str, signature: str) -> str:
    # Keep multiple expensive component revisions in the same Streamlit session.
    # ``put_bound`` itself stores only one value per key, so the signature is
    # part of the key as well as the BoundResult envelope.
    return f"{key}::{signature}"


def _component_get(key: str, signature: str):
    return legacy._bound_get(_component_cache_key(key, signature), signature)


def _component_put(key: str, signature: str, value: Any) -> None:
    legacy._bound_put(_component_cache_key(key, signature), signature, value)


def _prune_component_cache_for_new_base(base_signature: str) -> None:
    previous = st.session_state.get("research_v3_base_signature")
    if previous == base_signature:
        return
    prefixes = (
        "research_ct_component_v3::",
        "research_bj_component_v3::",
        "research_ptmc_component_v3::",
        "research_unified_report::",
        "research_equivalence_report_v3::",
        "research_mm_audit_v3::",
    )
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith(prefixes):
            st.session_state.pop(key, None)
    # Never let a result from a previous calculated crystallographic state leak
    # into pole/EBSD/double-shear tabs while the new state has not been compared.
    st.session_state.pop("research_current_unified_signature", None)
    st.session_state.pop("research_unified_report", None)
    st.session_state.pop("research_equivalence_report", None)
    st.session_state["research_v3_base_signature"] = base_signature


def _assemble_unified_report(
    project,
    transformation_id: str,
    base_signature: str,
    *,
    include_closing: bool,
    include_super: bool,
    ptmc_mode: str,
    experiments,
    status,
) -> tuple[UnifiedTheoryReport, str]:
    """Compute/cache theory components independently, then assemble one report."""
    adapter = StagedTheoryComparisonAdapter(project, transformation_id)

    ct_signature = fingerprint(
        base_signature,
        "ct-component-v3",
        include_closing,
        include_super,
    )
    ct_bundle = _component_get("research_ct_component_v3", ct_signature)
    if ct_bundle is None:
        status.write("CT: solving CMC/SMC, transformation twins and requested CT extensions…")
        ct_bundle = adapter.ct_component(
            natural_orientation=None,
            include_closing_gap=include_closing,
            include_supercompatibility=include_super,
        )
        _component_put("research_ct_component_v3", ct_signature, ct_bundle)
    else:
        status.write("CT: reused cached result for this calculated state.")
    ct_report = ct_bundle.report

    bj_signature = fingerprint(base_signature, "ball-james-component-v3", 101)
    bj_bundle = _component_get("research_bj_component_v3", bj_signature)
    if bj_bundle is None:
        status.write("Ball–James/cofactor: solving independent rank-one and cofactor branches…")
        bj_bundle = adapter.ball_james_component(fraction_samples=101)
        _component_put("research_bj_component_v3", bj_signature, bj_bundle)
    else:
        status.write("Ball–James/cofactor: reused cached result for this calculated state.")

    mode = PTMCMode(ptmc_mode)
    ptmc_signature = fingerprint(base_signature, "ptmc-component-v3", mode.value)
    ptmc_bundle = _component_get("research_ptmc_component_v3", ptmc_signature)
    if ptmc_bundle is None:
        if mode is PTMCMode.NONE:
            ptmc_bundle = adapter.ptmc_component(mode=PTMCMode.NONE)
            status.write("PTMC: disabled for this run.")
        else:
            status.write("PTMC: solving the full twinning-LIS branch set. This is the deep stage…")
            ptmc_bundle = adapter.ptmc_component(
                mode=mode,
                request=None,
                base_variant_index=None,
                dilatational_factor=1.0,
            )
        _component_put("research_ptmc_component_v3", ptmc_signature, ptmc_bundle)
    else:
        status.write("PTMC: reused cached result for this calculated state and branch mode.")

    unified = adapter.assemble(
        ct_bundle,
        bj_bundle,
        ptmc_bundle,
        experiments=tuple(experiments),
    )

    # Clarify the exact/approximate contract without changing native backend data.
    warnings = list(unified.warnings)
    if not ct_report.exact_habit_planes:
        warnings = [
            warning for warning in warnings
            if not warning.startswith("CT has no exact finite A/M habit-plane branch")
        ]
        warnings.append(
            "CT has no exact finite A/M habit-plane branch for these inputs; approximate CMC rows are nearest-degeneracy diagnostics and are excluded from exact equivalence assignment."
        )
    unified = UnifiedTheoryReport(
        transformation_id=unified.transformation_id,
        parent_phase_id=unified.parent_phase_id,
        product_phase_id=unified.product_phase_id,
        rows=unified.rows,
        ct_report=unified.ct_report,
        ball_james_report=unified.ball_james_report,
        ptmc_report=unified.ptmc_report,
        warnings=tuple(warnings),
        notes=unified.notes + (
            "CT approximate A/M habit diagnostics are preserved but excluded from exact-equivalence assignment.",
        ),
    )

    experiment_payload = [asdict(item) for item in experiments]
    theory_signature = fingerprint(
        base_signature,
        "unified-theory-v3",
        include_closing,
        include_super,
        mode.value,
        experiment_payload,
    )
    return unified, theory_signature


def _prediction_inventory(unified: UnifiedTheoryReport) -> pd.DataFrame:
    counts = Counter((row.theory.value, row.prediction_kind.value) for row in unified.rows)
    rows = []
    for (theory, kind), count in sorted(counts.items()):
        exact = sum(
            row.exact is True
            for row in unified.rows
            if row.theory.value == theory and row.prediction_kind.value == kind
        )
        approximate = sum(
            row.exact is False
            for row in unified.rows
            if row.theory.value == theory and row.prediction_kind.value == kind
        )
        rows.append(
            {
                "theory": theory,
                "prediction kind": kind,
                "native rows": count,
                "exact": exact,
                "approximate / diagnostic": approximate,
            }
        )
    return pd.DataFrame(rows)


def _unique_unmatched_rows(equivalence) -> list[dict[str, Any]]:
    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    families: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in equivalence.unmatched:
        key = (item.theory.value, item.row_id)
        aggregated.setdefault(
            key,
            {
                "theory": item.theory.value,
                "row_id": item.row_id,
                "branch": item.branch,
            },
        )
        families[key].add(item.family.value)
    output = []
    for key, row in aggregated.items():
        output.append({**row, "comparison families": ", ".join(sorted(families[key]))})
    return sorted(output, key=lambda row: (row["theory"], row["row_id"]))


def _diagnostic_nearest_rows(project, transformation_id: str, unified, tolerances) -> list[dict[str, Any]]:
    """Nearest matches for CT approximate A/M rows, explicitly outside equivalence claims."""
    approximate = [
        row
        for row in unified.rows
        if row.theory is TheoryKind.CAYRON_CT
        and row.prediction_kind is PredictionKind.CT_AM_HABIT
        and row.exact is False
    ]
    if not approximate:
        return []

    matcher = CachedPhysicalBranchMatcher(project, transformation_id)
    output: list[dict[str, Any]] = []
    target_specs = (
        (TheoryKind.BALL_JAMES, (PredictionKind.BALL_JAMES_AM,)),
        (TheoryKind.PTMC, (PredictionKind.PTMC_HABIT,)),
    )
    for theory, kinds in target_specs:
        targets = [
            row
            for row in unified.rows
            if row.theory is theory and row.prediction_kind in kinds
        ]
        if not targets:
            continue
        matches, _ = matcher.match_pair(
            approximate,
            targets,
            family=MatchFamily.AM_INTERFACE,
            scales=AssignmentScales(),
            tolerances=tolerances,
        )
        for item in matches:
            r = item.residuals
            output.append(
                {
                    "CT diagnostic": item.left_branch,
                    "nearest theory": item.right_theory.value,
                    "nearest branch": item.right_branch,
                    "habit Δ (deg)": r.habit_plane_angle_deg,
                    "shape Δ projective (deg)": r.shape_direction_projective_deg,
                    "shape magnitude rel.": r.shape_magnitude_relative,
                    "rank-one tensor rel.": r.rank_one_tensor_relative,
                    "status": "nearest diagnostic only — not an exact-equivalence claim",
                }
            )
    return output


def _render_mm_audit(audit: Mapping[str, Any] | None) -> None:
    if not audit:
        return
    st.markdown("#### Independent CT ↔ Mallard ↔ Ball–James M/M validation")
    cols = st.columns(4)
    cols[0].metric("Validation", "passed" if audit.get("success") else "failed")
    cols[1].metric("Relations checked", audit.get("relation_count", 0))
    cols[2].metric("CT coverage", "complete" if audit.get("ct_coverage_complete") else "incomplete")
    cols[3].metric("Classification", "exact" if audit.get("classification_exact") else "not exact")
    residuals = pd.DataFrame(
        [
            {"residual": "CT geometry angle (deg)", "maximum": audit.get("max_ct_geometry_angle_deg")},
            {"residual": "Ball–James geometry angle (deg)", "maximum": audit.get("max_ball_james_geometry_angle_deg")},
            {"residual": "shear relative", "maximum": audit.get("max_shear_relative_residual")},
            {"residual": "rank-one", "maximum": audit.get("max_rank_one_residual")},
            {"residual": "rotation", "maximum": audit.get("max_rotation_residual")},
        ]
    )
    st.dataframe(residuals, hide_index=True, use_container_width=True)
    st.caption(
        "This audit performs the required branch-specific configuration handling; it is the authoritative full M/M cross-check."
    )


def _render_unified_v3(project, transformation_id: str, base_signature: str) -> None:
    st.subheader("Unified physical comparison")
    st.caption(
        "CT, Ball–James/cofactor and PTMC are solved independently from the same crystallographic state. "
        "Exact-equivalence assignment excludes approximate CT A/M diagnostics."
    )

    controls = st.columns(4)
    include_closing = controls[0].checkbox(
        "CT closing-gap ORs",
        value=False,
        key="research_v3_ct_closing",
        help="Deep CT extension. All candidates are retained unless a natural OR is supplied.",
    )
    include_super = controls[1].checkbox(
        "CT supercompatibility",
        value=False,
        key="research_v3_ct_super",
        help="Evaluate CT A/M/M supercompatibility diagnostics in addition to ordinary CT branches.",
    )
    ptmc_mode = controls[2].selectbox(
        "PTMC branch set",
        options=[PTMCMode.NONE.value, PTMCMode.ALL_TWINNING.value],
        index=0,
        format_func=lambda value: "Off" if value == PTMCMode.NONE.value else "All twinning LIS — deep run",
        key="research_v3_ptmc_mode",
    )
    include_audit = controls[3].checkbox(
        "Independent M/M audit",
        value=True,
        key="research_v3_mm_audit",
        help="Authoritative CT ↔ Mallard ↔ Ball–James M/M validation; cached by calculated state.",
    )

    if ptmc_mode == PTMCMode.ALL_TWINNING.value:
        st.info(
            "Deep PTMC is enabled. The first run can be substantially heavier; later runs reuse the cached PTMC branch set unless the calculated crystallographic state changes."
        )

    with st.expander("Numerical agreement tolerances", expanded=False):
        st.caption(
            "These thresholds classify already-computed physical residuals. Changing them does not rerun CT, Ball–James or PTMC."
        )
        tcols = st.columns(5)
        plane_tol = tcols[0].number_input(
            "Plane angle (deg)", min_value=1.0e-10, value=1.0e-4, format="%.8g", key="research_v3_tol_plane"
        )
        direction_tol = tcols[1].number_input(
            "Direction angle (deg)", min_value=1.0e-10, value=1.0e-4, format="%.8g", key="research_v3_tol_direction"
        )
        magnitude_tol = tcols[2].number_input(
            "Relative magnitude", min_value=1.0e-12, value=1.0e-6, format="%.8g", key="research_v3_tol_magnitude"
        )
        tensor_tol = tcols[3].number_input(
            "Rank-one tensor relative", min_value=1.0e-12, value=1.0e-6, format="%.8g", key="research_v3_tol_tensor"
        )
        or_tol = tcols[4].number_input(
            "OR disorientation (deg)", min_value=1.0e-10, value=1.0e-4, format="%.8g", key="research_v3_tol_or"
        )
    tolerances = EquivalenceTolerances(
        plane_angle_deg=float(plane_tol),
        direction_angle_deg=float(direction_tol),
        relative_magnitude=float(magnitude_tol),
        rank_one_tensor_relative=float(tensor_tol),
        or_disorientation_deg=float(or_tol),
    )

    try:
        experiments = legacy._build_experiments(project, transformation_id, base_signature)
    except Exception as exc:
        legacy._render_failure("Experimental observation is invalid", exc)
        experiments = tuple()

    experiment_payload = [asdict(item) for item in experiments]
    theory_signature = fingerprint(
        base_signature,
        "unified-theory-v3",
        include_closing,
        include_super,
        ptmc_mode,
        experiment_payload,
    )
    equivalence_signature = fingerprint(
        theory_signature,
        "physical-equivalence-v3",
        tolerances.to_dict(),
    )
    audit_signature = fingerprint(base_signature, "independent-mm-audit-v3")

    calculate = st.button("Calculate / update comparison", type="primary", key="research_v3_run")

    unified = _component_get("research_unified_report", theory_signature)
    equivalence = _component_get("research_equivalence_report_v3", equivalence_signature)
    audit = _component_get("research_mm_audit_v3", audit_signature) if include_audit else None

    if calculate:
        try:
            with st.status("Running branch-resolved comparison…", expanded=True) as status:
                if unified is None:
                    unified, computed_signature = _assemble_unified_report(
                        project,
                        transformation_id,
                        base_signature,
                        include_closing=include_closing,
                        include_super=include_super,
                        ptmc_mode=ptmc_mode,
                        experiments=experiments,
                        status=status,
                    )
                    if computed_signature != theory_signature:
                        raise AssertionError("Theory signature mismatch")
                    _component_put("research_unified_report", theory_signature, unified)
                else:
                    status.write("Theory solvers: reused cached unified result.")

                if equivalence is None:
                    status.write("Physical matcher: assigning exact branches with cached physical representations…")
                    equivalence = build_equivalence_report_optimized(
                        project,
                        transformation_id,
                        unified,
                        tolerances=tolerances,
                    )
                    _component_put("research_equivalence_report_v3", equivalence_signature, equivalence)
                else:
                    status.write("Physical matcher: reused cached residual/assignment result.")

                if include_audit and audit is None:
                    status.write("Independent M/M audit: validating CT ↔ Mallard ↔ Ball–James relations…")
                    audit = independent_mm_audit(project, transformation_id)
                    _component_put("research_mm_audit_v3", audit_signature, audit)
                elif include_audit:
                    status.write("Independent M/M audit: reused cached validation.")

                status.update(label="Comparison complete", state="complete", expanded=False)
        except Exception as exc:
            legacy._render_failure("Theory comparison failed", exc)

    if unified is None or equivalence is None:
        st.session_state.pop("research_current_unified_signature", None)
        st.info("Calculate the comparison for the current state and settings.")
        return

    if include_audit:
        equivalence = attach_independent_mm_audit(equivalence, audit)

    # Preserve the legacy state contract used by pole, double-shear and EBSD tabs.
    st.session_state["research_current_unified_signature"] = theory_signature
    legacy._bound_put("research_unified_report", theory_signature, unified)
    legacy._bound_put("research_equivalence_report", theory_signature, equivalence)

    exact_agreements = sum(
        item.completeness is MatchCompleteness.FULL
        and item.all_required_components_within_tolerance is True
        for item in equivalence.matches
    )
    outside = sum(
        item.completeness is MatchCompleteness.FULL
        and item.all_required_components_within_tolerance is False
        for item in equivalence.matches
    )
    partial = sum(item.completeness is MatchCompleteness.PARTIAL for item in equivalence.matches)
    mm_validated = (
        int(audit.get("relation_count", 0))
        if include_audit and audit and audit.get("success")
        else 0
    )

    summary = st.columns(5)
    summary[0].metric("Candidate branch assignments", len(equivalence.matches))
    summary[1].metric("Full agreements", exact_agreements)
    summary[2].metric("Independent M/M validated", mm_validated)
    summary[3].metric("Partial comparisons", partial)
    summary[4].metric("Full pairs outside tolerance", outside)

    st.caption(
        "Candidate assignment is only a one-to-one pairing step; it is not an agreement claim. "
        "The independent M/M audit is reported separately because it performs the required configuration push-forward."
    )

    inventory = _prediction_inventory(unified)
    with st.expander("Native prediction inventory", expanded=False):
        st.dataframe(inventory, use_container_width=True, hide_index=True)
        st.caption(
            f"Total native rows: {len(unified.rows)}. Continuum and parameter-diagnostic PTMC rows are counted separately from discrete habit solutions."
        )

    for warning in unified.warnings:
        st.warning(warning)
    for warning in equivalence.warnings:
        st.warning(warning)

    checks = equivalence.classification_checks
    exact_ct_habits = sum(
        row.theory is TheoryKind.CAYRON_CT
        and row.prediction_kind is PredictionKind.CT_AM_HABIT
        and row.exact is True
        for row in unified.rows
    )
    approx_ct_habits = sum(
        row.theory is TheoryKind.CAYRON_CT
        and row.prediction_kind is PredictionKind.CT_AM_HABIT
        and row.exact is False
        for row in unified.rows
    )
    bj_am = sum(
        row.theory is TheoryKind.BALL_JAMES and row.prediction_kind is PredictionKind.BALL_JAMES_AM
        for row in unified.rows
    )
    st.markdown("#### Independent A/M existence check")
    c = st.columns(4)
    c[0].metric("CT exact A/M", "yes" if checks.get("ct_am_exact_compatible") else "no")
    c[1].metric("CT exact habit branches", exact_ct_habits)
    c[2].metric("Ball–James exact A/M branches", bj_am)
    c[3].metric(
        "CT / Ball–James existence",
        "agree" if checks.get("ct_vs_ball_james_am_existence_agreement") else "differ",
    )
    if approx_ct_habits:
        st.caption(
            f"CT also produced {approx_ct_habits} nearest-degeneracy A/M habit diagnostic(s). They are not exact compatibility branches and are excluded from the equivalence assignment below."
        )

    exact_rows = legacy._match_rows(equivalence)
    st.markdown("#### Exact-theory branch assignments and physical residuals")
    if exact_rows:
        st.dataframe(legacy._scalar_table(exact_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No exact cross-theory branch pair shared enough physical observables for assignment.")

    diagnostics = _diagnostic_nearest_rows(project, transformation_id, unified, tolerances)
    if diagnostics:
        with st.expander("Approximate CT A/M diagnostics — nearest branches only", expanded=False):
            st.warning(
                "These rows are nearest-degeneracy diagnostics. They are intentionally excluded from exact-equivalence counts and must not be read as CT/PTMC or CT/Ball–James equivalence."
            )
            st.dataframe(pd.DataFrame(diagnostics), use_container_width=True, hide_index=True)

    _render_mm_audit(audit if include_audit else None)

    experiment_table = legacy._experiment_rows(equivalence)
    if experiment_table:
        st.markdown("#### Theory ↔ experiment residuals")
        st.dataframe(legacy._scalar_table(experiment_table), use_container_width=True, hide_index=True)

    unique_unmatched = _unique_unmatched_rows(equivalence)
    if unique_unmatched:
        with st.expander(f"Unique unassigned native branches ({len(unique_unmatched)})"):
            st.caption(
                f"{len(equivalence.unmatched)} unassigned comparison instances collapse to {len(unique_unmatched)} unique native theory rows."
            )
            st.dataframe(pd.DataFrame(unique_unmatched), use_container_width=True, hide_index=True)

    with st.expander("All native prediction rows"):
        st.dataframe(
            legacy._scalar_table(unified.table_rows()),
            use_container_width=True,
            hide_index=True,
        )

    exports = st.columns(2)
    exports[0].download_button(
        "Unified theory report",
        data=json.dumps(unified.to_dict(), indent=2, default=str),
        file_name="cualni_ct_unified_theory_report.json",
        mime="application/json",
        key="research_v3_download_unified",
        use_container_width=True,
    )
    exports[1].download_button(
        "Physical equivalence report",
        data=json.dumps(equivalence.to_dict(), indent=2, default=str),
        file_name="cualni_ct_physical_equivalence_report.json",
        mime="application/json",
        key="research_v3_download_equivalence",
        use_container_width=True,
    )


def render_research_extension() -> None:
    st.header("CT equivalence laboratory")
    payload = st.session_state.get("current_project_payload")
    if payload is None:
        st.info(
            "No calculated ProjectState is active in this session. Open the main workstation, define or load the transformation, and calculate it first."
        )
        return

    st.caption(
        "Branch-resolved comparison of correspondence theory, nonlinear-elastic compatibility, PTMC and experiment."
    )

    if st.session_state.get("requires_recalculation", False):
        st.warning(
            "Setup has changed since the last calculation. Recalculate the transformation before running research comparisons."
        )
        return

    try:
        loaded = project_from_dict(dict(payload), source="streamlit:ct-equivalence-laboratory-v3")
        project = loaded.project
    except Exception as exc:
        st.error(f"The active ProjectState cannot be loaded: {exc}")
        return

    ids = [item.transformation_id for item in project.transformations]
    if not ids:
        st.info("The active project contains no transformation.")
        return
    transformation_id = ids[0] if len(ids) == 1 else st.selectbox(
        "Transformation", ids, key="research_v3_transformation_id"
    )

    comparison_tab, atlas_tab, pole_tab, independent_tab, ebsd_tab = st.tabs(
        [
            "Unified comparison",
            "Compatibility atlas",
            "Pole figures",
            "Invariant line / double shear",
            "EBSD pipeline",
        ]
    )
    base_signature = legacy._base_signature(dict(payload), transformation_id)
    _prune_component_cache_for_new_base(base_signature)

    with comparison_tab:
        _render_unified_v3(project, transformation_id, base_signature)
    with atlas_tab:
        legacy._render_atlas(project, transformation_id, base_signature)
    with pole_tab:
        legacy._render_poles(project, transformation_id)
    with independent_tab:
        legacy._render_invariant_and_double(project, transformation_id, base_signature)
    with ebsd_tab:
        legacy._render_ebsd(project, transformation_id, base_signature)
