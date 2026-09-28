from __future__ import annotations

"""Question-first research extension for CT/Ball--James/PTMC/experiment work.

Scientific calculations remain in the existing backends. This module changes
only orchestration visibility and presentation: conclusions first, derivation
and evidence second, exhaustive branch bookkeeping last.
"""

from dataclasses import asdict
import json
from typing import Any, Mapping

import pandas as pd
import streamlit as st

import app.research_workspaces as legacy
import app.research_workspaces_v3 as v3
from app.scientific_interpretation import (
    am_existence_finding,
    cofactor_finding,
    experiment_finding,
    mm_audit_finding,
    or_comparison_finding,
    ptmc_finding,
    supercompatibility_finding,
)
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
)
from app.session_state import fingerprint
from cualni_cryst.project_io import project_from_dict
from cualni_cryst.theory_equivalence import EquivalenceTolerances, MatchCompleteness
from cualni_cryst.theory_unified import PTMCMode, PredictionKind, TheoryKind



def _latest_bound_value(key: str):
    item = st.session_state.get(key)
    return getattr(item, "value", None)


def _render_atlas_summary(report) -> None:
    total = sum(int(v) for v in report.counts.values())
    both = int(report.counts.get("both", 0))
    ct_only = int(report.counts.get("ct_only", 0))
    cofactor_only = int(report.counts.get("cofactor_only", 0))
    neither = int(report.counts.get("neither", 0))
    not_eval = int(report.counts.get("not_evaluable", 0))
    from app.scientific_types import Evidence, Finding
    render_finding(Finding(
        title="Compatibility map summary",
        conclusion=f"{both}/{total} sampled lattice states satisfy both CT supercompatibility and the cofactor classification.",
        rationale=f"CT-only={ct_only}, cofactor-only={cofactor_only}, neither={neither}, not evaluable={not_eval}.",
        how="Only symmetry-independent product-cell parameters are varied. At every sampled state, CT and cofactor conditions are evaluated independently and then classified on the matched physical relation.",
        physical_meaning="The map shows where the two compatibility descriptions occupy the same or different regions of lattice-parameter space.",
        limitation="A sampled compatibility locus is not an experimental fit unless measured lattice parameters and uncertainties are explicitly overlaid.",
        evidence=(
            Evidence("Sampled states", "requested grid", total, ""),
            Evidence("Both", "CT and cofactor satisfied", both, ""),
            Evidence("CT only", "CT satisfied, cofactor not", ct_only, ""),
            Evidence("Cofactor only", "cofactor satisfied, CT not", cofactor_only, ""),
            Evidence("Neither", "neither condition satisfied", neither, ""),
        ),
        tone="neutral",
    ))


def _render_invariant_summary(report) -> None:
    from app.scientific_types import Evidence, Finding
    solutions = list(getattr(report, "solutions", ()) or ())
    if solutions:
        max_inv = max(abs(float(item.invariant_residual)) for item in solutions)
        max_det = max(abs(float(item.determinant_residual)) for item in solutions)
        render_finding(Finding(
            title="Independent invariant-line result",
            conclusion=f"{len(solutions)} real invariant-line branch(es) satisfy the supplied correlated-pair construction.",
            rationale=f"Maximum invariant-vector residual = {sci(max_inv,3)}; maximum det(RD−I) residual = {sci(max_det,3)}.",
            how="The invariant-line solver uses only the supplied geometric constraints and correlated pairs. It does not call CT, PTMC or a preselected OR to create the result.",
            physical_meaning="These branches are an independent geometric comparator that can later be checked against CT/PTMC OR predictions.",
            limitation="Agreement with another theory must be established by a separate physical OR comparison; this solver is not claimed to be Cayron CT or PTMC.",
            evidence=(
                Evidence("Real branches", "admissible invariant-line solutions", len(solutions), ""),
                Evidence("Max invariant residual", "≈ 0", sci(max_inv,3), ""),
                Evidence("Max determinant residual", "≈ 0", sci(max_det,3), ""),
            ),
            tone="good",
        ))
    else:
        render_finding(Finding(
            title="Independent invariant-line result",
            conclusion="No real invariant-line branch exists for the supplied correlated pairs.",
            rationale="The independent geometric constraints do not admit a real branch under the current inputs.",
            how="The solver tests the correlated pairs and constraint directly, without borrowing a CT/PTMC branch.",
            physical_meaning="This particular geometric hypothesis is not satisfied for the chosen inputs.",
            limitation="Changing the correlated pairs changes the hypothesis; absence here is not a general statement about the phase transformation.",
            tone="warn",
        ))


def _render_double_summary(report) -> None:
    from app.scientific_types import Evidence, Finding
    solutions = list(getattr(report, "solutions", ()) or ())
    continuum = getattr(report, "continuum", None)
    semantics = getattr(getattr(report, "parameter_semantics", None), "value", getattr(report, "parameter_semantics", "—"))
    if continuum is not None:
        conclusion = "The selected two-LIS construction admits a continuum of compatible parameter values."
        tone = "good"
    elif solutions:
        conclusion = f"The selected two-LIS construction gives {len(solutions)} compatible parameter root(s)."
        tone = "good"
    else:
        conclusion = "No compatible double-shear root exists at the selected fixed parameter."
        tone = "warn"
    residuals = [abs(float(item.middle_stretch_residual)) for item in solutions]
    best = min(residuals) if residuals else None
    render_finding(Finding(
        title="Double-shear PTMC result",
        conclusion=conclusion,
        rationale=f"Parameter semantics: {semantics}; best middle-stretch residual = {sci(best,3)}.",
        how="The two rank-one LIS increments are composed according to the explicitly selected additive-laminate or sequential finite-shear model. Their parameters retain that model's native semantics.",
        physical_meaning="This tests whether two independent lattice-invariant shears can restore a macroscopic compatibility condition beyond the single-shear case.",
        limitation="Finite-shear multipliers are not phase fractions unless the selected composition explicitly defines three-variant fractions.",
        evidence=(
            Evidence("Discrete roots", "compatible solutions", len(solutions), ""),
            Evidence("Continuum", "native solver result", "yes" if continuum is not None else "no", ""),
            Evidence("Best middle-stretch residual", "≈ 0", sci(best,3), ""),
        ),
        tone=tone,
    ))


def _render_ebsd_summary(result) -> None:
    from app.scientific_types import Evidence, Finding
    summary = result.get("summary", {}) if isinstance(result, Mapping) else {}
    segmentation = summary.get("segmentation", {}) if isinstance(summary, Mapping) else {}
    boundaries = summary.get("boundaries", {}) if isinstance(summary, Mapping) else {}
    parent = summary.get("parent_reconstruction", {}) if isinstance(summary, Mapping) else {}
    n_grains = segmentation.get("n_grains", "—")
    n_bound = boundaries.get("n_product_product_boundaries", "—")
    accepted = boundaries.get("n_operator_accepted", "—")
    ambiguous = boundaries.get("n_operator_ambiguous", "—")
    candidates = parent.get("n_domain_candidates", "—")
    render_finding(Finding(
        title="EBSD reconstruction summary",
        conclusion=f"The completed run segmented {n_grains} grain(s) and evaluated {n_bound} product/product boundary relation(s).",
        rationale=f"Accepted operator assignments={accepted}; ambiguous assignments={ambiguous}; parent-domain candidates={candidates}.",
        how="The pipeline applies only the explicit file convention, phase mapping, quality filters, segmentation settings, OR hypothesis/refinement bounds and reconstruction thresholds supplied in the run configuration.",
        physical_meaning="These counts describe how much of the measured map can be assigned consistently to the crystallographic hypotheses without forcing ambiguous cases.",
        limitation="Accepted/ambiguous counts are threshold-dependent; the configuration and file hash must accompany any interpretation.",
        evidence=(
            Evidence("Grains", "segmentation result", n_grains, ""),
            Evidence("Product/product boundaries", "boundary graph", n_bound, ""),
            Evidence("Accepted operators", "configured residual/margin thresholds", accepted, ""),
            Evidence("Ambiguous operators", "ambiguity retained", ambiguous, ""),
            Evidence("Parent-domain candidates", "reconstruction result", candidates, ""),
        ),
        tone="neutral",
    ))

def _science_match_table(equivalence) -> pd.DataFrame:
    rows = legacy._match_rows(equivalence)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    preferred = [
        "family",
        "left theory",
        "left branch",
        "right theory",
        "right branch",
        "habit Δ (deg)",
        "shape Δ projective (deg)",
        "shape magnitude rel.",
        "rank-one tensor rel.",
        "twin-plane Δ (deg)",
        "twin-direction Δ (deg)",
        "shear rel.",
        "OR disorientation (deg)",
        "comparison",
        "agreement",
        "missing required",
    ]
    cols = [name for name in preferred if name in frame.columns]
    return format_dataframe_scientific(frame[cols])


def _prediction_inventory(unified) -> pd.DataFrame:
    return v3._prediction_inventory(unified)


def _calculation_scope() -> tuple[bool, bool, str, bool, EquivalenceTolerances]:
    with st.expander("Calculation scope", expanded=True):
        st.caption("Only enabled theories/extensions are computed. Heavy PTMC is opt-in and cached separately from CT and Ball–James.")
        controls = st.columns(4)
        include_closing = controls[0].checkbox(
            "CT closing-gap ORs",
            value=False,
            key="research_v4_ct_closing",
            help="Generate every CT closing-gap candidate unless a natural OR is explicitly supplied. No preferred OR is fabricated.",
        )
        include_super = controls[1].checkbox(
            "CT supercompatibility",
            value=False,
            key="research_v4_ct_super",
            help="Evaluate the CT A/M/M shear–shear condition only from exact CT A/M habits.",
        )
        ptmc_mode = controls[2].selectbox(
            "PTMC",
            options=[PTMCMode.NONE.value, PTMCMode.ALL_TWINNING.value],
            index=0,
            format_func=lambda value: "Off" if value == PTMCMode.NONE.value else "All twinning LIS — deep run",
            key="research_v4_ptmc_mode",
        )
        include_audit = controls[3].checkbox(
            "Independent M/M audit",
            value=True,
            key="research_v4_mm_audit",
            help="Authoritative CT ↔ Mallard ↔ Ball–James physical twin validation.",
        )

        with st.expander("Numerical agreement tolerances", expanded=False):
            st.caption("These thresholds classify already-computed cross-theory residuals; changing them does not alter the native theory calculations.")
            t = st.columns(5)
            plane = t[0].number_input("Plane Δ (deg)", min_value=1e-10, value=1e-4, format="%.8g", key="research_v4_tol_plane")
            direction = t[1].number_input("Direction Δ (deg)", min_value=1e-10, value=1e-4, format="%.8g", key="research_v4_tol_direction")
            magnitude = t[2].number_input("Magnitude relative", min_value=1e-12, value=1e-6, format="%.8g", key="research_v4_tol_mag")
            tensor = t[3].number_input("Rank-one relative", min_value=1e-12, value=1e-6, format="%.8g", key="research_v4_tol_tensor")
            or_tol = t[4].number_input("OR Δ (deg)", min_value=1e-10, value=1e-4, format="%.8g", key="research_v4_tol_or")
        tolerances = EquivalenceTolerances(
            plane_angle_deg=float(plane),
            direction_angle_deg=float(direction),
            relative_magnitude=float(magnitude),
            rank_one_tensor_relative=float(tensor),
            or_disorientation_deg=float(or_tol),
        )
    return include_closing, include_super, ptmc_mode, include_audit, tolerances


def _run_or_load(project, transformation_id: str, base_signature: str, *, include_closing: bool, include_super: bool, ptmc_mode: str, include_audit: bool, tolerances: EquivalenceTolerances):
    try:
        experiments = legacy._build_experiments(project, transformation_id, base_signature)
    except Exception as exc:
        legacy._render_failure("Experimental observation is invalid", exc)
        experiments = tuple()

    experiment_payload = [asdict(item) for item in experiments]
    theory_signature = fingerprint(
        base_signature,
        "unified-theory-v4",
        include_closing,
        include_super,
        ptmc_mode,
        experiment_payload,
    )
    equivalence_signature = fingerprint(theory_signature, "physical-equivalence-v4", tolerances.to_dict())
    audit_signature = fingerprint(base_signature, "independent-mm-audit-v4")

    unified = v3._component_get("research_unified_report_v4", theory_signature)
    equivalence = v3._component_get("research_equivalence_report_v4", equivalence_signature)
    audit = v3._component_get("research_mm_audit_v4", audit_signature) if include_audit else None

    if st.button("Calculate / update", type="primary", key="research_v4_run", use_container_width=True):
        try:
            with st.status("Running the requested independent theory calculations…", expanded=True) as status:
                if unified is None:
                    unified, _ = v3._assemble_unified_report(
                        project,
                        transformation_id,
                        base_signature,
                        include_closing=include_closing,
                        include_super=include_super,
                        ptmc_mode=ptmc_mode,
                        experiments=experiments,
                        status=status,
                    )
                    v3._component_put("research_unified_report_v4", theory_signature, unified)
                else:
                    status.write("Native theory results reused from cache.")

                if equivalence is None:
                    status.write("Cross-theory physical residuals: assigning comparable exact branches…")
                    equivalence = v3.build_equivalence_report_optimized(
                        project,
                        transformation_id,
                        unified,
                        tolerances=tolerances,
                    )
                    v3._component_put("research_equivalence_report_v4", equivalence_signature, equivalence)
                else:
                    status.write("Cross-theory residuals reused from cache.")

                if include_audit and audit is None:
                    status.write("Independent M/M audit: CT ↔ Mallard ↔ Ball–James…")
                    audit = v3.independent_mm_audit(project, transformation_id)
                    v3._component_put("research_mm_audit_v4", audit_signature, audit)
                elif include_audit:
                    status.write("Independent M/M audit reused from cache.")
                status.update(label="Calculation complete", state="complete", expanded=False)
        except Exception as exc:
            legacy._render_failure("Theory comparison failed", exc)

    return unified, equivalence, audit, experiments, theory_signature


def _render_conclusions(project, transformation_id: str, base_signature: str) -> None:
    section_header(
        "Theory comparison",
        "For this exact crystallographic state, which physical predictions agree, which do not, and why?",
        answer_hint="A candidate branch assignment is never treated as evidence by itself. Each conclusion below states its native criterion and numerical residual.",
    )
    include_closing, include_super, ptmc_mode, include_audit, tolerances = _calculation_scope()
    if ptmc_mode == PTMCMode.ALL_TWINNING.value:
        st.info("Deep PTMC is enabled. Its first calculation may be slow; later runs reuse the cached branch set unless the crystallographic state changes.")

    unified, equivalence, audit, experiments, theory_signature = _run_or_load(
        project,
        transformation_id,
        base_signature,
        include_closing=include_closing,
        include_super=include_super,
        ptmc_mode=ptmc_mode,
        include_audit=include_audit,
        tolerances=tolerances,
    )
    if unified is None or equivalence is None:
        st.info("Choose the scope and calculate. No scientific conclusion is shown before the requested theories have actually been evaluated.")
        return

    if include_audit:
        equivalence = v3.attach_independent_mm_audit(equivalence, audit)

    # Keep compatibility with the atlas/pole/double-shear/EBSD tabs.
    st.session_state["research_current_unified_signature"] = theory_signature
    legacy._bound_put("research_unified_report", theory_signature, unified)
    legacy._bound_put("research_equivalence_report", theory_signature, equivalence)

    checks = equivalence.classification_checks
    exact_ct_habits = sum(
        row.theory is TheoryKind.CAYRON_CT and row.prediction_kind is PredictionKind.CT_AM_HABIT and row.exact is True
        for row in unified.rows
    )
    approx_ct_habits = sum(
        row.theory is TheoryKind.CAYRON_CT and row.prediction_kind is PredictionKind.CT_AM_HABIT and row.exact is False
        for row in unified.rows
    )
    bj_am = sum(row.theory is TheoryKind.BALL_JAMES and row.prediction_kind is PredictionKind.BALL_JAMES_AM for row in unified.rows)

    st.markdown("### Scientific conclusions")
    render_finding(am_existence_finding(checks, exact_ct_habits=exact_ct_habits, bj_am=bj_am, approx_ct_habits=approx_ct_habits))

    mm = mm_audit_finding(audit if include_audit else None)
    if mm is not None:
        render_finding(mm)

    render_finding(ptmc_finding(unified, enabled=ptmc_mode != PTMCMode.NONE.value))
    render_finding(
        or_comparison_finding(
            equivalence,
            enabled=include_closing and ptmc_mode != PTMCMode.NONE.value,
            tolerance_deg=tolerances.or_disorientation_deg,
        )
    )
    render_finding(cofactor_finding(unified))
    algebraic_tol = float(getattr(getattr(project, "numerical_policy", None), "algebraic", 1.0e-10))
    render_finding(supercompatibility_finding(unified, requested=include_super, algebraic_tolerance=algebraic_tol))
    exp = experiment_finding(equivalence)
    if exp is not None:
        render_finding(exp)
    elif experiments:
        st.warning("Experimental observations were supplied, but no comparable theory/experiment residual record was produced. This is not interpreted as agreement or disagreement.")
    else:
        st.caption("No experimental observation is attached yet. At this stage the workstation compares theories with one another, not which theory best describes the specimen.")

    st.markdown("### Compact evidence")
    ptmc_exact = sum(row.theory is TheoryKind.PTMC and row.prediction_kind is PredictionKind.PTMC_HABIT and row.exact is True for row in unified.rows)
    super_rows = [row for row in unified.rows if row.prediction_kind is PredictionKind.CT_SUPERCOMPATIBILITY]
    compact_metrics([
        ("Exact CT A/M habits", exact_ct_habits),
        ("Exact BJ A/M branches", bj_am),
        ("Exact PTMC laminate branches", ptmc_exact if ptmc_mode != PTMCMode.NONE.value else "not run"),
        ("Independent M/M relations", int(audit.get("relation_count", 0)) if audit else "not run"),
        ("CT supercompatibility rows", len(super_rows) if include_super else "not run"),
    ], columns=5)

    with audit_expander("Branch-level evidence", expanded=False):
        frame = _science_match_table(equivalence)
        if frame.empty:
            st.info("No exact cross-theory branch pair exposed enough common physical observables for assignment.")
        else:
            st.caption("These are one-to-one physical assignments. 'Partial' means a required observable is intentionally not compared because the native configurations differ; it is not a theory failure.")
            st.dataframe(frame, hide_index=True, use_container_width=True)

        diagnostics = v3._diagnostic_nearest_rows(project, transformation_id, unified, tolerances)
        if diagnostics:
            st.markdown("#### Approximate CT A/M diagnostics")
            st.warning("Nearest-degeneracy CT rows are diagnostic only and are excluded from exact-equivalence claims.")
            st.dataframe(format_dataframe_scientific(pd.DataFrame(diagnostics)), hide_index=True, use_container_width=True)

        experiment_table = legacy._experiment_rows(equivalence)
        if experiment_table:
            st.markdown("#### Theory ↔ experiment residuals")
            st.dataframe(format_dataframe_scientific(pd.DataFrame(experiment_table)), hide_index=True, use_container_width=True)

    with audit_expander("Native theory inventory and provenance", expanded=False):
        st.dataframe(_prediction_inventory(unified), hide_index=True, use_container_width=True)
        for warning in unified.warnings:
            st.warning(warning)
        for note in unified.notes:
            st.caption(str(note))

    show_bookkeeping = st.checkbox("Show developer matcher bookkeeping", value=False, key="research_v4_show_bookkeeping")
    if show_bookkeeping:
        with audit_expander("Matcher bookkeeping — not a scientific conclusion", expanded=True):
            st.warning("Unassigned rows are family-specific matcher bookkeeping. They do not mean that a theory branch is physically wrong, and disabled/not-comparable pathways must not be interpreted as failures.")
            unique = v3._unique_unmatched_rows(equivalence)
            if unique:
                st.dataframe(pd.DataFrame(unique), hide_index=True, use_container_width=True)
            with st.expander("All native prediction rows", expanded=False):
                st.dataframe(legacy._scalar_table(unified.table_rows()), hide_index=True, use_container_width=True)

    exports = st.columns(2)
    exports[0].download_button(
        "Unified native theory report",
        data=json.dumps(unified.to_dict(), indent=2, default=str),
        file_name="cualni_ct_unified_theory_report.json",
        mime="application/json",
        use_container_width=True,
        key="research_v4_download_unified",
    )
    exports[1].download_button(
        "Physical equivalence report",
        data=json.dumps(equivalence.to_dict(), indent=2, default=str),
        file_name="cualni_ct_physical_equivalence_report.json",
        mime="application/json",
        use_container_width=True,
        key="research_v4_download_equivalence",
    )


def _render_atlas(project, transformation_id: str, base_signature: str) -> None:
    section_header(
        "Compatibility map",
        "If the product lattice changes slightly, where do CT supercompatibility and the classical cofactor conditions appear or disappear?",
        answer_hint="Selected lattice parameters are varied without breaking the registered crystal setting; CT and cofactor conditions are solved independently at every state.",
    )
    existing = _latest_bound_value("research_atlas_report")
    if existing is not None:
        _render_atlas_summary(existing)
    with st.expander("Sweep controls and full atlas analysis", expanded=existing is None):
        legacy._render_atlas(project, transformation_id, base_signature)
    if existing is None:
        created = _latest_bound_value("research_atlas_report")
        if created is not None:
            _render_atlas_summary(created)

def _render_poles(project, transformation_id: str) -> None:
    section_header(
        "Pole figures",
        "Do predicted habit planes, twin systems or OR-transformed pole families overlap in the same reference frame?",
        answer_hint="Pole overlap is a geometric comparison. It does not by itself prove two theories have the same derivation or physical mechanism.",
    )
    with st.expander("Pole selection and stereographic plot", expanded=True):
        legacy._render_poles(project, transformation_id)

def _render_independent(project, transformation_id: str, base_signature: str) -> None:
    section_header(
        "Invariant-line & double-shear checks",
        "What additional compatibility is obtained by an independent invariant-line condition or by two lattice-invariant shears?",
        answer_hint="These are independent comparators. Their parameters retain their native meaning; shear multipliers are never relabelled as phase fractions unless the composition model actually defines them that way.",
    )
    inv_existing = _latest_bound_value("research_invariant_report")
    ds_existing = _latest_bound_value("research_double_shear_report")
    if inv_existing is not None:
        _render_invariant_summary(inv_existing)
    if ds_existing is not None:
        _render_double_summary(ds_existing)
    with st.expander("Inputs and full independent-comparator analysis", expanded=inv_existing is None and ds_existing is None):
        legacy._render_invariant_and_double(project, transformation_id, base_signature)
    if inv_existing is None:
        created = _latest_bound_value("research_invariant_report")
        if created is not None:
            _render_invariant_summary(created)
    if ds_existing is None:
        created = _latest_bound_value("research_double_shear_report")
        if created is not None:
            _render_double_summary(created)

def _render_ebsd(project, transformation_id: str, base_signature: str) -> None:
    section_header(
        "EBSD ↔ theory",
        "Which theory branches approach the measured orientations, boundaries and traces after every convention and threshold is made explicit?",
        answer_hint="This is ultimately the decisive layer: theory-to-theory agreement is useful, but measured residuals determine which branch describes the specimen.",
    )
    existing = _latest_bound_value("research_ebsd_result")
    if existing is not None:
        _render_ebsd_summary(existing)
    with st.expander("EBSD run configuration and full analysis", expanded=existing is None):
        legacy._render_ebsd(project, transformation_id, base_signature)
    if existing is None:
        created = _latest_bound_value("research_ebsd_result")
        if created is not None:
            _render_ebsd_summary(created)

def render_research_extension() -> None:
    install_styles()
    with st.sidebar:
        custom_sidebar_navigation(current="research")

    page_header(
        "Theory comparison laboratory",
        "CT, Ball–James/Mallard, PTMC and experiment are kept independent until their physical observables are explicitly brought into a common comparison frame.",
        kicker="CuAlNi-CT research workspace",
    )

    payload = st.session_state.get("current_project_payload")
    if payload is None:
        st.info("No calculated crystallographic state is active. Define and calculate the state in Workbench first.")
        return
    if st.session_state.get("requires_recalculation", False):
        st.warning("The Setup draft changed after the last calculation. Recalculate before running theory comparisons; stale results are intentionally blocked.")
        return

    try:
        loaded = project_from_dict(dict(payload), source="streamlit:ct-equivalence-laboratory-v4")
        project = loaded.project
    except Exception as exc:
        st.error(f"The active ProjectState cannot be loaded: {exc}")
        return

    ids = [item.transformation_id for item in project.transformations]
    if not ids:
        st.info("The active project contains no transformation.")
        return
    transformation_id = ids[0] if len(ids) == 1 else st.selectbox("Transformation", ids, key="research_v4_transformation_id")
    base_signature = legacy._base_signature(dict(payload), transformation_id)
    v3._prune_component_cache_for_new_base(base_signature)

    comparison_tab, atlas_tab, pole_tab, independent_tab, ebsd_tab = st.tabs([
        "Conclusions",
        "Compatibility map",
        "Pole figures",
        "Invariant line / double shear",
        "EBSD ↔ theory",
    ])
    with comparison_tab:
        _render_conclusions(project, transformation_id, base_signature)
    with atlas_tab:
        _render_atlas(project, transformation_id, base_signature)
    with pole_tab:
        _render_poles(project, transformation_id)
    with independent_tab:
        _render_independent(project, transformation_id, base_signature)
    with ebsd_tab:
        _render_ebsd(project, transformation_id, base_signature)
