from __future__ import annotations

"""Read-only, opt-in comparison of already implemented CT/BJ/PTMC theory backends.

The numerical engines and twin-mode classifications are not modified here.
The comparison covers the whole input transformation, not the selected UI mode.
"""

import json

import pandas as pd
import streamlit as st


THEORIES = (
    ("Cayron CT", "Which variants, correspondence operators and interface candidates follow from symmetry and lattice correspondence?"),
    ("Ball–James", "Which individual martensite pairs and austenite interfaces satisfy nonlinear rank-one compatibility?"),
    ("PTMC", "Can a twinned martensite laminate form a compatible habit plane against the parent phase, and at what fraction?"),
    ("Cofactor conditions", "How closely does a Ball–James twin satisfy the stronger CC1, CC2 and CC3 compatibility conditions?"),
)


def overview_rows(unified):
    """Count calculated rows, not observed variants or independent experiments."""
    from cualni_cryst.theory_unified import TheoryKind
    bj_twins = unified.ball_james_report.martensite_twin_branches
    cofactor_all = sum(
        bool(b.cofactor.cc1_satisfied and b.cofactor.cc2_satisfied and b.cofactor.cc3_satisfied)
        for b in bj_twins
    )
    return [
        {"Approach": "Cayron CT", "Calculated prediction rows": len(unified.rows_for(TheoryKind.CAYRON_CT)),
         "Question": "Correspondence / interface geometry"},
        {"Approach": "Ball–James", "Calculated prediction rows": len(unified.rows_for(TheoryKind.BALL_JAMES)),
         "Question": "Rank-one compatibility"},
        {"Approach": "PTMC", "Calculated prediction rows": len(unified.rows_for(TheoryKind.PTMC)),
         "Question": "Twinned-laminate habit planes"},
        {"Approach": "Cofactor conditions", "Calculated prediction rows": len(bj_twins),
         "Question": f"BJ twin branches evaluated; {cofactor_all} satisfy CC1–CC3 at solver tolerance"},
    ]


def matched_rows(equivalence):
    """Preserve each match's physical family and raw, separately labelled residuals."""
    result = []
    for match in equivalence.matches:
        d = match.residuals.to_dict()
        result.append({
            "Physical question": match.family.value.replace("_", " "),
            "Theory A": match.left_theory.value.replace("_", " "),
            "Theory B": match.right_theory.value.replace("_", " "),
            "A branch": match.left_branch,
            "B branch": match.right_branch,
            "Habit plane gap (deg)": d["habit_plane_angle_deg"],
            "Twin plane gap (deg)": d["twin_plane_angle_deg"],
            "Twin shear relative gap": d["shear_relative"],
            "Rank-one tensor relative gap": d["rank_one_tensor_relative"],
            "OR gap (deg)": d["or_disorientation_deg"],
            "Evidence": match.completeness.value,
            "All required quantities within tolerance": match.all_required_components_within_tolerance,
        })
    return result


def _prediction_rows(unified):
    result = []
    for item in unified.table_rows():
        result.append({
            "Theory": item["theory"], "Prediction": item["kind"],
            "Branch": item["branch"],
            "A/M habit (parent)": str(item["habit_plane_parent"]),
            "M/M twin plane (product)": str(item["twin_plane_product"]),
            "Shear": str(item["shear"]),
            "Numerical residuals": str(item["residuals"]),
        })
    return result


def calculate_full_theory_report(project_payload, transformation_id, *, progress=None):
    """Re-use tested staged theory backends and optimized, physical-frame matching.

    Numerical algorithms, tolerances, and branch lists remain those of the backend.
    The independent CT/Mallard/Ball–James audit is retained, never silently omitted.
    """
    from dataclasses import replace
    import numpy as np
    from cualni_cryst.project_io import project_from_dict
    from cualni_cryst.theory_unified import PTMCMode
    from cualni_cryst.theory_unified_components import StagedTheoryComparisonAdapter
    from cualni_cryst.theory_equivalence_optimized import (
        build_equivalence_report_optimized,
        independent_mm_audit,
        attach_independent_mm_audit,
    )

    def announce(message):
        if progress is not None:
            progress(message)

    announce("Validating crystal structures and correspondence…")
    project = project_from_dict(
        dict(project_payload), source="twin_app:theory_comparison"
    ).project
    adapter = StagedTheoryComparisonAdapter(project, transformation_id)

    announce("Cayron CT: twins, closing-gap ORs and shear/shear compatibility…")
    ct = adapter.ct_component(
        include_closing_gap=True, include_supercompatibility=True,
    )
    announce("Ball–James: rank-one interfaces and cofactor conditions…")
    bj = adapter.ball_james_component(fraction_samples=101)
    announce("PTMC: all exact twinning mechanisms and habit solutions…")
    ptmc = adapter.ptmc_component(mode=PTMCMode.ALL_TWINNING)
    unified = adapter.assemble(ct, bj, ptmc)

    announce("Matching only physically comparable theoretical quantities…")
    equivalence = build_equivalence_report_optimized(
        project, transformation_id, unified,
    )

    announce("Independent M/M audit: CT versus Mallard versus Ball–James…")
    try:
        audit = independent_mm_audit(project, transformation_id)
    except (AssertionError, ValueError, np.linalg.LinAlgError) as exc:
        # A failed audit is an explicit warning, never a numerical pass.
        equivalence = replace(
            equivalence,
            warnings=(*equivalence.warnings,
                      f"Independent M/M audit not evaluable: {type(exc).__name__}: {exc}"),
        )
    else:
        equivalence = attach_independent_mm_audit(equivalence, audit)

    return unified, equivalence



def thesis_findings(unified, equivalence):
    """Answer the physical questions, not simply count raw data rows."""
    from cualni_cryst.theory_equivalence import MatchFamily
    from cualni_cryst.theory_unified import PredictionKind, TheoryKind

    def count(kind, exact=False):
        return sum(r.prediction_kind is kind and (not exact or r.exact is True)
                   for r in unified.rows)

    def compare(family, right):
        matches = [m for m in equivalence.matches
                   if m.family is family
                   and m.left_theory is TheoryKind.CAYRON_CT
                   and m.right_theory is right]
        return (
            sum(m.all_required_components_within_tolerance is True for m in matches),
            sum(m.all_required_components_within_tolerance is False for m in matches),
            sum(m.all_required_components_within_tolerance is None for m in matches),
        )

    mm_yes, mm_no, mm_partial = compare(MatchFamily.MM_TWIN, TheoryKind.BALL_JAMES)
    audit = equivalence.mm_independent_audit
    if audit is None:
        audit_text = "Independent M/M verification was not evaluable."
    elif (audit.get("success") is True and
          audit.get("ct_coverage_complete") is True and
          audit.get("classification_exact") is True):
        audit_text = "Independent CT/Mallard/Ball–James audit passed."
    else:
        audit_text = "Independent M/M audit did not confirm complete agreement."

    if mm_yes + mm_no + mm_partial:
        mm_text = (
            f"{mm_yes} complete CT/BJ twin assignments agree within tolerance, "
            f"{mm_no} disagree, and {mm_partial} are only partial. {audit_text}"
        )
    else:
        mm_text = "No CT/BJ twin branches were assigned for comparison. " + audit_text

    ct_habit = count(PredictionKind.CT_AM_HABIT, True)
    bj_habit = count(PredictionKind.BALL_JAMES_AM, True)
    ptmc_habit = count(PredictionKind.PTMC_HABIT, True)
    am_bj, _, _ = compare(MatchFamily.AM_INTERFACE, TheoryKind.BALL_JAMES)
    am_ptmc, _, _ = compare(MatchFamily.AM_INTERFACE, TheoryKind.PTMC)

    or_agreed, _, _ = compare(MatchFamily.ORIENTATION, TheoryKind.PTMC)
    or_angles = [m.residuals.or_disorientation_deg for m in equivalence.matches
                 if m.family is MatchFamily.ORIENTATION
                 and m.left_theory is TheoryKind.CAYRON_CT
                 and m.right_theory is TheoryKind.PTMC
                 and m.residuals.or_disorientation_deg is not None]
    or_text = (
        f"CT produced {count(PredictionKind.CT_CLOSING_GAP_OR)} OR candidate records; "
        f"{or_agreed} fully comparable CT/PTMC OR assignments agree."
    )
    if or_angles:
        or_text += f" Minimum assigned rotation difference: {min(or_angles):.5g}°."
    else:
        or_text += " No CT/PTMC OR angle could be compared."

    twins = unified.ball_james_report.martensite_twin_branches
    cc = sum(b.cofactor.cc1_satisfied and b.cofactor.cc2_satisfied
             and b.cofactor.cc3_satisfied for b in twins)
    eps = [r.residuals["ct_supercompatibility_dimensionless"]
           for r in unified.rows
           if r.prediction_kind is PredictionKind.CT_SUPERCOMPATIBILITY
           and "ct_supercompatibility_dimensionless" in r.residuals]
    strong = f"{cc}/{len(twins)} BJ twin records satisfy CC1–CC3 at numerical tolerances. "
    strong += (f"Minimum CT shear/shear ε = {min(eps):.5g}; ideal zero."
               if eps else "No CT shear/shear residual for these inputs.")

    return (
        ("M/M twins — what actually agrees?",
         mm_text,
         "CT twin constructions and BJ ordered branches are different records, "
         "not counts of unique observed twins. The independent M/M audit and "
         "the branch matcher test different criteria: an audit pass does not "
         "upgrade a partial assignment to full agreement. To establish full "
         "equivalence, compare plane normals, shears and coordinate frames."),
        ("A/M habit planes — why do the counts differ?",
         f"SINGLE-VARIANT exact A/M records: CT {ct_habit}, BJ {bj_habit}. "
         f"TWO-VARIANT LAMINATE exact A/M records: PTMC {ptmc_habit}. "
         f"Complete CT/BJ A/M assignments: {am_bj}; "
         f"complete CT/PTMC A/M assignments: {am_ptmc}.",
         "These are NOT counts of the same physical problem. PTMC uses a twinned "
         "laminate with a fraction f, whereas the displayed CT/BJ checks "
         "concern single-variant A/M interfaces. Zero single-variant habits "
         "and many laminate habits are therefore NOT contradictory. "
         "PTMC records can include symmetry/sign-equivalent solutions; "
         "384 records are not 384 proven distinct physical planes. "
         "Next: group equivalent PTMC solutions by habit normal and twin pair."),
        ("Do CT and PTMC orientation predictions agree?",
         or_text,
         "The smallest assigned angle is NOT confirmation of theory agreement "
         "when no full match passes. Compare candidate rotations in a common "
         "coordinate frame and under both phases' crystal symmetries, "
         "then state an explicit angular tolerance."),
        ("Is the transformation especially compatible?",
         strong,
         "Cofactor conditions demand stronger compatibility than ordinary "
         "rank-one twinning. Failing CC1–CC3 does NOT invalidate BJ twins or "
         "PTMC laminate habits. If CT shear/shear is unavailable, it was "
         "NOT tested and should not be interpreted as a failed condition. "
         "Next: inspect CC1, CC2, CC3 separately and vary the lattice metrics."),
    )


def thesis_notes(unified, equivalence, project_payload, signature):
    """Export editable numerical findings with explicit limits."""
    lines = [
        "# Crystallographic theory comparison — numerical thesis working notes",
        "",
        f"Input fingerprint: {signature}",
        "Scope: the whole transformation, not only the selected twin mode.",
        "All counts refer to prediction records, not observed variants.",
        "",
        "## Input crystallography",
    ]
    for phase in project_payload.get("phases", ()):
        cell = phase.get("cell", {})
        metrics = ", ".join(
            f"{k}={cell[k]}" for k in
            ("a", "b", "c", "alpha_deg", "beta_deg", "gamma_deg") if k in cell
        )
        lines.append(f"- {phase.get('phase_id', '?')}: point group "
                     f"{phase.get('point_group', '?')}, {metrics}, "
                     f"unit={cell.get('length_unit', '?')}")
    lines.append("")
    lines.append("## Numerical findings and defensible interpretation")
    for question, result, meaning in thesis_findings(unified, equivalence):
        lines.extend(("", "### " + question, result, "", meaning))
    from twin_app.theory_scientific_explanations import scientific_notes
    lines.extend(("", scientific_notes(unified, equivalence)))
    lines.extend((
        "", "## Scientific limitations",
        "The findings assume the entered lattice metrics, crystal settings, "
        "correspondence matrix and numerical tolerances. The calculations "
        "do not prove experimentally formed variants, observed habit planes, "
        "actual variant selection, kinetics, or hysteresis.",
        "Unmatched and partial predictions cannot establish which theory is wrong.",
        "Report independent residuals and sensitivity to lattice parameters "
        "before treating these notes as thesis results.",
        "", "## Backend warnings",
    ))
    lines.extend("- " + x for x in (*unified.warnings, *equivalence.warnings))
    return "\n".join(lines) + "\n"


def render_theory_comparison(project_payload, transformation_id: str, signature: str) -> None:
    """Only run expensive independent calculations after an explicit button press."""
    st.divider()
    st.subheader("04  /  Compare crystallographic theories")
    with st.container(border=True):
        st.caption(
            "Interpret the numerical evidence for this transformation. "
            "Do not compare a martensite twin (M/M), a single-variant "
            "austenite interface (A/M), and a PTMC two-variant laminate "
            "as if they were identical problems."
        )
        with st.expander("Which physical question does each theory answer?", expanded=False):
            for name, description in THEORIES:
                st.markdown(f"**{name}** — {description}")
        with st.expander("Methods, assumptions and agreement criteria", expanded=False):
            st.markdown(
                "**CT:** correspondence variants, operators and metric compatibility. "
                "**Ball–James:** stretch tensors, M/M and A/M rank-one interfaces. "
                "**PTMC:** lattice-invariant twinning, laminate fraction and A/M habit solutions. "
                "**Cofactor:** CC1 (middle stretch), CC2 and CC3 margins for BJ twin branches."
            )
            st.markdown(
                "The same entered lattice metrics and correspondence feed every independent "
                "backend. PTMC additionally assumes a lattice-invariant twin mechanism. "
                "Without an independent natural OR, closing-gap solutions remain candidates. "
                "Numerical agreement is **not** experimental confirmation or proof of theory equivalence."
            )

        if st.button("Calculate theories and numerical checks (no EBSD)",
                     key="tf_theory_compare_run", type="secondary"):
            saved = st.session_state.get("tf_theory_comparison_result")
            if saved is not None and saved[0] == signature:
                st.success("Numerical comparison is already calculated for these inputs.")
            else:
                st.session_state.pop("tf_theory_comparison_result", None)
                try:
                    with st.status("Calculating theories", expanded=True) as status:
                        unified, equivalence = calculate_full_theory_report(
                            project_payload, transformation_id, progress=status.write,
                        )
                        status.update(
                            label="Theory calculations and numerical checks completed",
                            state="complete", expanded=False,
                        )
                    st.session_state["tf_theory_comparison_result"] = (
                        signature, unified, equivalence
                    )
                except Exception as exc:
                    st.error(
                        f"Independent comparison was not completed: {type(exc).__name__}: {exc}"
                    )
                    st.info(
                        "Your existing twin results are unchanged. "
                        "No theoretical prediction was invented."
                    )

        stored = st.session_state.get("tf_theory_comparison_result")
        if not stored or stored[0] != signature:
            st.caption(
                "Press Calculate to produce actual numerical results for this input. "
                "This covers the full transformation; it is not restricted to the selected Mode."
            )
            return

        _, unified, equivalence = stored
        st.markdown("### Four scientifically defensible findings")
        st.caption(
            "Answers concern all theoretical variants, not only the selected Mode. "
            "No experimental EBSD is needed."
        )
        for i, (question, result, meaning) in enumerate(
            thesis_findings(unified, equivalence), start=1
        ):
            st.markdown(f"**{i}. {question}**")
            st.write(result)
            st.caption("Interpretation and useful next numerical check: " + meaning)
        with st.expander("How to turn these results into thesis evidence", expanded=False):
            st.markdown(
                "1. **Unique PTMC habits:** reduce sign/symmetry-equivalent branches "
                "and report actual unique habit-plane families.\n"
                "2. **Comparable twins:** transform CT/BJ interface observables "
                "to common coordinate frames and evaluate full residuals.\n"
                "3. **Robustness:** rerun with justified 2H/18R lattice-parameter "
                "ranges; identify which predictions persist."
            )
        from twin_app.theory_scientific_explanations import render_scientific_explanations
        render_scientific_explanations(unified, equivalence)

        st.download_button(
            "Download numerical thesis notes (.md)",
            data=thesis_notes(unified, equivalence, project_payload, signature),
            file_name="theory_thesis_notes.md",
            mime="text/markdown",
            key="tf_thesis_notes",
        )

        with st.expander("Detailed theory predictions and matches (advanced)", expanded=False):
            st.markdown("**Calculated prediction records (not observed twins)**")
            st.table(pd.DataFrame(overview_rows(unified)))
            st.markdown("**Numerical assignments and residuals**")
            rows = matched_rows(equivalence)
            if rows:
                st.caption(
                    f"{len(rows)} matched physical branches across the complete input transformation. "
                    "Only the first 100 are displayed; download JSON for every branch. "
                    "The assignment is geometric, not an overall theory ranking."
                )
                st.dataframe(
                    pd.DataFrame(rows[:100]),
                    hide_index=True, width="stretch",
                )
            else:
                st.info("No independently matched branches for the selected inputs and assumptions. This is not proof that the theories disagree.")

        st.markdown("#### Further independent numerical checks — no EBSD needed")
        from twin_app.theory_numerical_ui import render_numerical_checks
        render_numerical_checks(unified, equivalence)

        with st.expander("Explore all calculated predictions (advanced)", expanded=False):
            predictions = _prediction_rows(unified)
            st.caption(
                f"{len(predictions)} total branch records; first 150 shown here. "
                "Source records and their numerical residuals are preserved in the JSON export."
            )
            st.dataframe(
                pd.DataFrame(predictions[:150]).fillna("N/A"),
                hide_index=True, width="stretch",
            )
            if equivalence.unmatched:
                st.warning(
                    f"{len(equivalence.unmatched)} branches were unmatched. "
                    "Unmatched does not automatically mean mathematically incorrect."
                )
                st.dataframe(
                    pd.DataFrame([x.to_dict() for x in equivalence.unmatched[:60]]),
                    hide_index=True, width="stretch",
                )

        with st.expander("Ball–James cofactor CC1 / CC2 / CC3 (advanced)", expanded=False):
            twin_branches = unified.ball_james_report.martensite_twin_branches
            if not twin_branches:
                st.info("No Ball–James martensite–martensite branches for the supplied inputs.")
            else:
                records = []
                for twin in twin_branches:
                    c = twin.cofactor
                    records.append({
                        "Stretch pair": f"U{twin.base_variant_index}–U{twin.other_variant_index}",
                        "Branch": twin.branch,
                        "CC1 λ₂−1": c.cc1_lambda2_minus_one,
                        "CC2 value": c.cc2_value,
                        "CC3 margin": c.cc3_margin,
                        "CC1 / CC2 / CC3 passed": (
                            c.cc1_satisfied and c.cc2_satisfied and c.cc3_satisfied
                        ),
                    })
                st.dataframe(
                    pd.DataFrame(records[:100]), hide_index=True, width="stretch"
                )
                st.caption("These are calculated compatibility conditions, not experimentally measured reversibility.")

        if unified.warnings or equivalence.warnings:
            with st.expander("Scientific limitations and warnings", expanded=False):
                for warning in (*unified.warnings, *equivalence.warnings):
                    st.warning(warning)

        st.download_button(
            "Download complete theory comparison (JSON)",
            data=json.dumps({
                "input_signature": signature,
                "scope": "whole_transformation_not_selected_mode",
                "unified_report": unified.to_dict(),
                "branch_comparison": equivalence.to_dict(),
            }, indent=2, default=str),
            file_name="independent_theory_comparison.json",
            mime="application/json",
            key="tf_theory_compare_export",
        )
        st.caption(
            "The EBSD/CTF input is above, under Crystal setup. "
            "Its orientation-operator matches do not by themselves certify physical twins."
        )
