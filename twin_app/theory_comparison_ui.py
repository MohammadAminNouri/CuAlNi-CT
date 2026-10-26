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


def render_theory_comparison(project_payload, transformation_id: str, signature: str) -> None:
    """Only run expensive independent calculations after an explicit button press."""
    st.divider()
    st.subheader("04  /  Compare crystallographic theories")
    with st.container(border=True):
        st.markdown("**First: what does each theory tell us?**")
        for name, description in THEORIES:
            st.markdown(f"**{name}** — {description}")
        st.caption(
            "The theories address overlapping but different physical questions. "
            "A twin interface (M/M) is not the same thing as an austenite–martensite "
            "habit plane (A/M). A missing prediction is N/A, not an invented zero."
        )
        with st.expander("Read more: methods and what counts as agreement", expanded=False):
            st.markdown(
                "**CT:** correspondence variants, operators and metric compatibility. "
                "**Ball–James:** stretch tensors, M/M and A/M rank-one interfaces. "
                "**PTMC:** lattice-invariant twinning, laminate fraction and A/M habit solutions. "
                "**Cofactor:** CC1 (middle stretch), CC2 and CC3 margins for BJ twin branches."
            )
            st.markdown(
                "The same entered lattice metrics and correspondence feed every independent "
                "backend. PTMC additionally assumes a lattice-invariant twin mechanism. "
                "No natural CT orientation relationship is supplied in this quick comparison. "
                "Numerical agreement is **not** experimental confirmation or proof of theory equivalence."
            )

        if st.button("Calculate theories and numerical checks (no EBSD)",
                     key="tf_theory_compare_run", type="secondary"):
            st.session_state.pop("tf_theory_comparison_result", None)
            try:
                from cualni_cryst.project_io import project_from_dict
                from cualni_cryst.theory_unified import PTMCMode, TheoryComparisonAdapter
                from cualni_cryst.theory_equivalence import build_equivalence_report

                project = project_from_dict(
                    dict(project_payload), source="twin_app:theory_comparison"
                ).project
                with st.spinner("Calculating separate theory branches and matching comparable physical quantities…"):
                    unified = TheoryComparisonAdapter(
                        project, transformation_id
                    ).compare(
                        ptmc_mode=PTMCMode.ALL_TWINNING,
                        include_ct_closing_gap=True,
                        include_ct_supercompatibility=True,
                    )
                    equivalence = build_equivalence_report(
                        project, transformation_id, unified,
                        include_independent_mm_audit=True,
                    )
                st.session_state["tf_theory_comparison_result"] = (
                    signature, unified, equivalence
                )
            except Exception as exc:
                st.error(
                    f"Independent comparison was not completed: {type(exc).__name__}: {exc}"
                )
                st.info("Existing twin-mode results remain unchanged. No missing theory prediction was filled in.")

        stored = st.session_state.get("tf_theory_comparison_result")
        if not stored or stored[0] != signature:
            st.caption(
                "Press Calculate to produce actual numerical results for this input. "
                "This covers the full transformation; it is not restricted to the selected Mode."
            )
            return

        _, unified, equivalence = stored
        st.markdown("#### 1. What each theory calculated")
        st.table(pd.DataFrame(overview_rows(unified)))

        st.markdown("#### 2. Where the theories can be compared")
        rows = matched_rows(equivalence)
        if rows:
            st.caption(
                f"{len(rows)} matched physical branches across the complete input transformation. "
                "Only the first 100 are displayed; download JSON for every branch. "
                "The assignment is geometric, not an overall theory ranking."
            )
            st.dataframe(
                pd.DataFrame(rows[:100]).fillna("N/A"),
                hide_index=True, width="stretch",
            )
        else:
            st.info("No independently matched branches for the selected inputs and assumptions. This is not proof that the theories disagree.")

        st.markdown("#### 3. Numerical verification — no EBSD needed")
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
