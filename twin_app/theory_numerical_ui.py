from __future__ import annotations

"""Concise, read-only numerical CT/OR/M-M checks from the existing theory reports.

Never reruns, changes, or replaces the scientific calculations. No EBSD required.
"""

import pandas as pd
import streamlit as st


def numerical_summary_rows(unified, equivalence):
    """Describe only evidence present in backend reports; absence is never a pass."""
    from cualni_cryst.theory_equivalence import MatchFamily
    from cualni_cryst.theory_unified import PredictionKind

    or_rows = [r for r in unified.rows if r.prediction_kind is PredictionKind.CT_CLOSING_GAP_OR]
    super_rows = [r for r in unified.rows if r.prediction_kind is PredictionKind.CT_SUPERCOMPATIBILITY]
    or_matches = equivalence.matches_for(family=MatchFamily.ORIENTATION)
    angles = [
        float(m.residuals.or_disorientation_deg) for m in or_matches
        if m.residuals.or_disorientation_deg is not None
    ]

    if or_rows:
        or_result = f"{len(or_rows)} candidate ORs"
        or_details = "All candidates retained; no natural orientation relationship was supplied."
        if angles:
            or_details += (
                f" {len(angles)} CT/PTMC candidate matches; "
                f"smallest angle = {min(angles):.5g}°."
            )
        else:
            or_details += " No comparable PTMC OR found."
    else:
        or_result = "No candidate ORs"
        or_details = "No CT closing-gap OR candidates were calculated for these inputs."

    if super_rows:
        epsilon = [
            float(r.residuals["ct_supercompatibility_dimensionless"]) for r in super_rows
            if "ct_supercompatibility_dimensionless" in r.residuals
        ]
        super_result = f"{len(super_rows)} habit/twin pairs"
        super_details = (
            f"Smallest shear/shear residual ε = {min(epsilon):.6g}; the ideal value is zero."
            if epsilon else "No numerical shear/shear residual available."
        )
    elif not unified.ct_report.exact_habit_planes:
        super_result = "Not applicable"
        super_details = "No exact CT austenite/martensite habit plane for these inputs."
    else:
        super_result = "No candidate pairs"
        super_details = "No CT habit/twin pairing; this is not a zero residual."

    audit = equivalence.mm_independent_audit
    if audit is None:
        mm_result = "Not evaluable"
        mm_details = "The independent check could not finish; read scientific warnings."
    elif (audit.get("success") is True and
          audit.get("ct_coverage_complete") is True and
          audit.get("classification_exact") is True):
        mm_result = "Numerical agreement"
        mm_details = (
            f"{audit.get('relation_count', 'N/A')} relationships checked; "
            "CT, Mallard and Ball–James matched within the audit criteria."
        )
    else:
        mm_result = "Mismatch / incomplete"
        mm_details = (
            f"{audit.get('relation_count', 'N/A')} relationships examined. "
            "See the residuals and checks below; no agreement is claimed."
        )

    return [
        {"Check": "CT closing-gap OR", "Numerical result": or_result,
         "What it means": or_details},
        {"Check": "CT shear/shear compatibility", "Numerical result": super_result,
         "What it means": super_details},
        {"Check": "Independent M/M twin audit", "Numerical result": mm_result,
         "What it means": mm_details},
    ]


def render_numerical_checks(unified, equivalence):
    from cualni_cryst.theory_unified import PredictionKind

    st.caption(
        "From the same lattice parameters and correspondence; no experimental EBSD needed. "
        "Numerical agreement is not proof that a twin occurs in the crystal."
    )
    st.dataframe(
        pd.DataFrame(numerical_summary_rows(unified, equivalence)),
        hide_index=True, width="stretch",
    )

    with st.expander("What exactly was checked? (numerical details)", expanded=False):
        st.markdown(
            "**CT closing-gap OR:** possible parent–martensite orientations. "
            "**CT supercompatibility:** whether a CT habit plane and twin satisfy "
            "the shear/shear equation (ε = 0 ideally). "
            "**Independent M/M audit:** whether CT, Mallard and Ball–James give "
            "consistent twin planes, shears and classifications."
        )
        or_rows = [
            r for r in unified.rows
            if r.prediction_kind is PredictionKind.CT_CLOSING_GAP_OR
        ]
        st.markdown("**1. CT orientation-relationship candidates**")
        if or_rows:
            st.dataframe(
                pd.DataFrame([
                    {
                        "Candidate": row.branch_label,
                        "OR matrix": str(row.or_parent_from_product),
                        "Rotation residual": row.residuals.get("rotation"),
                    }
                    for row in or_rows[:80]
                ]).fillna("N/A"),
                hide_index=True, width="stretch",
            )
            st.caption(
                f"First {min(80, len(or_rows))} of {len(or_rows)} candidates. "
                "None selected without a known natural OR."
            )
        else:
            st.caption("No CT closing-gap OR candidates.")

        st.markdown("**2. CT shear/shear compatibility residuals**")
        super_rows = [
            r for r in unified.rows
            if r.prediction_kind is PredictionKind.CT_SUPERCOMPATIBILITY
        ]
        if super_rows:
            ordered = sorted(
                super_rows,
                key=lambda row: float(row.residuals.get(
                    "ct_supercompatibility_dimensionless", float("inf")
                )),
            )
            st.dataframe(
                pd.DataFrame([
                    {
                        "Habit/twin pair": row.branch_label,
                        "Twin class": row.metadata.get("twin_classification", "N/A"),
                        "ε (dimensionless)": row.residuals.get("ct_supercompatibility_dimensionless"),
                    }
                    for row in ordered[:80]
                ]).fillna("N/A"),
                hide_index=True, width="stretch",
            )
            st.caption(
                f"Smallest residuals shown first; {len(super_rows)} total pairings. "
                "A nonzero ε is not an exact solution."
            )
        else:
            st.caption("No exact CT habit/twin pairing exists for these inputs.")

        st.markdown("**3. Independent CT / Mallard / Ball–James twin audit**")
        if equivalence.mm_independent_audit is None:
            st.warning("Audit not evaluable. See scientific warnings; no pass is claimed.")
        else:
            st.json(equivalence.mm_independent_audit)
            st.caption("The audit checks numerical consistency, not experimental twin formation.")
        st.caption("All records, including unmatched branches and warnings, remain in the full JSON export.")
