from __future__ import annotations

"""Cayron 2026's geometry-first comparison using EXISTING solver results.

The sequence follows Acta Materialia 316 (2026) 122399:
M/M transformation twins (Eqs. 19-25) -> individual A/M IPS from CMC
(Eqs. 37, 39, 41) -> A/M/M shear/shear (Eq. 43) -> the one-way
implication to cofactor conditions (Eqs. 1-3).
PTMC's two-variant laminate is displayed separately (paper Section 1).
This module performs no crystallographic calculations, introduces no
input, optimization, angle metric, tolerance or experimental claim.
"""

import math

from cualni_cryst.theory_unified import PredictionKind as K


def _vector(value):
    if value is None:
        return "not reported"
    return "(" + ", ".join(f"{float(v):.8g}" for v in value) + ")"


def _number(value):
    if value is None:
        return "not calculated"
    x = float(value)
    return f"{x:.9g}" if math.isfinite(x) else "not finite"


def cayron_paper_comparison(unified, equivalence):
    """Return source-ordered, branch-resolved observations with no new solving."""
    analysis = unified.ct_report.analysis
    mm_rows = [r for r in unified.rows if r.prediction_kind is K.CT_MM_TWIN]
    am_rows = [r for r in unified.rows if r.prediction_kind is K.CT_AM_HABIT]
    ss_rows = [r for r in unified.rows if r.prediction_kind is K.CT_SUPERCOMPATIBILITY]

    twins = [{
        "CT branch": r.branch_label,
        "Operator": r.metadata.get("operator_index"),
        "Twin classification": r.metadata.get("twin_classification", "unreported"),
        "Rational element": _vector(r.metadata.get("rational_element"))
            if isinstance(r.metadata.get("rational_element"), (tuple, list))
            else str(r.metadata.get("rational_element", "unreported")),
        "Twin plane in parent crystal": _vector(r.twin_plane_parent_crystal),
        "Shear direction in parent crystal": _vector(r.twin_direction_parent_crystal),
        "Twin shear |s|": r.shear_magnitude,
    } for r in mm_rows]

    habits = [{
        "CT A/M branch": r.branch_label,
        "Exact CMC habit": r.exact is True,
        "Habit plane mA in parent crystal": _vector(r.habit_plane_parent_crystal),
        "IPS shear vector dA": _vector(r.shape_vector_parent_crystal),
        "IPS shear magnitude": r.shape_vector_magnitude,
        "Nearest CMC eigenvalue residual": r.residuals.get(
            "cmc_nearest_zero_dimensionless"),
    } for r in am_rows]

    shear = [{
        "CT interface combination": r.branch_label,
        "Operator": r.metadata.get("operator_index"),
        "Twin index": r.metadata.get("twin_index"),
        "Habit index": r.metadata.get("habit_index"),
        "Twin class": r.metadata.get("twin_classification", "unreported"),
        "Habit plane mA": _vector(r.habit_plane_parent_crystal),
        "Twin plane n": _vector(r.twin_plane_parent_crystal),
        "Twin shear direction": _vector(r.twin_direction_parent_crystal),
        "IPS shear dA": _vector(r.shape_vector_parent_crystal),
        "Twin shear |s|": r.shear_magnitude,
        "Shear/shear ε (dimensionless)": r.residuals[
            "ct_supercompatibility_dimensionless"],
    } for r in ss_rows]
    shear.sort(key=lambda x: x["Shear/shear ε (dimensionless)"])

    bj = unified.ball_james_report
    cofactor = [{
        "Twin pair": f"U{t.base_variant_index}–U{t.other_variant_index}",
        "BJ branch": t.branch,
        "Rank-one residual": t.rank_one_residual,
        "CC1: λ2−1": t.cofactor.cc1_lambda2_minus_one,
        "CC2": t.cofactor.cc2_value,
        "CC3 margin": t.cofactor.cc3_margin,
        "CC1–CC3 jointly satisfied": bool(t.cofactor.all_satisfied),
    } for t in bj.martensite_twin_branches]

    matches = []
    for match in equivalence.matches:
        if (match.left_theory.value != "cayron_ct"
                or match.family.value not in (
                    "martensite_martensite_twin", "austenite_martensite_interface")):
            continue
        matches.append({
            "Physical interface": match.family.value.replace("_", " "),
            "Cayron CT branch": match.left_branch,
            "Compared theory": match.right_theory.value,
            "Compared branch": match.right_branch,
            "Comparison outcome": (
                "all required observables agree"
                if match.all_required_components_within_tolerance is True
                else "at least one required observable disagrees"
                if match.all_required_components_within_tolerance is False
                else "not fully comparable"
            ),
            "Component residuals": str(match.residuals.to_dict()),
            "Missing required observables": ", ".join(match.missing_required_components),
        })

    ptmc_report = unified.ptmc_report
    ptmc = []
    if ptmc_report is not None:
        for s in ptmc_report.solutions:
            ptmc.append({
                "PTMC LIS": s.lis_type,
                "Martensite pair": (
                    f"U{s.base_variant_index}–U{s.other_variant_index}"
                    if s.other_variant_index is not None
                    else f"U{s.base_variant_index} (no twin pair)"
                ),
                "Twin branch": s.twin_branch,
                "Laminate fraction f": s.other_variant_volume_fraction,
                "A/M habit plane in parent": _vector(s.habit_plane_parent_crystal),
                "δ": s.dilatational_factor,
                "Middle stretch residual": s.middle_stretch_residual,
                "A/M rank-one residual": s.rank_one_residual,
                "True invariant plane": bool(s.true_invariant_plane),
            })

    return {
        "scope": "same input correspondence and lattice metrics; no new physics or inputs",
        "mm_twins": twins,
        "am_habits": habits,
        "shear_pairs": shear,
        "cofactor_pairs": cofactor,
        "cross_theory_matches": matches,
        "ptmc_laminate_habits": ptmc,
        "ct_cmc": {
            "exact_compatible": bool(analysis.exact_compatible),
            "degeneracy_order": analysis.degeneracy_order,
            "reason": analysis.reason,
            "normalized_eigenvalue_shifts": [float(q) for q in analysis.eigenvalues],
            "nearest_zero_residual": float(analysis.nearest_zero_residual),
        },
        "counts_are_records_not_distinct_experimental_variants": True,
    }


def render_cayron_paper_comparison(unified, equivalence):
    """Primary presentation: reproduce the paper's logical argument, not theory rankings."""
    import pandas as pd
    import streamlit as st

    data = cayron_paper_comparison(unified, equivalence)
    st.markdown("### Cayron's compatibility argument — calculated for these inputs")
    st.caption(
        "Following Cayron, Acta Materialia 316 (2026) 122399, §§2–6. "
        "One entered parent/product metric pair and correspondence, and the "
        "already calculated CT, Ball–James and PTMC records. "
        "No extra assumptions, orientation angles or fitted comparison scores."
    )

    st.markdown("#### 1. M/M — transformation twins from correspondence and symmetry")
    st.latex(r"C_{\mathrm{int}}=C\,g_A\,C^{-1}\quad"
             r"\Longrightarrow\quad(K_1,\eta_1,s)_{\mathrm{Type\,I}},\ "
             r"(\eta_2,K_2,s)_{\mathrm{Type\,II}}")
    st.write(
        f"CT constructed {len(data['mm_twins'])} operator/twin records "
        "through the reflection and twofold routes of Eqs. (19)–(25). "
        "These are calculated transformation twins, not observed twins "
        "and not necessarily distinct physical pairs."
    )
    with st.expander("CT twin operators, rational elements, planes, directions and shear", expanded=False):
        if data["mm_twins"]:
            st.dataframe(pd.DataFrame(data["mm_twins"][:120]),
                         hide_index=True, width="stretch")
            st.caption("At most 120 rows shown; the complete original solver records are in the JSON export.")
        else:
            st.info("No CT reflection/twofold transformation twin was returned.")

    st.markdown("#### 2. A/M — individual-variant invariant-plane strain from CMC")
    st.latex(r"\mathrm{CMC}=C^{\mathsf T}M_MC-M_A,\qquad"
             r"(C^{\mathsf T}M_MC)v_i=\mu_i M_Av_i")
    st.latex(r"\mathrm{SMC}=M_A^{-1}-C^{-1}M_M^{-1}C^{-\mathsf T},"
             r"\qquad d_A=\mathrm{SMC}\,m_A")
    cmc = data["ct_cmc"]
    st.write(
        f"CMC: {'compatible at the backend numerical tolerance' if cmc['exact_compatible'] else 'no exact compatible single-variant IPS at the backend tolerance'}. "
        f"Degeneracy order: {cmc['degeneracy_order']}. "
        f"Normalized eigenvalue shifts (μ−1): {', '.join(_number(q) for q in cmc['normalized_eigenvalue_shifts'])}. "
        f"Nearest |μ−1|: {_number(cmc['nearest_zero_residual'])}. "
        f"{cmc['reason']}"
    )
    with st.expander("A/M habit planes and IPS shear vectors for these metrics", expanded=False):
        if data["am_habits"]:
            st.dataframe(pd.DataFrame(data["am_habits"]), hide_index=True, width="stretch")
            st.caption("A non-exact row is a nearest-zero diagnostic, not an exact coherent habit plane.")
        else:
            st.info("No isolated A/M habit plane was calculated. Higher-order degeneracy can lack a finite plane list.")
    st.caption(
        "This A/M test is for an individual correspondence variant. "
        "It is not PTMC's habit plane of a two-variant laminate."
    )

    st.markdown("#### 3. A/M/M — Cayron's shear/shear intercompatibility")
    st.latex(r"2(m_A^{\mathsf T}n)d_A=a_{\mathrm{twin}},\qquad"
             r"\varepsilon=\frac{\|2(m_A^{\mathsf T}n)d_A-a_{\mathrm{twin}}\|_{M_A}}{|s|}")
    shear = data["shear_pairs"]
    if shear:
        minimum = shear[0]
        st.write(
            f"{len(shear)} exact-CMC habit/twin combinations were evaluated. "
            f"Smallest calculated geometric residual ε = {_number(minimum['Shear/shear ε (dimensionless)'])}, "
            f"for {minimum['CT interface combination']}. "
            "Ideal shear/shear compatibility requires ε = 0; a small number "
            "is not an experimentally measured formation probability."
        )
    else:
        st.write(
            "No exact A/M habit/twin combination was available for the "
            "shear/shear test. It was not evaluated, which is not the same as failure."
        )
    with st.expander("All A/M/M combinations, ranked by the existing ε", expanded=False):
        if shear:
            st.dataframe(pd.DataFrame(shear[:120]), hide_index=True, width="stretch")
            st.caption("The ordering is by Cayron's existing ε, not by a newly invented score.")
        else:
            st.info("No exact A/M/M shear/shear pairs.")

    st.markdown("#### 4. Cofactor conditions — the implication, not a theory contest")
    st.latex(r"\left\{\mathrm{A/M\ IPS,\ M/M\ twin,\ shear/shear}\right\}"
             r"\Longrightarrow\left\{\mathrm{SC1,\ SC2,\ SC3}\right\}")
    st.write(
        "Cayron derives this forward implication in §2.3 and discusses it "
        "in §6. The paper does not prove the converse. "
        "Ball–James cofactor values are therefore compared only for their "
        "own valid M/M rank-one branches; they cannot be substituted for "
        "an unmatched CT twin."
    )
    with st.expander("Ball–James CC1–CC3 for the already computed M/M branches", expanded=False):
        if data["cofactor_pairs"]:
            st.dataframe(pd.DataFrame(data["cofactor_pairs"][:120]),
                         hide_index=True, width="stretch")
        else:
            st.info("No Ball–James M/M rank-one twin branch; cofactor assessment unavailable.")

    st.markdown("#### 5. PTMC — the alternative laminate construction")
    st.latex(r"R\,F(f)-\delta I=b\otimes m,\qquad"
             r"F(f)=U_i+f(a\otimes n),\quad 0\leq f\leq1")
    st.write(
        f"The current PTMC backend returned {len(data['ptmc_laminate_habits'])} "
        "discrete habit-plane records for the entered transformation. "
        "PTMC's laminate-fraction problem and CT's individual-variant "
        "CMC problem are different, so their solution counts must not be equated. "
        "The full comparison is made through common physical observables where available."
    )
    with st.expander("PTMC laminate fractions, habit planes and rank-one residuals", expanded=False):
        if data["ptmc_laminate_habits"]:
            st.dataframe(pd.DataFrame(data["ptmc_laminate_habits"][:120]),
                         hide_index=True, width="stretch")
        else:
            st.info("No discrete PTMC laminate habit solution was returned.")
    with st.expander("CT/BJ/PTMC assignments in common physical quantities", expanded=False):
        if data["cross_theory_matches"]:
            st.dataframe(pd.DataFrame(data["cross_theory_matches"][:120]),
                         hide_index=True, width="stretch")
        else:
            st.info("No physical branch assignment was found; no claim of disagreement follows.")

    st.caption(
        "This panel reports Cayron's compatibility chain and the separate "
        "PTMC formulation using existing results only. No new theory, "
        "orientation input, stress model, angle criterion or experimental prediction."
    )
    return data


def cayron_paper_thesis_notes(unified, equivalence):
    """Use identical scientific statements in the downloadable thesis notes."""
    data = cayron_paper_comparison(unified, equivalence)
    cmc = data["ct_cmc"]
    eps = data["shear_pairs"]
    return "\n".join([
        "## Comparison structured as Cayron (Acta Materialia 316, 2026, 122399)",
        "",
        "Existing lattice metrics and correspondence are used without fitting or new angles.",
        "The logical order is M/M twins -> A/M CMC/SMC -> shear/shear -> one-way cofactor implication.",
        "PTMC's two-variant laminate habit is a different physical construction.",
        "",
        f"1. M/M: {len(data['mm_twins'])} CT operator/twin construction records.",
        f"2. A/M: CMC compatible at solver tolerance = {cmc['exact_compatible']}; "
        f"degeneracy order {cmc['degeneracy_order']}; generalized eigenvalue shifts "
        f"{', '.join(_number(x) for x in cmc['normalized_eigenvalue_shifts'])}; "
        f"{len(data['am_habits'])} exact-or-diagnostic plane records.",
        f"3. A/M/M: {len(eps)} evaluated shear/shear combinations; "
        + (f"minimum ε = {_number(eps[0]['Shear/shear ε (dimensionless)'])}."
           if eps else "not evaluable for the present exact A/M/M candidates."),
        f"4. Cofactor: {len(data['cofactor_pairs'])} BJ twin branches evaluated separately; "
        "the reverse implication is not established by the source.",
        f"5. PTMC: {len(data['ptmc_laminate_habits'])} calculated laminate habit records, "
        f"{len(data['cross_theory_matches'])} assigned CT-versus-other records.",
        "",
        "Numbers count calculation records, not unique observed twins. "
        "The complete branch-by-branch values and assignment diagnostics are in "
        "the exported JSON; results do not establish real specimen selection.",
    ]) + "\n"
