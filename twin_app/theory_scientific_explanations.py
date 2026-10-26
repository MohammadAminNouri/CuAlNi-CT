from __future__ import annotations

"""Scientific provenance for calculated CT, Ball–James, PTMC and cofactor results.

Interpret only values returned by the existing verified backends. This file
does not calculate twins, change tolerances, or infer experimental outcomes.
"""

import math

# Each entry is (theory name, question, observed result, equations, reasoning,
# defensible conclusion, literature/implementation provenance).
# Equation strings are LaTeX source consumed by Streamlit's math renderer.


def scientific_explanations(unified, equivalence):
    from cualni_cryst.theory_equivalence import MatchFamily
    from cualni_cryst.theory_unified import PredictionKind as K, TheoryKind as T

    def rows(kind):
        return [r for r in unified.rows if r.prediction_kind is kind]

    def matched(family, right):
        candidates = [
            m for m in equivalence.matches
            if m.family is family and m.left_theory is T.CAYRON_CT
            and m.right_theory is right
        ]
        return (
            sum(m.all_required_components_within_tolerance is True for m in candidates),
            sum(m.all_required_components_within_tolerance is False for m in candidates),
            sum(m.all_required_components_within_tolerance is None for m in candidates),
        )

    # Cayron CT: exact CMC degeneracy and generated correspondence/twin branches.
    analysis = unified.ct_report.analysis
    ct_habit = unified.ct_report.exact_habit_planes
    ct_twins = rows(K.CT_MM_TWIN)
    ct_or = rows(K.CT_CLOSING_GAP_OR)
    ct_super = rows(K.CT_SUPERCOMPATIBILITY)
    eps = [
        float(r.residuals["ct_supercompatibility_dimensionless"])
        for r in ct_super
        if "ct_supercompatibility_dimensionless" in r.residuals
        and math.isfinite(float(r.residuals["ct_supercompatibility_dimensionless"]))
    ]
    nearest = float(analysis.nearest_zero_residual)
    if analysis.exact_compatible and len(ct_habit):
        cmc_decision = (
            f"Exact CMC compatibility: yes; {len(ct_habit)} finite A/M habit-plane "
            f"records from degeneracy order {analysis.degeneracy_order}."
        )
    elif analysis.exact_compatible:
        cmc_decision = (
            f"CMC is exactly degenerate (order {analysis.degeneracy_order}), "
            "but the solver does not enumerate isolated finite habit planes."
        )
    else:
        cmc_decision = (
            "Exact CMC compatibility: no for these input metrics and tolerance."
        )
    ct_result = (
        f"{len(ct_twins)} CT twin-construction records; {len(ct_or)} closing-gap OR "
        f"candidate records. {cmc_decision} Smallest |μ−1| = {nearest:.6g}."
    )
    ct_result += (
        f" Smallest computed shear/shear ε = {min(eps):.6g} "
        f"over {len(ct_super)} A/M–M/M pair records."
        if eps else
        " CT shear/shear comparison unavailable because no exact habit/twin pairing was returned."
    )
    ct_logic = (
        "Pull the product metric back through the entered correspondence C. "
        "A/M compatibility follows when the normalized CMC quadratic form has "
        "the required zero eigenvalue and opposite signs on the remaining two "
        "directions (or an admissible higher-order degeneracy). Symmetry-related "
        "intercorrespondence operators generate twin candidates. Only when exact "
        "A/M habit planes exist can the additional shear/shear residual be evaluated."
    )
    ct_conclusion = (
        "CT finds mathematically admissible correspondence/twin candidates. "
        + ("The entered metrics satisfy an exact CMC condition." if analysis.exact_compatible
           else "The entered metrics do not satisfy the exact CMC condition.")
        + " Closing-gap orientations are predictions, not experimentally selected ORs. "
        + ("A nonzero ε is a mismatch, not exact supercompatibility." if eps
           else "No claim about CT shear/shear supercompatibility follows.")
    )

    # Ball–James: independent nonlinear rank-one constructions.
    bj = unified.ball_james_report
    bj_twins = bj.martensite_twin_branches
    bj_habits = bj.austenite_martensite_branches
    bj_valid, bj_failed, bj_partial = matched(MatchFamily.MM_TWIN, T.BALL_JAMES)
    worst_mismatch = min(
        (max(abs(float(t.rank_one_residual)), abs(float(t.rotation_residual)))
         for t in bj_twins),
        default=None,
    )
    bj_result = (
        f"{len(bj.variants)} distinct stretch variants; "
        f"{len(bj_twins)} ordered M/M rank-one branch records and "
        f"{len(bj_habits)} single-variant A/M rank-one branch records. "
        f"CT/BJ assigned M/M branches: {bj_valid} fully agree within tolerance, "
        f"{bj_failed} fail, {bj_partial} are incomplete."
    )
    if worst_mismatch is not None:
        bj_result += f" Smallest branch maximum(rank-one, rotation residual) = {worst_mismatch:.6g}."
    bj_logic = (
        "Construct U as the positive square root of the pulled-back product "
        "metric. For each variant pair, form A = U_j U_i^{-1}; its relative "
        "Cauchy–Green tensor AᵀA must have middle eigenvalue μ₂ = 1. "
        "Then solve both admissible R U_j − U_i = a⊗n branches and verify "
        "the reconstructed rank-one jump and proper rotation. Single-variant "
        "A/M interfaces are checked independently against the parent identity."
    )
    bj_conclusion = (
        f"Ball–James identifies {len(bj_twins)} *calculated branch records* "
        "under the stated lattice metrics, not the number of twins in a crystal. "
        + (f"{bj_valid} full CT/BJ assignments numerically agree. " if bj_valid
           else "No complete CT/BJ agreement was established by the matched assignments. ")
        + "Unmatched, failed or incomplete comparisons cannot certify equivalent twins."
    )

    # PTMC: invariant-plane solution of a twinned laminate, not the M/M interface itself.
    ptmc = unified.ptmc_report
    ptmc_rows = rows(K.PTMC_HABIT)
    ptmc_exact = [r for r in ptmc_rows if r.exact is True]
    ptmc_cont = rows(K.PTMC_CONTINUUM)
    ptmc_agreed, ptmc_failed, ptmc_partial = matched(MatchFamily.AM_INTERFACE, T.PTMC)
    example = ptmc_exact[0] if ptmc_exact else (ptmc_rows[0] if ptmc_rows else None)
    ptmc_example = ""
    if example is not None:
        f = example.metadata.get("other_variant_volume_fraction")
        parameter = example.metadata.get("parameter_value")
        frac = f if f is not None else parameter
        mr = example.residuals.get("middle_stretch")
        ptmc_example = (
            f" Example calculated branch: {example.branch_label}; "
            f"parameter f = {float(frac):.6g}"
            if frac is not None else
            f" Example calculated branch: {example.branch_label}"
        )
        if mr is not None:
            ptmc_example += f"; |σ₂−δ| = {float(mr):.6g}."
        else:
            ptmc_example += "."
    ptmc_result = (
        f"{len(ptmc_rows)} PTMC habit-solution records "
        f"({len(ptmc_exact)} true invariant-plane records at the selected δ), "
        f"{len(ptmc_cont)} continuum-family records. "
        f"CT/PTMC assigned A/M interfaces: {ptmc_agreed} full agreements, "
        f"{ptmc_failed} disagreements, {ptmc_partial} incomplete."
        + ptmc_example
    )
    ptmc_logic = (
        "For an admissible twinned pair the pre-habit deformation is "
        "F(f) = U_i + f(a⊗n), with 0 ≤ f ≤ 1. "
        "The solver finds roots of det(FᵀF − δ²I) = 0, checks the middle "
        "singular value σ₂(F) = δ, and independently reconstructs a proper "
        "rotation and rank-one A/M interface R F − δI = b⊗m. "
        "At the default δ = 1 this is a true invariant-plane condition; "
        "for δ ≠ 1, it is a dilatational variant rather than an invariant plane."
    )
    ptmc_conclusion = (
        ("PTMC returns calculated laminate fractions and A/M habit candidates "
         "for these inputs; compare their normals and rank-one tensors with "
         "other theories only when those observables are defined."
         if ptmc_rows or ptmc_cont else
         "PTMC returned no A/M habit solution under the entered twinning "
         "mechanism and parameters; that is not proof transformation is impossible.")
        + " A PTMC habit is not automatically the plane between two martensite plates."
    )

    # Cofactor: branch-specific conjunction; never inferred from a single CC1.
    cc_all = [
        t for t in bj_twins
        if t.cofactor.cc1_satisfied and t.cofactor.cc2_satisfied
        and t.cofactor.cc3_satisfied
    ]
    representative = (
        min(bj_twins, key=lambda t: abs(float(t.cofactor.cc1_lambda2_minus_one)))
        if bj_twins else None
    )
    cc_result = (
        f"{len(cc_all)} of {len(bj_twins)} Ball–James M/M branch records satisfy "
        "CC1, CC2 and CC3 simultaneously at the backend's declared tolerance."
    )
    if representative is not None:
        c = representative.cofactor
        cc_result += (
            " Illustrative branch selected by smallest |CC1|: "
            f"U{representative.base_variant_index}–U{representative.other_variant_index}, "
            f"branch {representative.branch}; "
            f"CC1 = {float(c.cc1_lambda2_minus_one):.6g}, "
            f"CC2 = {float(c.cc2_value):.6g}, "
            f"CC3 margin = {float(c.cc3_margin):.6g}; "
            f"passes = {bool(c.all_satisfied)}."
        )
    cc_logic = (
        "For each Ball–James twin, evaluate the middle principal stretch of "
        "the base U and the twin shear a⊗n. CC1 and CC2 require residuals "
        "near zero, while CC3 requires its reported margin to be nonnegative "
        "within tolerance. The test passes only if all three conditions pass; "
        "checking λ₂ alone is insufficient."
    )
    cc_conclusion = (
        f"The code verifies stronger, branch-specific geometric compatibility "
        f"for {len(cc_all)} of {len(bj_twins)} M/M records. "
        "These conditions do not measure fatigue, hysteresis, transformation "
        "temperatures or whether a twin actually forms."
    )

    return (
        {
            "name": "Cayron correspondence theory (CT)",
            "result": ct_result,
            "equations": (
                r"\mathrm{CMC}=C^{\mathsf T}M_M C-M_A,\quad"
                r" (C^{\mathsf T}M_M C)v=\mu M_Av",
                r"\mathrm{SMC}=M_A^{-1}-C^{-1}M_M^{-1}C^{-\mathsf T},"
                r"\quad d_A=\mathrm{SMC}\,m_A",
                r"\varepsilon_{\mathrm{CT}}="
                r"\frac{\|2(m_A^{\mathsf T}n)d_A-a\|_{M_A}}{|s|}"
            ),
            "logic": ct_logic,
            "conclusion": ct_conclusion,
            "reference": "Cayron, Acta Materialia 316 (2026), 122399; ct.py",
        },
        {
            "name": "Ball–James nonlinear compatibility",
            "result": bj_result,
            "equations": (
                r"U^2=M_A^{-1/2}C^{\mathsf T}M_MC\,M_A^{-1/2}",
                r"R\,U_j-U_i=a\otimes n,\quad"
                r"\mu_2\!\left[(U_jU_i^{-1})^{\mathsf T}(U_jU_i^{-1})\right]=1",
            ),
            "logic": bj_logic,
            "conclusion": bj_conclusion,
            "reference": "Ball–James rank-one compatibility; ball_james_adapter.py",
        },
        {
            "name": "Phenomenological theory of martensite crystallography (PTMC)",
            "result": ptmc_result,
            "equations": (
                r"F(f)=U_i+f(a\otimes n),\quad 0\leq f\leq1",
                r"\det\!\left(F(f)^{\mathsf T}F(f)-\delta^2I\right)=0,"
                r"\quad\sigma_2(F(f))=\delta",
                r"R\,F(f)-\delta I=b\otimes m",
            ),
            "logic": ptmc_logic,
            "conclusion": ptmc_conclusion,
            "reference": "Classical PTMC and Ball–James habit condition; ptmc_adapter.py",
        },
        {
            "name": "Cofactor conditions (Chen et al., 2013)",
            "result": cc_result,
            "equations": (
                r"\mathrm{CC1}:\ \lambda_2(U)-1=0",
                r"\mathrm{CC2}:\ a\cdot U\,\operatorname{cof}(U^2-I)n=0",
                r"\mathrm{CC3}:\ \operatorname{tr}(U^2)-\det(U^2)"
                r"-\tfrac14\|a\|^2\|n\|^2-2\geq0",
            ),
            "logic": cc_logic,
            "conclusion": cc_conclusion,
            "reference": "Chen–Srivastava–Dabade–James (2013), Theorem 2; ball_james_adapter.py",
        },
    )


def scientific_notes(unified, equivalence):
    lines = [
        "## Per-theory results, equations, reasoning and scientific conclusions",
        "",
        "All quantities come from the input-specific backend reports. "
        "Exact is defined by the solver tolerances, not by measurement uncertainty.",
    ]
    for entry in scientific_explanations(unified, equivalence):
        lines.extend((
            "",
            "### " + entry["name"],
            "**Calculated result:** " + entry["result"],
            "",
            "**Equations (LaTeX):**",
        ))
        lines.extend("- $" + equation + "$" for equation in entry["equations"])
        lines.extend((
            "",
            "**Reasoning:** " + entry["logic"],
            "",
            "**Conclusion:** " + entry["conclusion"],
            "",
            "**Provenance:** " + entry["reference"],
        ))
    return "\n".join(lines) + "\n"


def render_scientific_explanations(unified, equivalence):
    import streamlit as st

    st.markdown("### How each theory reaches its conclusion")
    st.caption(
        "Each section uses the numerical results above. Read in order: "
        "result → equations → mathematical reasoning → defensible conclusion. "
        "All results cover the full entered transformation, not just the selected twin mode."
    )
    for entry in scientific_explanations(unified, equivalence):
        with st.expander(entry["name"], expanded=False):
            st.markdown("**1. Calculated result**")
            st.write(entry["result"])
            st.markdown("**2. Governing equations**")
            for equation in entry["equations"]:
                st.latex(equation)
            st.markdown("**3. How the equation produces the result**")
            st.write(entry["logic"])
            st.markdown("**4. Scientific conclusion**")
            st.write(entry["conclusion"])
            st.caption("Source / implementation: " + entry["reference"])
