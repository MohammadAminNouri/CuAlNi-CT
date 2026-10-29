from __future__ import annotations

"""Interactive Cayron-compatible lattice metric design, non-destructive.

Calculates metric targets first (lambda2 only). At the user's request each
candidate runs independent CT + Ball--James/cofactor solvers through the
existing supercompatibility atlas; complete PTMC requires a second explicit
action. Optimization never changes the Streamlit crystal editor or commits a
guessed experimental alloy recipe.
"""
import json
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_lattice_design_advisor(project_payload, transformation_id: str,
                                  signature: str) -> None:
    with st.expander(
        "Explore lattice changes for compatibility (Cayron 2026)",
        expanded=False,
    ):
        st.caption(
            "Independent metric-design experiment, using the entered "
            "correspondence and crystal settings. Proposals are hypothetical "
            "lattice metrics, not composition or processing instructions."
        )
        from cualni_cryst.project_io import project_from_dict
        from cualni_cryst.supercompatibility_atlas import (
            independent_product_lattice_parameters,
        )
        from twin_app.metric_design_targets import (
            metric_stretches, scan_lambda2_targets, state_at_parameter,
        )
        try:
            original = project_from_dict(
                dict(project_payload), source="twin_app:metric_design_advisor"
            ).project
            baseline = metric_stretches(original, transformation_id)
            independent = independent_product_lattice_parameters(
                original, transformation_id
            )
        except (ValueError, ArithmeticError, KeyError) as exc:
            st.error(f"Metric-design input invalid: {exc}")
            return

        a,b,c = st.columns(3)
        for col, lab, number in zip((a,b,c),("λ₁","λ₂","λ₃"),baseline):
            col.metric(lab,f"{number:.8f}")
        st.caption(
            f"Current |λ₂−1| = {abs(baseline[1]-1.):.7g}. "
            "λ₂=1 is the single-variant A/M condition, not full "
            "A/M/M supercompatibility."
        )

        length_percent = float(st.slider(
            "Maximum allowed lattice-length change (%)",
            min_value=0.5,max_value=20.,value=5.,step=0.5,
            key="tf_metric_design_length_bound",
        ))
        angle_window = float(st.slider(
            "Maximum angle change (degrees)",
            min_value=0.5,max_value=10.,value=3.,step=0.5,
            key="tf_metric_design_angle_bound",
        ))
        _render_joint_compatibility_design(
            original, transformation_id, signature,
            length_percent/100.0, angle_window,
        )
        key=(signature,length_percent,angle_window)
        if st.button(
            "Find λ₂=1 lattice targets",
            key="tf_metric_design_scan",
        ):
            with st.spinner("Scanning independent crystal parameters…"):
                results={}
                for param in ("uniform_scale",*independent):
                    try:
                        results[param]=scan_lambda2_targets(
                            original,transformation_id,param,
                            length_relative_bound=length_percent/100.,
                            angle_bound_deg=angle_window,
                        )
                    except (ValueError,ArithmeticError) as exc:
                        st.warning(f"{param}: {exc}")
                st.session_state["tf_metric_design_scan_result"]=(key,results)

        saved=st.session_state.get("tf_metric_design_scan_result")
        if not saved or saved[0]!=key:
            st.caption("Calculate targets to see which metric change may reach λ₂=1.")
            _show_niti_reference_example(project_payload, original, transformation_id)
            return

        results=saved[1]
        parameters = []
        candidate_options=[]
        for param,res in results.items():
            candidate=res.candidates[0] if res.candidates else None
            if candidate is not None:
                candidate_options.extend((param,c) for c in res.candidates)
            parameters.append({
                "Free parameter": param,
                "λ₂ at original cell":res.baseline_lambda2,
                "∂λ₂/∂parameter":res.derivative,
                "Target": (
                    float(candidate["parameter_value"])
                    if candidate is not None else None
                ),
                "Required change": (
                    float(candidate["change"])
                    if candidate is not None else None
                ),
                "Residual at target":(
                    float(candidate["abs_lambda2_minus_one"])
                    if candidate is not None else None
                ),
                "Result":(
                    "λ₂=1 candidate (NOT supercompatible)"
                    if candidate is not None else "Not found within search range"
                ),
            })
        st.dataframe(
            pd.DataFrame(parameters),
            hide_index=True,width="stretch",
        )
        st.caption(
            "Scale parameters are relative to the original length (1=unchanged). "
            "β offsets are in degrees. 'Not found' means only that the "
            "bounded search did not resolve a root; tangencies or solutions "
            "outside the window are possible."
        )
        selected=st.selectbox(
            "Inspect parameter response",
            list(results),
            key="tf_metric_design_selected_axis",
        )
        result=results[selected]
        fig=go.Figure()
        fig.add_trace(go.Scatter(
            x=[r["parameter_value"] for r in result.scan],
            y=[r["lambda2_minus_one"] for r in result.scan],
            mode="lines+markers",name="λ₂−1",
        ))
        fig.add_hline(y=0,line_dash="dash")
        fig.update_layout(
            height=300,margin=dict(l=24,r=15,t=18,b=35),
            xaxis_title=selected,
            yaxis_title="λ₂ − 1",
        )
        st.plotly_chart(fig,width="stretch")
        st.caption(
            "The plotted curve uses the exact supplied C and independent "
            "product metric changes. No symmetry or correspondence is fitted."
        )

        _show_niti_reference_example(project_payload, original, transformation_id)

        if not candidate_options:
            st.info(
                "No proposed target reached λ₂=1 inside the allowed window. "
                "Broaden physically plausible bounds or study another "
                "independent parameter; do not infer that the material "
                "cannot satisfy an A/M IPS."
            )
            return
        selected_idx=st.selectbox(
            "Candidate to assess with the complete crystallographic model",
            range(len(candidate_options)),
            format_func=lambda i: (
                f"{candidate_options[i][0]} = "
                f"{candidate_options[i][1]['parameter_value']:.9g}; "
                f"|λ₂−1|={candidate_options[i][1]['abs_lambda2_minus_one']:.3g}"
            ),
            key="tf_metric_design_candidate_choice",
        )
        param,candidate=candidate_options[selected_idx]
        candidate_key=(key,param,float(candidate["parameter_value"]))
        # Second-stage Cayron design: stay on the lambda2=1 CMC target
        # while varying a second crystal parameter; check the SAME
        # physically paired CT shear/shear and BJ cofactor branch.
        other_axes=[axis for axis in independent if axis!=param]
        if other_axes and st.checkbox(
            "Design toward full A/M/M shear/shear compatibility",
            key="tf_metric_design_continuation_open",value=False,
        ):
            st.caption(
                "Maintain λ₂≈1 by retuning the selected first parameter, "
                "then vary a second independent product-cell parameter. "
                "Each state is independently tested for exact CT A/M habit, "
                "M/M twin, Cayron Eq. 43 shear/shear ε, and BJ CC1–CC3 "
                "on physically matched twin relations. This is a bounded "
                "sampled search, not a proof that a solution exists."
            )
            secondary=st.selectbox(
                "Second independent lattice parameter",
                other_axes,key="tf_metric_design_secondary",
            )
            nstates=st.select_slider(
                "Number of full CT/BJ evaluations",
                options=(3,5,7),value=3,
                key="tf_metric_design_full_states",
            )
            second_bound=st.slider(
                "Second parameter length range (±%)",
                min_value=0.5,max_value=10.,value=2.,step=.5,
                key="tf_metric_design_secondary_bound",
            )
            continuation_key=(
                signature,param,secondary,int(nstates),
                float(second_bound),length_percent,angle_window,
            )
            if st.button(
                "Suggest lattice changes for Cayron shear/shear",
                key="tf_metric_design_continue_run",
            ):
                try:
                    from twin_app.metric_design_continuation import (
                        continue_supercompatibility_design,
                    )
                    with st.status(
                        "Re-solving CT and Ball–James for each constrained cell",
                        expanded=True,
                    ) as status:
                        full_design=continue_supercompatibility_design(
                            original,transformation_id,
                            primary_parameter=param,
                            secondary_parameter=secondary,
                            secondary_relative_bound=float(second_bound)/100.,
                            angle_bound_deg=angle_window,
                            primary_relative_bound=length_percent/100.,
                            primary_angle_bound_deg=angle_window,
                            steps=int(nstates),
                        )
                        status.update(
                            label="Metric continuation finished",
                            state="complete",expanded=False,
                        )
                    st.session_state["tf_metric_design_continuation"]=(
                        continuation_key,full_design,
                    )
                except Exception as exc:
                    st.error(
                        "Conditional A/M/M evaluation was not completed: "
                        f"{type(exc).__name__}: {exc}"
                    )
            cont=st.session_state.get("tf_metric_design_continuation")
            if cont and cont[0]==continuation_key:
                data=cont[1]
                st.markdown("**Suggested direction from the sampled, matched branches**")
                st.write(data["proposed_direction"])
                st.caption(data["scientific_limit"])
                displays=[]
                for row in data["samples"]:
                    pair=row.get("paired_CT_BJ_branch") or {}
                    displays.append({
                        "Δ second parameter":row["secondary_delta"],
                        "Target first parameter":row.get("primary_value"),
                        "a":row.get("a"),"b":row.get("b"),
                        "c":row.get("c"),"β":row.get("beta_deg"),
                        "|λ₂−1|":row.get("lambda2_residual"),
                        "Exact CT A/M":row.get("exact_CT_AM"),
                        "CT shear/shear ε":pair.get("ct_shear_shear_epsilon"),
                        "CC1 |λ₂−1|":pair.get("CC1_abs_lambda2_minus_one"),
                        "CC2 |value|":pair.get("CC2_abs"),
                        "CC3 margin":pair.get("CC3_margin"),
                        "Matched branch":(
                            f"{pair['operator_index']} / {pair['twin_index']}"
                            if pair else "unavailable"
                        ),
                        "Match complete":row.get("matching_is_complete",False),
                        "State":row["status"],
                    })
                st.dataframe(
                    pd.DataFrame(displays),hide_index=True,width="stretch",
                )
                fig=go.Figure()
                fig.add_trace(go.Scatter(
                    x=[r["secondary_delta"] for r in displays],
                    y=[r["CT shear/shear ε"] for r in displays],
                    mode="markers",name="Matched CT ε",
                    customdata=[r["Matched branch"] for r in displays],
                    hovertemplate="Δ parameter: %{x:.6g}<extra>%{customdata}</extra>",
                ))
                fig.update_layout(
                    height=270,margin=dict(l=25,r=15,t=18,b=38),
                    xaxis_title="Change in second parameter (relative or degrees)",
                    yaxis_title="Cayron shear/shear ε",
                )
                st.plotly_chart(fig,width="stretch")
                st.caption(
                    "Points may represent different twin branches. "
                    "They are deliberately NOT connected by a line, "
                    "and no interpolated ε=0 solution is asserted. "
                    "Missing points mean that the exact CMC habit or "
                    "a complete matched CT/BJ relation was unavailable."
                )
                st.download_button(
                    "Export complete sampled A/M/M design (.json)",
                    data=json.dumps(data,indent=2,default=str),
                    file_name="cayron_2026_conditional_shear_shear_design.json",
                    mime="application/json",
                    key="tf_metric_design_continuation_export",
                )


        if st.button(
            "Verify CT shear/shear and BJ cofactor conditions",
            key="tf_metric_design_verify",
        ):
            try:
                from cualni_cryst.supercompatibility_atlas import (
                    evaluate_supercompatibility_state,
                )
                with st.spinner("Running independent existing CT and Ball–James solvers…"):
                    changed=state_at_parameter(
                        original,transformation_id,param,
                        candidate["parameter_value"],
                    )
                    result=evaluate_supercompatibility_state(
                        changed,transformation_id,
                        state_id="metric_design_candidate",
                        metadata={
                            "design_parameter":param,
                            "selected_value":candidate["parameter_value"],
                            "original_input_signature":signature,
                            "goal":"lambda2=1 only",
                        },
                    )
                st.session_state["tf_metric_design_verified"]=(
                    candidate_key,result.to_dict()
                )
            except Exception as exc:
                st.error(
                    "Independent compatibility verification failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        verified=st.session_state.get("tf_metric_design_verified")
        if verified and verified[0]==candidate_key:
            data=verified[1]
            cols=st.columns(3)
            cols[0].metric("Exact CT A/M",str(data["ct_am_exact_compatible"]))
            cols[1].metric("CT A/M/M",str(data["ct_supercompatible"]))
            cols[2].metric("BJ cofactor",str(data["cofactor_compatible"]))
            st.write(
                "**Matched twin relations:** "+str(data["matched_relation_count"])
            )
            st.caption(
                "CT shear/shear and cofactor refer to specific M/M twin branches. "
                "An unavailable exact CMC habit is not a failed shear/shear "
                "test, and a nearby lattice target is not a confirmed alloy."
            )
            st.json({
                "CT minimum shear/shear ε":data["ct_best_supercompatibility_residual"],
                "Minimum |CC1|":data["cofactor_best_cc1_abs"],
                "Minimum |CC2|":data["cofactor_best_cc2_abs"],
                "Maximum CC3 margin":data["cofactor_best_cc3_margin"],
                "branch_matching":data["pair_results"],
                "errors":data["error"],
            })
            st.download_button(
                "Export metric design and CT/cofactor audit (.json)",
                data=json.dumps({
                    "input_signature":signature,
                    "baseline_lambda123":baseline,
                    "parameter":param,
                    "proposed_target":candidate,
                    "validation":data,
                    "source_doi":"10.1016/j.actamat.2026.122399",
                    "scope":"theoretical crystallographic metrics only",
                },indent=2,default=str),
                file_name="metric_design_supercompatibility_audit.json",
                mime="application/json",
                key="tf_metric_design_export",
            )
            if st.button(
                "Run CT / BJ / PTMC full theory comparison at this target",
                key="tf_metric_design_complete",
            ):
                try:
                    from twin_app.theory_comparison_ui import (
                        calculate_full_theory_report,
                        overview_rows,
                    )
                    with st.spinner("Evaluating all original CT/BJ/PTMC solvers…"):
                        changed=state_at_parameter(
                            original,transformation_id,param,
                            candidate["parameter_value"],
                        )
                        unified, equivalence=calculate_full_theory_report(
                            changed.to_dict(),transformation_id,
                        )
                    st.session_state["tf_metric_design_full"]=(
                        candidate_key,
                        {"summary":overview_rows(unified),
                         "comparisons":equivalence.to_dict(),
                         "unified":unified.to_dict()},
                    )
                except Exception as exc:
                    st.error(
                        "Full theory comparison failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
            full=st.session_state.get("tf_metric_design_full")
            if full and full[0]==candidate_key:
                st.dataframe(
                    pd.DataFrame(full[1]["summary"]),
                    hide_index=True,width="stretch",
                )
                st.download_button(
                    "Export complete candidate CT/BJ/PTMC calculations",
                    data=json.dumps(full[1],indent=2,default=str),
                    file_name="metric_design_full_theory_comparison.json",
                    mime="application/json",key="tf_metric_design_full_export",
                )


def _show_niti_reference_example(project_payload, project, transformation_id):
    """Paper-specific analytical example; never silently map it to Cu–Al–Ni."""
    if st.checkbox(
        "Show Cayron 2026 B2→B19′ analytical example (NiTi only)",
        value=False,key="tf_metric_design_niti_examples",
    ):
        st.caption(
            "Eq. (47), Table 2 and §5.3 are specific to Cayron's chosen "
            "B2→B19′ correspondence. These equations do not apply "
            "directly to other Cu–Al–Ni or NiTi correspondences."
        )
        from twin_app.cayron_2026_reference import niti_b19prime_cmc_domains
        t=project.transformation(transformation_id)
        parent=project.phase(t.parent_phase_id)
        product=project.phase(t.product_phase_id)
        pgroup=parent.point_group_symbol
        mgroup=product.point_group_symbol
        if pgroup not in {"m-3m","Oh","m3m"} or mgroup not in {"2/m","C2h"}:
            st.info("The current crystal point groups are not B2 cubic → monoclinic B19′.")
            return
        st.warning(
            "Crystal symmetry alone does not certify Cayron's exact B2→B19′ "
            "correspondence. The following values are conditional references, "
            "not automatic validation of the entered transformation."
        )
        a0=parent.lattice.a
        aa=product.lattice.a/a0
        bb=product.lattice.b/a0
        cc=product.lattice.c/a0
        beta=product.lattice.beta_deg
        domain=niti_b19prime_cmc_domains(aa,bb,cc,beta)
        st.dataframe(pd.DataFrame(domain),hide_index=True,width="stretch")
        btarget=math.sqrt(2)*a0
        inequality=2*aa*aa+cc*cc-aa*aa*cc*cc*math.sin(math.radians(beta))**2
        st.write(
            f"For this specific B2→B19′ normalization, the C₁ equality "
            f"requires b = √2 a₀ = {btarget:.8g} "
            f"(current b = {product.lattice.b:.8g}, same length unit). "
            f"Keeping a, c, β fixed, the second C₁ inequality "
            f"{'is' if inequality>=2 else 'is NOT'} fulfilled "
            f"(left side = {inequality:.7g}, required ≥2)."
        )
        st.warning(
            "The paper's C2b Table-2 / Eq.-47 inequality conflict and O₂ "
            "Type-I published-vs-derived discrepancy remain unresolved. "
            "Do not advertise those printed families as independently "
            "validated supercompatibility design prescriptions."
        )


def _render_joint_compatibility_design(
    original, transformation_id, signature, length_relative_bound, angle_bound_deg,
):
    """Opt-in multi-axis A/M design, followed by independent twin branch checks.

    Important: obtaining a lambda2 root does not solve the CT Eq.43 shear/shear
    or the Ball--James CC2/CC3 constraints. Preserve that epistemic boundary.
    """
    from twin_app.joint_metric_design_2026 import (
        find_joint_lambda2_target, state_at_joint_target,
    )

    key = (signature, length_relative_bound, angle_bound_deg)
    with st.container(border=True):
        st.markdown("**Coupled lattice changes: closest bounded A/M target**")
        st.caption(
            "Unlike the one-parameter scan, this varies all independent product "
            "cell parameters together. It minimizes the sum of squared changes "
            "relative to your permitted parameter bounds, with λ₂=1 as a "
            "numerical equality constraint. This is a local, bounded search, "
            "not a unique global optimum or a full supercompatibility design."
        )
        if st.button(
            "Find coupled λ₂=1 candidate",
            key="tf_metric_design_joint_search",
        ):
            try:
                with st.spinner("Solving bounded coupled metric problem…"):
                    report = find_joint_lambda2_target(
                        original, transformation_id,
                        length_relative_bound=length_relative_bound,
                        angle_bound_deg=angle_bound_deg,
                    )
                st.session_state["tf_metric_design_joint"] = (key, report)
            except Exception as exc:
                st.error(
                    "Coupled metric design failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                return

        saved = st.session_state.get("tf_metric_design_joint")
        if not saved or saved[0] != key:
            return
        report = saved[1]
        st.caption(
            f"Status: {report['status']}. "
            f"{report['states_evaluated']} metric trial states; "
            "search is not exhaustive."
        )
        candidate = report["candidate"]
        if candidate is None:
            st.info(
                "No λ₂=1 point was resolved inside these parameter bounds. "
                "This is not proof that no compatible cell exists."
            )
            return

        st.metric(
            "Rechecked |λ₂−1|",
            f"{abs(candidate['lambda2_residual']):.3g}",
        )
        old = candidate["baseline_cell"]
        new = candidate["proposed_cell"]
        st.dataframe(
            pd.DataFrame([
                {
                    "Cell parameter": name,
                    "Entered": old[name],
                    "Proposed": new[name],
                    "Change": new[name]-old[name],
                    "Unit": (
                        "degree" if name.endswith("_deg") else
                        "same length unit as entered"
                    ),
                }
                for name in ("a", "b", "c", "alpha_deg", "beta_deg", "gamma_deg")
            ]),
            hide_index=True, width="stretch",
        )
        if candidate["near_repeated_stretches"]:
            st.warning(
                "Near-repeated principal stretches: individual λ₂ "
                "sensitivities can be non-smooth. Check perturbations "
                "and lattice measurement uncertainty independently."
            )
        st.warning(
            "This candidate solves only the single-variant A/M metric "
            "condition numerically. The CT twin-specific shear/shear "
            "equation and Ball–James cofactor conditions have NOT yet "
            "been verified. No alloy composition or processing route "
            "is inferred."
        )

        if st.button(
            "Verify coupled target with CT and cofactor",
            key="tf_metric_design_joint_verify",
        ):
            try:
                from cualni_cryst.supercompatibility_atlas import (
                    evaluate_supercompatibility_state,
                )
                changed = state_at_joint_target(
                    original, transformation_id, candidate["parameter_values"],
                )
                with st.spinner("Evaluating exact CMC and twin-specific CT/BJ conditions…"):
                    result = evaluate_supercompatibility_state(
                        changed, transformation_id,
                        state_id="joint_metric_candidate",
                        metadata={
                            "design_method": report["method"],
                            "metric_only": True,
                            "input_signature": signature,
                        },
                    )
                st.session_state["tf_metric_design_joint_audit"] = (
                    key, candidate["parameter_values"], result.to_dict(),
                )
            except Exception as exc:
                st.error(
                    "Twin-specific independent audit failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                return

        verified = st.session_state.get("tf_metric_design_joint_audit")
        if (
            verified and verified[0] == key
            and verified[1] == candidate["parameter_values"]
        ):
            audit = verified[2]
            st.markdown("**Independent theory checks for this exact candidate**")
            st.dataframe(
                pd.DataFrame([
                    {"Check": "CT exact A/M at numerical tolerance",
                     "Result": str(audit["ct_am_exact_compatible"])},
                    {"Check": "CT A/M/M shear/shear",
                     "Result": str(audit["ct_supercompatible"])},
                    {"Check": "Cofactor conditions, evaluated by BJ",
                     "Result": str(audit["cofactor_compatible"])},
                    {"Check": "Matched M/M branch relations",
                     "Result": str(audit["matched_relation_count"])},
                ]), hide_index=True, width="stretch",
            )
            st.caption(
                "These are independently computed theory outputs. "
                "An aggregate CT and cofactor pass may involve different "
                "twin branches; inspect matched pair records before claiming "
                "agreement on one physical twin. Missing/partial matches "
                "are not proofs of equivalence."
            )
            st.json({
                "CT shear/shear residual": audit["ct_best_supercompatibility_residual"],
                "Cofactor CC1 residual": audit["cofactor_best_cc1_abs"],
                "Cofactor CC2 residual": audit["cofactor_best_cc2_abs"],
                "Cofactor CC3 margin": audit["cofactor_best_cc3_margin"],
                "Matched branch details": audit["pair_results"],
                "Not evaluable/error": audit["error"],
            })
            st.download_button(
                "Export coupled lattice design and independent audit",
                data=json.dumps({
                    "original_signature": signature,
                    "source_doi": "10.1016/j.actamat.2026.122399",
                    "method": report["method"],
                    "all_metric_candidates_verified": False,
                    "metric_design": report,
                    "theory_audit": audit,
                    "experimental_feasibility": "not assessed",
                }, indent=2, default=str),
                file_name="coupled_metric_design_ct_bj_audit.json",
                mime="application/json",
                key="tf_joint_metric_download",
            )
