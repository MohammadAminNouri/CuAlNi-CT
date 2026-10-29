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
        _render_full_paper_equation_audit(original, transformation_id, signature)
        # Independent source-specific Cu-Al-Ni papers: no specimen values are
        # inserted into the user's input or crystallographic inference.
        from twin_app.cualni_campaign_ui import render_cualni_literature_campaign
        render_cualni_literature_campaign(signature)

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
        _render_metric_failure_diagnosis(
            original, transformation_id, signature,
            length_percent/100.0, angle_window,
        )
        _render_same_twin_supercompatibility_inverse(
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
            if secondary.endswith("_offset_deg"):
                second_bound=2.0  # Not applied to angle controls by the solver.
                st.caption(
                    f"Angle variation for {secondary}: ±{angle_window:.3g} degrees "
                    "(controlled by the angle-window slider above)."
                )
            else:
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
                exploratory = data.get("ct_only_exploratory_target")
                if exploratory is not None:
                    st.markdown("**Exploratory CT-only lattice target (not verified supercompatibility)**")
                    st.write(
                        f"Sample {exploratory['sample_id']}: change "
                        f"{exploratory['secondary_parameter']} by "
                        f"{exploratory['secondary_parameter_delta']:+.8g} "
                        f"({'degrees' if secondary.endswith('_offset_deg') else 'relative scale'}). "
                        f"Recomputed minimum CT shear/shear ε within this state: "
                        f"{exploratory['best_CT_epsilon_in_that_state']:.8g}."
                    )
                    st.dataframe(pd.DataFrame([
                        {"Cell parameter": name, "Proposed value": value}
                        for name,value in exploratory["proposed_cell"].items()
                    ]),hide_index=True,width="stretch")
                    st.warning(
                        "This target minimizes the sampled CT shear/shear residual over "
                        "possibly different twin branches at different lattice states. "
                        "It does NOT prove the same twin improves, satisfy BJ cofactor "
                        "conditions, or attain full supercompatibility. Investigate "
                        "physical branch continuity before using it as a design target."
                    )
                st.caption(data["scientific_limit"])
                st.warning(
                    data.get("candidate_comparison_scope",
                             "Cross-state twin identity has not been validated.")
                )
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
                from twin_app.metric_failure_diagnostics import (
                    summarize_sampled_changes, _cell,
                )
                cause_report = summarize_sampled_changes(
                    data, _cell(original, transformation_id),
                )
                st.markdown("**Why each sampled target remains non-supercompatible**")
                st.dataframe(pd.DataFrame([
                    {
                        "Sample": item["sample_id"],
                        "Physical cell change": ", ".join(
                            f"{k}: {v:+.7g}" for k, v in
                            item["physical_cell_deltas"].items()
                        ),
                        "Failure / missing evidence": (
                            "; ".join(item["unmet_or_unevaluable_gates"])
                            or "No unresolved mathematical gate for this assigned pair"
                        ),
                        "CT ε": item["CT_epsilon"],
                        "CC1": item["CC1_abs"],
                        "CC2": item["CC2_abs"],
                        "CC3": item["CC3_margin"],
                    } for item in cause_report["samples"]
                ]), hide_index=True, width="stretch")
                st.caption(cause_report["scope"])
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


def _render_metric_failure_diagnosis(
    original, transformation_id, signature, length_bound, angle_bound,
) -> None:
    """Diagnostic reason -> quantified hypothetical metric change -> independent audit."""
    from twin_app.metric_failure_diagnostics import (
        metric_failure_analysis, supercompatibility_failure_analysis,
    )
    from twin_app.metric_design_targets import state_at_parameter
    from cualni_cryst.supercompatibility_atlas import evaluate_supercompatibility_state

    key = (signature, length_bound, angle_bound)
    with st.container(border=True):
        st.markdown("**Why compatibility fails · which parameter and by how much?**")
        st.caption(
            "Step A identifies the exact middle-stretch defect from the entered "
            "parent/product metric and correspondence. It tests each allowed "
            "parameter change within your bounds and reports an actual proposed "
            "cell in the entered units. Step B separately checks the CT habit, "
            "M/M twins, shear/shear equation and Ball–James cofactors."
        )
        if st.button(
            "Diagnose A/M failure and calculate parameter corrections",
            key="tf_failure_diagnose_metric",
        ):
            try:
                with st.spinner("Measuring the lambda2 defect and solving bounded metric targets…"):
                    result=metric_failure_analysis(
                        original, transformation_id,
                        length_relative_bound=length_bound,
                        angle_bound_deg=angle_bound,
                    )
                st.session_state["tf_failure_metric_report"]=(key,result)
            except Exception as exc:
                st.error(f"Failed metric diagnosis: {type(exc).__name__}: {exc}")
        stored=st.session_state.get("tf_failure_metric_report")
        if stored and stored[0]==key:
            metric=stored[1]
            st.write(metric["reason"])
            st.metric("Original λ₂−1",f"{metric['lambda2_minus_one']:+.9g}")
            candidates=metric["candidate_changes"]
            if candidates:
                st.markdown("**Numerically verified corrections for A/M λ₂=1**")
                st.dataframe(pd.DataFrame([{
                    "Change parameter":c["parameter"],
                    "Change magnitude":f"{c['abs_change']:.7g} {c['change_unit']}",
                    "Exact cell change":c["physical_change"],
                    "Recomputed |λ₂−1|":c["lambda2_residual"],
                    "Supercompatibility": "NOT VERIFIED",
                } for c in candidates]),hide_index=True,width="stretch")
                st.caption(
                    "These rows vary one independent parameter at a time. "
                    "The coupled A/M optimizer above can combine parameters; "
                    "neither option is an alloy-composition prescription."
                )
                index=st.selectbox(
                    "Select one proposed cell for the full mathematical audit",
                    range(len(candidates)),
                    format_func=lambda i: (
                        f"{candidates[i]['parameter']}: "
                        f"{candidates[i]['physical_change']}"
                    ),
                    key="tf_failure_metric_candidate",
                )
                selected=candidates[index]
                target_key=(
                    key,selected["parameter"],selected["target_parameter_value"],
                )
                if st.button(
                    "Identify remaining CT shear/shear and CC1–CC3 failures at this cell",
                    key="tf_failure_verify_candidate",
                ):
                    try:
                        new_state=state_at_parameter(
                            original, transformation_id, selected["parameter"],
                            selected["target_parameter_value"],
                        )
                        with st.spinner("Re-solving existing CT and Ball–James equations…"):
                            audit=evaluate_supercompatibility_state(
                                new_state, transformation_id,
                                state_id="inverse_metric_diagnostic",
                                metadata={
                                    "proposed_cell":selected["proposed_cell"],
                                    "hypothetical":True,
                                },
                            ).to_dict()
                        st.session_state["tf_failure_target_audit"]=(target_key,audit)
                    except Exception as exc:
                        st.error(f"Target CT/BJ verification failed: {type(exc).__name__}: {exc}")
                saved=st.session_state.get("tf_failure_target_audit")
                if saved and saved[0]==target_key:
                    assessment=supercompatibility_failure_analysis(saved[1])
                    _display_condition_diagnosis(assessment)
                    st.download_button(
                        "Export full diagnosis and proposed cell",
                        data=json.dumps({
                            "entered_signature":signature,
                            "metric_diagnosis":metric,
                            "target":selected,
                            "exact_theory_audit":saved[1],
                            "failure_analysis":assessment,
                        }, indent=2,default=str),
                        file_name="compatibility_failure_and_target_audit.json",
                        mime="application/json",
                        key="tf_failure_export_target",
                    )
            else:
                st.info(
                    "No independent parameter target reached λ₂=1 inside the tested bounds. "
                    "That does not prove nonexistence. Try broader physically admissible "
                    "bounds or coupled changes and inspect the full audit."
                )
            st.caption(metric["warning"])
        if st.button(
            "Explain current full CT/Ball–James compatibility",
            key="tf_failure_current_verify",
        ):
            try:
                with st.spinner("Auditing exact CMC, CT M/M and shear/shear, and cofactor conditions…"):
                    audit=evaluate_supercompatibility_state(
                        original,transformation_id,
                        state_id="entered_crystal_metric",
                    ).to_dict()
                st.session_state["tf_failure_current_audit"]=(signature,audit)
            except Exception as exc:
                st.error(f"Current-state compatibility audit failed: {type(exc).__name__}: {exc}")
        current=st.session_state.get("tf_failure_current_audit")
        if current and current[0]==signature:
            st.markdown("**Why the entered state does or does not satisfy each condition**")
            _display_condition_diagnosis(
                supercompatibility_failure_analysis(current[1])
            )


def _display_condition_diagnosis(analysis: dict) -> None:
    st.caption(analysis["cause_note"])
    st.markdown(f"**Result: {analysis['status'].replace('_',' ')}**")
    st.dataframe(pd.DataFrame([
        {
            "Mathematical condition":x["condition"],
            "State":x["status"],
            "Calculated observation":(
                json.dumps(x["observed"], default=str)
                if isinstance(x["observed"], dict) else x["observed"]
            ),
            "Why it failed or cannot be concluded":x["interpretation"],
        } for x in analysis["conditions"]
    ]),hide_index=True,width="stretch")
    st.caption(analysis["recommended_sequence"])


def _render_full_paper_equation_audit(
    original, transformation_id: str, signature: str,
) -> None:
    """Cayron's intermediate equations, completely separate from our CT solver."""
    from twin_app.paper_methodology_2022_2026 import (
        audit_project_paper_equations,
    )
    with st.expander(
        "Full paper-method audit · 2022/2026 CMC, SMC and all A/M habits",
        expanded=False,
    ):
        st.markdown(
            "**2022 correspondence:** The rational parent/product correspondence "
            "generates symmetry cosets (M variants) and double cosets (operators). "
            "A correspondence is not an observed orientation relationship."
        )
        st.markdown(
            "**2026 compatibility:** CMC Eq.32 → CMC quadratic degeneracy "
            "Eqs.35–39 → each A/M IPS habit → SMC Eqs.40–41 → "
            "M/M transformation twin → shear/shear Eq.42/43. "
            "BJ CC1/CC2/CC3 must also refer to the same valid twin."
        )
        if st.button(
            "Compute all CMC/SMC intermediate values independently",
            key="tf_full_paper_cmcsms_run",
        ):
            try:
                with st.spinner("Recalculating Cayron's direct-metric equations…"):
                    result=audit_project_paper_equations(original,transformation_id)
                st.session_state["tf_full_paper_cmcsms"]=(signature,result)
            except Exception as exc:
                st.error(f"Paper equations unavailable: {type(exc).__name__}: {exc}")
        saved=st.session_state.get("tf_full_paper_cmcsms")
        if not saved or saved[0]!=signature:
            return
        d=saved[1]
        st.markdown(f"**CMC classification:** {d['degeneracy_class'].replace('_',' ')}")
        st.dataframe(pd.DataFrame([
            {"Check":"Principal stretches λ₁, λ₂, λ₃",
             "Value":", ".join(f"{x:.10g}" for x in d["principal_stretches"])},
            {"Check":"λ₂−1","Value":f"{d['lambda2_minus_one']:+.10g}"},
            {"Check":"CMC inertia (negative, zero, positive)",
             "Value":str(d["cmc_inertia"])},
            {"Check":"Degeneracy order","Value":d["degeneracy_order"]},
            {"Check":"Individual A/M compatible","Value":d["individual_AM_compatible"]},
            {"Check":"Number of A/M habit planes",
             "Value":d["theoretical_habit_plane_count"]},
        ]),hide_index=True,width="stretch")
        left,right=st.columns(2)
        with left:
            st.markdown("**CMC · Eq. 32**")
            st.code("\n".join("  ".join(f"{v:+.9g}" for v in row)
                              for row in d["CMC_matrix"]),language="text")
        with right:
            st.markdown("**SMC · Eq. 41**")
            st.code("\n".join("  ".join(f"{v:+.9g}" for v in row)
                              for row in d["SMC_matrix"]),language="text")
        for i,h in enumerate(d["habit_planes"],1):
            with st.expander(f"Individual A/M IPS habit {i}",expanded=i==1):
                st.write("Habit-plane covector · parent reciprocal crystal:",
                         h["habit_plane_parent_reciprocal"])
                st.write("IPS shear vector · parent direct crystal:",
                         h["IPS_shear_parent_direct"])
                st.write("Parent-metric shear magnitude:",
                         h["IPS_shear_metric_norm"])
                st.write("pᵀd (IPS normal strain):",
                         h["volume_change_p_dot_d"])
                st.caption(
                    "This IPS shear is NOT the martensite–martensite twin shear. "
                    "Eq.42 requires the selected M/M twin plane and direction."
                )
        st.warning(d["scientific_limit"])
        st.download_button(
            "Export complete source-equation audit (JSON)",
            data=json.dumps(d,indent=2,default=str),
            file_name="cayron_2022_2026_full_cmc_smc_audit.json",
            mime="application/json",
            key="tf_full_paper_cmcsms_download",
        )


def _render_same_twin_supercompatibility_inverse(
    original, transformation_id: str, signature: str,
    length_bound: float, angle_bound: float,
) -> None:
    """Isolated, source-provenance-preserving inverse design for fixed CT twin."""
    from twin_app.same_twin_continuity_2026 import enumerate_twin_anchors
    from twin_app.same_twin_inverse_design_2026 import (
        optimize_one_tracked_twin, verify_candidate_same_twin_cofactor,
    )
    with st.expander(
        "Inverse supercompatibility · one continuously tracked physical twin",
        expanded=False,
    ):
        st.markdown(
            "**Cayron 2022 identity:** a fixed rational parent mirror or "
            "180° axis defines the generic Type-I or Type-II twin. "
            "Its identity is NEVER an arbitrary operator/twin array index. "
            "The app checks how its plane, direction and shear evolve."
        )
        st.markdown(
            "**Cayron 2026 inverse design:** enforce λ₂=1 analytically, "
            "track one specific CMC habit, minimize Eq. 42 "
            "ε = ‖2(mᵀn)d−sη‖/|s|, and independently check whether "
            "CC1–CC3 agree on the same physical CT/BJ twin relation."
        )
        st.warning(
            "The solver is bounded and local, not exhaustive. "
            "It proposes a HYPOTHETICAL crystal cell, not a measured alloy "
            "or chemical recipe. A completed CT-only calculation is NOT "
            "a full Ball–James/cofactor supercompatibility certificate."
        )
        if st.button(
            "Enumerate physically identifiable CT twin generators",
            key="tf_same_twin_enumerate",
        ):
            try:
                with st.spinner("Enumerating exact parent symmetry generators and CT twins…"):
                    anchors=enumerate_twin_anchors(original, transformation_id)
                st.session_state["tf_same_twin_anchors"]=(signature,anchors)
            except Exception as exc:
                st.error(f"CT twin enumeration failed: {type(exc).__name__}: {exc}")
        inventory=st.session_state.get("tf_same_twin_anchors")
        if not inventory or inventory[0]!=signature:
            st.caption("Enumerate exact CT twin generators to choose the physical branch.")
            return
        anchors=inventory[1]
        st.metric("Trackable exact CT construction branches",len(anchors))
        st.caption(
            "A compound physical twin can appear through both Type-I and "
            "Type-II generators; the two construction routes remain separate "
            "for continuity. Weak-plane candidates are not declared exact twins."
        )
        if not anchors:
            st.info("No nonzero-shear Type-I/II CT generators for these inputs.")
            return
        idx=st.selectbox(
            "Choose a frozen parent symmetry generator / construction",
            range(len(anchors)),
            format_func=lambda i: (
                f"{anchors[i].construction_route} · "
                f"{anchors[i].branch_id} · s={anchors[i].initial_shear:.7g}"
            ),
            key="tf_same_twin_selected",
        )
        chosen=anchors[idx]
        st.code(chosen.generator_key,language="text")
        st.caption(
            "This exact rational generator, the parent/product symmetries and "
            "the correspondence are frozen. Geometry discontinuity or "
            "loss of this twin makes the proposed path inadmissible."
        )
        a,b=st.columns(2)
        with a:
            habit=st.selectbox(
                "CMC A/M habit to follow",
                (0,1),
                format_func=lambda x:f"Habit {x+1} (from λ₂-corrected anchor)",
                key="tf_same_twin_habit",
            )
        with b:
            budget=st.slider(
                "Maximum attempted optimization evaluations",
                24,180,72,12,
                key="tf_same_twin_budget",
            )
        cc2_weight=st.slider(
            "Relative weight of same-twin CC2-equivalent squared-norm gap",
            min_value=0.0,max_value=2.0,value=.25,step=.05,
            key="tf_same_twin_cc2_weight",
        )
        st.caption(
            "Cayron/Zhang: Type I uses ‖U⁻¹e‖=1; Type II uses ‖Ue‖=1. "
            "This is an independent metric-equivalent CC2 criterion. "
            "A full Ball–James CC3 test still requires a geometrically "
            "comparable M/M twin after the inverse search."
        )
        steps=st.select_slider(
            "Intermediate continuity checks along the complete cell change",
            options=(5,7,9,11),value=7,
            key="tf_same_twin_steps",
        )
        uniform_bound=min(.35,max(.05,length_bound*1.6))
        key=(signature,chosen.branch_id,habit,length_bound,
             angle_bound,uniform_bound,int(budget),int(steps),float(cc2_weight))
        if st.button(
            "Find exact parameter changes for this same twin's Eq. 42",
            key="tf_same_twin_inverse_run",
        ):
            try:
                with st.spinner(
                    "Solving exact λ₂ constraint, same-twin Eq.42, "
                    "habit continuity and individual parameter counterfactuals…"
                ):
                    outcome=optimize_one_tracked_twin(
                        original,transformation_id,
                        branch_id=chosen.branch_id,habit_index=int(habit),
                        relative_length_bound=length_bound,
                        angle_bound_deg=angle_bound,
                        max_uniform_scale_offset=uniform_bound,
                        max_evaluations=int(budget),
                        cc2_equivalent_weight=float(cc2_weight),
                        continuity_steps=int(steps),
                    )
                st.session_state["tf_same_twin_inverse_result"]=(key,outcome)
            except Exception as exc:
                st.error(f"Same-twin optimizer failed: {type(exc).__name__}: {exc}")
        saved=st.session_state.get("tf_same_twin_inverse_result")
        if not saved or saved[0]!=key:
            return
        result=saved[1]
        st.markdown(f"**Result:** {result['status'].replace('_',' ')}")
        st.caption(
            f"Unique physical twin generator: {result['branch_id']}. "
            f"States evaluated: {result.get('states_evaluated','N/A')}. "
            "No global-optimum or experimental-attainability guarantee."
        )
        candidate=result.get("candidate")
        if candidate is None:
            st.warning(
                result.get("reason",
                    "No admissible continuous same-twin/habit inverse candidate "
                    "was resolved in the selected bounds."
                )
            )
            st.json(result.get("rejected_paths",[]))
            return
        d1,d2,d3=st.columns(3)
        d1.metric("Original Eq.42 ε",f"{candidate['baseline_epsilon']:.7g}")
        d2.metric("Proposed Eq.42 ε",f"{candidate['candidate_epsilon']:.7g}")
        d3.metric("Reduction in ε",f"{candidate['epsilon_reduction_absolute']:+.7g}")
        st.dataframe(pd.DataFrame([
            {
                "Lattice parameter":name,
                "Original":candidate["baseline_cell"][name],
                "Proposed":value,
                "Physical change":(
                    f"{candidate['length_change_percent'][name]:+.7g}%"
                    if name in candidate["length_change_percent"]
                    else f"{candidate['angle_change_deg'][name]:+.7g}°"
                ),
            }
            for name,value in candidate["proposed_cell"].items()
        ]),hide_index=True,width="stretch")
        st.markdown("**Which lattice change matters for this tracked twin?**")
        st.dataframe(pd.DataFrame([
            {
                "Independent axis":x["parameter"],
                "Proposed scale/angle":x["proposed_value"],
                "Normalized change":x["normalized_displacement"],
                "ε if this factor were not changed":x["epsilon_without_this_change"],
                "CC2 gap if factor removed":x[
                    "CC2_equivalent_gap_without_this_change"
                ],
                "Conditional decrease from changing it":x[
                    "conditional_epsilon_reduction_due_to_change"
                ],
                "Attribution available?":x["removal_status"],
            }
            for x in candidate["factor_by_factor_counterfactuals"]
        ]),hide_index=True,width="stretch")
        st.caption(
            "Counterfactuals keep the other independent parameters fixed and "
            "retune the λ₂=1 uniform scale. Contributions are nonadditive; "
            "a positive value means that factor helped reduce ε, not that "
            "it independently causes supercompatibility in a real alloy."
        )
        st.markdown("**Why Eq.42 still fails or closes**")
        eq=candidate["eq42_vector_decomposition"]
        st.dataframe(pd.DataFrame([
            {"Constraint":"λ₂−1 (exact A/M metric)",
             "Value":candidate["lambda2_residual"]},
            {"Constraint":"Eq.42 normalized ε",
             "Value":candidate["candidate_epsilon"]},
            {"Constraint":"Same-generator CC2-equivalent squared-norm minus one",
             "Value":candidate[
                "candidate_same_generator_CC2_equivalent"
             ]["squared_norm_minus_one"]},
            {"Constraint":"CC2-equivalent metric equality reached?",
             "Value":candidate["numeric_same_generator_CC2_equivalent_pass"]},
            {"Constraint":"Signed shear amplitude difference",
             "Value":eq["longitudinal_magnitude_gap"]},
            {"Constraint":"Shear direction transverse gap",
             "Value":eq["transverse_direction_gap"]},
            {"Constraint":"Signed direction angle (°)",
             "Value":eq["direction_angle_deg"]},
        ]),hide_index=True,width="stretch")
        st.markdown("**Same-twin and same-habit continuity across the entire change**")
        st.dataframe(pd.DataFrame(candidate["continuation_path"]),
                     hide_index=True,width="stretch")
        st.caption(
            "The recorded path uses one exact generator and continuity of "
            "the twin plane, shear direction, shear magnitude and CMC habit. "
            "It cannot establish external mechanical, chemical or EBSD validity."
        )
        if st.button(
            "Verify THIS fixed CT generator and habit against BJ CC1–CC3",
            key="tf_same_twin_bj_verify",
        ):
            try:
                with st.spinner("Recomputing CT/BJ physical assignments at the proposed cell…"):
                    bj=verify_candidate_same_twin_cofactor(
                        original,transformation_id,result,
                        length_bound=length_bound,
                        angle_bound_deg=angle_bound,
                        uniform_scale_bound=uniform_bound,
                    )
                st.session_state["tf_same_twin_bj_outcome"]=(key,bj)
            except Exception as exc:
                st.error(f"Same-pair cofactor verification failed: {type(exc).__name__}: {exc}")
        bj_cache=st.session_state.get("tf_same_twin_bj_outcome")
        verification=None
        if bj_cache and bj_cache[0]==key:
            verification=bj_cache[1]
            st.markdown("**Same-physical-pair Ball–James verification**")
            st.write(verification["status"])
            st.json(verification)
            if not verification["same_pair_full_supercompatibility_verified"]:
                st.warning(
                    "No combined certificate has been established. A BJ "
                    "cofactor pass on any *other* twin must not be combined "
                    "with this CT branch. Partial physical assignments "
                    "are UNRESOLVED, not automatically false."
                )
        st.download_button(
            "Download fixed-twin optimization and scientific evidence (JSON)",
            data=json.dumps({
                "method":"Cayron 2022 symmetry identity + 2026 Eq42 inverse",
                "source_dois":[
                    "10.3390/cryst12020130",
                    "10.1016/j.actamat.2026.122399"
                ],
                "exact_input_signature":signature,
                "local_CT_inverse":result,
                "independent_same_pair_BJ_audit":verification,
                "experimental_feasibility":"NOT_ASSESSED",
            },indent=2,default=str),
            file_name="same_twin_full_supercompatibility_inverse_audit.json",
            mime="application/json",
            key="tf_same_twin_export",
        )
