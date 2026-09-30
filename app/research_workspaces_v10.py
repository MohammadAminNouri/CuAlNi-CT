from __future__ import annotations

"""Scientifically explicit Cayron-CT assessment layered over V9.

Presentation only: this module does not call or replace any CT, Ball--James,
PTMC, twinning, group-theory, orientation, EBSD or numerical solver.
"""

from typing import Any, Mapping

import pandas as pd
import streamlit as st

import app.research_workspaces as rw
import app.research_workspaces_v4 as v4
import app.research_workspaces_v9 as v9
from app.cayron_martensite_assessment import (
    AssessmentStep,
    build_cayron_martensite_assessment,
)


_original_conclusions = getattr(
    v4._render_conclusions,
    "_cayron_v10_original_conclusions",
    v4._render_conclusions,
)


def _status_box(status: str, answer: str) -> None:
    positive = {
        "validated",
        "reached",
        "exact reached",
        "CT-consistent martensitic crystallography",
    }
    neutral = {
        "not requested",
        "not evaluated",
        "not available",
        "diagnostic only",
        "not evaluable",
        "not reachable exactly",
        "not uniquely defined",
        "incomplete CT inventory",
        "partial CT martensitic construction",
    }
    if status in positive:
        st.success(f"**Answer — {status}.** {answer}")
    elif status in neutral:
        st.info(f"**Answer — {status}.** {answer}")
    else:
        st.warning(f"**Answer — {status}.** {answer}")


def _pretty_value(value: Any) -> Any:
    if value is None:
        return "not available"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) < 1.0e-4 or abs(value) >= 1.0e5:
            return f"{value:.6e}"
        return f"{value:.10g}"
    if isinstance(value, (tuple, list)):
        if not value:
            return "none"
        if all(not isinstance(item, (tuple, list, Mapping)) for item in value):
            return "[" + ", ".join(str(_pretty_value(item)) for item in value) + "]"
        return value
    return value


def _render_evidence(step: AssessmentStep) -> None:
    if not step.evidence:
        return

    simple_rows: list[dict[str, Any]] = []
    complex_rows: list[tuple[str, Any]] = []
    for label, value in step.evidence:
        shown = _pretty_value(value)
        if isinstance(shown, (str, int, float)):
            simple_rows.append({"quantity / check": label, "obtained": shown})
        else:
            complex_rows.append((label, shown))

    if simple_rows:
        st.dataframe(
            pd.DataFrame(simple_rows),
            hide_index=True,
            use_container_width=True,
        )

    for label, value in complex_rows:
        st.markdown(f"**{label}**")
        try:
            st.dataframe(
                pd.DataFrame(value),
                hide_index=True,
                use_container_width=True,
            )
        except Exception:
            st.write(value)


def _render_object_identity_guide(*, compact: bool = False) -> None:
    """Explain representation identities without conflating Cayron objects."""

    st.markdown(
        "**Object and notation discipline — C, T, R_F, F and U are distinct**"
    )

    rows = [
        {
            "symbol": "C_(M←A) [app]",
            "what it is": (
                "Lattice correspondence acting on crystallographic direct "
                "coordinates: u_M = C u_A. This is the app's canonical backend "
                "convention."
            ),
            "Cayron crosswalk": (
                "Same numerical matrix as Cayron C^(M→A). "
                "Cayron C^(A→M) = C^(-1)."
            ),
        },
        {
            "symbol": "T_(A→M) [Cayron]",
            "what it is": (
                "Passive crystallographic coordinate-transformation matrix "
                "representing an orientation relationship for a fixed physical vector."
            ),
            "Cayron crosswalk": (
                "An OR object; not the lattice correspondence C and not the "
                "transformation distortion F."
            ),
        },
        {
            "symbol": "R or R_F",
            "what it is": (
                "Proper rotation in an orthonormal/metric Cartesian representation. "
                "R_F denotes specifically the polar rotation in F = R_F U."
            ),
            "Cayron crosswalk": (
                "A Cartesian rotation representation must not be numerically identified "
                "with Cayron's passive T without an explicit frame/convention conversion."
            ),
        },
        {
            "symbol": "F",
            "what it is": (
                "Physical transformation distortion/deformation gradient. "
                "It acts on material vectors and can change lengths and angles."
            ),
            "Cayron crosswalk": "Not a correspondence or a mere OR matrix.",
        },
        {
            "symbol": "U",
            "what it is": (
                "Right stretch in the polar decomposition F = R_F U, expressed "
                "in an orthonormal Cartesian representation."
            ),
            "Cayron crosswalk": "Stretch only; not C, T, R_F or the full F.",
        },
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("**Correspondence notation crosswalk**")
    st.latex(
        r"C_{M\leftarrow A}^{\mathrm{app}}"
        r"\equiv C_{\mathrm{Cayron}}^{M\to A},"
        r"\qquad u_M=C_{M\leftarrow A}^{\mathrm{app}}u_A"
    )
    st.latex(
        r"C_{\mathrm{Cayron}}^{A\to M}"
        r"=(C_{M\leftarrow A}^{\mathrm{app}})^{-1}"
    )

    if compact:
        st.caption(
            "The superscript arrows in Cayron's correspondence notation name his "
            "basis/correspondence convention; the app instead writes the actual "
            "coordinate action explicitly as target ← source."
        )
        return

    st.markdown("**Direct vectors and reciprocal plane covectors transform dually**")
    st.latex(r"u_M=C\,u_A")
    st.latex(r"p_M=C^{-T}p_A")
    st.latex(r"p_M^Tu_M=p_A^Tu_A")
    st.caption(
        "u is a direct-lattice coordinate vector. p is a reciprocal-space "
        "plane covector/Miller-index object. A plane therefore uses the inverse transpose."
    )

    st.markdown("**Crystallographic coordinates are not Cartesian coordinates**")
    st.latex(
        r"M=B^TB,\qquad x_{\mathrm{cart}}=Bu,\qquad "
        r"n_{\mathrm{cart}}\propto B^{-T}p"
    )
    st.caption(
        "The metric M carries the non-orthonormal crystallographic geometry. "
        "Cartesian representations are bridges, not replacements for metric-native objects."
    )

    st.markdown("**CMC and SMC in the app convention**")
    st.latex(r"CMC=C^TM_MC-M_A")
    st.latex(r"SMC=M_A^{-1}-C^{-1}M_M^{-1}C^{-T}")
    st.caption(
        "These are metric/correspondence constructions. They are not definitions "
        "of an OR, F or U."
    )

    st.markdown("**Correspondence variants and intercorrespondence operators**")
    st.latex(r"H_C^A=G_A\cap C^{-1}G_MC")
    st.latex(
        r"\mathrm{variants}:~G_A/H_C^A,\qquad"
        r"\mathrm{operators}:~H_C^A\backslash G_A/H_C^A"
    )
    st.caption(
        "The operator count depends on the exact subgroup embedding produced by "
        "the current correspondence. Seven operators is the published B2→B19′ "
        "NiTi benchmark, not a universal hard-coded count."
    )


def _render_step(step: AssessmentStep) -> None:
    st.markdown(f"### {step.question}")
    _status_box(step.status, step.answer)

    if step.formulae:
        st.markdown("**Equations / CT construction used**")
        for formula in step.formulae:
            st.latex(formula)

    if step.step_id == "state":
        st.markdown(
            r"**Interpretation:** $C_{M\leftarrow A}$ is the lattice "
            r"correspondence used by the app. It is the same numerical matrix "
            r"that Cayron denotes $C^{M\to A}$, and it is distinct from the "
            r"passive OR transform $T$, the polar rotation $R_F$, the "
            r"distortion $F$, and the right stretch $U$."
        )
        with st.expander(
            "Cayron object identities and coordinate conventions",
            expanded=True,
        ):
            _render_object_identity_guide(compact=False)

    st.markdown("**Why this answer follows**")
    st.write(step.reasoning)

    st.markdown("**Physical meaning**")
    st.write(step.physical_meaning)

    st.caption(f"Limitation — {step.limitation}")

    if step.evidence:
        with st.expander(
            "Numerical evidence / provenance for this step",
            expanded=False,
        ):
            _render_evidence(step)


def _response_mapping(response: Any, name: str) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        value = response.get(name, {})
    else:
        value = getattr(response, name, {})
    return value if isinstance(value, Mapping) else {}


def _render_full_ct_audit(response: Any) -> None:
    result = _response_mapping(response, "result")
    metric = result.get("metric", {})
    ct = result.get("ct_detail", {})
    representation = result.get("representation", {})
    project = _response_mapping(response, "project_payload")

    if not isinstance(metric, Mapping):
        metric = {}
    if not isinstance(ct, Mapping):
        ct = {}
    if not isinstance(representation, Mapping):
        representation = {}

    with st.expander(
        "Full Cayron numerical audit — exact input, metrics and normalized spectrum",
        expanded=False,
    ):
        st.caption(
            "Read-only audit of quantities already calculated by the backend. "
            "Nothing in this expander recalculates CT."
        )

        _render_object_identity_guide(compact=True)

        transformations = project.get("transformations", [])
        if isinstance(transformations, list) and transformations:
            transformation = transformations[0]
            if isinstance(transformation, Mapping):
                C = transformation.get(
                    "correspondence_M_from_A",
                    transformation.get("correspondence"),
                )
                if C is not None:
                    st.markdown(
                        "**Exact canonical input correspondence "
                        "C_(M←A) = Cayron C^(M→A)**"
                    )
                    try:
                        st.dataframe(
                            pd.DataFrame(C),
                            hide_index=True,
                            use_container_width=True,
                        )
                    except Exception:
                        st.write(C)

        matrices = (
            ("Parent metric M_A", "parent_metric"),
            ("Product metric M_M", "product_metric"),
            ("Pulled-back product metric Cᵀ M_M C", "pulled_product_metric"),
            ("Dimensional CMC = Cᵀ M_M C − M_A", "cmc_dimensional"),
            (
                "Metric-normalized CMC used for generalized degeneracy diagnostics",
                "cmc_normalized",
            ),
            (
                "Dimensional SMC = M_A⁻¹ − C⁻¹ M_M⁻¹ C⁻ᵀ",
                "smc_dimensional",
            ),
            ("Metric-derived right stretch U", "stretch"),
        )
        for title, key in matrices:
            value = metric.get(key)
            if value is None:
                continue
            st.markdown(f"**{title}**")
            st.dataframe(
                pd.DataFrame(value),
                hide_index=True,
                use_container_width=True,
            )

        eta = ct.get("eta_eigenvalues")
        mu = ct.get("generalized_mu")
        if isinstance(eta, (list, tuple)):
            rows = []
            for i, eta_value in enumerate(eta):
                mu_value = (
                    mu[i]
                    if isinstance(mu, (list, tuple)) and i < len(mu)
                    else None
                )
                rows.append(
                    {
                        "i": i + 1,
                        "generalized μ_i": _pretty_value(mu_value),
                        "normalized η_i = μ_i − 1": _pretty_value(eta_value),
                    }
                )
            st.markdown("**App generalized / normalized CMC spectrum**")
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "These η_i classify zero multiplicity and inertia of the "
                "metric-normalized CMC. They are not presented as the numerical "
                "dimensional q_i eigenvalues used in Cayron's orthonormal CMC diagonalization."
            )

        st.markdown("**Exact versus diagnostic classification**")
        st.write(
            {
                "exact CT A/M": (
                    "satisfied"
                    if bool(ct.get("exact_compatible", False))
                    else "not satisfied"
                ),
                "degeneracy order": ct.get("degeneracy_order"),
                "reason": ct.get("reason"),
                "nearest normalized η index": ct.get("nearest_zero_index"),
                "app normalized degeneracy residual": ct.get(
                    "nearest_zero_residual"
                ),
                "inertia (negative, zero, positive)": ct.get("inertia"),
            }
        )

        if representation:
            st.markdown("**Cartesian representation audit**")
            st.caption(
                "If present below, F, R_F and U are Cartesian representations of "
                "physical distortion/stretch objects. They are not the lattice correspondence."
            )
            for title, key in (
                ("Deformation gradient F", "deformation_gradient"),
                ("Polar rotation R_F from F = R_F U", "polar_rotation"),
                ("Right stretch U from F = R_F U", "right_stretch"),
            ):
                value = representation.get(key)
                if value is None:
                    continue
                st.markdown(f"**{title}**")
                st.dataframe(
                    pd.DataFrame(value),
                    hide_index=True,
                    use_container_width=True,
                )


def _render_cayron_martensite_assessment(
    project: Any,
    transformation_id: str,
    base_signature: str,
) -> None:
    del project, transformation_id, base_signature

    response = st.session_state.get("current_response")
    if response is None:
        st.info(
            "A calculated transformation state is required before the "
            "Cayron martensitic assessment can be shown."
        )
        return

    unified = rw._current_unified_report()
    closing_requested = bool(
        st.session_state.get("research_v4_ct_closing", False)
    )
    super_requested = bool(
        st.session_state.get("research_v4_ct_super", False)
    )

    assessment = build_cayron_martensite_assessment(
        response,
        unified,
        closing_gap_requested=closing_requested,
        supercompatibility_requested=super_requested,
    )

    v4.section_header(
        assessment.title,
        (
            "For this exact parent metric, product metric and lattice correspondence, "
            "what does Cayron correspondence theory actually construct and which "
            "compatibility conditions are exact?"
        ),
        answer_hint=(
            "The assessment separates correspondence topology, M/M twins, exact "
            "single-variant A/M compatibility, closing-gap orientation candidates, "
            "A/M/M supercompatibility and experimental SMA evidence."
        ),
    )

    st.markdown("## Overall CT answer")
    _status_box(assessment.overall_status, assessment.overall_answer)
    st.write(assessment.overall_reasoning)

    st.caption(
        "Scope — CT-only conclusion. Ball–James, PTMC and experiment may be shown "
        "elsewhere for comparison, but none is used to manufacture a Cayron-CT verdict."
    )

    st.info(
        "Notation: app C_(M←A) = Cayron C^(M→A) and u_M = C u_A. "
        "Cayron C^(A→M) = C^(-1). T is a passive crystallographic OR coordinate "
        "transform; R_F is the polar rotation of F; F is the transformation "
        "distortion; U is the right stretch. These objects are not interchangeable."
    )

    st.caption(
        "Current optional CT request state — "
        f"closing-gap ORs: {'requested' if closing_requested else 'not requested'}; "
        f"supercompatibility: {'requested' if super_requested else 'not requested'}; "
        f"unified CT inventory: {'available' if unified is not None else 'not yet calculated'}."
    )

    st.divider()
    for index, step in enumerate(assessment.steps):
        _render_step(step)
        if index != len(assessment.steps) - 1:
            st.divider()

    _render_full_ct_audit(response)


def _render_conclusions_v10(
    project: Any,
    transformation_id: str,
    base_signature: str,
) -> None:
    # Preserve the existing calculation workflow and render the CT-only
    # assessment only after its authoritative outputs have been made available.
    _original_conclusions(project, transformation_id, base_signature)
    st.divider()
    _render_cayron_martensite_assessment(
        project,
        transformation_id,
        base_signature,
    )


_render_conclusions_v10._cayron_v10_original_conclusions = (  # type: ignore[attr-defined]
    _original_conclusions
)
v4._render_conclusions = _render_conclusions_v10


def render_research_extension() -> None:
    v9.render_research_extension()
