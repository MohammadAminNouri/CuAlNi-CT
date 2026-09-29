from __future__ import annotations

"""Question-first Cayron-CT martensitic assessment over final Phase-2C/V9.

This module is presentation-only.  It does not call or replace any CT,
Ball--James, PTMC, twinning, group-theory, orientation or EBSD solver.
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
            rendered = (_pretty_value(item) for item in value)
            return "[" + ", ".join(str(item) for item in rendered) + "]"

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
        st.dataframe(pd.DataFrame(simple_rows), hide_index=True, use_container_width=True)
    for label, value in complex_rows:
        st.markdown(f"**{label}**")
        try:
            st.dataframe(pd.DataFrame(value), hide_index=True, use_container_width=True)
        except Exception:
            st.write(value)


def _render_step(step: AssessmentStep) -> None:
    st.markdown(f"### {step.question}")
    _status_box(step.status, step.answer)

    if step.formulae:
        st.markdown("**Formula / CT construction used**")
        for formula in step.formulae:
            st.latex(formula)

    st.markdown("**Why this answer follows**")
    st.write(step.reasoning)

    st.markdown("**Physical meaning**")
    st.write(step.physical_meaning)

    st.caption(f"Limitation — {step.limitation}")

    if step.evidence:
        with st.expander("Numerical evidence / provenance for this step", expanded=False):
            _render_evidence(step)


def _render_full_ct_audit(response: Any) -> None:
    result = getattr(response, "result", {})
    if not isinstance(result, Mapping):
        return
    metric = result.get("metric", {})
    ct = result.get("ct_detail", {})
    project = getattr(response, "project_payload", {})
    if not isinstance(metric, Mapping):
        metric = {}
    if not isinstance(ct, Mapping):
        ct = {}
    if not isinstance(project, Mapping):
        project = {}

    with st.expander("Full Cayron numerical audit — matrices, spectrum and exact input", expanded=False):
        st.caption(
            "This is an audit view of already-computed quantities. No matrix shown here is recalculated by the presentation layer."
        )

        transformations = project.get("transformations", [])
        if isinstance(transformations, list) and transformations:
            transformation = transformations[0]
            if isinstance(transformation, Mapping):
                C = transformation.get(
                    "correspondence_M_from_A",
                    transformation.get("correspondence"),
                )
                if C is not None:
                    st.markdown("**Exact input correspondence C(M←A)**")
                    try:
                        st.dataframe(pd.DataFrame(C), hide_index=True, use_container_width=True)
                    except Exception:
                        st.write(C)

        matrices = (
            ("Parent metric M_A", "parent_metric"),
            ("Product metric M_M", "product_metric"),
            ("Pulled-back product metric Cᵀ M_M C", "pulled_product_metric"),
            ("Dimensional CMC", "cmc_dimensional"),
            ("Normalized CMC", "cmc_normalized"),
            ("Dimensional SMC", "smc_dimensional"),
            ("Right stretch U", "stretch"),
        )
        for title, key in matrices:
            value = metric.get(key)
            if value is None:
                continue
            st.markdown(f"**{title}**")
            st.dataframe(pd.DataFrame(value), hide_index=True, use_container_width=True)

        spectrum_rows = []
        eta = ct.get("eta_eigenvalues")
        mu = ct.get("generalized_mu")
        if isinstance(eta, (list, tuple)):
            for index, eta_value in enumerate(eta):
                mu_value = mu[index] if isinstance(mu, (list, tuple)) and index < len(mu) else None
                spectrum_rows.append(
                    {
                        "i": index,
                        "μ_i": _pretty_value(mu_value),
                        "η_i = μ_i − 1": _pretty_value(eta_value),
                    }
                )
        if spectrum_rows:
            st.markdown("**Generalized CMC spectrum**")
            st.dataframe(pd.DataFrame(spectrum_rows), hide_index=True, use_container_width=True)

        st.markdown("**Exact/diagnostic classification**")
        st.write(
            {
                "exact CT A/M": "satisfied" if bool(ct.get("exact_compatible", False)) else "not satisfied",
                "degeneracy order": ct.get("degeneracy_order"),
                "reason": ct.get("reason"),
                "nearest zero index": ct.get("nearest_zero_index"),
                "nearest zero residual": ct.get("nearest_zero_residual"),
                "inertia (negative, zero, positive)": ct.get("inertia"),
            }
        )


def _render_cayron_martensite_assessment(
    project: Any,
    transformation_id: str,
    base_signature: str,
) -> None:
    del project, transformation_id, base_signature  # presentation reads only the frozen calculated outputs

    response = st.session_state.get("current_response")
    if response is None:
        st.info(
            "A calculated transformation state is required before the Cayron martensitic assessment can be shown."
        )
        return

    unified = rw._current_unified_report()
    assessment = build_cayron_martensite_assessment(
        response,
        unified,
        closing_gap_requested=bool(st.session_state.get("research_v4_ct_closing", False)),
        supercompatibility_requested=bool(st.session_state.get("research_v4_ct_super", False)),
    )

    v4.section_header(
        assessment.title,
        "For the supplied parent lattice, product lattice and correspondence: what can Cayron CT actually reach, and what does that say about martensitic crystallography?",
        answer_hint=(
            "The assessment never equates 'not exactly compatible' with 'not martensite'. "
            "It separates CT variant/twin crystallography, exact A/M interface compatibility, orientation constructions, supercompatibility and experimental SMA functionality."
        ),
    )

    st.markdown("## Overall CT answer")
    _status_box(assessment.overall_status, assessment.overall_answer)
    st.write(assessment.overall_reasoning)
    st.caption(
        "Scope — this is a Cayron-CT crystallographic assessment. It does not use Ball–James or PTMC to decide the CT conclusion, "
        "and it does not claim experimental proof of martensite kinetics, reversibility or shape-memory behavior."
    )

    st.divider()
    for index, step in enumerate(assessment.steps):
        _render_step(step)
        if index != len(assessment.steps) - 1:
            st.divider()

    _render_full_ct_audit(response)


def _render_conclusions_v10(project: Any, transformation_id: str, base_signature: str) -> None:
    # Keep the complete existing comparison workflow untouched so its Calculate /
    # update action executes first.  Then render the CT-only subsection from the
    # authoritative outputs produced in that same Streamlit run.
    _original_conclusions(project, transformation_id, base_signature)
    st.divider()
    _render_cayron_martensite_assessment(project, transformation_id, base_signature)


_render_conclusions_v10._cayron_v10_original_conclusions = _original_conclusions  # type: ignore[attr-defined]
v4._render_conclusions = _render_conclusions_v10


def render_research_extension() -> None:
    v9.render_research_extension()
