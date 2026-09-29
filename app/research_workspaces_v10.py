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
            simple_rows.append(
                {
                    "quantity / check": label,
                    "obtained": shown,
                }
            )
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
    """Explain Cayron object identities without altering any calculation.

    C, T/R, F and U are deliberately kept as distinct mathematical and
    crystallographic objects. This helper is presentation-only.
    """

    st.markdown(
        "**Object identity — do not interchange "
        "C, T/R, F and U**"
    )

    rows = [
        {
            "symbol": "C(M←A)",
            "what it is": (
                "Lattice correspondence. It maps parent direct-lattice "
                "coordinates/indices to product direct-lattice coordinates "
                "under the convention u_M = C u_A."
            ),
            "what it is NOT": (
                "Not an orientation relationship; not a physical deformation "
                "gradient; not a stretch tensor; not a rigid-body rotation."
            ),
        },
        {
            "symbol": "T/R(M←A)",
            "what it is": (
                "Orientation-relationship operator: the relative orientation "
                "of parent and product crystal frames in physical space. "
                "In an orthonormal Cartesian representation it is a proper "
                "rotation."
            ),
            "what it is NOT": (
                "Not the lattice correspondence C; not the full transformation "
                "deformation F; not the stretch U. It is also not automatically "
                "the polar rotation R_F unless that equality is independently "
                "established."
            ),
        },
        {
            "symbol": "F",
            "what it is": (
                "Transformation deformation gradient / distortion in physical "
                "space. It maps material vectors through x' = F x and generally "
                "changes lengths and/or angles."
            ),
            "what it is NOT": (
                "Not a mere relabelling of lattice indices; not the "
                "correspondence C; not, in general, a pure orientation rotation; "
                "not the stretch U alone."
            ),
        },
        {
            "symbol": "U",
            "what it is": (
                "Right stretch tensor from the polar decomposition F = R_F U, "
                "with U = (F^T F)^(1/2) in an orthonormal Cartesian "
                "representation. It contains the stretch/strain part of F."
            ),
            "what it is NOT": (
                "Not the correspondence C; not an orientation relationship; "
                "not a rigid-body rotation; not the complete deformation F."
            ),
        },
    ]

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
    )

    if compact:
        st.caption(
            "Object identity is physical, not merely numerical: two matrices "
            "can happen to contain the same numbers in a special representation "
            "and still represent different crystallographic objects."
        )
        return

    st.markdown(
        "**Direct and reciprocal objects transform dually "
        "under the correspondence**"
    )

    st.latex(
        r"\mathbf u_M=C_{M\leftarrow A}\mathbf u_A"
    )

    st.latex(
        r"\mathbf p_M=C_{M\leftarrow A}^{-T}\mathbf p_A"
    )

    st.latex(
        r"\mathbf p_M^T\mathbf u_M"
        r"=\mathbf p_A^T\mathbf u_A"
    )

    st.caption(
        "u is a direct-lattice coordinate vector. p is a reciprocal-space "
        "plane covector / Miller-index object. The inverse transpose is "
        "required for planes."
    )

    st.markdown(
        "**Crystallographic coordinates are not Cartesian coordinates**"
    )

    st.latex(
        r"M=B^TB,\qquad "
        r"\mathbf x=B\mathbf u,\qquad "
        r"\mathbf n_{\mathrm{cart}}\propto B^{-T}\mathbf p"
    )

    st.caption(
        "The lattice metric is part of the crystallography. Cartesian vectors "
        "are a representation bridge; they do not replace the metric-native "
        "objects."
    )

    st.markdown(
        "**The CT metric-compatibility objects remain metric-native**"
    )

    st.latex(
        r"CMC=C^TM_MC-M_A"
    )

    st.latex(
        r"SMC=M_A^{-1}-C^{-1}M_M^{-1}C^{-T}"
    )

    st.caption(
        "CMC and SMC are compatibility constructions built from the lattice "
        "metrics and the correspondence C. They are not orientation "
        "relationships and are not definitions of F or U."
    )

    st.markdown(
        "**Variants and operators are different symmetry objects**"
    )

    st.latex(
        r"H=G_A\cap C^{-1}G_MC"
    )

    st.latex(
        r"\text{variants: }G_A/H"
        r"\qquad"
        r"\text{operator classes: }H\backslash G_A/H"
    )

    st.caption(
        "A representative operator double coset has the form H g H. "
        "Variant count and operator-class count therefore answer different "
        "crystallographic questions."
    )


def _render_step(step: AssessmentStep) -> None:
    st.markdown(f"### {step.question}")
    _status_box(step.status, step.answer)

    if step.formulae:
        st.markdown("**Formula / CT construction used**")

        for formula in step.formulae:

            # The assessment backend historically supplied the shorthand
            # C != R. Do not render it as a literal matrix inequality.
            #
            # The scientifically correct statement is that correspondence
            # and orientation relationship are DIFFERENT KINDS OF OBJECTS.
            compact_formula = formula.replace(" ", "")

            if compact_formula in {
                r"C\neqR",
                r"C\neR",
            }:
                continue

            st.latex(formula)

    if step.step_id == "state":

        st.markdown(
            r"**Interpretation:** "
            r"$C_{M\leftarrow A}$ is the lattice correspondence. "
            r"It is **not** the orientation relationship $T/R$, "
            r"it is **not** the deformation gradient $F$, "
            r"and it is **not** the right stretch $U$."
        )

        with st.expander(
            "Cayron object identities and coordinate conventions",
            expanded=True,
        ):
            _render_object_identity_guide(
                compact=False,
            )

    st.markdown(
        "**Why this answer follows**"
    )
    st.write(step.reasoning)

    st.markdown(
        "**Physical meaning**"
    )
    st.write(step.physical_meaning)

    st.caption(
        f"Limitation — {step.limitation}"
    )

    if step.evidence:
        with st.expander(
            "Numerical evidence / provenance for this step",
            expanded=False,
        ):
            _render_evidence(step)


def _render_full_ct_audit(
    response: Any,
) -> None:

    result = getattr(
        response,
        "result",
        {},
    )

    if not isinstance(
        result,
        Mapping,
    ):
        return

    metric = result.get(
        "metric",
        {},
    )

    ct = result.get(
        "ct_detail",
        {},
    )

    representation = result.get(
        "representation",
        {},
    )

    project = getattr(
        response,
        "project_payload",
        {},
    )

    if not isinstance(metric, Mapping):
        metric = {}

    if not isinstance(ct, Mapping):
        ct = {}

    if not isinstance(
        representation,
        Mapping,
    ):
        representation = {}

    if not isinstance(project, Mapping):
        project = {}

    with st.expander(
        "Full Cayron numerical audit — matrices, spectrum and exact input",
        expanded=False,
    ):

        st.caption(
            "This is an audit view of already-computed quantities. "
            "No matrix shown here is recalculated by the presentation layer."
        )

        st.markdown(
            "### How to read the matrices below"
        )

        _render_object_identity_guide(
            compact=True,
        )

        transformations = project.get(
            "transformations",
            [],
        )

        if (
            isinstance(transformations, list)
            and transformations
        ):

            transformation = transformations[0]

            if isinstance(
                transformation,
                Mapping,
            ):

                C = transformation.get(
                    "correspondence_M_from_A",
                    transformation.get(
                        "correspondence"
                    ),
                )

                if C is not None:

                    st.markdown(
                        "**Exact input lattice correspondence C(M←A) — "
                        "not an OR, not F, not U**"
                    )

                    st.caption(
                        "C(M←A) acts on direct-lattice coordinates as "
                        "u_M = C u_A; reciprocal plane covectors transform "
                        "with C^(-T)."
                    )

                    try:
                        st.dataframe(
                            pd.DataFrame(C),
                            hide_index=True,
                            use_container_width=True,
                        )
                    except Exception:
                        st.write(C)

        metric_matrices = (
            (
                "Parent metric M_A",
                "parent_metric",
            ),
            (
                "Product metric M_M",
                "product_metric",
            ),
            (
                "Pulled-back product metric Cᵀ M_M C",
                "pulled_product_metric",
            ),
            (
                "Dimensional CMC = Cᵀ M_M C − M_A",
                "cmc_dimensional",
            ),
            (
                "Normalized CMC — representation/diagnostic form, "
                "not a replacement for dimensional CMC",
                "cmc_normalized",
            ),
            (
                "Dimensional SMC = M_A⁻¹ − C⁻¹ M_M⁻¹ C⁻ᵀ",
                "smc_dimensional",
            ),
            (
                "Metric-derived right stretch U — stretch only; "
                "not C and not an OR",
                "stretch",
            ),
        )

        for title, key in metric_matrices:

            value = metric.get(key)

            if value is None:
                continue

            st.markdown(
                f"**{title}**"
            )

            st.dataframe(
                pd.DataFrame(value),
                hide_index=True,
                use_container_width=True,
            )

        if representation:

            st.markdown(
                "### Cartesian representation audit"
            )

            st.caption(
                "These matrices are Cartesian representations derived from "
                "the same crystallographic state. They must not be confused "
                "with the lattice correspondence C."
            )

            representation_matrices = (
                (
                    "Deformation gradient F — physical distortion; "
                    "not C, not an OR, not U",
                    "deformation_gradient",
                ),
                (
                    "Polar rotation R_F from F = R_F U — rotation part of F; "
                    "not automatically the crystallographic OR T/R",
                    "polar_rotation",
                ),
                (
                    "Right stretch U from F = R_F U — stretch part of F; "
                    "not C, not T/R, not the full F",
                    "right_stretch",
                ),
            )

            for title, key in representation_matrices:

                value = representation.get(key)

                if value is None:
                    continue

                st.markdown(
                    f"**{title}**"
                )

                st.dataframe(
                    pd.DataFrame(value),
                    hide_index=True,
                    use_container_width=True,
                )

            parent_convention = representation.get(
                "parent_convention"
            )

            product_convention = representation.get(
                "product_convention"
            )

            parity_residual = representation.get(
                "maximum_parity_residual"
            )

            representation_rows = []

            if parent_convention is not None:
                representation_rows.append(
                    {
                        "representation item":
                            "parent Cartesian convention",
                        "value":
                            parent_convention,
                    }
                )

            if product_convention is not None:
                representation_rows.append(
                    {
                        "representation item":
                            "product Cartesian convention",
                        "value":
                            product_convention,
                    }
                )

            if parity_residual is not None:
                representation_rows.append(
                    {
                        "representation item":
                            "maximum representation-parity residual",
                        "value":
                            _pretty_value(
                                parity_residual
                            ),
                    }
                )

            if representation_rows:

                st.dataframe(
                    pd.DataFrame(
                        representation_rows
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

        spectrum_rows = []

        eta = ct.get(
            "eta_eigenvalues"
        )

        mu = ct.get(
            "generalized_mu"
        )

        if isinstance(
            eta,
            (list, tuple),
        ):

            for index, eta_value in enumerate(eta):

                mu_value = (
                    mu[index]
                    if isinstance(
                        mu,
                        (list, tuple),
                    )
                    and index < len(mu)
                    else None
                )

                spectrum_rows.append(
                    {
                        "i":
                            index,
                        "μ_i":
                            _pretty_value(
                                mu_value
                            ),
                        "η_i = μ_i − 1":
                            _pretty_value(
                                eta_value
                            ),
                    }
                )

        if spectrum_rows:

            st.markdown(
                "**Generalized CMC spectrum**"
            )

            st.dataframe(
                pd.DataFrame(
                    spectrum_rows
                ),
                hide_index=True,
                use_container_width=True,
            )

        st.markdown(
            "**Exact/diagnostic classification**"
        )

        st.write(
            {
                "exact CT A/M":
                    (
                        "satisfied"
                        if bool(
                            ct.get(
                                "exact_compatible",
                                False,
                            )
                        )
                        else "not satisfied"
                    ),
                "degeneracy order":
                    ct.get(
                        "degeneracy_order"
                    ),
                "reason":
                    ct.get(
                        "reason"
                    ),
                "nearest zero index":
                    ct.get(
                        "nearest_zero_index"
                    ),
                "nearest zero residual":
                    ct.get(
                        "nearest_zero_residual"
                    ),
                "inertia (negative, zero, positive)":
                    ct.get(
                        "inertia"
                    ),
            }
        )


def _render_cayron_martensite_assessment(
    project: Any,
    transformation_id: str,
    base_signature: str,
) -> None:

    del (
        project,
        transformation_id,
        base_signature,
    )

    response = st.session_state.get(
        "current_response"
    )

    if response is None:

        st.info(
            "A calculated transformation state is required before the "
            "Cayron martensitic assessment can be shown."
        )

        return

    unified = rw._current_unified_report()

    assessment = build_cayron_martensite_assessment(
        response,
        unified,
        closing_gap_requested=bool(
            st.session_state.get(
                "research_v4_ct_closing",
                False,
            )
        ),
        supercompatibility_requested=bool(
            st.session_state.get(
                "research_v4_ct_super",
                False,
            )
        ),
    )

    v4.section_header(
        assessment.title,
        (
            "For the supplied parent lattice, product lattice and "
            "correspondence: what can Cayron CT actually reach, and "
            "what does that say about martensitic crystallography?"
        ),
        answer_hint=(
            "The assessment never equates 'not exactly compatible' with "
            "'not martensite'. It separates CT variant/twin crystallography, "
            "exact A/M interface compatibility, orientation constructions, "
            "supercompatibility and experimental SMA functionality."
        ),
    )

    st.markdown(
        "## Overall CT answer"
    )

    _status_box(
        assessment.overall_status,
        assessment.overall_answer,
    )

    st.write(
        assessment.overall_reasoning
    )

    st.caption(
        "Scope — this is a Cayron-CT crystallographic assessment. "
        "It does not use Ball–James or PTMC to decide the CT conclusion, "
        "and it does not claim experimental proof of martensite kinetics, "
        "reversibility or shape-memory behavior."
    )

    st.info(
        "Notation discipline: "
        "C(M←A) = lattice correspondence; "
        "T/R = orientation relationship; "
        "F = deformation gradient/distortion; "
        "U = right stretch. "
        "These are distinct physical/mathematical objects "
        "and are not interchangeable."
    )

    st.divider()

    for index, step in enumerate(
        assessment.steps
    ):

        _render_step(step)

        if index != len(
            assessment.steps
        ) - 1:
            st.divider()

    _render_full_ct_audit(
        response
    )


def _render_conclusions_v10(
    project: Any,
    transformation_id: str,
    base_signature: str,
) -> None:

    # Keep the complete existing comparison workflow untouched so its
    # Calculate/update action executes first.
    #
    # Then render the CT-only subsection from the authoritative outputs
    # produced in that same Streamlit run.

    _original_conclusions(
        project,
        transformation_id,
        base_signature,
    )

    st.divider()

    _render_cayron_martensite_assessment(
        project,
        transformation_id,
        base_signature,
    )


_render_conclusions_v10._cayron_v10_original_conclusions = (  # type: ignore[attr-defined]
    _original_conclusions
)

v4._render_conclusions = (
    _render_conclusions_v10
)


def render_research_extension() -> None:
    v9.render_research_extension()