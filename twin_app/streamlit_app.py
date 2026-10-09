from __future__ import annotations

"""General, predictable entry point; no NiTi-vs-custom operating modes."""

import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

import streamlit as st

from app.application import CalculationRequest, TransformationInput
from app.errors import ApplicationError
from twin_app.input_ui import (
    render_correspondence_input, render_phase_input, render_weak_basis_input,
)
from twin_app.scientific_engine import build_twin_family_report
from twin_app.tree_renderer import render_report


APP_TITLE = "Martensitic Crystallography"

# Starting example INPUT only. The same inputs are fully editable; the solver
# never reads expected outputs from Bhattacharya's tables.
NITI_PARENT = (3.015, 3.015, 3.015, 90., 90., 90.)
NITI_PRODUCT = (2.889, 4.120, 4.622, 90., 96.8, 90.)
NITI_CORRESPONDENCE = (
    ("0", "0", "1"),
    ("1/2", "1/2", "0"),
    ("-1/2", "1/2", "0"),
)


def _style() -> None:
    st.markdown("""<style>
    :root { --tf-blue:#5C85AD; --tf-teal:#458C81; --tf-amber:#B68C4F;
            --tf-grey:#788491; }
    html { scroll-behavior: auto !important; }
    *, *::before, *::after {
      animation-duration: 0s !important; animation-delay: 0s !important;
      transition-duration: 0s !important;
    }
    .block-container { max-width: 1120px; padding-top: 1.7rem; padding-bottom: 3rem; }
    p, li { line-height: 1.65; font-size: 1.06rem; }
    label { font-size: 1.04rem !important; }
    [data-testid="stCaptionContainer"] p { font-size: 0.97rem; line-height:1.55; }
    div[data-baseweb="select"] { min-height: 48px; }
    [data-testid="stExpander"] { margin-top: 0.7rem; }
    .tf-alert-blue,.tf-alert-teal,.tf-alert-amber { padding: .85rem 1rem; border-left: 5px solid; border-radius: 5px; margin: .85rem 0 1rem; line-height: 1.6; }
    .tf-alert-blue { border-color: var(--tf-blue); background: rgba(92,133,173,.11); }
    .tf-alert-teal { border-color: var(--tf-teal); background: rgba(69,140,129,.11); }
    .tf-alert-amber { border-color: var(--tf-amber); background: rgba(182,140,79,.11); }
    h1 { font-size: 1.9rem !important; line-height:1.25; }
    h2 { font-size: 1.35rem !important; line-height:1.3; }
    h3,h4 { line-height:1.3; }
    button { font-weight: 600 !important; }
    .tf-inline { padding: .72rem .95rem; border: 1px solid rgba(128,140,154,.26);
       border-left-width: 5px; border-radius: .45rem; margin: .7rem 0 .9rem 0;
       font-weight: 600; line-height: 1.4; }
    .tf-selection { border-left-color:var(--tf-blue); }
    .tf-habit { border-left-color:var(--tf-teal); }
    .tf-uncertain { border-left-color:var(--tf-amber); }
    .tf-legend { display:flex; flex-wrap:wrap; gap:.45rem 1.05rem;
      margin:.2rem 0 1rem; font-size:.9rem; }
    .tf-dot { display:inline-block; vertical-align:middle; width:.68rem; height:.68rem;
      border-radius:100%; margin-right:.3rem; }
    .tf-blue { background:var(--tf-blue); }
    .tf-teal { background:var(--tf-teal); }
    .tf-grey { background:var(--tf-grey); }
    [data-testid="stCaptionContainer"] p { line-height: 1.45; }
    @media (prefers-reduced-motion: reduce) {
       *, *::before, *::after { animation:none !important; transition:none !important; }
    }
    </style>""", unsafe_allow_html=True)


def _fingerprint(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _basis_payload(basis_result: Any) -> object:
    if basis_result.basis is None:
        return None
    matrix = basis_result.basis.P_conventional_from_primitive
    return [[str(matrix[i, j]) for j in range(3)] for i in range(3)]


def _load_niti_example() -> None:
    """Resets input fields only; never supplies published twin/habit results."""
    for role, family, group, lattice in (
        ("parent", "cubic", "m-3m", NITI_PARENT),
        ("product", "monoclinic", "2/m", NITI_PRODUCT),
    ):
        st.session_state[f"{role}_family"] = family
        st.session_state[f"{role}_point_group_{family}"] = group
        for label, value in zip(("a", "b", "c", "alpha", "beta", "gamma"), lattice):
            st.session_state[f"{role}_{label}"] = value
    st.session_state["correspondence_direction_choice"] = "A → M"
    for i, row in enumerate(NITI_CORRESPONDENCE):
        for j, value in enumerate(row):
            st.session_state[f"correspondence_A_TO_M_{i}_{j}"] = value
    st.session_state["twin_length_unit"] = "angstrom"
    for key in (
        "twin_family_report", "twin_family_result_signature",
        "twin_tree_family_view", "twin_selected_couple",
    ):
        st.session_state.pop(key, None)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")
    _style()
    st.title(APP_TITLE)
    st.markdown("**Twin families and habit-plane compatibility**")
    st.caption("Research workbench · exact crystal symmetry, correspondence and laminate compatibility")
    st.markdown("**Crystal definitions** → **Calculate** → **Explore all families** → **Inspect one couple**")

    existing = st.session_state.get("twin_family_report")
    st.markdown("### Crystal state")
    st.caption("The editable fields use the published NiTi lattice and correspondence as a starting example, not as fixed material answers.")
    with st.expander("View or change parent, product and correspondence", expanded=existing is None):
        st.button("Restore published NiTi crystal inputs", on_click=_load_niti_example, help="Restores the input lattice and correspondence only. It does not load calculated or published twin results.")

        try:
            length_unit = st.selectbox(
                "Lattice length unit",
                ("angstrom", "nanometer", "picometer", "micrometer", "meter"),
                key="twin_length_unit",
            )
            parent_ui = render_phase_input(
                prefix="parent", heading="Parent crystal A", role="parent", length_unit=length_unit,
                default_family="cubic", default_point_group="m-3m", defaults=NITI_PARENT,
            )
            st.divider()
            product_ui = render_phase_input(
                prefix="product", heading="Product crystal M", role="product", length_unit=length_unit,
                default_family="monoclinic", default_point_group="2/m", defaults=NITI_PRODUCT,
            )
            st.divider()
            correspondence_ui = render_correspondence_input(
                prefix="correspondence", default_rows=NITI_CORRESPONDENCE,
            )
            weak_basis_ui = render_weak_basis_input()

            transformation = TransformationInput.from_rows(
                "A_to_M", parent_ui.phase.phase_id, product_ui.phase.phase_id,
                correspondence_ui.canonical_rows, label="Entered lattice correspondence",
            )
            request = CalculationRequest(
                project_id="twin_family_workbench", title="Crystal twin transformation",
                parent=parent_ui.phase, product=product_ui.phase, transformation=transformation,
                notes="Crystallography derived only from entered parent/product metrics and correspondence.",
            )
            payload = request.to_project_payload()
        except (ApplicationError, ValueError, ArithmeticError, AssertionError) as exc:
            st.warning("One input is incomplete or incompatible. Fix it before calculation.")
            with st.expander("Which input failed?", expanded=False):
                st.code(f"{type(exc).__name__}: {exc}")
            return

    signature = _fingerprint({"project": payload, "weak_basis": _basis_payload(weak_basis_ui)})
    previous = st.session_state.get("twin_family_result_signature")
    if previous is not None and previous != signature:
        st.info("Your inputs changed. The previous calculation is hidden until you calculate again.")

    if st.button("Calculate twin families and habit planes", type="primary", use_container_width=True):
        st.session_state.pop("twin_family_report", None)
        st.session_state.pop("twin_family_result_signature", None)
        try:
            with st.spinner("Calculating exact symmetry and compatible twin interfaces…"):
                report = build_twin_family_report(
                    payload, transformation.transformation_id,
                    product_node_basis=weak_basis_ui.basis,
                )
        except (ApplicationError, ValueError, ArithmeticError, AssertionError) as exc:
            st.error(f"Calculation could not finish: {exc}")
        except Exception as exc:
            st.error("Calculation failed. No partial or invented result will be shown.")
            with st.expander("Technical error", expanded=False):
                st.code("".join(traceback.format_exception(exc)))
        else:
            st.session_state.pop("twin_tree_family_view", None)
            st.session_state.pop("twin_selected_couple", None)
            st.session_state.pop("tf_selected_couple_v6", None)
            st.session_state["twin_family_report"] = report
            st.session_state["twin_family_result_signature"] = signature

    report = st.session_state.get("twin_family_report")
    if report is not None and st.session_state.get("twin_family_result_signature") == signature:
        render_report(report)
    else:
        st.caption("The calculated tree will appear below. All families will be shown together; published numerical results are never substituted for calculations.")


if __name__ == "__main__":
    main()
