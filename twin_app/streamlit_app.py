from __future__ import annotations

"""General, predictable entry point; no NiTi-vs-custom operating modes."""

from dataclasses import asdict
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
from cualni_cryst.numerics import DEFAULT_NUMERICAL_POLICY
from twin_app.input_ui import (
    render_correspondence_input, render_phase_input, render_weak_basis_input,
)
from twin_app.scientific_engine import build_twin_family_report
from twin_app.tree_renderer import render_report
from twin_app.report_export import make_export


APP_TITLE = "Martensitic Crystallography · Research Workbench"

# Starting example INPUT only. The same inputs are fully editable; the solver
# never reads expected outputs from Bhattacharya's tables.
NITI_PARENT = (3.015, 3.015, 3.015, 90., 90., 90.)
NITI_PRODUCT = (2.889, 4.120, 4.622, 90., 96.8, 90.)
NITI_CORRESPONDENCE = (
    ("0", "0", "1"),
    ("1/2", "1/2", "0"),
    ("-1/2", "1/2", "0"),
)


def _style(*, appearance: str = "dark", font_scale: str = "standard") -> None:
    base_background = {"dark": "#101922", "light": "#fff", "contrast": "#050505"}[appearance]
    text_color = {"dark": "#f2f5f8", "light": "#1b3240", "contrast": "#fff"}[appearance]
    fontsize = "1.12rem" if font_scale == "large" else "1rem"
    st.markdown("""<style>
    :root { --tf-blue:#5C85AD; --tf-teal:#458C81; --tf-amber:#B68C4F;
            --tf-grey:#788491; }
    html { scroll-behavior: auto !important; }
    *, *::before, *::after {
      animation-duration: 0s !important; animation-delay: 0s !important;
      transition-duration: 0s !important;
    }
    .block-container { max-width: 1320px; padding-top: 1.25rem; padding-bottom: 3rem; }
    p, li { line-height: 1.65; font-size: 1.06rem; }
    label { font-size: 1.04rem !important; }
    [data-testid="stCaptionContainer"] p { font-size: 0.97rem; line-height:1.55; }
    div[data-baseweb="select"] { min-height: 48px; }
    [data-testid="stExpander"] { margin-top: 0.7rem; }
    .tf-alert-blue,.tf-alert-teal,.tf-alert-amber { padding: .85rem 1rem; border-left: 5px solid; border-radius: 5px; margin: .85rem 0 1rem; line-height: 1.6; }
    .tf-alert-blue { border-color: var(--tf-blue); background: rgba(92,133,173,.11); }
    .tf-alert-teal { border-color: var(--tf-teal); background: rgba(69,140,129,.11); }
    .tf-alert-amber { border-color: var(--tf-amber); background: rgba(182,140,79,.11); }
    h1 { font-size: 2.1rem !important; line-height:1.25; font-weight:750 !important; }
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
    .tf-pane-heading { color:#8fb4d3; font-size:.81rem; font-weight:800; letter-spacing:.075em; margin-top:1rem; }
    .tf-quantity { border:1px solid rgba(124,146,163,.38); border-radius:11px; padding:15px 15px 14px; margin:8px 0 13px; background:rgba(115,136,152,.045); min-height:124px; }
    .tf-quantity-label {font-weight:700; font-size:1rem; margin-bottom:10px;}
    .tf-quantity-label span {font-weight:500; opacity:.8;}
    .tf-quantity-value {font-family:ui-monospace,Consolas,monospace; font-size:1.17rem; word-break:break-word; line-height:1.55; font-weight:600;}
    .tf-quantity-help {font-size:.91rem; opacity:.83; margin-top:9px; line-height:1.4;}
    .tf-banner {border:1px solid rgba(124,146,163,.4); border-left:5px solid #8592a0; border-radius:10px; padding:13px 16px; margin:16px 0; line-height:1.65;}
    .tf-banner-blue {border-left-color:#85b4e3;background:rgba(111,156,201,.11);}
    .tf-banner-amber {border-left-color:#e3bd81;background:rgba(227,189,129,.12);}
    .tf-banner-exact {border-left-color:#75bfb5;background:rgba(117,191,181,.11);}
    .tf-banner-neutral {border-left-color:#8995a2;background:rgba(125,145,160,.08);}
    [data-testid="stWidgetLabel"] {font-size:1rem;font-weight:650;}
    [data-testid="stExpander"] summary { min-height:48px; align-items:center; }
    </style>""" + f"<style>:root{{--tf-fsize:{fontsize}}} .stApp{{background:{base_background};color:{text_color}}} .stApp p, .stApp label{{font-size:{fontsize}}}</style>", unsafe_allow_html=True)


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
        "tf_selected_couple_v8", "tf_text_finder_v8",
    ):
        st.session_state.pop(key, None)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")
    with st.sidebar:
        st.subheader("Display preferences")
        appearance = st.radio("Appearance", ("Dark", "Light", "High contrast"), horizontal=False, key="tf_appearance_v8", help="Choose what is easiest to read. This never changes the scientific result.")
        font_scale = st.radio("Text size", ("Standard", "Large"), key="tf_font_scale_v8", horizontal=True)
        density = st.radio("Tree spacing", ("Comfortable", "Compact"), key="tf_density_v8", horizontal=True)
        st.caption("Display choices are not scientific inputs; no calculation is restarted when you change them.")
    theme = {"Dark": "dark", "Light": "light", "High contrast": "contrast"}[appearance]
    _style(appearance=theme, font_scale=font_scale.lower())
    st.title("Martensitic Crystallography")
    st.markdown("**A research workbench for twins, symmetry and habit-plane compatibility**")
    st.caption("Define the two crystals → Calculate → Explore the genealogy → Inspect verified results")

    existing = st.session_state.get("twin_family_report")
    st.subheader("Crystallographic inputs")
    st.caption("The editable NiTi demonstration contains crystal parameters only. All twin and habit results are computed from the entered parameters.")
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
            st.session_state.pop("tf_selected_couple_v8", None)
            st.session_state.pop("tf_text_finder_v8", None)
            st.session_state["twin_family_report"] = report
            st.session_state["twin_family_result_signature"] = signature

    report = st.session_state.get("twin_family_report")
    if report is not None and st.session_state.get("twin_family_result_signature") == signature:
        render_report(report, theme=theme, density=density.lower())
        with st.expander("Export reproducible scientific report", expanded=False):
            st.caption("Includes your input, coordinate conventions, every calculated variant, family and habit solution, and numerical residuals. Published benchmark outputs are not inserted.")
            record = make_export(report, input_payload=payload, policy=asdict(DEFAULT_NUMERICAL_POLICY))
            st.download_button(
                "Download complete analysis as JSON",
                data=json.dumps(record, indent=2, default=str),
                file_name="twin_family_research_record.json",
                mime="application/json",
                key="tf_export_v8",
            )
    else:
        st.caption("Calculate to create the full twin-family genealogy. The solver does not use published result tables as answers.")


if __name__ == "__main__":
    main()
