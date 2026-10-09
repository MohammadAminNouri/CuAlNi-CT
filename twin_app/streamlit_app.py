from __future__ import annotations

"""Interactive, low-clutter Streamlit entry point for the twin family workbench."""

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
    render_correspondence_input,
    render_phase_input,
    render_weak_basis_input,
)
from twin_app.scientific_engine import build_twin_family_report
from twin_app.tree_renderer import render_report
from twin_app.point_group_explainer import explain_group


APP_TITLE = "Twin-family tree"

# Example INPUTS only; none of Bhattacharya's twin/habit ANSWERS appear in
# production code. The same calculation works for any valid custom input.
NITI_PARENT = (3.015, 3.015, 3.015, 90., 90., 90.)
NITI_PRODUCT = (2.889, 4.120, 4.622, 90., 96.8, 90.)
NITI_CORRESPONDENCE = (
    ("0", "0", "1"),
    ("1/2", "1/2", "0"),
    ("-1/2", "1/2", "0"),
)


def _style() -> None:
    st.markdown("""<style>
    html { scroll-behavior: auto !important; }
    *, *::before, *::after {
        animation-duration: 0s !important;
        animation-delay: 0s !important;
        transition-duration: 0s !important;
    }
    .block-container { max-width: 1200px; padding-top: 1.2rem; padding-bottom: 3.5rem; }
    p, li { line-height: 1.5; }
    h1, h2, h3 { line-height: 1.25; }
    button { font-weight: 600 !important; }
    </style>""", unsafe_allow_html=True)


def _fingerprint(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _basis_payload(basis_result: Any) -> object:
    if basis_result.basis is None:
        return None
    matrix = basis_result.basis.P_conventional_from_primitive
    return [[str(matrix[i, j]) for j in range(3)] for i in range(3)]


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")
    _style()
    st.title(APP_TITLE)
    st.caption("Choose inputs → Calculate → select any twin couple in the tree → see its book-style results below.")

    state = st.radio(
        "Starting crystallographic state",
        ("NiTi book example · inputs only", "Custom transformation"),
        horizontal=True,
        help="NiTi loads published lattice/correspondence inputs, NOT twin/habit answers.",
        key="tree_input_state",
    )
    niti = state.startswith("NiTi")
    prefix = "niti" if niti else "custom"
    try:
        with st.expander("Crystal inputs and point groups", expanded=not niti):
            length_unit = st.selectbox(
                "Lattice-length unit", ("angstrom", "nanometer", "picometer", "micrometer", "meter"),
                key=f"{prefix}_length_unit",
            )
            parent_ui = render_phase_input(
                prefix=f"{prefix}_parent",
                heading="Parent A",
                role="parent",
                length_unit=length_unit,
                default_family="cubic",
                default_point_group="m-3m",
                defaults=NITI_PARENT if niti else (1., 1., 1., 90., 90., 90.),
            )
            product_ui = render_phase_input(
                prefix=f"{prefix}_product",
                heading="Product M",
                role="product",
                length_unit=length_unit,
                default_family="monoclinic",
                default_point_group="2/m",
                defaults=NITI_PRODUCT if niti else (1., 1., 1., 90., 90., 90.),
            )
            correspondence_ui = render_correspondence_input(
                prefix=f"{prefix}_correspondence",
                default_rows=NITI_CORRESPONDENCE if niti else (
                    ("1", "0", "0"), ("0", "1", "0"), ("0", "0", "1")
                ),
            )
            weak_basis_ui = render_weak_basis_input()

        transformation = TransformationInput.from_rows(
            "A_to_M", parent_ui.phase.phase_id, product_ui.phase.phase_id,
            correspondence_ui.canonical_rows,
            label="User-defined lattice correspondence",
        )
        request = CalculationRequest(
            project_id="twin_family_workbench",
            title="Twin-family tree input",
            parent=parent_ui.phase, product=product_ui.phase,
            transformation=transformation,
            notes="Calculated from crystallographic input only.",
        )
        project_payload = request.to_project_payload()
    except (ApplicationError, ValueError, ArithmeticError, AssertionError) as exc:
        st.error("Check the crystal inputs before calculating.")
        with st.expander("Input error details", expanded=False):
            st.code(f"{type(exc).__name__}: {exc}")
        return

    st.markdown("**Symmetry of the selected transformation**")
    for role, chosen in (("Parent A", parent_ui), ("Product M", product_ui)):
        inventory = chosen.inventory
        content = explain_group(
            inventory.symbol, dict(inventory.counts), inventory.expected_order,
            tuple(op.determinant for op in inventory.operations),
        )
        st.markdown(f"**{role} · {inventory.symbol}** — {content['signature']}")
        st.caption(
            f"{inventory.expected_order} operations · {content['proper']} proper · "
            f"{content['improper']} improper · inversion "
            f"{'yes' if content['inversion'] else 'no'}"
        )
    st.caption(
        "These symmetries enter the correspondence subgroup H. "
        "The twin tree below contains only results calculated after compatibility checks. "
        "Open crystal inputs to inspect exact axes and plane covectors."
    )
    signature = _fingerprint({
        "project": project_payload, "weak_basis": _basis_payload(weak_basis_ui),
    })
    previous = st.session_state.get("twin_family_result_signature")
    if previous is not None and previous != signature:
        st.info("Inputs changed: the previous calculation is hidden. Press Calculate again.")

    calculate = st.button(
        "Calculate twin family and habit planes",
        type="primary", use_container_width=True,
    )
    if calculate:
        st.session_state.pop("twin_family_report", None)
        st.session_state.pop("twin_family_result_signature", None)
        try:
            with st.spinner("Calculating symmetry families, rank-one twins and exact habit planes…"):
                report = build_twin_family_report(
                    project_payload,
                    transformation.transformation_id,
                    product_node_basis=weak_basis_ui.basis,
                )
        except ApplicationError as exc:
            st.error(str(exc))
        except (ValueError, ArithmeticError, AssertionError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error("No partial result was displayed because the calculation failed.")
            with st.expander("Technical error", expanded=False):
                st.code("".join(traceback.format_exception(exc)))
        else:
            # A new calculation must not inherit a stale couple/family choice
            # from a different crystallographic state. This enables the tree's
            # accurate "first family with exact habit solution" default.
            st.session_state.pop("twin_tree_family_view", None)
            st.session_state.pop("twin_selected_couple", None)
            st.session_state["twin_family_report"] = report
            st.session_state["twin_family_result_signature"] = signature

    report = st.session_state.get("twin_family_report")
    if report is not None and st.session_state.get("twin_family_result_signature") == signature:
        render_report(report)
    else:
        st.caption("The twin couples and A/M habit solutions will appear after calculation. No literature answers are inserted.")


if __name__ == "__main__":
    main()
