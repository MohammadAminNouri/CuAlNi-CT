from __future__ import annotations

"""Standalone Streamlit front end for the calculated twin-family tree."""

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


APP_TITLE = "Twin crystallography workbench"


def _style() -> None:
    st.markdown(
        """
        <style>
        html { scroll-behavior: auto !important; }
        *, *::before, *::after {
            animation-duration: 0s !important;
            animation-delay: 0s !important;
            transition-duration: 0s !important;
        }
        .block-container {
            max-width: 900px;
            padding-top: 1.7rem;
            padding-bottom: 4rem;
        }
        h1, h2, h3, h4, h5 { line-height: 1.3; }
        p, li { line-height: 1.6; }
        code { font-size: 0.95em; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 0.55rem;
        }
        div[data-testid="stMetric"] {
            padding: 0.15rem 0;
        }
        button { font-weight: 600 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _fingerprint(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _basis_payload(basis_result: Any) -> object:
    if basis_result.basis is None:
        return None
    matrix = basis_result.basis.P_conventional_from_primitive
    return [[str(matrix[i, j]) for j in range(3)] for i in range(3)]


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=None,
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    _style()

    st.title(APP_TITLE)
    st.markdown(
        "Enter one parent → product transformation. Editing the fields updates only the "
        "input and point-group preview; it does **not** run the twin-family or habit-plane solver."
    )
    st.markdown(
        "**How this page works**  \n"
        "1. Define parent A and product M.  \n"
        "2. Enter the correspondence and, only if needed, the weak-plane node basis.  \n"
        "3. Press **Calculate twin family and habit planes** once.  \n"
        "4. Read the result vertically: **Root → family → pair → twin branch → habit plane**."
    )

    st.markdown("## 1 · Transformation input")
    length_unit = st.selectbox(
        "Lattice-length unit",
        ("angstrom", "nanometer", "picometer", "micrometer", "meter"),
        index=0,
        help="Unit metadata are explicit; values are not silently rescaled.",
    )

    try:
        parent_ui = render_phase_input(
            prefix="parent",
            heading="Parent phase A",
            role="parent",
            length_unit=length_unit,
            default_family="cubic",
            default_point_group="m-3m",
            defaults=(1.0, 1.0, 1.0, 90.0, 90.0, 90.0),
        )
        st.divider()
        product_ui = render_phase_input(
            prefix="product",
            heading="Product phase M",
            role="product",
            length_unit=length_unit,
            default_family="monoclinic",
            default_point_group="2/m",
            defaults=(1.0, 1.0, 1.0, 90.0, 90.0, 90.0),
        )
        st.divider()
        correspondence_ui = render_correspondence_input()
        weak_basis_ui = render_weak_basis_input()

        transformation = TransformationInput.from_rows(
            "A_to_M",
            parent_ui.phase.phase_id,
            product_ui.phase.phase_id,
            correspondence_ui.canonical_rows,
            label="User-defined parent-to-product transformation",
        )
        request = CalculationRequest(
            project_id="twin_family_workbench",
            title="Twin family workbench state",
            parent=parent_ui.phase,
            product=product_ui.phase,
            transformation=transformation,
            notes="Standalone twin-family calculation; runtime outputs are input-derived.",
        )
        project_payload = request.to_project_payload()
    except ApplicationError as exc:
        st.info("Complete or correct the transformation input before calculation.")
        with st.expander("Input detail", expanded=False):
            st.code(str(exc), language="text")
        return
    except (ValueError, ArithmeticError, AssertionError) as exc:
        st.info("Complete or correct the transformation input before calculation.")
        with st.expander("Input detail", expanded=False):
            st.code(f"{type(exc).__name__}: {exc}", language="text")
        return

    state_signature = _fingerprint(
        {
            "project": project_payload,
            "weak_basis": _basis_payload(weak_basis_ui),
        }
    )
    stored_signature = st.session_state.get("twin_family_result_signature")
    stale = stored_signature is not None and stored_signature != state_signature
    if stale:
        st.info("Inputs changed. The previous calculation is hidden until you calculate this state.")

    calculate = st.button(
        "Calculate twin family and habit planes",
        type="secondary",
        use_container_width=True,
    )
    if calculate:
        st.session_state.pop("twin_family_report", None)
        st.session_state.pop("twin_family_result_signature", None)
        try:
            report = build_twin_family_report(
                project_payload,
                transformation.transformation_id,
                product_node_basis=weak_basis_ui.basis,
            )
        except ApplicationError as exc:
            st.error(str(exc))
            detail = getattr(exc, "detail", None)
            if detail:
                with st.expander("Technical detail", expanded=False):
                    st.code(str(detail), language="text")
        except (ValueError, ArithmeticError, AssertionError) as exc:
            st.error(str(exc))
            with st.expander("Technical detail", expanded=False):
                st.code(f"{type(exc).__name__}: {exc}", language="text")
        except Exception as exc:  # defensive UI boundary; never hides the traceback permanently
            st.error("The calculation did not complete. No partial scientific result is displayed.")
            with st.expander("Technical detail", expanded=False):
                st.code("".join(traceback.format_exception(exc)), language="text")
        else:
            st.session_state["twin_family_report"] = report
            st.session_state["twin_family_result_signature"] = state_signature

    report = st.session_state.get("twin_family_report")
    if report is not None and st.session_state.get("twin_family_result_signature") == state_signature:
        render_report(report)


if __name__ == "__main__":
    main()
