from __future__ import annotations

"""Standalone AR · Gen · Cry Streamlit application.

Deploy THIS entry point, not app/streamlit_app.py. There are no imported twin
application modules, no accidental legacy pages, and no executable old GUI.
"""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from cualni_cryst.legacy_suite_bridge import Phase
from cualni_cryst.ar_gen_cry_examples import (
    EXAMPLES, NI_TI_REFERENCE_PAIRS, provenance, parse_cif_phase,
    correspondence_from_pairs, invert_correspondence,
)
from cualni_cryst.ar_gen_cry_pipeline import (
    ar_parent_fit, ar_stage, cry_stage, gen_stage, report_json,
)

STEPS = ("1 · Cry", "2 · Gen", "3 · AR", "4 · Report")
POINT_GROUPS = ("m-3m", "mmm", "4/mmm", "6/mmm", "2/m", "222", "2", "-1", "1")

st.set_page_config(page_title="AR · Gen · Cry", page_icon="🔬", layout="wide",
                   initial_sidebar_state="expanded")

DEMO_ANG = (
    b"# GRID: SqrGrid\n# XSTEP: 1\n# YSTEP: 1\n"
    b"0 0 0 0 0 100 0.99 1 0 0\n"
    b"0 0 0 1 0 100 0.99 1 0 0\n"
    b"0 0 0 0 1 100 0.99 1 0 0\n"
)

if "ar_gen_cry_step" not in st.session_state:
    st.session_state["ar_gen_cry_step"] = STEPS[0]


def go(step: str) -> None:
    if step not in STEPS:
        raise ValueError("Unknown workflow step")
    st.session_state["ar_gen_cry_step"] = step


def clear_from_cry() -> None:
    for key in ("cry_result", "gen_result", "ar_result", "ar_fit_result", "ar_bytes", "ar_sha256"):
        st.session_state.pop(key, None)


def clear_from_gen() -> None:
    for key in ("gen_result", "ar_fit_result"):
        st.session_state.pop(key, None)


def parse_triplet(text: str, label: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(x) for x in text.replace(",", " ").split())
    except ValueError as exc:
        raise ValueError(f"{label}: enter three numbers separated by spaces") from exc
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError(f"{label}: exactly three finite numbers required")
    return values


def phase_form(prefix: str, base: Phase, *, editable: bool = True, scope: str = "custom") -> Phase:
    if not editable:
        st.markdown(f"**{prefix}: {base.name}**")
        st.caption(
            f"{base.point_group}  ·  a {base.a:g}  ·  b {base.b:g}  ·  c {base.c:g} Å"
            f"  ·  α {base.alpha:g}°  β {base.beta:g}°  γ {base.gamma:g}°"
        )
        return base
    st.markdown(f"#### {prefix}")
    name = st.text_input(f"{prefix} label", value=base.name, key=f"{scope}_{prefix}_name")
    group = st.selectbox(f"{prefix} symmetry", POINT_GROUPS,
                         index=POINT_GROUPS.index(base.point_group),
                         key=f"{scope}_{prefix}_point_group")
    a, b, c = st.columns(3)
    values = (
        a.number_input(f"{prefix} a (Å)", min_value=0.001, value=float(base.a),
                       format="%.5f", key=f"{scope}_{prefix}_a"),
        b.number_input(f"{prefix} b (Å)", min_value=0.001, value=float(base.b),
                       format="%.5f", key=f"{scope}_{prefix}_b"),
        c.number_input(f"{prefix} c (Å)", min_value=0.001, value=float(base.c),
                       format="%.5f", key=f"{scope}_{prefix}_c"),
    )
    with st.expander(f"{prefix} cell angles", expanded=False):
        x, y, z = st.columns(3)
        angles = (
            x.number_input("α (°)", min_value=0.001, max_value=179.999,
                           value=float(base.alpha), key=f"{scope}_{prefix}_alpha"),
            y.number_input("β (°)", min_value=0.001, max_value=179.999,
                           value=float(base.beta), key=f"{scope}_{prefix}_beta"),
            z.number_input("γ (°)", min_value=0.001, max_value=179.999,
                           value=float(base.gamma), key=f"{scope}_{prefix}_gamma"),
        )
    return Phase(name, group, *values, *angles)


def summary_status() -> None:
    st.sidebar.markdown("### Progress")
    for step, key in (("Cry", "cry_result"), ("Gen", "gen_result"),
                      ("AR", "ar_result")):
        done = key in st.session_state
        st.sidebar.caption(f"{'✓' if done else '○'}  {step} — {'calculated' if done else 'not run'}")
    st.sidebar.caption("AR can be skipped when no EBSD data are available.")


st.title("AR · Gen · Cry")
st.caption("Crystal geometry → variants and operators → EBSD. Four steps, one workspace.")
st.sidebar.radio("Workflow", STEPS, key="ar_gen_cry_step")
summary_status()
st.sidebar.progress(sum(k in st.session_state for k in ("cry_result", "gen_result", "ar_result")) / 3)
step = st.session_state["ar_gen_cry_step"]
cry = st.session_state.get("cry_result")
gen = st.session_state.get("gen_result")
ar = st.session_state.get("ar_result")
fit = st.session_state.get("ar_fit_result")

if step == STEPS[0]:
    st.subheader("1. Cry · Define the two crystals")
    st.write("Enter measured lattice parameters and point-group symmetry. "
             "All dimensions are in ångströms; all angles are in degrees.")
    st.info("Initial values are synthetic demonstration cells, NOT validated Cu–Al–Ni 2H or 18R data.")
    if cry:
        saved_p = Phase(**cry["parent_phase"])
        saved_m = Phase(**cry["product_phase"])
    else:
        saved_p = Phase("Illustrative parent", "m-3m", 5.8, 5.8, 5.8)
        saved_m = Phase("Illustrative product", "mmm", 4.3, 5.3, 4.5)
    with st.form("cry_form", clear_on_submit=False):
        left, right = st.columns(2, gap="large")
        with left:
            p = phase_form("Parent", saved_p)
        with right:
            m = phase_form("Product", saved_m)
        u = st.text_input("Direction [uvw]", value="1 0 0")
        h = st.text_input("Plane (hkl)", value="1 0 0")
        run = st.form_submit_button("Run Cry", type="primary", use_container_width=True)
    if run:
        try:
            result = cry_stage(p, m, parse_triplet(u, "Direction"), parse_triplet(h, "Plane"))
        except (ValueError, ArithmeticError, AssertionError, TypeError) as exc:
            st.error(f"Cry could not validate the crystal model: {exc}")
        else:
            st.session_state["cry_result"] = result
            for key in ("gen_result", "ar_result", "ar_fit_result", "ar_bytes",
                        "ar_sha256"):
                st.session_state.pop(key, None)
            cry, gen, ar, fit = result, None, None, None
            st.success("Both crystal models passed metric/symmetry validation.")
    if cry:
        a, b = st.columns(2)
        a.metric("Parent d-spacing (Å)", f"{cry['parent']['d_spacing_angstrom']:.5g}")
        b.metric("Product d-spacing (Å)", f"{cry['product']['d_spacing_angstrom']:.5g}")
        with st.expander("Inspect both metrics", expanded=False):
            st.json({"parent": cry["parent"], "product": cry["product"]})
        st.button("Continue to Gen", on_click=go, args=(STEPS[1],),
                  type="primary", use_container_width=True)

elif step == STEPS[1]:
    st.subheader("2. Gen · Calculate variants and operators")
    if not cry:
        st.warning("Run Cry first. Gen needs the validated parent/product cell and symmetry.")
        st.button("Go to Cry", on_click=go, args=(STEPS[0],))
        st.stop()
    st.caption(f"{cry['parent_phase']['name']} → {cry['product_phase']['name']}")
    st.write("Enter the exact 3 × 3 lattice correspondence **C (M ← A)**. "
             "Integers and fractions are accepted.")
    st.warning("C is NOT an experimental orientation relationship. "
               "The identity default is an illustrative input, not a Cu–Al–Ni assignment.")
    with st.form("gen_form"):
        c_text = st.text_area("Correspondence matrix (one row per line)",
                              value="1 0 0\n0 1 0\n0 0 1", height=120)
        run = st.form_submit_button("Run Gen", type="primary",
                                    use_container_width=True)
    if run:
        try:
            result = gen_stage(cry, c_text)
        except (ValueError, ArithmeticError, AssertionError, TypeError) as exc:
            st.error(f"Gen calculation stopped: {exc}")
        else:
            st.session_state["gen_result"] = result
            st.session_state.pop("ar_result", None)
            st.session_state.pop("ar_fit_result", None)
            gen, ar, fit = result, None, None
            st.success("Exact cosets and operator consistency checks passed.")
    if gen:
        x, y, z = st.columns(3)
        x.metric("Variants", gen["variants"])
        y.metric("Operator classes", gen["operator_classes"])
        z.metric("Subgroup order", gen["subgroup_order"])
        with st.expander("Operator adjacency table"):
            st.dataframe(pd.DataFrame(gen["adjacency"]), use_container_width=True)
        with st.expander("Exact C and audit"):
            st.json({k: v for k, v in gen.items() if k != "adjacency"})
        st.button("Continue to AR", on_click=go, args=(STEPS[2],),
                  type="primary", use_container_width=True)

elif step == STEPS[2]:
    st.subheader("3. AR · Inspect EBSD data")
    st.write("Upload Oxford CTF or EDAX ANG orientations to inspect phase counts. "
             "An experimental parent fit is optional and requires a verified orientation relationship.")
    if not cry:
        st.warning("Run Cry first to establish the parent and product crystal models.")
        st.button("Go to Cry", on_click=go, args=(STEPS[0],))
        st.stop()
    if not gen:
        st.info("Gen has not been run. You may inspect EBSD independently; "
                "the complete chain will remain marked incomplete.")
    uploaded = st.file_uploader("EBSD map (.ctf or .ang), maximum 8 MiB",
                                type=["ctf", "ang"], key="ar_upload")
    if uploaded is not None:
        current_sha = hashlib.sha256(uploaded.getvalue()).hexdigest()
        previous_sha = st.session_state.get("ar_sha256")
        if previous_sha is not None and current_sha != previous_sha:
            for stale_key in ("ar_result", "ar_fit_result", "ar_bytes", "ar_sha256"):
                st.session_state.pop(stale_key, None)
            ar = None
            fit = None
            st.warning("EBSD upload changed. Old AR results were cleared; inspect the new file.")
        st.caption(f"Selected: {uploaded.name} ({uploaded.size:,} bytes)")
        if st.button("Inspect EBSD", type="primary", use_container_width=True):
            try:
                blob = uploaded.getvalue()
                result = ar_stage(uploaded.name, blob)
            except (ValueError, ArithmeticError, UnicodeError, OSError) as exc:
                st.error(f"AR import failed: {exc}")
            else:
                st.session_state["ar_result"] = result
                st.session_state["ar_bytes"] = blob
                st.session_state["ar_sha256"] = hashlib.sha256(blob).hexdigest()
                st.session_state.pop("ar_fit_result", None)
                ar, fit = result, None
                st.success("EBSD orientations imported. The reference frame is not inferred.")
    if ar:
        if uploaded is not None and hashlib.sha256(uploaded.getvalue()).hexdigest() != st.session_state.get("ar_sha256"):
            st.warning("File changed. Inspect the new upload before attempting a parent fit.")
        st.json(ar)
        with st.expander("Optional experimental parent fit", expanded=False):
            st.caption("Exploratory single-parent fit, not a complete segmented parent-grain map.")
            chosen = st.number_input("Indexed product phase ID", min_value=1,
                                     step=1, value=1)
            supplied_or = st.text_area(
                "Verified R(parent ← product), JSON list of 3×3 Cartesian rotations",
                placeholder='[[[1,0,0],[0,1,0],[0,0,1]]]', value=""
            )
            n = st.slider("Maximum sampled points", 2, 48, 12)
            verified = st.checkbox("I independently verified the OR, phase identity and coordinate frames.")
            if st.button("Fit parent", disabled=not verified, type="primary"):
                if uploaded is not None and hashlib.sha256(uploaded.getvalue()).hexdigest() != st.session_state.get("ar_sha256"):
                    st.error("Selected upload has changed. Inspect it again.")
                else:
                    try:
                        result = ar_parent_fit(
                            ar["filename"], st.session_state["ar_bytes"], cry,
                            int(chosen), supplied_or, max_points=n
                        )
                    except (ValueError, TypeError, ArithmeticError, AssertionError) as exc:
                        st.error(f"Parent fit rejected: {exc}")
                    else:
                        st.session_state["ar_fit_result"] = result
                        fit = result
                        st.success("Exploratory parent fit calculated from supplied ORs.")
        if fit:
            with st.expander("Exploratory parent orientation and residuals", expanded=True):
                st.json(fit)
    if not ar:
        st.caption("No EBSD yet. The report can still document Cry and Gen.")
    st.button("Continue to Report", on_click=go, args=(STEPS[3],),
              use_container_width=True)

else:
    st.subheader("4. Report · Export what was actually calculated")
    if not cry:
        st.warning("Cry has not been calculated.")
        st.button("Go to Cry", on_click=go, args=(STEPS[0],))
        st.stop()
    st.markdown(f"**Cry:** complete · **Gen:** {'complete' if gen else 'not run'} · "
                f"**AR:** {'complete' if ar else 'not run'}")
    if fit:
        st.caption("An exploratory experimental parent fit is included, with its limitations.")
    st.info("Uncalculated stages appear as null in the report. "
            "Experimental ORs are never generated from C alone.")
    try:
        payload = report_json(cry, gen, ar, fit)
    except (ValueError, TypeError) as exc:
        st.error(f"Cannot export results: {exc}")
    else:
        st.download_button("Download reproducible JSON report", data=payload,
                           file_name="ar_gen_cry_results.json", mime="application/json",
                           type="primary", use_container_width=True)
        with st.expander("Preview results"):
            st.json(json.loads(payload))
    st.button("Back to Cry", on_click=go, args=(STEPS[0],))
