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


def clear_from_ar() -> None:
    for key in ("ar_result", "ar_fit_result", "ar_bytes", "ar_sha256"):
        st.session_state.pop(key, None)


class _SyntheticEBSD:
    name = "synthetic_demo.ang"
    size = len(DEMO_ANG)

    def getvalue(self) -> bytes:
        return DEMO_ANG


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
    st.markdown(":blue[**01 / CRY**]")
    st.subheader("Crystal setup")
    st.caption("Define parent and product cells. Lengths in Å, angles in degrees.")
    mode = st.radio(
        "Input source", ("Archive example", "Manual", "CIF files"),
        horizontal=True, key="cry_input_mode", on_change=clear_from_cry,
    )

    archive_reference = None
    if mode == "Archive example":
        name = st.selectbox(
            "Choose a material", tuple(EXAMPLES),
            key="cry_example", on_change=clear_from_cry,
        )
        example = EXAMPLES[name]
        st.caption(example.description + " · Original 2021 phase records")
        edit = st.checkbox(
            "Edit example parameters", key="cry_edit_example", on_change=clear_from_cry,
        )
        saved_p, saved_m = example.parent, example.product
        archive_reference = provenance(name, edited=edit)
        scope = "archive_" + str(list(EXAMPLES).index(name))
    elif mode == "CIF files":
        st.caption("Upload two CIFs. Cell parameters are read from each file; select point groups yourself.")
        col_a, col_b = st.columns(2)
        with col_a:
            file_p = st.file_uploader(
                "Parent CIF", type=["cif"], key="cry_parent_cif", on_change=clear_from_cry,
            )
            group_p = st.selectbox(
                "Parent point group", POINT_GROUPS, key="cry_parent_cif_group", on_change=clear_from_cry,
            )
        with col_b:
            file_m = st.file_uploader(
                "Product CIF", type=["cif"], key="cry_product_cif", on_change=clear_from_cry,
            )
            group_m = st.selectbox(
                "Product point group", POINT_GROUPS, key="cry_product_cif_group", on_change=clear_from_cry,
            )
        if file_p is None or file_m is None:
            st.info("Add both CIF files to continue.")
            st.stop()
        try:
            saved_p = parse_cif_phase(file_p.getvalue(), Path(file_p.name).stem, group_p)
            saved_m = parse_cif_phase(file_m.getvalue(), Path(file_m.name).stem, group_m)
        except (ValueError, TypeError) as exc:
            st.error(f"CIF input rejected: {exc}")
            st.stop()
        archive_reference = {
            "kind": "uploaded_cif",
            "parent_sha256": hashlib.sha256(file_p.getvalue()).hexdigest(),
            "product_sha256": hashlib.sha256(file_m.getvalue()).hexdigest(),
            "point_group_note": "user-selected, not automatically inferred",
        }
        edit = False
        scope = "cif_readonly"
    else:
        saved_p = Phase("Parent", "m-3m", 3, 3, 3)
        saved_m = Phase("Product", "2/m", 2.9, 4.1, 4.6, 90, 97, 90)
        edit = True
        scope = "manual"

    with st.form("cry_form", clear_on_submit=False):
        left, right = st.columns(2, gap="large")
        with left:
            p = phase_form("Parent", saved_p, editable=edit, scope=scope)
        with right:
            m = phase_form("Product", saved_m, editable=edit, scope=scope)
        with st.expander("Geometry probe · optional", expanded=False):
            u = st.text_input("Direction [uvw]", value="1 0 0")
            h = st.text_input("Plane (hkl)", value="1 0 0")
        run = st.form_submit_button("Run Cry", type="primary", use_container_width=True)
    if run:
        try:
            result = cry_stage(p, m, parse_triplet(u, "Direction"), parse_triplet(h, "Plane"))
        except (ValueError, ArithmeticError, AssertionError, TypeError) as exc:
            st.error(f"Crystal model rejected: {exc}")
        else:
            clear_from_cry()
            result["input_source"] = archive_reference or {
                "kind": "manual",
                "note": "Cell parameters and symmetry are user-supplied",
            }
            st.session_state["cry_result"] = result
            cry, gen, ar, fit = result, None, None, None
            st.success("Cell metrics and point-group compatibility checked.")
    if cry:
        a, b = st.columns(2)
        a.metric("Parent d-spacing (Å)", f"{cry['parent']['d_spacing_angstrom']:.5g}")
        b.metric("Product d-spacing (Å)", f"{cry['product']['d_spacing_angstrom']:.5g}")
        with st.expander("Metrics and provenance"):
            st.json({
                "parent": cry["parent"], "product": cry["product"],
                "input_source": cry.get("input_source", {}),
            })
        st.button(
            "Continue to Gen", on_click=go, args=(STEPS[1],),
            type="primary", use_container_width=True,
        )

elif step == STEPS[1]:
    st.markdown(":violet[**02 / GEN**]")
    st.subheader("Variants and operators")
    if not cry:
        st.warning("Run Cry first. Gen needs the validated parent/product cell and symmetry.")
        st.button("Go to Cry", on_click=go, args=(STEPS[0],))
        st.stop()
    st.caption(f"{cry['parent_phase']['name']}  →  {cry['product_phase']['name']}")
    input_mode = st.radio(
        "Correspondence input", ("3×3 matrix", "Three direction pairs"),
        horizontal=True, key="gen_input_mode", on_change=clear_from_gen,
    )
    if cry.get("input_source", {}).get("example") == "NiTi · B2 → B19′":
        with st.expander("NiTi archive · reference entries"):
            st.caption("Two paired entries were found in the NiTi transition file. "
                       "Their coordinate conventions need verification, and they do not "
                       "determine a unique 3D correspondence. They are not applied automatically.")
            for first, second in NI_TI_REFERENCE_PAIRS:
                st.code(f"{list(first)}  |  {list(second)}", language=None)
    with st.form("gen_form"):
        if input_mode == "3×3 matrix":
            st.caption("C maps direct directions from parent A into product M. Exact fractions are accepted.")
            c_text = st.text_area(
                "C (M ← A), one row per line",
                value="1 0 0\n0 1 0\n0 0 1", height=110,
            )
            reverse = st.checkbox("Matrix entered as A ← M · invert before calculation")
        else:
            st.caption("Enter three corresponding, linearly independent direction pairs. "
                       "Each line is one [u v w] vector in the specified crystal basis.")
            col1, col2 = st.columns(2)
            with col1:
                parent_rows = st.text_area(
                    "Parent directions", "1 0 0\n0 1 0\n0 0 1", height=110
                )
            with col2:
                product_rows = st.text_area(
                    "Product directions", "1 0 0\n0 1 0\n0 0 1", height=110
                )
            reverse = False
        st.caption("The default identity mapping is a software demonstration, not a NiTi OR.")
        run = st.form_submit_button("Run Gen", type="primary", use_container_width=True)
    if run:
        try:
            matrix = (
                invert_correspondence(c_text) if reverse
                else c_text
            ) if input_mode == "3×3 matrix" else correspondence_from_pairs(
                parent_rows, product_rows
            )
            result = gen_stage(cry, matrix)
        except (ValueError, ArithmeticError, AssertionError, TypeError, ZeroDivisionError) as exc:
            st.error(f"Correspondence input rejected: {exc}")
        else:
            clear_from_gen()
            result["input_method"] = input_mode
            result["resolved_matrix_M_from_A"] = matrix
            result["demo_only"] = (
                matrix.strip().replace(" ", "") == "100\n010\n001"
            )
            st.session_state["gen_result"] = result
            gen, fit = result, None
            st.success("Cosets, operators and consistency checks passed.")
    if gen:
        a, b, c = st.columns(3)
        a.metric("Variants", gen["variants"])
        b.metric("Operators", gen["operator_classes"])
        c.metric("Subgroup |H|", gen["subgroup_order"])
        if gen.get("demo_only"):
            st.caption("Illustrative identity correspondence. Do not interpret as an experimental OR.")
        with st.expander("Exact correspondence and operator table"):
            st.code(gen.get("resolved_matrix_M_from_A", ""), language=None)
            st.dataframe(pd.DataFrame(gen["adjacency"]), use_container_width=True)
            st.json({k: v for k, v in gen.items() if k not in ("adjacency", "cry_snapshot")})
        st.button(
            "Continue to AR", on_click=go, args=(STEPS[2],),
            type="primary", use_container_width=True,
        )

elif step == STEPS[2]:
    st.markdown(":teal[**03 / AR**]")
    st.subheader("EBSD inspection")
    st.write("Upload Oxford CTF or EDAX ANG orientations to inspect phase counts. "
             "An experimental parent fit is optional and requires a verified orientation relationship.")
    if not cry:
        st.warning("Run Cry first to establish the parent and product crystal models.")
        st.button("Go to Cry", on_click=go, args=(STEPS[0],))
        st.stop()
    if not gen:
        st.info("Gen has not been run. You may inspect EBSD independently; "
                "the complete chain will remain marked incomplete.")
    ar_mode = st.radio(
        "EBSD source", ("Upload file", "Synthetic demo"), horizontal=True,
        key="ar_source_mode", on_change=clear_from_ar,
    )
    if ar_mode == "Synthetic demo":
        st.caption("Three generated identity orientations · not measured NiTi or Cu–Al–Ni data.")
        uploaded = _SyntheticEBSD()
    else:
        uploaded = st.file_uploader(
            "EBSD file (.ctf or .ang), up to 8 MiB",
            type=["ctf", "ang"], key="ar_upload", on_change=clear_from_ar,
        )
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
                result["input_kind"] = (
                    "synthetic_software_demo" if ar_mode == "Synthetic demo"
                    else "user_ebsd_upload"
                )
                st.session_state["ar_result"] = result
                st.session_state["ar_bytes"] = blob
                st.session_state["ar_sha256"] = hashlib.sha256(blob).hexdigest()
                st.session_state.pop("ar_fit_result", None)
                ar, fit = result, None
                st.success("EBSD orientations imported. The reference frame is not inferred.")
    if ar:
        if uploaded is not None and hashlib.sha256(uploaded.getvalue()).hexdigest() != st.session_state.get("ar_sha256"):
            st.warning("File changed. Inspect the new upload before attempting a parent fit.")
        p, i = st.columns(2)
        p.metric("EBSD points", ar["points"])
        i.metric("Indexed", ar["indexed_points"])
        with st.expander("Phase counts and import details"):
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
