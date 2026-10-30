from __future__ import annotations

"""Standalone AR · Gen · Cry Streamlit application.

Deploy THIS entry point, not app/streamlit_app.py. There are no imported twin
application modules, no accidental legacy pages, and no executable old GUI.
"""

import hashlib
import json
from fractions import Fraction
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from cualni_cryst.legacy_suite_bridge import Phase
from cualni_cryst.ebsd_map import EBSDPhase
from cualni_cryst.ebsd_h5oina import load_h5oina_sample
from cualni_cryst.parent_fullmap_h5oina import reconstruct_conditional_parent_h5oina
from cualni_cryst.parent_grain_table_replay import replay_grain_boundary_tables
from cualni_cryst.parent_replay_exchange import (
    forward_inverse_roundtrip, export_replay_zip,
)

from cualni_cryst.parent_or_evidence import (
    audit_direct_basis, rebase_orientation_relationship,
)
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
    for key in ("cry_result", "gen_result", "ar_result", "ar_fit_result", "ar_bytes", "ar_sha256", "parent_interop_report", "parent_interop_zip", "parent_interop_inputs", "parent_full_h5", "parent_full_report", "parent_boundary_replay"):
        st.session_state.pop(key, None)


def clear_parent_fit() -> None:
    st.session_state.pop("ar_fit_result", None)
    st.session_state.pop("parent_interop_report", None)
    st.session_state.pop("parent_interop_zip", None)
    st.session_state.pop("parent_interop_inputs", None)
    st.session_state.pop("parent_full_h5", None)
    st.session_state.pop("parent_full_report", None)
    st.session_state.pop("parent_boundary_replay", None)


def clear_boundary_replay() -> None:
    st.session_state.pop("parent_boundary_replay", None)


def clear_from_ar() -> None:
    for key in ("ar_result", "ar_fit_result", "ar_bytes", "ar_sha256", "parent_interop_report", "parent_interop_zip"):
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
    st.write("Inspect Oxford H5OINA / CTF or EDAX ANG EBSD orientations. "
             "Parent fitting requires an independently verified orientation relationship.")
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
            "EBSD file (.h5oina up to 120 MiB, or .ctf/.ang up to 8 MiB)",
            type=["h5oina", "ctf", "ang"], key="ar_upload", on_change=clear_from_ar,
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
        if ar.get("source_format") == "h5oina":
            st.info(
                "Oxford H5OINA is read directly in radians. For speed, the parent "
                "pilot operates on a deterministic indexed-point sample, not the full "
                "pixel grid. The sample reference frame is not corrected automatically."
            )
            for phase_record in ar.get("phase_geometry", []):
                st.caption(
                    f"Indexed phase {phase_record['phase_id']}: {phase_record['name']} "
                    f"({phase_record['space_group']})"
                )
        with st.expander("18R basis and parent evidence", expanded=False):
            st.caption(
                "A conventional 18R reference OR must be rebased to the precise "
                "indexed 18R cell before use. The CIF and explicit integer "
                "change-of-basis P are needed; they are not inferred from EBSD."
            )
            conventional_cif = st.file_uploader(
                "Conventional 18R reference CIF (optional)", type=["cif"],
                key="ar_conventional_cif", on_change=clear_parent_fit,
            )
            basis_text = st.text_area(
                "P: columns of indexed basis in conventional direct coordinates",
                "1 0 0\\n0 1 0\\n0 0 1", height=100,
                key="ar_direct_basis", on_change=clear_parent_fit,
            )
            comparison_mode = st.radio(
                "Lattice-metric comparison",
                ("Strict indexing match", "Approximate literature screen"),
                horizontal=True, key="ar_basis_comparison_mode",
                on_change=clear_parent_fit,
            )
            basis_audit = None
            if conventional_cif is not None:
                try:
                    old_18r = parse_cif_phase(
                        conventional_cif.getvalue(),
                        Path(conventional_cif.name).stem, "2/m"
                    )
                    values = [
                        [float(Fraction(t)) for t in line.replace(",", " ").split()]
                        for line in basis_text.strip().splitlines()
                        if line.strip()
                    ]
                    new_18r = Phase(**cry["product_phase"])
                    tolerance = 5e-4 if comparison_mode == "Strict indexing match" else 3e-2
                    basis_audit = audit_direct_basis(
                        old_18r.lattice(), new_18r.lattice(), values,
                        relative_tolerance=tolerance,
                    )
                    st.success(
                        f"Basis metric compatible. Relative residual "
                        f"{basis_audit.metric_relative_residual:.2e}."
                    )
                    st.caption(
                        "A metric-only screen. Fractional basis coefficients may "
                        "describe a pseudocell; without atomic correspondence this "
                        "does NOT validate physical 18R translations or an OR."
                    )
                except (ValueError, TypeError, ArithmeticError) as exc:
                    st.error(f"Basis rejected: {exc}")
            st.caption(
                "A known retained/reverted beta orientation (or two noncollinear "
                "indexed beta directions) is needed to anchor an absolute parent."
            )
            known_parent_json = st.text_area(
                "Measured beta orientation g(parent crystal → same specimen frame), JSON 3×3",
                placeholder="[[1,0,0],[0,1,0],[0,0,1]]",
                key="ar_parent_anchor", on_change=clear_parent_fit,
            )
            st.caption(
                "Oxford's scanning-rotation angle is not automatically applied as a "
                "laboratory rotation. Use an independently calibrated frame matrix "
                "only if raw CS1 must be transformed to your reference specimen frame."
            )
            frame_correction_json = st.text_area(
                "Explicit specimen-frame rotation, JSON 3×3 (optional)",
                placeholder="[[1,0,0],[0,1,0],[0,0,1]]",
                key="ar_frame_correction", on_change=clear_parent_fit,
            )
            specimen_frame_ok = st.checkbox(
                "I cross-checked the Oxford Euler/specimen frame against an independent direction",
                key="ar_frame_externally_checked", on_change=clear_parent_fit,
            )

        with st.expander("Optional experimental parent fit", expanded=False):
            st.caption("Exploratory single-parent fit, not a complete segmented parent-grain map.")
            chosen = st.number_input("Indexed product phase ID", min_value=1,
                                     step=1, value=1)
            supplied_or = st.text_area(
                "Verified R(parent ← product), JSON list of 3×3 Cartesian rotations",
                placeholder='[[[1,0,0],[0,1,0],[0,0,1]]]', value=""
            )
            in_conventional = st.checkbox(
                "Supplied OR matrices refer to the CONVENTIONAL 18R basis",
                key="ar_or_uses_conventional_basis", on_change=clear_parent_fit,
            )
            n = st.slider("Maximum sampled points", 2, 48, 12)
            verified = st.checkbox("I independently verified the OR, phase identity and coordinate frames.")
            if st.button("Fit parent", disabled=not verified, type="primary"):
                if uploaded is not None and hashlib.sha256(uploaded.getvalue()).hexdigest() != st.session_state.get("ar_sha256"):
                    st.error("Selected upload has changed. Inspect it again.")
                else:
                    try:
                        or_for_indexed = supplied_or
                        if in_conventional:
                            if basis_audit is None or comparison_mode != "Strict indexing match":
                                raise ValueError(
                                    "OR conversion requires strict crystallographic metric "
                                    "agreement. A 3% literature pseudocell screen is NOT "
                                    "sufficient to establish a physical OR."
                                )
                            refs = np.asarray(json.loads(supplied_or), float)
                            if refs.ndim != 3 or refs.shape[1:] != (3, 3):
                                raise ValueError("OR must contain one or more 3×3 rotations")
                            parent_lattice = Phase(**cry["parent_phase"]).lattice()
                            identity = audit_direct_basis(
                                parent_lattice, parent_lattice, np.eye(3, dtype=int)
                            )
                            remapped = [
                                rebase_orientation_relationship(R, identity, basis_audit).tolist()
                                for R in refs
                            ]
                            or_for_indexed = json.dumps(remapped)
                        result = ar_parent_fit(
                            ar["filename"], st.session_state["ar_bytes"], cry,
                            int(chosen), or_for_indexed, max_points=n,
                            parent_anchor_orientation_json=known_parent_json,
                            indexed_18r_basis_verified=(basis_audit is not None
                                and comparison_mode == "Strict indexing match"),
                            specimen_frame_verified=specimen_frame_ok,
                            specimen_frame_correction_json=frame_correction_json,
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
        with st.expander("Parent ↔ 18R / MTEX & cross-tool replay", expanded=False):
            st.caption(
                "Forward parent → every proper 18R variant → inverse parent. "
                "Numeric tests only; the fitted OR is not evidence of a physical "
                "Cu–Al–Ni OR or the actual parent crystal."
            )
            st.caption(
                "Import the frozen Site 8 JSON manifest from your local "
                "reproducibility kit, or provide a proper parent←18R Cartesian "
                "OR and candidate β orientation as 3×3 JSON matrices. "
                "The app does not invent or silently modify these rotations."
            )
            manifest_file = st.file_uploader(
                "Fitted OR manifest (optional JSON, not an EBSD map)",
                type=["json"], key="parent_interop_manifest",
                on_change=clear_parent_fit,
            )
            or_text = st.text_area(
                "Q: 18R daughter crystal → cubic β parent crystal, 3×3 JSON",
                placeholder="[[...], [...], [...]]",
                key="parent_interop_q", on_change=clear_parent_fit,
            )
            parent_text = st.text_area(
                "Candidate β crystal → raw Oxford sample CS1, 3×3 JSON",
                placeholder="[[...], [...], [...]]",
                key="parent_interop_parent", on_change=clear_parent_fit,
            )
            indexed_cif_file = st.file_uploader(
                "Original indexed 18R CIF for AZtec Crystal (optional)",
                type=["cif"], key="parent_interop_indexed_cif",
                on_change=clear_parent_fit,
            )
            parent_cif_file = st.file_uploader(
                "Original measured β parent CIF (optional)",
                type=["cif"], key="parent_interop_parent_cif",
                on_change=clear_parent_fit,
            )
            accept_angle = st.slider(
                "Round-trip acceptance angle (degrees)", 1.0, 10.0, 3.0, 0.5,
                key="parent_interop_tolerance", on_change=clear_parent_fit,
            )
            n_sample = st.slider(
                "Measured daughter orientations sampled", 16, 256, 128, 16,
                key="parent_interop_sample_size", on_change=clear_parent_fit,
            )
            if ar.get("source_format") != "h5oina":
                st.info("A native H5OINA file is required for this release of the measured-orientation replay.")
            if st.button("Test parent ↔ daughter and build MTEX kit",
                         disabled=(ar.get("source_format") != "h5oina"),
                         type="primary"):
                try:
                    embedded = (
                        json.loads(manifest_file.getvalue())
                        if manifest_file is not None else {}
                    )
                    Q_input = (
                        json.loads(or_text) if or_text.strip() else
                        embedded.get("rotation_matrix_Q_parent_from_daughter")
                        or embedded.get("or", {}).get("Q_parent_from_daughter")
                    )
                    gA_input = (
                        json.loads(parent_text) if parent_text.strip() else
                        embedded.get("candidate_beta_parent_crystal_to_OxfordCS1_matrix")
                        or embedded.get("parent_crystal_to_sample_matrix")
                        or (fit or {}).get("parent_crystal_to_sample_rotation")
                    )
                    if Q_input is None or gA_input is None:
                        raise ValueError(
                            "Supply both Q and a candidate parent orientation. "
                            "One fitted OR alone cannot fix the absolute parent."
                        )
                    if not st.session_state.get("ar_bytes"):
                        raise ValueError("Reinspect the current H5OINA upload")
                    sampled, original_audit = load_h5oina_sample(
                        ar["filename"], st.session_state["ar_bytes"],
                        maximum_sample=int(n_sample),
                    )
                    product_phase = Phase(**cry["product_phase"])
                    parent_phase = Phase(**cry["parent_phase"])
                    parent_ops = EBSDPhase.from_point_group(
                        1, parent_phase.name, parent_phase.lattice(),
                        parent_phase.point_group,
                    ).proper_symmetry_cartesian
                    daughter_ops = EBSDPhase.from_point_group(
                        1, product_phase.name, product_phase.lattice(),
                        product_phase.point_group,
                    ).proper_symmetry_cartesian
                    if (not original_audit.get("phase_geometry") or
                        len(original_audit["phase_geometry"]) != 1):
                        raise ValueError(
                            "The measured map needs one indexed daughter phase"
                        )
                    observed_cell = original_audit["phase_geometry"][0]
                    expected_cell = np.array([
                        product_phase.a, product_phase.b, product_phase.c,
                        product_phase.alpha, product_phase.beta, product_phase.gamma,
                    ])
                    indexed_cell = np.array(
                        observed_cell["abc_angstrom"] + observed_cell["angles_deg"]
                    )
                    if not (
                        np.allclose(expected_cell[:3], indexed_cell[:3], rtol=0, atol=0.02)
                        and np.allclose(expected_cell[3:], indexed_cell[3:], rtol=0, atol=0.2)
                    ):
                        raise ValueError(
                            "Cry product cell does not match Oxford indexed 18R. "
                            "Choose the exact reduced C2/m cell first."
                        )
                    diagnostic = forward_inverse_roundtrip(
                        gA_input, Q_input, tuple(parent_ops), tuple(daughter_ops),
                        observed_daughter_g=sampled.orientations,
                        accepted_deg=float(accept_angle),
                    )
                    kit = export_replay_zip(
                        gA_input, Q_input, tuple(parent_ops), tuple(daughter_ops),
                        h5oina_filename=ar["filename"],
                        h5oina_sha256=st.session_state["ar_sha256"],
                        parent_cell=cry["parent_phase"],
                        daughter_cell=cry["product_phase"],
                        observed_daughter_g=sampled.orientations,
                        accepted_deg=float(accept_angle),
                        indexed_daughter_cif=(
                            indexed_cif_file.getvalue()
                            if indexed_cif_file is not None else None
                        ),
                        parent_cif=(
                            parent_cif_file.getvalue()
                            if parent_cif_file is not None else None
                        ),
                    )
                    st.session_state["parent_interop_inputs"] = {"Q": Q_input, "gA": gA_input}
                    st.session_state["parent_interop_report"] = diagnostic
                    st.session_state["parent_interop_zip"] = kit
                except (ValueError, TypeError, ArithmeticError,
                        AssertionError, UnicodeDecodeError, KeyError) as exc:
                    st.error(f"Round-trip rejected: {exc}")
                else:
                    st.success(
                        "Forward/inverse checks computed. External MTEX package "
                        "prepared; MATLAB execution and physical OR remain unverified."
                    )
            if st.session_state.get("parent_interop_report"):
                diagnostic = st.session_state["parent_interop_report"]
                k1, k2, k3 = st.columns(3)
                k1.metric("Daughter variants", diagnostic["n_predicted_variants"])
                k2.metric("Median fit", f'{diagnostic["median_forward_residual_deg"]:.2f}°')
                k3.metric("Within threshold",
                          f'{diagnostic["weighted_fraction_within_threshold"]:.1%}')
                st.caption(
                    "Measured point sampling is deterministic; no full MTEX grain graph "
                    "or MCL run is claimed. The inverse residual equals the forward "
                    "residual by rotation algebra, not independent confirmation."
                )
                with st.expander("Residuals, variant IDs and frame convention"):
                    st.json(diagnostic)
                st.download_button(
                    "Download MTEX, parent and 18R replay kit",
                    data=st.session_state["parent_interop_zip"],
                    file_name="cualni_parent_18r_cross_tool_replay.zip",
                    mime="application/zip", use_container_width=True,
                )
                st.caption(
                    "AZtec/Crystal requires an independently validated crystal "
                    "structure/CIF and correct sample frame; orientation CSVs "
                    "are reference outputs, not an AZtec-native phase file."
                )
                with st.expander("Replay daughter-grain boundaries (MTEX-style edge scores)"):
                    st.caption(
                        "Upload the exported grain quaternion CSV and its "
                        "boundary-pair CSV. Recomputes OR operator residuals "
                        "and Gaussian edge probabilities using the frozen Q. "
                        "This is not MTEX Markov clustering."
                    )
                    grain_csv=st.file_uploader(
                        "Daughter grain orientations CSV",type=["csv"],
                        key="parent_interop_grain_csv",
                        on_change=clear_boundary_replay,
                    )
                    boundary_csv=st.file_uploader(
                        "Boundary grain pairs CSV",type=["csv"],
                        key="parent_interop_boundary_csv",
                        on_change=clear_boundary_replay,
                    )
                    if st.button(
                        "Recalculate boundary operator scores",
                        disabled=(grain_csv is None or boundary_csv is None),
                    ):
                        try:
                            params=st.session_state["parent_interop_inputs"]
                            pa=Phase(**cry["parent_phase"])
                            ma=Phase(**cry["product_phase"])
                            p_sym=EBSDPhase.from_point_group(
                                1,pa.name,pa.lattice(),pa.point_group,
                            ).proper_symmetry_cartesian
                            m_sym=EBSDPhase.from_point_group(
                                1,ma.name,ma.lattice(),ma.point_group,
                            ).proper_symmetry_cartesian
                            boundary_report=replay_grain_boundary_tables(
                                grain_csv.getvalue(),boundary_csv.getvalue(),
                                params["Q"],tuple(p_sym),tuple(m_sym),
                            )
                            st.session_state["parent_boundary_replay"]=boundary_report
                        except (ValueError,TypeError,ArithmeticError,
                                AssertionError,KeyError) as exc:
                            st.error(f"Boundary replay rejected: {exc}")
                        else:
                            st.success(
                                "Boundary operator residuals independently "
                                "recomputed from supplied grain tables."
                            )
                    if st.session_state.get("parent_boundary_replay"):
                        replay=st.session_state["parent_boundary_replay"]
                        b1,b2=st.columns(2)
                        b1.metric("Boundary pairs",replay["n_adjacency_edges"])
                        b2.metric("Median misfit",
                                  f'{replay["median_operator_misfit_deg"]:.3f}°')
                        st.caption(
                            "A good boundary fit is not independent validation "
                            "of a physical β→18R relationship."
                        )
                        st.download_button(
                            "Download boundary evidence JSON",
                            json.dumps(replay,indent=2,allow_nan=False),
                            file_name="parent_boundary_replay.json",
                            mime="application/json",use_container_width=True,
                        )

                with st.expander("Native full-resolution parent HDF5", expanded=False):
                    st.caption(
                        "Fixed-Q, fixed-parent per-pixel mapping with MAD and "
                        "connected-variant support filters. NOT MTEX's "
                        "calcVariantGraph or a verified beta-grain reconstruction."
                    )
                    mad_max = st.slider(
                        "Maximum MAD (degrees)", 0.5, 3.0, 1.5, 0.25,
                        key="parent_full_mad",
                    )
                    min_patch = st.number_input(
                        "Minimum connected variant-patch area (pixels)",
                        min_value=1,max_value=10000,value=50,step=10,
                        key="parent_full_min_patch",
                    )
                    if st.button("Build full-resolution conditional parent map",
                                 type="primary"):
                        try:
                            inputs = st.session_state["parent_interop_inputs"]
                            pc = Phase(**cry["parent_phase"])
                            mc = Phase(**cry["product_phase"])
                            sym_p = EBSDPhase.from_point_group(
                                1,pc.name,pc.lattice(),pc.point_group,
                            ).proper_symmetry_cartesian
                            sym_m = EBSDPhase.from_point_group(
                                1,mc.name,mc.lattice(),mc.point_group,
                            ).proper_symmetry_cartesian
                            binary, summary = reconstruct_conditional_parent_h5oina(
                                ar["filename"],st.session_state["ar_bytes"],
                                inputs["gA"],inputs["Q"],tuple(sym_p),tuple(sym_m),
                                max_mad_deg=float(mad_max),
                                max_fit_deg=float(accept_angle),
                                min_variant_patch_pixels=int(min_patch),
                            )
                            st.session_state["parent_full_h5"] = binary
                            st.session_state["parent_full_report"] = summary
                        except (ValueError,TypeError,ArithmeticError,
                                AssertionError,KeyError,OverflowError) as exc:
                            st.error(f"Full-resolution export rejected: {exc}")
                        else:
                            st.success("Numerical parent map generated; not physically certified.")
                    if st.session_state.get("parent_full_report"):
                        full_report=st.session_state["parent_full_report"]
                        a1,a2=st.columns(2)
                        a1.metric("Within OR tolerance",
                                  full_report["n_fitted_within_deg"])
                        a2.metric("Variant-patch supported",
                                  full_report["n_variant_patch_supported"])
                        st.caption("Rejected pixels have NaN parent orientations.")
                        st.download_button(
                            "Download conditional parent HDF5",
                            data=st.session_state["parent_full_h5"],
                            file_name="cualni_parent_conditional_UNVERIFIED.h5",
                            mime="application/x-hdf5",use_container_width=True,
                        )
                        with st.expander("Full-grid evidence audit"):
                            st.json(full_report)

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
        payload = report_json(cry, gen, ar, fit,
                              st.session_state.get("parent_interop_report"),
                              st.session_state.get("parent_boundary_replay"),
                              st.session_state.get("parent_full_report"))
    except (ValueError, TypeError) as exc:
        st.error(f"Cannot export results: {exc}")
    else:
        st.download_button("Download reproducible JSON report", data=payload,
                           file_name="ar_gen_cry_results.json", mime="application/json",
                           type="primary", use_container_width=True)
        with st.expander("Preview results"):
            st.json(json.loads(payload))
    st.button("Back to Cry", on_click=go, args=(STEPS[0],))
