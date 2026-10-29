from __future__ import annotations

"""Coherent three-stage Streamlit laboratory; preserves the existing application."""

import sys
import json
import numpy as np
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cualni_cryst.legacy_suite_bridge import (
    Phase, crystal_geometry, inspect_ebsd, variant_groupoid, reconstruct_ebsd_parent
)

st.set_page_config(page_title="AR · Gen · Cry", layout="wide")
st.title("AR · Gen · Cry")
st.caption("Modern Python 3 workbench built on CuAlNi-CT's validated core.")
st.warning("Lattice correspondence C is not an experimentally measured orientation relationship (OR). "
           "Crystallographic variant counts cannot by themselves reconstruct EBSD parent grains.")


def input_phase(prefix: str, initial: tuple) -> Phase:
    st.subheader(prefix)
    name = st.text_input(f"{prefix} phase", initial[0], key=f"{prefix}_name")
    group_options = ["1", "-1", "2", "2/m", "222", "mmm", "4/mmm", "6/mmm", "m-3m"]
    pg = st.selectbox(f"{prefix} point group", group_options,
                      index=group_options.index(initial[1]), key=f"{prefix}_pg")
    cols = st.columns(3)
    lengths = [cols[i].number_input(f"{prefix}: {c} (Å)", min_value=0.001,
               value=float(initial[i+2]), format="%.6f", key=f"{prefix}_{c}")
               for i, c in enumerate("abc")]
    angles = [cols[i].number_input(f"{prefix}: {c} (°)", min_value=0.001,
              max_value=179.999, value=float(initial[i+5]), key=f"{prefix}_{c}")
              for i, c in enumerate(("alpha", "beta", "gamma"))]
    return Phase(name, pg, *lengths, *angles)


left, right = st.columns(2)
with left:
    parent = input_phase("Parent", ("Illustrative cubic", "m-3m", 5.8, 5.8, 5.8, 90, 90, 90))
with right:
    product = input_phase("Product", ("Illustrative orthorhombic", "mmm", 4.3, 5.3, 4.5, 90, 90, 90))

with st.expander("01 · Cry: lattice and reciprocal-space geometry", expanded=True):
    direction = st.text_input("Direct [uvw]", "1 0 0")
    plane = st.text_input("Reciprocal plane (hkl)", "1 0 0")
    if st.button("Calculate metric geometry"):
        try:
            u = tuple(float(v) for v in direction.replace(",", " ").split())
            h = tuple(float(v) for v in plane.replace(",", " ").split())
            st.json({"parent": crystal_geometry(parent, u, h),
                     "product": crystal_geometry(product, u, h)})
        except (ValueError, ArithmeticError) as exc:
            st.error(str(exc))

with st.expander("02 · Gen: correspondence variants and operator groupoid", expanded=True):
    st.info("The identity matrix is a software demonstration, not a Cu–Al–Ni crystallographic assignment.")
    c_text = st.text_area("Exact C_(M←A) (3 rows)", "1 0 0\n0 1 0\n0 0 1", height=100)
    if st.button("Calculate exact groupoid"):
        try:
            r = variant_groupoid(parent, product, c_text)
            a, b, c = st.columns(3)
            a.metric("Variants", r["variants"])
            b.metric("Operator classes", r["operator_classes"])
            c.metric("Intersection |H|", r["subgroup_order"])
            st.dataframe(pd.DataFrame(r["adjacency"]), use_container_width=True)
            st.json({k: v for k, v in r.items() if k != "adjacency"})
        except (ValueError, ArithmeticError, AssertionError) as exc:
            st.error(str(exc))

with st.expander("03 · AR: EBSD inspection and reconstruction handoff", expanded=True):
    file = st.file_uploader("Oxford CTF or EDAX ANG EBSD data", type=["ctf", "ang"])
    if file is not None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ("upload" + Path(file.name).suffix.lower())
            path.write_bytes(file.getvalue())
            try:
                ebsd, report = inspect_ebsd(path)
                st.json(report)
                st.info("Optional single-parent reconstruction below requires independently "
                        "verified Cartesian OR matrices and an explicit common reference frame.")
                phase_ids = sorted(int(v) for v in np.unique(
                    ebsd.phase_id[ebsd.indexed]) if int(v) > 0)
                if phase_ids:
                    phase_id = st.selectbox("Indexed product phase ID", phase_ids)
                    or_text = st.text_area(
                        "Verified OR matrices R_(parent←product) as JSON list of 3×3 rotations",
                        placeholder='[[[1,0,0],[0,1,0],[0,0,1]]]',
                        value="",
                    )
                    approved = st.checkbox(
                        "I verified these ORs, their coordinate convention, "
                        "and the input phase identities independently."
                    )
                    max_points = st.slider("Maximum sampled EBSD points", 2, 48, 12)
                    if st.button("Exploratory parent reconstruction", disabled=not approved):
                        try:
                            matrices = np.asarray(json.loads(or_text), dtype=float)
                            recon = reconstruct_ebsd_parent(
                                ebsd, parent, product, phase_id, matrices,
                                maximum_points=max_points,
                            )
                            st.json(recon)
                        except (ValueError, TypeError, ArithmeticError, AssertionError) as exc:
                            st.error(f"Parent reconstruction stopped: {exc}")
            except (ValueError, UnicodeError, OSError) as exc:
                st.error(str(exc))
st.caption("Historical original Python 2 code is not executed in the Streamlit server.")
