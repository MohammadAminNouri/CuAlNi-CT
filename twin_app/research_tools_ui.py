from __future__ import annotations

"""A compact home for existing orbit analysis and the complete lossless export.

Only UI placement changes: all backend calculations, data and identifiers remain intact.
"""

from dataclasses import asdict
import json

import streamlit as st

from cualni_cryst.numerics import DEFAULT_NUMERICAL_POLICY
from .p1_ui import render_p1_symmetry_orbit_panel
from .report_export import make_export


def render_research_tools(catalogue, report, payload, signature):
    with st.expander("Research tools and complete scientific export", expanded=False):
        orbit_tab, export_tab = st.tabs(("Symmetry-equivalent solutions", "Export all results"))
        with orbit_tab:
            st.caption(
                "Optional grouping of symmetry-equivalent macroscopic tensors. "
                "An orbit count does not establish distinct microstructures."
            )
            render_p1_symmetry_orbit_panel(report, payload, signature, inline=True)
        with export_tab:
            st.caption(
                "Complete unabridged analysis: entered inputs, crystal conventions, "
                "variant pairs, habit solutions, numerical residuals, classification "
                "evidence and provenance. Published benchmark outputs are not inserted."
            )
            record = make_export(report, input_payload=payload, policy=asdict(DEFAULT_NUMERICAL_POLICY))
            record["physical_mode_catalogue"] = {
                "derivation": "parent metric-symmetry orbits of actual stretch-variant pairs; no reference outputs loaded",
                "modes": [
                    {
                        "mode_id": m.mode_id,
                        "symmetry_equivalent_stretch_pairs": m.stretch_pairs,
                        "shear_magnitude": m.shear,
                        "verified_types_and_unresolved": m.types,
                        "source_records": [
                            {"operator_family": e.operator_family,
                             "pair_id": e.pair.pair_id,
                             "construction_id": e.construction.construction_id,
                             "classification": e.construction.classification,
                             "classification_status": e.construction.classification_status}
                            for e in m.entries
                        ],
                    }
                    for m in catalogue.modes
                ],
                "incompatible_pair_orbits": catalogue.incompatible_pair_orbits,
            }
            st.download_button(
                "Download complete analysis as JSON",
                data=json.dumps(record, indent=2, default=str),
                file_name="twin_family_research_record.json",
                mime="application/json",
                key="tf_export_v8",
            )
