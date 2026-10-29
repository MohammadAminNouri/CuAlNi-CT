from __future__ import annotations

"""Source-by-source Cu–Al–Ni validation; read-only scientific evidence UI."""
import json
import pandas as pd
import streamlit as st

from twin_app.cualni_paper_campaign_2026 import campaign_registry,run_campaign


def render_cualni_literature_campaign(signature: str) -> None:
    with st.expander(
        "Cu–Al–Ni literature · 18-paper reproducibility and missing-evidence audit",
        expanded=False,
    ):
        data=campaign_registry()
        st.markdown(
            "**Eighteen independently identified Cu–Al–Ni papers**, spanning "
            "experimental EBSD, TEM, diffraction, orthorhombic 2H, modified "
            "18R and Ball–James/PTMC 2H crystal mechanics. "
            "The app never constructs an unmeasured parent/product "
            "correspondence from a phase name or combines two specimens."
        )
        st.warning(
            "Source-input checks and numerical metric self-consistency "
            "are NOT experimental verification of Cayron's full A/M/M "
            "shear/shear Eq. 42 or of Ball–James CC1–CC3. Missing 3D twin "
            "geometry, parent reconstruction, C or uncertainty stays "
            "NOT IDENTIFIABLE. Reused published lattice cells count as "
            "borrowed evidence rather than new experimental measurements."
        )
        papers=data["papers"]
        st.dataframe(pd.DataFrame([
            {
                "Paper":p["id"],
                "Year":p["year"],
                "DOI":p["doi"],
                "Source evidence":p["source_access"],
                "What it can test":p["validation_level"],
                "Full experimental Eq.42": "NOT IDENTIFIABLE",
            } for p in papers
        ]),hide_index=True,width="stretch")
        picked=st.selectbox(
            "Inspect the published input, expected observations and limitations",
            range(len(papers)),
            format_func=lambda i:f"{papers[i]['year']} · {papers[i]['title']}",
            key="tf_cualni_campaign_paper_choice",
        )
        selected=papers[picked]
        st.markdown(f"**{selected['title']}**")
        st.caption(f"DOI: {selected['doi']}")
        st.write("Source data extracted for this test:")
        st.json(selected["observables"])
        st.write("What cannot be concluded from this source:")
        st.write(selected["limitation"])
        st.link_button("Open the source paper",selected["url"])
        if st.button(
            "Run the full 18-paper independent numerical and scope campaign",
            key="tf_cualni_campaign_execute",
        ):
            try:
                with st.spinner(
                    "Cross-checking the 18 sources, crystal metrics, Hane–Shield "
                    "twin shears, exact PTMC fractions and all 96 habits…"
                ):
                    result=run_campaign()
                st.session_state["tf_cualni_campaign_execution"]=(signature,result)
            except Exception as exc:
                st.error(
                    "Scientific benchmark failed; do not display a false pass: "
                    f"{type(exc).__name__}: {exc}"
                )
        saved=st.session_state.get("tf_cualni_campaign_execution")
        if saved is None or saved[0]!=signature:
            st.caption(
                "A full benchmark must be explicitly triggered. Inspect source "
                "records above even before running expensive numerical tests."
            )
            return
        result=saved[1]
        a,b,c=st.columns(3)
        a.metric("Unique checked papers",result["papers_evaluated"])
        b.metric("Numeric/input checks completed",result[
            "paper_input_or_lattice_numerical_passes"
        ])
        c.metric("Full empirical Eq.42/CC1–CC3 passes",result[
            "full_Cayron_AMM_or_BJ_experimental_predictions_validated"
        ])
        st.dataframe(pd.DataFrame([
            {
                "Paper":r["id"],
                "Evaluation":r["outcome"],
                "Source metric / input reproduced":r["paper_numeric_input_reproduced"],
                "Published 3D theoretical habit reproduced":r[
                    "paper_theoretical_3d_twin_habit_prediction_reproduced"
                ],
                "Observed 3D twin / habit validated":r[
                    "experimental_3d_twin_habit_observation_validated"
                ],
                "Why full comparison is unavailable":r["reason"],
            } for r in result["source_records"]
        ]),hide_index=True,width="stretch")
        st.caption(result["warning"])
        chosen=next((r for r in result["source_records"]
                     if r["id"]==selected["id"]),None)
        if chosen:
            st.markdown("**Calculated evidence for this paper**")
            st.json(chosen["numeric_checks"])
        st.download_button(
            "Export all 18 source-linked scientific results (JSON)",
            data=json.dumps(result,indent=2,default=str),
            file_name="cualni_18_paper_scientific_campaign_2026.json",
            mime="application/json",
            key="tf_cualni_campaign_download",
        )
