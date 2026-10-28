from __future__ import annotations

"""One CTF input, two scientific routes: measured EBSD and conditional OR fits.

OR-free maps NEVER assert phase identity, transformation OR, crystallographic
correspondence or a physical twin type. OR-dependent outputs require
independent input consistency and the separately calculated theory report.
"""

import hashlib
import io
import json

import numpy as np
import pandas as pd
import streamlit as st

from .ctf_independent import (
    compare_ctf_model_settings, declared_ctf_phases,
    independent_ctf_cartography, plot_ctf_field,
)
from .ctf_rich_io import (
    inspect_ctf_measurements, plot_ctf_numeric_map, ctf_field_histogram,
    raw_catalog_json, raw_catalog_npz, ctf_evidence_bundle,
    suggested_ctf_product_phase_id,
)
from .ebsd_integration import (
    MAX_CTF_BYTES, ctf_analysis_signature, inspect_ctf_upload,
    parse_explicit_or, run_ctf_observation, summarize_twin_candidates,
)


@st.cache_data(show_spinner=False, max_entries=4)
def _ctf_preview(file_bytes: bytes, filename: str, unit=None) -> dict:
    return inspect_ctf_upload(
        file_bytes, filename=filename, three_dimensional_angle_unit=unit,
    )


@st.cache_data(show_spinner=False, max_entries=3, ttl=3600)
def _ctf_catalog(file_bytes: bytes, filename: str) -> dict:
    return inspect_ctf_measurements(file_bytes, filename)


@st.cache_data(show_spinner=False, max_entries=3, ttl=3600)
def _ipf_calculation(raw, filename, phase_id, direction, unit):
    from twin_app.ctf_ipf import symmetry_ipf_colors
    return symmetry_ipf_colors(
        raw,filename,phase_id=phase_id,sample_direction=direction,
        three_dimensional_angle_unit=unit,
    )


@st.cache_data(show_spinner=False, max_entries=2, ttl=3600)
def _zip_ctf_evidence(raw_catalog, file_bytes, measured=None):
    return ctf_evidence_bundle(
        raw_catalog, file_bytes, independent=measured,
    )


@st.cache_data(show_spinner=False, max_entries=2, ttl=3600)
def _independent_cartography(file_bytes, filename, threshold, nmin, unit=None):
    return independent_ctf_cartography(
        file_bytes, filename,
        misorientation_threshold_deg=threshold,
        minimum_grain_pixels=nmin,
        three_dimensional_angle_unit=unit,
    )


def _map_download(result) -> bytes:
    payload = io.BytesIO()
    np.savez_compressed(
        payload,
        **{name: np.asarray(values) for name, values in result["maps"].items()},
    )
    return payload.getvalue()


def _show_independent_cartography(result):
    st.markdown("**Measured EBSD — no OR supplied**")
    a, b, c = st.columns(3)
    a.metric("Indexed pixels", f"{result['n_indexed']:,}")
    b.metric("Retained grains", f"{result['n_grains']:,}")
    c.metric("Grain adjacencies", f"{result['n_distinct_grain_adjacencies']:,}")
    st.caption(
        f"Grain threshold {result['threshold_deg']}° · KAM neighbor cutoff "
        f"{result['kam_cutoff_deg']}° · CTF-declared phase symmetries only"
    )
    retained = result["n_retained_indexed"]
    st.caption(
        f"{retained:,} of {result['n_indexed']:,} indexed pixels "
        "belong to orientation regions retained by the minimum-size cutoff."
    )
    jumps = result["neighbor_disorientation"]
    if jumps["median_deg"] is not None:
        st.caption(
            f"Adjacent indexed-pixel disorientation: median "
            f"{jumps['median_deg']:.2f}° · "
            f"{100*jumps['fraction_above_30_deg']:.1f}% over 30°"
        )
        if jumps["fraction_above_30_deg"] > 0.25:
            st.warning(
                "Many adjacent indexed pixels have large orientation jumps. "
                "Check AZtec indexing, pseudosymmetry and fine twins before "
                "interpreting segmented orientation regions as physical grains."
            )
    field_labels = {
        "Phase ID (AZtec indexing)": "phase",
        "Segmented grain ID": "grain_id",
        "Kernel average misorientation, °": "kam_deg",
        "Grain orientation spread, °": "gos_deg",
        "Maximum indexed-neighbor disorientation, °": "neighbor_max_deg",
        "Segmented boundary-pixel mask": "grain_boundary_mask",
        "Mean angular deviation (MAD, AZtec)": "MAD",
        "Band contrast (BC)": "BC",
        "Band slope (BS)": "BS",
    }
    selected = st.selectbox(
        "EBSD map", list(field_labels),
        key="tf_ctf_map_channel",
    )
    derived_maps = result["maps"]
    zlevels = np.unique(np.round(derived_maps["z"], 6))
    if len(zlevels) > 1:
        zselect = float(st.select_slider(
            "Analysis Z slice", options=[float(z) for z in zlevels],
            key="tf_ctf_analysis_z",
        ))
        mask = np.isclose(derived_maps["z"],zselect,atol=1e-5,rtol=0)
        shown_maps = {
            key:np.asarray(values)[mask]
            for key,values in derived_maps.items()
        }
    else:
        shown_maps = derived_maps
    st.plotly_chart(
        plot_ctf_field(shown_maps, field_labels[selected]),
        width="stretch",
    )
    with st.expander("Measured grains and boundaries", expanded=False):
        st.dataframe(
            pd.DataFrame(result["grains"]).drop(
                columns=["mean_crystal_to_sample_rotation"],
                errors="ignore",
            ),
            hide_index=True, width="stretch",
        )
        st.caption(
            "These are orientation-segmented grains, not Cayron CT variants. "
            "A 2-D trace does not determine a unique 3-D twin plane."
        )
    st.download_button(
        "Export all pixel maps (.npz)",
        data=_map_download(result),
        file_name="measured_ebsd_maps.npz",
        mime="application/octet-stream",
        key="tf_ctf_independent_npz",
    )
    summary = {k: v for k, v in result.items() if k != "maps"}
    st.download_button(
        "Export measured EBSD audit (.json)",
        data=json.dumps(summary, default=str, indent=2),
        file_name="measured_ebsd_independent.json",
        mime="application/json", key="tf_ctf_independent_json",
    )
    st.warning(
        "NOT CERTIFIED: CTF phase labels and reconstructed orientation grains "
        "are not independent 2H/18R diffraction indexing, a parent OR, "
        "or a Type I/II/Compound twin assignment."
    )


def render_ctf_panel(report, project_payload, transformation_id, signature,
                     *, provided_ctf=None):
    """Compact Step-01 EBSD, optionally reusing the same top-level CTF upload."""
    if provided_ctf is None:
        upload = st.file_uploader(
            "Oxford / AZtec indexed CTF map", type=["ctf"],
            key="tf_ctf_v1_file", max_upload_size=35,
            help="Indexed CTF, maximum 35 MB.",
        )
        if upload is None:
            return
        filename, content = upload.name, upload.getvalue()
    else:
        filename, content = provided_ctf
    if len(content) > MAX_CTF_BYTES:
        st.error("CTF exceeds the 35 MB analysis limit.")
        return
    # Always show the complete vendor measurement inventory FIRST.
    # Native orientation and the harder symmetry analysis may be unavailable
    # on otherwise readable CTFs; do not hide their raw quality maps.
    try:
        catalog = _ctf_catalog(content, upload.name)
    except (ValueError, RuntimeError, UnicodeError, OSError) as exc:
        st.error(f"CTF table could not be read: {exc}")
        return

    st.caption(
        f"{catalog['indexed_count']:,}/{catalog['n_points']:,} indexed points "
        f"({100*catalog['indexed_fraction']:.1f}%) · "
        f"{len(catalog['numeric_columns'])} original measurement columns"
    )
    with st.expander("CTF maps, quality and acquisition data", expanded=True):
        a, b, c = st.columns(3)
        a.metric("Pixels", f"{catalog['n_points']:,}")
        b.metric("Unindexed", f"{catalog['unindexed_count']:,}")
        c.metric("Phase labels", len(catalog["phase_rows"])-int(
            any(p["phase_id"]==0 for p in catalog["phase_rows"])
        ))
        grid = catalog["grid"]
        st.caption(
            f"Coordinate grid {grid['n_x']} × {grid['n_y']} × {grid['n_z']}"
            f" · duplicate coordinates {grid['duplicate_coordinates']}"
            f" · {'complete' if grid['complete_unique_cartesian_grid'] else 'irregular'}"
        )
        possible = [
            field for field in catalog["maps"] if field not in {"x","y","z"}
        ]
        chosen = st.selectbox(
            "Map channel", possible, key="tf_ctf_vendor_map",
        )
        zvalues = np.unique(np.round(catalog["maps"]["z"], 6))
        layer = (
            float(st.select_slider(
                "Z slice (CTF units)", options=[float(v) for v in zvalues],
                key="tf_ctf_z_slice",
            ))
            if len(zvalues)>1 else None
        )
        st.plotly_chart(
            plot_ctf_numeric_map(catalog, chosen, z_value=layer),
            width="stretch",
        )
        with st.expander("Distribution and original acquisition details",expanded=False):
            st.plotly_chart(
                ctf_field_histogram(catalog,chosen),
                width="stretch",
            )
            st.dataframe(
                pd.DataFrame(catalog["quality_summary"]),
                hide_index=True,width="stretch",
            )
            st.dataframe(
                pd.DataFrame(catalog["phase_rows"]),
                hide_index=True,width="stretch",
            )
            st.json({
                "grid":catalog["grid"],
                "acquisition":catalog["header"],
                "numeric_columns":catalog["numeric_columns"],
            })
        left,right = st.columns(2)
        left.download_button(
            "Download all raw pixel maps (.npz)",
            data=raw_catalog_npz(catalog),
            file_name="ctf_raw_measurements.npz",
            mime="application/octet-stream",
            key="tf_ctf_raw_npz",
        )
        right.download_button(
            "Download complete CTF audit (.json)",
            data=raw_catalog_json(catalog),
            file_name="ctf_original_header_and_quality.json",
            mime="application/json",
            key="tf_ctf_raw_json",
        )
        st.caption(
            "Vendor-reported Euler angles, phase labels and quality values. "
            "Unindexed Euler/MAD placeholders are masked. No Kikuchi "
            "reindexing, IPF coloring or parent OR has been inferred."
        )
    # 3-D CTF Euler units differ across historical AZtec exports.
    # Never guess degrees vs radians from the angles themselves.
    three_d = int(float(catalog["header"].get("ZCELLS","1"))) > 1
    unit = None
    if three_d:
        choice = st.selectbox(
            "3-D CTF Euler-angle unit",
            ("Select the documented unit","Degrees","Radians"),
            key="tf_ctf_3d_angle_units",
        )
        unit = {"Degrees":"degree","Radians":"radian"}.get(choice)
        if unit is None:
            st.info(
                "Raw measurement maps are available. Select the documented "
                "Euler-angle unit to unlock 3-D orientation/grain analysis."
            )
            return
    try:
        preview = _ctf_preview(content, upload.name, unit)
    except (ValueError, RuntimeError, UnicodeError, OSError) as exc:
        st.info(f"Orientation analysis unavailable: {exc}")
        return
    try:
        phases_declared = declared_ctf_phases(content)
    except (ValueError, RuntimeError, UnicodeError) as exc:
        phases_declared = ()
        st.warning(f"CTF phase definitions need review: {exc}")

    indexed_counts = preview["phase_counts"]
    st.caption(
        f"{preview['indexed_pixels']:,}/{preview['total_pixels']:,} indexed pixels · "
        + ", ".join(f"phase {k}: {v:,}" for k,v in indexed_counts.items())
    )
    if phases_declared:
        st.dataframe(pd.DataFrame([
            {
                "CTF ID": p["phase_id"], "Header label": p["name"],
                "Space group (declared)": p["space_group"],
                "Point group": p["declared_point_group"],
                "Indexed pixels": indexed_counts.get(p["phase_id"], 0),
            } for p in phases_declared
        ]), hide_index=True, width="stretch")
    else:
        st.warning("The CTF has no complete phase definitions. No symmetry-based independent analysis is possible.")
    st.caption("AZtec phase names are indexing assumptions, not independent diffraction results.")
    if phases_declared:
        with st.expander("IPF map · phase-symmetry Laue color key", expanded=False):
            active_ids = sorted(indexed_counts)
            ipf_id = st.selectbox(
                "Indexed phase", active_ids,
                format_func=lambda n: f"CTF phase {n} — "+next(
                    (p["name"] for p in phases_declared if p["phase_id"]==n),
                    "not defined",
                ),
                key="tf_ctf_ipf_phase",
            )
            ref_axis=st.selectbox(
                "Sample reference direction", ("Z","X","Y"),
                key="tf_ctf_ipf_sample_axis",
            )
            if st.button("Calculate symmetry-aware IPF colors",
                         key="tf_ctf_ipf_calculate"):
                try:
                    ipf=_ipf_calculation(
                        content,filename,ipf_id,ref_axis,unit,
                    )
                    st.session_state["tf_ctf_ipf"] = (
                        (catalog["file_sha256"],ipf_id,ref_axis,unit),ipf,
                    )
                except (ValueError,RuntimeError,ImportError) as exc:
                    st.error(f"IPF unavailable: {exc}")
            saved_ipf=st.session_state.get("tf_ctf_ipf")
            if saved_ipf and saved_ipf[0] == (
                catalog["file_sha256"],ipf_id,ref_axis,unit,
            ):
                ipf=saved_ipf[1]
                from twin_app.ctf_ipf import plot_symmetry_ipf
                zvals=np.unique(np.round(ipf["z"],6))
                ipf_z=(float(st.select_slider(
                    "IPF Z slice",options=[float(v) for v in zvals],
                    key="tf_ctf_ipf_z",
                )) if len(zvals)>1 else None)
                st.plotly_chart(plot_symmetry_ipf(ipf,z_value=ipf_z),
                                width="stretch")
                from orix.quaternion.symmetry import get_point_group
                from orix.plot import IPFColorKeyTSL
                from orix.vector import Vector3d
                sg=next(
                    p["space_group_number"] for p in phases_declared
                    if p["phase_id"]==ipf_id
                )
                axis={"X":(1,0,0),"Y":(0,1,0),"Z":(0,0,1)}[ref_axis]
                legend=IPFColorKeyTSL(get_point_group(sg),
                                      direction=Vector3d(axis))
                st.pyplot(legend.plot(return_figure=True),clear_figure=True)
                st.caption(
                    "Standard TSL Laue IPF convention via orix. "
                    "Crystal/sample frames follow the raw AZtec settings. "
                    "The color key does not independently validate phase indexing "
                    "or specimen-axis registration."
                )
                out=io.BytesIO()
                np.savez_compressed(
                    out,x=ipf["x"],y=ipf["y"],z=ipf["z"],
                    rgb=ipf["rgb"],active_mask=ipf["active_mask"],
                )
                st.download_button(
                    "Download phase IPF color arrays (.npz)",
                    data=out.getvalue(),file_name="ctf_ipf_colors.npz",
                    mime="application/octet-stream",
                    key="tf_ctf_ipf_export",
                )

    # A CTF's indexed phase ID is a vendor label, NOT proof that the
    # structure is the martensite. However the UI must not default to the
    # absent 2H phase when an 18R-only file contains only phase 3.
    # On a newly uploaded file, suggest its unique indexed non-parent phase;
    # never override an explicit user adjustment during the same file.
    suggested = suggested_ctf_product_phase_id(indexed_counts)
    file_key = catalog["file_sha256"]
    if st.session_state.get("tf_ctf_phase_id_source") != file_key:
        st.session_state["tf_ctf_v1_parent"] = 1
        st.session_state["tf_ctf_v1_product"] = suggested
        st.session_state["tf_ctf_phase_id_source"] = file_key
    st.caption(
        f"Suggested product phase ID {suggested} from indexed CTF labels "
        "when unique. Verify its real crystallographic phase."
    )
    a, b, c = st.columns(3)
    phase_a = int(a.number_input(
        "Parent phase ID", min_value=1, value=1, step=1,
        key="tf_ctf_v1_parent",
        help="The parent may be absent among indexed pixels.",
    ))
    phase_m = int(b.number_input(
        "Martensite phase ID", min_value=1, value=suggested, step=1,
        key="tf_ctf_v1_product",
        help="AZtec indexing label only; verify the structure independently.",
    ))
    nmin = int(c.number_input(
        "Minimum pixels per grain", min_value=1, max_value=1000,
        value=3, step=1, key="tf_ctf_v1_minpix",
    ))
    threshold = float(st.number_input(
        "Orientation segmentation threshold (degrees)", min_value=0.1,
        max_value=30.0, value=3.0, step=0.5, key="tf_ctf_independent_threshold",
    ))
    invalid = False
    if phase_a == phase_m:
        st.error("Parent and martensite must use different phase IDs.")
        invalid = True
    if phase_m not in indexed_counts:
        st.error(f"No indexed martensite pixels for phase {phase_m}.")
        invalid = True
    extras = sorted(set(indexed_counts).difference({phase_a,phase_m}))
    if extras:
        st.error(
            f"Indexed phases {extras} are outside the selected two-phase "
            "model. Split or explicitly remap the phase data first."
        )
        invalid = True

    # Report a mismatch even when the user only runs OR-free measured maps.
    # This does not disable independent EBSD cartography.
    if phases_declared and phase_a != phase_m:
        try:
            active_setting_audit = compare_ctf_model_settings(
                phases_declared, project_payload, transformation_id,
                phase_a, phase_m,
            )
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            st.caption(f"Model-to-CTF setting check unavailable: {exc}")
        else:
            if not active_setting_audit["pass"]:
                st.warning(
                    "The active crystal setup differs from the CTF phase "
                    "definitions. Measured maps remain available, but "
                    "EBSD/theory comparison is blocked until settings match."
                )

    # This branch computes directly from the supplied CTF and its
    # declared point groups; no OR, CIF or completed CT solver is required.
    independent_key = hashlib.sha256(
        content + json.dumps(
            {"threshold":threshold,"nmin":nmin,"euler_unit":unit},sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    auto = st.toggle(
        "Calculate grain and misorientation maps on upload",
        value=True,key="tf_ctf_auto_measure",
        help=(
            "Uses the CTF-declared phase symmetry and measured Euler angles. "
            "The result is numerical orientation segmentation, NOT "
            "independent confirmation of physical grains or twins."
        ),
    )
    requested = st.button(
        "Calculate grain/KAM maps",
        key="tf_ctf_v1_measure",disabled=not bool(phases_declared),
    )
    # Raw maps are always immediate. The expensive phase-symmetry
    # calculation is cached by bytes, segmentation threshold and min size.
    if phases_declared and (auto or requested):
        try:
            with st.spinner("Calculating measured orientation regions and KAM…"):
                result = _independent_cartography(
                    content, upload.name, threshold, nmin, unit,
                )
            st.session_state["tf_ctf_independent"] = (independent_key,result)
        except (ValueError, RuntimeError, ArithmeticError, OSError) as exc:
            st.error(f"EBSD measurement failed: {exc}")
        except Exception as exc:
            st.error(f"EBSD measurement failed ({type(exc).__name__}): {exc}")
    saved_independent = st.session_state.get("tf_ctf_independent")
    measured = (
        saved_independent[1]
        if saved_independent and saved_independent[0] == independent_key
        else None
    )
    if measured is not None:
        _show_independent_cartography(measured)
        if phases_declared:
            with st.expander(
                "Explore parent-symmetry embedding from martensite boundaries",
                expanded=False,
            ):
                jumps=measured["neighbor_disorientation"]
                st.caption(
                    "Blind inverse hypothesis: tests inter-region operators "
                    "using parent/product declared symmetries; never supplies "
                    "a unique measured OR or correspondence C."
                )
                if jumps.get("fraction_above_30_deg",0.)>0.25:
                    st.warning(
                        "More than 25% of indexed neighbor pairs show >30° "
                        "jumps. Pseudosymmetry or indexing errors may dominate "
                        "the inferred orientation regions."
                    )
                acknowledge=st.checkbox(
                    "I will interpret the result only as an exploratory "
                    "observable symmetry subgroup, not an experimental OR.",
                    key="tf_ctf_blind_ack",value=False,
                )
                if st.button(
                    "Search child/child operator compatibility",
                    key="tf_ctf_blind_run",
                    disabled=not acknowledge,
                ):
                    try:
                        from twin_app.ctf_blind_or import explore_blind_or
                        with st.spinner("Searching SO(3) observable classes…"):
                            blind=explore_blind_or(
                                measured,phases_declared,
                                parent_phase_id=phase_a,
                                product_phase_id=phase_m,
                            )
                        st.session_state["tf_ctf_blind_result"]=(
                            (independent_key,phase_a,phase_m),blind,
                        )
                    except (ValueError,ArithmeticError,RuntimeError) as exc:
                        st.error(f"Blind hypothesis search failed: {exc}")
                saved_blind=st.session_state.get("tf_ctf_blind_result")
                if (saved_blind and saved_blind[0]==
                        (independent_key,phase_a,phase_m)):
                    blind=saved_blind[1]
                    st.write(f"**Observable status:** {blind['status']}")
                    st.caption(blind.get("scientific_limit",blind.get("message","")))
                    st.json({
                        k:v for k,v in blind.items()
                        if k not in {"alternatives"}
                    })
                    st.download_button(
                        "Export exploratory OR evidence (.json)",
                        data=json.dumps(blind,indent=2,default=str),
                        file_name="ctf_blind_or_observable_class.json",
                        mime="application/json",
                        key="tf_ctf_blind_export",
                    )
    st.download_button(
        "Download complete CTF research package (.zip)",
        data=_zip_ctf_evidence(catalog, content, measured),
        file_name="ctf_complete_research_package.zip",
        mime="application/zip",
        key="tf_ctf_complete_bundle",
    )

    st.divider()
    st.markdown("**Compare with crystallographic theory (optional)**")
    has_or = st.checkbox(
        "I have an independently established parent-from-martensite OR",
        value=False, key="tf_ctf_v1_has_or",
        help="A CTF or CIF alone does not determine a unique transformation OR.",
    )
    if not has_or:
        st.caption("No OR is needed for measured EBSD maps and grain statistics.")
        return

    if not phases_declared:
        st.error("CTF lacks phase definitions needed to check the theory model.")
        return
    try:
        model_audit = compare_ctf_model_settings(
            phases_declared, project_payload, transformation_id,
            phase_a, phase_m,
        )
    except (ValueError, RuntimeError, ArithmeticError) as exc:
        st.error(f"Crystal-model check failed: {exc}")
        return
    if not model_audit["pass"]:
        st.error("CTF cells/point groups do not match the selected model's conventional settings.")
        with st.expander("Resolve phase and basis mismatches",expanded=False):
            st.dataframe(pd.DataFrame(model_audit["comparisons"]),
                         hide_index=True, width="stretch")
            st.caption(model_audit["scope"])

    matrix_txt = st.text_area(
        "Parent ← martensite OR (3×3 proper rotation)",
        value="", placeholder="r11 r12 r13\nr21 r22 r23\nr31 r32 r33",
        key="tf_ctf_v1_or",
        help=(
            "Matrix in corresponding orthonormal crystal Cartesian frames: "
            "x_parent=R x_martensite. Not the lattice correspondence C."
        ),
    )
    selected = ctf_analysis_signature(
        content, signature, matrix_txt, phase_a, phase_m, nmin,
        segmentation_threshold_deg=threshold,
        three_dimensional_angle_unit=unit,
    )
    confirm = st.checkbox(
        "I verified the phase IDs, indexing, crystal axes and OR frames.",
        value=False, key="tf_ctf_v1_confirm_" + selected[:20],
    )
    if report is None:
        st.caption("Calculate crystallographic modes to enable theory comparison.")
    ready = bool(
        report is not None and not invalid and model_audit["pass"]
        and matrix_txt.strip() and confirm
    )
    if st.button(
        "Compare EBSD with theory",
        key="tf_ctf_v1_analyze", disabled=not ready,
    ):
        try:
            R = parse_explicit_or(matrix_txt)
            with st.spinner("Reconstructing candidate parent domains and boundary operators…"):
                result = run_ctf_observation(
                    content,project_payload,transformation_id,
                    parent_phase_id=phase_a,product_phase_id=phase_m,
                    verified_or=R,filename=upload.name,
                    minimum_grain_points=nmin,
                    segmentation_threshold_deg=threshold,
                    three_dimensional_angle_unit=unit,
                )
            st.session_state["tf_ctf_v1_result"]=(selected,result)
        except (ValueError, RuntimeError, ArithmeticError, OSError) as exc:
            st.error(f"Experimental operator comparison failed: {exc}")
        except Exception as exc:
            st.error(f"Experimental operator comparison failed ({type(exc).__name__}): {exc}")

    saved = st.session_state.get("tf_ctf_v1_result")
    if (not saved or saved[0]!=selected or not ready):
        return
    result = saved[1]
    audit = summarize_twin_candidates(report,result)
    c1,c2,c3 = st.columns(3)
    c1.metric("Observed boundaries",audit["observed_boundaries"])
    c2.metric("Operator fits",audit["accepted_operator_fits"])
    c3.metric("Ambiguous",audit["ambiguous_operator_fits"])
    st.warning(
        "NOT CERTIFIED: operator fits alone do not identify physical Type I, "
        "Type II or Compound twins, nor a 3D habit plane."
    )
    with st.expander("EBSD comparison details",expanded=False):
        st.json(audit)
        st.json(result.summary)
        if result.boundaries:
            st.dataframe(pd.DataFrame(result.boundaries[:100]),
                         hide_index=True, width="stretch")
    st.download_button(
        "Download complete EBSD pipeline (.zip)",
        data=result.evidence_archive,
        file_name="ctf_ebsd_pipeline_all_outputs.zip",
        mime="application/zip",key="tf_ctf_v1_full_zip",
    )
    st.download_button(
        "Export EBSD/theory comparison (.json)",
        data=json.dumps({
            "source_sha256":result.sha256,"model_signature":signature,
            "CTF_parent_phase_id":phase_a,
            "CTF_martensite_phase_id":phase_m,
            "grain_segmentation_threshold_deg": threshold,
            "ctf_3d_euler_unit":unit,
            "verified_R_parent_from_product":parse_explicit_or(matrix_txt).tolist(),
            "model_consistency":model_audit,
            "audit":audit,"summary":result.summary,
            "boundaries":result.boundaries,
            "assignments":result.grain_assignments,
        },default=str,indent=2),
        file_name="ctf_orientation_evidence.json",
        mime="application/json",key="tf_ctf_v1_export",
    )
