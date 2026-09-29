from __future__ import annotations

"""Phase-2A research-workspace hardening.

This layer keeps every scientific backend frozen and changes only display
semantics, compact numerical presentation and pole-series geometric deduplication.
"""

from typing import Any, Mapping

import pandas as pd
import streamlit as st

import app.research_workspaces as rw
import app.research_workspaces_v4 as v4
import app.research_workspaces_v6 as v6
import app.scientific_interpretation_v7 as interp
from app.phase2_contracts import (
    compact_scalar_table,
    deduplicate_pole_entries,
    humanize_scientific_frame,
    physical_pole_label,
    sanitize_finding,
)
from app.scientific_ui import render_finding as _base_render_finding
from app.session_state import fingerprint, get_bound, get_bound_envelope, put_bound
from cualni_cryst.representation import CartesianConvention


# Professor-facing finding corrections.
v4.am_existence_finding = interp.am_existence_finding
v4.mm_audit_finding = interp.mm_audit_finding
v4.ptmc_finding = interp.ptmc_finding
v4.or_comparison_finding = interp.or_comparison_finding
v4.cofactor_finding = interp.cofactor_finding
v4.supercompatibility_finding = interp.supercompatibility_finding
v4.experiment_finding = interp.experiment_finding


def _render_finding_phase2(finding: Any, *args: Any, **kwargs: Any) -> None:
    _base_render_finding(sanitize_finding(finding), *args, **kwargs)


v4.render_finding = _render_finding_phase2


_original_frame_formatter = v4.format_dataframe_scientific


def _format_dataframe_phase2(frame: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
    return humanize_scientific_frame(_original_frame_formatter(frame, *args, **kwargs))


v4.format_dataframe_scientific = _format_dataframe_phase2
rw._scalar_table = compact_scalar_table


def _unique_labels(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for entry in entries:
        base = str(entry["display_label"])
        counts[base] = counts.get(base, 0) + 1
        label = base if counts[base] == 1 else f"{base} · geometry {counts[base]}"
        labels[label] = entry
    return labels


def _render_poles(project: Any, transformation_id: str) -> None:
    v4.section_header(
        "Pole figures",
        "Do selected predicted planes/directions overlap geometrically in one explicit reference frame?",
        answer_hint=(
            "Native generator/sign branches that generate the same projective pole family are grouped into one plotted series. "
            "Their native labels remain visible in provenance."
        ),
    )
    unified = rw._current_unified_report()
    if unified is None:
        st.info("Calculate the unified theory comparison first.")
        return

    transformation = project.transformation(transformation_id)
    parent_id = transformation.parent_phase_id
    product_id = transformation.product_phase_id
    mode = st.radio(
        "Pole analysis",
        ["Predicted interfaces / twins", "OR pole overlay"],
        horizontal=True,
        key="research_v7_pole_mode",
    )
    entries: list[dict[str, Any]] = []
    signature_parts: list[Any] = [
        st.session_state.get("research_current_unified_signature"),
        transformation_id,
        mode,
    ]

    if mode == "Predicted interfaces / twins":
        object_kind = st.selectbox(
            "Physical object",
            ["A/M habit plane", "Twin plane", "Twin direction"],
            key="research_v7_pole_kind",
        )
        for row in unified.rows:
            coeffs, kind = rw._row_object(row, object_kind)
            if coeffs is None:
                continue
            source = str(getattr(getattr(row, "theory", None), "value", getattr(row, "theory", "")))
            display = physical_pole_label(row)
            family = rw.pole_family(
                project,
                parent_id,
                kind=kind,
                coefficients=coeffs,
                reference_phase_id=parent_id,
                projective=True,
                label=display,
                source=source,
            )
            entries.append(
                {
                    "family": family,
                    "row": row,
                    "display_label": display,
                    "native_label": str(getattr(row, "branch_label", "")),
                    "kind": kind,
                    "phase_id": parent_id,
                    "source": source,
                    "provenance": "no OR applied; object is expressed natively in the parent crystal/reference frame",
                }
            )
        signature_parts.append(object_kind)
    else:
        c1, c2 = st.columns(2)
        pole_kind = c1.radio(
            "Product object type",
            ["direction", "plane"],
            horizontal=True,
            key="research_v7_or_pole_kind",
        )
        seed_text = c2.text_input(
            "Product indices",
            value="0 0 1",
            key="research_v7_or_pole_seed",
        )
        try:
            seed = rw._parse_vector(seed_text, name="product pole")
        except ValueError as exc:
            st.error(str(exc))
            return
        for row in unified.rows:
            if getattr(row, "or_parent_from_product", None) is None:
                continue
            R = rw._reexpress_or(
                project,
                transformation_id,
                rw.np.asarray(row.or_parent_from_product, dtype=float),
                source=CartesianConvention.PTCLAB_A_X_C_XZ,
                target=CartesianConvention.SYMMETRIC_METRIC,
            )
            source = str(getattr(getattr(row, "theory", None), "value", getattr(row, "theory", "")))
            display = physical_pole_label(row)
            family = rw.pole_family(
                project,
                product_id,
                kind=pole_kind,
                coefficients=seed,
                reference_phase_id=parent_id,
                R_reference_from_phase=R,
                projective=True,
                label=display,
                source=source,
            )
            entries.append(
                {
                    "family": family,
                    "row": row,
                    "display_label": display,
                    "native_label": str(getattr(row, "branch_label", "")),
                    "kind": pole_kind,
                    "phase_id": product_id,
                    "source": source,
                    "provenance": str(getattr(row, "provenance", "") or source),
                }
            )
        signature_parts += [pole_kind, seed_text]

    grouped = deduplicate_pole_entries(entries)
    options = _unique_labels(grouped)
    selected = st.multiselect(
        "Physical/geometric pole series",
        options=list(options),
        default=[],
        max_selections=12,
        key="research_v7_pole_branches",
        help=(
            "Selections are deduplicated by the actual projective pole family. Native generator/sign branches remain in the provenance table."
        ),
    )
    signature_parts.append(selected)
    signature = fingerprint("pole-figure-v7", *signature_parts)
    envelope = get_bound_envelope(st.session_state, "research_pole_report_v7")
    if envelope is not None and envelope.signature != signature:
        st.warning("Pole settings changed — regenerate the figure. The previous figure is retained but is not displayed as current.")

    if st.button(
        "Generate / update pole figure",
        type="primary",
        disabled=not selected,
        use_container_width=True,
        key="research_v7_pole_generate",
    ):
        families = []
        metadata = []
        for option in selected:
            entry = options[option]
            family = entry["family"]
            # Make the plotted label exactly the human physical/geometric option.
            try:
                family = type(family)(
                    phase_id=family.phase_id,
                    kind=family.kind,
                    label=option,
                    source=family.source,
                    projective=family.projective,
                    points=family.points,
                )
            except Exception:
                pass
            families.append(family)
            row = entry["row"]
            exact = getattr(row, "exact", None)
            metadata.append(
                {
                    "series": f"{entry['source']} · {option}",
                    "phase": entry["phase_id"],
                    "object type": entry["kind"],
                    "reference frame": f"{parent_id} Cartesian reference frame",
                    "projective convention": "unoriented pole; v ~ -v; upper hemisphere",
                    "scientific status": "exact" if exact is True else "approximate diagnostic" if exact is False else "status not encoded",
                    "theory source": entry["source"],
                    "OR provenance": entry["provenance"],
                    "native branch count": len(entry.get("native_labels", [])),
                    "native branches": " | ".join(entry.get("native_labels", [])),
                }
            )
        meta_map = {item["series"]: item for item in metadata}
        rows = v6._pole_rows(families, reference_phase_id=parent_id, metadata_by_series=meta_map)
        put_bound(
            st.session_state,
            "research_pole_report_v7",
            signature,
            {
                "rows": rows,
                "mode": mode,
                "series_metadata": metadata,
                "reference_phase_id": parent_id,
            },
        )

    payload = get_bound(st.session_state, "research_pole_report_v7", signature)
    if not isinstance(payload, Mapping):
        return
    rows = list(payload["rows"])
    frame = pd.DataFrame(rows)
    if frame.empty:
        st.info("No pole coordinates were produced for the selected series.")
        return

    st.altair_chart(rw._stereographic_chart(frame), use_container_width=False)
    best, pair = v6.minimum_projective_separation_deg(rows)
    st.markdown("### Geometric overlap check")
    if best is None or pair is None:
        st.caption("At least two distinct plotted series are required for an inter-series separation.")
    else:
        st.write(
            f"Closest projective pole separation: **{best:.6g}°** between **{pair[0]}** and **{pair[1]}**."
        )
        st.caption(
            "A small angular separation shows geometric overlap in this reference frame. It does not prove identical derivation, mechanism, branch identity or theory equivalence."
        )

    metadata = list(payload.get("series_metadata", []))
    if metadata:
        st.markdown("### Pole-series provenance")
        st.dataframe(pd.DataFrame(metadata), hide_index=True, use_container_width=True)
    with st.expander("Pole coordinates / provenance", expanded=False):
        st.dataframe(frame, hide_index=True, use_container_width=True)
        st.caption("Projective antipodes are one unoriented pole. Native branch multiplicity is retained in provenance rather than plotted as duplicate geometry.")
    st.download_button(
        "Pole coordinates CSV",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name="cualni_ct_poles.csv",
        mime="text/csv",
        key="research_v7_pole_csv",
    )


v4._render_poles = _render_poles


def render_research_extension() -> None:
    v6.restore_persistent_widget_state(st.session_state)
    try:
        v4.render_research_extension()
    finally:
        v6.snapshot_persistent_widget_state(st.session_state)
