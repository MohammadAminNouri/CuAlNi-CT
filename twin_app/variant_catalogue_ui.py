from __future__ import annotations

"""Read-only variant catalogue built from calculated twin-family report.

A correspondence variant M_i is not the same object as a distinct stretch U_j.
Variant counts come ONLY from the engine's immutable report. Pair and
construction records are deduplicated by stable report IDs, not interpreted
as experimentally observed twins or variants.
"""

from collections import defaultdict
from typing import Any


def build_variant_catalogue(report: Any) -> dict[str, Any]:
    records = tuple(report.correspondence_variants)
    total = int(report.audit.topology_variant_count)
    if len(records) != total:
        raise ValueError("Variant inventory must match the computed symmetry topology")
    identities = [int(x.variant_index) for x in records]
    if len(set(identities)) != total or any(i < 1 for i in identities):
        raise ValueError("Duplicate or invalid crystallographic variant identity")

    involvement: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"partners": set(), "pairs": set(), "constructions": set(),
                 "types": set(), "habits": set(), "U_partners": set(),
                 "unresolved": set()}
    )
    unique_pairs = set()
    all_constructions = set()
    for family in report.families:
        for pair in family.pair_records:
            i, j = int(pair.variant_i), int(pair.variant_j)
            if i not in identities or j not in identities or i == j:
                raise ValueError("Variant pair references an unavailable or identical variant")
            pair_key = tuple(sorted((i, j)))
            unique_pairs.add(pair_key)
            for v, other, other_U in ((i, j, pair.stretch_j),
                                      (j, i, pair.stretch_i)):
                row = involvement[v]
                row["partners"].add(other)
                row["pairs"].add(pair_key)
                if other_U is not None:
                    row["U_partners"].add(int(other_U))
                if not pair.constructions:
                    row["unresolved"].add(pair_key)
            for c in pair.constructions:
                cid = (pair_key, str(c.construction_id))
                all_constructions.add(cid)
                for v in (i, j):
                    item = involvement[v]
                    item["constructions"].add(cid)
                    label = str(c.classification)
                    if label in {"Type I", "Type II", "Compound"}:
                        item["types"].add(label)
                    else:
                        item["unresolved"].add(cid)
                    for habit in c.habit_solutions:
                        item["habits"].add(
                            (cid, int(habit.habit_branch),
                             round(float(habit.other_variant_volume_fraction), 12),
                             tuple(round(float(t), 10) for t in habit.habit_normal_parent_cartesian))
                        )
    rows = []
    for r in sorted(records, key=lambda x: int(x.variant_index)):
        idx = int(r.variant_index)
        v = involvement[idx]
        rows.append({
            "variant_index": idx,
            "label": f"M{idx}",
            "stretch_variant_index": (
                None if r.stretch_variant_index is None else int(r.stretch_variant_index)
            ),
            "stretch_label": (
                "unmapped" if r.stretch_variant_index is None
                else f"U{int(r.stretch_variant_index)}"
            ),
            "mapping_status": str(r.mapping_status),
            "stretch_mapping_residual": (
                None if r.stretch_mapping_residual is None
                else float(r.stretch_mapping_residual)
            ),
            "parent_symmetry_determinant": int(r.representative_determinant),
            "parent_symmetry_matrix": r.representative_parent_symmetry,
            "correspondence_M_from_A": r.correspondence_M_from_A,
            "partners": tuple(sorted(v["partners"])),
            "partner_stretches": tuple(sorted(v["U_partners"])),
            "unordered_correspondence_pairs": len(v["pairs"]),
            "twin_construction_records": len(v["constructions"]),
            "types": tuple(sorted(v["types"])),
            "habit_solution_records": len(v["habits"]),
            "unresolved_record_count": len(v["unresolved"]),
        })
    mapped = sum(row["stretch_variant_index"] is not None for row in rows)
    distinct = len({row["stretch_variant_index"] for row in rows
                    if row["stretch_variant_index"] is not None})
    if mapped != int(report.audit.mapped_correspondence_variant_count):
        raise ValueError("Mapped variant count disagrees with engine audit")
    if mapped-distinct != int(report.audit.collapsed_correspondence_variant_count):
        raise ValueError("Variant stretch-collapsing count disagrees with engine audit")
    return {
        "transformation_id": report.transformation_id,
        "total_correspondence_variants": total,
        "distinct_stretch_variants": int(report.audit.stretch_variant_count),
        "mapped_correspondence_variants": mapped,
        "collapsed_correspondence_variants": mapped-distinct,
        "unmapped_correspondence_variants": total-mapped,
        "unique_unordered_correspondence_pairs": len(unique_pairs),
        "twin_construction_records": len(all_constructions),
        "rows": rows,
        "scope": (
            "M = correspondence variants derived from crystallographic symmetry; "
            "U = distinct stretch tensors; one U may receive multiple M labels. "
            "Construction counts are computational records, NOT distinct observed twins. "
            "No EBSD observation or population is inferred."
        ),
    }


def render_variant_catalogue(report: Any) -> None:
    import json
    import pandas as pd
    import streamlit as st

    catalogue = build_variant_catalogue(report)
    st.subheader("03  /  All calculated martensite variants")
    a,b,c = st.columns(3)
    a.metric("Correspondence variants · M", catalogue["total_correspondence_variants"])
    b.metric("Distinct stretch variants · U", catalogue["distinct_stretch_variants"])
    c.metric("Unmapped correspondence variants", catalogue["unmapped_correspondence_variants"])
    st.caption(catalogue["scope"])
    # Cayron (Crystals 2022), Sec. 3.3 and Table 2: variants are
    # simple cosets, whereas operators classifying variant pairs are double
    # cosets. Never identify these two counts or an arbitrary operator index
    # with the author's GenOVa O0–O6 labels.
    group_order = int(report.audit.correspondence_subgroup_order) * (
        int(catalogue["total_correspondence_variants"])
    )
    total_possible_pairs = (
        catalogue["total_correspondence_variants"]
        * (catalogue["total_correspondence_variants"]-1) // 2
    )
    st.markdown("**2022 Correspondence Theory · complete symmetry census**")
    st.dataframe(pd.DataFrame([
        {"Quantity":"Implied parent symmetry-group order |G_A|",
         "Calculated":group_order,
         "Equation":"|G_A| = |H_C| × N_M (Lagrange)"},
        {"Quantity":"Correspondence subgroup order |H_C|",
         "Calculated":report.audit.correspondence_subgroup_order,
         "Equation":"Common correspondence symmetries"},
        {"Quantity":"Correspondence variants N_M",
         "Calculated":catalogue["total_correspondence_variants"],
         "Equation":"Left cosets G_A/H_C"},
        {"Quantity":"Distinct inter-variant operators",
         "Calculated":report.audit.operator_count,
         "Equation":"Double cosets H_C\\G_A/H_C"},
        {"Quantity":"Possible unordered M_i–M_j pairs",
         "Calculated":total_possible_pairs,
         "Equation":"N_M(N_M−1)/2 (not all must be exact twins)"},
        {"Quantity":"Pairs represented in output",
         "Calculated":catalogue["unique_unordered_correspondence_pairs"],
         "Equation":"Computed pairs, deduplicated by M_i/M_j"},
    ]),hide_index=True,width="stretch")
    st.caption(
        "For Cayron's published NiTi B2→B19′ case (Crystals 2022, "
        "Tables 1–2): |G_A|=48, |H_C|=4, 12 correspondence variants "
        "and 7 operators. These are REFERENCE numbers, not assumed "
        "to apply to your Cu–Al–Ni phase pair. Published O0–O6 labels "
        "depend on GenOVa numbering and cannot be mapped by index alone."
    )
    if catalogue["collapsed_correspondence_variants"]:
        st.info(
            f"{catalogue['collapsed_correspondence_variants']} correspondence variant(s) "
            "map to stretch tensors already represented by another M variant. "
            "Do not add M and U counts together."
        )
    rows = catalogue["rows"]
    summary = [{
        "M variant": x["label"], "Stretch": x["stretch_label"],
        "Mapping": x["mapping_status"],
        "Partner M variants": ", ".join(f"M{p}" for p in x["partners"]) or "none",
        "Twin construction records": x["twin_construction_records"],
        "Verified types in records": ", ".join(x["types"]) or "none classified",
        "Habit solution records": x["habit_solution_records"],
        "Unresolved records": x["unresolved_record_count"],
    } for x in rows]
    st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
    if not rows:
        return
    selected = st.selectbox(
        "Inspect an individual correspondence variant",
        range(len(rows)),
        format_func=lambda i: (
            f"M{rows[i]['variant_index']} → {rows[i]['stretch_label']} · "
            f"{len(rows[i]['partners'])} computed partner(s)"
        ),
        key="tf_all_variants_selected",
    )
    detail = rows[selected]
    st.markdown(f"**{detail['label']} · {detail['stretch_label']}**")
    st.write(
        f"Computed partner variants: "
        + (", ".join(f"M{k}" for k in detail["partners"]) or "none")
    )
    st.write(f"Mapping: {detail['mapping_status']}")
    if detail["stretch_mapping_residual"] is not None:
        st.caption(
            "Correspondence-to-stretch mapping residual: "
            f"{detail['stretch_mapping_residual']:.6g}"
        )
    x, y = st.columns(2)
    with x:
        st.markdown("**Parent-symmetry representative**")
        st.code("\n".join("  ".join(row) for row in detail["parent_symmetry_matrix"]))
        st.caption(
            f"Determinant = {detail['parent_symmetry_determinant']}. "
            "Representative symmetry operation, not an experimental orientation relationship."
        )
    with y:
        st.markdown("**Parent → product correspondence matrix**")
        st.code("\n".join("  ".join(row) for row in detail["correspondence_M_from_A"]))
        st.caption("The reported C matrix is a crystallographic correspondence, not an EBSD-derived OR.")
    st.download_button(
        "Export complete variant inventory (JSON)",
        data=json.dumps(catalogue, indent=2, default=str),
        file_name="calculated_correspondence_and_stretch_variants.json",
        mime="application/json",
        key="tf_all_variant_export",
    )
