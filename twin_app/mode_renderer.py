from __future__ import annotations

"""Accessible, progressive Mode → verified Type → twin → habit presentation.

All numerical and classification values come from the already-calculated
scientific report. Mode letters are display ordinals only; they are NEVER
silently equated with a published paper's mode labels.
"""

from contextlib import nullcontext
from typing import Iterable

import streamlit as st

from .mode_catalogue import ModeCatalogue
from .mode_tree_component import build_tree_model, render_mode_tree
from .mode_view_model import ModeSummary, ModeView, build_mode_view, entry_caption
from .scientific_models import TwinFamilyReport


def _value(x: float, digits: int = 9) -> str:
    return f"{float(x):.{digits}g}"


def _vec(values: Iterable[float], *, plane: bool = False) -> str:
    text = ", ".join(_value(v) for v in values)
    return f"({text})" if plane else f"[{text}]"


def _select_source(mode: ModeSummary, twin_type) -> object:
    entries = twin_type.entries
    st.caption(f"Showing an actual calculated source: **{entry_caption(entries[0])}**. "
               "The first source is chosen for convenience, never as an experimental prediction.")
    if len(entries) < 2:
        return entries[0]
    expanded = st.checkbox(
        f"Browse other equivalent sources ({len(entries) - 1} additional records)",
        key=f"ct_ux_advanced_source_{mode.mode_id}_{twin_type.label}",
        help="Sources are symmetry-related descriptions; viewing another does not create a new mode.",
    )
    if not expanded:
        return entries[0]
    option = st.selectbox(
        "Choose a computed variant pair and twin branch",
        options=tuple(range(len(entries))),
        format_func=lambda i: ("Default representative · " if i == 0 else "") + entry_caption(entries[i]),
        key=f"ct_ux_source_{mode.mode_id}_{twin_type.label}",
    )
    return entries[option]


def _show_habits(entry: object, *, selection_key: str) -> None:
    """Make the selected-branch numerical conclusion legible without hiding evidence."""
    twin = entry.construction
    st.markdown("#### 4 · Austenite–martensite habit-plane compatibility")
    st.caption(
        "Scope: this selected calculated twin **source branch** and its entered "
        "lattice parameters. Equivalent sources and other modes may give different results."
    )
    if twin.continuum_fraction:
        st.markdown("**A/M result: Continuous compatible laminate-fraction family.**")
        st.write(
            "The solver returns a continuum of compatible laminate fractions. "
            "It does not select one unique habit orientation."
        )
        st.caption(
            "Scientific interpretation: a predicted continuum is not a measured "
            "variant fraction or an experimentally confirmed habit plane."
        )
        return
    habits = tuple(twin.habit_solutions)
    if not habits:
        st.markdown("**A/M result: No exact habit-plane solution found for this branch.**")
        st.write(
            "**Interpretation:** The M/M twin interface calculated above remains "
            "a separate mathematical result. For this branch and these lattice "
            "parameters, the twinned-laminate calculation returned no exact "
            "compatible parent–martensite habit plane."
        )
        st.caption(
            "**Thesis-ready statement:** No exact A/M habit-plane solution "
            "was obtained for the selected calculated twin branch under the "
            "entered crystallographic parameters and numerical criteria."
        )
        st.caption(
            "This does not exclude the calculated M/M twin, other twin branches "
            "or the transformation itself; nor does it establish that every "
            "physically possible habit plane is absent."
        )
        return
    st.markdown(
        f"**A/M result: {len(habits)} exact numerical habit-plane record(s) "
        "for this selected twin branch.**"
    )
    st.caption(
        "These are candidate parent–martensite interfaces of a twinned laminate, "
        "not the M/M twin plane or necessarily distinct observed habit planes."
    )
    chosen = 0
    if len(habits) > 1:
        chosen = st.radio(
            "Choose a habit solution",
            options=tuple(range(len(habits))),
            format_func=lambda i: f"Habit {i + 1} · λ = {_value(habits[i].other_variant_volume_fraction)} "
                                  f"· branch {habits[i].habit_branch:+d}",
            key=f"ct_ux_habit_{selection_key}",
            horizontal=len(habits) <= 3,
        )
    habit = habits[chosen]
    a, b, c = st.columns(3)
    a.metric("Other-variant fraction · λ", _value(habit.other_variant_volume_fraction))
    st.caption(
        "λ is the model-predicted laminate fraction, not a measured specimen fraction."
    )
    with b:
        st.markdown("**A/M habit-plane normal · m**")
        st.code(_vec(habit.habit_normal_parent_cartesian), language="text")
    with c:
        st.markdown("**Shape-strain vector · b**")
        st.code(_vec(habit.shape_vector_parent_cartesian), language="text")
    st.caption("m and b use the **parent Cartesian frame**. m is unit length; b contains magnitude and direction. "
               "These are not Miller indices.")
    with st.expander("Crystal-coordinate vectors and numerical compatibility checks", expanded=False):
        st.write(f"Parent reciprocal habit-plane coordinates: {_vec(habit.habit_plane_parent_crystal, plane=True)}")
        st.write(f"Parent direct shape-vector coordinates: {_vec(habit.shape_vector_parent_crystal)}")
        st.code(
            f"Rank-one residual       {habit.rank_one_residual:.3e}\n"
            f"Rotation residual       {habit.rotation_residual:.3e}\n"
            f"Middle-stretch residual {habit.middle_stretch_residual:.3e}\n"
            f"Plane frame residual    {habit.frame_plane_residual:.3e}\n"
            f"Vector frame residual   {habit.frame_shape_vector_residual:.3e}",
            language="text",
        )


def _show_audit(view: ModeView, catalogue: ModeCatalogue, report: TwinFamilyReport, *, inline: bool = False) -> None:
    """Preserve the complete audit, optionally displayed inside the third tab."""
    context = nullcontext() if inline else st.expander(
        "Unresolved records, incompatible pairs and provenance", expanded=False
    )
    with context:
        st.markdown("**What the main navigator does not show**")
        st.write(f"Unresolved rank-one source records: {view.unresolved_record_count}")
        st.write(f"Mode groups containing only unresolved classifications: {len(view.diagnostic_modes)}")
        st.write(f"Non-twinning stretch-pair orbits: {len(catalogue.incompatible_pair_orbits)}")
        st.write(f"All computed twin source records retained: {view.all_source_record_count}")
        st.write(f"All correspondence couples: {catalogue.correspondence_pair_count}")
        st.write(f"Correspondence operator classes (not modes): {report.audit.operator_count}")
        if view.unresolved_record_count:
            st.warning("Some calculated rank-one geometries lack direct cross-lock or proved symmetry-transport "
                       "support for Type I, Type II or Compound. They are not promoted to verified choices.")
        st.caption("The complete provenance remains in the exported scientific report. "
                   "Expand here only when checking a specific discrepancy.")
        if st.checkbox("Inspect all unresolved records and correspondence mappings",
                       value=False, key="ct_ux_full_audit_records"):
            for mode in tuple(view.visible_modes) + tuple(view.diagnostic_modes):
                if not mode.unresolved_entries:
                    continue
                st.markdown(f"**{mode.display_name} — {mode.mode_id}**: {len(mode.unresolved_entries)} unresolved record(s)")
                for entry in mode.unresolved_entries:
                    st.caption(f"{entry_caption(entry)} · s={_value(entry.construction.shear_magnitude)} · "
                               f"{entry.construction.classification_status}")
            for mode in view.diagnostic_modes:
                st.caption(f"{mode.mode_id}: no independently verified type; omitted from the primary picker.")
            st.markdown("**Correspondence-to-stretch mapping**")
            for record in report.correspondence_variants:
                mapped = f"U{record.stretch_variant_index}" if record.stretch_variant_index is not None else "unmapped"
                st.write(f"M{record.variant_index} → {mapped} · {record.mapping_status}")
            for warning in report.audit.warnings:
                st.warning(warning)


def render_mode_report(catalogue: ModeCatalogue, report: TwinFamilyReport) -> None:
    """Render a real, connected, interactive mode → verified-type tree.

    Selection is the only state crossing the custom-component boundary.
    Source tensors, classification and habit results remain the scientific
    report's responsibility, never values supplied by the browser.
    """
    view = build_mode_view(catalogue)
    st.header("Calculated physical twinning modes")
    st.caption(
        "Select a node in the connected tree. Blue = calculated mode; "
        "teal = Type I; violet = Type II; amber = Compound. "
        "Only directly cross-locked or proved symmetry-transported types are shown. "
        "Mode letters are display labels, not literature assignments. "
        "Pair counts measure classification coverage within one symmetry orbit, "
        "not distinct measured twin interfaces."
    )
    st.caption(
        f"{len(view.visible_modes)} calculated twinning modes · "
        f"{catalogue.mapped_stretch_count} distinct stretch variants. "
        "Unclassified records and incompatible symmetry orbits are retained "
        "in the expandable scientific audit."
    )
    if not view.visible_modes:
        st.info("No Type I, Type II or Compound twinning mode supported by direct cross-lock "
                "or proved symmetry transport was identified for these inputs. No type is invented.")
        _show_audit(view, catalogue, report)
        return

    # Invalidate selection on a changed scientific report, even when its visible
    # mode letters or counts happen to be identical. Include source identities.
    fingerprint = repr(tuple(
        (mode.mode_id, round(mode.shear, 14), mode.symmetry_pairs,
         tuple((item.label, item.source_records, item.stretch_couples,
                tuple((entry_caption(e), e.construction.classification,
                       e.construction.classification_status,
                       tuple(e.construction.twin_plane_product_crystal),
                       tuple(e.construction.shear_direction_product_crystal),
                       tuple((h.other_variant_volume_fraction,
                              tuple(h.habit_normal_parent_cartesian),
                              tuple(h.shape_vector_parent_cartesian))
                             for h in e.construction.habit_solutions))
                      for e in item.entries)) for item in mode.types))
        for mode in view.visible_modes
    ))
    import hashlib
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    lookup = {mode.mode_id: mode for mode in view.visible_modes}
    if st.session_state.get("ct_tree_science_digest") != digest:
        st.session_state["ct_tree_science_digest"] = digest
        st.session_state["ct_ux_mode_selection"] = view.visible_modes[0].mode_id
        for key in list(st.session_state):
            if key.startswith(("ct_ux_type_", "ct_ux_source_", "ct_ux_habit_", "ct_ux_advanced_source_")):
                st.session_state.pop(key, None)

    mode_id = st.session_state.get("ct_ux_mode_selection", view.visible_modes[0].mode_id)
    if mode_id not in lookup:
        mode_id = view.visible_modes[0].mode_id
    selected_mode = lookup[mode_id]
    type_key = f"ct_ux_type_{mode_id}"
    available = {t.label: t for t in selected_mode.types}
    selected_type = st.session_state.get(type_key)
    if selected_type not in available:
        selected_type = selected_mode.types[0].label
    active = (mode_id, selected_type)

    # The frontend receives only a small, verified set of display metadata;
    # it can REQUEST (mode_id, type), but cannot create scientific output.
    graph_model = build_tree_model(view.visible_modes)
    new_active = render_mode_tree(graph_model, selection=active,
                                  key=f"ct_real_physical_mode_tree_{digest}")
    if new_active is not None and new_active != active:
        st.session_state["ct_ux_mode_selection"] = new_active[0]
        st.session_state[f"ct_ux_type_{new_active[0]}"] = new_active[1]
        mode_id, selected_type = new_active
        selected_mode = lookup[mode_id]
    twin_type = next(item for item in selected_mode.types if item.label == selected_type)

    st.divider()
    st.markdown(f"### {selected_mode.display_name} → {twin_type.label} · full numerical result")
    st.caption(
        f"Calculated twin shear magnitude: s = {_value(selected_mode.shear, 8)}. "
        "The displayed type refers to the selected mode; inspect the geometry and "
        "independent classification evidence below."
    )
    # Verify the numerator against the actual U_i–U_j identities at presentation
    # time. Different twin types may classify overlapping pairs, so never sum
    # these pair counts as if they were disjoint physical interfaces.
    unique_pairs = tuple(sorted({
        tuple(sorted((int(e.pair.stretch_i), int(e.pair.stretch_j))))
        for e in twin_type.entries
    }))
    if len(unique_pairs) != twin_type.stretch_couples:
        raise ValueError("Selected type pair count disagrees with calculated U-pair identities")
    with st.expander("Scientific audit: which variant pairs belong to this mode?", expanded=False):
        st.markdown(
            f"**{twin_type.stretch_couples} of {selected_mode.symmetry_pairs} pairs** "
            f"in {selected_mode.display_name} have a supported {twin_type.label} classification."
        )
        st.write(
            "A stretch variant Uᵢ is one calculated positive-definite transformation "
            "stretch tensor. A pair Uᵢ–Uⱼ represents two distinct stretch variants "
            "being compared for compatibility. The two orders Uᵢ–Uⱼ and Uⱼ–Uᵢ "
            "count once. Applying parent point-group symmetry generates an orbit "
            "of equivalent unordered pairs. Its size is the denominator. "
            "The numerator counts pairs whose indicated Type I, Type II, or "
            "Compound classification is independently cross-checked or rigorously "
            "transported by symmetry. A pair may carry more than one classification; "
            "therefore counts across Type I and Type II must not be added. "
            "These are neither measured twins nor distinct physical samples."
        )
        st.code(
            "  ·  ".join(f"U{a}–U{b}" for a, b in unique_pairs),
            language="text",
        )
        st.caption(
            f"{len(twin_type.entries)} underlying computed source records support "
            f"{len(unique_pairs)} distinct stretch pairs for this type. "
            "The expanded scientific audit retains the precise construction IDs, "
            "correspondence variants and classification evidence."
        )
    if twin_type.stretch_couples < selected_mode.symmetry_pairs:
        st.warning(
            f"Some symmetry-related pairs have no verified {twin_type.label} classification. "
            "The remaining pairs are unclassified for this type, not necessarily "
            "incompatible or untwinnable; inspect the audit before drawing conclusions."
        )
    entry = _select_source(selected_mode, twin_type)
    twin, pair = entry.construction, entry.pair
    st.markdown("#### Calculated martensite–martensite twin interface")
    st.caption(f"Source: {entry_caption(entry)}")
    x, y, z = st.columns(3)
    x.metric("Physical twin shear · s", _value(twin.shear_magnitude, 8))
    with y:
        st.markdown("**Twin plane · K₁**")
        st.code(_vec(twin.twin_plane_product_crystal, plane=True), language="text")
    with z:
        st.markdown("**Shear direction · η₁**")
        st.code(_vec(twin.shear_direction_product_crystal), language="text")
    st.caption("K₁ is in daughter reciprocal-crystal coordinates; η₁ is in daughter "
               "direct-crystal coordinates. Compare after accounting for symmetry, "
               "conventional settings and published rounding.")
    _show_habits(entry, selection_key=f"{mode_id}_{selected_type}_{entry.operator_family}_"
                                    f"{pair.variant_i}_{pair.variant_j}_{twin.branch}")
    from .p1_ui import render_p1_source_evidence
    with st.expander("Verify this result and inspect its scientific evidence", expanded=False):
        numerical_tab, sources_tab, unresolved_tab = st.tabs((
            "Numerical checks", "Variant pairs and sources", "Unresolved records"
        ))
        with numerical_tab:
            st.caption(
                "Numerical residuals test equations, not whether a physical "
                "interface was observed. All source-level evidence is retained."
            )
            render_p1_source_evidence(entry, report, inline=True)
            st.markdown("**Independent checks for the selected M/M interface**")
            st.write(f"Rank-one residual: {twin.rank_one_residual:.3e}")
            st.write(f"Proper-rotation residual: {twin.rotation_residual:.3e}")
            if twin.independent_outer_product_residual is not None:
                st.write(f"Independent tensor residual: {twin.independent_outer_product_residual:.3e}")
            if twin.independent_shear_relative_residual is not None:
                st.write(f"Independent shear relative residual: {twin.independent_shear_relative_residual:.3e}")
        with sources_tab:
            st.caption(
                "Classification and source records supporting the displayed twin type. "
                "Symmetry-related source records are not distinct observed twins."
            )
            st.write("Classification evidence:", twin.classification_status)
            st.write("CT classical system IDs:", ", ".join(twin.classical_system_ids) or "none")
            for i, source in enumerate(twin_type.entries, start=1):
                st.write(f"{i}. {entry_caption(source)} · s={_value(source.construction.shear_magnitude)} · "
                         f"{len(source.construction.habit_solutions)} recorded habit branches")
        with unresolved_tab:
            _show_audit(view, catalogue, report, inline=True)
