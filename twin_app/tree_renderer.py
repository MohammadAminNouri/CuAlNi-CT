from __future__ import annotations

"""One complete crystallographic family tree and one stable interpretation pane.

The full family/couple graph is the default, never an arbitrarily selected
family. All outputs come from the active scientific report, not benchmark data.
"""

from typing import Iterable

import streamlit as st

from .family_tree_graph import CoupleNode, normalized_rank_one, plot_family_tree, tree_layout
from .scientific_models import PairTwinConstruction, TwinFamilyReport, TwinFamilyRecord
from .ux_language import twin_name, representative_habit_solutions


def _fmt(value: float, precision: int = 5) -> str:
    v = float(value)
    return "0" if abs(v) < 5e-10 else (f"{v:.3e}" if abs(v) < 1e-4 or abs(v) >= 1e5 else f"{v:.{precision}g}")


def _vec(values: Iterable[float], *, plane: bool = False, precision: int = 6) -> str:
    parts = ", ".join(_fmt(v, precision) for v in values)
    return f"({parts})" if plane else f"[{parts}]"


def _habit_count(node: CoupleNode) -> int:
    return sum(len(c.habit_solutions) for c in node.pair.constructions) if node.pair else 0


def _status_for(node: CoupleNode) -> str:
    if node.pair is None:
        return "Geometry not evaluated"
    if _habit_count(node):
        return "Compatible A/M interface"
    if any(c.continuum_fraction for c in node.pair.constructions):
        return "Continuous compatibility"
    if node.pair.constructions:
        return "No exact A/M interface"
    return "No exact classical twin"


def _selected_details(node: CoupleNode) -> None:
    st.divider()
    st.subheader(f"Selected variant couple: {node.label}")
    st.markdown(
        f"**Family {node.family.family_id}** · {_status_for(node)}.  "
        "The M numbers are calculated correspondence-variant IDs—not Bhattacharya's published mode numbering."
    )
    pair = node.pair
    if node.family.route == "axial_weak":
        st.markdown("**Higher-order symmetry relation · potential weak twin**")
        st.write("This is a symmetry relationship, not automatically an exact classical twin or austenite–martensite habit plane.")
        if node.family.weak_candidates:
            for item in node.family.weak_candidates:
                st.write(f"Rotation order {item.parent_rotation_order} · candidate weak planes {_vec(item.plane1_primitive, plane=True)} and {_vec(item.plane2_primitive, plane=True)}")
                st.caption(f"Calculated generalized shear: {_fmt(item.generalized_shear)}")
        else:
            st.info("Weak-plane indices were not determined. A product primitive-node basis may be required.")
        return
    if pair is None:
        st.info("A correspondence relation exists, but no pair-specific twin geometry is available.")
        return
    if not pair.constructions:
        st.info("No exact classical twin was found for this pair with the current lattice input.")
        return

    options = tuple(range(len(pair.constructions)))
    key = f"tf_branch_{node.key}"
    if st.session_state.get(key) not in options:
        st.session_state[key] = next((i for i,c in enumerate(pair.constructions) if c.habit_solutions), 0)
    chosen = st.selectbox(
        "Twin solution for this couple",
        options, key=key,
        format_func=lambda i: f"Solution {i + 1} — {twin_name(pair.constructions[i].classification)}; " +
        ("habit plane found" if pair.constructions[i].habit_solutions else "no discrete habit plane"),
    )
    twin: PairTwinConstruction = pair.constructions[chosen]
    unresolved = "unresolved" in twin.classification.lower() or "unresolved" in twin.classification_status.lower()
    if unresolved:
        st.markdown('<div class="tf-alert-amber"><strong>Classification not confirmed</strong> · A rank-one twin relation was computed, but the independent symmetry checks do not agree on Type I or II.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="tf-alert-blue"><strong>{twin_name(twin.classification)}</strong> · independently checked twin classification</div>', unsafe_allow_html=True)

    st.markdown("#### Martensite–martensite twin")
    st.markdown(f"**Amount of shear:** `{_fmt(twin.shear_magnitude)}`  ·  conventional symbol **s**")
    st.markdown(f"**Twin interface plane:** `{_vec(twin.twin_plane_product_crystal, plane=True)}`  ·  **K₁**, indices (hkl) in the product crystal")
    st.markdown(f"**Direction of shear:** `{_vec(twin.shear_direction_product_crystal)}`  ·  **η₁**, direction [uvw] in the product crystal")
    with st.expander("Detailed twin deformation · a and n̂", expanded=False):
        a, n = normalized_rank_one(twin.a_parent_cartesian, twin.n_parent_cartesian)
        st.markdown(f"**a · shear vector:** `{_vec(a)}` (parent Cartesian)")
        st.markdown(f"**n̂ · unit normal to twin interface:** `{_vec(n)}` (parent Cartesian)")
        st.caption("The displayed a is rescaled with n̂ so a ⊗ n̂ equals the computed tensor. This is NOT a unit-cell (hkl) index.")

    st.markdown("#### Austenite–martensite habit plane")
    if twin.continuum_fraction:
        st.markdown('<div class="tf-alert-teal"><strong>Continuous exact compatibility</strong> · no single discrete habit-plane branch.</div>', unsafe_allow_html=True)
    elif not twin.habit_solutions:
        st.info("No exact austenite–martensite interface exists for this twin solution. This is a calculated result, not missing data.")
    else:
        st.markdown('<div class="tf-alert-teal"><strong>Exact compatible interface found</strong> · the values below come from this twin solution.</div>', unsafe_allow_html=True)
        shown, extra = representative_habit_solutions(twin.habit_solutions)
        st.markdown(f"**Second-variant volume fraction λ:** `{_fmt(shown[0].other_variant_volume_fraction)}` (between 0 and 1)")
        st.caption("A compatible interface is not the same thing as the martensite–martensite twin plane K₁ above.")
        for index, solution in enumerate(shown, start=1):
            name = "+" if solution.habit_branch > 0 else "−" if solution.habit_branch < 0 else "0"
            st.markdown(f"**Habit alternative {index} ({name})**")
            if abs(solution.other_variant_volume_fraction - shown[0].other_variant_volume_fraction) > 1e-8:
                st.markdown(f"Volume fraction for this alternative: `{_fmt(solution.other_variant_volume_fraction)}`")
            st.markdown(f"**Shape-strain vector b{name}:** `{_vec(solution.shape_vector_parent_cartesian)}`")
            st.markdown(f"**Habit-plane normal m{name}:** `{_vec(solution.habit_normal_parent_cartesian)}`")
            st.caption("b includes magnitude and direction; m is a unit normal. Both are in the parent Cartesian frame, not Miller-index (hkl) coordinates.")
        if extra:
            with st.expander(f"Show {len(extra)} additional equivalent/complementary solutions", expanded=False):
                st.caption("Complementary variant fractions and alternative branch descriptions are preserved without averaging or inventing data.")
                for sol in extra:
                    st.write(f"λ {_fmt(sol.other_variant_volume_fraction)} · branch {sol.habit_branch:+d} · b {_vec(sol.shape_vector_parent_cartesian)} · m {_vec(sol.habit_normal_parent_cartesian)}")

    with st.expander("Research checks and coordinate frames", expanded=False):
        st.write(f"Classification cross-check: {twin.classification_status}")
        st.write(f"Rank-one residual: {_fmt(twin.rank_one_residual)}; rotation residual: {_fmt(twin.rotation_residual)}")
        st.latex(r"R U_j-U_i=a\otimes n")
        if pair.stretch_i is not None and pair.stretch_j is not None:
            st.write(f"Stretch variants: U{pair.stretch_i} and U{pair.stretch_j}")
        for sol in twin.habit_solutions:
            st.write(f"λ={_fmt(sol.other_variant_volume_fraction, 9)}, m_A (reciprocal)={_vec(sol.habit_plane_parent_crystal, plane=True)}, b_A (direct)={_vec(sol.shape_vector_parent_crystal)}")
            st.caption(f"Frame residuals: m={_fmt(sol.frame_plane_residual)}, b={_fmt(sol.frame_shape_vector_residual)}, habit={_fmt(sol.rank_one_residual)}")


def render_report(report: TwinFamilyReport) -> None:
    """Always render every scientific family first; do not default to F1 only."""
    layout = tree_layout(report)
    st.header("Calculated twin-family tree")
    if not layout.couples:
        st.info("There are no nonidentity variant couples for this input.")
        return

    st.markdown(
        f"**{len(layout.families)} families · {len(layout.couples)} variant couples** · "
        "Every family and couple belongs to the same transformation root."
    )
    st.caption("Root = your A → M transformation. F1, F2, … = symmetry-related twin families. M1 ↔ M2 = one pair of correspondence variants. All families remain visible; zoom or pan if the tree is crowded.")

    lookup = {node.key: node for node in layout.couples}
    choice_key, chart_key = "tf_selected_couple_v6", "tf_all_families_chart_v6"
    if st.session_state.get(choice_key) not in lookup:
        st.session_state[choice_key] = next((n.key for n in layout.couples if _habit_count(n)), layout.couples[0].key)

    def _select_from_chart() -> None:
        state = st.session_state.get(chart_key)
        try:
            points = state["selection"]["points"]
        except (KeyError, TypeError):
            return
        for point in reversed(points):
            if point.get("curve_number") != 2:
                continue
            raw = point.get("customdata")
            key = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
            if key in lookup:
                st.session_state[choice_key] = key
                return

    st.plotly_chart(
        plot_family_tree(report, st.session_state[choice_key], focus_family=None),
        use_container_width=True, key=chart_key,
        on_select=_select_from_chart, selection_mode="points",
        config={"displaylogo": False, "scrollZoom": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )
    st.caption("Blue circle = selected couple · Teal diamond = a calculated exact habit plane · Grey circle = another couple. Colour is never the only indicator.")
    st.selectbox(
        "Find and select a couple (keyboard or screen-reader accessible)",
        options=list(lookup), key=choice_key,
        format_func=lambda k: f"Family {lookup[k].family.family_id} — {lookup[k].label} — {_status_for(lookup[k])}",
    )
    _selected_details(lookup[st.session_state[choice_key]])

    with st.expander("Full correspondence inventory and technical warnings", expanded=False):
        st.write(f"Correspondence variants: {report.audit.topology_variant_count}; stretch variants: {report.audit.stretch_variant_count}")
        st.write(f"Correspondence subgroup size: {report.audit.correspondence_subgroup_order}; operator classes: {report.audit.operator_count}")
        for record in report.correspondence_variants:
            st.write(f"M{record.variant_index} → U{record.stretch_variant_index if record.stretch_variant_index is not None else 'unmapped'} · {record.mapping_status}")
        for warning in report.audit.warnings:
            st.warning(warning)
