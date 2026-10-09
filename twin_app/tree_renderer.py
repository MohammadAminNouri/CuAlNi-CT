from __future__ import annotations

"""Single-row connected family tree, one selected-couple pane, layered precision.

The diagram carries topology only. The lower pane presents genuine backend
solutions using the same symbols as Bhattacharya's Tables 5.1 and 7.3;
NO published answer is used to render a runtime result.
"""

from typing import Iterable

import streamlit as st

from .family_tree_graph import (
    CoupleNode, couple_habit_count, normalized_rank_one,
    plot_family_tree, tree_layout,
)
from .scientific_models import HabitPlaneSolution, PairTwinConstruction, TwinFamilyReport, TwinFamilyRecord
from .ux_language import QUANTITY_HELP, habit_status, twin_name, representative_habit_solutions


def _fmt(value: float, *, precision: int = 5) -> str:
    v = float(value)
    if abs(v) < 5e-10:
        return "0"
    if abs(v) < 1e-4 or abs(v) >= 1e5:
        return f"{v:.3e}"
    return f"{v:.{precision}g}"


def _vec(values: Iterable[float], *, plane: bool = False, precision: int = 5) -> str:
    inner = ", ".join(_fmt(x, precision=precision) for x in values)
    return f"({inner})" if plane else f"[{inner}]"


def _default_couple(nodes: tuple[CoupleNode, ...]) -> str:
    return next((n.key for n in nodes if couple_habit_count(n) > 0), nodes[0].key)


def _panel(text: str, kind: str = "selection") -> None:
    """One labelled accent, not bright badges or colour-only meaning."""
    css = {"selection": "tf-selection", "habit": "tf-habit", "uncertain": "tf-uncertain"}[kind]
    st.markdown(f'<div class="tf-inline {css}">{text}</div>', unsafe_allow_html=True)


def _weak_details(family: TwinFamilyRecord) -> None:
    _panel("Higher-order weak relation · not an exact classical twin", "uncertain")
    if not family.weak_candidates:
        st.write("No weak-plane coordinates were calculated. An explicit primitive-node basis may be required.")
        return
    for c in family.weak_candidates:
        st.write(f"Parent axis {_vec(c.parent_axis)} → product axis {_vec(c.product_axis)}")
        st.write(f"Candidate plane pair {_vec(c.plane1_primitive,plane=True)} ↔ {_vec(c.plane2_primitive,plane=True)}")
        st.write(f"Generalized shear {_fmt(c.generalized_shear)}")
    st.caption("These weak relations are not automatically M/M rank-one twins or A/M habit planes.")
    with st.expander("Weak calculation details", expanded=False):
        for c in family.weak_candidates:
            st.write(f"Generalized twin index: {c.generalized_twin_index} · intraplanar distortion: {_fmt(c.intraplanar_distortion)}")



def _shape_name(branch: int) -> str:
    return "+" if branch > 0 else "−" if branch < 0 else "0"


def _book_twin_elements(c: PairTwinConstruction) -> None:
    """First recognized K1/eta1/s, then the exact a,n pair on request."""
    st.markdown("#### Twin plane and shear")
    st.caption("Standard crystallographic notation · Bhattacharya Table 5.1 convention")
    st.markdown(f"**Twin plane K₁**  `{_vec(c.twin_plane_product_crystal,plane=True)}`")
    st.caption("K₁ is a plane (hkl) expressed in the product crystal.")
    st.markdown(f"**Shear direction η₁**  `{_vec(c.shear_direction_product_crystal)}`")
    st.caption("η₁ is a direction [uvw] in the product crystal.")
    st.markdown(f"**Twin shear s**  `{_fmt(c.shear_magnitude)}`")
    with st.expander("Show full Table 5.1-style twin vectors a and n̂", expanded=False):
        a, nhat = normalized_rank_one(c.a_parent_cartesian, c.n_parent_cartesian)
        st.markdown(f"**a**  `{_vec(a,precision=7)}` · parent Cartesian")
        st.caption(QUANTITY_HELP["a"])
        st.markdown(f"**n̂**  `{_vec(nhat,precision=7)}` · parent Cartesian")
        st.caption(QUANTITY_HELP["n̂"])
        st.caption("n̂ is normalized. a is rescaled so the calculated rank-one tensor a ⊗ n̂ is unchanged.")


def _book_habit_elements(c: PairTwinConstruction) -> None:
    st.markdown("#### Austenite–martensite interface (habit plane)")
    st.caption("Bhattacharya Table 7.3 convention · all b and m values here are parent Cartesian")
    if c.continuum_fraction:
        _panel(habit_status(0, True), "habit")
        st.write("There is a continuous set of compatible fractions, not one discrete habit plane.")
        return
    if not c.habit_solutions:
        st.write("**No exact austenite–martensite habit plane** for this twin branch.")
        st.caption("This is a valid calculated outcome, not missing information.")
        return

    _panel("Exact A/M habit plane found", "habit")
    shown, extra = representative_habit_solutions(c.habit_solutions)
    if not shown:
        return
    representative_fraction = shown[0].other_variant_volume_fraction
    st.markdown(f"**Martensite fraction λ**  `{_fmt(representative_fraction)}`")
    st.caption("λ = fraction of the second variant in the twinned region. The complementary description uses 1 − λ.")
    for sol in shown:
        symbol = _shape_name(sol.habit_branch)
        st.markdown(f"**Habit alternative {symbol}**")
        st.markdown(f"**b{symbol}**  `{_vec(sol.shape_vector_parent_cartesian,precision=6)}`")
        st.caption("Shape strain: magnitude and direction together.")
        st.markdown(f"**m{symbol}**  `{_vec(sol.habit_normal_parent_cartesian,plane=True,precision=6)}`")
        st.caption("Habit-plane unit normal (parent Cartesian). m and −m represent the same plane.")

    if extra:
        with st.expander(f"Show {len(extra)} more calculated interface descriptions", expanded=False):
            st.caption("These include complementary fractions and symmetry-related configurations. None are deleted or averaged.")
            for sol in extra:
                sym = _shape_name(sol.habit_branch)
                st.markdown(
                    f"λ `{_fmt(sol.other_variant_volume_fraction,precision=7)}` · "
                    f"alternative {sym} · b `{_vec(sol.shape_vector_parent_cartesian,precision=7)}` · "
                    f"m `{_vec(sol.habit_normal_parent_cartesian,plane=True,precision=7)}`"
                )


def _selected_details(node: CoupleNode) -> None:
    """Exactly one lower pane regardless of family/couple selection."""
    st.divider()
    st.subheader(f"Selected couple · {node.label}")
    st.caption("Each M is a correspondence variant. U denotes a stretch variant; M and U are not interchangeable.")
    pair = node.pair
    if pair is None:
        if node.family.route == "axial_weak":
            _weak_details(node.family)
        else:
            st.info("Correspondence couple identified; no physical twin geometry evaluated here.")
        return
    if node.family.route == "axial_weak":
        _weak_details(node.family)
        return
    if not pair.constructions:
        st.write("No exact classical M/M twin found for this couple.")
        with st.expander("Scientific status", expanded=False):
            st.write(pair.status)
        return

    options = list(range(len(pair.constructions)))
    key = f"selected_twin_branch_{node.key}"
    if st.session_state.get(key) not in options:
        st.session_state[key] = next(
            (i for i, c in enumerate(pair.constructions) if c.habit_solutions), 0
        )
    st.caption("A couple can have different twin descriptions. Choose one; the tree stays in place.")
    selected = st.selectbox(
        "Twin description",
        options,
        key=key,
        format_func=lambda i: (
            f"{twin_name(pair.constructions[i].classification)}"
            + (" · exact habit found" if pair.constructions[i].habit_solutions else
               " · continuous compatibility" if pair.constructions[i].continuum_fraction else
               " · no exact habit")
        ),
        help="Type I: mirror-related. Type II: 180°-rotation-related. Compound: both descriptions for the same twin. Unverified means the independent classification checks disagree.",
    )
    twin = pair.constructions[selected]
    unresolved = "unresolved" in twin.classification.lower() or "unresolved" in twin.classification_status.lower()
    if unresolved:
        _panel("Twin geometry calculated · Type I/II label not yet verified", "uncertain")
        st.caption("The physical rank-one solution exists, but independent classification routes have not agreed. No type is guessed.")
    else:
        _panel(twin_name(twin.classification), "selection")

    _book_twin_elements(twin)
    _book_habit_elements(twin)
    with st.expander("Research details · frames, symmetry and numerical checks", expanded=False):
        st.write(f"Family {node.family.family_id} · operator pair O{pair.operator_forward}/O{pair.operator_reverse}")
        if pair.stretch_i is not None and pair.stretch_j is not None:
            st.write(f"Stretch connection U{pair.stretch_i} ↔ U{pair.stretch_j}")
        st.write(f"Classification audit: {twin.classification_status}")
        st.write(f"Rank-one residual {_fmt(twin.rank_one_residual)} · rotation residual {_fmt(twin.rotation_residual)}")
        st.latex(r"R U_j-U_i=a\otimes n")
        st.caption("Exactly these a, n and U variants generated the displayed habit solutions.")
        for sol in twin.habit_solutions:
            sym = _shape_name(sol.habit_branch)
            st.write(f"Habit {sym}: λ={_fmt(sol.other_variant_volume_fraction,precision=9)}")
            st.write(f"Parent reciprocal plane m_A = {_vec(sol.habit_plane_parent_crystal,plane=True,precision=9)}")
            st.write(f"Parent direct shape b_A = {_vec(sol.shape_vector_parent_crystal,precision=9)}")
            st.caption(f"Frame residuals: plane {_fmt(sol.frame_plane_residual)}, shape {_fmt(sol.frame_shape_vector_residual)}, rank-one {_fmt(sol.rank_one_residual)}")


def render_report(report: TwinFamilyReport) -> None:
    st.divider()
    st.header("Twin-family tree")
    layout = tree_layout(report)
    if not layout.couples:
        st.info("There are no nonidentity variant couples for this transformation.")
        return
    st.caption("Select a twin couple in the connected tree. Its calculations will appear in ONE panel below.")

    family_ids = [n.family.family_id for n in layout.families]
    available = family_ids + ["All families"]
    view_key = "twin_tree_family_view"
    if st.session_state.get(view_key) not in available:
        first_habit = next((c.family.family_id for c in layout.couples if couple_habit_count(c)), None)
        st.session_state[view_key] = first_habit or family_ids[0]

    family_choice = st.selectbox(
        "Show twin family",
        available,
        key=view_key,
        format_func=lambda v: (
            "All families (scroll the wide tree)" if v == "All families"
            else f"Family {v} · {sum(c.family.family_id == v for c in layout.couples)} couples"
        ),
        help="All couples in the selected family remain on the same row. Choose All families to inspect the complete genealogy.",
    )
    focused = None if family_choice == "All families" else family_choice
    visible = tuple(c for c in layout.couples if focused is None or c.family.family_id == focused)
    lookup = {c.key: c for c in visible}
    choice_key = "twin_selected_couple"
    chart_key = "twin_all_couples_chart"
    if st.session_state.get(choice_key) not in lookup:
        st.session_state[choice_key] = _default_couple(visible)

    def _select_from_chart() -> None:
        chart = st.session_state.get(chart_key)
        if chart is None:
            return
        try:
            points = chart["selection"]["points"]
        except (KeyError, TypeError):
            return
        for point in reversed(points):
            if point.get("curve_number") != 2:
                continue
            raw = point.get("customdata")
            name = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
            if name in lookup:
                st.session_state[choice_key] = name
                break

    st.plotly_chart(
        plot_family_tree(report, st.session_state[choice_key], focus_family=focused),
        use_container_width=True,
        key=chart_key,
        on_select=_select_from_chart,
        selection_mode="points",
        config={"displaylogo": False, "scrollZoom": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )
    st.markdown(
        '<div class="tf-legend"><span class="tf-dot tf-blue"></span> Selected couple'
        '&nbsp; <span class="tf-dot tf-teal"></span> Exact habit exists'
        '&nbsp; <span class="tf-dot tf-grey"></span> Other couple</div>',
        unsafe_allow_html=True,
    )
    st.selectbox(
        "Twin couple (keyboard-friendly alternative to clicking the tree)",
        options=list(lookup),
        key=choice_key,
        format_func=lambda k: (
            f"{lookup[k].label}"
            + (" · exact habit" if couple_habit_count(lookup[k]) else " · no exact habit")
        ),
    )
    _selected_details(lookup[st.session_state[choice_key]])

    with st.expander("All variants, mappings and scientific warnings (advanced)", expanded=False):
        st.write(f"Correspondence variants: {report.audit.topology_variant_count} · stretch variants: {report.audit.stretch_variant_count}")
        st.write(f"Correspondence subgroup order: {report.audit.correspondence_subgroup_order} · operator classes: {report.audit.operator_count}")
        for record in report.correspondence_variants:
            target = f"U{record.stretch_variant_index}" if record.stretch_variant_index is not None else "unmapped"
            st.write(f"M{record.variant_index} → {target} · {record.mapping_status}")
        for warning in report.audit.warnings:
            st.warning(warning)
