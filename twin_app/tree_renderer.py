from __future__ import annotations

"""Three-level connected tree with ONE lower selected-couple result pane.

Root -> family -> all correspondence-variant couples at y=0.
Couples are never replaced by a vertical stack. Physical twins and exact
A/M habit results appear only in the single detail pane below the diagram.
"""

from typing import Iterable

import streamlit as st

from .family_tree_graph import (
    CoupleNode, couple_habit_count, normalized_rank_one, plot_family_tree, tree_layout,
)
from .scientific_models import PairTwinConstruction, TwinFamilyReport, TwinFamilyRecord


def _fmt(value: float) -> str:
    v = float(value)
    return f"{v:.4e}" if v != 0 and (abs(v) < 1.e-4 or abs(v) >= 1.e4) else f"{v:.6g}"


def _vec(values: Iterable[float], *, plane: bool = False) -> str:
    content = ", ".join(_fmt(v) for v in values)
    return f"({content})" if plane else f"[{content}]"


def _default_couple(nodes: tuple[CoupleNode, ...]) -> str:
    """Select a real habit calculation, if the chosen family has one."""
    return next(
        (n.key for n in nodes if couple_habit_count(n) > 0),
        nodes[0].key,
    )


def _weak_details(family: TwinFamilyRecord) -> None:
    if not family.weak_candidates:
        st.info("Weak family identified; no weak-plane coordinates calculated for this basis.")
        return
    st.markdown("**Higher-order weak candidates**")
    rows = []
    for c in family.weak_candidates:
        rows.append({
            "Rotation": f"{c.parent_rotation_order}-fold",
            "Parent axis [uvw]": _vec(c.parent_axis),
            "Product axis [uvw]": _vec(c.product_axis),
            "Primitive plane pair": f"{_vec(c.plane1_primitive,plane=True)} ↔ {_vec(c.plane2_primitive,plane=True)}",
            "qg": str(c.generalized_twin_index),
            "Shear": _fmt(c.generalized_shear),
        })
    st.table(rows)
    st.caption("Weak geometric relations are not automatically exact rank-one twins or A/M habit planes.")


def _physical_twin_table(construction: PairTwinConstruction) -> None:
    """Readable, explicitly framed physical observables—not a wide data grid."""
    a, n_hat = normalized_rank_one(construction.a_parent_cartesian, construction.n_parent_cartesian)
    st.markdown("**Twin elements · M/M**")
    st.table([
        {"Quantity": "Twin type", "Calculated value": construction.classification},
        {"Quantity": "Shear s", "Calculated value": _fmt(construction.shear_magnitude)},
        {"Quantity": "a · parent Cartesian", "Calculated value": _vec(a)},
        {"Quantity": "n̂ · parent Cartesian", "Calculated value": _vec(n_hat)},
        {"Quantity": "K₁ · product reciprocal (hkl)", "Calculated value": _vec(construction.twin_plane_product_crystal, plane=True)},
        {"Quantity": "η₁ · product direct [uvw]", "Calculated value": _vec(construction.shear_direction_product_crystal)},
    ])
    st.caption("n̂ is unit length; a is rescaled so a ⊗ n̂ is unchanged. K₁ is a plane, η₁ a direction.")


def _habit_table(construction: PairTwinConstruction) -> None:
    """Both ± A/M habit branches, always in the single selected-couple pane."""
    st.markdown("**Exact habit planes · A/M**")
    if construction.continuum_fraction:
        st.info("Continuous compatibility family. No arbitrary discrete habit plane is invented.")
        return
    if not construction.habit_solutions:
        st.info("No exact A/M habit solution (habit plane) for this twin branch.")
        return

    st.caption("b includes magnitude and direction · m is a unit habit normal · all vectors in parent Cartesian frame")
    rows = []
    for sol in sorted(construction.habit_solutions, key=lambda s: -s.habit_branch):
        sign = "+" if sol.habit_branch > 0 else "−" if sol.habit_branch < 0 else "0"
        rows.append({
            "Habit": sign,
            "λ": _fmt(sol.other_variant_volume_fraction),
            "b · parent Cartesian": _vec(sol.shape_vector_parent_cartesian),
            "m · parent Cartesian": _vec(sol.habit_normal_parent_cartesian, plane=True),
        })
    st.table(rows)
    st.caption("λ is the other-variant fraction. Signs/variants may differ by valid symmetry or branch conventions.")


def _details(node: CoupleNode) -> None:
    """Only ONE detail pane exists below the complete diagram."""
    st.divider()
    st.subheader(f"Selected twin couple · {node.label}")
    pair = node.pair
    if pair is None:
        st.caption(f"Family {node.family.family_id} · crystallographic correspondence couple")
        if node.family.route == "axial_weak":
            _weak_details(node.family)
        else:
            st.info("This couple has no evaluated physical rank-one construction.")
        return
    mapping = (
        f"U{pair.stretch_i} ↔ U{pair.stretch_j}"
        if pair.stretch_i is not None and pair.stretch_j is not None
        else "Stretch mapping unresolved"
    )
    st.caption(f"Family {node.family.family_id} · {mapping} · O{pair.operator_forward} / O{pair.operator_reverse}")
    if node.family.route == "axial_weak":
        _weak_details(node.family)
        return
    constructions = pair.constructions
    if not constructions:
        st.info(f"No exact classical twin relation for this couple. {pair.status}")
        return

    # Choose one rank-one branch in the SAME fixed pane. Favour an exact habit
    # solution when available; never show a blank habit section by default.
    choices = list(range(len(constructions)))
    key = f"selected_twin_branch_{node.key}"
    fav = next((i for i, c in enumerate(constructions) if c.habit_solutions), 0)
    if key not in st.session_state:
        st.session_state[key] = fav
    selection = st.selectbox(
        "Physical twin branch",
        choices,
        key=key,
        format_func=lambda i: (
            f"{constructions[i].classification} · branch {constructions[i].branch:+d} · "
            f"s={_fmt(constructions[i].shear_magnitude)} · "
            f"{len(constructions[i].habit_solutions)} habit solution(s)"
        ),
        help="Switch among this couple's distinct rank-one twin branches. The tree does not move.",
    )
    twin = constructions[selection]
    if twin.classification_status and "unresolved" in twin.classification_status.lower():
        st.warning("Twin classification cross-check unresolved. Numerical geometry is shown separately.")
    _physical_twin_table(twin)
    _habit_table(twin)

    with st.expander("More: crystal indices, validation and symmetry provenance", expanded=False):
        st.markdown(f"**{twin.classification} · rank-one audit**")
        st.caption(twin.classification_status)
        st.markdown(
            f"Rank-one residual `{_fmt(twin.rank_one_residual)}` · "
            f"rotation residual `{_fmt(twin.rotation_residual)}`"
        )
        st.markdown("$R U_j - U_i = a\\otimes n$ · independent CT/Ball–James cross-lock")
        st.caption("CT route identifiers: " + (", ".join(twin.classical_system_ids) or "unresolved"))
        for sol in twin.habit_solutions:
            st.markdown(
                f"Habit {sol.habit_branch:+d} · λ={_fmt(sol.other_variant_volume_fraction)} · "
                f"m_A (parent reciprocal/projective)={_vec(sol.habit_plane_parent_crystal,plane=True)} · "
                f"b_A (parent direct)={_vec(sol.shape_vector_parent_crystal)}"
            )
            st.caption(
                f"Residuals: parent plane `{_fmt(sol.frame_plane_residual)}` · "
                f"parent shape `{_fmt(sol.frame_shape_vector_residual)}` · "
                f"rank one `{_fmt(sol.rank_one_residual)}` · "
                f"middle stretch `{_fmt(sol.middle_stretch_residual)}`"
            )


def render_report(report: TwinFamilyReport) -> None:
    """Fixed root + family focus + one same-level couple row + one lower pane."""
    st.divider()
    st.header("Twin-family tree")
    layout = tree_layout(report)
    if not layout.couples:
        st.info("No nonidentity correspondence-variant couples for this transformation.")
        return
    st.caption(
        f"|H_C^A| = {report.audit.correspondence_subgroup_order} · "
        f"{report.audit.topology_variant_count} correspondence variants M · "
        f"{report.audit.stretch_variant_count} stretches U · "
        f"{report.audit.operator_count} operator classes · "
        f"{len(layout.families)} families · {len(layout.couples)} couples"
    )

    # Family zoom is a VIEW FILTER. Every couple stays in the complete layout.
    # It fixes the prior Plotly defect: many nodes + cropped offscreen root.
    all_family_ids = [f.family.family_id for f in layout.families]
    view_choices = all_family_ids + ["All families"]
    view_key = "twin_tree_family_view"
    if st.session_state.get(view_key) not in view_choices:
        # When any exact A/M habit exists, show its family first; otherwise
        # start from the first actual family. This never fabricates a solution.
        habit_family = next((
            c.family.family_id for c in layout.couples if couple_habit_count(c) > 0
        ), None)
        st.session_state[view_key] = habit_family or all_family_ids[0]
    family_view = st.selectbox(
        "Family to display",
        view_choices,
        key=view_key,
        format_func=lambda v: (
            "Full family tree (pan to explore)" if v == "All families" else
            f"Family {v} · {sum(c.family.family_id == v for c in layout.couples)} couples"
        ),
    )
    focused = None if family_view == "All families" else family_view
    if focused is not None:
        node = next(f for f in layout.families if f.family.family_id == focused)
        indices = ", ".join(f"O{v}" for v in node.family.operator_indices)
        st.caption(
            f"{focused} · operator family {indices} · {node.family.route.replace('_', ' ')} · "
            f"{node.count} equivalent couples"
        )
    visible = tuple(c for c in layout.couples if focused is None or c.family.family_id == focused)
    choices = {node.key: node for node in visible}
    picker_key = "twin_selected_couple"
    chart_key = "twin_all_couples_chart"
    if st.session_state.get(picker_key) not in choices:
        st.session_state[picker_key] = _default_couple(visible)

    def _sync_chart_selection() -> None:
        state = st.session_state.get(chart_key)
        if state is None:
            return
        try:
            points = state["selection"]["points"]
        except (KeyError, TypeError):
            return
        for point in reversed(points):
            if point.get("curve_number") != 2:
                continue
            custom = point.get("customdata")
            node_key = custom[0] if isinstance(custom, (list, tuple)) and custom else custom
            if node_key in choices:
                st.session_state[picker_key] = node_key
                break

    st.plotly_chart(
        plot_family_tree(report, st.session_state[picker_key], focus_family=focused),
        use_container_width=True,
        key=chart_key,
        on_select=_sync_chart_selection,
        selection_mode="points",
        config={
            "displaylogo": False, "scrollZoom": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    st.caption(
        "Root → operator family → all couples on one row. "
        "Filled circle = exact habit available; open circle = none shown. "
        "Select a circle or use the keyboard control below."
    )
    st.selectbox(
        "Twin couple (keyboard-accessible selection)",
        options=list(choices),
        key=picker_key,
        format_func=lambda k: (
            f"{choices[k].label} · {choices[k].family.family_id}" +
            (" · exact habit available" if couple_habit_count(choices[k]) else "")
        ),
    )
    _details(choices[st.session_state[picker_key]])

    if report.audit.warnings:
        with st.expander("Scientific warnings", expanded=False):
            for warning in report.audit.warnings:
                st.warning(warning)
    with st.expander("Full correspondence Mᵢ → stretch Uⱼ map", expanded=False):
        st.table([{
            "Correspondence": f"M{r.variant_index}",
            "Stretch": f"U{r.stretch_variant_index}" if r.stretch_variant_index is not None else "—",
            "Mapping status": r.mapping_status,
        } for r in report.correspondence_variants])
        st.caption("M and U denote different objects; metric degeneracy may collapse multiple M variants onto one U.")
