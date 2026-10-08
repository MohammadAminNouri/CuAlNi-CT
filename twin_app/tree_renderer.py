from __future__ import annotations

"""One clickable rooted twin-family tree, one persistent lower results panel.

Root -> operator/inverse family -> ALL variant couples. All couples share
exactly the same y-coordinate. Selecting a couple changes only the lower
book-style numerical pane, never the tree topology or the other pair nodes.
"""

from typing import Iterable

import streamlit as st

from .family_tree_graph import CoupleNode, normalized_rank_one, plot_family_tree, tree_layout
from .scientific_models import PairTwinConstruction, TwinFamilyReport, TwinFamilyRecord


def _fmt(value: float) -> str:
    x = float(value)
    return f"{x:.4e}" if 0 < abs(x) < 1e-4 else f"{x:.6g}"


def _vec(values: Iterable[float], *, plane: bool = False) -> str:
    content = ", ".join(_fmt(x) for x in values)
    return f"({content})" if plane else f"[{content}]"


def _status(node: CoupleNode) -> str:
    if node.pair is None:
        return "Correspondence couple found · no evaluated rank-one branch"
    pair = node.pair
    if not pair.constructions:
        return pair.status
    count = sum(len(c.habit_solutions) for c in pair.constructions)
    return f"{len(pair.constructions)} physical twin branch(es) · {count} A/M habit solution(s)"


def _weak_details(family: TwinFamilyRecord) -> None:
    st.markdown("**Higher-order weak-twin route**")
    st.caption(family.weak_status)
    if not family.weak_candidates:
        st.info("No weak-plane coordinates are available for this input and node basis.")
        return
    st.table([{
        "Order": c.parent_rotation_order,
        "Parent axis [uvw]": str(c.parent_axis),
        "Product axis [uvw]": str(c.product_axis),
        "Weak plane 1": str(c.plane1_primitive),
        "Weak plane 2": str(c.plane2_primitive),
        "q₍g₎": c.generalized_twin_index,
        "Generalized shear": _fmt(c.generalized_shear),
        "Generalized strain": _fmt(c.generalized_strain),
    } for c in family.weak_candidates])
    st.caption("Weak geometry is not an exact Ball–James/PTMC twin or habit plane.")


def _physical_twin_table(constructions: tuple[PairTwinConstruction, ...]) -> None:
    """Bhattacharya-style observables; every value is calculated, not memorized."""
    st.markdown("**M/M twinning elements**")
    rows = []
    for c in constructions:
        a, n_hat = normalized_rank_one(c.a_parent_cartesian, c.n_parent_cartesian)
        rows.append({
        "Branch": f"{c.branch:+d}",
        "Classification": c.classification,
        "a · parent Cartesian": _vec(a),
        "n̂ · parent Cartesian": _vec(n_hat),
        "K₁ · product (hkl)": _vec(c.twin_plane_product_crystal, plane=True),
        "η₁ · product [uvw]": _vec(c.shear_direction_product_crystal),
        "s": _fmt(c.shear_magnitude),
        })
    st.table(rows)


def _habit_table(constructions: tuple[PairTwinConstruction, ...]) -> None:
    st.markdown("**A/M habit planes**")
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for c in constructions:
        if c.continuum_fraction:
            missing.append(f"Twin {c.branch:+d}: continuous compatibility (not sampled)")
        elif not c.habit_solutions:
            missing.append(f"Twin {c.branch:+d}: no exact A/M interface")
        else:
            for solution in sorted(c.habit_solutions, key=lambda s: (s.other_variant_volume_fraction, -s.habit_branch)):
                branch_label = "+" if solution.habit_branch > 0 else "−" if solution.habit_branch < 0 else "0"
                rows.append({
                    "Twin": f"{c.classification} {c.branch:+d}",
                    "λ": _fmt(solution.other_variant_volume_fraction),
                    "Habit": branch_label,
                    "b · parent Cartesian": _vec(solution.shape_vector_parent_cartesian),
                    "m · parent Cartesian": _vec(solution.habit_normal_parent_cartesian, plane=True),
                })
    if rows:
        st.table(rows)
        st.caption("Book-style b includes magnitude and direction; m is a unit plane normal. λ is the other-variant volume fraction. Signs refer to the calculated habit branches; symmetry/branch conventions may differ from a printed table.")
    for message in missing:
        st.info(message)
    if not rows and not missing:
        st.info("No exact A/M habit solution is available for this couple.")


def _details(node: CoupleNode) -> None:
    pair = node.pair
    st.divider()
    st.subheader(f"Selected couple · {node.label}")
    st.caption(f"Family {node.family.family_id} · {_status(node)}")
    if pair is None:
        if node.family.route == "axial_weak":
            _weak_details(node.family)
        else:
            st.info("This reported couple has no pair-specific physical solution record.")
        return
    mapping = (
        f"U{pair.stretch_i} ↔ U{pair.stretch_j}"
        if pair.stretch_i is not None and pair.stretch_j is not None
        else "U mapping unresolved"
    )
    st.caption(f"Stretch mapping: {mapping} · operator O{pair.operator_forward} / O{pair.operator_reverse}")
    if node.family.route == "axial_weak":
        _weak_details(node.family)
        return
    if not pair.constructions:
        st.info(f"No exact classical twin solution: {pair.status}")
        return

    _physical_twin_table(pair.constructions)
    _habit_table(pair.constructions)

    # Expensive detail and type-I/type-II cross-lock provenance do not obscure
    # the book-style quantities users came to inspect.
    with st.expander("More: crystal indices, validation and symmetry provenance", expanded=False):
        for c in pair.constructions:
            st.markdown(f"**{c.classification} · twin {c.branch:+d}**")
            st.caption(c.classification_status)
            st.markdown(
                f"Rank-one residual: `{_fmt(c.rank_one_residual)}` · "
                f"rotation residual: `{_fmt(c.rotation_residual)}`"
            )
            st.markdown(
                f"$R U_j-U_i=a\\otimes n$ · "
                f"Type-I/II discrete lock: {', '.join(c.classical_system_ids) or 'unresolved'}"
            )
            for sol in c.habit_solutions:
                st.markdown(
                    f"λ=`{_fmt(sol.other_variant_volume_fraction)}` · "
                    f"branch `{sol.habit_branch:+d}` · "
                    f"m_A (parent reciprocal/projective): `{_vec(sol.habit_plane_parent_crystal, plane=True)}` · "
                    f"b_A (parent direct): `{_vec(sol.shape_vector_parent_crystal)}`"
                )
                st.caption(
                    f"Frame residuals: plane {_fmt(sol.frame_plane_residual)} · "
                    f"shape {_fmt(sol.frame_shape_vector_residual)} · "
                    f"rank-one {_fmt(sol.rank_one_residual)} · "
                    f"middle stretch {_fmt(sol.middle_stretch_residual)}"
                )
        if node.family.classical_systems:
            st.markdown("**Discrete CT routes (not interchangeable with physical K₁, η₁)**")
            for system in node.family.classical_systems:
                for rep in system.representations:
                    st.caption(
                        f"{system.system_id} · {rep.route}: "
                        f"{rep.plane_symbol}={_vec(rep.plane_product_crystal, plane=True)}; "
                        f"{rep.direction_symbol}={_vec(rep.direction_product_crystal)}; "
                        f"s={_fmt(rep.shear_magnitude)}"
                    )


def render_report(report: TwinFamilyReport) -> None:
    """Visible tree first; ONE lower panel only for the selected twin couple."""
    st.divider()
    st.header("Twin-family tree")
    layout = tree_layout(report)
    if not layout.couples:
        st.info("No non-identity correspondence-variant couple for this state.")
        return

    st.caption(
        f"{report.audit.topology_variant_count} M variants · "
        f"{report.audit.stretch_variant_count} U variants · "
        f"{len(layout.families)} families · {len(layout.couples)} couples"
    )
    # A keyboard-operable native control is mandatory: plotting libraries do
    # not guarantee that their pointer-selection interaction is accessible.
    choices = {node.key: node for node in layout.couples}
    picker_key = "twin_selected_couple"
    chart_key = "twin_all_couples_chart"
    if st.session_state.get(picker_key) not in choices:
        # Show a real habit result first when one exists. If none exists,
        # default to the center couple, without fabricating a solution.
        preferred = next(
            (node.key for node in layout.couples
             if node.pair is not None
             and any(c.habit_solutions for c in node.pair.constructions)),
            layout.couples[len(layout.couples) // 2].key,
        )
        st.session_state[picker_key] = preferred

    def _sync_chart_selection() -> None:
        chart_state = st.session_state.get(chart_key)
        if chart_state is None:
            return
        try:
            points = chart_state["selection"]["points"]
        except (KeyError, TypeError):
            return
        for point in reversed(points):
            if point.get("curve_number") != 2:
                continue
            datum = point.get("customdata")
            key = datum[0] if isinstance(datum, (list, tuple)) and datum else datum
            if key in choices:
                st.session_state[picker_key] = key
                break

    st.plotly_chart(
        plot_family_tree(report, st.session_state[picker_key]),
        use_container_width=True,
        key=chart_key,
        on_select=_sync_chart_selection,
        selection_mode="points",
        config={"displaylogo": False, "scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )
    st.caption("All couples are on one level. Click a couple, or choose it below. Drag to pan; scroll to zoom.")
    st.selectbox(
        "Twin couple (keyboard-accessible selection)",
        options=list(choices),
        key=picker_key,
        format_func=lambda key: f"{choices[key].label} · {choices[key].family.family_id}",
    )
    _details(choices[st.session_state[picker_key]])

    if report.audit.warnings:
        with st.expander("Scientific warnings", expanded=False):
            for warning in report.audit.warnings:
                st.warning(warning)
    with st.expander("Full correspondence Mᵢ → stretch Uⱼ map", expanded=False):
        st.table([{
            "Correspondence": f"M{item.variant_index}",
            "Stretch": f"U{item.stretch_variant_index}" if item.stretch_variant_index is not None else "—",
            "State": item.mapping_status,
        } for item in report.correspondence_variants])
        st.caption("M and U are different objects; several M may map to one U in a metric-degenerate state.")
