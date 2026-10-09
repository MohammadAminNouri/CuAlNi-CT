from __future__ import annotations

"""Accessible tree and a stable Bhattacharya-style scientific interpretation pane.

The custom navigator is an *interface*, not an independent source of physics.
All values, classifications and no-solution outcomes originate in TwinFamilyReport.
"""

from html import escape
from typing import Iterable

import streamlit as st

from .family_tree_graph import CoupleNode, normalized_rank_one, tree_layout
from .navigator_component import render_navigator
from .navigator_model import build_navigator_model, first_interpretable_couple
from .scientific_models import PairTwinConstruction, TwinFamilyReport
from .ux_language import twin_name, representative_habit_solutions


def _fmt(value: float, precision: int = 6) -> str:
    v = float(value)
    if abs(v) < 5e-11:
        return "0"
    return f"{v:.{precision}g}" if 1.e-4 <= abs(v) < 1.e5 else f"{v:.3e}"


def _vec(values: Iterable[float], *, plane: bool = False, precision: int = 5) -> str:
    parts = ", ".join(_fmt(v, precision) for v in values)
    return f"({parts})" if plane else f"[{parts}]"


def _habit_count(node: CoupleNode) -> int:
    return sum(len(c.habit_solutions) for c in node.pair.constructions) if node.pair is not None else 0


def _status(construction: PairTwinConstruction) -> tuple[str, str]:
    if "unresolved" in construction.classification.lower() or "unresolved" in construction.classification_status.lower():
        return "amber", "Twin type not verified"
    return "blue", twin_name(construction.classification)


def _named_value(label: str, symbol: str, meaning: str, value: str) -> None:
    st.markdown(
        '<div class="tf-quantity"><div class="tf-quantity-label">'
        f'{escape(label)} <span>· {escape(symbol)}</span></div>'
        f'<div class="tf-quantity-value">{escape(value)}</div>'
        f'<div class="tf-quantity-help">{escape(meaning)}</div></div>',
        unsafe_allow_html=True,
    )


def _habit_branches(construction: PairTwinConstruction) -> None:
    st.markdown("#### Austenite–martensite compatibility")
    if construction.continuum_fraction:
        st.markdown('<div class="tf-banner tf-banner-exact"><strong>Continuous compatible family</strong> · There is no unique discrete habit plane.</div>', unsafe_allow_html=True)
        return
    if not construction.habit_solutions:
        st.markdown(
            '<div class="tf-banner tf-banner-neutral"><strong>No exact A/M habit plane for this twin branch.</strong> '
            'This is a valid calculated outcome for the entered crystal parameters. Other couples may have solutions.</div>',
            unsafe_allow_html=True,
        )
        return
    st.markdown('<div class="tf-banner tf-banner-exact"><strong>Compatible interface calculated</strong> · Each alternative below is a separate exact solution.</div>', unsafe_allow_html=True)
    shown, extras = representative_habit_solutions(construction.habit_solutions)
    st.caption("λ is the volume fraction of the other martensite variant. The normal m and shape vector b are expressed in the parent Cartesian frame.")
    for i, solution in enumerate(shown, 1):
        branch = "+" if solution.habit_branch > 0 else "−" if solution.habit_branch < 0 else "0"
        with st.container(border=True):
            st.markdown(f"**Habit alternative {i} · {branch}**")
            _named_value("Second-variant fraction", "λ", "A fraction from 0 to 1", _fmt(solution.other_variant_volume_fraction))
            _named_value("Shape-strain vector", f"b{branch}", "Magnitude and direction; parent Cartesian", _vec(solution.shape_vector_parent_cartesian))
            _named_value("Habit-plane normal", f"m{branch}", "Unit normal; parent Cartesian, not Miller indices", _vec(solution.habit_normal_parent_cartesian, plane=True))
    if extras:
        with st.expander(f"Other valid solutions ({len(extras)})", expanded=False):
            st.caption("Complementary fractions and branch conventions are preserved without inventing or averaging any value.")
            for i, s in enumerate(extras, 1):
                st.markdown(f"**Additional solution {i}** · λ = `{_fmt(s.other_variant_volume_fraction)}` · branch {s.habit_branch:+d}")
                st.code(f"b = {_vec(s.shape_vector_parent_cartesian)}\nm = {_vec(s.habit_normal_parent_cartesian, plane=True)}", language="text")


def _selected_details(node: CoupleNode) -> None:
    st.markdown("<div class='tf-pane-heading'>SELECTED TWIN COUPLE</div>", unsafe_allow_html=True)
    st.subheader(f"{node.label} · {node.family.family_id}")
    st.caption("M-identifiers are computed correspondence variants; they are not Bhattacharya's printed variant numbering.")
    pair = node.pair
    if node.family.route == "axial_weak":
        st.markdown('<div class="tf-banner tf-banner-amber"><strong>Higher-order weak candidate.</strong> This is a symmetry relationship, not proof of an exact classical twin.</div>', unsafe_allow_html=True)
        if node.family.weak_candidates:
            with st.expander("Calculated weak-plane candidates", expanded=False):
                for w in node.family.weak_candidates:
                    st.write(f"Parent rotation order {w.parent_rotation_order} · primitive planes {w.plane1_primitive} / {w.plane2_primitive} · generalized shear {_fmt(w.generalized_shear)}")
        else:
            st.caption("A primitive product-node basis is required to enumerate weak planes. No indices are invented.")
        return
    if pair is None:
        st.info("The correspondence couple exists, but physical twin geometry was not evaluated.")
        return
    if not pair.constructions:
        st.info("No exact martensite–martensite rank-one twin relation for this couple and these inputs.")
        return
    indices = list(range(len(pair.constructions)))
    chosen_key = f"tf_v8_branch_{node.key}"
    if st.session_state.get(chosen_key) not in indices:
        st.session_state[chosen_key] = next((i for i, branch in enumerate(pair.constructions) if branch.habit_solutions), 0)
    if len(indices) > 1:
        st.radio(
            "Twin solution",
            indices,
            key=chosen_key,
            horizontal=True,
            format_func=lambda i: f"Solution {i + 1} · {twin_name(pair.constructions[i].classification).split(' · ')[0]}",
            help="Some couples have two exact rank-one branches. Choose which physical twin to inspect. This does not change the calculation.",
        )
    else:
        st.session_state[chosen_key] = 0
    branch = pair.constructions[st.session_state[chosen_key]]
    kind, label = _status(branch)
    st.markdown(f'<div class="tf-banner tf-banner-{kind}"><strong>{escape(label)}</strong> · {escape("Mathematical rank-one twin geometry calculated")}</div>', unsafe_allow_html=True)
    st.markdown("#### Martensite–martensite twinning elements")
    columns = st.columns(3)
    with columns[0]:
        _named_value("Twin shear", "s", "Relative shear amount", _fmt(branch.shear_magnitude))
    with columns[1]:
        _named_value("Twin plane", "K₁", "Product reciprocal (hkl)", _vec(branch.twin_plane_product_crystal, plane=True))
    with columns[2]:
        _named_value("Shear direction", "η₁", "Product direct [uvw]", _vec(branch.shear_direction_product_crystal))
    _habit_branches(branch)
    with st.expander("Scientific verification and full coordinates", expanded=False):
        st.markdown("**Exact twin equation**")
        st.latex(r"R_t U_j-U_i=a\otimes n")
        a, n = normalized_rank_one(branch.a_parent_cartesian, branch.n_parent_cartesian)
        st.code(f"a = {_vec(a)}  · parent Cartesian\nn = {_vec(n)}  · unit normal, parent Cartesian", language="text")
        st.write("Classification check:", branch.classification_status)
        st.caption(f"Twin rank-one residual {_fmt(branch.rank_one_residual, 9)} · proper rotation residual {_fmt(branch.rotation_residual, 9)}")
        st.markdown("**Exact A/M compatibility equation**")
        st.latex(r"R_h(U_i+\lambda a\otimes n)-I=b\otimes m")
        for sol in branch.habit_solutions:
            st.code(
                f"λ={_fmt(sol.other_variant_volume_fraction, 9)}; branch {sol.habit_branch:+d}\n"
                f"m in parent reciprocal coordinates = {_vec(sol.habit_plane_parent_crystal, plane=True)}\n"
                f"b in parent direct coordinates = {_vec(sol.shape_vector_parent_crystal)}\n"
                f"rank-one residual={_fmt(sol.rank_one_residual, 9)}\n"
                f"rotation residual={_fmt(sol.rotation_residual, 9)}\n"
                f"middle stretch residual={_fmt(sol.middle_stretch_residual, 9)}\n"
                f"frame residuals=({_fmt(sol.frame_plane_residual, 9)}, {_fmt(sol.frame_shape_vector_residual, 9)})",
                language="text",
            )
        st.caption("A/m compatibility is numerical for the entered lattice parameters; it does not establish experimental exactness within measurement uncertainty.")


def render_report(report: TwinFamilyReport, *, theme: str = "dark", density: str = "comfortable") -> None:
    """Show every family in a single navigator and a single fixed details pane."""
    st.header("Twin-family navigator")
    model = build_navigator_model(report)
    summary = model["summary"]
    if not summary["couple_count"]:
        st.info("No non-identity correspondence couples for these entered crystals.")
        return
    st.markdown(
        f"**{summary['family_count']} families** · {summary['couple_count']} couples · "
        f"**{summary['couples_with_habit']} couples with exact habit solutions**"
    )
    st.caption("All families belong to one transformation. Expand the couples for the full genealogy; select a node to inspect its result below.")
    lookup = {node.key: node for node in tree_layout(report).couples}
    state_key = "tf_selected_couple_v8"
    if st.session_state.get(state_key) not in lookup:
        st.session_state[state_key] = first_interpretable_couple(model)
    default = st.session_state[state_key]
    if summary["couples_with_habit"]:
        if st.button("Show a couple with an exact habit plane", key="tf_go_to_habit_v8", help="Selects an already calculated compatible couple. Does not recompute results."):
            st.session_state[state_key] = first_interpretable_couple(model)
            default = st.session_state[state_key]
    selected = render_navigator(model, selected=default, theme=theme, density=density, key="tf_browser_family_tree_v8")
    if selected is not None and selected != st.session_state[state_key]:
        st.session_state[state_key] = selected
    if (st.session_state.get("tf_text_finder_v8") not in lookup or
        (selected is not None and selected == st.session_state[state_key])):
        st.session_state["tf_text_finder_v8"] = st.session_state[state_key]
    def _select_from_text() -> None:
        choice = st.session_state.get("tf_text_finder_v8")
        if choice in lookup:
            st.session_state[state_key] = choice
    with st.expander("Find a couple by name (text and keyboard alternative)", expanded=False):
        st.selectbox(
            "Calculated correspondence couple",
            options=list(lookup), index=list(lookup).index(st.session_state[state_key]),
            format_func=lambda k: f"{lookup[k].family.family_id} · {lookup[k].label}",
            key="tf_text_finder_v8", on_change=_select_from_text,
        )
    _selected_details(lookup[st.session_state[state_key]])
    with st.expander("Correspondence genealogy and all numerical warnings", expanded=False):
        st.markdown(f"**Correspondence variants:** {report.audit.topology_variant_count} · **Distinct stretch variants:** {report.audit.stretch_variant_count}")
        st.markdown(f"**Operator classes:** {report.audit.operator_count} · **Correspondence subgroup order:** {report.audit.correspondence_subgroup_order}")
        for r in report.correspondence_variants:
            target = f"U{r.stretch_variant_index}" if r.stretch_variant_index is not None else "unmapped"
            st.write(f"M{r.variant_index} → {target} · {r.mapping_status}")
        for warning in report.audit.warnings:
            st.warning(warning)
