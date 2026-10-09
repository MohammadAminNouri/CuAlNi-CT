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
from .branch_presentation import distinct_twin_branches
from .interface_interpretation import compare_interface_geometries, interface_choice_description


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


def _habit_branches(construction: PairTwinConstruction, *, all_solutions: tuple | None = None) -> None:
    st.markdown("#### Austenite–martensite compatibility")
    if construction.continuum_fraction:
        st.markdown('<div class="tf-banner tf-banner-exact"><strong>Continuous compatible family</strong> · No unique discrete habit plane.</div>', unsafe_allow_html=True)
        return
    candidates = construction.habit_solutions if all_solutions is None else all_solutions
    if not candidates:
        st.markdown(
            '<div class="tf-banner tf-banner-neutral"><strong>No exact A/M habit plane for this twin branch.</strong> '
            'This is a calculated outcome for the entered metrics, not missing data. Other twin couples may have compatible interfaces.</div>',
            unsafe_allow_html=True,
        )
        return
    st.markdown('<div class="tf-banner tf-banner-exact"><strong>Compatible A/M interface found</strong> · Habit alternatives are separate from M/M interface A/B.</div>', unsafe_allow_html=True)
    shown, extras = representative_habit_solutions(candidates)
    alternatives = tuple(shown) + tuple(extras)
    st.caption("Choose ONE habit orientation below. λ is the other martensite variant's fraction; b includes magnitude and direction; m is a Cartesian unit normal, NOT Miller indices.")
    if not alternatives:
        return
    habit_key = f"tf_v9_habit_{construction.construction_id}"
    options = tuple(range(len(alternatives)))
    if st.session_state.get(habit_key) not in options:
        st.session_state[habit_key] = 0
    if len(options) > 1:
        st.radio(
            "Habit interface orientation",
            options, key=habit_key, horizontal=True,
            format_func=lambda i: f"Habit {i+1} · {'+' if alternatives[i].habit_branch > 0 else '−' if alternatives[i].habit_branch < 0 else '0'}",
            help="These are A/M habit solutions of the selected M/M interface, not alternative twinning classifications.",
        )
    sol = alternatives[st.session_state[habit_key]]
    branch = '+' if sol.habit_branch > 0 else '−' if sol.habit_branch < 0 else '0'
    _named_value("Second-variant fraction", "λ", "0 is none of the other variant; 1 is entirely the other variant", _fmt(sol.other_variant_volume_fraction))
    _named_value("Habit-plane normal", f"m{branch}", "Unit normal; parent Cartesian, not Miller indices", _vec(sol.habit_normal_parent_cartesian, plane=True))
    _named_value("Shape-strain vector", f"b{branch}", "Physical shape strain with magnitude and direction; parent Cartesian", _vec(sol.shape_vector_parent_cartesian))
    with st.expander("View other calculated habit solutions", expanded=False):
        st.caption("Every admissible calculated solution is preserved. Complementary fractions are not automatically discarded.")
        for i, extra in enumerate(alternatives):
            st.write(f"Habit {i+1} · λ = {_fmt(extra.other_variant_volume_fraction, 8)} · sign {extra.habit_branch:+d}")


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
<<<<<<< HEAD
    # Display physically DISTINCT rank-one tensors, not anonymous ± solver signs.
    interfaces = distinct_twin_branches(pair.constructions)
    choice_key = f"tf_v9_interface_{node.key}"
    options = tuple(item.label for item in interfaces)
    if st.session_state.get(choice_key) not in options:
        st.session_state[choice_key] = next(
            (item.label for item in interfaces if item.representative.habit_solutions),
            options[0],
=======
    # Display physically DISTINCT rank-one tensors, not anonymous ± solver signs.
    interfaces = distinct_twin_branches(pair.constructions)
    choice_key = f"tf_v9_interface_{node.key}"
    options = tuple(item.label for item in interfaces)
    if st.session_state.get(choice_key) not in options:
        st.session_state[choice_key] = next(
            (item.label for item in interfaces if item.representative.habit_solutions),
            options[0],
        )
    st.markdown("**Martensite–martensite interface geometry**")
    st.caption("Each interface is a distinct solution of R·Uⱼ − Uᵢ = a⊗n. Interface A/B is not the same as Type I/Type II, and is not the +/− habit-plane designation.")
    if len(interfaces) > 1:
        st.radio(
            "Choose an interface geometry",
            options, key=choice_key, horizontal=True,
            format_func=lambda label: (
                label + " · " + twin_name(next(x for x in interfaces if x.label == label).representative.classification).split(" · ")[0]
            ),
            help="These are distinct rank-one interface geometries for this one martensite couple. Selecting one does not recalculate anything.",
>>>>>>> 2fce910d70d4b7cc9f6cc900b26f34ec5c719f62
        )
<<<<<<< HEAD
    st.markdown("#### The possible twin interfaces")
    st.write(
        "The **same two martensite variants** can sometimes meet through two different "
        "planar interfaces. Each option is a separately calculated solution of the "
        "martensite–martensite rank-one equation. You are selecting a *geometry*, "
        "not changing the material or choosing Type I versus Type II. "
        "Interface A/B is not the same as Type I/Type II."
    )
    if len(interfaces) > 1:
        comparison = compare_interface_geometries(interfaces[0].representative, interfaces[1].representative)
        st.caption(comparison.narrative)
    else:
        st.caption("One physically distinct interface was obtained. Duplicate algebraic records are combined for presentation only.")
    # Larger, clear selection surfaces; one stable details pane below.
    columns = st.columns(min(len(interfaces), 2), gap="medium")
    for i, interface in enumerate(interfaces):
        with columns[i % len(columns)]:
            active = st.session_state[choice_key] == interface.label
            with st.container(border=True):
                st.markdown(f"**{interface.label}**{' · Currently selected' if active else ''}")
                st.write(interface_choice_description(interface))
                st.caption(
                    f"Twin plane (product crystal): {_vec(interface.representative.twin_plane_product_crystal, plane=True)}"
                )
                has_habit = bool(interface.all_habit_solutions) or bool(interface.representative.continuum_fraction)
                st.caption("A/M compatible habit: " + ("calculated for this branch" if has_habit else "no discrete exact habit calculated for this branch"))
                if st.button(
                    "Currently selected" if active else f"Inspect {interface.label}",
                    key=f"tf_v91_select_{node.key}_{i}",
                    use_container_width=True,
                    disabled=active,
                ):
                    st.session_state[choice_key] = interface.label
    st.caption(
        "Interface A/B are screen labels, not published crystallographic modes. "
        "A verified Type I, Type II or Compound label is a separate scientific conclusion. "
        "Habit-plane alternatives belong *inside* whichever twin interface you inspect."
    )
    selected_interface = next(item for item in interfaces if item.label == st.session_state[choice_key])
    branch = selected_interface.representative
    if len(selected_interface.equivalent_source_indices) > 1:
        st.caption(f"{len(selected_interface.equivalent_source_indices)} algebraic records describe this same physical rank-one tensor; all distinct habit solutions and the original records are preserved.")
    if len(selected_interface.classifications) > 1:
        st.warning("Conflicting classification evidence exists for this physical interface. Review all recorded routes before identifying its twin type.")
=======
    selected_interface = next(item for item in interfaces if item.label == st.session_state[choice_key])
    branch = selected_interface.representative
    if len(selected_interface.equivalent_source_indices) > 1:
        st.caption(f"{len(selected_interface.equivalent_source_indices)} algebraic records describe this same physical rank-one tensor; all distinct habit solutions and the original records are preserved.")
    if len(selected_interface.classifications) > 1:
        st.warning("Conflicting classification evidence exists for this physical interface. Review all recorded routes before identifying its twin type.")
>>>>>>> 2fce910d70d4b7cc9f6cc900b26f34ec5c719f62
    kind, label = _status(branch)
<<<<<<< HEAD
    if len(selected_interface.classifications) > 1:
        kind, label = "amber", "Twin geometry calculated · conflicting type evidence; not verified"
    st.markdown(f'<div class="tf-banner tf-banner-{kind}"><strong>{escape(label)}</strong></div>', unsafe_allow_html=True)
    if kind == "amber":
        st.write(
            "The rank-one equation has a numerical solution, but the independent "
            "crystallographic checks do not yet establish a Type-I, Type-II or Compound label. "
            "This does not mean the interface geometry is absent."
        )
=======
    if len(selected_interface.classifications) > 1:
        kind, label = "amber", "Twin geometry calculated · conflicting type evidence; not verified"
    st.markdown(f'<div class="tf-banner tf-banner-{kind}"><strong>{escape(label)}</strong> · {escape("Mathematical rank-one twin geometry calculated")}</div>', unsafe_allow_html=True)
>>>>>>> 2fce910d70d4b7cc9f6cc900b26f34ec5c719f62
    st.markdown("#### Martensite–martensite twinning elements")
    columns = st.columns(3)
    with columns[0]:
        _named_value("Twin shear magnitude", "s", "How much one martensite variant shears relative to the other (dimensionless)", _fmt(branch.shear_magnitude))
    with columns[1]:
        _named_value("Twin plane", "K₁", "Which plane can separate the two martensite variants; product-crystal (hkl)", _vec(branch.twin_plane_product_crystal, plane=True))
    with columns[2]:
<<<<<<< HEAD
        _named_value("Shear direction", "η₁", "Crystal direction associated with the twin shear; product-crystal [uvw]", _vec(branch.shear_direction_product_crystal))
    _habit_branches(branch, all_solutions=selected_interface.all_habit_solutions)
=======
        _named_value("Shear direction", "η₁", "Product direct [uvw]", _vec(branch.shear_direction_product_crystal))
    _habit_branches(branch, all_solutions=selected_interface.all_habit_solutions)
>>>>>>> 2fce910d70d4b7cc9f6cc900b26f34ec5c719f62
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
