from __future__ import annotations

"""Linear, low-stimulation family-tree renderer for the twin workbench.

The renderer is deliberately presentation-only. It never changes a scientific
classification or recomputes a vector.  The heading structure follows the
scientific tree explicitly:

    Root -> family -> variant pair -> twin branch -> A/M habit branch.

Technical matrices and residuals stay available through optional expanders.
"""

from collections import Counter
from typing import Iterable

import streamlit as st

from .scientific_models import (
    ClassicalTwinSystem,
    PairTwinConstruction,
    TwinFamilyRecord,
    TwinFamilyReport,
    TwinElementRepresentation,
    VariantPairRecord,
)


def _number(value: float) -> str:
    x = float(value)
    if x == 0.0:
        return "0"
    if abs(x) < 1.0e-4 or abs(x) >= 1.0e4:
        return f"{x:.4e}"
    return f"{x:.6g}"


def _vector(values: Iterable[float], *, brackets: str = "[]") -> str:
    left, right = brackets[0], brackets[1]
    return left + ", ".join(_number(float(value)) for value in values) + right


def _pairs(pairs: tuple[tuple[int, int], ...]) -> str:
    return ", ".join(f"M{i} ↔ M{j}" for i, j in pairs)


def _route_label(route: str) -> str:
    return {
        "classical_exact": "Classical twin family",
        "axial_weak": "Higher-order weak family",
        "unsupported": "No supported twin route",
    }.get(route, route.replace("_", " ").title())


def _representation(rep: TwinElementRepresentation) -> None:
    st.markdown(f"**{rep.route} representation · {rep.plane_symbol} / {rep.direction_symbol}**")
    st.markdown(
        f"- shear `s = {_number(rep.shear_magnitude)}`\n"
        f"- product plane {rep.plane_symbol}: "
        f"`{_vector(rep.plane_product_crystal, brackets='()')}`\n"
        f"- product direction {rep.direction_symbol}: "
        f"`{_vector(rep.direction_product_crystal)}`"
    )
    with st.expander("Crystallographic provenance", expanded=False):
        st.markdown(
            f"Parent plane/covector: `{_vector(rep.plane_parent_crystal, brackets='()')}`  \n"
            f"Parent direct direction: `{_vector(rep.direction_parent_crystal)}`"
        )
        st.code(
            "\n".join("[ " + "  ".join(row) + " ]" for row in rep.generator_parent_symmetry),
            language="text",
        )
        st.caption("Exact parent symmetry generator used by the discrete construction.")


def _classical_system(system: ClassicalTwinSystem) -> None:
    st.markdown(
        f"**{system.system_id} · {system.classification} · "
        f"s = {_number(system.shear_magnitude)}**"
    )
    for rep in system.representations:
        _representation(rep)


def _construction(
    construction: PairTwinConstruction,
    systems: dict[str, ClassicalTwinSystem],
) -> None:
    st.markdown(
        f"##### Twin branch {construction.construction_id} · {construction.classification}"
    )
    st.markdown(f"**Classification check:** {construction.classification_status}")

    st.markdown(
        f"- shear `s = {_number(construction.shear_magnitude)}`\n"
        f"- physical twin plane `K₁` in product reciprocal coordinates: "
        f"`{_vector(construction.twin_plane_product_crystal, brackets='()')}`\n"
        f"- physical shear line `η₁` in product direct coordinates: "
        f"`{_vector(construction.shear_direction_product_crystal)}`"
    )

    if construction.classical_system_ids:
        st.caption(
            "Discrete-family cross-lock: "
            + ", ".join(construction.classical_system_ids)
        )

    with st.expander("Rank-one calculation and classification audit", expanded=False):
        st.markdown(
            "Pair-specific nonlinear elasticity:  "
            r"$R\,U_j-U_i=a\otimes n$."
        )
        st.markdown(
            f"`a = {_vector(construction.a_parent_cartesian)}`  \n"
            f"`n = {_vector(construction.n_parent_cartesian)}`  \n"
            f"rank-one residual = `{_number(construction.rank_one_residual)}`  \n"
            f"rotation residual = `{_number(construction.rotation_residual)}`"
        )
        if construction.independent_outer_product_residual is not None:
            st.markdown(
                f"independent tensor cross-check = "
                f"`{_number(construction.independent_outer_product_residual)}`  \n"
                f"independent shear cross-check = "
                f"`{_number(construction.independent_shear_relative_residual or 0.0)}`"
            )
        if construction.discrete_shear_relative_residual is not None:
            lines = [
                "discrete-system shear residual = "
                f"`{_number(construction.discrete_shear_relative_residual)}`"
            ]
            if construction.discrete_plane_angle_deg is not None:
                lines.append(
                    "K₁ projective angle = "
                    f"`{_number(construction.discrete_plane_angle_deg)}°`"
                )
            if construction.discrete_direction_angle_deg is not None:
                lines.append(
                    "η₁ projective angle = "
                    f"`{_number(construction.discrete_direction_angle_deg)}°`"
                )
            st.markdown("  \n".join(lines))

    st.markdown("**Austenite–martensite habit-plane result for this twin branch**")
    if construction.continuum_fraction:
        st.info(
            "Calculated result: a continuous exact compatible-fraction family exists. "
            "The backend does not replace that continuum with arbitrary sampled habit planes."
        )
    elif not construction.habit_solutions:
        st.info("Calculated result: no exact A/M habit-plane solution for this twin branch.")
    else:
        st.caption(
            r"For each solution, $F_\lambda=U_i+\lambda\,a\otimes n$ and "
            r"$R_hF_\lambda-I=b\otimes m$.  Plane covectors are projective: "
            r"$m$ and $-m$ describe the same plane."
        )
        for solution in construction.habit_solutions:
            with st.container(border=True):
                st.markdown(f"**Habit branch {solution.habit_branch:+d}**")
                st.markdown(
                    f"- other-variant fraction `λ = {_number(solution.other_variant_volume_fraction)}`\n"
                    f"- base-variant fraction `1 − λ = {_number(solution.base_variant_volume_fraction)}`\n"
                    f"- `m_A` parent reciprocal/projective coefficients: "
                    f"`{_vector(solution.habit_plane_parent_crystal, brackets='()')}`\n"
                    f"- `b_A` parent direct coefficients: "
                    f"`{_vector(solution.shape_vector_parent_crystal)}`"
                )
                with st.expander("Habit-plane numerical audit", expanded=False):
                    st.markdown(
                        f"m (parent orthonormal frame) = "
                        f"`{_vector(solution.habit_normal_parent_cartesian)}`  \n"
                        f"b (parent orthonormal frame) = "
                        f"`{_vector(solution.shape_vector_parent_cartesian)}`  \n"
                        f"m (base product reciprocal coefficients) = "
                        f"`{_vector(solution.habit_plane_product_crystal_base, brackets='()')}`  \n"
                        f"parent plane-frame residual = "
                        f"`{_number(solution.frame_plane_residual)}`  \n"
                        f"parent b-frame residual = "
                        f"`{_number(solution.frame_shape_vector_residual)}`  \n"
                        f"rank-one residual = `{_number(solution.rank_one_residual)}`  \n"
                        f"rotation residual = `{_number(solution.rotation_residual)}`  \n"
                        f"middle-stretch residual = `{_number(solution.middle_stretch_residual)}`"
                    )


def _pair(pair: VariantPairRecord, systems: dict[str, ClassicalTwinSystem]) -> None:
    mapping = (
        f"U{pair.stretch_i} ↔ U{pair.stretch_j}"
        if pair.stretch_i is not None and pair.stretch_j is not None
        else "stretch mapping incomplete"
    )
    with st.container(border=True):
        st.markdown(f"#### Variant pair {pair.pair_id}")
        st.caption(
            f"{mapping} · directed operator O{pair.operator_forward} / "
            f"inverse O{pair.operator_reverse}"
        )
        st.markdown(f"**Pair status:** {pair.status}")
        if not pair.constructions:
            return
        for index, construction in enumerate(pair.constructions):
            if index:
                st.divider()
            _construction(construction, systems)


def _weak_family(family: TwinFamilyRecord) -> None:
    st.markdown(f"**Weak-plane status:** {family.weak_status}")
    if not family.weak_candidates:
        return
    for rank, candidate in enumerate(family.weak_candidates, start=1):
        with st.container(border=True):
            st.markdown(
                f"**Weak candidate {rank} · parent rotation order "
                f"{candidate.parent_rotation_order}**"
            )
            st.markdown(
                f"- parent direct axis: `{candidate.parent_axis}`\n"
                f"- product direct axis: `{candidate.product_axis}`\n"
                f"- primitive weak-plane pair: `{candidate.plane1_primitive}` ↔ "
                f"`{candidate.plane2_primitive}`\n"
                f"- generalized twin index: `{candidate.generalized_twin_index}`\n"
                f"- generalized shear: `{_number(candidate.generalized_shear)}`\n"
                f"- in-plane distortion: `{_number(candidate.intraplanar_distortion)}`"
            )
            st.caption(
                "Higher-order weak candidate only. It is not relabelled as an exact "
                "Type-I/Type-II rank-one twin, so no exact laminate habit plane is "
                "fabricated beneath it."
            )


def _family(family: TwinFamilyRecord) -> None:
    st.markdown(f"### Branch {family.family_id} · {_route_label(family.route)}")
    st.markdown(
        f"**Representative correspondence pair:** "
        f"M{family.representative_pair[0]} ↔ M{family.representative_pair[1]}  \n"
        f"**Operator class family:** "
        + ", ".join(f"O{value}" for value in family.operator_indices)
    )

    if len(family.equivalent_pairs) > 1:
        with st.expander(
            f"Show {len(family.equivalent_pairs)} symmetry-equivalent correspondence pairs",
            expanded=False,
        ):
            st.markdown(_pairs(family.equivalent_pairs))
    else:
        st.caption(f"Equivalent correspondence pair: {_pairs(family.equivalent_pairs)}")

    systems = {system.system_id: system for system in family.classical_systems}
    if systems:
        with st.expander("Family-level discrete crystallographic twin systems", expanded=False):
            for index, system in enumerate(systems.values()):
                if index:
                    st.divider()
                _classical_system(system)

    if family.route == "axial_weak":
        _weak_family(family)
        return

    representative = next(
        (
            pair
            for pair in family.pair_records
            if (pair.variant_i, pair.variant_j) == family.representative_pair
        ),
        family.pair_records[0] if family.pair_records else None,
    )
    if representative is not None:
        st.markdown("**Representative pair calculation**")
        _pair(representative, systems)

    remaining = tuple(pair for pair in family.pair_records if pair is not representative)
    if remaining:
        with st.expander(
            f"Show {len(remaining)} additional symmetry-equivalent pair calculation(s)",
            expanded=False,
        ):
            for index, pair in enumerate(remaining):
                if index:
                    st.divider()
                _pair(pair, systems)


def render_report(report: TwinFamilyReport) -> None:
    st.divider()
    st.header("Calculated twin-family root tree")
    st.caption(
        "Calculated only from the entered lattices, point groups and correspondence. "
        "Literature values are not runtime inputs."
    )

    with st.container(border=True):
        st.markdown(
            f"### Root · {report.parent_phase_id} → {report.product_phase_id} transformation"
        )
        st.markdown(
            "**Read the result from top to bottom:**  \n"
            "Root → operator/twin family → correspondence-variant pair → "
            "physical twin branch → exact A/M habit-plane branch."
        )
        st.markdown(
            f"- correspondence subgroup order `|H_C^A| = {report.audit.correspondence_subgroup_order}`\n"
            f"- correspondence variants `Mᵢ`: `{report.audit.topology_variant_count}`\n"
            f"- distinct stretch variants `Uⱼ`: `{report.audit.stretch_variant_count}`\n"
            f"- operator classes: `{report.audit.operator_count}`\n"
            f"- non-identity family branches shown below: `{len(report.families)}`"
        )

    if report.audit.warnings:
        for warning in report.audit.warnings:
            st.warning(warning)

    with st.expander("Show explicit correspondence-to-stretch mapping Mᵢ → Uⱼ", expanded=False):
        for item in report.correspondence_variants:
            target = "—" if item.stretch_variant_index is None else f"U{item.stretch_variant_index}"
            residual = "—" if item.stretch_mapping_residual is None else _number(item.stretch_mapping_residual)
            st.markdown(
                f"**M{item.variant_index} → {target}**  \n"
                f"{item.mapping_status}  \n"
                f"mapping residual: `{residual}`"
            )
        st.caption(
            "Mᵢ and Uⱼ are different objects. Several correspondence variants may "
            "collapse onto one stretch variant for a degenerate metric."
        )

    route_counts = Counter(family.route for family in report.families)
    if route_counts:
        st.markdown(
            "**Branch inventory:** "
            + " · ".join(
                f"{count} {_route_label(route).lower()}"
                for route, count in sorted(route_counts.items())
            )
        )

    for family in report.families:
        _family(family)

    with st.expander("Global numerical audit", expanded=False):
        st.markdown(
            f"correspondence subgroup order = `{report.audit.correspondence_subgroup_order}`  \n"
            f"maximum M→U mapping residual = "
            f"`{_number(report.audit.maximum_correspondence_to_stretch_residual)}`  \n"
            f"PTMC maximum residual = `{_number(report.audit.ptmc_maximum_residual)}`  \n"
            f"independent nonlinear-elasticity maximum residual = "
            f"`{_number(report.audit.ball_james_maximum_residual)}`"
        )
