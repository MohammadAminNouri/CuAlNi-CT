from __future__ import annotations

"""Low-clutter input UI for the standalone twin-family application."""

from dataclasses import dataclass
from typing import Sequence

import streamlit as st
import sympy as sp

from app.application import LatticeInput, PhaseInput
from cualni_cryst.lattice import Lattice
from cualni_cryst.point_groups import point_group_table
from cualni_cryst.weak_twins import BravaisNodeBasis

from .input_logic import canonical_correspondence_rows, displayed_relation, exact_matrix
from .symmetry_inventory import SymmetryInventory, build_symmetry_inventory
from .point_group_explainer import explain_group, simple_group_label
from .point_group_guide import group_guide, explanation_for_operation
from .input_explainer import preview_correspondence, validate_metric_parameters


FAMILY_ORDER = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
)


@dataclass(frozen=True)
class PhaseUIResult:
    phase: PhaseInput
    inventory: SymmetryInventory


@dataclass(frozen=True)
class CorrespondenceUIResult:
    canonical_rows: tuple[tuple[str, str, str], ...]
    input_direction: str
    display_arrow: str
    display_equation: str


@dataclass(frozen=True)
class WeakBasisUIResult:
    basis: BravaisNodeBasis | None
    label: str


def _point_group_metadata() -> tuple[dict[str, object], ...]:
    return point_group_table()


def _locked_number(label: str, value: float, *, key: str, angle: bool = False, length_unit: str | None = None) -> float:
    # Derived lattice constraints are displayed, never edited or persisted as
    # independent widget state. Include the exact value in the widget identity
    # so a change in the controlling length cannot display a stale locked value.
    kwargs: dict[str, object] = {
        "label": label,
        "value": float(value),
        "disabled": True,
        "key": f"{key}_{float(value).hex()}",
        "format": "%.8e" if length_unit == "meter" and not angle else "%.8f",
    }
    if angle:
        kwargs.update(min_value=0.000001, max_value=179.999999)
    else:
        kwargs.update(min_value=1.0e-12)
    st.number_input(**kwargs)
    return float(value)


def _number(label: str, value: float, *, key: str, angle: bool = False, length_unit: str | None = None) -> float:
    kwargs: dict[str, object] = {
        "label": label,
        "value": float(value),
        "key": key,
        "format": "%.8e" if length_unit == "meter" and not angle else "%.8f",
    }
    if angle:
        kwargs.update(min_value=0.000001, max_value=179.999999)
    else:
        kwargs.update(min_value=1.0e-12)
    return float(st.number_input(**kwargs))


def _operation_label(operation: object) -> str:
    """Tell the user what an operation *does* before exposing G-indices."""
    kind = str(operation.kind)
    axis = operation.axis_or_plane
    if kind == "identity":
        return "No change · identity"
    if kind == "inversion":
        return "Flip every direction · inversion"
    if axis is not None:
        indices = ",".join(str(v) for v in axis)
        if kind == "mirror reflection":
            return f"Reflect across plane ({indices})"
        if "rotation" in kind:
            if operation.order == 2:
                return f"Rotate 180° about [{indices}]"
            # Group order alone does NOT distinguish, e.g., +90° from −90°.
            return f"{operation.order}-fold rotation about [{indices}]"
    return kind.replace("order-", "order ")


def _coordinate_rule(matrix: tuple[tuple[str, str, str], ...]) -> str:
    """Explain an exact symmetry action on fractional crystal coordinates."""
    symbols = ("x", "y", "z")
    import sympy as sp
    M = sp.Matrix([[sp.Rational(x) for x in row] for row in matrix])
    v = M * sp.Matrix(sp.symbols("x y z"))
    return "(x, y, z) → (" + ", ".join(str(sp.simplify(expr)) for expr in v) + ")"


def _render_operation_summary(inventory: SymmetryInventory, *, parent: bool, cell: tuple[float, float, float, float, float, float]) -> None:
    """Scientific meaning beside the selected group, not a large operations dump."""
    guide = group_guide(inventory, parent=parent)
    with st.container(border=True):
        st.markdown(f"**{guide.symbol} · {guide.family.capitalize()} crystal class**")
        st.caption(f"Selected Hermann–Mauguin point group · {guide.setting}")
        st.write(guide.signature)
        st.caption(
            f"Exact group order: {guide.order} operations · "
            f"{guide.proper} proper (det = +1) · {guide.improper} improper (det = −1) · "
            f"Inversion: {'yes' if guide.inversion else 'no'}."
        )
        # Counts are from actual exact matrices, never prose/table expected values.
        families = tuple(f for f in guide.families if f.name != "Identity")
        if families:
            preview = []
            for family in families:
                kind = family.name.lower()
                label = f"{family.operations} {kind}"
                if family.geometric_elements:
                    axes = family.geometric_elements
                    representative = axes[0]
                    form = "(" + ", ".join(map(str, representative)) + ")" if "plane" in family.coordinate_type else "[" + ", ".join(map(str, representative)) + "]"
                    label += f" · {len(axes)} distinct {'planes' if 'plane' in family.coordinate_type else 'axes'}, e.g. {form}"
                preview.append(label)
            st.markdown("**Operations belonging to this group**")
            st.write("  ·  ".join(preview))
        else:
            st.write("Identity only; no nontrivial point-symmetry operation.")
        mirror_count = next((f.operations for f in guide.families if f.name == "Mirror reflections"), 0)
        twofold_count = next((f.operations for f in guide.families if f.name == "Proper 2-fold rotation"), 0)
        if parent:
            st.caption(
                f"Role in the calculation: these exact parent operations act on the calculated "
                f"stretch variants and determine which unordered variant pairs belong to one "
                f"symmetry orbit. The group contains {mirror_count} mirror operation(s) and "
                f"{twofold_count} proper half-turn(s). They are candidate symmetry relations, "
                "not automatically physical Type I or Type II twins: rank-one compatibility, "
                "twin geometry and independent classification checks must also pass."
            )
        else:
            st.caption(
                "Role in the calculation: these product symmetries identify equivalent "
                "crystal-coordinate descriptions; plane (hkl) and direction [uvw] indices "
                "refer to this product's selected conventional setting. The product point "
                "group alone cannot establish a measured twin."
            )
        st.caption(
            f"Metric check: maximum relative preservation residual "
            f"{inventory.metric_preservation_maximum_residual:.2e}. "
            "This checks compatibility with the entered lattice metric, "
            "NOT whether the specimen's atoms actually possess this point group."
        )

        with st.expander("Operation counts, indexed elements and primary references", expanded=False):
            st.caption(
                "Counts below are matrix operations, not counts of independent axes. "
                "For example, opposite quarter-turns share a fourfold axis. "
                "Axis [uvw] is a direct-space direction; plane (hkl) is a reciprocal covector."
            )
            for family in guide.families:
                indices = family.geometric_elements
                label = f"**{family.name}: {family.operations} operation(s)**"
                if indices:
                    example = indices[:6]
                    formatted = ", ".join(
                        "(" + ",".join(map(str, v)) + ")" if "plane" in family.coordinate_type
                        else "[" + ",".join(map(str, v)) + "]"
                        for v in example
                    )
                    label += f" · {len(indices)} indexed elements · {formatted}"
                    if len(indices) > 6:
                        label += f" · +{len(indices)-6} more"
                st.write(label)
            st.markdown(
                "Crystallographic nomenclature and operation definitions: "
                "[IUCr — Symmetry and 32 point groups]"
                "(https://www.iucr.org/what-we-do/education/pamphlets/symmetry) · "
                "[IUCr — Matrices, mappings and crystallographic symmetry]"
                "(https://www.iucr.org/what-we-do/education/pamphlets/matrices-mappings-and-crystallographic-symmetry) · "
                "[IUCr — Definition of symmetry elements]"
                "(https://www.iucr.org/who-we-are/commissions/commission-on-crystallographic-nomenclature/published-reports/symel)"
            )
            st.caption(
                "Hermann–Mauguin symbols and group orders follow the 32 crystallographic "
                "point classes. A space group additionally describes translations, "
                "including glides and screw axes; this tool does not infer one from lattice angles."
            )

        with st.expander("Inspect one symmetry operation (optional) · action, axis and matrix", expanded=False):
            ops = list(inventory.operations)
            default = next((i for i, item in enumerate(ops) if item.kind == "proper 2-fold rotation"), 0)
            index = st.selectbox(
                "Symmetry operation", list(range(len(ops))), index=default,
                format_func=lambda i: _operation_label(ops[i]),
                key=f"inspect_{'parent' if parent else 'product'}_{inventory.symbol}",
                help="Select one operation to see exactly what happens to crystal coordinates.",
            )
            op = ops[index]
            meaning, limitation = explanation_for_operation(op)
            st.markdown(f"**{_operation_label(op)}**")
            st.write(meaning)
            st.write(f"**Action on a point:** {_coordinate_rule(op.matrix)}")
            st.caption(limitation)
            with st.expander("3D geometry in the entered cell", expanded=False):
                from .point_group_visualizer import cell_basis, make_operation_scene
                from .operation_witness import operation_witness
                B = cell_basis(*[float(v) for v in cell])
                witness = operation_witness(B, op.matrix)
                fig = make_operation_scene(
                    B, op.kind, op.axis_or_plane, operation_matrix=op.matrix
                )
                st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": False})
                st.caption(
                    "Blue/orange arrows: one generic test direction before/after "
                    "the exact selected point-group operation; not a measured vector. "
                    f"Relative metric-length residual: {witness.relative_metric_residual:.2e}. "
                    "The vectors share one illustrative display scale."
                )
                if op.axis_or_plane is None or not (
                    "rotation" in op.kind or "mirror" in op.kind
                ):
                    st.info(
                        "A dedicated axis/plane 3D symmetry overlay is not implemented for this operation. "
                        "The before/after arrows still show its exact coordinate action."
                    )
                st.caption("Geometry only: this is not a stereographic projection, EBSD pattern, or physical twin prediction.")
            with st.expander("Show exact coordinate matrix and numerical check", expanded=False):
                st.code("\n".join("[ " + "  ".join(row) + " ]" for row in op.matrix), language="text")
                st.caption(f"Metric-preservation maximum residual over the selected group: {inventory.metric_preservation_maximum_residual:.2e}")


def render_phase_input(
    *,
    prefix: str,
    heading: str,
    role: str,
    length_unit: str,
    default_family: str,
    default_point_group: str,
    defaults: tuple[float, float, float, float, float, float],
) -> PhaseUIResult:
    st.subheader(heading)
    metadata = _point_group_metadata()
    by_symbol = {str(row["symbol"]): row for row in metadata}
    families = [
        family
        for family in FAMILY_ORDER
        if any(str(row["crystal_family"]).lower() == family for row in metadata)
    ]
    st.caption("Crystal system = shape of the unit cell; point group = rotations and reflections that keep it unchanged.")
    family = st.selectbox(
        "Crystal system",
        families,
        index=families.index(default_family) if default_family in families else 0,
        key=f"{prefix}_family",
    )
    symbols = [
        str(row["symbol"])
        for row in metadata
        if str(row["crystal_family"]).lower() == family
    ]
    selected_default = default_point_group if default_point_group in symbols else symbols[0]
    point_group = st.selectbox(
        "Point group",
        symbols,
        index=symbols.index(selected_default),
        key=f"{prefix}_point_group_{family}",
        format_func=lambda symbol: (
            f"{symbol} — {simple_group_label(symbol)}"
        ),
        help="Symbol followed by a plain-language name. Choose a group to see what it means directly below.",
    )

    # Reserve the location right below the selection. Fill it once the real
    # edited cell metric has been validated (never display a guessed count).
    selected_group_explanation = st.container()

    a0, b0, c0, alpha0, beta0, gamma0 = defaults
    st.markdown("**Conventional cell**")
    lengths = st.columns(3)
    with lengths[0]:
        a = _number("a", a0, key=f"{prefix}_a", length_unit=length_unit)
    with lengths[1]:
        if family in {"cubic", "tetragonal", "hexagonal", "trigonal"}:
            b = _locked_number("b", a, key=f"{prefix}_b_locked_{family}", length_unit=length_unit)
        else:
            b = _number("b", b0, key=f"{prefix}_b", length_unit=length_unit)
    with lengths[2]:
        if family == "cubic":
            c = _locked_number("c", a, key=f"{prefix}_c_locked_{family}", length_unit=length_unit)
        else:
            c = _number("c", c0, key=f"{prefix}_c", length_unit=length_unit)

    angles = st.columns(3)
    with angles[0]:
        if family == "triclinic":
            alpha = _number("α (deg)", alpha0, key=f"{prefix}_alpha", angle=True)
        else:
            alpha = _locked_number("α (deg)", 90.0, key=f"{prefix}_alpha_locked_{family}", angle=True)
    with angles[1]:
        if family in {"triclinic", "monoclinic"}:
            beta = _number("β (deg)", beta0, key=f"{prefix}_beta", angle=True)
        else:
            beta = _locked_number("β (deg)", 90.0, key=f"{prefix}_beta_locked_{family}", angle=True)
    with angles[2]:
        if family == "triclinic":
            gamma = _number("γ (deg)", gamma0, key=f"{prefix}_gamma", angle=True)
        elif family in {"hexagonal", "trigonal"}:
            gamma = _locked_number("γ (deg)", 120.0, key=f"{prefix}_gamma_locked_{family}", angle=True)
        else:
            gamma = _locked_number("γ (deg)", 90.0, key=f"{prefix}_gamma_locked_{family}", angle=True)

    family_notes = {
        "cubic": "a = b = c; α = β = γ = 90°.",
        "tetragonal": "a = b; α = β = γ = 90°.",
        "orthorhombic": "α = β = γ = 90°.",
        "hexagonal": "hexagonal axes: a = b; α = β = 90°, γ = 120°.",
        "trigonal": "built-in trigonal groups use hexagonal axes: a = b; α = β = 90°, γ = 120°.",
        "monoclinic": "built-in monoclinic groups use the unique-b setting: α = γ = 90°.",
        "triclinic": "all six cell parameters are independent.",
    }
    st.caption(family_notes[family])

    label = st.text_input(
        "Phase label",
        value="Parent" if role == "parent" else "Product",
        key=f"{prefix}_label",
    )
    phase_id = "A" if role == "parent" else "M"
    representation = str(by_symbol[point_group]["conventional_setting"])
    phase = PhaseInput(
        phase_id=phase_id,
        label=label,
        physical_phase=label,
        cell_representation=representation,
        point_group=point_group,
        lattice=LatticeInput(a, b, c, alpha, beta, gamma, length_unit),
    )
    lattice = Lattice(a, b, c, alpha, beta, gamma, label=label, length_unit=length_unit)
    inventory = build_symmetry_inventory(point_group, lattice.metric())
    volume, condition = validate_metric_parameters((a, b, c), (alpha, beta, gamma))
    st.caption(f"Valid positive-definite cell metric · volume {volume:.5g} {length_unit}³ · normalized metric condition {condition:.3g}")
    with selected_group_explanation:
        _render_operation_summary(inventory, parent=(role == "parent"), cell=(a,b,c,alpha,beta,gamma))
    return PhaseUIResult(phase=phase, inventory=inventory)


def render_correspondence_input(
    *,
    prefix: str = "correspondence",
    default_rows: Sequence[Sequence[str]] = (
        ("1", "0", "0"), ("0", "1", "0"), ("0", "0", "1"),
    ),
) -> CorrespondenceUIResult:
    """Cayron 2019: paired direct vectors OR exact rational matrix, one solver C.

    Mode and direction switches carry over the same exact correspondence.
    The below direction probe always uses canonical C_(M <- A).
    """
    from .correspondence_explorer import (
        calculate_action, images_of_parent_basis, vector_text,
    )
    from .cayron_correspondence import (
        calculate_from_paired_vectors, parse_triplet, seed_paired_vectors,
        plane_image, entered_matrix_from_canonical, direction_labels,
    )

    baseline = canonical_correspondence_rows(default_rows, direction="A_TO_M")
    method_key = f"{prefix}_entry_method"
    last_method_key = f"{prefix}_last_method"
    last_direction_key = f"{prefix}_last_direction"
    warning_key = f"{prefix}_conversion_warning"
    methods = ("Three paired lattice vectors", "Exact rational 3 × 3 matrix")
    directions = {"Parent crystal → Product crystal": "A_TO_M", "Product crystal → Parent crystal": "M_TO_A"}
    dir_key = "correspondence_direction_choice"
    # Existing Streamlit sessions may still store the old internal A/M labels.
    # Migrate them before registering the widget, keeping the exact C unchanged.
    legacy_directions = {
        "A → M": "Parent crystal → Product crystal",
        "M → A": "Product crystal → Parent crystal",
    }
    if st.session_state.get(dir_key) in legacy_directions:
        st.session_state[dir_key] = legacy_directions[st.session_state[dir_key]]

    def matrix_keys(direction: str):
        return tuple(tuple(f"{prefix}_{direction}_{i}_{j}" for j in range(3)) for i in range(3))

    def paired_keys(direction: str, end: str):
        return tuple(f"{prefix}_supercell_{direction}_{end}_{j}" for j in range(3))

    def read_editor(method: str, direction: str):
        if method == methods[1]:
            keys = matrix_keys(direction)
            if all(key in st.session_state for row in keys for key in row):
                typed = [[st.session_state[key] for key in row] for row in keys]
                return canonical_correspondence_rows(typed, direction=direction)
        else:
            source_keys, dest_keys = paired_keys(direction, "src"), paired_keys(direction, "dst")
            if all(k in st.session_state for k in (*source_keys, *dest_keys)):
                src = [parse_triplet(st.session_state[k]) for k in source_keys]
                dst = [parse_triplet(st.session_state[k]) for k in dest_keys]
                return calculate_from_paired_vectors(src, dst, direction=direction).canonical_M_from_A
        return baseline

    def set_editor(method: str, direction: str, canonical):
        D = exact_matrix(entered_matrix_from_canonical(canonical, direction=direction))
        if method == methods[1]:
            keys = matrix_keys(direction)
            if all(k in st.session_state for row in keys for k in row):
                try:
                    prior = canonical_correspondence_rows(
                        [[st.session_state[k] for k in row] for row in keys],
                        direction=direction,
                    )
                    if prior == canonical:
                        return  # Retain exact user spellings rather than rewrite them.
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
            for i, row in enumerate(keys):
                for j, key in enumerate(row):
                    st.session_state[key] = str(D[i, j])
        else:
            sources = paired_keys(direction, "src")
            destinations = paired_keys(direction, "dst")
            if all(k in st.session_state for k in (*sources, *destinations)):
                try:
                    prior = calculate_from_paired_vectors(
                        [parse_triplet(st.session_state[k]) for k in sources],
                        [parse_triplet(st.session_state[k]) for k in destinations],
                        direction=direction,
                    ).canonical_M_from_A
                    if prior == canonical:
                        return  # Preserve the actual paired-supercell witnesses.
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
            source_seed, dest_seed = seed_paired_vectors(canonical, direction=direction)
            for key, value in zip(sources, source_seed):
                st.session_state[key] = value
            for key, value in zip(destinations, dest_seed):
                st.session_state[key] = value

    def synchronize_editors():
        previous_method = st.session_state.get(last_method_key, methods[0])
        previous_direction = st.session_state.get(last_direction_key, "A_TO_M")
        new_method = st.session_state.get(method_key, methods[0])
        new_direction = directions.get(st.session_state.get(dir_key, "Parent crystal → Product crystal"), "A_TO_M")
        try:
            canonical = read_editor(previous_method, previous_direction)
            set_editor(new_method, new_direction, canonical)
            st.session_state.pop(warning_key, None)
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            # Do not strand the user in a fresh default editor: keep the invalid
            # draft visible in its original direction/format so it can be fixed.
            st.session_state[method_key] = previous_method
            st.session_state[dir_key] = (
                "Parent crystal → Product crystal" if previous_direction == "A_TO_M" else "Product crystal → Parent crystal"
            )
            st.session_state[warning_key] = (
                "Cannot convert an invalid correspondence: " + str(exc)
                + ". Correct the original entries before switching."
            )

    st.subheader("Crystallographic correspondence")
    st.caption(
        "Specify how direct-lattice directions in the parent crystal correspond "
        "to directions in the product crystal. Either input method produces "
        "the same exact correspondence matrix."
    )
    st.caption("This C is NOT the orientation matrix T or the distortion F. "
               "Parent and product indices use their respective conventional crystal bases.")
    method = st.radio(
        "Input format", methods, horizontal=True, key=method_key,
        on_change=synchronize_editors,
        help="Paired vectors use C = Q P⁻¹; matrix entries may be exact integers or fractions.",
    )
    direction_label = st.radio(
        "Enter correspondence in direction",
        ("Parent crystal → Product crystal", "Product crystal → Parent crystal"),
        horizontal=True, key=dir_key,
        on_change=synchronize_editors,
        help="Changing direction computes an exact inverse; it does not infer an orientation relationship.",
    )
    direction = directions[direction_label]
    source_name, destination_name = direction_labels(direction)
    if direction == "A_TO_M":
        st.latex(
            r"\mathbf u_{\mathrm{product}}="
            r"C_{\mathrm{parent}\to\mathrm{product}}"
            r"\mathbf u_{\mathrm{parent}}"
        )
    else:
        st.latex(
            r"\mathbf u_{\mathrm{parent}}="
            r"C_{\mathrm{product}\to\mathrm{parent}}"
            r"\mathbf u_{\mathrm{product}}"
        )
    st.caption(f"Reading order: **{source_name} → {destination_name}**. "
               "Source is always on the left, destination on the right.")
    st.session_state[last_method_key] = method
    st.session_state[last_direction_key] = direction
    if st.session_state.get(warning_key):
        st.warning(st.session_state.pop(warning_key))

    with st.expander("How vectors and Miller planes transform", expanded=False):
        st.write("Corresponding direct lattice vectors form columns of B_source and B_destination. "
                 "C = Q P⁻¹. Reversing the input direction uses C⁻¹. "
                 "Miller-plane covectors transform by C⁻ᵀ, not C.")
        st.caption("Integers and rational entries p/q are preferred. A typed finite decimal "
                   "is treated as its exact base-10 fraction; 0.3333 does not mean 1/3. "
                   "Cell parameters, atom positions and point-group settings do not determine C by themselves.")

    if method == methods[0]:
        st.markdown(f"**Three corresponding direct-space lattice vectors · {source_name} → {destination_name}**")
        st.caption("Each pair gives the source and destination components of one direct lattice vector. "
                   "Use independent vectors in exact integer/fraction notation, in the basis of each crystal.")
        default_source, default_dest = seed_paired_vectors(baseline, direction=direction)
        guides = st.columns([0.85, 2.6, 0.4, 2.6], gap="small")
        guides[1].markdown(f"**Source · {source_name}**")
        guides[2].markdown("**→**")
        guides[3].markdown(f"**Destination · {destination_name}**")
        source = []
        destination = []
        for j, label in enumerate(("Pair 1", "Pair 2", "Pair 3")):
            cols = st.columns([0.85, 2.6, 0.4, 2.6], gap="small")
            cols[0].markdown(f"**{label}**")
            src = cols[1].text_input(
                f"{label} in {source_name}", value=default_source[j],
                key=paired_keys(direction, "src")[j],
            )
            cols[2].markdown("**→**")
            dst = cols[3].text_input(
                f"Corresponding {label} in {destination_name}", value=default_dest[j],
                key=paired_keys(direction, "dst")[j],
            )
            source.append(src)
            destination.append(dst)
        try:
            paired = calculate_from_paired_vectors(
                [parse_triplet(v) for v in source],
                [parse_triplet(v) for v in destination],
                direction=direction,
            )
            canonical = paired.canonical_M_from_A
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            st.error(f"Corresponding vectors cannot define C: {exc}")
            st.stop()
        st.latex(
            r"C_{\mathrm{entered}}= B_{\mathrm{destination}}"
            r"B_{\mathrm{source}}^{-1}="
            + sp.latex(exact_matrix(paired.source_to_destination))
        )
        st.caption("The initial integer pairs are algebraically generated, not experimentally observed. "
                   "Verify cell settings, centering and physical correspondence before using the predictions.")
    else:
        st.markdown(f"**Exact 3 × 3 correspondence · {source_name} → {destination_name}**")
        st.caption("Read source (left) → destination (right). Each COLUMN is a "
                   "source basis vector, with destination components in its rows. "
                   "Use exact p/q fractions rather than rounded approximations.")
        D_default = exact_matrix(entered_matrix_from_canonical(baseline, direction=direction))
        headings = st.columns([1.6, 1, 1, 1])
        for col, name in zip(headings, ("Destination ↓ / source →", "a", "b", "c")):
            col.markdown(f"**{name}**")
        typed = []
        for i, label in enumerate(("a", "b", "c")):
            cols = st.columns([1.6, 1, 1, 1], gap="small")
            cols[0].markdown(f"**{label}**")
            row = []
            for j in range(3):
                row.append(cols[j + 1].text_input(
                    f"{source_name} → {destination_name}: destination component {label}, source axis {j+1}",
                    value=str(D_default[i, j]),
                    key=matrix_keys(direction)[i][j], label_visibility="collapsed",
                    max_chars=80,
                ))
            typed.append(row)
        try:
            canonical = canonical_correspondence_rows(typed, direction=direction)
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            st.error(f"Exact matrix is unusable: {exc}")
            st.stop()

    C = exact_matrix(canonical)
    from .numeric_domain import validate_solver_correspondence
    try:
        validate_solver_correspondence(C)
    except ValueError as exc:
        st.error(f"Unsafe numerical correspondence: {exc}")
        st.stop()
    if canonical != baseline:
        st.info("Correspondence differs from the starting editable example. "
                "The calculation will use only these explicitly entered directions.")

    def restore_starting_correspondence():
        set_editor(method, direction, baseline)

    if canonical != baseline:
        st.button("Restore starting correspondence", key=f"{prefix}_restore_C",
                  on_click=restore_starting_correspondence)

    entered_matrix = entered_matrix_from_canonical(canonical, direction=direction)
    D = exact_matrix(entered_matrix)
    if direction == "M_TO_A":
        if D * C != sp.eye(3) or C * D != sp.eye(3):
            raise AssertionError("Reverse editor and canonical solver mapping are not exact inverses")
        label = r"C_{\mathrm{product}\to\mathrm{parent}}"
    else:
        if D != C:
            raise AssertionError("Forward editor disagrees with canonical solver mapping")
        label = r"C_{\mathrm{parent}\to\mathrm{product}}"
    st.markdown(f"**Selected direction: {source_name} → {destination_name}**")
    st.latex(label + "=" + sp.latex(D))
    st.caption(
        f"det(selected matrix) = {D.det()}. This is the exact matrix for the "
        "direction selected above, including any edits you have made."
    )
    if direction == "M_TO_A":
        st.caption(
            "The calculation uses the inverse of this matrix to express the "
            "same transformation from parent to product. Reversing the input "
            "direction alone must not change the calculated physical result."
        )
    with st.expander("Advanced: exact parent-to-product solver matrix", expanded=False):
        st.latex(r"C_{\mathrm{parent}\to\mathrm{product}}=" + sp.latex(C))
        st.caption(
            "The scientific engine always uses this parent-to-product matrix. "
            "It remains unchanged when you merely reverse the direction of entry, "
            "because the inverse is computed exactly."
        )
        if direction == "M_TO_A":
            st.latex(
                r"C_{\mathrm{product}\to\mathrm{parent}}"
                r"C_{\mathrm{parent}\to\mathrm{product}}=I"
            )
        st.caption(
            f"det(parent-to-product matrix) = {C.det()}. An invertible matrix "
            "alone does not establish a physically observed correspondence."
        )
    st.markdown(f"**Interactive vector mapping · {source_name} → {destination_name}**")
    choice = st.selectbox(
        f"Direction in {source_name}",
        ("a-axis [1,0,0]", "b-axis [0,1,0]", "c-axis [0,0,1]", "Custom [u,v,w]"),
        key=f"{prefix}_probe_choice",
    )
    axes = (("1", "0", "0"), ("0", "1", "0"), ("0", "0", "1"))
    if choice == "Custom [u,v,w]":
        fields = st.columns(3, gap="small")
        parent_vector = tuple(
            fields[k].text_input(label.replace("parent", source_name), value=v, key=f"{prefix}_probe_{k}")
            for k, (label, v) in enumerate(zip(("u (parent)", "v (parent)", "w (parent)"), ("1", "1", "0")))
        )
    else:
        parent_vector = axes[("a-axis [1,0,0]", "b-axis [0,1,0]", "c-axis [0,0,1]").index(choice)]

    try:
        action = calculate_action(entered_matrix, parent_vector)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        st.error(f"Invalid direction: {exc}; enter rational integers or p/q fractions.")
    else:
        st.markdown(f"**{source_name} {vector_text(action.parent)} → {destination_name} {vector_text(action.daughter)}**")
        st.latex(sp.latex(D) + r"\," + sp.latex(sp.Matrix(action.parent)) + "=" + sp.latex(sp.Matrix(action.daughter)))

        with st.expander("More coordinate mappings and optional 3D drawings", expanded=False):
            st.caption(f"For the three source axes in {source_name}, each a COLUMN of the entered matrix:")
            actions = images_of_parent_basis(entered_matrix)
            for label, obj in zip(("a", "b", "c"), actions):
                st.write(f"{source_name} {label} → {destination_name} {vector_text(obj.daughter)}")
            st.caption("These components are expressed on **separate coordinate axes**. "
                       "They cannot be compared as common-frame Cartesian vectors.")
            if st.checkbox("Show 3D coordinate-space sketches", value=False,
                           key=f"{prefix}_show_coordinate_arrows"):
                import plotly.graph_objects as go

                def arrow_figure(values: Sequence[sp.Rational], title: str):
                    coords = [float(v) for v in values]
                    limit = 1.2 * max(1.0, *(abs(v) for v in coords))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter3d(
                        x=[0, coords[0]], y=[0, coords[1]], z=[0, coords[2]],
                        mode="lines+markers", line=dict(color="#38bdf8", width=8),
                        marker=dict(size=4, color="#38bdf8"), showlegend=False,
                    ))
                    fig.update_layout(height=260, margin=dict(l=0, r=0, t=12, b=0),
                        scene=dict(xaxis=dict(title="u", range=[-limit,limit]),
                                   yaxis=dict(title="v", range=[-limit,limit]),
                                   zaxis=dict(title="w", range=[-limit,limit]),
                                   aspectmode="cube"))
                    return fig

                left, right = st.columns(2)
                with left:
                    st.caption(f"{source_name} direct-coordinate components")
                    st.plotly_chart(arrow_figure(action.parent, "Parent"),
                                    width="stretch", key=f"{prefix}_plot_parent")
                with right:
                    st.caption(f"{destination_name} direct-coordinate components")
                    st.plotly_chart(arrow_figure(action.daughter, "Daughter"),
                                    width="stretch", key=f"{prefix}_plot_daughter")
                st.caption("These drawings use separate coordinate axes. No physical angle can be deduced.")

    with st.expander("Advanced: inverse matrix and reciprocal-plane correspondence", expanded=False):
        st.code("\n".join("[ " + "  ".join(row) + " ]" for row in canonical), language="text")
        st.write(f"det(C) = {C.det()}")
        st.code(str(C.inv()), language="text")
        st.caption("C.inv() gives the reverse DIRECT correspondence, not an independently observed OR.")
        plane = plane_image(entered_matrix, ("1", "0", "0"))
        st.write(f"Reciprocal plane (1,0,0) in {source_name} → ({', '.join(plane)}) in {destination_name}.")
        st.latex(r"\mathbf h_{\mathrm{destination}}=C_{\mathrm{source}\to\mathrm{destination}}^{-T}\mathbf h_{\mathrm{source}}")
        st.caption("This is a reciprocal covector mapping; do not transform Miller planes by direct C.")


    arrow, equation = displayed_relation(direction)
    return CorrespondenceUIResult(
        canonical_rows=canonical, input_direction=direction,
        display_arrow=arrow, display_equation=equation,
    )

def _standard_basis(name: str) -> sp.Matrix:
    half = sp.Rational(1, 2)
    if name == "P — primitive conventional cell":
        return sp.eye(3)
    if name == "A — A-centered conventional cell":
        return sp.Matrix([[1, 0, 0], [0, half, -half], [0, half, half]])
    if name == "B — B-centered conventional cell":
        return sp.Matrix([[half, 0, -half], [0, 1, 0], [half, 0, half]])
    if name == "C — C-centered conventional cell":
        return sp.Matrix([[half, -half, 0], [half, half, 0], [0, 0, 1]])
    if name == "I — body-centered conventional cell":
        return sp.Matrix([[-half, half, half], [half, -half, half], [half, half, -half]])
    if name == "F — face-centered conventional cell":
        return sp.Matrix([[0, half, half], [half, 0, half], [half, half, 0]])
    raise KeyError(name)


def render_weak_basis_input() -> WeakBasisUIResult:
    with st.expander("Optional: higher-order weak-twin plane search", expanded=False):
        st.caption(
            "Only needed if you want plane indices for higher-order weak-twin candidates. "
            "A point group describes rotational symmetry but does not specify where lattice nodes sit. "
            "For ordinary exact Type-I/Type-II twins, leave this at 'Not specified'."
        )
        choices = (
            "Not specified — identify weak families only",
            "P — primitive conventional cell",
            "A — A-centered conventional cell",
            "B — B-centered conventional cell",
            "C — C-centered conventional cell",
            "I — body-centered conventional cell",
            "F — face-centered conventional cell",
            "Custom exact primitive-node basis",
        )
        choice = st.selectbox("Product lattice node arrangement (only for weak-plane enumeration)", choices)
        if choice.startswith("Not specified"):
            return WeakBasisUIResult(None, choice)
        if choice.startswith("Custom"):
            st.caption(r"Enter P in u_conventional = P u_primitive using exact integers/decimals/fractions.")
            values: list[list[str]] = []
            for i in range(3):
                cols = st.columns(3)
                row: list[str] = []
                for j in range(3):
                    with cols[j]:
                        row.append(
                            st.text_input(
                                f"P{i + 1}{j + 1}",
                                value="1" if i == j else "0",
                                key=f"weak_basis_{i}_{j}",
                                label_visibility="collapsed",
                            )
                        )
                values.append(row)
            matrix = exact_matrix(values)
            return WeakBasisUIResult(
                BravaisNodeBasis(matrix, "user-supplied exact primitive-node basis"),
                choice,
            )
        matrix = _standard_basis(choice)
        return WeakBasisUIResult(BravaisNodeBasis(matrix, choice), choice)
