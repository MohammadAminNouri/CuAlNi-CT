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


def _locked_number(label: str, value: float, *, key: str, angle: bool = False) -> float:
    st.session_state[key] = float(value)
    kwargs: dict[str, object] = {
        "label": label,
        "value": float(value),
        "disabled": True,
        "key": key,
        "format": "%.8f",
    }
    if angle:
        kwargs.update(min_value=0.000001, max_value=179.999999)
    else:
        kwargs.update(min_value=1.0e-12)
    return float(st.number_input(**kwargs))


def _number(label: str, value: float, *, key: str, angle: bool = False) -> float:
    kwargs: dict[str, object] = {
        "label": label,
        "value": float(value),
        "key": key,
        "format": "%.8f",
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
        st.markdown(f"**{guide.symbol} · {guide.title}**")
        st.write(guide.signature)
        st.caption(f"{guide.family.capitalize()} · {guide.setting} · {guide.order} total point operations")
        st.markdown(
            f"**Verified contents:**  {guide.proper} proper (orientation-preserving) operations; "
            f"{guide.improper} improper (orientation-reversing) operations. "
            f"Inversion centre: {'present' if guide.inversion else 'absent'}."
        )
        st.markdown("**Why it matters for this calculation**")
        st.write(guide.role_explanation)
        st.caption(
            "These operations are an exact inventory of the *selected* point group, "
            "verified against the entered lattice metric. Lattice parameters alone do not "
            "establish atomic-structure or space-group symmetry."
        )
<<<<<<< HEAD
=======
        op = items[selection]
        st.markdown(f"**{_operation_label(op)}**")
        st.write("**Action in fractional crystal coordinates:** " + _coordinate_rule(op.matrix))
        # An arbitrary illustrative point, not a Miller index or a measured direction.
        import sympy as sp
        demo = sp.Matrix([sp.Rational(1, 4), sp.Rational(1, 3), sp.Rational(1, 5)])
        matrix = sp.Matrix([[sp.Rational(x) for x in row] for row in op.matrix])
        mapped = matrix * demo
        st.caption("Example point (fractional coordinates): " +
                   "(" + ", ".join(str(x) for x in demo) + ") → (" +
                   ", ".join(str(x) for x in mapped) + "). This example is illustrative, not an experimental coordinate.")
        st.write(f"**Why this matters:** {op.twin_role}.")
        with st.expander("Visualize this axis or plane within the unit cell", expanded=False):
            from .point_group_visualizer import cell_basis, make_operation_scene
            from numpy import asarray
            # Use the actual entered cell, not a hard-coded cubic scene.
            B = cell_basis(*[float(v) for v in cell])
            fig = make_operation_scene(B, op.kind, op.axis_or_plane)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": False})
            st.caption("A geometrical view of the chosen symmetry element in the entered conventional cell; not a full stereographic projection of the group.")
        with st.expander("Show exact coordinate matrix and numerical check", expanded=False):
            st.code("\n".join("[ " + "  ".join(row) + " ]" for row in op.matrix), language="text")
            st.caption(f"Full symmetry group metric-preservation residual: {inventory.metric_preservation_maximum_residual:.2e}")
            st.caption("rotation axis [uvw] is a direct-space direction; mirror plane (hkl) is a reciprocal-space covector. They are not interchangeable.")
>>>>>>> 2fce910d70d4b7cc9f6cc900b26f34ec5c719f62

        with st.expander("Understand the symmetry elements and their directions", expanded=False):
            st.write(
                "Each operation maps the crystal onto itself. The operation count is NOT "
                "always the number of geometric axes: +90° and −90° can share one axis."
            )
            for family in guide.families:
                line = f"**{family.name}** — {family.operations} operation(s)"
                if family.geometric_elements:
                    axes = family.geometric_elements
                    label = ('(' + ', '.join(str(v) for v in axes[0]) + ')') if 'plane' in family.coordinate_type else ('[' + ', '.join(str(v) for v in axes[0]) + ']')
                    line += f" · {len(axes)} distinct indexed {'planes' if 'plane' in family.coordinate_type else 'axes'} · e.g. {label}"
                st.markdown(line)
            st.caption(
                "A rotation axis [uvw] denotes a direct-space direction. A mirror plane (hkl) "
                "is a reciprocal-space covector. They are not interchangeable. "
                "These indices follow the selected conventional setting."
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
                B = cell_basis(*[float(v) for v in cell])
                fig = make_operation_scene(B, op.kind, op.axis_or_plane)
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": False})
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
        a = _number("a", a0, key=f"{prefix}_a")
    with lengths[1]:
        if family in {"cubic", "tetragonal", "hexagonal", "trigonal"}:
            b = _locked_number("b", a, key=f"{prefix}_b_locked_{family}")
        else:
            b = _number("b", b0, key=f"{prefix}_b")
    with lengths[2]:
        if family == "cubic":
            c = _locked_number("c", a, key=f"{prefix}_c_locked_{family}")
        else:
            c = _number("c", c0, key=f"{prefix}_c")

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
        ("1", "0", "0"),
        ("0", "1", "0"),
        ("0", "0", "1"),
    ),
) -> CorrespondenceUIResult:
    st.subheader("Lattice correspondence")
    st.caption("This 3×3 matrix tells which parent crystal directions map to which product directions. It does not represent a symmetry operation.")
    choice = st.radio(
        "Which way are the crystal directions mapped?",
        ("A → M", "M → A"),
        horizontal=True,
        key=f"{prefix}_direction_choice",
        help=(
            "Choose the direction of the matrix you are entering. The scientific backend always "
            "receives the canonical A → M action. A reverse matrix is inverted exactly."
        ),
    )
    direction = "A_TO_M" if choice == "A → M" else "M_TO_A"
    arrow, equation = displayed_relation(direction)
    st.markdown(f"**{arrow}**")
    st.caption(equation)

    rows: list[list[str]] = []
    for i in range(3):
        cols = st.columns(3)
        row: list[str] = []
        for j in range(3):
            with cols[j]:
                row.append(
                    st.text_input(
                        f"C{i + 1}{j + 1}",
                        value=str(default_rows[i][j]),
                        key=f"{prefix}_{direction}_{i}_{j}",
                        label_visibility="collapsed",
                    )
                )
        rows.append(row)

    canonical = canonical_correspondence_rows(rows, direction=direction)
    preview = preview_correspondence(rows, direction=direction)
    st.markdown("**How this matrix maps the parent basis**")
    for parent_axis, product_coords in preview.basis_mappings:
        st.markdown(f"**{parent_axis}** → **{product_coords}**")
    st.caption("Every column is one parent basis direction expressed in product coordinates. This is a lattice correspondence, not a rigid rotation or an orientation relationship.")
    st.caption(f"Exact correspondence determinant: {preview.determinant} (nonzero = invertible).")
    with st.expander("Show the exact canonical A → M matrix sent to the solver", expanded=False):
        st.code("\n".join("[ " + "  ".join(row) + " ]" for row in canonical), language="text")
        if direction == "M_TO_A":
            st.caption("The displayed matrix is the exact symbolic inverse of the entered M → A matrix.")
    return CorrespondenceUIResult(
        canonical_rows=canonical,
        input_direction=direction,
        display_arrow=arrow,
        display_equation=equation,
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
