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


def _render_operation_summary(inventory: SymmetryInventory, *, parent: bool) -> None:
    """Compact but *specific* composition of the actual selected finite group.

    Counts and axes/planes come from exact metric-validated operations. A
    geometric candidate is NEVER presented as a predicted physical twin.
    """
    counts = dict(inventory.counts)
    short = {
        "identity": "Identity",
        "inversion": "Inversion",
        "mirror reflection": "Mirror",
        "proper 2-fold rotation": "180° rotation",
        "proper 3-fold rotation": "120° rotation",
        "proper 4-fold rotation": "90° rotation",
        "proper 6-fold rotation": "60° rotation",
    }
    display = " · ".join(
        f"{count} {short.get(name, name)}"
        for name, count in inventory.counts
        if count > 0
    )
    st.markdown(
        f"**{inventory.symbol}** · {inventory.crystal_family.capitalize()} · "
        f"**{inventory.expected_order} operations**"
    )
    st.caption(f"Conventional setting: {inventory.conventional_setting}")
    st.caption(display)
    st.caption(
        "Parent symmetry generates candidate twin routes; product symmetry identifies equivalent "
        "correspondences." if parent else
        "Product symmetry identifies equivalent correspondences; it does not itself "
        "assign Type I, Type II or Compound."
    )

    with st.expander("Point-group contents: operations, axes and planes", expanded=False):
        st.caption(
            "Every row is derived from the selected exact group. "
            "A mirror is a plane, while a rotation is about an axis. "
            "No single operation proves a physical twin."
        )
        rows = []
        for kind, count in inventory.counts:
            members = [op for op in inventory.operations if op.kind == kind]
            items = []
            for op in members:
                if op.axis_or_plane is None:
                    continue
                notation = ("(" + ",".join(map(str, op.axis_or_plane)) + ")") if kind == "mirror reflection" else (
                    "[" + ",".join(map(str, op.axis_or_plane)) + "]"
                )
                if notation not in items:
                    items.append(notation)
            rows.append({
                "Symmetry element": short.get(kind, kind),
                "Count": count,
                "Axis [uvw] / plane (hkl)": ", ".join(items[:9]) + (
                    f" · +{len(items)-9} more" if len(items)>9 else ""
                ) if items else "—",
            })
        st.table(rows)
        st.markdown(
            "**What these elements mean**"
            "\n- **Identity**: no change of coordinates."
            "\n- **Inversion**: maps (x,y,z) to (−x,−y,−z)."
            "\n- **Mirror**: reflection in a plane (hkl)."
            "\n- **Proper n-fold rotation**: turns 360°/n about [uvw]."
        )
        st.caption(
            "rotation axis [uvw] is a direct-space direction; "
            "mirror-plane covector (hkl) is reciprocal-space. "
            "These are not the same kind of object."
        )
        options = tuple(range(len(inventory.operations)))
        operation_id = st.selectbox(
            "Inspect one exact symmetry operation",
            options,
            format_func=lambda i: (
                f"G{inventory.operations[i].index} · "
                f"{inventory.operations[i].kind} · order {inventory.operations[i].order}"
            ),
            key=f"inspect_group_{'parent' if parent else 'product'}_{inventory.symbol}",
        )
        op = inventory.operations[operation_id]
        if op.kind == "mirror reflection" and op.axis_or_plane is not None:
            st.markdown(f"**mirror-plane covector (hkl)**: `{op.axis_or_plane}`")
        elif "rotation" in op.kind and op.axis_or_plane is not None:
            st.markdown(f"**rotation axis [uvw]**: `{op.axis_or_plane}`")
        st.caption(op.twin_role)
        st.code("\n".join("[ " + "  ".join(row) + " ]" for row in op.matrix), language="text")
        st.caption(
            f"Metric-preservation residual (maximum over group): "
            f"{inventory.metric_preservation_maximum_residual:.3e}"
        )

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
            f"{symbol}  ·  {by_symbol[symbol]['conventional_setting']}  ·  "
            f"order {by_symbol[symbol]['order']}"
        ),
        help="The exact operations of the selected group are shown below after the cell is defined.",
    )

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
    _render_operation_summary(inventory, parent=(role == "parent"))
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
    choice = st.radio(
        "Matrix direction",
        ("A → M", "M → A"),
        horizontal=True,
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
    with st.expander("Weak-plane lattice basis (advanced)", expanded=False):
        st.caption(
            "Point-group symmetry does not determine Bravais centering. Weak-plane enumeration "
            "therefore requires an explicit primitive-node basis. Leaving this unspecified is safe: "
            "higher-order families are still identified, but no weak plane is invented."
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
        choice = st.selectbox("Product Bravais-node basis", choices)
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
