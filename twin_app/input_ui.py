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
    """Compact, auditable Cayron parent→daughter rational-coordinate correspondence.

    Only the exact C returned from this function enters the crystallographic solver.
    The probe, inverse and optional 3D views are explanatory and read-only.
    """
    from .correspondence_explorer import (
        calculate_action, images_of_parent_basis, vector_text,
    )

    st.subheader("1 · Correspondence: which daughter coordinates match each parent direction?")
    st.latex(r"[\mathbf u]_{B_c^{\gamma}}=C_c^{\alpha\rightarrow\gamma}[\mathbf u]_{B_c^{\alpha}}")
    st.caption("Parent α (austenite) → daughter γ (martensite). **C maps coordinates**, "
               "not a physical rotation, orientation relationship or deformation F.")
    with st.expander("What exactly do the rows and columns mean?", expanded=False):
        st.write("Each **column** tells where one parent crystal basis vector maps in daughter "
                 "direct coordinates. Each **row** is a daughter coordinate component. "
                 "Enter exact integers or rational fractions p/q, such as `1/2` or `-1/2`.")
        st.write("The crystal lattice metrics are independent inputs. "
                 "Coordinate components from different bases must not be used to infer a Cartesian angle.")

    baseline = tuple(tuple(str(v) for v in row) for row in default_rows)
    keys = tuple(tuple(f"{prefix}_A_TO_M_{i}_{j}" for j in range(3)) for i in range(3))

    def restore_starting_matrix() -> None:
        # This is solely a user-requested restoration of the original INPUT preset,
        # never a hardcoded numerical scientific answer or classification.
        for i in range(3):
            for j in range(3):
                st.session_state[keys[i][j]] = baseline[i][j]

    st.markdown("**Edit the exact 3 × 3 matrix C**")
    st.caption("Columns: parent a, b, c → · Rows: daughter a, b, c ↓")
    header = st.columns([0.75, 1, 1, 1], gap="small")
    for col, title in zip(header, ("**M ↓ / A →**", "**aₐ**", "**bₐ**", "**cₐ**")):
        with col:
            st.markdown(title)
    rows: list[list[str]] = []
    for i, label in enumerate(("**aₘ**", "**bₘ**", "**cₘ**")):
        columns = st.columns([0.75, 1, 1, 1], gap="small")
        with columns[0]:
            st.markdown(label)
        row = []
        for j in range(3):
            with columns[j + 1]:
                row.append(st.text_input(
                    f"Daughter {i+1}, parent {j+1}",
                    value=baseline[i][j],
                    key=keys[i][j], label_visibility="collapsed",
                ))
        rows.append(row)

    # An accidental input change previously produced nine different NiTi-like
    # modes while the user believed they were viewing the published NiTi preset.
    # Highlight any departure from the starting correspondence without claiming
    # that the edited matrix is necessarily scientifically invalid.
    changed = [(i, j) for i in range(3) for j in range(3)
               if str(rows[i][j]).strip() != baseline[i][j]]
    if changed:
        cells = ", ".join(f"C{i+1}{j+1}" for i, j in changed)
        st.warning(f"**Edited correspondence ({len(changed)} entries: {cells}).** "
                   "These results will NOT necessarily reproduce the starting published example. "
                   "This may be intentional for a different transformation.")
        st.button("Restore starting correspondence entries", key=f"{prefix}_restore_C",
                  on_click=restore_starting_matrix, width="content")
    try:
        canonical = canonical_correspondence_rows(rows, direction="A_TO_M")
        C = exact_matrix(canonical)
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        st.error(f"Correspondence is not usable: {exc}")
        st.stop()
        raise AssertionError("unreachable")

    st.caption(f"Exact rational C accepted · determinant = {C.det()} · invertible")
    st.markdown("**Try a parent direction → see its daughter coordinates**")
    choice = st.selectbox(
        "Parent direction",
        ("a-axis [1,0,0]", "b-axis [0,1,0]", "c-axis [0,0,1]", "Custom [u,v,w]"),
        key=f"{prefix}_probe_choice",
    )
    axes = (("1", "0", "0"), ("0", "1", "0"), ("0", "0", "1"))
    if choice == "Custom [u,v,w]":
        fields = st.columns(3, gap="small")
        parent_vector = tuple(
            fields[k].text_input(label, value=v, key=f"{prefix}_probe_{k}")
            for k, (label, v) in enumerate(zip(("u (parent)", "v (parent)", "w (parent)"), ("1", "1", "0")))
        )
    else:
        parent_vector = axes[("a-axis [1,0,0]", "b-axis [0,1,0]", "c-axis [0,0,1]").index(choice)]

    try:
        action = calculate_action(canonical, parent_vector)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        st.error(f"Invalid direction: {exc}; enter rational integers or p/q fractions.")
    else:
        st.markdown(f"**Parent {vector_text(action.parent)}  →  Daughter {vector_text(action.daughter)}**")
        st.latex(sp.latex(C) + r"\," + sp.latex(sp.Matrix(action.parent)) + "=" + sp.latex(sp.Matrix(action.daughter)))

        with st.expander("More coordinate mappings and optional 3D drawings", expanded=False):
            st.caption("For the three basis axes (each vector is a COLUMN of C):")
            actions = images_of_parent_basis(canonical)
            for label, obj in zip(("aₐ", "bₐ", "cₐ"), actions):
                st.write(f"{label} → daughter {vector_text(obj.daughter)}")
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
                    st.caption("Parent direct-coordinate components")
                    st.plotly_chart(arrow_figure(action.parent, "Parent"),
                                    width="stretch", key=f"{prefix}_plot_parent")
                with right:
                    st.caption("Daughter direct-coordinate components")
                    st.plotly_chart(arrow_figure(action.daughter, "Daughter"),
                                    width="stretch", key=f"{prefix}_plot_daughter")
                st.caption("These drawings use separate coordinate axes. No physical angle can be deduced.")

    with st.expander("Expert details: exact C, inverse and determinant", expanded=False):
        st.code("\n".join("[ " + "  ".join(row) + " ]" for row in canonical), language="text")
        st.write(f"det(C) = {C.det()}")
        st.code(str(C.inv()), language="text")
        st.caption("C.inv() is the exact reverse coordinate mapping. It is not another independently measured correspondence.")

    arrow, equation = displayed_relation("A_TO_M")
    return CorrespondenceUIResult(
        canonical_rows=canonical, input_direction="A_TO_M",
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
