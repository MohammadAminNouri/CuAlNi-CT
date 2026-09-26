from __future__ import annotations

"""Presentation-only helpers for the public crystallography workstation.

The helpers in this module never evaluate CT, PTMC, Ball-James, Mallard,
cofactor, correspondence, or orientation equations.  They collect user input,
apply conventional crystal-system cell constraints for convenience, and render
scientific objects returned by the Python backend.
"""

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st

from .application import LatticeInput, PhaseInput
from .errors import ApplicationError


@dataclass(frozen=True)
class PhaseDefaults:
    phase_id: str
    label: str
    point_group: str
    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    representation: str = "conventional"


@dataclass(frozen=True)
class ConstrainedCell:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    independent: tuple[str, ...]
    note: str


def constrain_cell(
    crystal_family: str,
    *,
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> ConstrainedCell:
    """Apply the standard conventional-cell constraints used by the UI.

    This is an input convenience only.  The project loader remains the final
    scientific validator and checks that the selected point group preserves the
    resulting metric.
    """

    family = str(crystal_family).strip().lower()
    if family == "cubic":
        return ConstrainedCell(a, a, a, 90.0, 90.0, 90.0, ("a",), "a=b=c; α=β=γ=90°")
    if family == "tetragonal":
        return ConstrainedCell(a, a, c, 90.0, 90.0, 90.0, ("a", "c"), "a=b; α=β=γ=90°")
    if family == "orthorhombic":
        return ConstrainedCell(a, b, c, 90.0, 90.0, 90.0, ("a", "b", "c"), "α=β=γ=90°")
    if family in {"hexagonal", "trigonal"}:
        note = "hexagonal axes: a=b; α=β=90°, γ=120°"
        if family == "trigonal":
            note += "; built-in trigonal point groups use hexagonal axes"
        return ConstrainedCell(a, a, c, 90.0, 90.0, 120.0, ("a", "c"), note)
    if family == "monoclinic":
        return ConstrainedCell(a, b, c, 90.0, beta, 90.0, ("a", "b", "c", "beta"), "unique-b setting: α=γ=90°")
    if family == "triclinic":
        return ConstrainedCell(a, b, c, alpha, beta, gamma, ("a", "b", "c", "alpha", "beta", "gamma"), "all six cell parameters are independent")
    return ConstrainedCell(a, b, c, alpha, beta, gamma, ("a", "b", "c", "alpha", "beta", "gamma"), "no UI constraint applied")


def _number(
    label: str,
    value: float,
    *,
    key: str,
    disabled: bool = False,
    angle: bool = False,
) -> float:
    kwargs: dict[str, object] = {
        "label": label,
        "value": float(value),
        "format": "%.8f",
        "key": key,
        "disabled": disabled,
    }
    if angle:
        kwargs.update(min_value=0.000001, max_value=179.999999)
    else:
        kwargs.update(min_value=1.0e-12)
    return float(st.number_input(**kwargs))


def smart_phase_editor(
    *,
    prefix: str,
    heading: str,
    defaults: PhaseDefaults,
    point_groups: Sequence[dict[str, object]],
    length_unit: str,
) -> PhaseInput:
    """Compact crystal-family-aware phase editor."""

    st.markdown(f"### {heading}")
    label = st.text_input("Phase name", value=defaults.label, key=f"{prefix}_label")

    symbols = [str(row["symbol"]) for row in point_groups]
    metadata = {str(row["symbol"]): row for row in point_groups}
    labels = {
        symbol: (
            f"{symbol}  ·  {metadata[symbol]['crystal_family']}  ·  "
            f"{metadata[symbol]['conventional_setting']}"
        )
        for symbol in symbols
    }
    default_index = symbols.index(defaults.point_group) if defaults.point_group in symbols else 0
    point_group = st.selectbox(
        "Point group",
        symbols,
        index=default_index,
        format_func=lambda symbol: labels[symbol],
        key=f"{prefix}_point_group",
        help=(
            "The built-in registry contains all 32 conventional crystallographic point groups. "
            "The scientific loader independently verifies that the selected symmetry preserves the entered metric."
        ),
    )
    family = str(metadata[point_group]["crystal_family"])
    setting = str(metadata[point_group]["conventional_setting"])

    st.caption(f"{family.capitalize()} · {setting}")

    # Independent values are collected first. Values that are symmetry-constrained
    # are derived below rather than silently accepted and discarded.
    if family == "cubic":
        a = _number("a", defaults.a, key=f"{prefix}_a")
        cell = constrain_cell(family, a=a, b=a, c=a, alpha=90, beta=90, gamma=90)
        st.caption("b = c = a; α = β = γ = 90°")
    elif family in {"tetragonal", "hexagonal", "trigonal"}:
        col1, col2 = st.columns(2)
        with col1:
            a = _number("a", defaults.a, key=f"{prefix}_a")
        with col2:
            c = _number("c", defaults.c, key=f"{prefix}_c")
        cell = constrain_cell(family, a=a, b=a, c=c, alpha=90, beta=90, gamma=(120 if family in {"hexagonal", "trigonal"} else 90))
        st.caption(cell.note)
    elif family == "orthorhombic":
        cols = st.columns(3)
        with cols[0]:
            a = _number("a", defaults.a, key=f"{prefix}_a")
        with cols[1]:
            b = _number("b", defaults.b, key=f"{prefix}_b")
        with cols[2]:
            c = _number("c", defaults.c, key=f"{prefix}_c")
        cell = constrain_cell(family, a=a, b=b, c=c, alpha=90, beta=90, gamma=90)
        st.caption(cell.note)
    elif family == "monoclinic":
        cols = st.columns(4)
        with cols[0]:
            a = _number("a", defaults.a, key=f"{prefix}_a")
        with cols[1]:
            b = _number("b", defaults.b, key=f"{prefix}_b")
        with cols[2]:
            c = _number("c", defaults.c, key=f"{prefix}_c")
        with cols[3]:
            beta = _number("β (deg)", defaults.beta, key=f"{prefix}_beta", angle=True)
        cell = constrain_cell(family, a=a, b=b, c=c, alpha=90, beta=beta, gamma=90)
        st.caption(cell.note)
    else:
        lengths = st.columns(3)
        with lengths[0]:
            a = _number("a", defaults.a, key=f"{prefix}_a")
        with lengths[1]:
            b = _number("b", defaults.b, key=f"{prefix}_b")
        with lengths[2]:
            c = _number("c", defaults.c, key=f"{prefix}_c")
        angles = st.columns(3)
        with angles[0]:
            alpha = _number("α (deg)", defaults.alpha, key=f"{prefix}_alpha", angle=True)
        with angles[1]:
            beta = _number("β (deg)", defaults.beta, key=f"{prefix}_beta", angle=True)
        with angles[2]:
            gamma = _number("γ (deg)", defaults.gamma, key=f"{prefix}_gamma", angle=True)
        cell = constrain_cell(family, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

    with st.expander("Advanced phase metadata", expanded=False):
        phase_id = st.text_input(
            "Phase ID",
            value=defaults.phase_id,
            key=f"{prefix}_phase_id",
            help="Stable identifier stored in the project file; ordinary calculations do not require changing it.",
        )
        representation = st.text_input(
            "Cell representation label",
            value=defaults.representation,
            key=f"{prefix}_representation",
        )
        physical_phase = st.text_input(
            "Physical phase label",
            value=defaults.label,
            key=f"{prefix}_physical_phase",
        )

    return PhaseInput(
        phase_id=phase_id,
        label=label,
        physical_phase=physical_phase,
        cell_representation=representation,
        point_group=point_group,
        lattice=LatticeInput(
            cell.a,
            cell.b,
            cell.c,
            cell.alpha,
            cell.beta,
            cell.gamma,
            length_unit,
        ),
    )


def correspondence_editor(
    *,
    defaults: Sequence[Sequence[str]] = (
        ("1", "0", "0"),
        ("0", "1", "0"),
        ("0", "0", "1"),
    ),
    key_prefix: str = "C",
) -> tuple[tuple[str, str, str], tuple[str, str, str], tuple[str, str, str]]:
    st.markdown("### Lattice correspondence")
    st.latex(r"C_{M\leftarrow A}:\;u_M=C_{M\leftarrow A}u_A")
    st.caption(
        "Enter C explicitly. Integers, decimals and exact fractions (for example 1/2) are accepted. "
        "Plane covectors are mapped by the inverse transpose in the backend."
    )
    rows: list[tuple[str, str, str]] = []
    for i in range(3):
        cols = st.columns(3)
        row: list[str] = []
        for j in range(3):
            with cols[j]:
                row.append(
                    st.text_input(
                        f"C{i + 1}{j + 1}",
                        value=str(defaults[i][j]),
                        key=f"{key_prefix}_{i}_{j}",
                        label_visibility="collapsed",
                    )
                )
        rows.append(tuple(row))  # type: ignore[arg-type]
    return tuple(rows)  # type: ignore[return-value]


def matrix_editor(
    title: str,
    *,
    default: Sequence[Sequence[float]] | None = None,
    key_prefix: str,
    help_text: str = "",
) -> list[list[float]]:
    st.markdown(f"#### {title}")
    if help_text:
        st.caption(help_text)
    base = default or ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    rows: list[list[float]] = []
    for i in range(3):
        cols = st.columns(3)
        row: list[float] = []
        for j in range(3):
            with cols[j]:
                row.append(
                    float(
                        st.number_input(
                            f"{key_prefix}{i + 1}{j + 1}",
                            value=float(base[i][j]),
                            format="%.10g",
                            key=f"{key_prefix}_{i}_{j}",
                            label_visibility="collapsed",
                        )
                    )
                )
        rows.append(row)
    return rows


def vector_editor(
    title: str,
    *,
    default: Sequence[float] = (1.0, 0.0, 0.0),
    key_prefix: str,
) -> list[float]:
    st.markdown(f"##### {title}")
    cols = st.columns(3)
    out: list[float] = []
    for i in range(3):
        with cols[i]:
            out.append(
                float(
                    st.number_input(
                        f"{key_prefix}{i + 1}",
                        value=float(default[i]),
                        format="%.10g",
                        key=f"{key_prefix}_{i}",
                        label_visibility="collapsed",
                    )
                )
            )
    return out


def matrix_frame(matrix: object, *, digits: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame(matrix, index=["1", "2", "3"], columns=["1", "2", "3"])
    return frame.applymap(lambda x: round(float(x), digits) if isinstance(x, (int, float)) else x)


def vector_frame(vectors: Iterable[Sequence[float]], *, prefix: str) -> pd.DataFrame:
    rows = []
    for index, vector in enumerate(vectors, start=1):
        rows.append({"solution": f"{prefix}{index}", "x1": vector[0], "x2": vector[1], "x3": vector[2]})
    return pd.DataFrame(rows)


def compact_key_value(rows: Sequence[tuple[str, object]]) -> None:
    st.dataframe(pd.DataFrame(rows, columns=["quantity", "value"]), hide_index=True, use_container_width=True)


def render_application_error(exc: ApplicationError) -> None:
    info = exc.info
    st.error(f"**{info.title}**\n\n{info.message}")
    if info.hint:
        st.info(info.hint)
    if info.technical_detail:
        with st.expander("Technical detail", expanded=False):
            st.code(info.technical_detail)


def result_badge(label: str, ok: bool | None) -> None:
    if ok is True:
        st.success(label)
    elif ok is False:
        st.warning(label)
    else:
        st.info(label)


def variant_table(variants: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    rows = []
    for item in variants:
        rows.append(
            {
                "variant": int(item.get("index", 0)) + 1,
                "disorientation from base (deg)": item.get("misorientation_from_base_deg"),
                "equivalent matrices": item.get("equivalent_proper_matrix_count"),
                "reference symmetry": item.get("reference_symmetry_index"),
            }
        )
    return pd.DataFrame(rows)
