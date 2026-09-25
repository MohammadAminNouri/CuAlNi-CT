from __future__ import annotations

"""Small presentation helpers for the Streamlit workbench.

No function in this module performs crystallographic mathematics.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

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


def phase_editor(
    *,
    prefix: str,
    heading: str,
    defaults: PhaseDefaults,
    point_groups: Sequence[dict[str, object]],
    length_unit: str,
) -> PhaseInput:
    st.markdown(f"### {heading}")
    id_col, pg_col = st.columns([1.15, 1.0])
    with id_col:
        phase_id = st.text_input(
            "Phase ID",
            value=defaults.phase_id,
            key=f"{prefix}_phase_id",
            help="Stable identifier used inside the project file.",
        )
    with pg_col:
        symbols = [str(row["symbol"]) for row in point_groups]
        labels = {
            str(row["symbol"]): (
                f"{row['symbol']} · {row['crystal_family']} · {row['conventional_setting']}"
            )
            for row in point_groups
        }
        default_index = symbols.index(defaults.point_group) if defaults.point_group in symbols else 0
        point_group = st.selectbox(
            "Point group",
            symbols,
            index=default_index,
            format_func=lambda symbol: labels[symbol],
            key=f"{prefix}_point_group",
            help=(
                "All 32 conventional crystallographic point groups are supported. "
                "The selected conventional symmetry must preserve the entered metric."
            ),
        )

    label = st.text_input(
        "Display name",
        value=defaults.label,
        key=f"{prefix}_label",
    )

    st.caption("Lattice parameters")
    lengths = st.columns(3)
    a = lengths[0].number_input("a", min_value=1.0e-12, value=float(defaults.a), format="%.8f", key=f"{prefix}_a")
    b = lengths[1].number_input("b", min_value=1.0e-12, value=float(defaults.b), format="%.8f", key=f"{prefix}_b")
    c = lengths[2].number_input("c", min_value=1.0e-12, value=float(defaults.c), format="%.8f", key=f"{prefix}_c")

    angles = st.columns(3)
    alpha = angles[0].number_input("α (deg)", min_value=0.000001, max_value=179.999999, value=float(defaults.alpha), format="%.8f", key=f"{prefix}_alpha")
    beta = angles[1].number_input("β (deg)", min_value=0.000001, max_value=179.999999, value=float(defaults.beta), format="%.8f", key=f"{prefix}_beta")
    gamma = angles[2].number_input("γ (deg)", min_value=0.000001, max_value=179.999999, value=float(defaults.gamma), format="%.8f", key=f"{prefix}_gamma")

    with st.expander("Representation metadata", expanded=False):
        representation = st.text_input(
            "Cell representation",
            value=defaults.representation,
            key=f"{prefix}_representation",
            help="Descriptive label only; no hidden basis conversion is inferred.",
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
        lattice=LatticeInput(a, b, c, alpha, beta, gamma, length_unit),
    )


def correspondence_editor(
    *,
    defaults: Sequence[Sequence[str]] = (
        ("1", "0", "0"),
        ("0", "1", "0"),
        ("0", "0", "1"),
    ),
) -> tuple[tuple[str, str, str], tuple[str, str, str], tuple[str, str, str]]:
    st.markdown("### Correspondence matrix  C(M ← A)")
    st.caption(
        "Enter the crystallographic correspondence explicitly. Integer, decimal, "
        "and exact rational entries such as 1/2 are accepted; the UI does not infer C."
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
                        key=f"C_{i}_{j}",
                        label_visibility="collapsed",
                    )
                )
        rows.append(tuple(row))  # type: ignore[arg-type]
    return tuple(rows)  # type: ignore[return-value]


def matrix_frame(matrix: object) -> pd.DataFrame:
    return pd.DataFrame(
        matrix,
        index=["1", "2", "3"],
        columns=["1", "2", "3"],
    )


def vector_frame(vectors: Iterable[Sequence[float]], *, prefix: str) -> pd.DataFrame:
    rows = []
    for index, vector in enumerate(vectors, start=1):
        rows.append({"solution": f"{prefix}{index}", "x1": vector[0], "x2": vector[1], "x3": vector[2]})
    return pd.DataFrame(rows)


def render_application_error(exc: ApplicationError) -> None:
    info = exc.info
    st.error(f"**{info.title}**\n\n{info.message}")
    if info.hint:
        st.info(info.hint)
    if info.technical_detail:
        with st.expander("Technical detail", expanded=False):
            st.code(info.technical_detail)
