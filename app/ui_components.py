from __future__ import annotations

"""Presentation helpers for the Streamlit crystallography workstation.

These helpers do not implement CT, PTMC, Ball-James, Mallard, cofactor,
correspondence, OR, EBSD, or crystallographic-metric mathematics.  They only
collect explicit user input and render backend results.
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


FAMILY_ORDER = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
)


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
    family = str(crystal_family).strip().lower()
    if family == "cubic":
        return ConstrainedCell(
            a, a, a, 90.0, 90.0, 90.0, ("a",),
            "Cubic constraint: a = b = c and α = β = γ = 90°."
        )
    if family == "tetragonal":
        return ConstrainedCell(
            a, a, c, 90.0, 90.0, 90.0, ("a", "c"),
            "Tetragonal constraint: a = b and α = β = γ = 90°."
        )
    if family == "orthorhombic":
        return ConstrainedCell(
            a, b, c, 90.0, 90.0, 90.0, ("a", "b", "c"),
            "Orthorhombic constraint: α = β = γ = 90°."
        )
    if family in {"hexagonal", "trigonal"}:
        note = "Hexagonal-axis constraint: a = b, α = β = 90°, γ = 120°."
        if family == "trigonal":
            note += " The built-in trigonal entries use hexagonal axes."
        return ConstrainedCell(
            a, a, c, 90.0, 90.0, 120.0, ("a", "c"), note
        )
    if family == "monoclinic":
        return ConstrainedCell(
            a, b, c, 90.0, beta, 90.0, ("a", "b", "c", "beta"),
            "Unique-b monoclinic setting: α = γ = 90°."
        )
    if family == "triclinic":
        return ConstrainedCell(
            a, b, c, alpha, beta, gamma,
            ("a", "b", "c", "alpha", "beta", "gamma"),
            "Triclinic cell: all six parameters are independent."
        )
    return ConstrainedCell(
        a, b, c, alpha, beta, gamma,
        ("a", "b", "c", "alpha", "beta", "gamma"),
        "No UI symmetry constraint is defined for this setting."
    )


def _number(
    label: str,
    value: float,
    *,
    key: str,
    angle: bool = False,
) -> float:
    kwargs: dict[str, object] = {
        "label": label,
        "value": float(value),
        "format": "%.8f",
        "key": key,
    }
    if angle:
        kwargs.update(min_value=0.000001, max_value=179.999999)
    else:
        kwargs.update(min_value=1.0e-12)
    return float(st.number_input(**kwargs))


def _locked_number(
    label: str, value: float, *, key: str, angle: bool = False
) -> float:
    # Synchronize a constrained display field before the widget is instantiated.
    # This prevents stale disabled values when an independent lattice parameter changes.
    st.session_state[key] = float(value)
    kwargs: dict[str, object] = {
        "label": label,
        "value": float(value),
        "format": "%.8f",
        "disabled": True,
        "key": key,
    }
    if angle:
        kwargs.update(min_value=0.000001, max_value=179.999999)
    else:
        kwargs.update(min_value=1.0e-12)
    return float(st.number_input(**kwargs))


def _default_family(
    defaults: PhaseDefaults, point_groups: Sequence[dict[str, object]]
) -> str:
    for row in point_groups:
        if str(row.get("symbol")) == defaults.point_group:
            return str(row.get("crystal_family", "triclinic")).lower()
    return "triclinic"


def _representation_label(family: str, setting: str) -> str:
    family = family.lower()
    setting_l = setting.lower().replace("_", " ")
    if family == "cubic":
        return "conventional cubic"
    if family == "monoclinic" and "unique" in setting_l and "b" in setting_l:
        return "conventional unique-b"
    if family in {"hexagonal", "trigonal"}:
        return "conventional hexagonal axes"
    return f"conventional {family}"


def smart_phase_editor(
    *,
    prefix: str,
    heading: str,
    defaults: PhaseDefaults,
    point_groups: Sequence[dict[str, object]],
    length_unit: str,
) -> PhaseInput:
    """Crystal-system-first phase editor with all six cell parameters visible."""

    st.markdown(f"### {heading}")
    label = st.text_input("Phase name", value=defaults.label, key=f"{prefix}_label")

    metadata = {str(row["symbol"]): row for row in point_groups}
    available = {str(row["crystal_family"]).lower() for row in point_groups}
    families = [name for name in FAMILY_ORDER if name in available]
    default_family = _default_family(defaults, point_groups)
    family = st.selectbox(
        "Crystal system",
        families,
        index=families.index(default_family) if default_family in families else 0,
        key=f"{prefix}_crystal_family",
    )

    symbols = [
        str(row["symbol"])
        for row in point_groups
        if str(row["crystal_family"]).lower() == family
    ]
    family_default = defaults.point_group if defaults.point_group in symbols else symbols[0]
    point_group = st.selectbox(
        "Point group",
        symbols,
        index=symbols.index(family_default),
        key=f"{prefix}_point_group_{family}",
        format_func=lambda symbol: (
            f"{symbol}  ·  {metadata[symbol]['conventional_setting']}"
        ),
        help=(
            "Only point groups belonging to the selected crystal system are shown. "
            "The backend still verifies symmetry/metric consistency independently."
        ),
    )
    setting = str(metadata[point_group]["conventional_setting"])

    st.markdown("##### Conventional cell")
    st.caption("Locked fields are symmetry constraints; they are shown rather than hidden.")

    # Lengths.  Independent fields keep stable session keys.  Constrained fields
    # intentionally have no key so their displayed value tracks the independent
    # parameter immediately instead of retaining stale widget state.
    lcols = st.columns(3)
    with lcols[0]:
        a = _number("a", defaults.a, key=f"{prefix}_a")
    with lcols[1]:
        if family in {"cubic", "tetragonal", "hexagonal", "trigonal"}:
            b = _locked_number("b", a, key=f"{prefix}_locked_b_{family}_{point_group}")
        else:
            b = _number("b", defaults.b, key=f"{prefix}_b")
    with lcols[2]:
        if family == "cubic":
            c = _locked_number("c", a, key=f"{prefix}_locked_c_{family}_{point_group}")
        else:
            c = _number("c", defaults.c, key=f"{prefix}_c")

    acols = st.columns(3)
    with acols[0]:
        if family == "triclinic":
            alpha = _number("α (deg)", defaults.alpha, key=f"{prefix}_alpha", angle=True)
        else:
            alpha = _locked_number("α (deg)", 90.0, key=f"{prefix}_locked_alpha_{family}_{point_group}", angle=True)
    with acols[1]:
        if family in {"triclinic", "monoclinic"}:
            beta = _number("β (deg)", defaults.beta, key=f"{prefix}_beta", angle=True)
        else:
            beta = _locked_number("β (deg)", 90.0, key=f"{prefix}_locked_beta_{family}_{point_group}", angle=True)
    with acols[2]:
        if family == "triclinic":
            gamma = _number("γ (deg)", defaults.gamma, key=f"{prefix}_gamma", angle=True)
        elif family in {"hexagonal", "trigonal"}:
            gamma = _locked_number("γ (deg)", 120.0, key=f"{prefix}_locked_gamma_{family}_{point_group}", angle=True)
        else:
            gamma = _locked_number("γ (deg)", 90.0, key=f"{prefix}_locked_gamma_{family}_{point_group}", angle=True)

    cell = constrain_cell(
        family,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    st.caption(cell.note)

    representation = _representation_label(family, setting)
    with st.expander("Phase metadata", expanded=False):
        phase_id = st.text_input(
            "Phase ID",
            value=defaults.phase_id,
            key=f"{prefix}_phase_id",
            help="Stable identifier used by the project file and analysis links.",
        )
        st.text_input(
            "Cell representation",
            value=representation,
            disabled=True,
            key=f"{prefix}_representation_{family}_{point_group}",
            help="Derived from the selected crystallographic setting; not free text.",
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
    """Collect a correspondence matrix without changing the backend contract.

    The backend always receives the canonical app convention

        u_M = C_(M<-A) u_A.

    Cayron's paper writes the same numerical matrix as C^(M->A) while still
    using u_M = C^(M->A) u_A.  The selector below therefore changes notation
    only; it does NOT invert or transpose the matrix entries.
    """

    st.markdown("### Lattice correspondence")

    # Always show our actual solver contract first.
    st.caption("Canonical solver convention")
    st.latex(r"C_{M\leftarrow A}:\;u_M=C_{M\leftarrow A}u_A")

    notation = st.radio(
        "Input notation",
        options=("app", "cayron"),
        index=0,
        horizontal=True,
        key=f"{key_prefix}_notation",
        format_func=lambda value: (
            "App notation  C_(M←A)"
            if value == "app"
            else "Cayron paper notation  C^(M→A)"
        ),
        help=(
            "Both choices describe the same numerical correspondence matrix "
            "u_M = C u_A. The backend contract never changes."
        ),
    )

    if notation == "cayron":
        st.caption("Cayron paper notation")
        st.latex(
            r"C_{\mathrm{Cayron}}^{M\to A}:\;"
            r"u_M=C_{\mathrm{Cayron}}^{M\to A}u_A"
        )
        st.info(
            "Cayron's C^(M→A) is the same numerical matrix that the app stores "
            "as C_(M←A). The UI normalizes only the notation before calculation; "
            "the matrix entries are NOT inverted or transposed. "
            "Do not paste Cayron's inverse C^(A→M) in this mode."
        )
    else:
        st.caption(
            "Enter the canonical app matrix directly: parent/austenite "
            "coordinates in, product/martensite coordinates out."
        )

    st.caption(
        "Enter the correspondence explicitly. Integers, decimals and exact fractions "
        "such as 1/3 are accepted. Plane covectors use the inverse transpose."
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

    if notation == "cayron":
        st.caption(
            "Canonical matrix sent to the solver: "
            "C_(M←A) = C_Cayron^(M→A) — same numerical entries."
        )

    # CRITICAL: return exactly the same object shape and numerical entries as
    # before.  This keeps every existing backend/solver path untouched.
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
    base = default or (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
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
    default: Sequence[float] = (0.0, 0.0, 0.0),
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


def scientific_number(value: object, *, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if number == 0.0:
            return "0"
        if abs(number) < 1.0e-4 or abs(number) >= 1.0e5:
            return f"{number:.{digits}e}"
        return f"{number:.{digits}g}"
    return str(value)


def matrix_frame(matrix: object, *, digits: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame(matrix)
    frame.index = [str(i + 1) for i in range(frame.shape[0])]
    frame.columns = [str(i + 1) for i in range(frame.shape[1])]

    def formatter(value: object) -> object:
        if isinstance(value, (int, float)):
            number = float(value)
            if number != 0.0 and abs(number) < 10 ** (-max(4, digits // 2)):
                return f"{number:.4e}"
            return round(number, digits)
        return value

    if hasattr(frame, "map"):
        return frame.map(formatter)
    return frame.applymap(formatter)  # pragma: no cover - old pandas fallback


def vector_frame(
    vectors: Iterable[Sequence[float]], *, prefix: str
) -> pd.DataFrame:
    rows = []
    for index, vector in enumerate(vectors, start=1):
        rows.append(
            {
                "solution": f"{prefix}{index}",
                "x1": vector[0],
                "x2": vector[1],
                "x3": vector[2],
            }
        )
    return pd.DataFrame(rows)


def compact_key_value(rows: Sequence[tuple[str, object]]) -> None:
    st.dataframe(
        pd.DataFrame(rows, columns=["quantity", "value"]),
        hide_index=True,
        use_container_width=True,
    )


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
        # OrientationVariant.index is already the scientific variant number.
        rows.append(
            {
                "variant": int(item.get("index", 0)),
                "disorientation from base (deg)": item.get(
                    "misorientation_from_base_deg"
                ),
                "equivalent matrices": item.get("equivalent_proper_matrix_count"),
                "reference symmetry": item.get("reference_symmetry_index"),
            }
        )
    return pd.DataFrame(rows)
