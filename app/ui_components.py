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
