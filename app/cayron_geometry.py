from __future__ import annotations

"""Read-only Cayron-CT geometry for the theory-comparison presentation layer.

The functions in this module never invoke a CT, Ball--James, PTMC, twinning,
orientation, or EBSD solver.  They only turn already-calculated CT outputs into
metric-orthonormal plotting geometry.
"""

from typing import Any, Mapping

import numpy as np
import pandas as pd
import streamlit as st


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _result(response: Any) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        return _mapping(response.get("result", response))
    return _mapping(getattr(response, "result", {}))


def _row_attr(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _enum_text(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value))


def _ct_rows(unified: Any, kind: str) -> tuple[Any, ...]:
    if unified is None:
        return ()
    raw = unified.get("rows", ()) if isinstance(unified, Mapping) else getattr(unified, "rows", ())
    selected: list[Any] = []
    for row in tuple(raw or ()):
        theory = _enum_text(_row_attr(row, "theory"))
        row_kind = _enum_text(_row_attr(row, "prediction_kind", _row_attr(row, "kind")))
        if theory == "cayron_ct" and row_kind == kind:
            selected.append(row)
    return tuple(selected)


def _symmetric_sqrt(metric: Any) -> np.ndarray:
    matrix = np.asarray(metric, dtype=float).reshape(3, 3)
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    if np.any(values <= 0.0):
        raise ValueError("Parent metric must be positive definite for plotting")
    return (vectors * np.sqrt(values)) @ vectors.T


def _symmetric_inv_sqrt(metric: Any) -> np.ndarray:
    matrix = np.asarray(metric, dtype=float).reshape(3, 3)
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    if np.any(values <= 0.0):
        raise ValueError("Parent metric must be positive definite for plotting")
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def plane_covector_to_orthonormal_normal(parent_metric: Any, plane: Any) -> np.ndarray:
    """Convert a parent crystal plane covector to the plotted orthonormal normal."""

    p = np.asarray(plane, dtype=float).reshape(3)
    normal = _symmetric_inv_sqrt(parent_metric) @ p
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-15:
        raise ValueError("Plane covector cannot be zero")
    return normal / norm


def direct_vector_to_orthonormal(parent_metric: Any, vector: Any) -> np.ndarray:
    """Convert parent direct-lattice coordinates into the plotted orthonormal frame."""

    v = _symmetric_sqrt(parent_metric) @ np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(v))
    if norm <= 1.0e-15:
        raise ValueError("Direct vector cannot be zero")
    return v / norm


def _plane_mesh(normal: Any, *, extent: float = 1.35, samples: int = 21) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=float).reshape(3)
    n /= np.linalg.norm(n)
    seed = np.array([1.0, 0.0, 0.0]) if abs(float(n[0])) < 0.85 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    grid = np.linspace(-extent, extent, samples)
    uu, vv = np.meshgrid(grid, grid)
    xyz = uu[..., None] * e1 + vv[..., None] * e2
    return xyz[..., 0], xyz[..., 1], xyz[..., 2]


def cmc_cone_mesh(cmc_normalized: Any, *, extent: float = 1.35, axis_samples: int = 35, angular_samples: int = 72) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return the non-degenerate real zero-cone of the normalized CMC.

    This is a plotting transform only.  Exact/approximate compatibility labels
    always come from the authoritative backend result, never from this mesh.
    """

    cmc = np.asarray(cmc_normalized, dtype=float).reshape(3, 3)
    cmc = 0.5 * (cmc + cmc.T)
    values, vectors = np.linalg.eigh(cmc)
    scale = max(float(np.max(np.abs(values))), 1.0)
    tol = 1.0e-12 * scale
    pos = [i for i, value in enumerate(values) if value > tol]
    neg = [i for i, value in enumerate(values) if value < -tol]
    if not ((len(pos) == 2 and len(neg) == 1) or (len(pos) == 1 and len(neg) == 2)):
        return None

    if len(neg) == 1:
        axis = neg[0]
        radial = pos
        axis_value = -values[axis]
        radial_coeff = [float(np.sqrt(axis_value / values[index])) for index in radial]
    else:
        axis = pos[0]
        radial = neg
        axis_value = values[axis]
        radial_coeff = [float(np.sqrt(axis_value / (-values[index]))) for index in radial]

    t = np.linspace(-extent, extent, axis_samples)
    theta = np.linspace(0.0, 2.0 * np.pi, angular_samples)
    tt, aa = np.meshgrid(t, theta, indexing="ij")
    coordinates = np.zeros((3, axis_samples, angular_samples), dtype=float)
    coordinates[axis] = tt
    coordinates[radial[0]] = np.abs(tt) * radial_coeff[0] * np.cos(aa)
    coordinates[radial[1]] = np.abs(tt) * radial_coeff[1] * np.sin(aa)
    rotated = np.einsum("ij,jkl->ikl", vectors, coordinates)
    return rotated[0], rotated[1], rotated[2]


def _new_figure(title: str):
    try:
        import plotly.graph_objects as go
    except Exception:
        return None, None
    figure = go.Figure()
    figure.update_layout(
        title=title,
        scene={
            "xaxis_title": "x1",
            "yaxis_title": "x2",
            "zaxis_title": "x3",
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 48},
        height=560,
        showlegend=True,
    )
    return go, figure


def _add_plane(figure: Any, go: Any, normal: Any, *, name: str, opacity: float) -> None:
    x, y, z = _plane_mesh(normal)
    figure.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            opacity=opacity,
            showscale=False,
            name=name,
            hovertemplate=name + "<extra></extra>",
        )
    )


def _add_vector(figure: Any, go: Any, vector: Any, *, name: str, scale: float = 1.2) -> None:
    v = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(v))
    if norm <= 1.0e-15:
        return
    v = scale * v / norm
    figure.add_trace(
        go.Scatter3d(
            x=[0.0, float(v[0])],
            y=[0.0, float(v[1])],
            z=[0.0, float(v[2])],
            mode="lines+markers",
            name=name,
        )
    )


def _render_cmc_geometry(response: Any) -> None:
    result = _result(response)
    metric = _mapping(result.get("metric"))
    ct = _mapping(result.get("ct_detail"))
    cmc = metric.get("cmc_normalized")
    parent_metric = metric.get("parent_metric")
    if cmc is None or parent_metric is None:
        st.info("CMC geometry is unavailable because the calculated result does not expose the parent metric and normalized CMC.")
        return

    st.latex(r"u_A^T\,CMC\,u_A=0")
    st.caption(
        "The surface shows directions whose length is preserved by the correspondence. "
        "It is rendered in the parent metric-orthonormal frame. Plotting never changes the backend exact/diagnostic classification."
    )

    go, figure = _new_figure("Cayron CMC preserved-length geometry")
    if figure is None:
        st.warning("3D plotting support is unavailable in this environment. The numerical CT result is unaffected.")
        return

    mesh = cmc_cone_mesh(cmc)
    if mesh is not None:
        x, y, z = mesh
        figure.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                opacity=0.58,
                showscale=False,
                name="current CMC zero-surface",
                hovertemplate="current CMC zero-surface<extra></extra>",
            )
        )

    exact = bool(ct.get("exact_compatible", False))
    order = int(ct.get("degeneracy_order", 0) or 0)
    exact_planes = tuple(ct.get("exact_habit_planes_parent_covectors", ()) or ())
    approximate = _mapping(ct.get("approximate_diagnostic"))
    diagnostic_planes = tuple(approximate.get("candidate_planes_parent_covectors", ()) or ())

    if exact and order == 3:
        st.success("Third-order CMC degeneracy: CMC = 0 in the exact classification, so every direction is preserved and no unique finite habit-plane list exists.")
    elif exact_planes:
        for index, plane in enumerate(exact_planes):
            normal = plane_covector_to_orthonormal_normal(parent_metric, plane)
            _add_plane(figure, go, normal, name=f"exact CT habit plane {index + 1}", opacity=0.48)
        st.success(f"Exact CT degeneracy: {len(exact_planes)} exact habit-plane branch(es) are overlaid.")
    elif exact:
        st.info("Exact compatibility is classified by the backend, but no explicit finite habit-plane branch is exposed for this degeneracy order.")
    else:
        if diagnostic_planes:
            for index, plane in enumerate(diagnostic_planes):
                normal = plane_covector_to_orthonormal_normal(parent_metric, plane)
                _add_plane(figure, go, normal, name=f"diagnostic projected plane {index + 1}", opacity=0.22)
            st.warning(
                "The translucent planes are nearest-degeneracy diagnostics only. "
                "They are NOT exact CT habit planes and are never used as an exact supercompatibility seed."
            )
        else:
            st.info("No exact or diagnostic habit-plane branch is available for the current CMC state.")

    if mesh is None and not exact_planes and not (exact and order == 3):
        st.caption("The current quadratic zero-set is not a non-degenerate real double-cone under the plotted numerical spectrum.")

    st.plotly_chart(figure, use_container_width=True, key="cayron_v11_cmc_geometry")

    eta = ct.get("eta_eigenvalues")
    mu = ct.get("generalized_mu")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "classification": "exact CT A/M",
                    "value": "yes" if exact else "no",
                },
                {"classification": "degeneracy order", "value": order},
                {"classification": "eta spectrum", "value": str(eta)},
                {"classification": "mu spectrum", "value": str(mu)},
                {
                    "classification": "nearest-zero residual",
                    "value": ct.get("nearest_zero_residual"),
                },
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )


def _render_mm_geometry(response: Any, unified: Any) -> None:
    rows = _ct_rows(unified, "ct_mm_twin")
    if not rows:
        st.info("No CT M/M twin rows are available in the current theory inventory. Run Theory comparison → Conclusions → Calculation scope → Calculate / update.")
        return
    result = _result(response)
    parent_metric = _mapping(result.get("metric")).get("parent_metric")
    if parent_metric is None:
        st.info("Parent metric is unavailable for the M/M geometry view.")
        return

    labels = [str(_row_attr(row, "branch_label", f"CT twin {index}")) for index, row in enumerate(rows)]
    selected_index = st.selectbox(
        "CT M/M branch",
        options=list(range(len(rows))),
        format_func=lambda index: labels[index],
        key="cayron_v11_mm_branch",
    )
    row = rows[int(selected_index)]
    plane = _row_attr(row, "twin_plane_parent_crystal")
    direction = _row_attr(row, "twin_direction_parent_crystal")
    if plane is None or direction is None:
        st.info("The selected native CT row does not expose both parent-basis twin plane and direction.")
        return

    normal = plane_covector_to_orthonormal_normal(parent_metric, plane)
    shear_direction = direct_vector_to_orthonormal(parent_metric, direction)
    go, figure = _new_figure("Selected CT martensite/martensite twin geometry")
    if figure is None:
        st.warning("3D plotting support is unavailable in this environment.")
        return
    _add_plane(figure, go, normal, name="M/M twin plane", opacity=0.45)
    _add_vector(figure, go, shear_direction, name="M/M twin shear direction")
    st.plotly_chart(figure, use_container_width=True, key="cayron_v11_mm_geometry")
    st.caption(
        f"Native branch: {labels[int(selected_index)]}. Shear magnitude = {_row_attr(row, 'shear_magnitude', 'N/A')}. "
        "This is a CT transformation-twin construction; experimental occurrence remains a separate question."
    )


def _render_am_shear_geometry(response: Any, unified: Any) -> None:
    rows = tuple(
        row
        for row in _ct_rows(unified, "ct_am_habit")
        if _row_attr(row, "exact") is True
        and _row_attr(row, "habit_plane_parent_crystal") is not None
        and _row_attr(row, "shape_vector_parent_crystal") is not None
    )
    result = _result(response)
    metric = _mapping(result.get("metric"))
    parent_metric = metric.get("parent_metric")
    if parent_metric is None:
        st.info("Parent metric is unavailable for the A/M shear geometry view.")
        return
    if not rows:
        if metric.get("smc_dimensional") is not None:
            st.info(
                "The SMC matrix is available, but an exact Cayron d_A vector is plotted only when an exact CT habit-plane seed exists in the current theory inventory. "
                "Diagnostic near-degeneracy planes are not promoted to exact A/M shear."
            )
        else:
            st.info("No exact CT A/M shear row is available in the current theory inventory.")
        return

    labels = [str(_row_attr(row, "branch_label", f"CT A/M habit {index}")) for index, row in enumerate(rows)]
    selected_index = st.selectbox(
        "Exact CT A/M branch",
        options=list(range(len(rows))),
        format_func=lambda index: labels[index],
        key="cayron_v11_am_branch",
    )
    row = rows[int(selected_index)]
    normal = plane_covector_to_orthonormal_normal(parent_metric, _row_attr(row, "habit_plane_parent_crystal"))
    d_vector = direct_vector_to_orthonormal(parent_metric, _row_attr(row, "shape_vector_parent_crystal"))
    go, figure = _new_figure("Exact CT A/M habit plane and SMC-derived d_A")
    if figure is None:
        st.warning("3D plotting support is unavailable in this environment.")
        return
    _add_plane(figure, go, normal, name="exact A/M habit plane", opacity=0.45)
    _add_vector(figure, go, d_vector, name="Cayron d_A")
    st.plotly_chart(figure, use_container_width=True, key="cayron_v11_am_shear_geometry")
    st.caption("Cayron d_A is the SMC-derived A/M displacement/shear vector. It is not relabelled as the Ball–James shape vector b.")


def _render_super_geometry(response: Any, unified: Any) -> None:
    rows = _ct_rows(unified, "ct_supercompatibility")
    if not rows:
        st.info(
            "No CT supercompatibility rows are available. Exact evaluation requires an exact A/M habit/shear seed and an eligible M/M twin; when those exist, enable Theory comparison → Conclusions → Calculation scope → CT supercompatibility and Calculate / update."
        )
        return
    result = _result(response)
    parent_metric = _mapping(result.get("metric")).get("parent_metric")
    if parent_metric is None:
        st.info("Parent metric is unavailable for the A/M/M geometry view.")
        return

    labels = [str(_row_attr(row, "branch_label", f"CT super branch {index}")) for index, row in enumerate(rows)]
    selected_index = st.selectbox(
        "CT A/M/M branch",
        options=list(range(len(rows))),
        format_func=lambda index: labels[index],
        key="cayron_v11_super_branch",
    )
    row = rows[int(selected_index)]
    habit = _row_attr(row, "habit_plane_parent_crystal")
    twin_plane = _row_attr(row, "twin_plane_parent_crystal")
    d_vector = _row_attr(row, "shape_vector_parent_crystal")
    twin_direction = _row_attr(row, "twin_direction_parent_crystal")
    if any(item is None for item in (habit, twin_plane, d_vector, twin_direction)):
        st.info("The selected CT supercompatibility row does not expose all four geometric objects required for the overlay.")
        return

    habit_normal = plane_covector_to_orthonormal_normal(parent_metric, habit)
    twin_normal = plane_covector_to_orthonormal_normal(parent_metric, twin_plane)
    d_plot = direct_vector_to_orthonormal(parent_metric, d_vector)
    a_plot = direct_vector_to_orthonormal(parent_metric, twin_direction)
    go, figure = _new_figure("CT A/M/M shear–shear geometry")
    if figure is None:
        st.warning("3D plotting support is unavailable in this environment.")
        return
    _add_plane(figure, go, habit_normal, name="A/M habit plane", opacity=0.32)
    _add_plane(figure, go, twin_normal, name="M/M twin plane", opacity=0.32)
    _add_vector(figure, go, d_plot, name="A/M d_A")
    _add_vector(figure, go, a_plot, name="M/M twin direction a")
    st.plotly_chart(figure, use_container_width=True, key="cayron_v11_super_geometry")
    residuals = _mapping(_row_attr(row, "residuals", {}))
    st.latex(r"2(m_A^T n)d_A=a")
    st.caption(
        "Displayed from one already-calculated CT A/M/M row. "
        f"Native supercompatibility residual: {residuals.get('ct_supercompatibility_dimensionless', 'N/A')}."
    )


def render_cayron_geometry(response: Any, unified: Any) -> None:
    """Render read-only geometric interpretations of existing Cayron outputs."""

    with st.expander(
        "Cayron geometry — CMC cone, habit planes, twins and shear/shear overlay",
        expanded=False,
    ):
        st.caption(
            "Presentation-only view of already-calculated CT quantities. "
            "Nothing in this panel changes the ProjectState, reruns a solver, widens a tolerance, or upgrades a diagnostic branch to an exact one."
        )
        cmc_tab, mm_tab, am_tab, super_tab = st.tabs(
            [
                "CMC cone / habit planes",
                "M/M twin geometry",
                "A/M SMC shear",
                "A/M/M supercompatibility",
            ]
        )
        with cmc_tab:
            _render_cmc_geometry(response)
        with mm_tab:
            _render_mm_geometry(response, unified)
        with am_tab:
            _render_am_shear_geometry(response, unified)
        with super_tab:
            _render_super_geometry(response, unified)
