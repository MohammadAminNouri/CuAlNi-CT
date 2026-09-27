from __future__ import annotations

"""Streamlit front end for the CuAlNi-CT scientific workbench.

The front end does not implement crystallographic theory.  Every scientific
result is produced by the existing application/workbench/backend services and
is bound to the exact inputs that produced it.
"""

from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from app.application import (
    CalculationRequest,
    TransformationInput,
    calculate_payload,
    calculate_request,
    point_group_options,
)
from app.errors import ApplicationError, input_error, stale_state_error
from app.session_state import (
    BoundResult,
    fingerprint,
    get_bound,
    invalidate_or_dependents,
    invalidate_transformation_dependents,
    mark_calculation_failure,
    mark_calculation_success,
    observe_draft,
    put_bound,
)
from app.ui_components import (
    PhaseDefaults,
    compact_key_value,
    correspondence_editor,
    matrix_editor,
    matrix_frame,
    render_application_error,
    scientific_number,
    smart_phase_editor,
    variant_table,
    vector_editor,
)
from app.workbench import (
    calpad_low_index_table,
    calpad_normal_conversion,
    calpad_phase_cell,
    manual_ptmc_cofactor_analysis,
    map_correspondence_object,
    map_orientation_object,
    martensite_variant_pair_analysis,
    orientation_from_euler,
    orientation_from_matrix,
    orientation_from_parallelisms,
    orientation_from_polar_correspondence,
    reconstruct_sample_orientations,
)
from cualni_cryst.ebsd_io import load_ang, load_ctf
from cualni_cryst.ebsd_map import AngleUnit, audit_map
from cualni_cryst.orientation import EulerConvention
from cualni_cryst.representation import CartesianConvention


# ---------------------------------------------------------------------------
# General rendering/state helpers
# ---------------------------------------------------------------------------


def _fmt(value: object, digits: int = 10) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number == 0.0:
            return "0"
        if abs(number) < 1.0e-4 or abs(number) >= 1.0e5:
            return f"{number:.{max(2, digits - 1)}e}"
        return f"{number:.{digits}g}"
    return str(value)


def _sci(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}e}"
    except (TypeError, ValueError):
        return str(value)


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _render_matrix(title: str, matrix: object, *, expanded: bool = False) -> None:
    with st.expander(title, expanded=expanded):
        st.dataframe(matrix_frame(matrix), use_container_width=True)


def _phase_names(payload: Mapping[str, object]) -> tuple[str, str, str, str]:
    phases = payload.get("phases", [])
    transformations = payload.get("transformations", [])
    if (
        not isinstance(phases, list)
        or len(phases) < 2
        or not isinstance(transformations, list)
        or not transformations
    ):
        return "Parent", "Product", "phase_A", "phase_M"
    t = transformations[0]
    if not isinstance(t, Mapping):
        return "Parent", "Product", "phase_A", "phase_M"
    parent_id = str(t.get("parent_phase_id", "phase_A"))
    product_id = str(t.get("product_phase_id", "phase_M"))
    by_id = {
        str(item.get("phase_id")): item
        for item in phases
        if isinstance(item, Mapping)
    }
    parent = by_id.get(parent_id, {})
    product = by_id.get(product_id, {})
    return (
        str(parent.get("label", parent_id)),
        str(product.get("label", product_id)),
        parent_id,
        product_id,
    )


def _require_project() -> tuple[dict[str, object], object]:
    if st.session_state.get("requires_recalculation", False):
        raise stale_state_error()
    payload = st.session_state.get("current_project_payload")
    response = st.session_state.get("current_response")
    if not isinstance(payload, dict) or response is None:
        raise input_error(
            "No calculated transformation is active.",
            hint="Define or open a project in Setup and calculate the transformation.",
        )
    return payload, response


def _normalize_compact_crystal_text(text: str) -> str:
    """Accept common compact 3-index notation such as [010] and [10-1].

    Multi-digit indices remain separator-explicit because [120] is inherently
    ambiguous between 1,2,0 and 12,0,... outside the conventional single-digit
    shorthand.
    """

    raw = str(text).strip()
    if len(raw) < 3 or raw[0] not in "[(" or raw[-1] not in "])":
        return raw
    body = raw[1:-1].strip()
    if " " in body or "," in body:
        return raw
    tokens: list[str] = []
    i = 0
    while i < len(body):
        if body[i] == "-":
            if i + 1 >= len(body) or not body[i + 1].isdigit():
                return raw
            tokens.append("-" + body[i + 1])
            i += 2
        elif body[i].isdigit():
            tokens.append(body[i])
            i += 1
        else:
            return raw
    if len(tokens) != 3:
        return raw
    return f"{raw[0]}{' '.join(tokens)}{raw[-1]}"


def _draft_snapshot(
    *,
    project_id: str,
    title: str,
    length_unit: str,
    parent: object,
    product: object,
    transformation_id: str,
    transformation_label: str,
    correspondence: Sequence[Sequence[str]],
) -> dict[str, object]:
    def phase_snapshot(phase: object) -> dict[str, object]:
        lattice = phase.lattice  # type: ignore[attr-defined]
        return {
            "phase_id": phase.phase_id,  # type: ignore[attr-defined]
            "label": phase.label,  # type: ignore[attr-defined]
            "physical_phase": phase.physical_phase,  # type: ignore[attr-defined]
            "point_group": phase.point_group,  # type: ignore[attr-defined]
            "cell": {
                "a": lattice.a,
                "b": lattice.b,
                "c": lattice.c,
                "alpha_deg": lattice.alpha_deg,
                "beta_deg": lattice.beta_deg,
                "gamma_deg": lattice.gamma_deg,
                "length_unit": lattice.length_unit,
            },
        }

    return {
        "project_id": str(project_id),
        "title": str(title),
        "length_unit": str(length_unit),
        "parent": phase_snapshot(parent),
        "product": phase_snapshot(product),
        "transformation": {
            "transformation_id": str(transformation_id),
            "label": str(transformation_label),
            "correspondence": [list(map(str, row)) for row in correspondence],
        },
    }


def _seed_setup_from_payload(payload: Mapping[str, object]) -> None:
    """Seed Setup widgets before they are instantiated after opening a project."""

    phases = payload.get("phases", [])
    transforms = payload.get("transformations", [])
    if not isinstance(phases, list) or len(phases) < 2 or not transforms:
        return
    transformation = transforms[0]
    if not isinstance(transformation, Mapping):
        return
    by_id = {
        str(item.get("phase_id")): item
        for item in phases
        if isinstance(item, Mapping)
    }
    parent_id = str(transformation.get("parent_phase_id", "phase_A"))
    product_id = str(transformation.get("product_phase_id", "phase_M"))
    pg_rows = point_group_options()
    pg_family = {
        str(row["symbol"]): str(row["crystal_family"]).lower() for row in pg_rows
    }

    st.session_state["project_title"] = str(payload.get("title", "Untitled transformation"))
    st.session_state["project_id"] = str(payload.get("project_id", "workbench_project"))

    for prefix, phase_id in (("parent", parent_id), ("product", product_id)):
        phase = by_id.get(phase_id)
        if not isinstance(phase, Mapping):
            continue
        cell = phase.get("cell", {})
        if not isinstance(cell, Mapping):
            cell = {}
        point_group = str(phase.get("point_group", "1"))
        family = pg_family.get(point_group, "triclinic")
        st.session_state[f"{prefix}_crystal_family"] = family
        st.session_state[f"{prefix}_point_group_{family}"] = point_group
        st.session_state[f"{prefix}_label"] = str(phase.get("label", phase_id))
        st.session_state[f"{prefix}_phase_id"] = phase_id
        st.session_state[f"{prefix}_physical_phase"] = str(
            phase.get("physical_phase", phase.get("label", phase_id))
        )
        for name, default in (
            ("a", 1.0),
            ("b", 1.0),
            ("c", 1.0),
            ("alpha", 90.0),
            ("beta", 90.0),
            ("gamma", 90.0),
        ):
            source = f"{name}_deg" if name in {"alpha", "beta", "gamma"} else name
            if source in cell:
                st.session_state[f"{prefix}_{name}"] = float(cell.get(source, default))
        if "length_unit" in cell:
            st.session_state["length_unit"] = str(cell["length_unit"])

    st.session_state["transformation_id"] = str(
        transformation.get("transformation_id", "A_to_M")
    )
    st.session_state["transformation_label"] = str(
        transformation.get("label", "Parent → Product")
    )
    matrix = transformation.get("correspondence_M_from_A", [])
    if isinstance(matrix, list) and len(matrix) == 3:
        for i, row in enumerate(matrix):
            if isinstance(row, list) and len(row) == 3:
                for j, value in enumerate(row):
                    st.session_state[f"C_{i}_{j}"] = str(value)


def _consume_pending_project_load() -> None:
    response = st.session_state.pop("pending_loaded_response", None)
    payload = st.session_state.pop("pending_loaded_payload", None)
    if response is None or not isinstance(payload, dict):
        return
    _seed_setup_from_payload(payload)
    st.session_state["current_response"] = response
    st.session_state["current_project_payload"] = payload
    st.session_state["pending_accept_loaded_draft"] = True
    st.session_state["requires_recalculation"] = False
    invalidate_transformation_dependents(st.session_state)


def _projective_display(values: Sequence[object]) -> list[float]:
    arr = np.asarray(values, dtype=float).reshape(3)
    scale = float(np.max(np.abs(arr)))
    if scale <= 0.0:
        return [0.0, 0.0, 0.0]
    return [float(x) for x in arr / scale]


def _primitive_exact(exact: Sequence[object]) -> list[int] | None:
    try:
        fractions = [Fraction(str(value)) for value in exact]
    except (ValueError, ZeroDivisionError):
        return None
    denominators = [item.denominator for item in fractions]
    lcm = 1
    for den in denominators:
        lcm = math.lcm(lcm, den)
    ints = [item.numerator * (lcm // item.denominator) for item in fractions]
    gcd = 0
    for item in ints:
        gcd = math.gcd(gcd, abs(item))
    if gcd:
        ints = [item // gcd for item in ints]
    return ints


def _render_mapping_result(title: str, result: Mapping[str, object]) -> None:
    st.caption(title)
    rows: list[tuple[str, object]] = [
        ("Source", result.get("source_phase_id")),
        ("Target", result.get("target_phase_id")),
        ("Object", result.get("object_kind")),
        ("Input", result.get("input")),
    ]
    exact = result.get("mapped_exact_coefficients")
    numeric = result.get("mapped_numeric_coefficients") or result.get("mapped_coefficients")
    if isinstance(exact, list):
        rows.append(("Mapped exact coefficients", exact))
        primitive = _primitive_exact(exact)
        if primitive is not None:
            rows.append(("Primitive exact ratio", primitive))
    if isinstance(numeric, list):
        rows.append(("Mapped coefficients", numeric))
        rows.append(("Scale-normalized", _projective_display(numeric)))
    rows.append(("Relation", result.get("relation") or result.get("mapping")))
    compact_key_value(rows)
    with st.expander("Raw result"):
        st.json(_jsonable(result))


def _relative_matrix_residual(a: np.ndarray, b: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0)
    return float(np.linalg.norm(a - b) / scale)


def _twin_groups(relations: Sequence[Mapping[str, object]]) -> list[list[Mapping[str, object]]]:
    groups: list[list[Mapping[str, object]]] = []
    for relation in relations:
        a = np.asarray(relation["twin_a"], dtype=float).reshape(3)
        n = np.asarray(relation["twin_n"], dtype=float).reshape(3)
        dyad = np.outer(a, n)
        placed = False
        for group in groups:
            ga = np.asarray(group[0]["twin_a"], dtype=float).reshape(3)
            gn = np.asarray(group[0]["twin_n"], dtype=float).reshape(3)
            gdyad = np.outer(ga, gn)
            if _relative_matrix_residual(dyad, gdyad) <= 1.0e-9:
                group.append(relation)
                placed = True
                break
        if not placed:
            groups.append([relation])
    return groups


def _clean_projective_notation(text: object, kind: str, sense: str) -> str:
    notation = str(text)
    if sense != "projective":
        return notation
    # Projective ± identification is not a crystallographic symmetry family.
    if kind == "plane" and notation.startswith("{") and notation.endswith("}"):
        return "(" + notation[1:-1] + ")"
    if kind == "direction" and notation.startswith("<") and notation.endswith(">"):
        return "[" + notation[1:-1] + "]"
    return notation


def _target_kind(text: str) -> str | None:
    stripped = str(text).strip()
    if stripped.startswith("["):
        return "direction"
    if stripped.startswith("("):
        return "plane"
    return None


def _normal_parallel_ranking(
    payload: Mapping[str, object],
    phase: str,
    target: str,
    candidate_kind: str,
    *,
    max_index: int,
    limit: int,
) -> dict[str, object]:
    source_kind = _target_kind(target)
    if source_kind is None or source_kind == candidate_kind:
        raise input_error(
            "Normal-parallel ranking is defined here only for direction↔plane comparisons.",
            hint="Use a [uvw] target with plane candidates, or an (hkl) target with direction candidates.",
        )
    conversion = calpad_normal_conversion(
        payload, phase, target, max_index=max_index
    )
    coeff = conversion["raw_target_coefficients"]
    if not isinstance(coeff, list) or len(coeff) != 3:
        raise input_error("Normal conversion did not return three target coefficients.")
    body = " ".join(f"{float(x):.17g}" for x in coeff)
    derived = f"({body})" if candidate_kind == "plane" else f"[{body}]"
    ranked = calpad_low_index_table(
        payload,
        phase,
        derived,
        candidate_kind=candidate_kind,
        max_index=max_index,
        limit=limit,
        angle_sense="projective",
    )
    ranked = dict(ranked)
    ranked["normal_parallel_source"] = {
        "original_target": target,
        "converted_target": derived,
        "conversion": conversion,
        "definition": (
            "Candidates are ranked by the physical angle between their normal/direction "
            "and the metric-correct normal of the target object."
        ),
    }
    return ranked


# ---------------------------------------------------------------------------
# Scientific result renderers
# ---------------------------------------------------------------------------


def _render_transformation(response: object) -> None:
    result = response.result  # type: ignore[attr-defined]
    summary = result["summary"]
    code = summary["classification"]
    if code == "EXACT_CT_COMPATIBLE":
        st.success(f"Exact CT compatibility · {summary['ct_reason']}")
    elif code == "NOT_EXACT_NEAREST_DEGENERACY_DIAGNOSTIC_AVAILABLE":
        st.warning(
            "Not exactly CT-compatible. A nearest-degeneracy diagnostic is available; "
            "it is not reported as an exact solution."
        )
    else:
        st.info(f"Not exactly CT-compatible · {summary['ct_reason']}")

    cols = st.columns(6)
    cols[0].metric("λ₁", _fmt(summary["lambda1"], 12))
    cols[1].metric("λ₂", _fmt(summary["lambda2"], 12))
    cols[2].metric("λ₃", _fmt(summary["lambda3"], 12))
    cols[3].metric("|λ₂−1|", _fmt(summary["lambda2_residual"], 6))
    cols[4].metric("Variants", summary["topological_variant_count"])
    cols[5].metric("Operators", summary["operator_count"])

    overview, metric_tab, ct_tab, bj_tab, variants_tab = st.tabs(
        ["Summary", "Metric / stretch", "Correspondence Theory", "Ball–James", "Variants / topology"]
    )
    with overview:
        compact_key_value(
            [
                ("CT exact compatible", summary["ct_exact_compatible"]),
                ("CT degeneracy order", summary["degeneracy_order"]),
                ("Exact CT habit planes", summary["ct_exact_habit_plane_count"]),
                ("Ball–James λ₂ condition", summary["ball_james_lambda2_exact"]),
                ("Ball–James rank-one solutions", summary["ball_james_solution_count"]),
                ("Stretch variants", summary["variant_count"]),
            ]
        )
        st.markdown("#### Principal stretches")
        st.bar_chart(
            pd.DataFrame(
                {
                    "stretch": ["λ₁", "λ₂", "λ₃"],
                    "value": [summary["lambda1"], summary["lambda2"], summary["lambda3"]],
                }
            ).set_index("stretch")
        )

    with metric_tab:
        metric = result["metric"]
        _render_matrix("Parent metric M_A", metric["parent_metric"])
        _render_matrix("Product metric M_M", metric["product_metric"])
        _render_matrix("Pulled product metric Cᵀ M_M C", metric["pulled_product_metric"])
        _render_matrix("Dimensional CMC", metric["cmc_dimensional"], expanded=True)
        _render_matrix("Normalized CMC", metric["cmc_normalized"])
        _render_matrix("Dimensional SMC", metric["smc_dimensional"])
        _render_matrix("Right stretch U", metric["stretch"], expanded=True)
        _render_matrix("Principal axes", metric["principal_axes"])

    with ct_tab:
        ct = result["ct_detail"]
        compact_key_value(
            [
                ("Classification", ct["classification"]),
                ("Degeneracy order", ct["degeneracy_order"]),
                ("Reason", ct["reason"]),
                ("Nearest-zero residual", _sci(ct["nearest_zero_residual"], 6)),
                ("Inertia (−,0,+)", ct["inertia"]),
            ]
        )
        st.markdown("#### Generalized metric spectrum")
        st.dataframe(
            pd.DataFrame(
                {"η = μ−1": ct["eta_eigenvalues"], "μ": ct["generalized_mu"]},
                index=[1, 2, 3],
            ),
            use_container_width=True,
        )
        st.markdown("#### Exact habit-plane covectors in parent coordinates")
        planes = ct["exact_habit_planes_parent_covectors"]
        if planes:
            st.dataframe(
                pd.DataFrame(planes, columns=["p₁", "p₂", "p₃"]),
                use_container_width=True,
            )
        else:
            st.caption("No exact CT habit plane exists for this state.")
        approx = ct["approximate_diagnostic"]
        if not ct["exact_compatible"] and approx["candidate_planes_parent_covectors"]:
            with st.expander("Nearest-degeneracy diagnostic (not an exact CT solution)"):
                st.warning(approx["explanation"])
                st.write("Residual:", _sci(approx["residual"], 6))
                st.dataframe(
                    pd.DataFrame(
                        approx["candidate_planes_parent_covectors"],
                        columns=["p₁", "p₂", "p₃"],
                    ),
                    use_container_width=True,
                )

    with bj_tab:
        bj = result["ball_james_detail"]
        st.write("λ₂ criterion satisfied:", bj["lambda2_exact"])
        if not bj["solutions"]:
            st.caption("No single-variant Ball–James rank-one solution exists.")
        for index, solution in enumerate(bj["solutions"], 1):
            with st.expander(
                f"Solution {index} · branch {solution['branch']}", expanded=index == 1
            ):
                left, right = st.columns(2)
                with left:
                    compact_key_value(
                        [
                            ("Residual", _sci(solution["residual"], 6)),
                            ("Eigenvalues C", solution["eigenvalues_C"]),
                        ]
                    )
                    st.write("Shape vector a")
                    st.dataframe(
                        pd.DataFrame([solution["shape_vector_a"]], columns=["a₁", "a₂", "a₃"]),
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.write("Habit normal n")
                    st.dataframe(
                        pd.DataFrame([solution["habit_normal_n"]], columns=["n₁", "n₂", "n₃"]),
                        hide_index=True,
                        use_container_width=True,
                    )
                with right:
                    st.write("Rotation R")
                    st.dataframe(matrix_frame(solution["rotation"]), use_container_width=True)

    with variants_tab:
        topology = result["topology"]
        stretch = result["stretch_variants"]
        cols = st.columns(4)
        cols[0].metric("Parent group", topology["parent_group_order"])
        cols[1].metric("Product group", topology["product_group_order"])
        cols[2].metric("Variants", topology["n_variants"])
        cols[3].metric("Operators", topology["n_operators"])
        if topology.get("operator_summaries"):
            st.dataframe(
                pd.DataFrame(topology["operator_summaries"]),
                hide_index=True,
                use_container_width=True,
            )
        mats = stretch.get("variants", [])
        if mats:
            selected = st.selectbox(
                "Stretch variant",
                range(len(mats)),
                format_func=lambda i: f"U{i + 1}",
                key="transformation_variant_view",
            )
            st.dataframe(matrix_frame(mats[selected]), use_container_width=True)


def _render_orientation(analysis: Mapping[str, object]) -> None:
    report = analysis["report"]
    state = analysis["state"]
    st.info(str(analysis.get("origin_note", "")))
    cols = st.columns(5)
    cols[0].metric("OR variants", report["orientation_variant_count"])
    cols[1].metric("OR operators", report["orientation_operator_count"])
    cols[2].metric("Hᵀ proper order", report["proper_orientation_intersection_order"])
    cols[3].metric("Rotation residual", _sci(report["audit"]["maximum_residual"], 3))
    cols[4].metric("Parity residual", _sci(report["parity"]["maximum_residual"], 3))

    r1, r2 = st.columns([1.2, 1.0])
    with r1:
        st.markdown("#### Base orientation relationship")
        st.caption("Convention: x_parent = R(parent←product) · x_product")
        st.dataframe(matrix_frame(state["R_reference_from_moving"]), use_container_width=True)
    with r2:
        compact_key_value(
            [
                ("Axis", report["axis_angle"]["axis"]),
                ("Angle (deg)", report["axis_angle"]["angle_deg"]),
                (
                    "Euler ZXZ active",
                    [
                        report["euler_zxz_active"]["phi1_deg"],
                        report["euler_zxz_active"]["Phi_deg"],
                        report["euler_zxz_active"]["phi2_deg"],
                    ],
                ),
                (
                    "Euler ZXZ passive",
                    [
                        report["euler_zxz_passive"]["phi1_deg"],
                        report["euler_zxz_passive"]["Phi_deg"],
                        report["euler_zxz_passive"]["phi2_deg"],
                    ],
                ),
            ]
        )

    variants = analysis.get("variants", [])
    if isinstance(variants, list) and variants:
        st.markdown("#### Orientation variants")
        st.caption(f"{len(variants)} symmetry-distinct variants are returned by the backend.")
        st.dataframe(variant_table(variants), hide_index=True, use_container_width=True)
        selected = st.selectbox(
            "Inspect OR variant",
            range(len(variants)),
            format_func=lambda i: f"Variant {variants[i].get('index', i + 1)}",
            key="or_variant_inspect",
        )
        item = variants[selected]
        left, right = st.columns(2)
        with left:
            st.caption("Parent ← Product")
            st.dataframe(matrix_frame(item["R_parent_from_product"]), use_container_width=True)
            st.write(item["forward_axis_angle"])
        with right:
            st.caption("Product ← Parent")
            st.dataframe(matrix_frame(item["R_product_from_parent"]), use_container_width=True)
            st.write(item["reverse_axis_angle"])

    operators = analysis.get("operators", [])
    if isinstance(operators, list) and operators:
        with st.expander("Orientation operator classes"):
            st.dataframe(pd.DataFrame(operators), hide_index=True, use_container_width=True)
    with st.expander("Topology / convention audit"):
        st.json(_jsonable(report["cayron_topology_audit"]))
        for warning in report.get("warnings", []):
            st.caption(str(warning))


def _render_manual_ptmc(result: Mapping[str, object]) -> None:
    st.markdown("##### Manual shear result")
    cofactor = result["cofactor"]
    cols = st.columns(4)
    cols[0].metric("CC1", "pass" if cofactor["cc1_satisfied"] else "fail")
    cols[1].metric("CC2", "pass" if cofactor["cc2_satisfied"] else "fail")
    cols[2].metric("CC3", "pass" if cofactor["cc3_satisfied"] else "fail")
    cols[3].metric("PTMC solutions", len(result["ptmc"]))
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "CC1 residual": _sci(cofactor["cc1_residual"], 6),
                    "CC2 residual": _sci(cofactor["cc2_residual"], 6),
                    "CC2 simplified": _sci(cofactor["cc2_simplified"], 6),
                    "CC3 margin": _sci(cofactor["cc3_margin"], 6),
                }
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    if result["ptmc"]:
        st.dataframe(pd.DataFrame(result["ptmc"]), hide_index=True, use_container_width=True)
    else:
        st.caption(result.get("ptmc_note") or "No admissible classical single-shear PTMC solution.")
    with st.expander("Raw result"):
        st.json(_jsonable(result))


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="CuAlNi-CT Workbench",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
<style>
.block-container {padding-top: 1.0rem; padding-bottom: 4rem; max-width: 1540px;}
[data-testid="stSidebar"] .block-container {padding-top: 1.0rem;}
[data-testid="stMetricValue"] {font-size: 1.25rem;}
.app-kicker {font-size:.72rem; letter-spacing:.11em; text-transform:uppercase; opacity:.64;}
.app-title {font-size:2.05rem; font-weight:760; line-height:1.08; margin:.10rem 0 .16rem 0;}
.app-subtitle {font-size:.98rem; opacity:.78; max-width:1040px; margin-bottom:.7rem;}
.project-strip {border:1px solid rgba(128,128,128,.25); border-radius:.55rem; padding:.5rem .75rem; margin:.25rem 0 .75rem 0;}
.stale-strip {border-left:4px solid #d9a400; padding:.45rem .7rem; background:rgba(217,164,0,.08); margin:.35rem 0 .7rem 0;}
</style>
""",
        unsafe_allow_html=True,
    )

    _consume_pending_project_load()

    st.markdown('<div class="app-kicker">Phase-transformation crystallography workstation</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-title">CuAlNi-CT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Define a two-phase crystallographic state, test compatibility, build orientation relationships, reconstruct parent/product orientations, evaluate martensitic twin/PTMC conditions, audit EBSD maps, and inspect metric-correct direct/reciprocal geometry.</div>',
        unsafe_allow_html=True,
    )

    # Sidebar ---------------------------------------------------------------
    with st.sidebar:
        st.markdown("## Project")
        project_title = st.text_input(
            "Title", value="Untitled transformation", key="project_title"
        )
        length_unit = st.selectbox(
            "Length unit", ["angstrom", "nm", "pm"], index=0, key="length_unit"
        )
        with st.expander("Project metadata"):
            project_id = st.text_input(
                "Project ID", value="workbench_project", key="project_id"
            )
        st.divider()
        st.markdown("### Open project")
        uploaded = st.file_uploader(
            "Project JSON", type=["json"], label_visibility="collapsed"
        )
        if st.button(
            "Open and calculate",
            disabled=uploaded is None,
            use_container_width=True,
        ):
            try:
                payload = json.loads(uploaded.getvalue().decode("utf-8"))  # type: ignore[union-attr]
                if not isinstance(payload, dict):
                    raise ValueError("Top-level JSON must be an object.")
                response = calculate_payload(payload, source=uploaded.name)  # type: ignore[union-attr]
                st.session_state["pending_loaded_response"] = response
                st.session_state["pending_loaded_payload"] = response.project_payload
                st.session_state.pop("app_error", None)
                st.rerun()
            except ApplicationError as exc:
                st.session_state["app_error"] = exc
            except Exception as exc:
                st.session_state["app_error"] = input_error(
                    f"Uploaded project could not be opened: {exc}"
                )

        if st.session_state.get("current_response") is not None:
            if st.session_state.get("requires_recalculation", False):
                st.warning("Draft changed — recalculate")
            else:
                st.success("Calculated state active")

    if "app_error" in st.session_state:
        render_application_error(st.session_state["app_error"])

    current_payload = st.session_state.get("current_project_payload")
    if isinstance(current_payload, dict):
        parent_name, product_name, _, _ = _phase_names(current_payload)
        state_label = "stale draft" if st.session_state.get("requires_recalculation", False) else "calculated"
        st.markdown(
            f'<div class="project-strip"><b>{current_payload.get("title", "Project")}</b> &nbsp; · &nbsp; {parent_name} → {product_name} &nbsp; · &nbsp; {state_label}</div>',
            unsafe_allow_html=True,
        )

    (
        setup_tab,
        transformation_tab,
        orientation_tab,
        martensite_tab,
        calpad_tab,
        ebsd_tab,
        diagnostics_tab,
    ) = st.tabs(
        [
            "Setup",
            "Transformation",
            "OR & reconstruction",
            "Martensite",
            "CalPad",
            "EBSD",
            "Diagnostics / export",
        ]
    )

    active_exports: dict[str, object] = {}
    point_groups = point_group_options()

    # Setup -----------------------------------------------------------------
    with setup_tab:
        st.markdown("## Define the two phases")
        st.caption(
            "Select the crystal system first, then the point group. All six conventional-cell "
            "parameters remain visible; symmetry-constrained values are locked."
        )
        left, right = st.columns(2, gap="large")
        with left:
            parent = smart_phase_editor(
                prefix="parent",
                heading="Parent phase (A)",
                defaults=PhaseDefaults(
                    "phase_A", "Parent phase", "m-3m", 5.8, 5.8, 5.8, 90, 90, 90,
                    "conventional cubic",
                ),
                point_groups=point_groups,
                length_unit=length_unit,
            )
        with right:
            product = smart_phase_editor(
                prefix="product",
                heading="Product / daughter phase (M)",
                defaults=PhaseDefaults(
                    "phase_M", "Product phase", "2/m", 4.4, 5.3, 13.8, 90, 100, 90,
                    "conventional unique-b",
                ),
                point_groups=point_groups,
                length_unit=length_unit,
            )

        st.divider()
        mcol, info = st.columns([1.0, 1.15], gap="large")
        with mcol:
            correspondence = correspondence_editor()
        with info:
            st.markdown("### Transformation")
            st.write("The correspondence matrix maps parent lattice directions to product lattice directions.")
            st.caption("No correspondence is inferred from the phase names or lattice parameters.")
            with st.expander("Transformation metadata"):
                transformation_id = st.text_input(
                    "Transformation ID", value="A_to_M", key="transformation_id"
                )
                transformation_label = st.text_input(
                    "Display name", value="Parent → Product", key="transformation_label"
                )

        snapshot = _draft_snapshot(
            project_id=project_id,
            title=project_title,
            length_unit=length_unit,
            parent=parent,
            product=product,
            transformation_id=transformation_id,
            transformation_label=transformation_label,
            correspondence=correspondence,
        )
        draft_signature = fingerprint("setup", snapshot)

        if st.session_state.pop("pending_accept_loaded_draft", False):
            st.session_state["calculated_draft_signature"] = draft_signature
            st.session_state["requires_recalculation"] = False

        stale = observe_draft(st.session_state, draft_signature)
        if stale and st.session_state.get("current_response") is not None:
            st.markdown(
                '<div class="stale-strip"><b>Recalculation required.</b> Analysis tabs are blocked until this draft is calculated successfully.</div>',
                unsafe_allow_html=True,
            )

        calculate = st.button(
            "Calculate transformation", type="primary", use_container_width=True
        )
        if calculate:
            try:
                request = CalculationRequest(
                    project_id=project_id,
                    title=project_title,
                    parent=parent,
                    product=product,
                    transformation=TransformationInput.from_rows(
                        transformation_id,
                        parent.phase_id,
                        product.phase_id,
                        correspondence,
                        label=transformation_label,
                    ),
                    notes="Created with the CuAlNi-CT workbench.",
                )
                response = calculate_request(request)
                st.session_state["current_response"] = response
                st.session_state["current_project_payload"] = response.project_payload
                st.session_state.pop("app_error", None)
                mark_calculation_success(st.session_state, draft_signature)
                st.success("Transformation calculated from the current Setup state.")
            except ApplicationError as exc:
                mark_calculation_failure(st.session_state, exc)
                render_application_error(exc)

    # Transformation --------------------------------------------------------
    with transformation_tab:
        st.markdown("## Transformation analysis")
        try:
            _, response = _require_project()
            _render_transformation(response)
        except ApplicationError as exc:
            render_application_error(exc)

    # OR & reconstruction ---------------------------------------------------
    with orientation_tab:
        st.markdown("## Orientation relationship & reconstruction")
        st.caption(
            "R is a physical Cartesian rotation. C is a lattice correspondence. "
            "They are never substituted for one another."
        )
        try:
            payload, _ = _require_project()
            project_sig = str(st.session_state.get("calculated_draft_signature", ""))
            parent_name, product_name, parent_id, product_id = _phase_names(payload)
            source_mode = st.radio(
                "OR source",
                [
                    "Polar rotation candidate",
                    "Rotation matrix",
                    "Euler ZXZ",
                    "Two crystallographic parallelisms",
                ],
                horizontal=True,
                key="or_source_mode",
            )

            analysis: Mapping[str, object] | None = None
            active_or_sig = ""

            if source_mode == "Polar rotation candidate":
                st.caption(
                    "Finite-strain polar rotation of the correspondence deformation. "
                    "This is a candidate rotation, not experimental OR data."
                )
                or_sig = fingerprint(project_sig, source_mode)
                if st.button("Analyze polar rotation", type="primary"):
                    try:
                        invalidate_or_dependents(st.session_state)
                        result = orientation_from_polar_correspondence(payload)
                        put_bound(st.session_state, "orientation_result", or_sig, result)
                    except ApplicationError as exc:
                        render_application_error(exc)
                value = get_bound(st.session_state, "orientation_result", or_sig)
                analysis = value if isinstance(value, Mapping) else None
                active_or_sig = or_sig

            elif source_mode == "Rotation matrix":
                matrix = matrix_editor(
                    "R(parent ← product)",
                    key_prefix="or_matrix",
                    help_text="x_parent = R · x_product. R must be a proper rotation unless explicit near-rotation projection is enabled.",
                )
                cols = st.columns(2)
                parent_conv = cols[0].selectbox(
                    "Parent Cartesian frame",
                    [item.value for item in CartesianConvention],
                    index=1,
                    key="or_parent_frame",
                )
                product_conv = cols[1].selectbox(
                    "Product Cartesian frame",
                    [item.value for item in CartesianConvention],
                    index=1,
                    key="or_product_frame",
                )
                repair = st.checkbox(
                    "Project an explicitly near-rotation to SO(3)",
                    value=False,
                    key="or_repair",
                    help="Never applied silently; the backend also enforces a maximum repair residual.",
                )
                or_sig = fingerprint(project_sig, source_mode, matrix, parent_conv, product_conv, repair)
                if st.button("Analyze OR matrix", type="primary"):
                    try:
                        invalidate_or_dependents(st.session_state)
                        result = orientation_from_matrix(
                            payload,
                            matrix,
                            parent_convention=parent_conv,
                            product_convention=product_conv,
                            repair=repair,
                        )
                        if repair:
                            result = dict(result)
                            repaired = np.asarray(
                                result["state"]["R_reference_from_moving"], dtype=float
                            )
                            entered = np.asarray(matrix, dtype=float)
                            correction = _relative_matrix_residual(entered, repaired)
                            result["origin_note"] = (
                                "User matrix explicitly projected to the nearest proper rotation. "
                                f"SO(3) correction residual = {correction:.3e}."
                            )
                            result["input_matrix_before_projection"] = matrix
                            result["so3_projection_correction_residual"] = correction
                        put_bound(st.session_state, "orientation_result", or_sig, result)
                    except ApplicationError as exc:
                        render_application_error(exc)
                value = get_bound(st.session_state, "orientation_result", or_sig)
                analysis = value if isinstance(value, Mapping) else None
                active_or_sig = or_sig

            elif source_mode == "Euler ZXZ":
                cols = st.columns(3)
                phi1 = cols[0].number_input("φ₁ (deg)", value=0.0, key="or_phi1")
                Phi = cols[1].number_input("Φ (deg)", value=0.0, key="or_Phi")
                phi2 = cols[2].number_input("φ₂ (deg)", value=0.0, key="or_phi2")
                cols2 = st.columns(3)
                euler_conv = cols2[0].selectbox(
                    "Euler convention",
                    [item.value for item in EulerConvention],
                    key="or_euler_convention",
                )
                parent_conv = cols2[1].selectbox(
                    "Parent Cartesian frame",
                    [item.value for item in CartesianConvention],
                    index=1,
                    key="euler_parent_frame",
                )
                product_conv = cols2[2].selectbox(
                    "Product Cartesian frame",
                    [item.value for item in CartesianConvention],
                    index=1,
                    key="euler_product_frame",
                )
                or_sig = fingerprint(
                    project_sig, source_mode, phi1, Phi, phi2, euler_conv, parent_conv, product_conv
                )
                if st.button("Analyze Euler OR", type="primary"):
                    try:
                        invalidate_or_dependents(st.session_state)
                        result = orientation_from_euler(
                            payload,
                            phi1,
                            Phi,
                            phi2,
                            euler_convention=euler_conv,
                            parent_convention=parent_conv,
                            product_convention=product_conv,
                        )
                        put_bound(st.session_state, "orientation_result", or_sig, result)
                    except ApplicationError as exc:
                        render_application_error(exc)
                value = get_bound(st.session_state, "orientation_result", or_sig)
                analysis = value if isinstance(value, Mapping) else None
                active_or_sig = or_sig

            else:
                st.caption(
                    "Two independent crystallographic parallelisms define exact proper-rotation candidates. "
                    "Internal angles are evaluated with the phase metrics."
                )
                c1, c2 = st.columns(2)
                p1 = c1.text_input(f"{parent_name}: first object", value="(1 1 1)", key="or_p1")
                m1 = c2.text_input(f"{product_name}: first object", value="(0 1 1)", key="or_m1")
                p2 = c1.text_input(f"{parent_name}: second object", value="[1 0 -1]", key="or_p2")
                m2 = c2.text_input(f"{product_name}: second object", value="[1 1 -1]", key="or_m2")
                p1n, m1n = _normalize_compact_crystal_text(p1), _normalize_compact_crystal_text(m1)
                p2n, m2n = _normalize_compact_crystal_text(p2), _normalize_compact_crystal_text(m2)
                solve_sig = fingerprint(project_sig, source_mode, p1n, m1n, p2n, m2n)
                if st.button("Solve exact OR from parallelisms", type="primary"):
                    try:
                        invalidate_or_dependents(st.session_state)
                        solved = orientation_from_parallelisms(payload, p1n, m1n, p2n, m2n)
                        put_bound(st.session_state, "parallelism_solve_result", solve_sig, solved)
                    except ApplicationError as exc:
                        render_application_error(exc)
                solved_value = get_bound(st.session_state, "parallelism_solve_result", solve_sig)
                if isinstance(solved_value, Mapping):
                    candidates = solved_value.get("candidates", [])
                    if isinstance(candidates, list) and candidates:
                        selected = st.selectbox(
                            "Exact OR candidate",
                            range(len(candidates)),
                            format_func=lambda i: f"Candidate {i + 1}",
                            key="parallel_candidate",
                        )
                        analysis = candidates[selected]
                        active_or_sig = fingerprint(solve_sig, selected)
                        st.caption(f"{len(candidates)} exact candidate(s) satisfy both parallelisms.")

            if analysis is None:
                if st.session_state.get("orientation_result") is not None or st.session_state.get("parallelism_solve_result") is not None:
                    st.info("OR inputs changed. Analyze the current OR before using mapping or reconstruction.")
            else:
                active_exports["orientation_analysis"] = analysis
                st.divider()
                _render_orientation(analysis)

                map_tab, recon_tab = st.tabs(
                    ["Map crystallographic objects", "Parent ↔ product reconstruction"]
                )

                with map_tab:
                    st.markdown("### Correspondence mapping vs physical OR mapping")
                    object_text_raw = st.text_input(
                        "Direction or plane",
                        value="[1 0 0]",
                        key="map_object",
                        help="[uvw] is a direct direction; (hkl) is a reciprocal plane. Compact [010]/(010) notation is accepted.",
                    )
                    object_text = _normalize_compact_crystal_text(object_text_raw)
                    source_phase = st.radio(
                        "Object belongs to",
                        ["parent", "product"],
                        horizontal=True,
                        key="map_source_phase",
                    )
                    c_sig = fingerprint(project_sig, "C", object_text, source_phase)
                    r_sig = fingerprint(project_sig, active_or_sig, "R", object_text, source_phase)

                    left, right = st.columns(2)
                    do_c = left.button("Map through correspondence C", use_container_width=True)
                    do_r = right.button("Map through physical OR R", use_container_width=True)

                    # Independent error boundaries: one action can fail without
                    # suppressing its sibling control or the rest of the panel.
                    if do_c:
                        try:
                            result_c = map_correspondence_object(
                                payload, object_text, source_phase=source_phase
                            )
                            put_bound(st.session_state, "correspondence_map_result", c_sig, result_c)
                        except ApplicationError as exc:
                            render_application_error(exc)
                    if do_r:
                        try:
                            base_R = analysis["state"]["R_reference_from_moving"]
                            result_r = map_orientation_object(
                                payload, base_R, object_text, source_phase=source_phase
                            )
                            put_bound(st.session_state, "orientation_map_result", r_sig, result_r)
                        except ApplicationError as exc:
                            render_application_error(exc)

                    c_value = get_bound(st.session_state, "correspondence_map_result", c_sig)
                    r_value = get_bound(st.session_state, "orientation_map_result", r_sig)
                    lres, rres = st.columns(2)
                    with lres:
                        if isinstance(c_value, Mapping):
                            _render_mapping_result("Crystallographic correspondence", c_value)
                            active_exports["correspondence_mapping"] = c_value
                    with rres:
                        if isinstance(r_value, Mapping):
                            _render_mapping_result("Physical orientation relationship", r_value)
                            active_exports["orientation_mapping"] = r_value

                with recon_tab:
                    st.markdown("### Reconstruct the other phase from an observed orientation")
                    st.caption(
                        "Matrix convention: x_sample = g(sample←crystal) · x_crystal. "
                        "No EBSD-vendor Euler convention is inferred here."
                    )
                    observed_phase = st.radio(
                        "Observed phase",
                        ["parent", "product"],
                        horizontal=True,
                        key="recon_observed_phase",
                    )
                    g = matrix_editor(
                        "Observed g(sample ← crystal)", key_prefix="sample_g"
                    )
                    recon_sig = fingerprint(project_sig, active_or_sig, observed_phase, g)
                    if st.button("Reconstruct symmetry-distinct candidates", type="primary"):
                        try:
                            result_recon = reconstruct_sample_orientations(
                                payload, analysis, g, observed_phase=observed_phase
                            )
                            put_bound(st.session_state, "reconstruction_result", recon_sig, result_recon)
                        except ApplicationError as exc:
                            render_application_error(exc)
                    recon = get_bound(st.session_state, "reconstruction_result", recon_sig)
                    if isinstance(recon, Mapping):
                        active_exports["reconstruction"] = recon
                        candidates = recon.get("candidates", [])
                        if isinstance(candidates, list) and candidates:
                            st.caption(f"{len(candidates)} symmetry-distinct candidate(s).")
                            table = pd.DataFrame(
                                [
                                    {
                                        # variant_index is already 1-based in the backend payload.
                                        "variant": item["variant_index"],
                                        "reconstructed phase": item["reconstructed_phase_id"],
                                        "φ₁ active": item["euler_zxz_active"]["phi1_deg"],
                                        "Φ active": item["euler_zxz_active"]["Phi_deg"],
                                        "φ₂ active": item["euler_zxz_active"]["phi2_deg"],
                                        "rotation residual": _sci(item["rotation_audit"]["maximum_residual"], 3),
                                    }
                                    for item in candidates
                                ]
                            )
                            st.dataframe(table, hide_index=True, use_container_width=True)
                            sel = st.selectbox(
                                "Inspect reconstructed candidate",
                                range(len(candidates)),
                                format_func=lambda i: f"Candidate {i + 1}",
                                key="recon_candidate",
                            )
                            st.dataframe(
                                matrix_frame(candidates[sel]["g_sample_from_reconstructed"]),
                                use_container_width=True,
                            )
                            st.caption(str(recon.get("note", "")))
        except ApplicationError as exc:
            render_application_error(exc)

    # Martensite ------------------------------------------------------------
    with martensite_tab:
        st.markdown("## Martensite: twins, classical PTMC & cofactor conditions")
        st.caption(
            "Selected stretch variants are tested through the backend Mallard, classical single-shear PTMC, "
            "and cofactor solvers. No twin or habit-plane solution is created when the prerequisites fail."
        )
        try:
            payload, response = _require_project()
            project_sig = str(st.session_state.get("calculated_draft_signature", ""))
            variants = response.result["stretch_variants"]["variants"]  # type: ignore[attr-defined]
            grouped: list[list[Mapping[str, object]]] = []
            if len(variants) < 2:
                st.info("Fewer than two stretch variants were generated; pair-twin analysis is unavailable.")
            else:
                cols = st.columns(2)
                vi = cols[0].selectbox(
                    "Variant i", range(len(variants)), format_func=lambda i: f"U{i + 1}", key="mart_vi"
                )
                vj_options = [i for i in range(len(variants)) if i != vi]
                vj = cols[1].selectbox(
                    "Variant j", vj_options, format_func=lambda i: f"U{i + 1}", key="mart_vj"
                )
                pair_sig = fingerprint(project_sig, vi, vj)
                if st.button("Find Mallard twins; evaluate PTMC / cofactor", type="primary"):
                    try:
                        pair_result = martensite_variant_pair_analysis(payload, vi, vj)
                        put_bound(st.session_state, "martensite_pair_result", pair_sig, pair_result)
                    except ApplicationError as exc:
                        render_application_error(exc)
                pair = get_bound(st.session_state, "martensite_pair_result", pair_sig)
                if isinstance(pair, Mapping):
                    active_exports["martensite_variant_pair"] = pair
                    relations = pair.get("relations", [])
                    if isinstance(relations, list):
                        grouped = _twin_groups(relations)
                    if not grouped:
                        st.info("No registered parent order-two symmetry produced a Mallard twin for this pair.")
                    else:
                        st.caption(
                            f"{len(relations)} generator/branch relation(s) collapse to {len(grouped)} unique physical twin system(s)."
                        )
                    for system_index, group in enumerate(grouped, 1):
                        relation = group[0]
                        branches = ", ".join(str(item["mallard_kind"]) for item in group)
                        with st.expander(
                            f"Twin system {system_index} · {len(group)} generator(s) · Mallard {branches}",
                            expanded=system_index == 1,
                        ):
                            top = st.columns(4)
                            top[0].metric("Twin shear", _fmt(relation["twin_shear_magnitude"], 8))
                            top[1].metric("Mallard residual", _sci(relation["mallard_residual"], 3))
                            top[2].metric("PTMC solutions", len(relation["ptmc"]))
                            top[3].metric(
                                "Cofactor CC1–3",
                                "all pass" if relation["cofactor"]["satisfied"] else "not all pass",
                            )
                            a_col, n_col = st.columns(2)
                            a_col.dataframe(
                                pd.DataFrame([relation["twin_a"]], columns=["a₁", "a₂", "a₃"]),
                                hide_index=True,
                                use_container_width=True,
                            )
                            n_col.dataframe(
                                pd.DataFrame([relation["twin_n"]], columns=["n₁", "n₂", "n₃"]),
                                hide_index=True,
                                use_container_width=True,
                            )
                            st.markdown("##### Equivalent Mallard generators")
                            st.dataframe(
                                pd.DataFrame(
                                    [
                                        {
                                            "parent symmetry index": item["parent_symmetry_index"],
                                            "Mallard type": item["mallard_kind"],
                                            "twofold axis": item["twofold_axis_parent_symmetric_cartesian"],
                                            "Mallard residual": _sci(item["mallard_residual"], 3),
                                        }
                                        for item in group
                                    ]
                                ),
                                hide_index=True,
                                use_container_width=True,
                            )
                            st.markdown("##### Cofactor conditions")
                            cofactor = relation["cofactor"]
                            st.dataframe(
                                pd.DataFrame(
                                    [
                                        {
                                            "CC1 residual": _sci(cofactor["cc1_residual"], 6),
                                            "CC2 residual": _sci(cofactor["cc2_residual"], 6),
                                            "CC2 simplified": _sci(cofactor["cc2_simplified"], 6),
                                            "CC3 margin": _sci(cofactor["cc3_margin"], 6),
                                            "CC1": cofactor["cc1_satisfied"],
                                            "CC2": cofactor["cc2_satisfied"],
                                            "CC3": cofactor["cc3_satisfied"],
                                        }
                                    ]
                                ),
                                hide_index=True,
                                use_container_width=True,
                            )
                            st.caption(
                                "CC2 residual and CC2 simplified are two equivalent zero tests; their numerical values need not be equal."
                            )
                            st.markdown("##### Classical PTMC")
                            if relation["ptmc"]:
                                st.dataframe(pd.DataFrame(relation["ptmc"]), hide_index=True, use_container_width=True)
                            else:
                                st.caption(
                                    relation.get("ptmc_note")
                                    or "No admissible classical single-shear PTMC volume fraction."
                                )

            with st.expander("Manual lattice-invariant shear a ⊗ n"):
                variant_index = (
                    st.selectbox(
                        "Stretch variant",
                        range(len(variants)),
                        format_func=lambda i: f"U{i + 1}",
                        key="manual_ptmc_variant",
                    )
                    if variants
                    else 0
                )
                if grouped:
                    load_index = st.selectbox(
                        "Load a computed twin system",
                        range(len(grouped)),
                        format_func=lambda i: f"Twin system {i + 1}",
                        key="manual_load_system",
                    )
                    if st.button("Load selected Mallard a, n"):
                        source = grouped[load_index][0]
                        for i, value in enumerate(source["twin_a"]):
                            st.session_state[f"manual_a_{i}"] = float(value)
                        for i, value in enumerate(source["twin_n"]):
                            st.session_state[f"manual_n_{i}"] = float(value)
                        st.rerun()
                st.caption("No scientific default is inserted. Enter a non-zero a and n, or load a computed Mallard system above.")
                a = vector_editor("a", default=(0.0, 0.0, 0.0), key_prefix="manual_a")
                n = vector_editor("n", default=(0.0, 0.0, 0.0), key_prefix="manual_n")
                ready = float(np.linalg.norm(a)) > 0.0 and float(np.linalg.norm(n)) > 0.0
                manual_sig = fingerprint(project_sig, variant_index, a, n)
                if st.button("Evaluate manual PTMC / cofactor", disabled=not ready):
                    try:
                        manual = manual_ptmc_cofactor_analysis(payload, variant_index, a, n)
                        put_bound(st.session_state, "manual_ptmc_result", manual_sig, manual)
                    except ApplicationError as exc:
                        render_application_error(exc)
                manual_value = get_bound(st.session_state, "manual_ptmc_result", manual_sig)
                if isinstance(manual_value, Mapping):
                    active_exports["manual_ptmc"] = manual_value
                    _render_manual_ptmc(manual_value)

            st.caption(
                "Scope: classical single-shear PTMC only. Double-shear PTMC is not labelled as available without a validated solver."
            )
        except ApplicationError as exc:
            render_application_error(exc)

    # CalPad ----------------------------------------------------------------
    with calpad_tab:
        st.markdown("## Metric-correct crystallography calculator")
        st.caption("Direct and reciprocal geometry remain distinct for non-cubic phases.")
        try:
            payload, _ = _require_project()
            project_sig = str(st.session_state.get("calculated_draft_signature", ""))
            parent_name, product_name, parent_id, product_id = _phase_names(payload)
            phase = st.selectbox(
                "Phase",
                [parent_id, product_id],
                format_func=lambda x: parent_name if x == parent_id else product_name,
                key="calpad_phase",
            )
            cell_tab, normal_tab, low_tab = st.tabs(
                ["Cell / reciprocal cell", "Plane ↔ physical normal", "Low-index ranking"]
            )

            with cell_tab:
                cell_sig = fingerprint(project_sig, phase, "cell")
                if st.button("Inspect phase cell"):
                    try:
                        value = calpad_phase_cell(payload, phase)
                        put_bound(st.session_state, "calpad_cell_result", cell_sig, value)
                    except ApplicationError as exc:
                        render_application_error(exc)
                cell = get_bound(st.session_state, "calpad_cell_result", cell_sig)
                if isinstance(cell, Mapping):
                    active_exports["calpad_cell"] = cell
                    compact_key_value(
                        [
                            ("Point group", cell["point_group_symbol"]),
                            ("Symmetry order", cell["symmetry_order"]),
                            ("Direct cell", cell["direct_cell"]),
                            ("Reciprocal cell", cell["reciprocal_cell"]),
                            ("M·M⁻¹ residual", _sci(cell["metric_inverse_residual"], 6)),
                        ]
                    )
                    _render_matrix("Direct metric", cell["direct_metric"], expanded=True)
                    _render_matrix("Reciprocal metric", cell["reciprocal_metric"])

            with normal_tab:
                obj_raw = st.text_input(
                    "Plane or direction", value="(1 0 1)", key="calpad_normal_object"
                )
                obj = _normalize_compact_crystal_text(obj_raw)
                max_index = st.number_input(
                    "Nearest low-index search bound",
                    min_value=1,
                    max_value=30,
                    value=12,
                    key="calpad_normal_bound",
                )
                normal_sig = fingerprint(project_sig, phase, obj, int(max_index))
                if st.button("Convert using the lattice metric"):
                    try:
                        report = calpad_normal_conversion(
                            payload, phase, obj, max_index=int(max_index)
                        )
                        put_bound(st.session_state, "calpad_normal_result", normal_sig, report)
                    except ApplicationError as exc:
                        render_application_error(exc)
                report = get_bound(st.session_state, "calpad_normal_result", normal_sig)
                if isinstance(report, Mapping):
                    active_exports["calpad_normal_conversion"] = report
                    compact_key_value(
                        [
                            ("Relation", report["relation"]),
                            ("Raw coefficients", report["raw_target_coefficients"]),
                            ("Scale-normalized", report["projective_target_coefficients"]),
                            ("Nearest low-index", report["nearest_low_index"]["notation"]),
                            ("Angular mismatch (deg)", report["nearest_low_index"]["angular_mismatch_deg"]),
                            ("Round-trip residual", _sci(report["roundtrip_projective_residual"], 6)),
                        ]
                    )
                    for warning in report.get("warnings", []):
                        st.warning(str(warning))

            with low_tab:
                target_raw = st.text_input(
                    "Target object", value="[1 0 1]", key="calpad_low_target"
                )
                target = _normalize_compact_crystal_text(target_raw)
                source_kind = _target_kind(target)
                cols = st.columns(4)
                kind = cols[0].selectbox(
                    "Candidates", ["direction", "plane"], key="calpad_candidate_kind"
                )
                cross_type = source_kind is not None and source_kind != kind
                if cross_type:
                    comparison = cols[1].selectbox(
                        "Cross-type comparison",
                        ["direction–plane incidence", "normal-parallel"],
                        key="calpad_cross_relation",
                    )
                    sense = "projective"
                else:
                    comparison = "same-type physical angle"
                    sense = cols[1].selectbox(
                        "Angle sense", ["projective", "oriented"], key="calpad_angle_sense"
                    )
                bound = cols[2].number_input(
                    "Max |index|", min_value=1, max_value=12, value=3, key="calpad_low_bound"
                )
                limit = cols[3].number_input(
                    "Rows", min_value=1, max_value=200, value=20, key="calpad_low_limit"
                )
                low_sig = fingerprint(
                    project_sig, phase, target, kind, comparison, sense, int(bound), int(limit)
                )
                if st.button("Rank low-index objects"):
                    try:
                        if comparison == "normal-parallel":
                            low_report = _normal_parallel_ranking(
                                payload,
                                phase,
                                target,
                                kind,
                                max_index=int(bound),
                                limit=int(limit),
                            )
                        else:
                            low_report = calpad_low_index_table(
                                payload,
                                phase,
                                target,
                                candidate_kind=kind,
                                max_index=int(bound),
                                limit=int(limit),
                                angle_sense=sense,
                            )
                        put_bound(st.session_state, "calpad_low_result", low_sig, low_report)
                    except ApplicationError as exc:
                        render_application_error(exc)
                low = get_bound(st.session_state, "calpad_low_result", low_sig)
                if isinstance(low, Mapping):
                    active_exports["calpad_low_index"] = low
                    rows = [dict(item) for item in low["rows"]]
                    for row in rows:
                        row["notation"] = _clean_projective_notation(
                            row.get("notation", ""), kind, sense
                        )
                        if row.get("angle_deg") is not None:
                            row["angle_deg"] = float(row["angle_deg"])
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                    st.caption("Projective means ± identified; parentheses/brackets above denote one object, not a symmetry family.")
                    with st.expander("Definition used for this ranking"):
                        if comparison == "normal-parallel":
                            st.write(
                                "The target is first converted with the lattice metric to its physical normal counterpart, then ranked in the corresponding direct/reciprocal space."
                            )
                            st.json(_jsonable(low.get("normal_parallel_source", {})))
                        elif cross_type:
                            st.latex(
                                r"\sin\theta=\frac{|p^T u|}{\sqrt{u^T M u}\sqrt{p^T M^{-1}p}}"
                            )
                            st.write(
                                "θ = 0° means the direction lies in the plane; θ = 90° means the direction is parallel to the plane normal."
                            )
                        elif kind == "plane":
                            st.latex(
                                r"\cos\theta=\frac{p^T M^{-1}q}{\sqrt{p^TM^{-1}p}\sqrt{q^TM^{-1}q}}"
                            )
                        else:
                            st.latex(
                                r"\cos\theta=\frac{u^T M v}{\sqrt{u^TMu}\sqrt{v^TMv}}"
                            )
                        with st.expander("Backend search provenance"):
                            st.json(_jsonable(low.get("derivation", {})))
        except ApplicationError as exc:
            render_application_error(exc)

    # EBSD ------------------------------------------------------------------
    with ebsd_tab:
        st.markdown("## EBSD map import & audit")
        st.caption(
            "This panel uses the repository's vendor-neutral EBSD adapters. Vendor/sample frame corrections are never guessed."
        )
        ebsd_file = st.file_uploader(
            "EBSD file", type=["ang", "ctf"], key="ebsd_upload"
        )
        unit_choice = st.selectbox(
            "CTF 3-D Euler-angle unit",
            ["auto / standard 2-D", "degrees", "radians"],
            help=(
                "Standard 2-D CTF is degrees. For 3-D CTF the backend requires an explicit unit; "
                "choose degrees or radians here."
            ),
            key="ebsd_ctf_unit",
        )
        if ebsd_file is not None:
            raw = ebsd_file.getvalue()
            file_hash = hashlib.sha256(raw).hexdigest()
            ebsd_sig = fingerprint(file_hash, ebsd_file.name, unit_choice)
            if st.button("Load and audit EBSD map", type="primary"):
                suffix = Path(ebsd_file.name).suffix.lower()
                tmp_name = ""
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                        handle.write(raw)
                        tmp_name = handle.name
                    if suffix == ".ang":
                        data = load_ang(tmp_name)
                    elif suffix == ".ctf":
                        unit = None
                        if unit_choice == "degrees":
                            unit = AngleUnit.DEGREE
                        elif unit_choice == "radians":
                            unit = AngleUnit.RADIAN
                        data = load_ctf(tmp_name, three_dimensional_angle_unit=unit)
                    else:
                        raise input_error("Only .ang and .ctf are enabled in this first EBSD UI layer.")

                    audit = audit_map(data)
                    stride = max(1, data.n_points // 8000)
                    idx = np.arange(0, data.n_points, stride, dtype=int)
                    preview = pd.DataFrame(
                        {
                            "x": data.x[idx],
                            "y": data.y[idx],
                            "phase": data.phase_id[idx].astype(str),
                            "indexed": data.indexed[idx],
                        }
                    )
                    result = {
                        "file_name": ebsd_file.name,
                        "sha256": file_hash,
                        "audit": asdict(audit),
                        "metadata": _jsonable(dict(data.metadata)),
                        "quality_fields": sorted(map(str, data.quality.keys())),
                        "preview": preview.to_dict(orient="records"),
                    }
                    put_bound(st.session_state, "ebsd_result", ebsd_sig, result)
                except ApplicationError as exc:
                    render_application_error(exc)
                except (ValueError, OSError, UnicodeError) as exc:
                    render_application_error(
                        input_error(
                            str(exc),
                            detail=f"{type(exc).__name__}: {exc}",
                            hint="Check the EBSD format and orientation-unit choice. No frame correction is inferred.",
                        )
                    )
                finally:
                    if tmp_name:
                        try:
                            Path(tmp_name).unlink(missing_ok=True)
                        except OSError:
                            pass

            result = get_bound(st.session_state, "ebsd_result", ebsd_sig)
            if isinstance(result, Mapping):
                active_exports["ebsd_import_audit"] = {
                    key: value for key, value in result.items() if key != "preview"
                }
                audit = result["audit"]
                cols = st.columns(5)
                cols[0].metric("Points", audit["n_points"])
                cols[1].metric("Indexed", audit["n_indexed"])
                cols[2].metric("Indexed fraction", _fmt(audit["indexed_fraction"], 5))
                cols[3].metric("Coordinate dimension", audit["coordinate_dimension"])
                cols[4].metric("SO(3) residual", _sci(audit["maximum_so3_residual"], 3))
                compact_key_value(
                    [
                        ("File SHA-256", result["sha256"]),
                        ("Phase counts", audit["phase_counts"]),
                        ("Duplicate coordinate pairs", audit["duplicate_coordinate_pairs"]),
                        ("Nearest-neighbour median", audit["nearest_neighbor_median"]),
                        ("Nearest-neighbour minimum", audit["nearest_neighbor_minimum"]),
                        ("Quality fields", result["quality_fields"]),
                    ]
                )
                preview = pd.DataFrame(result["preview"])
                if not preview.empty:
                    st.markdown("#### Coordinate preview")
                    st.scatter_chart(preview, x="x", y="y", color="phase")
                with st.expander("Import metadata / orientation convention"):
                    st.json(_jsonable(result["metadata"]))

        st.markdown("### Validated EBSD backend already present")
        st.dataframe(
            pd.DataFrame(
                [
                    ("ANG / CTF / explicit CSV-HDF5 I/O", "backend available", "No hidden frame correction"),
                    ("Map audit / quality fields", "UI enabled here", "Point count, indexing, SO(3), coordinates"),
                    ("Phase-aware grain segmentation", "backend available", "Next UI layer: explicit thresholds + sweep"),
                    ("KAM / GOS", "backend available", "Must retain explicit neighbourhood/threshold settings"),
                    ("Parent reconstruction / variant graph", "backend available", "Ambiguity must be retained, not forced"),
                    ("OR refinement", "backend available", "Configured OR and fitted OR must remain separate"),
                    ("Trace validation", "backend available", "Requires explicit specimen surface normal"),
                ],
                columns=["Capability", "Status", "UI contract"],
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "The deeper EBSD pipeline is intentionally not exposed with guessed defaults. Its next UI layer should collect an explicit, exportable run configuration before segmentation/reconstruction is executed."
        )

    # Diagnostics / export --------------------------------------------------
    with diagnostics_tab:
        st.markdown("## Diagnostics, provenance & export")
        try:
            payload, response = _require_project()
            diagnostics = response.result["diagnostics"]  # type: ignore[attr-defined]
            cols = st.columns(4)
            cols[0].metric("Metric solver", diagnostics["solver_source"] or "—")
            cols[1].metric("Precision escalated", "yes" if diagnostics["precision_escalated"] else "no")
            cols[2].metric("Representation parity", _sci(diagnostics["representation_parity_residual"], 3))
            cols[3].metric(
                "CT / Ball–James agreement",
                "yes" if diagnostics["exact_classification_agreement_ct_vs_ball_james"] else "no",
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        ("Generalized eigen equation", _sci(diagnostics["generalized_eigen_residual"], 6)),
                        ("Metric orthonormality", _sci(diagnostics["metric_orthonormality_residual"], 6)),
                        ("Route disagreement", _sci(diagnostics["route_disagreement"], 6)),
                        ("Representation parity", _sci(diagnostics["representation_parity_residual"], 6)),
                    ],
                    columns=["diagnostic", "residual"],
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "Precision escalation is triggered only when the binary64 decision is numerically ambiguous or fails the forward/error guards; it is not a quality score."
            )
            with st.expander("Theory-consistency audit"):
                st.json(_jsonable(response.result["contracts"]))  # type: ignore[attr-defined]

            st.markdown("### Export")
            project_json = response.project_json()  # type: ignore[attr-defined]
            result_json = response.to_json()  # type: ignore[attr-defined]
            workstation = {
                "calculated_draft_signature": st.session_state.get("calculated_draft_signature"),
                "analyses": _jsonable(active_exports),
            }
            workstation_json = json.dumps(
                workstation,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                default=str,
            )
            c1, c2, c3 = st.columns(3)
            c1.download_button(
                "Project JSON",
                project_json,
                "cualni_ct_project.json",
                "application/json",
                use_container_width=True,
            )
            c2.download_button(
                "Transformation results",
                result_json,
                "cualni_ct_results.json",
                "application/json",
                use_container_width=True,
            )
            c3.download_button(
                "Workbench analyses",
                workstation_json,
                "cualni_ct_workbench_analyses.json",
                "application/json",
                use_container_width=True,
            )

            with st.expander("Method scope"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            ("Metric / CT / SMC / stretch", "available", "general backend solver"),
                            ("Ball–James single-variant", "available", "general backend solver"),
                            ("Orientation variants / operators", "available", "Cayron-style topology"),
                            ("Parent ↔ product reconstruction", "available", "explicit OR + symmetry"),
                            ("Mallard twins", "available", "physical twins grouped from generator branches"),
                            ("Classical single-shear PTMC", "available", "exact-binary64 polynomial/root workflow"),
                            ("Cofactor CC1–CC3", "available", "user/Mallard twin system"),
                            ("Metric CalPad", "available", "direct/reciprocal geometry"),
                            ("EBSD import/audit", "available", "ANG/CTF in this UI layer"),
                            ("EBSD segmentation/reconstruction", "backend available", "UI configuration layer still to expose"),
                            ("Double-shear PTMC", "not claimed", "no validated solver exposed"),
                            ("E2E / 3-D NCS / diffraction simulation", "not claimed", "not exposed as validated workflow"),
                        ],
                        columns=["Method", "Status", "Scope"],
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
        except ApplicationError as exc:
            render_application_error(exc)
