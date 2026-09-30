from __future__ import annotations

"""Additive Cayron-geometry/navigation layer over the frozen V10 presentation.

V11 deliberately patches only the V10 step renderer. It does not alter the
calculation scope, session-state ownership, native theory solvers, matching,
EBSD, atlas, pole-figure, PTMC, or export pathways.
"""

from dataclasses import replace
import math
from typing import Any, Mapping

import pandas as pd
import streamlit as st

import app.research_workspaces as rw
import app.research_workspaces_v10 as v10
from app.cayron_geometry import render_cayron_geometry


_original_render_step = getattr(
    v10._render_step,
    "_cayron_v11_original_render_step",
    v10._render_step,
)


_GUIDANCE: dict[str, dict[str, str]] = {
    "state": {
        "route": "Workbench → State definition → Parent phase / Product phase → Lattice correspondence → Recalculate crystallographic state",
        "why": "This question requires a current calculated ProjectState with valid parent/product metrics and an explicit C(M←A).",
    },
    "topology": {
        "route": "Workbench → State definition → verify point groups + lattice correspondence → Recalculate crystallographic state; then Workbench → Compatibility → Full transformation audit",
        "why": "Variant/operator topology comes from the calculated symmetries and correspondence. If a current calculation still produces no topology, that is a scientific result rather than a missing UI step.",
    },
    "mm_twins": {
        "route": "Theory comparison → Conclusions → Calculation scope → Calculate / update; then inspect the native CT M/M twin systems shown below. For one selected physical pair, use Workbench → Twins & PTMC → Analyze this variant pair",
        "why": "The native CT M/M inventory is produced by the unified CT theory run. The Workbench pair analysis is a focused follow-up, not a substitute for the CT inventory.",
    },
    "am_exact": {
        "route": "Workbench → Compatibility → Why / how / verify and Full transformation audit. To explore nearby lattice states, use Theory comparison → Compatibility map",
        "why": "A 'not reached' answer here is already a valid CT result for the current lattice state. The compatibility map is for exploring how the result changes when selected lattice parameters change.",
    },
    "nearest": {
        "route": "Workbench → Compatibility → Why / how / verify. If no nearest-degeneracy residual is present, return to Workbench → State definition and recalculate the crystallographic state",
        "why": "The nearest CT residual belongs to the base calculated transformation state, not to an optional theory toggle.",
    },
    "habit": {
        "route": "Theory comparison → Conclusions → Branch-level evidence → Approximate CT A/M diagnostics for diagnostic planes; use Workbench → Numerical audit & export for the exact calculated state. For measured interfaces, use Theory comparison → EBSD ↔ theory",
        "why": "If exact CMC compatibility is absent, there is no exact CT habit plane to unlock elsewhere. Diagnostic planes must remain approximate; EBSD is the route for experimental interface/trace validation.",
    },
    "smc": {
        "route": "Theory comparison → Conclusions → Calculate / update → Native theory inventory and provenance. The SMC matrix itself is also visible in the Cayron Full numerical audit / Workbench → Numerical audit & export",
        "why": "An exact Cayron d_A needs an exact CT habit-plane seed. If that seed does not exist, the exact shear is scientifically blocked rather than merely hidden.",
    },
    "closing_gap": {
        "route": "Theory comparison → Conclusions → Calculation scope → enable CT closing-gap ORs → Calculate / update. Then inspect Native theory inventory and provenance. Use Workbench → Orientation & reconstruction for independent OR analysis",
        "why": "Closing-gap ORs are intentionally opt-in. The Workbench OR tools can inspect orientation hypotheses, but they do not manufacture a missing CT closing-gap branch.",
    },
    "supercompatibility": {
        "route": "If an exact A/M seed exists: Theory comparison → Conclusions → Calculation scope → enable CT supercompatibility → Calculate / update. If the exact A/M seed is absent: Workbench → Compatibility or Theory comparison → Compatibility map",
        "why": "Exact CT A/M/M supercompatibility cannot be evaluated from a nearest-degeneracy diagnostic habit. The prerequisite exact A/M seed must exist first.",
    },
    "overall": {
        "route": "Theory comparison → Conclusions → Calculation scope → request the missing CT extensions → Calculate / update; then inspect Native theory inventory and provenance",
        "why": "The overall statement is a synthesis of the individual CT questions and should only become more complete when the missing native CT inventory has actually been calculated.",
    },
}


def _guidance_needed(step: Any) -> bool:
    status = str(getattr(step, "status", "")).strip().lower()
    return status in {
        "not evaluated",
        "not requested",
        "not available",
        "not evaluable",
        "not reachable exactly",
        "not reached",
        "incomplete ct inventory",
        "partial ct martensitic construction",
        "ct martensitic construction not established",
    }


def _render_next_route(step: Any) -> None:
    if not _guidance_needed(step):
        return
    step_id = str(getattr(step, "step_id", ""))
    item = _GUIDANCE.get(step_id)
    if item is None:
        return

    status = str(getattr(step, "status", "")).strip().lower()

    # A failed supercompatibility residual is not a missing prerequisite:
    # the exact A/M seed existed and the A/M/M condition was actually tested.
    if step_id == "supercompatibility" and status == "not reached":
        with st.expander("Where to inspect this failed exact condition", expanded=False):
            st.markdown(
                "**Route:** Theory comparison → Conclusions → CT supercompatibility "
                "and Numerical evidence / provenance for this step"
            )
            st.write(
                "The exact A/M habit/shear seed exists and CT supercompatibility "
                "was evaluated. 'Not reached' here means every available "
                "shear–shear residual remains above the project's algebraic "
                "tolerance; no prerequisite is missing."
            )
            st.caption(
                "Changing pages or widening a display tolerance does not turn this "
                "state into an exact supercompatible state. A different lattice "
                "state or physical branch is required."
            )
        return

    definitive_negative = (
        step_id in {"am_exact", "habit", "smc", "supercompatibility"}
        and status in {"not reached", "not reachable exactly", "not evaluable"}
    )
    heading = (
        "Where to inspect / what prerequisite is missing"
        if definitive_negative
        else "Where to go next in the app"
    )
    with st.expander(heading, expanded=False):
        st.markdown(f"**Route:** {item['route']}")
        st.write(item["why"])
        if definitive_negative:
            st.caption(
                "Important: this navigation does not imply that another page will turn the current negative CT result into a positive one. "
                "It shows where to inspect the criterion, run the missing prerequisite, explore a different lattice state, or validate against experiment."
            )


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
    raw = (
        unified.get("rows", ())
        if isinstance(unified, Mapping)
        else getattr(unified, "rows", ())
    )
    selected: list[Any] = []
    for row in tuple(raw or ()):
        theory = _enum_text(_row_attr(row, "theory"))
        row_kind = _enum_text(
            _row_attr(row, "prediction_kind", _row_attr(row, "kind"))
        )
        if theory == "cayron_ct" and row_kind == kind:
            selected.append(row)
    return tuple(selected)


def _ct_mm_rows(unified: Any) -> tuple[Any, ...]:
    return _ct_rows(unified, "ct_mm_twin")


def _ct_super_rows(unified: Any) -> tuple[Any, ...]:
    return _ct_rows(unified, "ct_supercompatibility")


def _current_algebraic_tolerance() -> float:
    """Read the active project's numerical policy without changing it."""

    default = 1.0e-10
    response = st.session_state.get("current_response")
    if response is None:
        return default

    if isinstance(response, Mapping):
        project = response.get("project_payload", response.get("project", {}))
    else:
        project = getattr(response, "project_payload", {})

    if not isinstance(project, Mapping):
        return default

    policy = project.get("numerical_policy", {})
    if not isinstance(policy, Mapping):
        return default

    try:
        value = float(policy.get("algebraic", default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value > 0.0 else default


def _supercompatibility_residuals(rows: tuple[Any, ...]) -> tuple[float, ...]:
    """Return the native CT shear–shear residuals exposed by unified rows."""

    values: list[float] = []
    for row in rows:
        residual_map = _row_attr(row, "residuals", {})
        if not isinstance(residual_map, Mapping):
            continue
        raw = residual_map.get("ct_supercompatibility_dimensionless")
        if raw is None:
            continue
        try:
            value = abs(float(raw))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return tuple(values)


def _correct_supercompatibility_step(step: Any) -> Any:
    """Fix only the V10/V11 presentation classification for CT A/M/M.

    Unified ``row.exact`` records the provenance/exactness of the native branch.
    It is NOT the pass/fail flag for the final shear–shear equation.  The
    authoritative Conclusions panel already classifies supercompatibility from
    ``ct_supercompatibility_dimensionless`` against the algebraic tolerance.
    Mirror that exact rule here so Question 9 cannot contradict Conclusions.
    """

    if str(getattr(step, "step_id", "")) != "supercompatibility":
        return step

    rows = _ct_super_rows(rw._current_unified_report())
    residuals = _supercompatibility_residuals(rows)
    if not residuals:
        return step

    tolerance = _current_algebraic_tolerance()
    satisfied = sum(value <= tolerance for value in residuals)
    best = min(residuals)

    if satisfied:
        status = "reached"
        answer = (
            f"Yes. {satisfied}/{len(residuals)} evaluated A/M/M branch "
            "combination(s) satisfy the CT shear–shear residual within the "
            f"algebraic tolerance ({tolerance:.3e})."
        )
    else:
        status = "not reached"
        answer = (
            f"No. 0/{len(residuals)} evaluated A/M/M branch combinations "
            "satisfy the exact CT shear–shear condition. "
            f"Best |shear–shear residual| = {best:.10g}, versus algebraic "
            f"tolerance {tolerance:.3e}."
        )

    evidence = tuple(getattr(step, "evidence", ()) or ())
    evidence += (
        ("Evaluated CT A/M/M residuals", len(residuals)),
        ("Within algebraic tolerance", satisfied),
        ("Best |shear–shear residual|", best),
        ("Project algebraic tolerance", tolerance),
    )
    return replace(step, status=status, answer=answer, evidence=evidence)


def _metadata(row: Any) -> Mapping[str, Any]:
    value = _row_attr(row, "metadata", {})
    return value if isinstance(value, Mapping) else {}


def _first_present(
    row: Any, metadata: Mapping[str, Any], names: tuple[str, ...]
) -> Any:
    for name in names:
        value = _row_attr(row, name, None)
        if value is not None:
            return value
        value = metadata.get(name)
        if value is not None:
            return value
    return None


def _find_metadata_by_tokens(
    metadata: Mapping[str, Any], token_sets: tuple[tuple[str, ...], ...]
) -> Any:
    for tokens in token_sets:
        for key, value in metadata.items():
            lowered = str(key).lower()
            if all(token in lowered for token in tokens):
                return value
    return None


def _projective_text(value: Any, *, plane: bool) -> str:
    if value is None:
        return "N/A"
    try:
        values = [float(item) for item in value]
    except Exception:
        return str(value)
    if len(values) != 3:
        return str(value)
    scale = max(abs(item) for item in values)
    if scale <= 1.0e-15:
        return "N/A"

    reduced = [
        0.0 if abs(item / scale) <= 1.0e-12 else item / scale
        for item in values
    ]
    tokens: list[str] = []
    for item in reduced:
        nearest = round(item)
        tokens.append(
            str(int(nearest))
            if abs(item - nearest) <= 1.0e-10
            else f"{item:.6g}"
        )
    left, right = ("(", ")") if plane else ("[", "]")
    return left + " ".join(tokens) + right


def _render_native_ct_mm_twin_table() -> None:
    unified = rw._current_unified_report()
    rows = _ct_mm_rows(unified)
    if not rows:
        st.info(
            "No native CT M/M twin rows are available yet. "
            "Run Theory comparison → Conclusions → Calculation scope → Calculate / update."
        )
        return

    table_rows: list[dict[str, Any]] = []
    for row in rows:
        metadata = _metadata(row)

        plane = _first_present(
            row,
            metadata,
            (
                "twin_plane_product_crystal",
                "plane_product_crystal",
                "product_plane_crystal",
                "plane_m",
                "twin_plane_m",
            ),
        )
        if plane is None:
            plane = _find_metadata_by_tokens(
                metadata,
                (
                    ("plane", "product"),
                    ("plane", "martensite"),
                    ("plane", "_m"),
                ),
            )

        direction = _first_present(
            row,
            metadata,
            (
                "twin_direction_product_crystal",
                "direction_product_crystal",
                "product_direction_crystal",
                "direction_m",
                "twin_direction_m",
            ),
        )
        if direction is None:
            direction = _find_metadata_by_tokens(
                metadata,
                (
                    ("direction", "product"),
                    ("direction", "martensite"),
                    ("direction", "_m"),
                ),
            )

        operator_index = _first_present(
            row, metadata, ("operator_index", "ct_operator_index", "operator")
        )
        twin_index = _first_present(
            row, metadata, ("twin_index", "ct_twin_index")
        )
        classification = _first_present(
            row,
            metadata,
            ("twin_classification", "classification"),
        )
        construction_route = _first_present(
            row,
            metadata,
            ("twin_kind", "construction_route", "route"),
        )

        table_rows.append(
            {
                "internal operator": (
                    operator_index if operator_index is not None else "N/A"
                ),
                "twin": twin_index if twin_index is not None else "N/A",
                "classification": (
                    classification if classification is not None else "N/A"
                ),
                "route": (
                    construction_route
                    if construction_route is not None
                    else "N/A"
                ),
                "product plane": _projective_text(plane, plane=True),
                "product direction": _projective_text(direction, plane=False),
                "shear s": _row_attr(row, "shear_magnitude", "N/A"),
                "native branch": str(_row_attr(row, "branch_label", "")),
            }
        )

    with st.expander(
        "Native CT M/M twin systems — product plane, direction and shear",
        expanded=True,
    ):
        st.caption(
            "Read-only view of already-calculated Cayron CT rows. "
            "The operator number is the app's internal ordering and is not assumed to equal Cayron's published Oᵢ numbering. "
            "Plane and direction coefficients are only projectively rescaled for readability; no new low-index solution is invented."
        )
        st.dataframe(
            pd.DataFrame(table_rows),
            hide_index=True,
            use_container_width=True,
        )


def _render_step_v11(step: Any) -> None:
    step = _correct_supercompatibility_step(step)

    _original_render_step(step)
    _render_next_route(step)

    step_id = str(getattr(step, "step_id", ""))

    if step_id == "mm_twins":
        _render_native_ct_mm_twin_table()

    if step_id == "am_exact":
        response = st.session_state.get("current_response")
        if response is not None:
            render_cayron_geometry(response, rw._current_unified_report())


_render_step_v11._cayron_v11_original_render_step = _original_render_step  # type: ignore[attr-defined]
v10._render_step = _render_step_v11


def render_research_extension() -> None:
    v10.render_research_extension()
