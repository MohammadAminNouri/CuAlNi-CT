from __future__ import annotations

"""Additive Cayron-geometry/navigation layer over the frozen V10 presentation.

V11 deliberately patches only the V10 step renderer.  It does not alter the
calculation scope, session-state ownership, native theory solvers, matching,
EBSD, atlas, pole-figure, PTMC, or export pathways.
"""

from typing import Any

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
        "route": "Theory comparison → Conclusions → Calculation scope → Calculate / update; then open Native theory inventory and provenance. For one selected physical pair, use Workbench → Twins & PTMC → Analyze this variant pair",
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

    status = str(getattr(step, "status", ""))
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


def _render_step_v11(step: Any) -> None:
    _original_render_step(step)
    _render_next_route(step)

    if str(getattr(step, "step_id", "")) == "am_exact":
        response = st.session_state.get("current_response")
        if response is not None:
            render_cayron_geometry(response, rw._current_unified_report())


_render_step_v11._cayron_v11_original_render_step = _original_render_step  # type: ignore[attr-defined]
v10._render_step = _render_step_v11


def render_research_extension() -> None:
    v10.render_research_extension()
