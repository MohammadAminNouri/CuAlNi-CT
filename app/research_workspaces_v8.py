from __future__ import annotations

"""Phase-2B research orchestration over cumulative Phase-2A hardening."""

import streamlit as st

import app.research_workspaces as rw
import app.research_workspaces_v4 as v4
import app.research_workspaces_v7 as v7
import app.scientific_interpretation_v8 as interp
from app.browser_recovery import persist_browser_draft, restore_browser_draft


def _or_comparison_with_live_native_inventory(equivalence, *, enabled: bool, tolerance_deg: float):
    # The unified report is the native inventory.  The equivalence report alone
    # cannot distinguish "no PTMC OR observable exists" from disagreement.
    unified = rw._current_unified_report()
    return interp.or_comparison_finding(
        equivalence,
        enabled=enabled,
        tolerance_deg=tolerance_deg,
        unified=unified,
    )


# Replace only this conclusion gate; every native solver remains unchanged.
v4.or_comparison_finding = _or_comparison_with_live_native_inventory


def render_research_extension() -> None:
    restore_browser_draft(st.session_state, st.query_params)
    try:
        v7.render_research_extension()
    finally:
        persist_browser_draft(st.session_state, st.query_params)
