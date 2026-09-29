from __future__ import annotations

"""Phase-2B workbench entry: browser recovery + cumulative Phase-2A semantics."""

import streamlit as st

import app.streamlit_workstation_v4 as v4
import app.streamlit_workstation_v7 as v7
from app.browser_recovery import persist_browser_draft, restore_browser_draft


_original_setup = v4._render_setup


def _render_setup_phase2b(title: str, length_unit: str, project_id: str) -> None:
    if st.session_state.get("_browser_recovery_invalid", False):
        st.warning(
            "A browser draft token was present but failed integrity/version checks, so it was ignored. No scientific state was reconstructed from it."
        )
    if st.session_state.get("_browser_draft_recovered", False) and st.session_state.get("current_response") is None:
        st.info(
            "Editable scientific inputs were recovered after the app process restarted. Calculated results were deliberately not restored; run the crystallographic calculation explicitly before downstream analysis."
        )
    _original_setup(title, length_unit, project_id)


v4._render_setup = _render_setup_phase2b


def main() -> None:
    # Must happen before v7/v6 instantiate any widget.
    restore_browser_draft(st.session_state, st.query_params)
    try:
        v7.main()
    finally:
        # v7/v6 have already refreshed the detached whitelist mirror here.
        persist_browser_draft(st.session_state, st.query_params)


if __name__ == "__main__":
    main()
