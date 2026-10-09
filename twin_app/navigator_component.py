from __future__ import annotations

"""Resilient, offline Streamlit V1 twin-family navigator.

The static frontend must be committed at twin_app/navigator_frontend/index.html.
If an asset is missing or Streamlit cannot register the custom component,
retain a fully operational native Streamlit selection control instead of
crashing and hiding the already-calculated scientific results.
"""

from functools import lru_cache
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
FRONTEND = Path(__file__).resolve().parent / "navigator_frontend"


def _frontend_assets_present() -> bool:
    """Check the actual deployed files, not a manifest entry or local ZIP."""
    index = FRONTEND / "index.html"
    return FRONTEND.is_dir() and index.is_file() and index.stat().st_size > 100


@lru_cache(maxsize=1)
def _component():
    if not _frontend_assets_present():
        raise FileNotFoundError(
            "Twin navigator HTML is absent: "
            "twin_app/navigator_frontend/index.html"
        )
    import streamlit.components.v1 as components

    return components.declare_component(
        "twin_family_navigator_v8",
        path=str(FRONTEND),
    )


def _native_fallback(
    model: dict[str, Any], *, selected: str, key: str, reason: str
) -> str | None:
    """Keyboard/search-friendly and complete even without custom JS assets."""
    import streamlit as st

    st.warning(
        "The interactive family diagram is temporarily unavailable. "
        "Every calculated twin couple and its scientific results can still "
        "be explored using this list."
    )
    families = model.get("families", ())
    descriptions = {
        node["id"]: (
            f"{family['label']} · {node['label']} · "
            f"{node.get('description', node.get('state', ''))}"
        )
        for family in families
        for node in family.get("couples", ())
    }
    options = list(descriptions)
    if not options:
        st.info("The calculation contains no selectable twin couples.")
        return None
    LOGGER.warning("Twin-family diagram fallback activated: %s", reason)
    return st.selectbox(
        "Choose a twin couple (all calculated families)",
        options=options,
        index=options.index(selected) if selected in descriptions else 0,
        format_func=lambda option: descriptions[option],
        key=f"{key}_native_fallback",
        help="This list contains exactly the same calculated couples as the diagram.",
    )


def render_navigator(
    model: dict[str, Any],
    *,
    selected: str,
    theme: str = "dark",
    density: str = "comfortable",
    key: str = "twin_family_navigator_v8",
) -> str | None:
    allowed = {
        node["id"]
        for family in model.get("families", ())
        for node in family.get("couples", ())
    }
    if not _frontend_assets_present():
        return _native_fallback(
            model, selected=selected, key=key,
            reason="the committed HTML frontend asset is missing",
        )
    try:
        value = _component()(
            model=model,
            selected=selected,
            theme=theme,
            density=density,
            key=key,
            default=None,
        )
    except Exception as exc:
        # Only the navigator/Streamlit UI bridge is caught here. Mathematical
        # errors from build_twin_family_report are never swallowed or relabelled.
        LOGGER.exception("Could not mount the twin-family custom component")
        return _native_fallback(
            model, selected=selected, key=key,
            reason=f"Streamlit navigator registration failed ({type(exc).__name__})",
        )
    return value if isinstance(value, str) and value in allowed else None
