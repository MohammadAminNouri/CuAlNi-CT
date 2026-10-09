from __future__ import annotations

"""Offline, keyboard-usable Streamlit tree component.

Uses the documented V1 custom-component protocol so a github.dev upload can
ship a working, version-pinned UI without a Node/npm build or third-party CDN.
The scientific source of truth remains the Python report.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def _component():
    from streamlit.components.v1 import declare_component
    return declare_component(
        "twin_family_navigator_v8",
        path=str(Path(__file__).resolve().parent / "navigator_frontend"),
    )


def render_navigator(
    model: dict[str, Any],
    *,
    selected: str,
    theme: str = "dark",
    density: str = "comfortable",
    key: str = "twin_family_navigator_v8",
) -> str | None:
    value = _component()(
        model=model, selected=selected, theme=theme, density=density,
        key=key, default=None,
    )
    if isinstance(value, str):
        allowed = {node["id"] for family in model["families"] for node in family["couples"]}
        return value if value in allowed else None
    return None
