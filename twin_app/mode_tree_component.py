from __future__ import annotations

"""Interactive, offline, read-only physical-twin mode graph.

The graph is a PRESENTATION of the verified ModeView, never a source of physics.
Only independently cross-locked types from ModeSummary.types can become nodes.
JavaScript can request a selection, never supply a classification or result.
"""

from functools import lru_cache
import logging
from pathlib import Path
from typing import Any

from .mode_view_model import ModeSummary

_LOGGER = logging.getLogger(__name__)
_FRONTEND = Path(__file__).resolve().parent / "mode_tree_frontend"
_ALLOWED_TYPES = ("Type I", "Type II", "Compound")


def build_tree_model(modes: tuple[ModeSummary, ...]) -> dict[str, Any]:
    """Create a small, JSON-compatible view of the already-verified mode data."""
    seen: set[str] = set()
    output = []
    for mode in modes:
        if mode.mode_id in seen:
            raise ValueError("Repeated scientific mode identifier in navigator")
        seen.add(mode.mode_id)
        types = []
        if not mode.types:
            raise ValueError("A visible scientific mode must contain a verified twin type")
        for twin_type in mode.types:
            if twin_type.label not in _ALLOWED_TYPES:
                raise ValueError("An unsupported or unresolved twin type cannot enter the graph")
            if not twin_type.entries or twin_type.source_records != len(twin_type.entries):
                raise ValueError("The mode graph cannot show missing source evidence")
            count = int(twin_type.stretch_couples)
            if count < 1 or count > mode.symmetry_pairs:
                raise ValueError("Inconsistent verified symmetry-pair coverage")
            # Count unordered stretch-variant pairs, not experimental twins.
            # Type I and Type II may apply to the same pair and are not additive.
            pairs = tuple(sorted({
                tuple(sorted((int(e.pair.stretch_i), int(e.pair.stretch_j))))
                for e in twin_type.entries if hasattr(e, "pair")
            }))
            if pairs and (len(pairs) != count or
                          any(a < 1 or b < 1 or a == b for a, b in pairs)):
                raise ValueError("Displayed pair coverage disagrees with source identities")
            pair_labels = tuple("U{}–U{}".format(a, b) for a, b in pairs)
            types.append({
                "id": twin_type.label,
                "label": twin_type.label,
                "verifiedPairs": count,
                "totalPairs": int(mode.symmetry_pairs),
                "pairLabels": pair_labels,
                "sources": int(twin_type.source_records),
                "hasHabit": bool(twin_type.recorded_habit_solutions or twin_type.continuous_families),
                "habitCount": int(twin_type.recorded_habit_solutions),
            })
        if len({t["id"] for t in types}) != len(types):
            raise ValueError("Repeated twin type in a physical mode")
        output.append({
            "id": str(mode.mode_id),
            "label": str(mode.display_name),
            "shear": f"{float(mode.shear):.8g}",
            "pairs": int(mode.symmetry_pairs),
            "types": types,
        })
    return {"root": "Parent crystal → Product crystal", "modes": output}


def allowed_selections(model: dict[str, Any]) -> set[tuple[str, str]]:
    return {(m["id"], t["id"]) for m in model["modes"] for t in m["types"]}


def validate_selection(value: Any, model: dict[str, Any]) -> tuple[str, str] | None:
    """Fail closed on data not present in the solver-derived mode catalogue."""
    if not isinstance(value, dict):
        return None
    mode, kind = value.get("mode"), value.get("type")
    if not isinstance(mode, str) or not isinstance(kind, str):
        return None
    return (mode, kind) if (mode, kind) in allowed_selections(model) else None


@lru_cache(maxsize=1)
def _component():
    index = _FRONTEND / "index.html"
    if not index.is_file() or index.stat().st_size < 500:
        raise FileNotFoundError("Offline mode-tree frontend was not installed")
    import streamlit.components.v1 as components
    return components.declare_component("physical_twin_mode_graph_2026", path=str(_FRONTEND))


def _native_fallback(model: dict[str, Any], selection: tuple[str, str]) -> tuple[str, str] | None:
    """Operational even when the browser custom component cannot mount."""
    import streamlit as st
    options = tuple(sorted(allowed_selections(model)))
    if not options:
        return None
    labels = {
        (m["id"], t["id"]): f"{m['label']} · s={m['shear']} → {t['label']}"
        for m in model["modes"] for t in m["types"]
    }
    st.warning("The connected tree is unavailable. The complete verified mode/type list remains accessible below.")
    return st.selectbox(
        "Choose calculated mode and verified twin type",
        options=options,
        index=options.index(selection) if selection in options else 0,
        format_func=lambda x: labels[x],
        key="ct_graph_native_select_fallback",
    )


def render_mode_tree(model: dict[str, Any], *, selection: tuple[str, str], key: str) -> tuple[str, str] | None:
    """Return only validated user selections; never trust custom-component values."""
    import streamlit as st
    valid = allowed_selections(model)
    if not valid:
        return None
    if selection not in valid:
        raise ValueError("Selected mode/type is absent from the validated graph")
    appearance = st.session_state.get("tf_appearance_v8", "Dark")
    theme = {"Dark": "dark", "Light": "light", "High contrast": "contrast"}.get(appearance, "dark")
    density = "compact" if st.session_state.get("tf_density_v8") == "Compact" else "comfortable"
    font_scale = "large" if st.session_state.get("tf_font_scale_v8") == "Large" else "standard"
    try:
        value = _component()(
            model=model,
            selected={"mode": selection[0], "type": selection[1]},
            theme=theme,
            density=density,
            fontScale=font_scale,
            key=key,
            default=None,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        _LOGGER.warning("Mode-tree frontend fallback: %s", exc)
        return _native_fallback(model, selection)
    return validate_selection(value, model)
