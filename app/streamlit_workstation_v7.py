from __future__ import annotations

"""Phase-2A live-defect hardening over the green Phase-1/V6 workstation."""

from copy import deepcopy
from typing import Any, Mapping

import streamlit as st

import app.scientific_interpretation_v7 as interp
import app.streamlit_workstation as legacy
import app.streamlit_workstation_v4 as v4
import app.streamlit_workstation_v6 as v6
from app.phase2_contracts import (
    projective_representative_notation,
    sanitize_finding,
)
from app.scientific_ui import render_finding as _base_render_finding


# V6 already installs the verified solver/UI bindings.  Replace only the
# professor-facing interpretation hooks with Phase-2 wording corrections.
v4.transformation_findings = interp.transformation_findings
v4.topology_finding = interp.topology_finding
v4.orientation_finding = interp.orientation_finding
v4.martensite_pair_finding = interp.martensite_pair_finding
v4.ebsd_audit_finding = interp.ebsd_audit_finding


def _render_finding_phase2(finding: Any, *args: Any, **kwargs: Any) -> None:
    _base_render_finding(sanitize_finding(finding), *args, **kwargs)


v4.render_finding = _render_finding_phase2


_original_normal_conversion = v4.calpad_normal_conversion


def _normal_conversion_phase2(*args: Any, **kwargs: Any):
    result = _original_normal_conversion(*args, **kwargs)
    if not isinstance(result, Mapping):
        return result
    out = deepcopy(dict(result))
    nearest = out.get("nearest_low_index")
    if isinstance(nearest, Mapping):
        nearest_out = deepcopy(dict(nearest))
        target_kind = str(out.get("target_kind", ""))
        nearest_out["notation"] = projective_representative_notation(
            str(nearest_out.get("notation", "")),
            kind=target_kind,
        )
        out["nearest_low_index"] = nearest_out
    return out


v4.calpad_normal_conversion = _normal_conversion_phase2


def _project_status_phase2() -> None:
    payload = st.session_state.get("current_project_payload")
    if not isinstance(payload, dict):
        return
    parent, product, _, _ = legacy._phase_names(payload)
    title = str(payload.get("title") or "Project")
    transformations = payload.get("transformations")
    transformation_label = ""
    if isinstance(transformations, list) and transformations:
        first = transformations[0]
        if isinstance(first, Mapping):
            transformation_label = str(first.get("label") or first.get("transformation_id") or "").strip()
    middle = f" · {transformation_label}" if transformation_label else ""
    if st.session_state.get("requires_recalculation", False):
        v4.state_strip(f"{title}{middle} · {parent} → {product} · draft changed — recalculate before analysis")
    else:
        v4.state_strip(f"{title}{middle} · {parent} → {product} · calculated state active")


v4._project_status = _project_status_phase2


def main() -> None:
    v6.main()


if __name__ == "__main__":
    main()
