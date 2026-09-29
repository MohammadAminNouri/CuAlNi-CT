from __future__ import annotations

"""Final Phase-2C research entry: EBSD fail-closed preflight over cumulative 2B."""

import app.research_workspaces as rw
import app.research_workspaces_v8 as v8
from app.phase2_ebsd_guard import validate_ebsd_config_file


# Guard the UI-to-pipeline boundary only; the validated native EBSD solver is not
# modified.  Preserve the unwrapped delegate if this module is imported again.
_delegate = getattr(rw.run_pipeline, "_phase2c_original_delegate", rw.run_pipeline)


def _phase2c_guarded_run_pipeline(config_path):
    validate_ebsd_config_file(config_path)
    return _delegate(config_path)


_phase2c_guarded_run_pipeline._phase2c_original_delegate = _delegate  # type: ignore[attr-defined]
rw.run_pipeline = _phase2c_guarded_run_pipeline


def render_research_extension() -> None:
    v8.render_research_extension()
