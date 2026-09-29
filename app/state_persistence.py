from __future__ import annotations

"""Cross-page persistence for Streamlit widget drafts.

Streamlit may delete widget-owned session keys when a widget disappears on a
different page. The workbench therefore mirrors scientific input controls into
one detached ordinary session-state object. Returning to a workspace restores
those values before any widget is instantiated.

This module stores UI input only; calculated scientific outputs remain governed
by app.session_state signatures.
"""

from copy import deepcopy
from typing import Any, Mapping, MutableMapping

MIRROR_KEY = "_persistent_scientific_widget_draft_v1"

_EXACT_KEYS = {
    "project_title",
    "project_id",
    "length_unit",
    "transformation_id",
    "transformation_label",
    "workbench_v4_workspace",
    # Persistent scientific selections whose historical widget keys do not
    # share the newer semantic prefixes.
    "mart_vi",
    "mart_vj",
    "manual_load_system",
    "map_object",
    "map_source_phase",
    "parallel_candidate",
    "recon_observed_phase",
    "v4_twin_system",
}

_PREFIXES = (
    "parent_",
    "product_",
    "C_",
    "or_",
    "euler_",
    "parallelism_",
    "reconstruction_",
    "martensite_",
    "manual_ptmc_",
    "manual_a_",
    "manual_n_",
    "sample_g_",
    "calpad_",
    "ebsd_",
    "research_",
)

# Streamlit owns the values of upload widgets. Assigning even ``None`` to a
# file_uploader key through st.session_state before that widget is instantiated
# raises StreamlitValueAssignmentNotAllowedError.
#
# Keep uploader state out of our cross-page mirror entirely. The suffix rule
# protects future upload controls that follow the app's current naming
# convention; the explicit set documents the upload widgets known today.
_STREAMLIT_OWNED_EXACT_KEYS = {
    "research_ebsd_pipeline_upload",
    "workbench_v4_project_upload",
    "ebsd_upload",
}
_STREAMLIT_OWNED_SUFFIXES = (
    "_upload",
    "_uploader",
)

# These are actions or file handles, not durable inputs. Replaying them can
# trigger invalid Streamlit widget state or repeat an action unintentionally.
_EPHEMERAL_PREFIXES = (
    "research_run",
    "research_v4_run",
    "research_download",
    "research_v4_download",
    "workbench_v4_open",
    "workbench_v4_calculate",
    "workbench_v5_calculate",
    "workbench_v4_project_upload",
    "research_pole_generate",
    "research_v5_pole_generate",
    "research_v5_atlas_run",
    "research_v5_pole_csv",
    "research_v6_atlas_run",
    "research_v6_pole_generate",
    "research_v6_pole_csv",
    "research_v7_pole_generate",
    "research_v7_pole_csv",
    "workbench_v6_calculate",
)

