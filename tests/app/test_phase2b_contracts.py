from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.browser_recovery import (
    RECOVERY_PARAM,
    decode_recovery_token,
    encode_recovery_token,
    persist_browser_draft,
    restore_browser_draft,
)
from app.phase2_contracts import atlas_professor_audit_rows, ct_ptmc_or_availability
from app.state_persistence import MIRROR_KEY, snapshot_persistent_widget_state


@dataclass
class _Row:
    theory: str
    prediction_kind: str
    exact: bool
    or_parent_from_product: object | None = None


@dataclass
class _Unified:
    rows: tuple[_Row, ...]


@dataclass
class _AtlasState:
    state_id: str = "a_scale=1"
    metadata: dict = field(default_factory=lambda: {"a_scale": 1.0})
    classification: str = "not_evaluable"
    ct_am_exact_compatible: bool = False
    ct_supercompatible: bool | None = None
    cofactor_compatible: bool = False
    ct_best_supercompatibility_residual: float | None = None
    ct_compatible_branch_count: int = 0
    ct_supercompatible_branch_count: int = 0
    cofactor_compatible_branch_count: int = 0
    matched_relation_count: int = 0
    pair_agreement_count: int = 0
    pair_disagreement_count: int = 0


def test_browser_recovery_roundtrip_is_draft_only_and_stale():
    source = {
        "project_title": "Recovered project",
        "parent_a": 5.836,
        "product_beta": 100.0,
        "C_00": 1.0,
        "research_atlas_1d_axis1": "a_scale",
    }
    token = encode_recovery_token(source)
    assert decode_recovery_token(token) == source

    state: dict = {}
    restored = restore_browser_draft(state, {RECOVERY_PARAM: token})
    assert restored is True
    assert state["project_title"] == "Recovered project"
    assert state["parent_a"] == 5.836
    assert state["requires_recalculation"] is True
    assert "current_response" not in state
    assert "current_project_payload" not in state
    assert "calculated_draft_signature" not in state




def test_browser_recovery_never_clobbers_live_calculated_state():
    token = encode_recovery_token({"project_title": "browser"})
    live_response = object()
    state = {
        "current_response": live_response,
        "current_project_payload": {"title": "live"},
        "calculated_draft_signature": "live-signature",
    }
    assert restore_browser_draft(state, {RECOVERY_PARAM: token}) is False
    assert state["current_response"] is live_response
    assert state["current_project_payload"]["title"] == "live"

def test_browser_recovery_never_overwrites_live_detached_draft():
    token = encode_recovery_token({"project_title": "browser"})
    state = {MIRROR_KEY: {"project_title": "live"}}
    assert restore_browser_draft(state, {RECOVERY_PARAM: token}) is False
    assert state[MIRROR_KEY]["project_title"] == "live"


def test_browser_persistence_uses_existing_whitelist_mirror():
    state = {
        "project_title": "Persist me",
        "parent_a": 5.8,
        "research_v6_atlas_run": True,  # ephemeral action: must not be mirrored
    }
    snapshot_persistent_widget_state(state)
    qp: dict[str, str] = {}
    assert persist_browser_draft(state, qp) is True
    values = decode_recovery_token(qp[RECOVERY_PARAM])
    assert values["project_title"] == "Persist me"
    assert values["parent_a"] == 5.8
    assert "research_v6_atlas_run" not in values


def test_browser_recovery_rejects_tampering():
    token = encode_recovery_token({"project_title": "x"})
    broken = ("0" if token[0] != "0" else "1") + token[1:]
    state: dict = {}
    assert restore_browser_draft(state, {RECOVERY_PARAM: broken}) is False
    assert state["_browser_recovery_invalid"] is True


def test_ct_ptmc_or_gate_distinguishes_missing_observable_from_disagreement():
    unified = _Unified(
        rows=(
            _Row("cayron_ct", "ct_closing_gap_or", True, ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
        )
    )
    gate = ct_ptmc_or_availability(unified)
    assert gate == {
        "evaluable": False,
        "reason": "ptmc_or_unavailable",
        "ct_exact_or_rows": 1,
        "ptmc_exact_habit_rows": 0,
        "ptmc_exact_or_rows": 0,
    }

    unified2 = _Unified(
        rows=unified.rows
        + (
            _Row("ptmc", "ptmc_habit", True, ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
        )
    )
    assert ct_ptmc_or_availability(unified2)["evaluable"] is True


def test_atlas_professor_audit_has_no_raw_boolean_none_or_enum_token():
    row = atlas_professor_audit_rows([_AtlasState()], ["a_scale"])[0]
    assert row["classification"] == "not evaluable"
    assert row["CT exact A/M"] == "not satisfied"
    assert row["CT supercompatibility"] == "not evaluable"
    assert row["cofactor"] == "not satisfied"
    assert row["CT supercompatibility residual"] == "not evaluable"
    assert not any(value is None or isinstance(value, bool) for value in row.values())
    assert "not_evaluable" not in str(row)


def test_live_ui_source_contains_phase2b_zero_root_and_availability_guards():
    root = Path(__file__).resolve().parents[2]
    research = (root / "app" / "research_workspaces.py").read_text(encoding="utf-8")
    assert 'if rows:\n                st.dataframe(_scalar_table(rows)' in research
    assert "Evaluated — no admissible double-shear root exists at the selected fixed second parameter." in research
    assert "Only Cayron CT OR predictions are available for this invariant-line comparison" in research
    assert 'st.dataframe(\n                        _scalar_table(comparisons)' not in research.split('with st.expander("Invariant-line OR ↔ CT/PTMC OR comparison"):', 1)[1].split('with double_tab:', 1)[0].lstrip()[:120]


def test_atlas_machine_tokens_are_secondary_developer_audit():
    root = Path(__file__).resolve().parents[2]
    source = (root / "app" / "research_workspaces_v6.py").read_text(encoding="utf-8")
    assert "Atlas audit — human-readable" in source
    assert "Raw machine audit — developer / serialization" in source
    assert "atlas_professor_audit_rows(report.states, axes.keys())" in source


def test_cross_page_mirror_covers_legacy_scientific_selection_keys():
    state = {
        "mart_vi": 2,
        "mart_vj": 5,
        "manual_load_system": 1,
        "manual_a_0": 0.6,
        "manual_n_2": -0.8,
        "map_object": "[1 0 1]",
        "map_source_phase": "parent",
        "parallel_candidate": 3,
        "recon_observed_phase": "product",
        "sample_g_00": 1.0,
        "v4_twin_system": 2,
        "research_v7_pole_generate": True,
    }
    snapshot_persistent_widget_state(state)
    mirror = state[MIRROR_KEY]
    for key in (
        "mart_vi",
        "mart_vj",
        "manual_load_system",
        "manual_a_0",
        "manual_n_2",
        "map_object",
        "map_source_phase",
        "parallel_candidate",
        "recon_observed_phase",
        "sample_g_00",
        "v4_twin_system",
    ):
        assert key in mirror
    assert "research_v7_pole_generate" not in mirror
