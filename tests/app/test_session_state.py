from app.session_state import (
    BoundResult,
    CalculatedSnapshot,
    fingerprint,
    get_bound,
    mark_calculation_failure,
    mark_calculation_success,
    observe_draft,
    put_bound,
    record_draft,
)


def test_fingerprint_is_order_stable_for_mappings():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


def test_bound_result_is_visible_only_for_matching_signature():
    state = {}
    put_bound(state, "x", "abc", {"value": 7})
    assert get_bound(state, "x", "abc") == {"value": 7}
    assert get_bound(state, "x", "def") is None


def test_draft_change_marks_stale_but_preserves_previous_results():
    state = {"current_project_payload": {"project_id": "p"}, "current_response": object()}
    mark_calculation_success(state, "v1")
    put_bound(state, "orientation_result", "or-1", {"ok": True})
    assert observe_draft(state, "v2") is True
    assert isinstance(state["orientation_result"], BoundResult)
    assert state["calculated_draft_signature"] == "v1"


def test_editing_back_to_old_values_does_not_silently_revive_stale_results():
    state = {"current_project_payload": {"project_id": "p"}, "current_response": object()}
    mark_calculation_success(state, "v1")
    assert observe_draft(state, "v2") is True
    assert observe_draft(state, "v1") is True
    mark_calculation_success(state, "v1")
    assert observe_draft(state, "v1") is False


def test_failed_calculation_is_non_destructive():
    response = object()
    state = {"current_project_payload": {"project_id": "p"}, "current_response": response}
    mark_calculation_success(state, "valid")
    put_bound(state, "calpad_cell_result", "cell", {"ok": True})
    mark_calculation_failure(state, RuntimeError("bad draft"))
    assert state["requires_recalculation"] is True
    assert isinstance(state["calpad_cell_result"], BoundResult)
    assert state["current_response"] is response
    assert isinstance(state["calculated_snapshot"], CalculatedSnapshot)
    assert state["calculated_snapshot"].signature == "valid"


def test_success_replaces_calculated_snapshot_and_invalidates_old_dependents():
    state = {"current_project_payload": {"project_id": "p1"}, "current_response": "r1"}
    mark_calculation_success(state, "v1")
    put_bound(state, "orientation_result", "old", 1)
    state["current_project_payload"] = {"project_id": "p2"}
    state["current_response"] = "r2"
    mark_calculation_success(state, "v2")
    assert "orientation_result" not in state
    assert state["calculated_snapshot"].signature == "v2"
    assert state["calculated_snapshot"].project_payload == {"project_id": "p2"}


def test_draft_snapshot_is_independent_from_source_mapping():
    source = {"cell": {"a": 1.0}}
    state = {}
    record_draft(state, source, "sig")
    source["cell"]["a"] = 2.0
    assert state["draft_snapshot"]["cell"]["a"] == 1.0
