from app.session_state import (
    BoundResult,
    fingerprint,
    get_bound,
    mark_calculation_failure,
    mark_calculation_success,
    observe_draft,
    put_bound,
)


def test_fingerprint_is_order_stable_for_mappings():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


def test_bound_result_is_visible_only_for_matching_signature():
    state = {}
    put_bound(state, "x", "abc", {"value": 7})
    assert get_bound(state, "x", "abc") == {"value": 7}
    assert get_bound(state, "x", "def") is None


def test_draft_change_requires_recalculation_and_invalidates_dependents():
    state = {}
    mark_calculation_success(state, "v1")
    put_bound(state, "orientation_result", "or-1", {"ok": True})
    assert observe_draft(state, "v2") is True
    assert "orientation_result" not in state


def test_editing_back_does_not_silently_revive_old_result():
    state = {}
    mark_calculation_success(state, "v1")
    assert observe_draft(state, "v2") is True
    assert observe_draft(state, "v1") is True
    mark_calculation_success(state, "v1")
    assert observe_draft(state, "v1") is False


def test_failed_calculation_blocks_downstream_results():
    state = {}
    mark_calculation_success(state, "valid")
    put_bound(state, "calpad_cell_result", "cell", {"ok": True})
    mark_calculation_failure(state, RuntimeError("bad draft"))
    assert state["requires_recalculation"] is True
    assert "calpad_cell_result" not in state
