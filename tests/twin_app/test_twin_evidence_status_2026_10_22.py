"""Permanent regression for evidence provenance and separate unresolved status."""
from types import SimpleNamespace
import pytest
from twin_app.twin_evidence_status_2026_10_22 import (
    evidence_category, category_for_construction, NONCONVENTIONAL,
)
from twin_app.mode_catalogue import ModeEntry, PhysicalTwinMode


def _b(label, status):
    return SimpleNamespace(classification=label, classification_status=status,
                           habit_solutions=("habit1", "habit2"), branch=-1)


def test_direct_classical_claim_is_not_confused_with_proof():
    assert evidence_category("Type I", "cross-locked: tested elsewhere") == "direct_classical_claim"


def test_transported_is_separate_from_direct():
    status = ("symmetry-transported from directly cross-locked F1:M1-P1; "
              "G-index 2; ordered stretch and physical rank-one tensor residuals "
              "1e-15, 1e-15, 1e-15; this is NOT an independent discrete cross-lock "
              "for the target pair")
    assert evidence_category("Type II", status) == "symmetry_transported_classical_claim"


def test_parent_symmetry_excluded_is_not_unresolved():
    status = ("PROVED: neither a parent mirror nor a proper 180-degree rotation "
              "in the supplied parent group (48 operations) maps the two ordered "
              "stretch variants. Hence no classical parent Type-I mirror or "
              "Type-II twofold symmetry route exists. Rank-one compatibility remains valid")
    assert evidence_category(NONCONVENTIONAL, status) == "parent_mirror_twofold_excluded_under_supplied_group"
    assert evidence_category("Exact rank-one relation — classification cross-lock unresolved", "UNRESOLVED: insufficient evidence") == "unresolved_classical_provenance"


def test_invalid_claims_fail_closed():
    for label, status in [
        ("Compound", "unverified"),
        (NONCONVENTIONAL, "PROVED: impossible by assertion"),
        ("Type II", "symmetry-transported from anywhere"),
        ("TYPE III", "cross-locked: anything"),
        ("Exact rank-one relation — unresolved", ""),
    ]:
        with pytest.raises(ValueError):
            evidence_category(label, status)


def test_mode_entries_do_not_lose_geometry_when_categorized():
    status = ("PROVED: neither a parent mirror nor a proper 180-degree rotation "
              "in the supplied parent group (48 operations) maps the two ordered "
              "stretch variants. Hence no classical parent Type-I mirror or "
              "Type-II twofold symmetry route exists.")
    original = _b(NONCONVENTIONAL, status)
    unresolved = _b("Exact rank-one relation — operator family is not classical", "insufficient discrete operator route")
    a = ModeEntry(None, original, "F1")
    b = ModeEntry(None, unresolved, "F2")
    mode = PhysicalTwinMode("Mode 1", ((1,2),), (a,b), 0.08, ("Unresolved",))
    assert mode.by_type("Unresolved") == (a,b)  # existing UI contract unchanged
    assert mode.by_evidence_category("parent_mirror_twofold_excluded_under_supplied_group") == (a,)
    assert mode.by_evidence_category("unresolved_classical_provenance") == (b,)
    assert mode.evidence_counts()["parent_mirror_twofold_excluded_under_supplied_group"] == 1
    assert original.habit_solutions == ("habit1", "habit2")
    assert category_for_construction(original) == a.evidence_category
