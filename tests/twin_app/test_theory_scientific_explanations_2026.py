"""Numerical conclusions must follow exactly from actual prediction rows."""

from types import SimpleNamespace
from cualni_cryst.theory_unified import PredictionKind as K, TheoryKind as T
from cualni_cryst.theory_equivalence import MatchFamily as F

from twin_app.theory_scientific_explanations import (
    scientific_explanations, scientific_notes,
)


def make_match(agree):
    return SimpleNamespace(
        family=F.MM_TWIN, left_theory=T.CAYRON_CT,
        right_theory=T.BALL_JAMES,
        all_required_components_within_tolerance=agree,
        residuals=SimpleNamespace(or_disorientation_deg=None),
    )


def fake_report(*, exact=False, twin_count=0, habit_count=0, eps=None):
    ct = SimpleNamespace(
        analysis=SimpleNamespace(exact_compatible=exact,
                                 degeneracy_order=1 if exact else 0,
                                 nearest_zero_residual=0.012),
        exact_habit_planes=tuple(object() for _ in range(habit_count)),
    )
    bj = SimpleNamespace(
        variants=(), martensite_twin_branches=(),
        austenite_martensite_branches=(),
    )
    rows = [
        SimpleNamespace(prediction_kind=K.CT_MM_TWIN,
                        residuals={}, metadata={}, exact=True)
        for _ in range(twin_count)
    ]
    if eps is not None:
        rows.append(SimpleNamespace(
            prediction_kind=K.CT_SUPERCOMPATIBILITY,
            residuals={"ct_supercompatibility_dimensionless": eps},
            metadata={}, exact=True))
    return SimpleNamespace(
        rows=tuple(rows), ct_report=ct, ball_james_report=bj,
        ptmc_report=SimpleNamespace(), warnings=(),
    )


def fake_equivalence(matches=()):
    return SimpleNamespace(matches=matches, mm_independent_audit=None,
                           warnings=())


def test_all_four_theories_have_equations_logic_and_conclusions():
    entries = scientific_explanations(fake_report(), fake_equivalence())
    assert len(entries) == 4
    assert all(len(e["equations"]) > 0 and e["logic"] and e["conclusion"]
               and e["result"] and e["reference"] for e in entries)
    assert "CMC" in entries[0]["equations"][0]
    assert "CC3" in entries[3]["equations"][2]
    assert r"-\det(U^2)" in entries[3]["equations"][2]


def test_ct_absent_habit_is_not_labeled_supercompatible():
    entries = scientific_explanations(
        fake_report(exact=False, twin_count=4), fake_equivalence())
    assert "Exact CMC compatibility: no" in entries[0]["result"]
    assert "unavailable" in entries[0]["result"]
    assert "No claim" in entries[0]["conclusion"]


def test_shear_residual_is_calculated_only_when_supplied():
    entries = scientific_explanations(
        fake_report(exact=True, habit_count=2, eps=0.027),
        fake_equivalence())
    assert "0.027" in entries[0]["result"]
    assert "nonzero" in entries[0]["conclusion"].lower()


def test_partial_matches_are_not_called_full_agreement():
    entries = scientific_explanations(
        fake_report(), fake_equivalence((make_match(None), make_match(False))))
    assert "0 fully agree" in entries[1]["result"]
    assert "1 fail, 1 are incomplete" in entries[1]["result"]


def test_export_contains_equations_and_provenance():
    report = scientific_notes(fake_report(), fake_equivalence())
    assert "Cayron correspondence theory" in report
    assert "Ball–James nonlinear" in report
    assert "CC3" in report
    assert "ptmc_adapter.py" in report
    assert "Scientific conclusion" not in report or "**Conclusion:**" in report
