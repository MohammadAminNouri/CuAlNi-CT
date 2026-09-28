from __future__ import annotations

from types import SimpleNamespace

from app.scientific_interpretation import (
    am_existence_finding,
    or_comparison_finding,
    ptmc_finding,
    supercompatibility_finding,
    transformation_findings,
)


def enum(value: str):
    return SimpleNamespace(value=value)


def row(theory: str, kind: str, *, exact=None, metadata=None, residuals=None):
    return SimpleNamespace(
        theory=enum(theory),
        prediction_kind=enum(kind),
        exact=exact,
        metadata={} if metadata is None else metadata,
        residuals={} if residuals is None else residuals,
    )


def test_single_variant_absence_is_reported_as_valid_agreement_not_theory_failure():
    finding = am_existence_finding(
        {
            "ct_am_exact_compatible": False,
            "ct_vs_ball_james_am_existence_agreement": True,
        },
        exact_ct_habits=0,
        bj_am=0,
        approx_ct_habits=2,
    )
    assert "agree" in finding.conclusion.lower()
    assert "no exact single-variant" in finding.conclusion.lower()
    assert "nearest-degeneracy" in finding.limitation.lower()


def test_ptmc_disabled_is_not_misreported_as_no_solution():
    unified = SimpleNamespace(rows=())
    finding = ptmc_finding(unified, enabled=False)
    assert "not evaluated" in finding.conclusion.lower()
    assert "not attempted" in finding.limitation.lower()


def test_requested_supercompatibility_is_not_evaluated_from_approximate_habits():
    unified = SimpleNamespace(
        rows=(
            row("cayron_ct", "ct_am_habit", exact=False),
            row("cayron_ct", "ct_am_habit", exact=False),
        )
    )
    finding = supercompatibility_finding(
        unified,
        requested=True,
        algebraic_tolerance=1e-10,
    )
    assert "not evaluable" in finding.conclusion.lower()
    assert "no exact ct a/m habit" in finding.conclusion.lower()
    assert "does not promote" in finding.how.lower()


def test_or_gap_outside_tolerance_is_close_but_not_equivalent():
    match = SimpleNamespace(
        family=enum("orientation_relationship"),
        left_theory=enum("cayron_ct"),
        right_theory=enum("ptmc"),
        residuals=SimpleNamespace(or_disorientation_deg=0.10868836866302703),
    )
    equivalence = SimpleNamespace(matches=(match,))
    finding = or_comparison_finding(equivalence, enabled=True, tolerance_deg=1e-4)
    assert "no exact" in finding.conclusion.lower()
    assert "0.108688" in finding.conclusion
    assert "same derivation" in finding.limitation.lower()


def test_transformation_interpretation_keeps_ct_nearest_degeneracy_separate_from_exact():
    result = {
        "summary": {
            "ct_exact_compatible": False,
            "lambda2": 0.9980600226,
            "lambda2_residual": 0.0019399774,
        },
        "ct_detail": {
            "nearest_zero_residual": 0.0038761913,
            "degeneracy_order": 0,
            "inertia": [2, 0, 1],
            "reason": "no exact CMC degeneracy",
        },
        "ball_james_detail": {
            "lambda2_exact": False,
            "solutions": [],
        },
    }
    ct, bj, agreement = transformation_findings(result)
    assert "not satisfied" in ct.conclusion.lower()
    assert "approximate nearest-degeneracy planes are kept separate" in ct.status.lower()
    assert "not satisfied" in bj.conclusion.lower()
    assert "same a/m existence classification" in agreement.conclusion.lower()


def test_mm_audit_is_authoritative_and_explains_configuration_push_forward():
    from app.scientific_interpretation import mm_audit_finding

    finding = mm_audit_finding(
        {
            "success": True,
            "relation_count": 8,
            "ct_coverage_complete": True,
            "classification_exact": True,
            "max_ct_geometry_angle_deg": 6.95e-13,
            "max_ball_james_geometry_angle_deg": 7.53e-13,
            "max_shear_relative_residual": 7.82e-14,
            "max_rank_one_residual": 2.05e-14,
            "max_rotation_residual": 5.74e-16,
        }
    )
    assert finding is not None
    assert "all 8" in finding.conclusion.lower()
    assert "push-forward" in finding.rationale.lower()
    assert "partial" in finding.limitation.lower()


def test_ptmc_exact_laminate_is_distinguished_from_single_variant_compatibility():
    ptmc_row = SimpleNamespace(
        theory=enum("ptmc"),
        prediction_kind=enum("ptmc_habit"),
        exact=True,
        metadata={"true_invariant_plane": True},
        residuals={"rank_one": 2.0e-16, "middle_stretch": 1.0e-15},
    )
    finding = ptmc_finding(SimpleNamespace(rows=(ptmc_row,)), enabled=True)
    assert "1 exact twinned-laminate" in finding.conclusion.lower()
    assert "does not contradict" in finding.rationale.lower()
    assert "homogeneous" in finding.physical_meaning.lower()


def test_cofactor_conditions_stay_separate_and_are_not_collapsed_into_score():
    from app.scientific_interpretation import cofactor_finding

    rows = tuple(
        SimpleNamespace(
            theory=enum("ball_james"),
            prediction_kind=enum("ball_james_mm"),
            metadata={
                "cofactor_cc1_satisfied": False,
                "cofactor_cc2_satisfied": False,
                "cofactor_cc3_satisfied": i < 3,
                "cofactor_all_satisfied": False,
            },
        )
        for i in range(4)
    )
    finding = cofactor_finding(SimpleNamespace(rows=rows))
    assert "0/4" in finding.conclusion
    assert "reported separately" in finding.rationale.lower()
    quantities = {item.quantity for item in finding.evidence}
    assert {"CC1 satisfied", "CC2 satisfied", "CC3 satisfied", "CC1 ∧ CC2 ∧ CC3"} <= quantities


def test_supercompatibility_exact_rows_use_explicit_native_residual_tolerance():
    exact_habit = row("cayron_ct", "ct_am_habit", exact=True)
    super_row = row(
        "cayron_ct",
        "ct_supercompatibility",
        exact=True,
        residuals={"ct_supercompatibility_dimensionless": 5e-12},
    )
    finding = supercompatibility_finding(
        SimpleNamespace(rows=(exact_habit, super_row)),
        requested=True,
        algebraic_tolerance=1e-10,
    )
    assert "1/1" in finding.conclusion
    assert "ct-native" in finding.limitation.lower()
