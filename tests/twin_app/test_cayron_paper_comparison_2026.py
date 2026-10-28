"""Cayron 2026 paper-ordered comparison must use existing numerical outputs."""

from types import SimpleNamespace as NS

from cualni_cryst.theory_unified import PredictionKind as K
from twin_app.cayron_paper_comparison_ui import (
    cayron_paper_comparison,
    cayron_paper_thesis_notes,
)


def ns(**kwargs):
    return NS(**kwargs)


def result_fixture():
    twins = ns(
        prediction_kind=K.CT_MM_TWIN, branch_label="CT O2 twin",
        metadata={"operator_index": 2, "twin_index": 0,
                  "twin_classification": "Compound", "rational_element": "mirror"},
        twin_plane_parent_crystal=(0, 0, 1),
        twin_direction_parent_crystal=(0, 1, 0),
        shear_magnitude=0.27325,
    )
    habit = ns(
        prediction_kind=K.CT_AM_HABIT, branch_label="CT A/M +",
        metadata={}, exact=True,
        habit_plane_parent_crystal=(1, -1, 2.41966),
        shape_vector_parent_crystal=(.36938, -.36938, -.05378),
        shape_vector_magnitude=.525,
        residuals={"cmc_nearest_zero_dimensionless": 1e-14},
    )
    shear = ns(
        prediction_kind=K.CT_SUPERCOMPATIBILITY,
        branch_label="CT habit 0 / O2",
        metadata={"operator_index": 2, "twin_index": 0, "habit_index": 0,
                  "twin_classification": "Compound"},
        habit_plane_parent_crystal=(1, -1, 2.41966),
        twin_plane_parent_crystal=(0, 0, 1),
        twin_direction_parent_crystal=(0, 1, 0),
        shape_vector_parent_crystal=(.36938, -.36938, -.05378),
        shear_magnitude=.27325,
        residuals={"ct_supercompatibility_dimensionless": .215},
    )
    # Source labels: exact CMC for the hypothetical NiTi C1 benchmark, but
    # epsilon 0.215 is NOT exact A/M/M supercompatibility.
    ct = ns(analysis=ns(exact_compatible=True, degeneracy_order=1,
                        reason="first-order degeneracy: two compatible habit planes",
                        eigenvalues=(-.1, 0., .2), nearest_zero_residual=1e-14))
    bj_twin = ns(base_variant_index=0, other_variant_index=1,
                 branch=1, rank_one_residual=1e-11,
                 cofactor=ns(cc1_lambda2_minus_one=0., cc2_value=0.,
                             cc3_margin=-.01, all_satisfied=False))
    ptmc_solution = ns(
        lis_type="twin", base_variant_index=0, other_variant_index=1,
        twin_branch=1, other_variant_volume_fraction=.3,
        habit_plane_parent_crystal=(1, 2, 3),
        dilatational_factor=1., middle_stretch_residual=2e-11,
        rank_one_residual=3e-11, true_invariant_plane=True,
    )
    match = ns(
        left_theory=ns(value="cayron_ct"),
        right_theory=ns(value="ball_james"),
        family=ns(value="martensite_martensite_twin"),
        left_branch="CT O2 twin", right_branch="BJ U0–U1",
        all_required_components_within_tolerance=None,
        residuals=ns(to_dict=lambda: {"twin_plane_angle_deg": 0.1}),
        missing_required_components=("rank_one_tensor_relative",),
    )
    # Orientation matching is not part of Cayron 2026's A/M/M argument.
    or_match = ns(
        left_theory=ns(value="cayron_ct"),
        right_theory=ns(value="ptmc"),
        family=ns(value="orientation_relationship"),
        left_branch="CT OR", right_branch="PTMC OR",
        all_required_components_within_tolerance=True,
        residuals=ns(to_dict=lambda: {"or_disorientation_deg": 0.1}),
        missing_required_components=(),
    )
    unified = ns(
        rows=(twins, habit, shear),
        ct_report=ct,
        ball_james_report=ns(martensite_twin_branches=(bj_twin,)),
        ptmc_report=ns(solutions=(ptmc_solution,)),
    )
    return unified, ns(matches=(match, or_match))


def test_paper_order_uses_returned_ct_habits_and_transformation_twins():
    unified, equivalence = result_fixture()
    result = cayron_paper_comparison(unified, equivalence)
    assert result["mm_twins"][0]["Twin shear |s|"] == 0.27325
    assert result["am_habits"][0]["Exact CMC habit"] is True
    assert result["ct_cmc"]["degeneracy_order"] == 1
    assert result["shear_pairs"][0]["Shear/shear ε (dimensionless)"] == .215
    assert result["cofactor_pairs"][0]["CC1–CC3 jointly satisfied"] is False
    assert result["ptmc_laminate_habits"][0]["Laminate fraction f"] == .3


def test_comparison_keeps_partial_physical_matches_incomplete():
    unified, equivalence = result_fixture()
    result = cayron_paper_comparison(unified, equivalence)
    assert len(result["cross_theory_matches"]) == 1
    assert result["cross_theory_matches"][0]["Comparison outcome"] == "not fully comparable"
    assert "rank_one_tensor_relative" in result["cross_theory_matches"][0]["Missing required observables"]


def test_absence_of_exact_habit_is_not_presented_as_failed_shear_shear():
    unified, equivalence = result_fixture()
    unified.rows = tuple(row for row in unified.rows
                         if row.prediction_kind is K.CT_MM_TWIN)
    unified.ct_report.analysis.exact_compatible = False
    result = cayron_paper_comparison(unified, equivalence)
    assert result["shear_pairs"] == []
    assert result["am_habits"] == []
    assert result["ct_cmc"]["exact_compatible"] is False


def test_empty_ptmc_means_no_reported_solution_not_zero_physical_habits():
    unified, equivalence = result_fixture()
    unified.ptmc_report = None
    result = cayron_paper_comparison(unified, equivalence)
    assert result["ptmc_laminate_habits"] == []


def test_thesis_export_explicitly_preserves_cayron_scope():
    unified, equivalence = result_fixture()
    notes = cayron_paper_thesis_notes(unified, equivalence)
    assert "M/M twins -> A/M CMC/SMC -> shear/shear" in notes
    assert "0.215" in notes
    assert "different physical construction" in notes
    assert "not unique observed twins" in notes
