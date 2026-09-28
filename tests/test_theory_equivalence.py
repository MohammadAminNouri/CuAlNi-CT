
from dataclasses import replace

import numpy as np

from cualni_cryst.compatibility_atlas import nearest_exact_6m_beta_projection
from cualni_cryst.project_state import james_hane_6m_reference_project
from cualni_cryst.theory_equivalence import (
    MatchCompleteness,
    MatchFamily,
    build_equivalence_report,
)
from cualni_cryst.theory_unified import (
    ExperimentalObservation,
    PTMCMode,
    PredictionKind,
    TheoryComparisonAdapter,
    TheoryKind,
)


def _exact_beta_project():
    project = james_hane_6m_reference_project()
    transformation = project.transformation("do3_to_6m_reference")
    parent = project.phase(transformation.parent_phase_id)
    product = project.phase(transformation.product_phase_id)
    projected, _, _ = nearest_exact_6m_beta_projection(
        parent.lattice,
        product.lattice,
    )
    product2 = replace(product, lattice=projected)
    phases = tuple(
        product2 if phase.phase_id == product.phase_id else phase
        for phase in project.phases
    )
    result = replace(project, project_id="equivalence_exact_beta", phases=phases)
    result.validate().assert_passed()
    return result


def test_ct_am_ball_james_exact_branches_are_matched_one_to_one():
    project = _exact_beta_project()
    adapter = TheoryComparisonAdapter(project, "do3_to_6m_reference")
    unified = adapter.compare(
        ptmc_mode=PTMCMode.NONE,
        include_ct_closing_gap=False,
        include_ct_supercompatibility=False,
    )
    report = build_equivalence_report(
        project,
        "do3_to_6m_reference",
        unified,
        include_independent_mm_audit=False,
    )
    matches = report.matches_for(
        family=MatchFamily.AM_INTERFACE,
        left=TheoryKind.CAYRON_CT,
        right=TheoryKind.BALL_JAMES,
    )
    assert matches
    assert len({item.left_row_id for item in matches}) == len(matches)
    assert len({item.right_row_id for item in matches}) == len(matches)
    assert max(item.residuals.habit_plane_angle_deg for item in matches) < 2e-5
    assert all(item.residuals.rank_one_tensor_relative is not None for item in matches)
    assert max(item.residuals.rank_one_tensor_relative for item in matches) < 1e-8
    assert all(item.completeness is MatchCompleteness.FULL for item in matches)
    assert all(item.all_required_components_within_tolerance is True for item in matches)
    assert all(
        item.within_tolerance.get("habit_plane_angle_deg") is True
        for item in matches
    )
    assert report.classification_checks[
        "ct_vs_ball_james_am_existence_agreement"
    ] is True


def test_ct_ptmc_matcher_preserves_separate_physical_residual_components():
    project = _exact_beta_project()
    unified = TheoryComparisonAdapter(
        project, "do3_to_6m_reference"
    ).compare(
        ptmc_mode=PTMCMode.ALL_TWINNING,
        include_ct_closing_gap=True,
        include_ct_supercompatibility=True,
    )
    report = build_equivalence_report(
        project,
        "do3_to_6m_reference",
        unified,
        include_independent_mm_audit=False,
    )
    am = report.matches_for(
        family=MatchFamily.AM_INTERFACE,
        left=TheoryKind.CAYRON_CT,
        right=TheoryKind.PTMC,
    )
    assert am
    assert all(item.residuals.habit_plane_angle_deg is not None for item in am)
    # No aggregate "winner score" is part of the public result.
    assert all("score" not in item.to_dict() for item in report.matches)


def test_experiment_or_residual_uses_symmetry_reduced_orientation_distance():
    project = _exact_beta_project()
    adapter = TheoryComparisonAdapter(project, "do3_to_6m_reference")
    baseline = adapter.compare(
        ptmc_mode=PTMCMode.ALL_TWINNING,
        include_ct_closing_gap=True,
        include_ct_supercompatibility=False,
    )
    target = next(
        row
        for row in baseline.rows
        if row.prediction_kind is PredictionKind.PTMC_HABIT
        and row.or_parent_from_product is not None
    )
    observation = ExperimentalObservation(
        observation_id="exact_ptmc_or",
        or_parent_from_product=target.or_parent_from_product,
        uncertainty={"orientation_deg": 0.1},
    )
    unified = adapter.compare(
        ptmc_mode=PTMCMode.ALL_TWINNING,
        include_ct_closing_gap=True,
        include_ct_supercompatibility=False,
        experiments=(observation,),
    )
    report = build_equivalence_report(
        project,
        "do3_to_6m_reference",
        unified,
        include_independent_mm_audit=False,
    )
    exact = [
        item
        for item in report.experiment_residuals
        if item.observation_row_id == "experiment_exact_ptmc_or"
        and item.theory_row_id == target.row_id
    ]
    assert len(exact) == 1
    assert exact[0].residuals.or_disorientation_deg < 1e-8
    assert exact[0].uncertainty_normalized["or_disorientation_deg"] < 1e-7


def test_mixed_ct_ball_james_twin_direction_is_not_compared_across_configurations():
    project = _exact_beta_project()
    unified = TheoryComparisonAdapter(
        project, "do3_to_6m_reference"
    ).compare(
        ptmc_mode=PTMCMode.ALL_TWINNING,
        include_ct_closing_gap=False,
        include_ct_supercompatibility=True,
    )
    report = build_equivalence_report(
        project,
        "do3_to_6m_reference",
        unified,
        include_independent_mm_audit=True,
    )
    mm = report.matches_for(
        family=MatchFamily.MM_TWIN,
        left=TheoryKind.CAYRON_CT,
        right=TheoryKind.BALL_JAMES,
    )
    assert mm
    assert all(item.residuals.twin_direction_angle_deg is None for item in mm)
    assert all(item.completeness is MatchCompleteness.PARTIAL for item in mm)
    assert all(item.all_required_components_within_tolerance is None for item in mm)
    assert report.mm_independent_audit is not None


def test_equivalence_report_exports_component_tolerances_without_a_winner_score():
    project = _exact_beta_project()
    unified = TheoryComparisonAdapter(project, "do3_to_6m_reference").compare(
        ptmc_mode=PTMCMode.NONE,
        include_ct_closing_gap=False,
        include_ct_supercompatibility=False,
    )
    report = build_equivalence_report(
        project,
        "do3_to_6m_reference",
        unified,
        include_independent_mm_audit=False,
    )
    payload = report.to_dict()
    assert payload["tolerances"]["plane_angle_deg"] > 0.0
    assert payload["matches"]
    assert all("within_tolerance" in item for item in payload["matches"])
    assert "score" not in payload


def test_parent_reference_experimental_twin_direction_is_not_compared_to_bj_current_shear():
    project = _exact_beta_project()
    adapter = TheoryComparisonAdapter(project, "do3_to_6m_reference")
    baseline = adapter.compare(
        ptmc_mode=PTMCMode.NONE,
        include_ct_closing_gap=False,
        include_ct_supercompatibility=True,
    )
    target = next(
        row
        for row in baseline.rows
        if row.prediction_kind is PredictionKind.BALL_JAMES_MM
        and row.twin_direction_parent_cartesian is not None
    )
    observation = ExperimentalObservation(
        observation_id="parent_reference_twin_direction",
        twin_direction_parent_cartesian=target.twin_direction_parent_cartesian,
    )
    unified = adapter.compare(
        ptmc_mode=PTMCMode.NONE,
        include_ct_closing_gap=False,
        include_ct_supercompatibility=True,
        experiments=(observation,),
    )
    report = build_equivalence_report(
        project,
        "do3_to_6m_reference",
        unified,
        include_independent_mm_audit=False,
    )
    residual = next(
        item
        for item in report.experiment_residuals
        if item.observation_row_id == "experiment_parent_reference_twin_direction"
        and item.theory_row_id == target.row_id
    )
    assert residual.residuals.twin_direction_angle_deg is None
