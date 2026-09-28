from dataclasses import replace

import numpy as np

from cualni_cryst.compatibility_atlas import nearest_exact_6m_beta_projection
from cualni_cryst.project_state import james_hane_6m_reference_project
from cualni_cryst.theory_equivalence import (
    AssignmentScales,
    EquivalenceTolerances,
    MatchCompleteness,
    MatchFamily,
)
from cualni_cryst.theory_equivalence_optimized import (
    CachedPhysicalBranchMatcher,
    build_equivalence_report_optimized,
)
from cualni_cryst.theory_observable_contracts import (
    ShapeVectorRole,
    infer_native_shape_vector_role,
)
from cualni_cryst.theory_unified import (
    ComparisonRow,
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
    projected, _, _ = nearest_exact_6m_beta_projection(parent.lattice, product.lattice)
    product2 = replace(product, lattice=projected)
    phases = tuple(product2 if phase.phase_id == product.phase_id else phase for phase in project.phases)
    result = replace(project, project_id="optimized_equivalence_exact_beta", phases=phases)
    result.validate().assert_passed()
    return result


def test_optimized_matcher_preserves_full_exact_ct_ball_james_am_agreement():
    project = _exact_beta_project()
    unified = TheoryComparisonAdapter(project, "do3_to_6m_reference").compare(
        ptmc_mode=PTMCMode.NONE,
        include_ct_closing_gap=False,
        include_ct_supercompatibility=False,
    )
    report = build_equivalence_report_optimized(project, "do3_to_6m_reference", unified)
    matches = report.matches_for(
        family=MatchFamily.AM_INTERFACE,
        left=TheoryKind.CAYRON_CT,
        right=TheoryKind.BALL_JAMES,
    )
    assert matches
    assert all(item.completeness is MatchCompleteness.FULL for item in matches)
    assert all(item.all_required_components_within_tolerance is True for item in matches)
    assert max(item.residuals.habit_plane_angle_deg for item in matches) < 2e-5
    assert max(item.residuals.rank_one_tensor_relative for item in matches) < 1e-8


def test_ct_native_smc_vector_is_preserved_but_comparison_uses_audited_rank_one_observable():
    project = _exact_beta_project()
    unified = TheoryComparisonAdapter(project, "do3_to_6m_reference").compare(
        ptmc_mode=PTMCMode.NONE,
        include_ct_closing_gap=False,
        include_ct_supercompatibility=False,
    )
    ct = next(
        row for row in unified.rows
        if row.theory is TheoryKind.CAYRON_CT
        and row.prediction_kind is PredictionKind.CT_AM_HABIT
        and row.exact is True
    )
    assert infer_native_shape_vector_role(
        theory=ct.theory,
        prediction_kind=ct.prediction_kind,
        exact=ct.exact,
        metadata=ct.metadata,
    ) is ShapeVectorRole.CT_SMC_IPS_D

    matcher = CachedPhysicalBranchMatcher(project, "do3_to_6m_reference")
    native = matcher._stored_shape_vector(ct)
    comparable = matcher._am_rank_one_shape_vector(ct)
    normal = matcher._habit_normal(ct)
    assert native is not None and comparable is not None and normal is not None

    # Native CT d and classical b are different representations.  The code must
    # preserve d, then derive b only at the comparison boundary.
    assert not np.allclose(native, comparable, atol=1e-10, rtol=1e-10)
    audit = matcher._ct_am_bridge_audit(ct)
    assert audit is not None
    assert audit.maximum_residual <= matcher._am_bridge_guard_tolerance
    assert np.allclose(audit.rank_one_shape_cartesian, comparable)


def test_approximate_ct_habit_is_diagnostic_only_and_cannot_manufacture_rank_one_b():
    project = _exact_beta_project()
    transformation = project.transformation("do3_to_6m_reference")
    product = project.phase(transformation.product_phase_id)
    perturbed_product = replace(
        product,
        lattice=replace(product.lattice, beta_deg=float(product.lattice.beta_deg) + 0.2),
    )
    project = replace(
        project,
        project_id="optimized_equivalence_approximate_beta",
        phases=tuple(
            perturbed_product if phase.phase_id == product.phase_id else phase
            for phase in project.phases
        ),
    )
    project.validate().assert_passed()
    unified = TheoryComparisonAdapter(project, "do3_to_6m_reference").compare(
        ptmc_mode=PTMCMode.NONE,
        include_ct_closing_gap=False,
        include_ct_supercompatibility=False,
    )
    approximate = [
        row for row in unified.rows
        if row.theory is TheoryKind.CAYRON_CT
        and row.prediction_kind is PredictionKind.CT_AM_HABIT
        and row.exact is False
    ]
    assert approximate
    matcher = CachedPhysicalBranchMatcher(project, "do3_to_6m_reference")
    assert all(matcher._stored_shape_vector(row) is not None for row in approximate)
    assert all(matcher._am_rank_one_shape_vector(row) is None for row in approximate)
    assert all(matcher._am_rank_one_tensor(row) is None for row in approximate)

    report = build_equivalence_report_optimized(project, "do3_to_6m_reference", unified)
    exact_matches = report.matches_for(
        family=MatchFamily.AM_INTERFACE,
        left=TheoryKind.CAYRON_CT,
        right=TheoryKind.BALL_JAMES,
    )
    assert exact_matches == ()


def test_true_ips_ptmc_rank_one_vector_remains_physically_comparable_not_blanket_suppressed():
    project = _exact_beta_project()
    unified = TheoryComparisonAdapter(project, "do3_to_6m_reference").compare(
        ptmc_mode=PTMCMode.NONE,
        include_ct_closing_gap=False,
        include_ct_supercompatibility=False,
    )
    ct = next(
        row for row in unified.rows
        if row.theory is TheoryKind.CAYRON_CT
        and row.prediction_kind is PredictionKind.CT_AM_HABIT
        and row.exact is True
    )
    matcher = CachedPhysicalBranchMatcher(project, "do3_to_6m_reference")
    ct_b = matcher._am_rank_one_shape_vector(ct)
    ct_n = matcher._habit_normal(ct)
    assert ct_b is not None and ct_n is not None

    synthetic_ptmc = ComparisonRow(
        row_id="synthetic_true_ips_ptmc",
        theory=TheoryKind.PTMC,
        prediction_kind=PredictionKind.PTMC_HABIT,
        branch_label="synthetic true IPS",
        exact=True,
        habit_normal_parent_cartesian=tuple(float(x) for x in ct_n),
        shape_vector_parent_cartesian=tuple(float(x) for x in ct_b),
        shape_vector_magnitude=float(np.linalg.norm(ct_b)),
        metadata={"dilatational_factor": 1.0, "true_invariant_plane": True},
    )
    residuals = matcher._family_residuals(ct, synthetic_ptmc, MatchFamily.AM_INTERFACE)
    assert residuals.habit_plane_angle_deg < 1e-10
    assert residuals.rank_one_tensor_relative < 1e-10
    assert residuals.shape_direction_projective_deg < 1e-10
    assert residuals.shape_magnitude_relative < 1e-10


def test_dilated_ptmc_rank_one_vector_is_not_mislabeled_as_parent_identity_ips():
    project = _exact_beta_project()
    matcher = CachedPhysicalBranchMatcher(project, "do3_to_6m_reference")
    row = ComparisonRow(
        row_id="synthetic_dilated_ptmc",
        theory=TheoryKind.PTMC,
        prediction_kind=PredictionKind.PTMC_HABIT,
        branch_label="synthetic dilated plane",
        exact=False,
        habit_normal_parent_cartesian=(1.0, 0.0, 0.0),
        shape_vector_parent_cartesian=(0.1, 0.0, 0.0),
        shape_vector_magnitude=0.1,
        metadata={"dilatational_factor": 1.01, "true_invariant_plane": False},
    )
    assert matcher._stored_shape_vector(row) is not None
    assert matcher._am_rank_one_shape_vector(row) is None
    assert matcher._am_rank_one_tensor(row) is None


def test_vectorized_or_angle_matches_orientation_kernel_disorientation():
    project = _exact_beta_project()
    matcher = CachedPhysicalBranchMatcher(project, "do3_to_6m_reference")

    angle = np.deg2rad(11.0)
    Rz = (
        (float(np.cos(angle)), float(-np.sin(angle)), 0.0),
        (float(np.sin(angle)), float(np.cos(angle)), 0.0),
        (0.0, 0.0, 1.0),
    )
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    left = ComparisonRow(
        row_id="or_left",
        theory=TheoryKind.CAYRON_CT,
        prediction_kind=PredictionKind.CT_CLOSING_GAP_OR,
        branch_label="left",
        exact=True,
        or_parent_from_product=identity,
    )
    right = ComparisonRow(
        row_id="or_right",
        theory=TheoryKind.PTMC,
        prediction_kind=PredictionKind.PTMC_HABIT,
        branch_label="right",
        exact=True,
        or_parent_from_product=Rz,
    )

    left_R = matcher._or_in_symmetric_metric_frame(left)
    right_R = matcher._or_in_symmetric_metric_frame(right)
    expected = matcher.kernel.disorientation(left_R, right_R).angle_deg
    orbit = matcher._or_orbit(left)
    actual = matcher._angles_for_orbit_against_many(orbit, np.stack([right_R]))[0]
    assert abs(float(actual) - float(expected)) < 1e-10


def test_staged_unified_components_reassemble_same_native_rows_as_direct_compare():
    from cualni_cryst.theory_unified_components import StagedTheoryComparisonAdapter

    project = _exact_beta_project()
    staged = StagedTheoryComparisonAdapter(project, "do3_to_6m_reference")
    ct = staged.ct_component(include_closing_gap=False, include_supercompatibility=False)
    bj = staged.ball_james_component(fraction_samples=101)
    ptmc = staged.ptmc_component(mode=PTMCMode.NONE)
    assembled = staged.assemble(ct, bj, ptmc)

    direct = TheoryComparisonAdapter(project, "do3_to_6m_reference").compare(
        ptmc_mode=PTMCMode.NONE,
        include_ct_closing_gap=False,
        include_ct_supercompatibility=False,
        ball_james_fraction_samples=101,
    )

    assert [row.row_id for row in assembled.rows] == [row.row_id for row in direct.rows]
    assert assembled.warnings == direct.warnings
    assert assembled.notes == direct.notes
