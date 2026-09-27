from dataclasses import replace

import numpy as np

from cualni_cryst.compatibility_atlas import nearest_exact_6m_beta_projection
from cualni_cryst.project_state import james_hane_6m_reference_project
from cualni_cryst.theory_equivalence import MatchFamily
from cualni_cryst.theory_equivalence_optimized import (
    CachedPhysicalBranchMatcher,
    build_equivalence_report_optimized,
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


def test_optimized_matcher_preserves_exact_ct_ball_james_am_agreement():
    project = _exact_beta_project()
    unified = TheoryComparisonAdapter(project, "do3_to_6m_reference").compare(
        ptmc_mode=PTMCMode.NONE,
        include_ct_closing_gap=False,
        include_ct_supercompatibility=False,
    )
    report = build_equivalence_report_optimized(
        project,
        "do3_to_6m_reference",
        unified,
    )
    matches = report.matches_for(
        family=MatchFamily.AM_INTERFACE,
        left=TheoryKind.CAYRON_CT,
        right=TheoryKind.BALL_JAMES,
    )
    assert matches
    assert all(item.all_required_components_within_tolerance is True for item in matches)
    assert max(item.residuals.habit_plane_angle_deg for item in matches) < 2e-5


def test_ct_approximate_am_rows_are_excluded_from_exact_equivalence_pool():
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
        row
        for row in unified.rows
        if row.theory is TheoryKind.CAYRON_CT
        and row.prediction_kind is PredictionKind.CT_AM_HABIT
        and row.exact is False
    ]
    assert approximate, "reference project should expose nearest-degeneracy diagnostics"

    matcher = CachedPhysicalBranchMatcher(project, "do3_to_6m_reference")
    selected = matcher._rows(
        unified,
        TheoryKind.CAYRON_CT,
        (PredictionKind.CT_AM_HABIT,),
    )
    assert all(row.exact is True for row in selected)
    assert not {row.row_id for row in approximate} & {row.row_id for row in selected}


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
    ct = staged.ct_component(
        include_closing_gap=False,
        include_supercompatibility=False,
    )
    bj = staged.ball_james_component(fraction_samples=101)
    ptmc = staged.ptmc_component(mode=PTMCMode.NONE)
    assembled = staged.assemble(ct, bj, ptmc)

    direct = TheoryComparisonAdapter(project, "do3_to_6m_reference").compare(
        ptmc_mode=PTMCMode.NONE,
        include_ct_closing_gap=False,
        include_ct_supercompatibility=False,
        ball_james_fraction_samples=101,
    )

    assembled_ids = [row.row_id for row in assembled.rows]
    direct_ids = [row.row_id for row in direct.rows]
    assert assembled_ids == direct_ids
    assert assembled.warnings == direct.warnings
    assert assembled.notes == direct.notes
