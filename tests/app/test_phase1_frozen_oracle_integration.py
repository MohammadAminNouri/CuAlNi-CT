from __future__ import annotations

import numpy as np
import pytest

from app.application import CalculationRequest, LatticeInput, PhaseInput, TransformationInput, calculate_request
from app.phase1_contracts import frozen_cualni_oracle


def test_canonical_blind_cualni_state_matches_frozen_transformation_checkpoint() -> None:
    o = frozen_cualni_oracle()
    request = CalculationRequest(
        project_id="frozen_blind_cualni_phase1",
        title="Frozen blind Cu-Al-Ni checkpoint",
        parent=PhaseInput(
            "phase_A", "Parent A", LatticeInput(o.parent_a, o.parent_a, o.parent_a, 90.0, 90.0, 90.0),
            point_group="m-3m", cell_representation="conventional cubic",
        ),
        product=PhaseInput(
            "phase_M", "Product M", LatticeInput(o.product_a, o.product_b, o.product_c, 90.0, o.product_beta_deg, 90.0),
            point_group="2/m", cell_representation="conventional unique-b",
        ),
        transformation=TransformationInput.from_rows(
            "A_to_M", "phase_A", "phase_M", o.correspondence_m_from_a, label="Parent -> Product",
        ),
    )
    response = calculate_request(request)
    summary = response.result["summary"]
    ct = response.result["ct_detail"]
    bj = response.result["ball_james_detail"]

    assert np.isclose(summary["lambda2"], o.lambda2, rtol=0.0, atol=7.0e-13)
    assert summary["variant_count"] == o.stretch_variants
    assert summary["operator_count"] == o.operator_classes
    assert summary["ct_exact_compatible"] is o.exact_ct_am
    assert bj["lambda2_exact"] is o.exact_bj_am
    assert np.isclose(ct["nearest_zero_residual"], o.ct_nearest_zero_residual, rtol=0.0, atol=7.0e-13)
    assert ct["exact_habit_planes_parent_covectors"] == []


@pytest.mark.slow
def test_canonical_blind_cualni_deep_oracle_is_locked() -> None:
    from cualni_cryst.project_io import project_from_dict
    from cualni_cryst.theory_equivalence import MatchFamily, build_equivalence_report
    from cualni_cryst.theory_unified import PTMCMode, PredictionKind, TheoryComparisonAdapter, TheoryKind

    o = frozen_cualni_oracle()
    request = CalculationRequest(
        project_id="frozen_blind_cualni_phase1_deep",
        title="Frozen blind Cu-Al-Ni deep checkpoint",
        parent=PhaseInput(
            "phase_A", "Parent A", LatticeInput(o.parent_a, o.parent_a, o.parent_a, 90.0, 90.0, 90.0),
            point_group="m-3m", cell_representation="conventional cubic",
        ),
        product=PhaseInput(
            "phase_M", "Product M", LatticeInput(o.product_a, o.product_b, o.product_c, 90.0, o.product_beta_deg, 90.0),
            point_group="2/m", cell_representation="conventional unique-b",
        ),
        transformation=TransformationInput.from_rows(
            "A_to_M", "phase_A", "phase_M", o.correspondence_m_from_a, label="Parent -> Product",
        ),
    )
    project = project_from_dict(request.to_project_payload()).project
    unified = TheoryComparisonAdapter(project, "A_to_M").compare(
        ptmc_mode=PTMCMode.ALL_TWINNING,
        include_ct_closing_gap=True,
        include_ct_supercompatibility=True,
    )

    ptmc_rows = [
        row for row in unified.rows
        if row.theory is TheoryKind.PTMC
        and row.prediction_kind is PredictionKind.PTMC_HABIT
        and row.exact is True
    ]
    assert len(ptmc_rows) == o.native_ptmc_branches

    exact_ct_am = [
        row for row in unified.rows
        if row.theory is TheoryKind.CAYRON_CT
        and row.prediction_kind is PredictionKind.CT_AM_HABIT
        and row.exact is True
    ]
    assert exact_ct_am == []
    assert not any(row.prediction_kind is PredictionKind.CT_SUPERCOMPATIBILITY for row in unified.rows)

    cofactor_full = sum(
        1
        for branch in unified.ball_james_report.martensite_twin_branches
        if branch.cofactor.all_satisfied
    )
    assert cofactor_full == o.full_cofactor_solutions

    equivalence = build_equivalence_report(
        project,
        "A_to_M",
        unified,
        include_independent_mm_audit=True,
    )
    audit = equivalence.mm_independent_audit
    assert audit is not None
    assert audit["success"] is True
    assert audit["relation_count"] == o.authoritative_mm_relations
    assert audit["ct_coverage_complete"] is True
    assert audit["classification_exact"] is True

    or_matches = equivalence.matches_for(
        family=MatchFamily.ORIENTATION,
        left=TheoryKind.CAYRON_CT,
        right=TheoryKind.PTMC,
    )
    eligible = [
        item.residuals.or_disorientation_deg
        for item in or_matches
        if item.residuals.or_disorientation_deg is not None
    ]
    assert eligible
    assert np.isclose(
        min(eligible),
        0.10868836866302703,
        rtol=0.0,
        atol=2.0e-8,
    )
