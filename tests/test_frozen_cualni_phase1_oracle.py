from __future__ import annotations

from dataclasses import asdict, is_dataclass

import pytest

from app.application import CalculationRequest, LatticeInput, PhaseInput, TransformationInput, calculate_request
from app.phase1_contracts import frozen_cualni_oracle
from cualni_cryst.project_io import project_from_dict
from cualni_cryst.theory_equivalence import MatchCompleteness, MatchFamily, build_equivalence_report
from cualni_cryst.theory_unified import PTMCMode, PredictionKind, TheoryComparisonAdapter, TheoryKind


def _request() -> CalculationRequest:
    o = frozen_cualni_oracle()
    return CalculationRequest(
        project_id="frozen_blind_cualni_phase1",
        title="Frozen blind Cu-Al-Ni Phase-1 regression",
        parent=PhaseInput(
            "phase_A",
            "Parent DO3",
            LatticeInput(o.parent_a, o.parent_a, o.parent_a, 90.0, 90.0, 90.0),
            point_group="m-3m",
            cell_representation="conventional cubic",
        ),
        product=PhaseInput(
            "phase_M",
            "Product 6M",
            LatticeInput(o.product_a, o.product_b, o.product_c, 90.0, o.product_beta_deg, 90.0),
            point_group="2/m",
            cell_representation="conventional unique-b",
        ),
        transformation=TransformationInput.from_rows(
            "A_to_M",
            "phase_A",
            "phase_M",
            o.correspondence_m_from_a,
            label="DO3 -> 6M",
        ),
    )


def _walk_dicts(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_dicts(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_dicts(item)


@pytest.mark.slow
def test_frozen_blind_cualni_full_scientific_oracle() -> None:
    """Lock the canonical blind state through the real application + theory backends.

    This is intentionally a heavy scientific regression.  It must not be
    replaced by a constants-only test or weakened merely to make CI green.
    """

    o = frozen_cualni_oracle()
    response = calculate_request(_request())
    summary = response.result["summary"]

    assert float(summary["lambda2"]) == pytest.approx(o.lambda2, abs=2e-12)
    assert int(summary["variant_count"]) == o.stretch_variants
    assert int(summary["operator_count"]) == o.operator_classes
    assert summary["ct_exact_compatible"] is o.exact_ct_am
    assert summary["ball_james_lambda2_exact"] is o.exact_bj_am
    assert float(response.result["ct_detail"]["nearest_zero_residual"]) == pytest.approx(
        o.ct_nearest_zero_residual, abs=2e-12
    )

    project = project_from_dict(response.project_payload, source="frozen-phase1-regression").project
    unified = TheoryComparisonAdapter(project, "A_to_M").compare(
        ptmc_mode=PTMCMode.ALL_TWINNING,
        include_ct_closing_gap=True,
        include_ct_supercompatibility=True,
    )

    exact_ptmc = [
        row
        for row in unified.rows
        if row.theory is TheoryKind.PTMC
        and row.prediction_kind is PredictionKind.PTMC_HABIT
        and row.exact is True
    ]
    assert len(exact_ptmc) == o.native_ptmc_branches

    exact_ct_habits = [
        row for row in unified.rows
        if row.theory is TheoryKind.CAYRON_CT
        and row.prediction_kind is PredictionKind.CT_AM_HABIT
        and row.exact is True
    ]
    supercompat = [
        row for row in unified.rows
        if row.prediction_kind is PredictionKind.CT_SUPERCOMPATIBILITY
    ]
    assert exact_ct_habits == []
    assert supercompat == []  # exact condition is gated => UI status must be NOT EVALUABLE

    equivalence = build_equivalence_report(
        project,
        "A_to_M",
        unified,
        include_independent_mm_audit=True,
    )
    generic_ct_bj_mm = equivalence.matches_for(
        family=MatchFamily.MM_TWIN,
        left=TheoryKind.CAYRON_CT,
        right=TheoryKind.BALL_JAMES,
    )
    assert len(generic_ct_bj_mm) == o.generic_ct_bj_mm_assignments
    assert all(match.completeness is MatchCompleteness.PARTIAL for match in generic_ct_bj_mm)

    audit = equivalence.mm_independent_audit
    assert audit is not None
    assert audit["success"] is True
    assert int(audit["relation_count"]) == o.authoritative_mm_relations
    assert audit["ct_coverage_complete"] is True
    assert audit["classification_exact"] is True

    assigned_or = [
        match.residuals.or_disorientation_deg
        for match in equivalence.matches_for(
            family=MatchFamily.ORIENTATION,
            left=TheoryKind.CAYRON_CT,
            right=TheoryKind.PTMC,
        )
        if match.residuals.or_disorientation_deg is not None
    ]
    assert assigned_or
    assert min(assigned_or) == pytest.approx(o.closest_assigned_ct_ptmc_or_deg, abs=5e-6)

    cofactor_nodes = [
        node
        for node in _walk_dicts(unified.ball_james_report)
        if {"cc1_satisfied", "cc2_satisfied", "cc3_satisfied"}.issubset(node)
    ]
    assert cofactor_nodes
    assert sum(
        bool(node["cc1_satisfied"] and node["cc2_satisfied"] and node["cc3_satisfied"])
        for node in cofactor_nodes
    ) == o.full_cofactor_solutions
