from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

import cualni_cryst.theory_unified as unified_module
from cualni_cryst.project_state import james_hane_6m_reference_project
from cualni_cryst.theory_unified import (
    ExperimentalObservation,
    PredictionKind,
    PTMCMode,
    TheoryComparisonAdapter,
    TheoryKind,
)


def _adapter() -> TheoryComparisonAdapter:
    return TheoryComparisonAdapter(
        james_hane_6m_reference_project(),
        "do3_to_6m_reference",
    )


def test_ct_exact_symmetry_bridge_recovers_project_float_groups():
    project = james_hane_6m_reference_project()
    parent = project.phase("austenite_do3")
    product = project.phase("martensite_long_period")
    tolerance = max(
        project.numerical_policy.representation,
        project.numerical_policy.algebraic,
    )

    parent_exact = unified_module.exact_symmetry_group_for_ct(
        list(parent.symmetry_matrices()),
        parent.lattice.metric(),
        tolerance=tolerance,
        label="parent",
    )
    product_exact = unified_module.exact_symmetry_group_for_ct(
        list(product.symmetry_matrices()),
        product.lattice.metric(),
        tolerance=tolerance,
        label="product",
    )

    assert len(parent_exact) == 48
    assert len(product_exact) == 4
    assert any(
        np.array_equal(np.asarray(matrix, dtype=float), np.eye(3))
        for matrix in parent_exact
    )
    assert any(
        np.array_equal(np.asarray(matrix, dtype=float), np.eye(3))
        for matrix in product_exact
    )


def test_ct_exact_symmetry_bridge_refuses_non_group_float_input():
    bad = [
        np.eye(3),
        np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    ]
    with pytest.raises(ValueError, match="exact finite crystallographic group"):
        unified_module.exact_symmetry_group_for_ct(
            bad,
            np.eye(3),
            tolerance=1.0e-10,
            label="bad",
        )


def test_unified_layer_is_orchestrator_not_a_fourth_theory():
    tree = ast.parse(inspect.getsource(unified_module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    assert "ball_james_adapter" in imported
    assert "ptmc_adapter" in imported
    assert "ct" in imported
    source = inspect.getsource(unified_module)
    assert "cubic_proper_rotations" not in source
    assert "winner score" in source


def test_same_project_inputs_generate_ct_ball_james_and_ptmc_rows():
    report = _adapter().compare(
        include_ct_closing_gap=False,
        include_ct_supercompatibility=False,
        ball_james_fraction_samples=11,
        ptmc_mode=PTMCMode.ALL_TWINNING,
    )

    assert report.transformation_id == "do3_to_6m_reference"
    assert report.parent_phase_id == "austenite_do3"
    assert report.product_phase_id == "martensite_long_period"

    theories = {row.theory for row in report.rows}
    assert TheoryKind.CAYRON_CT in theories
    assert TheoryKind.BALL_JAMES in theories
    assert TheoryKind.PTMC in theories

    assert report.ct_report.analysis.nearest_zero_residual >= 0.0
    assert report.ball_james_report.audit.pulled_back_metric_residual < 1.0e-10
    assert report.ball_james_report.audit.metric_native_spectrum_residual < 1.0e-10
    assert report.ball_james_report.audit.maximum_martensite_rank_one_residual < 1.0e-8
    assert report.ptmc_report is not None
    assert np.isfinite(report.ptmc_report.audit.maximum_residual)
    ptmc_diagnostics = [
        row
        for row in report.rows
        if row.prediction_kind is PredictionKind.PTMC_PARAMETER_DIAGNOSTIC
    ]
    assert len(ptmc_diagnostics) == len(report.ptmc_report.parameter_roots)

    row_ids = [row.row_id for row in report.rows]
    assert len(row_ids) == len(set(row_ids))


def test_all_branches_are_preserved_and_missing_values_are_explicit_na():
    report = _adapter().compare(
        include_ct_closing_gap=False,
        include_ct_supercompatibility=False,
        ball_james_fraction_samples=7,
        ptmc_mode=PTMCMode.NONE,
    )

    bj_am = [
        row
        for row in report.rows
        if row.prediction_kind is PredictionKind.BALL_JAMES_AM
    ]
    bj_mm = [
        row
        for row in report.rows
        if row.prediction_kind is PredictionKind.BALL_JAMES_MM
    ]
    assert len(bj_am) == len(report.ball_james_report.austenite_martensite_branches)
    assert len(bj_mm) == len(report.ball_james_report.martensite_twin_branches)

    table = report.table_rows()
    assert table
    assert all("residuals" in row for row in table)
    assert any(row["OR_parent_from_product"] == "N/A" for row in table)
    assert not report.rows_for(TheoryKind.PTMC)
    assert any("PTMC was explicitly disabled" in item for item in report.warnings)


def test_experiment_is_pass_through_not_used_to_fit_predictors():
    experiment = ExperimentalObservation(
        observation_id="ebsd_001",
        label="synthetic experiment row",
        or_parent_from_product=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        shear_magnitude=0.123,
        uncertainty={"orientation_deg": 0.2},
    )

    report = _adapter().compare(
        include_ct_closing_gap=False,
        include_ct_supercompatibility=False,
        ptmc_mode="none",
        ball_james_fraction_samples=5,
        experiments=(experiment,),
    )

    rows = report.rows_for(TheoryKind.EXPERIMENT)
    assert len(rows) == 1
    row = rows[0]
    assert row.row_id == "experiment_ebsd_001"
    assert np.isclose(row.shear_magnitude, 0.123)
    assert row.residuals["uncertainty_orientation_deg"] == pytest.approx(0.2)


def test_ct_closing_gap_preserves_every_candidate_without_natural_or():
    report = _adapter().compare(
        include_ct_closing_gap=True,
        include_ct_supercompatibility=False,
        ptmc_mode="none",
        ball_james_fraction_samples=5,
    )

    rows = [
        row
        for row in report.rows
        if row.prediction_kind is PredictionKind.CT_CLOSING_GAP_OR
    ]
    assert rows
    assert all(row.or_parent_from_product is not None for row in rows)
    assert all(row.metadata["selected_within_twin"] is False for row in rows)
    assert any("all retained" in item for item in report.warnings)


@pytest.mark.parametrize(
    ("mode", "ptmc_request"),
    [
        ("slip", None),
        ("twin_plane", None),
        ("twin_pair", None),
    ],
)
def test_theory_specific_ptmc_requests_fail_loudly_when_missing(mode, ptmc_request):
    with pytest.raises(TypeError):
        _adapter().compare(
            include_ct_closing_gap=False,
            include_ct_supercompatibility=False,
            ptmc_mode=mode,
            ptmc_request=ptmc_request,
            ball_james_fraction_samples=5,
        )


def test_invalid_common_options_fail_loudly():
    adapter = _adapter()

    with pytest.raises(ValueError):
        adapter.compare(ball_james_fraction_samples=1)

    with pytest.raises(ValueError):
        adapter.compare(ptmc_dilatational_factor=0.0)

    with pytest.raises(ValueError):
        adapter.compare(ptmc_mode="not-a-mode")
