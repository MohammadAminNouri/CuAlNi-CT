from __future__ import annotations

"""Compatibility causality tests: exact metric counterfactuals, not alloy recipes."""
from pathlib import Path
import ast

import pytest

from cualni_cryst.project_state import james_hane_6m_reference_project
from twin_app.metric_design_targets import metric_stretches,state_at_parameter
from twin_app.metric_failure_diagnostics import (
    metric_failure_analysis, supercompatibility_failure_analysis,
    summarize_sampled_changes, _cell,
)

T="do3_to_6m_reference"


def test_quantified_metric_correction_recomputes_lambda2_and_cell_changes():
    original=james_hane_6m_reference_project()
    scale=1.02/metric_stretches(original,T)[1]
    shifted=state_at_parameter(original,T,"uniform_scale",scale)
    analysis=metric_failure_analysis(
        shifted,T,length_relative_bound=.05,angle_bound_deg=2.,
    )
    assert analysis["status"]=="AM_METRIC_CONDITION_NOT_SATISFIED"
    assert analysis["lambda2_minus_one"]==pytest.approx(.02,abs=1e-10)
    assert analysis["candidate_count"]>0
    uniform=next(x for x in analysis["candidate_changes"]
                 if x["parameter"]=="uniform_scale")
    assert uniform["target_parameter_value"]==pytest.approx(1/1.02,abs=1e-8)
    assert uniform["change_unit"]=="%"
    assert uniform["abs_change"]==pytest.approx(100*(1-1/1.02),abs=1e-6)
    assert uniform["lambda2_residual"] < 2e-8
    assert uniform["physical_change"].count("entered length units")==3
    assert uniform["full_supercompatibility_verified"] is False
    assert analysis["twin_specific_supercompatibility"]=="NOT_ASSESSED"


def test_zero_mid_stretch_defect_does_not_trigger_false_supercompatibility():
    p=james_hane_6m_reference_project()
    p=state_at_parameter(p,T,"uniform_scale",1/metric_stretches(p,T)[1])
    analysis=metric_failure_analysis(p,T,length_relative_bound=.01)
    assert analysis["status"]=="AM_METRIC_CONDITION_SATISFIED"
    assert analysis["twin_specific_supercompatibility"]=="NOT_ASSESSED"


def test_missing_CT_AM_seed_is_not_a_failed_shear_shear_equation():
    raw={"ct_am_exact_compatible":False,"ct_supercompatible":None,
         "cofactor_compatible":False,"pair_results":[],
         "ct_best_supercompatibility_residual":None}
    a=supercompatibility_failure_analysis(raw)
    assert a["status"]=="NO_SAME_PAIR_SUPERCOMPATIBILITY_CERTIFICATE"
    assert a["conditions"][1]["status"]=="NOT_EVALUABLE"
    assert a["conditions"][1]["observed"] is None


def test_partial_or_geometrically_disagreeing_pair_cannot_certificate():
    pair={"ct_row_id":"ct1","ball_james_row_id":"bj1",
          "match_completeness":"full","missing_required_components":[],
          "match_geometry_within_tolerance":False,
          "ct_supercompatibility_residual":0.,
          "cofactor_cc1_abs":0.,"cofactor_cc2_abs":0.,"cofactor_cc3_margin":1.,
          "ct_supercompatible":True,"cofactor_compatible":True}
    report={"ct_am_exact_compatible":True,"ct_supercompatible":True,
            "ct_best_supercompatibility_residual":0.,
            "pair_results":[pair]}
    a=supercompatibility_failure_analysis(report)
    assert a["physically_comparable_pairs"]==0
    assert a["status"]=="NO_SAME_PAIR_SUPERCOMPATIBILITY_CERTIFICATE"
    assert a["conditions"][-1]["status"]=="NOT_COMPARABLE"
    pair["match_geometry_within_tolerance"]=True
    b=supercompatibility_failure_analysis(report)
    assert b["status"]=="SAME_PAIR_NUMERICALLY_SATISFIED"
    report["ct_am_exact_compatible"]=False
    assert supercompatibility_failure_analysis(report)["status"]==(
        "NO_SAME_PAIR_SUPERCOMPATIBILITY_CERTIFICATE"
    )


def test_continuation_reports_actual_cell_delta_and_missing_constraints():
    baseline={"a":4.,"b":5.,"c":12.,"beta_deg":96.}
    state={"samples":[{
        "sample_id":0,"secondary_parameter":"beta_offset_deg",
        "secondary_delta":1.,"primary_value":0.982,
        "a":4.,"b":5.02,"c":12.,"beta_deg":97.,
        "lambda2_residual":1e-10,"exact_CT_AM":True,
        "matching_is_complete":False,
        "paired_CT_BJ_branch":{
            "ct_shear_shear_epsilon":.021,
            "CC1_abs_lambda2_minus_one":1e-10,
            "CC2_abs":.002,"CC3_margin":-.3,
            "ct_satisfied_for_this_pair":False,
            "cofactor_satisfied_for_this_pair":False,
            "both_satisfied_for_this_matched_pair":False,
        }
    }],
    "best_fully_matched_sample_id":None,
    "proposed_direction":"NO_BRANCH_RESOLVED"}
    result=summarize_sampled_changes(state,baseline)
    row=result["samples"][0]
    assert row["physical_cell_deltas"]["beta_deg"]==pytest.approx(1.)
    assert row["physical_cell_deltas"]["b"]==pytest.approx(.02)
    assert len(row["unmet_or_unevaluable_gates"])==3
    assert result["cross_state_twin_identity_verified"] is False


def test_failure_diagnosis_ui_remains_additive_to_all_existing_advisors():
    root=Path(__file__).resolve().parents[2]
    ui=(root/"twin_app"/"metric_design_ui.py").read_text()
    ast.parse(ui)
    for token in (
        "Find coupled λ₂=1 candidate",
        "Design toward full A/M/M shear/shear compatibility",
        "Find λ₂=1 lattice targets",
        "Diagnose A/M failure and calculate parameter corrections",
        "Explain current full CT/Ball–James compatibility",
        "Identify remaining CT shear/shear and CC1–CC3 failures at this cell",
    ):
        assert token in ui
