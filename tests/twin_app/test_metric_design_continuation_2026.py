"""Finite two-axis continuation: exact lambda2 roots and same-pair CT/CC."""
from types import SimpleNamespace as Record

import numpy as np
import pytest

from cualni_cryst.project_state import james_hane_6m_reference_project
from cualni_cryst.supercompatibility_atlas import independent_product_lattice_parameters
from twin_app.metric_design_targets import metric_stretches,state_at_parameter
from twin_app.metric_design_continuation import (
    _matched_branch_summary,continue_supercompatibility_design,
)

TRANSFORMATION="do3_to_6m_reference"


def _ready_project():
    initial=james_hane_6m_reference_project()
    scale=1./metric_stretches(initial,TRANSFORMATION)[1]
    return state_at_parameter(initial,TRANSFORMATION,"uniform_scale",scale)


def _fake_evaluator(*, complete=True, exact=True):
    def inner(project, transformation_id, *, state_id, metadata):
        index=int(state_id.rsplit("-",1)[-1])
        T=project.transformation(transformation_id)
        assert T.correspondence.C_M_from_A == (
            _ready_project().transformation(transformation_id).correspondence.C_M_from_A
        )
        pair=Record(
            ct_operator_index=2,ct_twin_index=1,ct_best_habit_index=0,
            ct_supercompatibility_residual=.003*abs(index-2),
            cofactor_cc1_abs=1.e-12,cofactor_cc2_abs=3.e-7,
            cofactor_cc3_margin=.2,cofactor_compatible=True,
            ct_supercompatible=index==2,
            match_completeness="full" if complete else "partial",
            missing_required_components=() if complete else ("twin_plane",),
        )
        return Record(
            pair_results=(pair,) if exact else (),
            ct_am_exact_compatible=exact,
            ct_supercompatible=(index==2) if exact else None,
            cofactor_compatible=True,
            matched_relation_count=1 if exact else 0,
            ct_best_supercompatibility_residual=pair.ct_supercompatibility_residual,
        )
    return inner


def test_continuation_uses_real_lambda2_root_and_specific_matched_branch():
    project=_ready_project()
    secondary=independent_product_lattice_parameters(project,TRANSFORMATION)[0]
    result=continue_supercompatibility_design(
        project,TRANSFORMATION,
        primary_parameter="uniform_scale",
        secondary_parameter=secondary,
        secondary_relative_bound=.01,steps=5,
        evaluator=_fake_evaluator(),
    )
    assert len(result["samples"])==5
    assert result["best_fully_matched_sample_id"]==2
    assert result["proposed_direction"].startswith("UNCHANGED")
    assert all(row["lambda2_residual"]<2e-8 for row in result["samples"])
    assert all(row["paired_CT_BJ_branch"]["cofactor_satisfied_for_this_pair"]
               for row in result["samples"])
    assert result["samples"][2]["paired_CT_BJ_branch"]["both_satisfied_for_this_matched_pair"]
    assert "not an exhaustive global optimum" in result["scientific_limit"]


def test_partial_or_missing_am_is_not_silently_promoted_to_supercompatibility():
    project=_ready_project()
    secondary=independent_product_lattice_parameters(project,TRANSFORMATION)[0]
    for evaluator in (_fake_evaluator(complete=False),_fake_evaluator(exact=False)):
        result=continue_supercompatibility_design(
            project,TRANSFORMATION,
            primary_parameter="uniform_scale",
            secondary_parameter=secondary,
            secondary_relative_bound=.005,steps=3,
            evaluator=evaluator,
        )
        assert result["best_fully_matched_sample_id"] is None
        assert result["proposed_direction"].startswith("NO_BRANCH_RESOLVED")


def test_independent_axis_and_finite_scan_validation():
    project=_ready_project()
    axis=independent_product_lattice_parameters(project,TRANSFORMATION)[0]
    with pytest.raises(ValueError,match="different"):
        continue_supercompatibility_design(
            project,TRANSFORMATION,
            primary_parameter=axis,secondary_parameter=axis,
            evaluator=_fake_evaluator(),
        )
    with pytest.raises(ValueError,match="3, 5 or 7"):
        continue_supercompatibility_design(
            project,TRANSFORMATION,
            primary_parameter="uniform_scale",secondary_parameter=axis,
            steps=99,evaluator=_fake_evaluator(),
        )
    with pytest.raises(ValueError,match="allowed"):
        continue_supercompatibility_design(
            project,TRANSFORMATION,
            primary_parameter="uniform_scale",secondary_parameter="gamma_offset_deg",
            evaluator=_fake_evaluator(),
        )


def test_pair_summaries_do_not_mix_cofactor_residuals_from_unrelated_branches():
    a=Record(
        ct_operator_index=1,ct_twin_index=0,ct_best_habit_index=1,
        ct_supercompatibility_residual=.0001,cofactor_cc1_abs=3.,
        cofactor_cc2_abs=4.,cofactor_cc3_margin=-9.,
        cofactor_compatible=False,ct_supercompatible=True,
        match_completeness="full",missing_required_components=(),
    )
    b=Record(
        ct_operator_index=6,ct_twin_index=8,ct_best_habit_index=7,
        ct_supercompatibility_residual=.05,cofactor_cc1_abs=1.e-10,
        cofactor_cc2_abs=1.e-11,cofactor_cc3_margin=3.,
        cofactor_compatible=True,ct_supercompatible=False,
        match_completeness="full",missing_required_components=(),
    )
    row=_matched_branch_summary(Record(pair_results=(a,b)))
    assert row["operator_index"]==1
    assert row["CC1_abs_lambda2_minus_one"]==3.
    assert not row["both_satisfied_for_this_matched_pair"]


def test_fully_matched_branch_not_masked_by_smaller_partial_residual():
    # The lowest epsilon alone is not valid branch evidence when matching
    # is incomplete. Select the best FULL match for the design comparison.
    partial=Record(
        ct_operator_index=4,ct_twin_index=2,ct_best_habit_index=0,
        ct_supercompatibility_residual=1e-10,
        cofactor_cc1_abs=0.,cofactor_cc2_abs=0.,cofactor_cc3_margin=1.,
        cofactor_compatible=True,ct_supercompatible=True,
        match_completeness="partial",
        missing_required_components=("twin_plane",),
    )
    complete=Record(
        ct_operator_index=5,ct_twin_index=3,ct_best_habit_index=1,
        ct_supercompatibility_residual=0.015,
        cofactor_cc1_abs=1e-8,cofactor_cc2_abs=4e-5,cofactor_cc3_margin=.1,
        cofactor_compatible=False,ct_supercompatible=False,
        match_completeness="full",
        missing_required_components=(),
    )
    result=_matched_branch_summary(Record(pair_results=(partial,complete)))
    assert result["operator_index"]==5
    assert result["ct_shear_shear_epsilon"]==pytest.approx(.015)
    assert result["match_completeness"]=="full"
    assert not result["both_satisfied_for_this_matched_pair"]
