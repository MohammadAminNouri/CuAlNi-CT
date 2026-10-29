from __future__ import annotations

"""2022 exact CT generator continuity and 2026 Eq.42 inverse-design guards.

No CI acceptance may relabel a nearest twin from a different symmetry
operation as the selected physical twin. Source constraints are unchanged.
"""
from dataclasses import replace
from pathlib import Path
import ast
import numpy as np
import pytest

from cualni_cryst.project_state import james_hane_6m_reference_project
from twin_app.metric_design_targets import state_at_parameter, metric_stretches
from twin_app.same_twin_continuity_2026 import (
    enumerate_twin_anchors, reconstruct_same_twin,
    fixed_twin_am_habits,
)
from twin_app.same_twin_inverse_design_2026 import (
    optimize_one_tracked_twin, verify_candidate_same_twin_cofactor,
)

TRANSFORMATION="do3_to_6m_reference"


@pytest.fixture(scope="module")
def original():
    return james_hane_6m_reference_project()


@pytest.fixture(scope="module")
def anchors(original):
    return enumerate_twin_anchors(original,TRANSFORMATION)


def test_all_2022_generic_twin_constructions_have_unique_exact_generator(anchors):
    assert anchors, "The known reference project must contain CT twins"
    assert len({r.branch_id for r in anchors})==len(anchors)
    assert len({r.generator_key for r in anchors})==len(anchors)
    for r in anchors:
        assert r.generator_key.startswith(("I:","II:"))
        assert r.initial_shear>0
        assert r.generator.shape==(3,3)
        assert np.array_equal(
            np.asarray(r.generator*r.generator,dtype=float),np.eye(3)
        )


def test_same_generator_is_not_reselected_by_twin_array_index(original,anchors):
    a=anchors[0]
    unchanged,trace=reconstruct_same_twin(original,TRANSFORMATION,a)
    assert trace["branch_id"]==a.branch_id
    assert trace["plane_step_deg"]<1.e-5
    assert trace["direction_step_deg"]<1.e-5
    assert trace["shear_ratio_step"]==pytest.approx(1.,abs=1e-8)
    modified=state_at_parameter(original,TRANSFORMATION,"uniform_scale",1.002)
    twin,record=reconstruct_same_twin(modified,TRANSFORMATION,a,
                                      previous_twin=unchanged)
    assert record["generator_key"]==a.generator_key
    assert record["status"]=="SAME_EXACT_GENERATOR_AND_CONTINUOUS_PHYSICAL_GEOMETRY"
    assert np.isfinite(twin.shear)


def test_cross_project_parent_metric_or_correspondence_change_is_not_the_same_twin(
    original,anchors
):
    a=anchors[0]
    wrong=replace(a,parent_metric=a.parent_metric*1.001)
    with pytest.raises(ValueError,match="Parent metric"):
        reconstruct_same_twin(original,TRANSFORMATION,wrong)
    wrong2=replace(a,product_group_signature=("false_product_group",))
    with pytest.raises(ValueError,match="Parent metric"):
        reconstruct_same_twin(original,TRANSFORMATION,wrong2)
    with pytest.raises(ValueError):
        reconstruct_same_twin(original,TRANSFORMATION,
                              replace(a,construction_route="III"))


def test_lambda2_only_is_never_claimed_as_shear_shear(original,anchors):
    corrected=state_at_parameter(
        original,TRANSFORMATION,"uniform_scale",
        1/metric_stretches(original,TRANSFORMATION)[1],
    )
    report=fixed_twin_am_habits(corrected,TRANSFORMATION,anchors[0])
    assert report["status"]=="EXACT_CT_AM_HABITS_AND_FIXED_MM_TWIN"
    assert report["ball_james_cofactor_on_this_twin"]=="NOT_YET_PHYSICALLY_ASSIGNED"
    assert report["same_generator_equivalent_CC2_metric"]["route"] in ("I","II")
    assert report["same_generator_equivalent_CC2_metric"]["CC3_verified"] is False
    assert report["habits"]
    for branch in report["habits"]:
        assert branch["epsilon"]>=0
        assert branch["epsilon"]==pytest.approx(
            branch["eq42"]["epsilon_normalized_by_twin_shear"],
            abs=5e-9,
        )


def test_inverse_requires_existing_generator_and_real_constraints(original):
    with pytest.raises(ValueError,match="does not exist"):
        optimize_one_tracked_twin(
            original,TRANSFORMATION,branch_id="invented branch",max_evaluations=12,
        )
    with pytest.raises(ValueError,match="Length bound"):
        optimize_one_tracked_twin(original,TRANSFORMATION,
                                  branch_id="anything",relative_length_bound=0.)


def test_inverse_never_joins_another_twin_and_can_fail_closed(original,anchors):
    report=optimize_one_tracked_twin(
        original,TRANSFORMATION,branch_id=anchors[0].branch_id,
        habit_index=0,relative_length_bound=.001,
        max_uniform_scale_offset=.001,
        max_evaluations=12,continuity_steps=5,
    )
    assert report["branch_id"]==anchors[0].branch_id
    assert report["search_exhaustive"] is False
    if report["candidate"] is not None:
        c=report["candidate"]
        assert c["data_status"]=="HYPOTHETICAL_TEST"
        assert abs(c["lambda2_residual"])<2e-8
        assert not report["cofactor_same_pair_certified"]
        assert c["same_parent_symmetry_generator"]==anchors[0].generator_key
    else:
        assert report["status"].startswith(("NO_EXACT_","NO_UNAMBIGUOUS_",
                                            "NO_VALID_","NO_CONTINUOUS_"))


def test_same_pair_cofactor_requires_explicit_final_lattice_candidate(original):
    with pytest.raises(ValueError,match="No resolved tracked-twin candidate"):
        verify_candidate_same_twin_cofactor(
            original,TRANSFORMATION,{"candidate":None},
            length_bound=.1,angle_bound_deg=3.,uniform_scale_bound=.15,
        )


def test_ui_is_additive_and_contains_original_and_new_controls():
    root=Path(__file__).resolve().parents[2]
    ui=(root/"twin_app"/"metric_design_ui.py").read_text()
    ast.parse(ui)
    for text in (
        "_render_joint_compatibility_design(",
        "_render_metric_failure_diagnosis(",
        "Design toward full A/M/M shear/shear compatibility",
        "Inverse supercompatibility · one continuously tracked physical twin",
        "Verify THIS fixed CT generator and habit against BJ CC1–CC3",
        "factor_by_factor_counterfactuals",
    ):
        assert text in ui
    backend=(root/"twin_app"/"same_twin_inverse_design_2026.py").read_text()
    ast.parse(backend)
    assert 'cofactor_same_pair_certified":False' in backend
    assert 'search_exhaustive":False' in backend


def test_known_exact_lambda2_input_retains_one_fixed_twin_as_a_valid_candidate(
    original,anchors
):
    """Positive control: the unchanged reference must not be dropped by Powell."""
    known=state_at_parameter(
        original,TRANSFORMATION,"uniform_scale",
        1./metric_stretches(original,TRANSFORMATION)[1],
    )
    reference=next(x for x in anchors if x.initial_shear>.02)
    # Anchor ids are exact parent symmetry keys and remain valid at the
    # analytically corrected source state, independent of metric changes.
    result=optimize_one_tracked_twin(
        known,TRANSFORMATION,branch_id=reference.branch_id,
        habit_index=0,relative_length_bound=.035,
        angle_bound_deg=1.,max_uniform_scale_offset=.05,
        max_evaluations=12,continuity_steps=5,
    )
    assert result["candidate"] is not None, result
    candidate=result["candidate"]
    assert result["branch_id"]==reference.branch_id
    assert result["states_evaluated"]<=12
    assert abs(candidate["lambda2_residual"])<2e-8
    assert candidate["candidate_epsilon"]>=0.
    assert candidate["data_status"]=="HYPOTHETICAL_TEST"
    assert len(candidate["continuation_path"])==5
    assert candidate["factor_by_factor_counterfactuals"]
    assert candidate["candidate_same_generator_CC2_equivalent"]["absolute_squared_norm_gap"]>=0.
    assert candidate["candidate_same_generator_CC2_equivalent"]["CC3_verified"] is False
    assert candidate["same_parent_symmetry_generator"]==reference.generator_key
    assert not result["cofactor_same_pair_certified"]
    for row in candidate["continuation_path"]:
        assert abs(row["lambda2_minus_one"])<2e-8
