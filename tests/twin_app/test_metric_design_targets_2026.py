"""Cayron 2026 metric design: exact physical control, no false SC certificate."""
import ast
from pathlib import Path

import numpy as np
import pytest

from cualni_cryst.project_state import james_hane_6m_reference_project
from cualni_cryst.supercompatibility_atlas import (
    independent_product_lattice_parameters, uniform_product_scale_sweep,
)
from twin_app.metric_design_targets import (
    metric_stretches, scan_lambda2_targets, state_at_parameter,
)


TRANSFORMATION="do3_to_6m_reference"


def test_metric_lambda2_uniform_scale_is_exact_linear_relation():
    project=james_hane_6m_reference_project()
    stretch=metric_stretches(project,TRANSFORMATION)
    assert 0<stretch[0]<=stretch[1]<=stretch[2]
    for scale in (0.97,1.02,1.05):
        scaled=uniform_product_scale_sweep(
            project,TRANSFORMATION,(scale,),
        )[0][1]
        np.testing.assert_allclose(
            metric_stretches(scaled,TRANSFORMATION),
            np.array(stretch)*scale,rtol=1e-9,atol=1e-9,
        )


def test_metric_lambda2_target_is_actual_root_not_low_score():
    project=james_hane_6m_reference_project()
    baseline=metric_stretches(project,TRANSFORMATION)[1]
    # Construct a provable lambda2=1 fixture, preserving the parent,
    # correspondence and all point-group cell metric constraints.
    exact_project=state_at_parameter(
        project,TRANSFORMATION,"uniform_scale",1./baseline,
    )
    target=scan_lambda2_targets(
        exact_project,TRANSFORMATION,"uniform_scale",
        length_relative_bound=.04,samples=21,
    )
    assert target.candidates
    assert abs(target.candidates[0]["parameter_value"]-1)<1e-8
    assert target.candidates[0]["abs_lambda2_minus_one"]<2e-8
    assert target.derivative==pytest.approx(1,abs=2e-4)


def test_uniform_scale_root_direction_matches_metric_derivative():
    project=james_hane_6m_reference_project()
    lam=metric_stretches(project,TRANSFORMATION)[1]
    # A 2% shift guarantees a root inside the 5% search bound.
    shifted=state_at_parameter(
        project,TRANSFORMATION,"uniform_scale",1.02/lam,
    )
    target=scan_lambda2_targets(
        shifted,TRANSFORMATION,"uniform_scale",
        length_relative_bound=.05,samples=21,
    )
    assert target.candidates
    root=target.candidates[0]
    assert root["parameter_value"]==pytest.approx(1./1.02,abs=1e-8)
    assert root["change"]<0
    assert root["abs_lambda2_minus_one"]<2e-8


def test_symmetry_independent_parameters_never_change_parent_or_C():
    project=james_hane_6m_reference_project()
    allowed=independent_product_lattice_parameters(project,TRANSFORMATION)
    assert allowed
    transformation=project.transformation(TRANSFORMATION)
    original_C=transformation.correspondence.C_M_from_A
    original_parent=project.phase(transformation.parent_phase_id)
    for parameter in allowed:
        new=state_at_parameter(
            project,TRANSFORMATION,parameter,
            .001 if parameter.endswith("offset_deg") else 1.001,
        )
        assert new.phase(transformation.parent_phase_id).to_dict() == original_parent.to_dict()
        assert new.transformation(TRANSFORMATION).correspondence.C_M_from_A==original_C
    with pytest.raises(ValueError,match="cannot vary independently"):
        state_at_parameter(project,TRANSFORMATION,"unknown_offset_deg",2.)
    with pytest.raises(ValueError):
        state_at_parameter(project,TRANSFORMATION,"a_scale",0.)


def test_scan_limits_and_no_false_supercompat_status():
    p=james_hane_6m_reference_project()
    with pytest.raises(ValueError):
        scan_lambda2_targets(p,TRANSFORMATION,"uniform_scale",
                             length_relative_bound=0)
    with pytest.raises(ValueError):
        scan_lambda2_targets(p,TRANSFORMATION,"unknown")
    res=scan_lambda2_targets(
        p,TRANSFORMATION,"uniform_scale",
        length_relative_bound=.002,samples=11,
    )
    assert len(res.scan)==11
    assert "SUPERCOMPATIBLE" not in res.status
    assert all("lambda2_minus_one" in item for item in res.scan)


def test_ui_keeps_metric_design_separate_from_primary_theory_solver():
    root=Path(__file__).resolve().parents[2]
    ui=(root/"twin_app"/"metric_design_ui.py").read_text()
    compare=(root/"twin_app"/"theory_comparison_ui.py").read_text()
    ast.parse(ui)
    assert 'render_lattice_design_advisor(project_payload, transformation_id, signature)' in compare
    assert 'Find λ₂=1 lattice targets' in ui
    assert 'Verify CT shear/shear and BJ cofactor conditions' in ui
    assert 'Run CT / BJ / PTMC full theory comparison at this target' in ui
    assert 'source_doi' in ui
    assert 'state_at_parameter(' in ui
    assert "st.session_state['parent_a']" not in ui
    assert "project.to_dict()" not in ui
