"""Scientific guards for bounded joint A/M metric suggestions, Cayron 2026."""
import ast
from pathlib import Path

import numpy as np
import pytest

from cualni_cryst.project_state import james_hane_6m_reference_project
from cualni_cryst.supercompatibility_atlas import uniform_product_scale_sweep
from twin_app.metric_design_targets import metric_stretches
from twin_app.joint_metric_design_2026 import (
    find_joint_lambda2_target, state_at_joint_target,
)

TRANSFORMATION = "do3_to_6m_reference"


def _known_close_input(multiplier=1.014):
    project = james_hane_6m_reference_project()
    root_scale = multiplier / metric_stretches(project, TRANSFORMATION)[1]
    return uniform_product_scale_sweep(
        project, TRANSFORMATION, (root_scale,),
    )[0][1]


def test_joint_candidate_recovers_proven_metric_root_and_preserves_inputs():
    project = _known_close_input(1.014)
    baseline = metric_stretches(project, TRANSFORMATION)
    assert baseline[1] == pytest.approx(1.014, abs=2e-12)
    report = find_joint_lambda2_target(
        project, TRANSFORMATION, length_relative_bound=0.05,
        angle_bound_deg=2.0,
    )
    assert report["status"] == "NUMERICAL_AM_LAMBDA2_TARGET_ONLY"
    assert report["search_exhaustive"] is False
    assert report["candidate"] is not None
    candidate = report["candidate"]
    assert abs(candidate["lambda2_residual"]) < 2e-8
    assert candidate["normalized_squared_change"] >= 0.0

    changed = state_at_joint_target(
        project, TRANSFORMATION, candidate["parameter_values"],
    )
    np.testing.assert_allclose(
        metric_stretches(changed, TRANSFORMATION), candidate["stretches"],
        atol=1e-11, rtol=1e-11,
    )
    transformation = project.transformation(TRANSFORMATION)
    assert changed.phase(transformation.parent_phase_id).to_dict() == (
        project.phase(transformation.parent_phase_id).to_dict()
    )
    assert (changed.transformation(TRANSFORMATION).correspondence.C_M_from_A ==
            transformation.correspondence.C_M_from_A)


def test_joint_neutral_point_is_not_mistaken_for_root():
    project = _known_close_input(1.03)
    allowed = find_joint_lambda2_target(
        project, TRANSFORMATION, length_relative_bound=0.001,
        angle_bound_deg=0.01,
    )
    assert allowed["status"] == "NO_NUMERICAL_ROOT_RESOLVED_WITHIN_BOUNDS"
    assert allowed["candidate"] is None


def test_coupled_solver_rejects_invalid_metric_design_controls():
    project = james_hane_6m_reference_project()
    for bound in (0.0, -0.1, 0.5, float("nan")):
        with pytest.raises(ValueError):
            find_joint_lambda2_target(
                project, TRANSFORMATION, length_relative_bound=bound,
            )
    with pytest.raises(ValueError):
        find_joint_lambda2_target(
            project, TRANSFORMATION, angle_bound_deg=float("nan"),
        )
    with pytest.raises(ValueError):
        state_at_joint_target(project, TRANSFORMATION, {"a_scale": 1.0})


def test_joint_design_does_not_claim_supercompatibility_or_edit_theory_engine():
    root = Path(__file__).resolve().parents[2]
    solver = (root / "twin_app" / "joint_metric_design_2026.py").read_text()
    ui = (root / "twin_app" / "metric_design_ui.py").read_text()
    ast.parse(solver)
    ast.parse(ui)
    assert "NUMERICAL_AM_LAMBDA2_TARGET_ONLY" in solver
    assert "NO_NUMERICAL_ROOT_RESOLVED_WITHIN_BOUNDS" in solver
    assert "_render_joint_compatibility_design(" in ui
    assert "Verify coupled target with CT and cofactor" in ui
