from dataclasses import replace

import numpy as np
import pytest

from cualni_cryst.orientation import OrientationService
from cualni_cryst.project_state import (
    OrientationState,
    OrientationTheoryOrigin,
    james_hane_6m_reference_project,
)


def test_project_rejects_orientation_with_unknown_phase():
    project = james_hane_6m_reference_project()
    state = OrientationState(
        orientation_id="bad",
        label="bad",
        reference_phase_id="austenite_do3",
        moving_phase_id="does_not_exist",
        R_reference_from_moving=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
    )
    with pytest.raises(ValueError, match="unknown moving phase"):
        replace(project, orientations=(state,))


def test_orientation_state_rejects_improper_matrix():
    with pytest.raises(ValueError, match="proper rotation"):
        OrientationState(
            orientation_id="improper",
            label="improper",
            reference_phase_id="a",
            moving_phase_id="b",
            R_reference_from_moving=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, -1.0),
            ),
        )


def test_registered_polar_orientation_project_validates():
    project = james_hane_6m_reference_project()
    service = OrientationService(project)
    polar = service.polar_orientation("do3_to_6m_reference")
    registered = replace(project, orientations=(polar,))
    registered.validate().assert_passed()
    assert polar.theory_origin is OrientationTheoryOrigin.POLAR_CORRESPONDENCE
    assert np.isclose(
        np.linalg.det(np.asarray(polar.R_reference_from_moving)),
        1.0,
        atol=1e-12,
    )
