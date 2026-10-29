"""Smoke-test the experimental-OR handoff using one synthetic single-parent patch."""

import numpy as np
import pytest

from cualni_cryst.ebsd_map import EBSDMap
from cualni_cryst.legacy_suite_bridge import Phase, reconstruct_ebsd_parent


def test_explicit_or_reconstructs_identity_single_parent():
    ebsd = EBSDMap(
        orientations=np.stack([np.eye(3), np.eye(3), np.eye(3)]),
        phase_id=np.array([1, 1, 1], dtype=int),
        indexed=np.array([True, True, True]),
        x=np.array([0.0, 1.0, 2.0]),
        y=np.array([0.0, 0.0, 0.0]),
        z=np.array([0.0, 0.0, 0.0]),
    )
    phase = Phase("cubic", "1", 2.0, 2.0, 2.0)
    outcome = reconstruct_ebsd_parent(
        ebsd, phase, phase, 1, np.eye(3)[None],
    )
    assert outcome["sampled_points"] == 3
    assert outcome["weighted_inlier_fraction"] == pytest.approx(1)
    assert outcome["mean_inlier_residual_deg"] == pytest.approx(0, abs=1e-7)


def test_rejects_nonrotation_or():
    ebsd = EBSDMap(
        orientations=np.stack([np.eye(3), np.eye(3)]),
        phase_id=np.array([1, 1]),
        indexed=np.array([True, True]),
        x=np.array([0., 1.]),
        y=np.array([0., 0.]),
        z=np.zeros(2),
    )
    phase = Phase("triclinic", "1", 2, 2, 2)
    with pytest.raises(ValueError):
        reconstruct_ebsd_parent(ebsd, phase, phase, 1, np.diag([1, 1, -1])[None])
