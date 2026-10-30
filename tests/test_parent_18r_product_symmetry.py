from __future__ import annotations

"""Regression: EBSD product symmetry copies must not produce false parents."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cualni_cryst.ebsd_reconstruction import VariantOR, reconstruct_parent
from cualni_cryst.ebsd_analysis import minimum_disorientation


def test_monoclinic_18r_parent_votes_include_both_c2_representatives():
    parent = Rotation.from_euler("xyz", [12, 18, 25], degrees=True).as_matrix()
    r_parent_from_child = Rotation.from_euler(
        "xyz", [23, -16, 13], degrees=True
    ).as_matrix()
    S = Rotation.from_euler("y", 180, degrees=True).as_matrix()
    g = parent @ r_parent_from_child
    # Two distinct EBSD files/indexer conventions may return equivalent
    # physical child orientations as g or g @ C2; this is *not* a new variant.
    measured = np.stack((g, g @ S, g, g @ S, g))
    result = reconstruct_parent(
        measured,
        [VariantOR(1, r_parent_from_child)],
        [np.eye(3), S],
        [np.eye(3)],
        inlier_tolerance_deg=0.1,
        maximum_seed_candidates=64,
        maximum_iterations=10,
    )
    assert result.weighted_inlier_fraction == pytest.approx(1.0, abs=1e-12)
    assert result.mean_inlier_residual_deg == pytest.approx(0, abs=1e-5)
    assert minimum_disorientation(
        result.parent_orientation, parent, [np.eye(3)]
    ).angle_deg < 1e-5
    assert all(x.accepted for x in result.variant_assignments)


def test_invalid_improper_daughter_symmetry_is_rejected():
    with pytest.raises(ValueError):
        reconstruct_parent(
            np.stack((np.eye(3), np.eye(3))),
            [VariantOR(1, np.eye(3))],
            [np.diag([1, 1, -1])], [np.eye(3)],
        )
