from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.phase1_contracts import (
    axis_angle_display,
    classify_scientific_error,
    frozen_cualni_metric_regression,
    frozen_cualni_oracle,
    metric_geometry_audit,
    physical_closing_gap_label,
)


def test_frozen_cualni_metric_checkpoint_is_reconstructed_from_inputs() -> None:
    oracle = frozen_cualni_oracle()
    result = frozen_cualni_metric_regression()
    assert abs(result["lambda2"] - oracle.lambda2) < 5.0e-13
    assert abs(result["ct_nearest_zero_residual"] - oracle.ct_nearest_zero_residual) < 5.0e-13
    np.testing.assert_allclose(
        result["lambdas"],
        [0.9222298692360634, 0.9980600225927657, 1.1169270102738489],
        rtol=0.0,
        atol=7.0e-13,
    )


def test_zero_rotation_axis_is_not_given_physical_meaning() -> None:
    shown = axis_angle_display([1.0, 0.0, 0.0], 0.0)
    assert shown["zero_rotation"] is True
    assert shown["axis"] is None
    assert "undefined" in shown["axis_text"]


def test_non_cubic_direction_and_plane_are_metric_distinct() -> None:
    beta = np.deg2rad(96.35)
    a, b, c = 4.392, 5.360, 12.940
    M = np.array(
        [
            [a * a, 0.0, a * c * np.cos(beta)],
            [0.0, b * b, 0.0],
            [a * c * np.cos(beta), 0.0, c * c],
        ]
    )
    audit = metric_geometry_audit(M, [1, 0, 1], [1, 0, 1])
    u = np.asarray(audit["direction_cartesian"])
    n = np.asarray(audit["plane_normal_cartesian"])
    assert not np.allclose(u / np.linalg.norm(u), n / np.linalg.norm(n), rtol=0.0, atol=1.0e-8)
    assert audit["metric_inverse_residual"] < 1.0e-12


def test_closing_gap_label_uses_physical_branch_metadata() -> None:
    row = SimpleNamespace(metadata={
        "operator_index": 1,
        "twin_index": 3,
        "candidate_index": 0,
        "twin_kind": "I",
        "twin_classification": "compound",
        "twin_representations": ["I", "II"],
    })
    label = physical_closing_gap_label(row)
    assert "Type-I" in label
    assert "operator 1" in label
    assert "twin 3" in label
    assert "sign branch A" in label
    assert "compound" in label


def test_scientific_error_taxonomy_is_specific() -> None:
    assert classify_scientific_error(ValueError("correspondence matrix is singular"))["title"] == "Singular correspondence"
    assert classify_scientific_error(ValueError("metric is not SPD / positive definite"))["title"] == "Non-SPD lattice metric"
    assert classify_scientific_error(ValueError("bad rational p/q matrix entry"))["title"] == "Malformed exact rational input"
    assert classify_scientific_error(ValueError("orientation is not a proper SO(3) rotation"))["title"] == "Invalid orientation relationship"
