from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.phase1_contracts import (
    axis_angle_display,
    classify_scientific_error,
    frozen_cualni_oracle,
    metric_geometry_audit,
    physical_closing_gap_label,
)


def test_zero_rotation_axis_is_explicitly_undefined() -> None:
    shown = axis_angle_display((1.0, 0.0, 0.0), 0.0)
    assert shown["zero_rotation"] is True
    assert shown["axis"] is None
    assert "undefined" in shown["axis_text"]


def test_physical_closing_gap_label_uses_real_metadata() -> None:
    row = SimpleNamespace(
        metadata={
            "operator_index": 4,
            "twin_index": 3,
            "twin_kind": "I",
            "twin_classification": "compound",
            "twin_representations": ("I", "II"),
            "candidate_index": 0,
        }
    )
    label = physical_closing_gap_label(row)
    assert "Type-I" in label
    assert "operator 4" in label
    assert "twin 3" in label
    assert "sign branch A" in label


def test_monoclinic_101_direction_is_not_plane_normal_by_index_text() -> None:
    o = frozen_cualni_oracle()
    beta = np.deg2rad(o.product_beta_deg)
    M = np.array(
        [
            [o.product_a**2, 0.0, o.product_a * o.product_c * np.cos(beta)],
            [0.0, o.product_b**2, 0.0],
            [o.product_a * o.product_c * np.cos(beta), 0.0, o.product_c**2],
        ]
    )
    result = metric_geometry_audit(M, (1, 0, 1), (1, 0, 1))
    r = np.asarray(result["direction_cartesian"], float)
    n = np.asarray(result["plane_normal_cartesian"], float)
    cosine = abs(float((r / np.linalg.norm(r)) @ (n / np.linalg.norm(n))))
    # Matching integer triplets do not make [101] parallel to the physical
    # normal of (101) in this monoclinic metric.
    assert cosine < 0.999
    assert result["metric_inverse_residual"] < 1e-12


def test_error_taxonomy_distinguishes_scientific_failures() -> None:
    assert classify_scientific_error(ValueError("correspondence matrix is singular"))["title"] == "Singular correspondence"
    assert classify_scientific_error(ValueError("metric is not positive definite"))["title"] == "Non-SPD lattice metric"
    assert classify_scientific_error(ValueError("entry is not a rational p/q"))["title"] == "Malformed exact rational input"
    assert classify_scientific_error(ValueError("rotation is outside SO(3)"))["title"] == "Invalid orientation relationship"


def test_frozen_oracle_inputs_are_exact_not_output_only() -> None:
    o = frozen_cualni_oracle()
    assert o.parent_a == 5.812
    assert (o.product_a, o.product_b, o.product_c, o.product_beta_deg) == (4.392, 5.360, 12.940, 96.35)
    assert o.correspondence_m_from_a[2] == ("0", "1/3", "-1/3")
