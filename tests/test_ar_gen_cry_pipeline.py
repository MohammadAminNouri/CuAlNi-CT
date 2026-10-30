from __future__ import annotations

"""Standalone workflow regression, numerical invariants, malformed input and EBSD fixtures."""

import json

import numpy as np
import pytest

from cualni_cryst.legacy_suite_bridge import Phase
from cualni_cryst.ar_gen_cry_pipeline import (
    MAX_EBSD_BYTES, ar_parent_fit, ar_stage, cry_stage, gen_stage, report_json,
)


@pytest.fixture
def cry() -> dict:
    return cry_stage(
        Phase("Parent", "m-3m", 2, 2, 2),
        Phase("Product", "mmm", 2, 3, 4),
        (1, 0, 0), (1, 0, 0),
    )


@pytest.fixture
def tiny_ang() -> bytes:
    return (
        b"# GRID: SqrGrid\n"
        b"# XSTEP: 1\n# YSTEP: 1\n"
        b"0 0 0 0 0 100 0.99 1 0 0\n"
        b"0 0 0 1 0 100 0.99 1 0 0\n"
        b"0 0 0 0 1 100 0.99 1 0 0\n"
    )


def test_cry_cubic_direct_and_reciprocal_geometry(cry):
    assert cry["parent"]["direction_length_angstrom"] == pytest.approx(2)
    assert cry["parent"]["d_spacing_angstrom"] == pytest.approx(2)
    assert cry["parent"]["point_group_order"] == 48
    assert cry["product"]["point_group_order"] == 8
    assert cry["provenance"].startswith("user input")


def test_cry_is_invariant_under_direction_plane_sign():
    p = Phase("Test", "mmm", 2.5, 3.5, 5.0)
    result_a = cry_stage(p, p, (2, 1, -3), (1, -2, 3))
    result_b = cry_stage(p, p, (-2, -1, 3), (-1, 2, -3))
    assert result_a["parent"]["direction_length_angstrom"] == pytest.approx(
        result_b["parent"]["direction_length_angstrom"], rel=1e-13)
    assert result_a["parent"]["d_spacing_angstrom"] == pytest.approx(
        result_b["parent"]["d_spacing_angstrom"], rel=1e-13)


def test_cry_metric_reciprocity_for_orthorhombic():
    p = Phase("orthorhombic", "mmm", 2, 3, 4)
    for axis, length in enumerate((2.0, 3.0, 4.0)):
        h = tuple(float(k == axis) for k in range(3))
        out = cry_stage(p, p, h, h)
        assert out["parent"]["direction_length_angstrom"] == pytest.approx(length)
        assert out["parent"]["d_spacing_angstrom"] == pytest.approx(length)


@pytest.mark.parametrize("direction,plane", [
    ((0, 0, 0), (1, 0, 0)),
    ((1, 0, 0), (0, 0, 0)),
    ((1, 0), (1, 0, 0)),
    ((float("nan"), 0, 1), (1, 0, 0)),
])
def test_cry_invalid_vectors_fail_closed(direction, plane):
    p = Phase("Test", "1", 2, 3, 4)
    with pytest.raises(ValueError):
        cry_stage(p, p, direction, plane)


def test_cry_rejects_incompatible_lattice_symmetry():
    p = Phase("Invalid cubic", "m-3m", 2, 3, 4)
    with pytest.raises(ValueError, match="not compatible"):
        cry_stage(p, p, (1, 0, 0), (1, 0, 0))


def test_gen_exact_groupoid_lagrange_burnside_and_adjacency(cry):
    result = gen_stage(cry, "1 0 0\n0 1 0\n0 0 1")
    assert result["variants"] == 6
    assert result["subgroup_order"] == 8
    assert result["variants"] * result["subgroup_order"] == 48
    assert result["operator_classes"] == result["burnside_count"]
    assert sum(result["class_sizes"]) == 48
    assert len(result["adjacency"]) == result["variants"]
    assert all(len(row) == result["variants"] for row in result["adjacency"])
    for i, row in enumerate(result["adjacency"]):
        assert row[i] == result["adjacency"][0][0]


@pytest.mark.parametrize("matrix", [
    "1 0 0\n0 0 0\n0 0 1",
    "1 0 0\n0 1 0",
    "1 0 0\n0 __import__('os') 0\n0 0 1",
    "1 0 0\n0 1/0 0\n0 0 1",
    "1 0 0\n0 nan 0\n0 0 1",
])
def test_gen_bad_matrices_rejected(cry, matrix):
    with pytest.raises((ValueError, TypeError, ZeroDivisionError)):
        gen_stage(cry, matrix)


def test_gen_requires_cry():
    with pytest.raises(ValueError, match="Cry"):
        gen_stage({}, "1 0 0\n0 1 0\n0 0 1")


def test_ar_ang_real_parser_and_counts(tiny_ang):
    report = ar_stage("data.ang", tiny_ang)
    assert report["points"] == 3
    assert report["indexed_points"] == 3
    assert report["phase_counts"] == {"1": 3}
    assert report["stage"] == "AR"
    assert "not" in report["provenance"].lower() or "unverified" in report["provenance"].lower()


@pytest.mark.parametrize("filename,data", [
    ("sample.pkl", b"not a pickle"),
    ("sample.ctf", b"\x00\x00bad"),
    ("sample.ang", b""),
    ("sample.ang", b"invalid ang"),
])
def test_ar_rejects_invalid_untrusted_upload(filename, data):
    with pytest.raises((ValueError, UnicodeError)):
        ar_stage(filename, data)


def test_ar_rejects_large_upload_before_parsing():
    with pytest.raises(ValueError, match="8 MiB"):
        ar_stage("big.ang", b"x" * (MAX_EBSD_BYTES + 1))


def test_ar_parent_fit_refuses_fabricated_or(cry, tiny_ang):
    with pytest.raises(ValueError, match="JSON"):
        ar_parent_fit("tiny.ang", tiny_ang, cry, 1, "there is no OR")
    with pytest.raises(ValueError, match="rotations|rotation|SO"):
        ar_parent_fit("tiny.ang", tiny_ang, cry, 1,
                      json.dumps([[[2, 0, 0], [0, 1, 0], [0, 0, 1]]]))


def test_ar_parent_fit_trivial_synthetic_case(tiny_ang):
    # Deliberately synthetic identity relationship to test code consistency.
    # Not a Cu-Al-Ni OR or a benchmark against an experimental measurement.
    one = Phase("Phase", "1", 3, 4, 5)
    case = cry_stage(one, one, (1, 0, 0), (1, 0, 0))
    fit = ar_parent_fit("tiny.ang", tiny_ang, case, 1,
                        json.dumps([np.eye(3).tolist()]), max_points=3)
    assert fit["sampled_points"] == 3
    assert fit["weighted_inlier_fraction"] == pytest.approx(1.0, abs=1e-8)
    assert fit["mean_inlier_residual_deg"] <= 1e-5
    assert fit["experimental_or_user_supplied"] is True


def test_report_preserves_missing_stages(cry):
    payload = json.loads(report_json(cry, None, None))
    assert payload["software"] == "AR · Gen · Cry"
    assert payload["cry"]["stage"] == "Cry"
    assert payload["gen"] is None and payload["ar"] is None
    assert payload["ar_parent_fit"] is None


def test_report_requires_crystal_setup():
    with pytest.raises(ValueError, match="Cry"):
        report_json(None, None, None)
