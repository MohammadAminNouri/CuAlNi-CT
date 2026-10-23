"""Regression gate for the synthetic cubic-parent variant-identity defect.

Run from the CuAlNi-CT repository root after unzipping this bundle:
    python -m pytest -q test_variant_identity_repair_2026_10_21.py

These tests are INTENTIONALLY red against the original 2026-10-20 solver. They
are NOT proof of experimentally observed Cu-Al-Ni variant counts. They require
resolvable tensor orbits and exact generator provenance, rather than merely a
numerical tolerance adjustment. Uses the independently authored audit helper.
"""
from __future__ import annotations

import pytest

from variant_identity_forensics_2026_10_21 import run_once


def assert_correct_orbit(result: dict, expected: int) -> None:
    # Tests of the independent reference should fail clearly if its test input
    # unexpectedly degenerates at IEEE binary64 resolution.
    assert result["exact_binary64_orbit"] == expected, result
    assert result["expected_resolved_count"] == expected, result
    for prefix, count_key, matched_key, bad_key in (
        ("PTMC", "ptmc_count", "ptmc_orbit_matched_count", "ptmc_false_generator_membership"),
        ("Ball-James", "ball_james_count", "bj_orbit_matched_count", "bj_false_generator_membership"),
    ):
        assert result[count_key] == expected, (
            prefix, "variant orbit incorrectly collapsed/split", result
        )
        assert result[matched_key] == expected, (
            prefix, "one-to-one orbit coverage incomplete", result
        )
        assert result[bad_key] == [], (
            prefix, "incorrect symmetry-generator provenance", result
        )


@pytest.mark.parametrize("epsilon, expected", [
    ("1e-8", 6),
    ("3e-9", 6),
    ("1e-9", 6),
    ("1e-11", 6),
    ("1e-12", 6),
    ("0", 3),
])
@pytest.mark.parametrize("order", ["forward", "reverse", "sorted"])
def test_resolvable_orbit_and_generator_provenance(epsilon, expected, order):
    report = run_once(epsilon, 1.0, 1e-10, 1e-9, order)
    assert_correct_orbit(report, expected)


@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
@pytest.mark.parametrize("epsilon", ["1e-9", "1e-11"])
def test_same_stretch_orbit_under_uniform_lattice_scaling(scale, epsilon):
    report = run_once(epsilon, scale, 1e-10, 1e-9, "forward")
    assert_correct_orbit(report, 6)


@pytest.mark.parametrize("algebraic_tolerance", [1e-8, 1e-10, 1e-12])
def test_algebraic_verification_tolerance_does_not_change_variant_identity(algebraic_tolerance):
    # Scientific requirement for the repair: algebraic tolerance is for
    # validating identities, not for redefining the number of distinct variants.
    report = run_once("1e-9", 1.0, algebraic_tolerance, 1e-9, "forward")
    assert_correct_orbit(report, 6)
