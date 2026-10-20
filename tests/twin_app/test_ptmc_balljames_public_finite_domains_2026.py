"""Public solver entry points reject NaN and infinity before material inference."""
from __future__ import annotations

import numpy as np
import pytest

from cualni_cryst.ball_james_adapter import analyze_ball_james
from cualni_cryst.ptmc_adapter import (
    PTMCTwinPlaneInput,
    _validate_ptmc_public_controls,
    analyze_ptmc_slip,
    analyze_ptmc_twinning,
)

INVALID = [float("nan"), float("inf"), -float("inf")]
PTMC_DEFAULTS = dict(
    dilatational_factor=1.0,
    algebraic_tolerance=1e-10,
    eigenvalue_tolerance=1e-8,
    middle_stretch_tolerance=1e-7,
    polynomial_tolerance=1e-10,
)


@pytest.mark.parametrize("invalid", INVALID)
@pytest.mark.parametrize("field", [
    "algebraic_tolerance", "eigenvalue_tolerance", "rank_one_tolerance",
    "cofactor_tolerance",
])
def test_ball_james_rejects_nonfinite_public_policy(field, invalid):
    with pytest.raises(ValueError, match=field):
        analyze_ball_james(
            np.eye(3), np.eye(3), np.eye(3), [],
            **{field: invalid},
        )


@pytest.mark.parametrize("invalid", INVALID)
@pytest.mark.parametrize("field", list(PTMC_DEFAULTS))
def test_ptmc_controls_reject_nonfinite_values_before_inference(field, invalid):
    args = dict(PTMC_DEFAULTS)
    args[field] = invalid
    with pytest.raises(ValueError, match=field):
        _validate_ptmc_public_controls(**args)


@pytest.mark.parametrize("invalid", INVALID)
def test_both_ptmc_public_routes_reject_nonfinite_dilatation(invalid):
    for entry, args in (
        (analyze_ptmc_twinning, (None, None, np.eye(3), [], [])),
        (analyze_ptmc_slip, (None, None, np.eye(3), [], [], None)),
    ):
        with pytest.raises(ValueError, match="dilatational_factor"):
            entry(*args, dilatational_factor=invalid)


@pytest.mark.parametrize("invalid", INVALID)
def test_ptmc_twin_plane_constructor_rejects_nonfinite_angle_gate(invalid):
    with pytest.raises(ValueError, match="plane_tolerance_deg"):
        PTMCTwinPlaneInput((1.0, 0.0, 0.0), plane_tolerance_deg=invalid)
