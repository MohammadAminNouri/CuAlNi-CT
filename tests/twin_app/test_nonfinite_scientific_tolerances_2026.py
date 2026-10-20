"""Non-finite tolerances must never silently establish scientific compatibility.

Regression cases for NaN/infinity entering CMC, PTMC, Ball–James and shared
classification policy.  Ordinary finite tolerances retain their behavior.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.ct import analyze_cmc, habit_planes_from_analysis
from cualni_cryst.ball_james_adapter import evaluate_cofactor_conditions
from cualni_cryst.ptmc_adapter import solve_affine_ptmc_parameters
from cualni_cryst.numerics import NumericalPolicy, ResidualClass, classify_residual


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_shared_numeric_policy_rejects_nonfinite_inputs(invalid):
    with pytest.raises(ValueError, match="positive"):
        NumericalPolicy(exact_eigenvalue=invalid)
    with pytest.raises(ValueError, match="finite"):
        classify_residual(0.0, invalid)
    with pytest.raises(ValueError, match="finite"):
        classify_residual(invalid, 1e-8)
    with pytest.raises(ValueError, match="finite"):
        classify_residual(0.0, 1e-8, near_factor=invalid)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_cmc_refuses_nonfinite_compatibility_tolerance(invalid):
    parent = np.eye(3)
    product = np.diag((1.3, 1.0, 0.8))
    correspondence = Correspondence(sp.eye(3))
    with pytest.raises(ValueError, match="finite"):
        analyze_cmc(parent, product, correspondence, tol=invalid)
    valid = analyze_cmc(parent, product, correspondence, tol=1e-9)
    assert valid.exact_compatible and valid.degeneracy_order == 1
    with pytest.raises(ValueError, match="finite"):
        habit_planes_from_analysis(parent, valid, tol=invalid)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_ball_james_refuses_nonfinite_cofactor_tolerance(invalid):
    with pytest.raises(ValueError, match="finite"):
        evaluate_cofactor_conditions(
            np.eye(3), np.array((1.0, 0.0, 0.0)),
            np.array((0.0, 1.0, 0.0)), tolerance=invalid,
        )


@pytest.mark.parametrize("field", ["polynomial_tolerance", "middle_stretch_tolerance"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_ptmc_refuses_nonfinite_parameter_tolerances(field, invalid):
    arguments = {field: invalid}
    with pytest.raises(ValueError, match="finite"):
        solve_affine_ptmc_parameters(
            np.eye(3),
            np.array(((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
            **arguments,
        )


def test_shared_policy_retains_finite_classification():
    assert classify_residual(0.0, 1e-8) is ResidualClass.PASS
    assert classify_residual(2e-8, 1e-8) is ResidualClass.NEAR
    assert classify_residual(2e-5, 1e-8) is ResidualClass.FAIL
    with pytest.raises(ValueError, match="overflows"):
        classify_residual(1.0, 1e308, near_factor=10.0)
