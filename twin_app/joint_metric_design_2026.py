from __future__ import annotations

"""Bounded, symmetry-respecting inverse design of the A/M IPS metric condition.

This is an A/M metric solver, NOT an inverse CT shear/shear or cofactor solver.
It optimizes the normalized squared distance from an entered product cell,
subject to lambda_2 = 1 at the FIXED parent metric and correspondence.
The result is a locally obtained numerical candidate, not a global minimum,
a supercompatibility certificate, or an experimentally attainable alloy.
"""

from math import isfinite
from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize

from cualni_cryst.supercompatibility_atlas import (
    independent_product_lattice_parameters,
    symmetry_preserving_product_grid,
)
from twin_app.metric_design_targets import metric_stretches


def state_at_joint_target(
    project: Any,
    transformation_id: str,
    parameter_values: Mapping[str, float],
) -> Any:
    """Set only crystallographically independent product-cell parameters."""
    allowed = set(independent_product_lattice_parameters(project, transformation_id))
    if not parameter_values or set(parameter_values) != allowed:
        raise ValueError("All and only the independent product parameters are required")
    axes = {}
    for name, val in parameter_values.items():
        value = float(val)
        if not isfinite(value) or (name.endswith("_scale") and value <= 0):
            raise ValueError("Product parameters must be finite and length scales positive")
        axes[name] = (value,)
    state = symmetry_preserving_product_grid(
        project, transformation_id, axes, max_states=1,
    )[0][1]
    state.validate().assert_passed()
    return state


def find_joint_lambda2_target(
    project: Any,
    transformation_id: str,
    *,
    length_relative_bound: float = 0.05,
    angle_bound_deg: float = 3.0,
    tolerance: float = 2e-8,
) -> dict:
    """Minimize squared changes in units of user-entered bounds, with lambda2=1.

    The nonlinear constrained search is local and finite. A feasible uniform
    length-scaling seed is constructed analytically because all principal
    stretches scale identically. Every returned candidate is re-evaluated
    against the exact generalized metric eigenproblem, not trusted merely
    because SLSQP reports success.

    This routine deliberately NEVER evaluates or optimizes CT M/M shear/shear,
    Ball--James cofactor conditions, chemical constraints, or measured errors.
    """
    if not (isfinite(length_relative_bound) and 0 < length_relative_bound < 0.5):
        raise ValueError("Length bound must be finite and between 0 and 50%")
    if not (isfinite(angle_bound_deg) and 0 < angle_bound_deg < 30):
        raise ValueError("Angle bound must be finite and between 0 and 30 degrees")
    if not (isfinite(tolerance) and 0 < tolerance < 1e-4):
        raise ValueError("Metric root tolerance must be finite and below 1e-4")
    axes = independent_product_lattice_parameters(project, transformation_id)
    baseline_stretches = metric_stretches(project, transformation_id)
    base = np.array([0.0 if name.endswith("_offset_deg") else 1.0 for name in axes])
    widths = np.array([
        angle_bound_deg if name.endswith("_offset_deg") else length_relative_bound
        for name in axes
    ])
    bounds = [(-1.0, 1.0) for _ in axes]
    cache = {}

    def unpack(x):
        return {name: float(base[i] + widths[i]*float(x[i]))
                for i, name in enumerate(axes)}

    def evaluate(x):
        key = tuple(round(float(v), 13) for v in x)
        if key not in cache:
            try:
                state = state_at_joint_target(project, transformation_id, unpack(x))
                spectrum = metric_stretches(state, transformation_id)
                cache[key] = (spectrum, "")
            except (AssertionError, ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                cache[key] = (None, f"{type(exc).__name__}: {exc}")
        return cache[key]

    def constraint(x):
        spectrum, _ = evaluate(x)
        if spectrum is None:
            # Never interpret invalid crystal metrics as a compatible state.
            return 1e3
        return float(spectrum[1] - 1.0)

    def score(x):
        return float(np.dot(x, x))

    zero = np.zeros(len(axes), dtype=float)
    initial = [zero]
    # Analytic witness, provided it is inside the allowed window.
    scale = 1.0/baseline_stretches[1]
    scaled = np.array([
        0.0 if name.endswith("_offset_deg")
        else (scale - 1.0)/widths[i]
        for i, name in enumerate(axes)
    ])
    if np.all(np.abs(scaled) <= 1.0):
        initial.append(scaled)
    # A least-norm linearized correction provides another deterministic start.
    gradients = []
    for i in range(len(axes)):
        perturb = np.zeros(len(axes))
        perturb[i] = 1e-4
        gradients.append(
            (constraint(perturb)-constraint(-perturb))/(2e-4)
        )
    grad = np.asarray(gradients)
    if np.all(np.isfinite(grad)) and float(grad @ grad) > 1e-20:
        linear = (1.0-baseline_stretches[1])*grad/float(grad @ grad)
        initial.append(np.clip(linear, -1.0, 1.0))

    successful = []
    for start in initial:
        # Keep the analytic witness even if a nonlinear optimizer fails.
        starts = [start]
        try:
            opt = minimize(
                score, start, method="SLSQP", bounds=bounds,
                constraints=({"type": "eq", "fun": constraint},),
                options={"maxiter": 140, "ftol": 1e-12, "disp": False},
            )
            starts.append(opt.x)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            pass
        for x in starts:
            x = np.asarray(x, dtype=float)
            if not np.isfinite(x).all() or np.max(np.abs(x)) > 1.00000001:
                continue
            x = np.clip(x, -1.0, 1.0)
            spectrum, _ = evaluate(x)
            if spectrum is not None and abs(spectrum[1]-1.0) <= tolerance:
                successful.append((score(x), x.copy(), spectrum))
    if not successful:
        return {
            "status": "NO_NUMERICAL_ROOT_RESOLVED_WITHIN_BOUNDS",
            "baseline_stretches": baseline_stretches,
            "parameters": axes, "candidate": None,
            "method": "bounded SLSQP with exact metric recheck",
            "search_exhaustive": False, "states_evaluated": len(cache),
        }
    successful.sort(key=lambda v: v[0])
    merit, best, spectrum = successful[0]
    values = unpack(best)
    product = project.phase(project.transformation(transformation_id).product_phase_id).lattice
    changed = state_at_joint_target(project, transformation_id, values)
    new = changed.phase(changed.transformation(transformation_id).product_phase_id).lattice
    return {
        "status": "NUMERICAL_AM_LAMBDA2_TARGET_ONLY",
        "baseline_stretches": baseline_stretches, "parameters": axes,
        "candidate": {
            "parameter_values": values,
            "normalized_changes": {name: float(best[i]) for i, name in enumerate(axes)},
            "baseline_cell": {name: float(getattr(product, name)) for name in
                              ("a", "b", "c", "alpha_deg", "beta_deg", "gamma_deg")},
            "proposed_cell": {name: float(getattr(new, name)) for name in
                              ("a", "b", "c", "alpha_deg", "beta_deg", "gamma_deg")},
            "stretches": spectrum, "lambda2_residual": float(spectrum[1]-1.0),
            "normalized_squared_change": merit,
            "near_repeated_stretches": min(spectrum[1]-spectrum[0],
                                          spectrum[2]-spectrum[1]) < 1e-7,
        },
        "method": "bounded SLSQP with exact metric recheck",
        "search_exhaustive": False, "states_evaluated": len(cache),
    }
