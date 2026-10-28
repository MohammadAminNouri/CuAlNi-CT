from __future__ import annotations

"""Metric-only inverse targets, never a supercompatibility certificate.

Cayron (2026), Acta Materialia 316, 122399, §4--5:
generalized eigenvalues of (C.T M_M C, M_A) are squared principal
stretches. Lambda_2=1 is the necessary A/M IPS condition, not the full
A/M/M shear/shear and cofactor conditions. A candidate found here must be
passed, unchanged, to the independent existing CT/BJ/PTMC solvers.

No phase symmetry, correspondence, parent metric, or unverified CTF
orientation relationship is adjusted by this parameter suggestion.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq, minimize_scalar

from cualni_cryst.supercompatibility_atlas import (
    independent_product_lattice_parameters,
    symmetry_preserving_product_grid,
    uniform_product_scale_sweep,
)

DOI = "10.1016/j.actamat.2026.122399"


@dataclass(frozen=True)
class MetricTarget:
    parameter: str
    baseline_lambda2: float
    derivative: float | None
    candidates: tuple[dict[str, float], ...]
    minimum_abs_residual: float
    scan: tuple[dict[str, float], ...]
    status: str


def metric_stretches(project: Any, transformation_id: str) -> tuple[float, float, float]:
    transform = project.transformation(transformation_id)
    parent = project.phase(transform.parent_phase_id)
    product = project.phase(transform.product_phase_id)
    MA = parent.lattice.metric()
    MM = product.lattice.metric()
    C = np.asarray(transform.correspondence.C_M_from_A, dtype=float)
    if C.shape != (3, 3) or not np.isfinite(C).all():
        raise ValueError("Correspondence must be a finite 3x3 matrix")
    G = C.T @ MM @ C
    G = 0.5 * (G + G.T)
    eigenvalues = eigh(G, MA, eigvals_only=True, check_finite=True)
    if np.any(eigenvalues <= 0) or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("Correspondence metrics have no positive stretch spectrum")
    return tuple(float(np.sqrt(v)) for v in eigenvalues)


def state_at_parameter(project, transformation_id, parameter: str, value: float):
    if not isfinite(value):
        raise ValueError("The parameter must be finite")
    if parameter == "uniform_scale":
        state = uniform_product_scale_sweep(
            project, transformation_id, [value],
        )[0][1]
    else:
        allowed = independent_product_lattice_parameters(project, transformation_id)
        if parameter not in allowed:
            raise ValueError(
                f"{parameter} cannot vary independently for this crystal setting; "
                f"available={allowed}"
            )
        state = symmetry_preserving_product_grid(
            project, transformation_id, {parameter: [value]}, max_states=1,
        )[0][1]
    state.validate().assert_passed()
    return state


def scan_lambda2_targets(
    project, transformation_id: str, parameter: str, *,
    length_relative_bound: float = .05,
    angle_bound_deg: float = 3.0,
    samples: int = 61,
    tolerance: float = 2e-8,
) -> MetricTarget:
    """Propose single-axis lambda2=1 metric targets within an explicit window.

    Sign-changing roots are bracketed with Brent. Local minima are also
    examined for tangencies; no finite sampling is called exhaustive.
    Near-zero eigenvalue degeneracy may invalidate derivative estimates and
    is flagged. Root suggestions are neither experimentally attainable
    compositions nor a full A/M/M compatibility result.
    """
    if not 0 < length_relative_bound < .5:
        raise ValueError("Length search window must lie between 0 and 50%")
    if not 0 < angle_bound_deg < 30:
        raise ValueError("Angle window must lie between 0 and 30 degrees")
    if not 9 <= samples <= 301:
        raise ValueError("Use 9 to 301 scan samples")
    if not 0 < tolerance < .001:
        raise ValueError("Target tolerance must be positive and below .001")
    allowed = ("uniform_scale",) + independent_product_lattice_parameters(
        project, transformation_id
    )
    if parameter not in allowed:
        raise ValueError(f"Not an independent parameter: {parameter}")
    baseline = 0.0 if parameter.endswith("_offset_deg") else 1.0
    bound = angle_bound_deg if parameter.endswith("_offset_deg") else length_relative_bound
    lo, hi = baseline - bound, baseline + bound
    parent_lambda2 = metric_stretches(project, transformation_id)[1]
    cache: dict[float, float] = {}

    def f(x: float) -> float:
        key = float(x)
        if key not in cache:
            candidate = state_at_parameter(
                project, transformation_id, parameter, key
            )
            cache[key] = metric_stretches(candidate, transformation_id)[1] - 1.0
        return cache[key]

    xs = np.linspace(lo, hi, samples)
    rows = []
    for x in xs:
        try:
            delta = f(float(x))
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            delta = float("nan")
        rows.append({"parameter_value": float(x), "lambda2_minus_one": float(delta)})
    finite = [(i, x, rows[i]["lambda2_minus_one"]) for i, x in enumerate(xs)
              if isfinite(rows[i]["lambda2_minus_one"])]
    if not finite:
        return MetricTarget(parameter, parent_lambda2, None, (),
                            float("inf"), tuple(rows), "NO_VALID_STATES")

    roots: list[float] = []
    for i in range(len(rows)-1):
        a, b = rows[i], rows[i+1]
        ya, yb = a["lambda2_minus_one"], b["lambda2_minus_one"]
        if not isfinite(ya) or not isfinite(yb):
            continue
        if abs(ya) <= tolerance:
            roots.append(a["parameter_value"])
        if ya * yb < 0:
            roots.append(float(brentq(
                f, a["parameter_value"], b["parameter_value"],
                xtol=1.e-12, rtol=1.e-13,
            )))
    if isfinite(rows[-1]["lambda2_minus_one"]) and abs(rows[-1]["lambda2_minus_one"]) <= tolerance:
        roots.append(float(xs[-1]))

    # Additional bounded minimizations can detect even-multiplicity/tangent
    # roots. These are tentative unless the exact metric residual passes.
    for i in range(1, len(rows)-1):
        a,b,c=(rows[k]["lambda2_minus_one"] for k in (i-1,i,i+1))
        if all(map(isfinite,(a,b,c))) and abs(b)<=abs(a) and abs(b)<=abs(c):
            opt = minimize_scalar(
                lambda x: abs(f(x)),
                bounds=(rows[i-1]["parameter_value"],
                        rows[i+1]["parameter_value"]),
                method="bounded",options={"xatol":1e-12},
            )
            if opt.success and abs(f(float(opt.x)))<=tolerance:
                roots.append(float(opt.x))
    roots.sort(key=lambda x: abs(x-baseline))
    unique: list[float] = []
    for x in roots:
        if all(abs(x-p)>1e-7 for p in unique):
            unique.append(x)
    candidates = tuple({
        "parameter_value": float(x),
        "change": float(x-baseline),
        "lambda2": float(f(x)+1.0),
        "abs_lambda2_minus_one": float(abs(f(x))),
    } for x in unique[:12] if abs(f(x))<=tolerance)
    step = min(bound/100.,1e-4 if parameter.endswith("_offset_deg") else 1e-5)
    try:
        derivative = float((f(baseline+step)-f(baseline-step))/(2*step))
    except (ValueError, ArithmeticError, np.linalg.LinAlgError):
        derivative = None
    minimum = min(abs(x[2]) for x in finite)
    status = (
        "LAMBDA2_ROOT_CANDIDATES_REQUIRE_FULL_COMPATIBILITY_CHECK"
        if candidates else "NO_ROOT_RESOLVED_IN_REQUESTED_WINDOW"
    )
    return MetricTarget(
        parameter, parent_lambda2, derivative, candidates, minimum,
        tuple(rows), status,
    )
