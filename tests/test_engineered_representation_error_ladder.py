from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.linalg import eigh

from cualni_cryst.numerics import DEFAULT_NUMERICAL_POLICY
from validation.engineered.jsonutil import to_jsonable
from validation.engineered.physical import (
    crystal_rebasis,
    first_order_case,
    incompatible_case,
)
from validation.engineered.production import evaluate


def _rational_of_binary64(x: float) -> sp.Rational:
    """Exact rational value of the actual IEEE-754 binary64 input."""
    num, den = float(x).as_integer_ratio()
    return sp.Rational(num, den)


def _certified_generalized_mu(
    M_a: np.ndarray,
    M_m: np.ndarray,
    C: np.ndarray,
    *,
    interval_decimal_digits: int = 35,
) -> np.ndarray:
    r"""Certified roots of det(C^T M_M C - mu M_A)=0.

    The matrices are converted from their *actual* IEEE-754 values to exact
    rationals using ``float.as_integer_ratio``.  No decimal string rounding and
    no Cholesky/SVD oracle is involved.

    SymPy real-root isolation supplies disjoint rational intervals for the
    cubic generalized characteristic polynomial.  The midpoint of each
    interval is returned only after the interval is much narrower than the
    project's 1e-8 exact-compatibility threshold.
    """
    mu = sp.symbols("mu")

    A = sp.Matrix(
        [[_rational_of_binary64(x) for x in row] for row in np.asarray(M_a, float)]
    )
    M = sp.Matrix(
        [[_rational_of_binary64(x) for x in row] for row in np.asarray(M_m, float)]
    )
    X = sp.Matrix(
        [[_rational_of_binary64(x) for x in row] for row in np.asarray(C, float)]
    )

    polynomial = sp.Poly(sp.expand((X.T * M * X - mu * A).det()), mu)

    eps = sp.Rational(1, 10**interval_decimal_digits)
    isolated = sp.polys.polytools.intervals(polynomial, eps=eps)

    roots: list[float] = []
    for (lo, hi), multiplicity in isolated:
        midpoint = (lo + hi) / 2
        value = float(midpoint)
        roots.extend([value] * int(multiplicity))

    roots = sorted(roots)
    if len(roots) != 3:
        raise AssertionError(
            "SPD 3x3 generalized metric pencil did not yield exactly three "
            f"certified real roots: {isolated}"
        )

    values = np.asarray(roots, dtype=float)
    if not np.all(np.isfinite(values)) or np.min(values) <= 0.0:
        raise AssertionError(f"certified generalized roots are not positive: {values}")
    return values


def _classify_mu(mu: np.ndarray, tol: float) -> tuple[bool, int]:
    q = np.sort(np.asarray(mu, float)) - 1.0
    zero = np.abs(q) <= tol
    nullity = int(np.sum(zero))
    if nullity == 3:
        return True, 3
    if nullity == 2:
        return True, 2
    if nullity == 1:
        other = q[~zero]
        ok = bool(other[0] * other[1] < 0.0)
        return ok, 1 if ok else 0
    return False, 0


def _factorized_double_mu(
    M_a: np.ndarray,
    M_m: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    r"""Double-precision evaluation of the same metric equation.

    M_M = B_M^T B_M by Cholesky, therefore

        C^T M_M C = (B_M C)^T(B_M C)

    algebraically.  This route is useful as a fast candidate, but this test
    deliberately does *not* treat it as an independent oracle.
    """
    M_a = 0.5 * (np.asarray(M_a, float) + np.asarray(M_a, float).T)
    M_m = 0.5 * (np.asarray(M_m, float) + np.asarray(M_m, float).T)
    C = np.asarray(C, float)

    B_m = np.linalg.cholesky(M_m).T
    X = B_m @ C
    G = X.T @ X
    G = 0.5 * (G + G.T)
    mu, _ = eigh(G, M_a, check_finite=False)
    mu = np.sort(np.asarray(mu, float))
    if np.min(mu) <= 0:
        raise AssertionError(f"factorized double route produced non-positive mu={mu}")
    return mu


def _adaptive_certified_mu(
    M_a: np.ndarray,
    M_m: np.ndarray,
    C: np.ndarray,
    *,
    exact_tol: float,
) -> tuple[np.ndarray, bool]:
    """Prototype precision-escalation policy.

    A fast double route is used away from the scientific decision boundary.
    Whenever any generalized eigenvalue is within 1024 exact-tolerance units
    of mu=1, the exact-binary rational pencil is solved with certified real-root
    isolation.

    This is a *candidate architecture only*.  Production is not modified by
    this test.
    """
    mu_double = _factorized_double_mu(M_a, M_m, C)
    guard = 1024.0 * float(exact_tol)

    if float(np.min(np.abs(mu_double - 1.0))) <= guard:
        return _certified_generalized_mu(M_a, M_m, C), True

    return mu_double, False


def _hidden_basis_mu(case) -> np.ndarray:
    left = np.asarray(case.B_product, float) @ np.asarray(case.C_m_from_a, float)
    F = np.linalg.solve(np.asarray(case.B_parent, float).T, left.T).T
    return np.sort(np.linalg.svd(F, compute_uv=False)) ** 2


def _current_production_mu(case) -> tuple[np.ndarray, tuple[bool, int]]:
    result = evaluate(case, tol=DEFAULT_NUMERICAL_POLICY.exact_eigenvalue)
    return (
        np.sort(np.asarray(result.mu, float)),
        (
            bool(result.ct_analysis.exact_compatible),
            int(result.ct_analysis.degeneracy_order),
        ),
    )


def _evaluate_one(case, tol: float) -> dict:
    C = np.asarray(case.C_m_from_a, float)
    M_a = np.asarray(case.M_parent, float)
    M_m = np.asarray(case.M_product, float)

    certified = _certified_generalized_mu(M_a, M_m, C)
    factorized = _factorized_double_mu(M_a, M_m, C)
    adaptive, escalated = _adaptive_certified_mu(
        M_a, M_m, C, exact_tol=tol
    )
    current, current_class = _current_production_mu(case)
    hidden = _hidden_basis_mu(case)

    exact_class = _classify_mu(certified, tol)
    factorized_class = _classify_mu(factorized, tol)
    adaptive_class = _classify_mu(adaptive, tol)

    # Compare literal and factorized assembly only as a conditioning diagnostic.
    G_literal = C.T @ M_m @ C
    B_m = np.linalg.cholesky(0.5 * (M_m + M_m.T)).T
    X = B_m @ C
    G_factorized = X.T @ X
    assembly_difference = float(
        np.linalg.norm(G_literal - G_factorized, ord="fro")
        / max(
            np.linalg.norm(G_literal, ord="fro"),
            np.linalg.norm(G_factorized, ord="fro"),
            np.finfo(float).tiny,
        )
    )

    return {
        "certified_mu": certified,
        "factorized_mu": factorized,
        "adaptive_mu": adaptive,
        "current_mu": current,
        "hidden_mu": hidden,
        "certified_class": exact_class,
        "factorized_class": factorized_class,
        "adaptive_class": adaptive_class,
        "current_class": current_class,
        "adaptive_precision_escalated": escalated,
        "current_to_certified_max": float(np.max(np.abs(current - certified))),
        "factorized_to_certified_max": float(np.max(np.abs(factorized - certified))),
        "adaptive_to_certified_max": float(np.max(np.abs(adaptive - certified))),
        "hidden_to_certified_max": float(np.max(np.abs(hidden - certified))),
        "assembly_relative_difference": assembly_difference,
        "condition_Ma": float(np.linalg.cond(M_a)),
        "condition_Mm": float(np.linalg.cond(M_m)),
        "condition_C": float(np.linalg.cond(C)),
    }


def test_certified_binary64_error_ladder_and_precision_escalation_candidate():
    """Audit current production against a genuinely independent certified oracle.

    This replaces the previous Cholesky/SVD "metric-only oracle", which shared
    too much numerical machinery with the factorized candidate.

    Authoritative reference:
        exact IEEE-754 inputs -> exact rational characteristic polynomial ->
        certified real-root isolation.

    No production equation or implementation is changed here.
    """
    tol = float(DEFAULT_NUMERICAL_POLICY.exact_eigenvalue)

    totals = {
        "representations_checked": 0,
        "certified_base_rebase_class_shifts": 0,
        "current_production_class_mismatches": 0,
        "factorized_double_class_mismatches": 0,
        "adaptive_candidate_class_mismatches": 0,
        "adaptive_precision_escalations": 0,
    }
    worst = {
        "current_to_certified_max": 0.0,
        "factorized_to_certified_max": 0.0,
        "adaptive_to_certified_max": 0.0,
        "hidden_to_certified_max": 0.0,
        "assembly_relative_difference": 0.0,
    }

    current_examples = []
    factorized_examples = []
    adaptive_examples = []
    precision_shift_examples = []

    families = (
        ("exact", first_order_case, 300),
        ("incompatible", incompatible_case, 300),
    )

    for family, factory, count in families:
        for index in range(count):
            base = factory(f"certified-ladder-{family}", index)
            rebased = crystal_rebasis(
                base,
                f"certified-ladder-{family}-rebasis",
                index,
            )

            rows = [
                ("base", _evaluate_one(base, tol)),
                ("rebased", _evaluate_one(rebased, tol)),
            ]
            totals["representations_checked"] += 2

            if rows[0][1]["certified_class"] != rows[1][1]["certified_class"]:
                totals["certified_base_rebase_class_shifts"] += 1
                if len(precision_shift_examples) < 20:
                    precision_shift_examples.append(
                        {
                            "family": family,
                            "index": index,
                            "base": rows[0][1],
                            "rebased": rows[1][1],
                        }
                    )

            for representation, row in rows:
                if row["adaptive_precision_escalated"]:
                    totals["adaptive_precision_escalations"] += 1

                if row["current_class"] != row["certified_class"]:
                    totals["current_production_class_mismatches"] += 1
                    if len(current_examples) < 30:
                        current_examples.append(
                            {
                                "family": family,
                                "index": index,
                                "representation": representation,
                                **row,
                            }
                        )

                if row["factorized_class"] != row["certified_class"]:
                    totals["factorized_double_class_mismatches"] += 1
                    if len(factorized_examples) < 30:
                        factorized_examples.append(
                            {
                                "family": family,
                                "index": index,
                                "representation": representation,
                                **row,
                            }
                        )

                if row["adaptive_class"] != row["certified_class"]:
                    totals["adaptive_candidate_class_mismatches"] += 1
                    if len(adaptive_examples) < 30:
                        adaptive_examples.append(
                            {
                                "family": family,
                                "index": index,
                                "representation": representation,
                                **row,
                            }
                        )

                for key in worst:
                    worst[key] = max(worst[key], float(row[key]))

    report = {
        "schema_version": 3,
        "purpose": (
            "certified exact-binary64 generalized metric-pencil audit; "
            "scientific equations unchanged"
        ),
        "exact_tolerance": tol,
        "certification": {
            "float_conversion": "float.as_integer_ratio -> exact SymPy Rational",
            "equation": "det(C.T*M_M*C - mu*M_A) = 0",
            "root_method": "certified disjoint rational real-root intervals",
            "interval_decimal_digits": 35,
        },
        "adaptive_candidate": {
            "double_route": "(chol(M_M).T @ C)^T(...) generalized SPD eigenproblem",
            "precision_escalation_guard": "min|mu-1| <= 1024*exact_tolerance",
            "fallback": "certified exact-binary64 rational characteristic polynomial",
        },
        "totals": totals,
        "worst": worst,
        "current_production_failure_examples": current_examples,
        "factorized_double_failure_examples": factorized_examples,
        "adaptive_candidate_failure_examples": adaptive_examples,
        "certified_input_precision_shift_examples": precision_shift_examples,
    }

    out = Path("/tmp/engineered_representation_error_ladder.json")
    out.write_text(
        json.dumps(to_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # The adaptive candidate is allowed to proceed toward a production design
    # only if its classification agrees with the certified exact-binary oracle
    # for every tested representation.
    assert totals["adaptive_candidate_class_mismatches"] == 0, (
        "Adaptive precision candidate is not yet reliable. "
        f"Full certified audit: {out}\n"
        + json.dumps(to_jsonable(adaptive_examples[:8]), indent=2, sort_keys=True)
    )

    # Keep the current production defect visible until production is actually
    # repaired.  This assertion is expected to remain red at the present
    # checkpoint.
    assert totals["current_production_class_mismatches"] == 0, (
        "Current production disagrees with the certified exact-binary64 "
        "generalized metric pencil. Do not fix this with tolerance changes. "
        f"Full audit: {out}\n"
        + json.dumps(to_jsonable(current_examples[:8]), indent=2, sort_keys=True)
    )
