from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from cualni_cryst.numerics import DEFAULT_NUMERICAL_POLICY
from validation.engineered.jsonutil import to_jsonable
from validation.engineered.physical import (
    crystal_rebasis,
    first_order_case,
    incompatible_case,
)
from validation.engineered.production import evaluate


def _canonical_basis(M: np.ndarray) -> np.ndarray:
    """Metric-only Cartesian realization B with B.T @ B = M."""
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)
    L = np.linalg.cholesky(M)  # M = L L.T
    return L.T


def _metric_only_lambdas(case) -> np.ndarray:
    """Independent oracle using only M_A, M_M, C -- no hidden generator bases."""
    B_a = _canonical_basis(case.M_parent)
    B_m = _canonical_basis(case.M_product)
    left = B_m @ np.asarray(case.C_m_from_a, dtype=float)
    F = np.linalg.solve(B_a.T, left.T).T
    return np.sort(np.linalg.svd(F, compute_uv=False))


def _hidden_basis_lambdas(case) -> np.ndarray:
    """Diagnostic only; hidden B_A/B_M contain more information than the metrics."""
    left = np.asarray(case.B_product, float) @ np.asarray(case.C_m_from_a, float)
    F = np.linalg.solve(np.asarray(case.B_parent, float).T, left.T).T
    return np.sort(np.linalg.svd(F, compute_uv=False))


def _factorized_same_equation_candidate_lambdas(case) -> np.ndarray:
    r"""Evaluate the SAME generalized metric equation through a safer assembly.

    The production equation is

        G = C.T @ M_M @ C,
        G v = mu M_A v.

    Since M_M = B_M.T @ B_M for any Cartesian realization B_M,

        C.T M_M C = (B_M C).T (B_M C)

    exactly in algebra.  Here B_M is obtained from a Cholesky factor of the
    metric input itself, not from the hidden generator basis.  No scientific
    equation is changed.
    """
    M_a = 0.5 * (np.asarray(case.M_parent, float) + np.asarray(case.M_parent, float).T)
    B_m = _canonical_basis(case.M_product)
    X = B_m @ np.asarray(case.C_m_from_a, dtype=float)
    G = X.T @ X
    G = 0.5 * (G + G.T)
    mu, _ = eigh(G, M_a, check_finite=False)
    mu = np.sort(np.asarray(mu, float))
    if np.min(mu) <= 0:
        raise AssertionError(f"factorized candidate produced non-positive mu={mu}")
    return np.sqrt(mu)


def _classify(lambdas: np.ndarray, tol: float) -> tuple[bool, int]:
    q = np.sort(np.asarray(lambdas, float)) ** 2 - 1.0
    zero = np.abs(q) <= tol
    nzero = int(np.sum(zero))
    if nzero == 3:
        return True, 3
    if nzero == 2:
        return True, 2
    if nzero == 1:
        other = q[~zero]
        ok = bool(other[0] * other[1] < 0.0)
        return ok, 1 if ok else 0
    return False, 0


def _evaluate_representation(case, tol: float) -> dict:
    metric_lam = _metric_only_lambdas(case)
    hidden_lam = _hidden_basis_lambdas(case)
    prod = evaluate(case, tol=tol)
    current_lam = np.sort(np.asarray(prod.lambdas, float))
    candidate_lam = _factorized_same_equation_candidate_lambdas(case)

    C = np.asarray(case.C_m_from_a, float)
    M_m = np.asarray(case.M_product, float)
    G_literal = C.T @ M_m @ C
    B_m = _canonical_basis(M_m)
    X = B_m @ C
    G_factorized = X.T @ X
    assembly_rel = float(
        np.linalg.norm(G_literal - G_factorized, ord="fro")
        / max(
            np.linalg.norm(G_literal, ord="fro"),
            np.linalg.norm(G_factorized, ord="fro"),
            np.finfo(float).tiny,
        )
    )

    return {
        "hidden_lambdas": hidden_lam,
        "metric_lambdas": metric_lam,
        "current_lambdas": current_lam,
        "candidate_lambdas": candidate_lam,
        "hidden_to_metric_max": float(np.max(np.abs(hidden_lam - metric_lam))),
        "current_to_metric_max": float(np.max(np.abs(current_lam - metric_lam))),
        "candidate_to_metric_max": float(np.max(np.abs(candidate_lam - metric_lam))),
        "metric_class": _classify(metric_lam, tol),
        "current_class": (
            bool(prod.ct_analysis.exact_compatible),
            int(prod.ct_analysis.degeneracy_order),
        ),
        "candidate_class": _classify(candidate_lam, tol),
        "assembly_relative_difference": assembly_rel,
        "condition_Ma": float(np.linalg.cond(case.M_parent)),
        "condition_Mm": float(np.linalg.cond(case.M_product)),
        "condition_C": float(np.linalg.cond(case.C_m_from_a)),
    }


def test_basis_change_error_ladder_uses_metric_only_oracle_and_same_equation_candidate():
    """Separate three distinct phenomena instead of conflating them.

    A. Hidden physical generator remains invariant.
    B. Finite-precision M_A/M_M/C may no longer encode exactly the same boundary
       problem after an extreme SL(3,Z) rebasing: INPUT PRECISION LIMIT.
    C. For the finite-precision metric problem actually supplied, production may
       disagree with an independent metric-only Cartesian oracle: PRODUCTION
       NUMERICAL FAILURE.

    The factorized candidate is evaluated only as a diagnostic implementation of
    the same C.T @ M_M @ C equation.  This test changes no production code.
    """
    tol = DEFAULT_NUMERICAL_POLICY.exact_eigenvalue

    totals = {
        "exact_cases": 0,
        "incompatible_cases": 0,
        "representations_checked": 0,
        "input_precision_class_shifts": 0,
        "current_production_class_mismatches": 0,
        "factorized_candidate_class_mismatches": 0,
    }
    precision_examples = []
    production_examples = []
    candidate_examples = []
    worst = {
        "hidden_to_metric_max": 0.0,
        "current_to_metric_max": 0.0,
        "candidate_to_metric_max": 0.0,
        "assembly_relative_difference": 0.0,
    }

    # Boundary-sensitive exact cases: enough deterministic samples to expose
    # basis/correspondence conditioning failures reproducibly.
    families = [
        ("exact", first_order_case, 300),
        ("incompatible", incompatible_case, 300),
    ]

    for family_name, factory, count in families:
        for i in range(count):
            base = factory(f"metric-ladder-{family_name}", i)
            rebased = crystal_rebasis(
                base,
                f"metric-ladder-{family_name}-rebasis",
                i,
            )
            b = _evaluate_representation(base, tol)
            r = _evaluate_representation(rebased, tol)

            totals[f"{family_name}_cases"] += 1
            totals["representations_checked"] += 2

            if b["metric_class"] != r["metric_class"]:
                totals["input_precision_class_shifts"] += 1
                if len(precision_examples) < 20:
                    precision_examples.append(
                        {"family": family_name, "index": i, "base": b, "rebased": r}
                    )

            for label, row in (("base", b), ("rebased", r)):
                if row["current_class"] != row["metric_class"]:
                    totals["current_production_class_mismatches"] += 1
                    if len(production_examples) < 30:
                        production_examples.append(
                            {
                                "family": family_name,
                                "index": i,
                                "representation": label,
                                **row,
                            }
                        )

                if row["candidate_class"] != row["metric_class"]:
                    totals["factorized_candidate_class_mismatches"] += 1
                    if len(candidate_examples) < 30:
                        candidate_examples.append(
                            {
                                "family": family_name,
                                "index": i,
                                "representation": label,
                                **row,
                            }
                        )

                for key in worst:
                    worst[key] = max(worst[key], float(row[key]))

    report = {
        "schema_version": 2,
        "purpose": "metric-only representation error ladder; production equations unchanged",
        "tolerance": float(tol),
        "totals": totals,
        "worst": worst,
        "input_precision_examples": precision_examples,
        "current_production_failure_examples": production_examples,
        "factorized_candidate_failure_examples": candidate_examples,
    }

    out = Path("/tmp/engineered_representation_error_ladder.json")
    out.write_text(
        json.dumps(to_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Candidate must first demonstrate that the equation-preserving numerical
    # route agrees with the independent metric-only oracle.  If this fails, no
    # production repair based on it is justified.
    assert totals["factorized_candidate_class_mismatches"] == 0, (
        "The proposed equation-preserving numerical route is not reliable. "
        f"See {out}; first examples:\n"
        + json.dumps(to_jsonable(candidate_examples[:5]), indent=2, sort_keys=True)
    )

    # Keep the existing production defect red until it is actually repaired.
    assert totals["current_production_class_mismatches"] == 0, (
        "Current production disagrees with the metric-only oracle even after "
        "input-encoding precision loss is separated out. This is a production "
        f"numerical defect, not a tolerance adjustment. Full report: {out}\n"
        + json.dumps(to_jsonable(production_examples[:8]), indent=2, sort_keys=True)
    )
