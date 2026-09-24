from __future__ import annotations

"""Metamorphic validation with explicit solver-vs-encoding error decomposition.

The core rule is that a representation transform is judged in three layers:

1. each encoded problem is checked independently against an exact/high-precision
   oracle for *that encoded binary64 input*;
2. the oracle measures how much the encoded problem itself moved under the
   requested transformation;
3. production cross-representation error is required to fit inside the
   triangle bound formed by oracle drift plus the two local production errors.

This prevents both false accusations (blaming the solver for binary64
re-encoding drift) and false passes (hiding a solver error behind a large,
fixed metamorphic tolerance).
"""

import numpy as np

from cualni_cryst.numerics import DEFAULT_NUMERICAL_POLICY

from .budget import error_budget
from .certified_binary64 import (
    certified_mu,
    classify_mu,
    cmc_literal_forward_error_bound,
    high_precision_cmc,
    high_precision_eigensystem,
    high_precision_smc,
)
from .compare import projective_family_distance, relative_matrix_residual
from .physical import crystal_rebasis, common_length_scale, reciprocal_swap
from .production import evaluate


_EPS = np.finfo(float).eps


def _fro(A) -> float:
    return float(np.linalg.norm(np.asarray(A, float), ord="fro"))


def _direct_F(case):
    left = case.B_product @ case.C_m_from_a
    return np.linalg.solve(case.B_parent.T, left.T).T


def _direct_lambdas(case):
    return np.sort(np.linalg.svd(_direct_F(case), compute_uv=False))


def _classify_q(q, tol):
    q = np.sort(np.asarray(q, float))
    zero = np.abs(q) <= tol
    n = int(np.sum(zero))
    if n == 3:
        return True, 3
    if n == 2:
        return True, 2
    if n == 1:
        others = q[~zero]
        ok = bool(others[0] * others[1] < 0.0)
        return ok, (1 if ok else 0)
    return False, 0


def _normalize_plane(p, M):
    p = np.asarray(p, float).reshape(3)
    q = float(p @ np.linalg.solve(np.asarray(M, float), p))
    if q <= 0.0:
        raise ArithmeticError(f"non-positive reciprocal plane norm: {q}")
    return p / np.sqrt(q)


def _oracle_planes(M_a, mu, V, tol):
    cls, order = classify_mu(mu, tol)
    if not cls or order == 3:
        return []
    eta = np.asarray(mu, float) - 1.0
    zero = np.abs(eta) <= tol
    M_a = 0.5 * (np.asarray(M_a, float) + np.asarray(M_a, float).T)
    V = np.asarray(V, float)
    if order == 2:
        k = int(np.where(~zero)[0][0])
        return [_normalize_plane(M_a @ V[:, k], M_a)]
    iz = int(np.where(zero)[0][0])
    i, j = [k for k in range(3) if k != iz]
    if eta[i] * eta[j] >= 0:
        return []
    ip, im = (i, j) if eta[i] > 0 else (j, i)
    p1 = M_a @ (np.sqrt(eta[ip]) * V[:, ip] + np.sqrt(-eta[im]) * V[:, im])
    p2 = M_a @ (np.sqrt(eta[ip]) * V[:, ip] - np.sqrt(-eta[im]) * V[:, im])
    return [_normalize_plane(p1, M_a), _normalize_plane(p2, M_a)]


def _physical_plane_normals(case, planes):
    out = []
    for p in planes:
        n = np.linalg.solve(case.B_parent.T, np.asarray(p, float))
        n /= np.linalg.norm(n)
        out.append(n)
    return out


def _safe_ratio(value: float, bound: float) -> float:
    value = float(value)
    bound = float(bound)
    if not np.isfinite(value):
        return float("inf")
    if bound <= 0.0:
        return 0.0 if value == 0.0 else float("inf")
    return value / bound


def _matrix_decomposition(
    prod_before,
    prod_after,
    oracle_before,
    oracle_after,
    transform,
) -> dict[str, float]:
    """Triangle decomposition for a linear matrix transformation."""
    P0 = np.asarray(prod_before, float)
    P1 = np.asarray(prod_after, float)
    O0 = np.asarray(oracle_before, float)
    O1 = np.asarray(oracle_after, float)

    TP0 = np.asarray(transform(P0), float)
    TO0 = np.asarray(transform(O0), float)
    TE0 = np.asarray(transform(P0 - O0), float)

    production_defect = _fro(P1 - TP0)
    oracle_drift = _fro(O1 - TO0)
    local_after = _fro(P1 - O1)
    transformed_local_before = _fro(TE0)

    scale = max(
        _fro(P1), _fro(TP0), _fro(O1), _fro(TO0),
        local_after, transformed_local_before, np.finfo(float).tiny
    )
    # Only protects evaluation of the diagnostic decomposition itself; it is
    # not a scientific tolerance.
    diagnostic_roundoff = 512.0 * _EPS * scale
    bound = (
        oracle_drift
        + local_after
        + transformed_local_before
        + diagnostic_roundoff
    )
    return {
        "production_defect_abs": production_defect,
        "oracle_encoding_drift_abs": oracle_drift,
        "local_after_abs": local_after,
        "transformed_local_before_abs": transformed_local_before,
        "diagnostic_roundoff_abs": diagnostic_roundoff,
        "decomposition_bound_abs": bound,
        "decomposition_ratio": _safe_ratio(production_defect, bound),
    }


def _spectrum_decomposition(prod0, prod1, oracle0, oracle1) -> dict[str, float]:
    p0 = np.sort(np.asarray(prod0, float))
    p1 = np.sort(np.asarray(prod1, float))
    o0 = np.sort(np.asarray(oracle0, float))
    o1 = np.sort(np.asarray(oracle1, float))
    production_delta = float(np.max(np.abs(p1 - p0)))
    oracle_delta = float(np.max(np.abs(o1 - o0)))
    local_before = float(np.max(np.abs(p0 - o0)))
    local_after = float(np.max(np.abs(p1 - o1)))
    scale = max(
        float(np.max(np.abs(p0))),
        float(np.max(np.abs(p1))),
        float(np.max(np.abs(o0))),
        float(np.max(np.abs(o1))),
        1.0,
    )
    diagnostic_roundoff = 64.0 * _EPS * scale
    bound = oracle_delta + local_before + local_after + diagnostic_roundoff
    return {
        "lambda_residual": production_delta,
        "oracle_lambda_residual": oracle_delta,
        "lambda_local_before_abs": local_before,
        "lambda_local_after_abs": local_after,
        "lambda_decomposition_bound": bound,
        "lambda_decomposition_ratio": _safe_ratio(production_delta, bound),
    }


def _habit_decomposition(prod_before, prod_after, oracle_before, oracle_after):
    production_delta = projective_family_distance(prod_before, prod_after)
    oracle_delta = projective_family_distance(oracle_before, oracle_after)
    local_before = projective_family_distance(prod_before, oracle_before)
    local_after = projective_family_distance(prod_after, oracle_after)

    if not all(np.isfinite(x) for x in (oracle_delta, local_before, local_after)):
        return {
            "physical_habit_family_distance": float(production_delta),
            "oracle_physical_habit_family_distance": float(oracle_delta),
            "habit_local_before_distance": float(local_before),
            "habit_local_after_distance": float(local_after),
            "habit_cross_test_applicable": False,
            "habit_decomposition_bound": float("inf"),
            "habit_decomposition_ratio": 0.0,
        }

    bound = oracle_delta + local_before + local_after + 64.0 * _EPS
    return {
        "physical_habit_family_distance": float(production_delta),
        "oracle_physical_habit_family_distance": float(oracle_delta),
        "habit_local_before_distance": float(local_before),
        "habit_local_after_distance": float(local_after),
        "habit_cross_test_applicable": True,
        "habit_decomposition_bound": float(bound),
        "habit_decomposition_ratio": _safe_ratio(production_delta, bound),
    }


def _stage_diagnostics(case, prod, tol):
    direct_lam = _direct_lambdas(case)
    direct_q = np.sort(direct_lam**2 - 1.0)
    direct_class = _classify_q(direct_q, tol)

    certified = certified_mu(case.M_parent, case.M_product, case.C_m_from_a)
    certified_class = classify_mu(certified, tol)
    hp_mu, hp_V = high_precision_eigensystem(
        case.M_parent, case.M_product, case.C_m_from_a, digits=110
    )
    oracle_planes = _oracle_planes(case.M_parent, certified, hp_V, tol)
    prod_planes = list(prod.ct_planes_crystal)

    prod_phys = _physical_plane_normals(case, prod_planes)
    oracle_phys = _physical_plane_normals(case, oracle_planes)
    habit_oracle_distance = projective_family_distance(prod_phys, oracle_phys)

    cmc_oracle = high_precision_cmc(
        case.M_parent, case.M_product, case.C_m_from_a, digits=110
    )
    smc_oracle = high_precision_smc(
        case.M_parent, case.M_product, case.C_m_from_a, digits=110
    )

    production_class = (
        bool(prod.ct_analysis.exact_compatible),
        int(prod.ct_analysis.degeneracy_order),
    )
    generalized_q = np.sort(np.asarray(prod.mu, float) - 1.0)

    cmc_abs_error = _fro(np.asarray(prod.cmc_dimensional) - cmc_oracle)
    cmc_bound = cmc_literal_forward_error_bound(
        case.M_parent, case.M_product, case.C_m_from_a
    )
    # One extra factor of two covers the final BLAS/storage evaluation details
    # not represented in the textbook dot-product model; this is still a
    # theorem-derived machine-roundoff envelope, not a fitted acceptance limit.
    cmc_bound *= 2.0

    smc_abs_error = _fro(np.asarray(prod.smc_dimensional) - smc_oracle)
    smc_scale = max(
        _fro(prod.smc_dimensional), _fro(smc_oracle), np.finfo(float).tiny
    )

    prod_lam = np.sqrt(np.asarray(prod.mu, float))
    cert_lam = np.sqrt(np.asarray(certified, float))

    return {
        "direct_cartesian_lambdas": direct_lam.tolist(),
        "direct_q": direct_q.tolist(),
        "direct_class": list(direct_class),
        "production_mu": np.asarray(prod.mu, float).tolist(),
        "production_lambdas": prod_lam.tolist(),
        "production_q": generalized_q.tolist(),
        "production_class": list(production_class),
        "certified_mu": certified.tolist(),
        "certified_lambdas": cert_lam.tolist(),
        "certified_class": list(certified_class),
        "high_precision_mu": hp_mu.tolist(),
        "production_matches_certified": production_class == certified_class,
        "production_certified_mu_residual": float(
            np.max(np.abs(np.asarray(prod.mu, float) - certified))
        ),
        "production_certified_lambda_residual": float(
            np.max(np.abs(prod_lam - cert_lam))
        ),
        "high_precision_certified_mu_residual": float(np.max(np.abs(hp_mu - certified))),
        "cmc_oracle_residual": relative_matrix_residual(prod.cmc_dimensional, cmc_oracle),
        "cmc_oracle_abs_error": cmc_abs_error,
        "cmc_literal_forward_error_bound_abs": cmc_bound,
        "cmc_forward_error_ratio": _safe_ratio(cmc_abs_error, cmc_bound),
        "smc_oracle_residual": smc_abs_error / smc_scale,
        "smc_oracle_abs_error": smc_abs_error,
        "habit_oracle_distance": float(habit_oracle_distance),
        "oracle_planes_crystal": [p.tolist() for p in oracle_planes],
        "production_planes_crystal": [np.asarray(p, float).tolist() for p in prod_planes],
        "direct_lambda2_residual": float(abs(direct_lam[1] - 1.0)),
        "production_lambda2_residual": float(abs(np.sqrt(prod.mu[1]) - 1.0)),
        "_cmc_oracle_matrix": cmc_oracle,
        "_smc_oracle_matrix": smc_oracle,
        "_production_physical_planes": prod_phys,
        "_oracle_physical_planes": oracle_phys,
    }


def _public_stage(stage: dict) -> dict:
    """Drop internal ndarray helpers before JSON serialization."""
    return {k: v for k, v in stage.items() if not k.startswith("_")}


def evaluate_metamorphic_family(case, index: int) -> list[dict]:
    tol = DEFAULT_NUMERICAL_POLICY.exact_eigenvalue
    base = evaluate(case, tol=tol)
    base_stage = _stage_diagnostics(case, base, tol)
    results = []

    rebased = crystal_rebasis(case, "metamorphic-rebasis", index)
    r = evaluate(rebased, tol=tol)
    re_stage = _stage_diagnostics(rebased, r, tol)

    Pa = np.linalg.solve(case.B_parent, rebased.B_parent)
    Pai = np.linalg.inv(Pa)

    spectrum = _spectrum_decomposition(
        np.sqrt(np.asarray(base.mu, float)),
        np.sqrt(np.asarray(r.mu, float)),
        np.sqrt(np.asarray(base_stage["certified_mu"], float)),
        np.sqrt(np.asarray(re_stage["certified_mu"], float)),
    )
    cmc_dec = _matrix_decomposition(
        base.cmc_dimensional,
        r.cmc_dimensional,
        base_stage["_cmc_oracle_matrix"],
        re_stage["_cmc_oracle_matrix"],
        lambda X: Pa.T @ X @ Pa,
    )
    smc_dec = _matrix_decomposition(
        base.smc_dimensional,
        r.smc_dimensional,
        base_stage["_smc_oracle_matrix"],
        re_stage["_smc_oracle_matrix"],
        lambda X: Pai @ X @ Pai.T,
    )
    habit_dec = _habit_decomposition(
        base_stage["_production_physical_planes"],
        re_stage["_production_physical_planes"],
        base_stage["_oracle_physical_planes"],
        re_stage["_oracle_physical_planes"],
    )

    results.append(
        {
            "name": "independent_parent_product_SL3",
            **spectrum,
            "direct_cartesian_lambda_residual": float(
                np.max(
                    np.abs(
                        np.asarray(re_stage["direct_cartesian_lambdas"])
                        - np.asarray(base_stage["direct_cartesian_lambdas"])
                    )
                )
            ),
            "cmc_covariance_residual": relative_matrix_residual(
                r.cmc_dimensional, Pa.T @ base.cmc_dimensional @ Pa
            ),
            "smc_covariance_residual": relative_matrix_residual(
                r.smc_dimensional, Pai @ base.smc_dimensional @ Pai.T
            ),
            "oracle_cmc_covariance_residual": relative_matrix_residual(
                re_stage["_cmc_oracle_matrix"],
                Pa.T @ base_stage["_cmc_oracle_matrix"] @ Pa,
            ),
            "oracle_smc_covariance_residual": relative_matrix_residual(
                re_stage["_smc_oracle_matrix"],
                Pai @ base_stage["_smc_oracle_matrix"] @ Pai.T,
            ),
            "cmc_decomposition": cmc_dec,
            "smc_decomposition": smc_dec,
            **habit_dec,
            "classification_same": base_stage["production_class"]
            == re_stage["production_class"],
            "certified_classification_same": base_stage["certified_class"]
            == re_stage["certified_class"],
            "encoded_problem_shifted_class": base_stage["certified_class"]
            != re_stage["certified_class"],
            "stage_before": _public_stage(base_stage),
            "stage_after": _public_stage(re_stage),
            "default_exact_tolerance": tol,
            "base_budget": error_budget(case).to_dict(),
            "transformed_budget": error_budget(rebased).to_dict(),
            "condition_parent_before": case.condition_parent_basis,
            "condition_parent_after": rebased.condition_parent_basis,
            "condition_product_before": case.condition_product_basis,
            "condition_product_after": rebased.condition_product_basis,
            "condition_C_before": case.condition_correspondence,
            "condition_C_after": rebased.condition_correspondence,
        }
    )

    for s in (1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9):
        scaled = common_length_scale(case, s)
        q = evaluate(scaled, tol=tol)
        stage = _stage_diagnostics(scaled, q, tol)

        spectrum = _spectrum_decomposition(
            np.sqrt(np.asarray(base.mu, float)),
            np.sqrt(np.asarray(q.mu, float)),
            np.sqrt(np.asarray(base_stage["certified_mu"], float)),
            np.sqrt(np.asarray(stage["certified_mu"], float)),
        )
        cmc_dec = _matrix_decomposition(
            base.cmc_dimensional,
            q.cmc_dimensional,
            base_stage["_cmc_oracle_matrix"],
            stage["_cmc_oracle_matrix"],
            lambda X, ss=s: (ss * ss) * X,
        )
        smc_dec = _matrix_decomposition(
            base.smc_dimensional,
            q.smc_dimensional,
            base_stage["_smc_oracle_matrix"],
            stage["_smc_oracle_matrix"],
            lambda X, ss=s: X / (ss * ss),
        )

        results.append(
            {
                "name": f"common_length_scale_{s:g}",
                **spectrum,
                "classification_same": stage["production_class"]
                == base_stage["production_class"],
                "certified_classification_same": stage["certified_class"]
                == base_stage["certified_class"],
                "cmc_dimensional_law": relative_matrix_residual(
                    q.cmc_dimensional, (s * s) * base.cmc_dimensional
                ),
                "smc_dimensional_law": relative_matrix_residual(
                    q.smc_dimensional, base.smc_dimensional / (s * s)
                ),
                "cmc_decomposition": cmc_dec,
                "smc_decomposition": smc_dec,
                "stage_before": _public_stage(base_stage),
                "stage_after": _public_stage(stage),
            }
        )

    swapped = reciprocal_swap(case)
    sw = evaluate(swapped, tol=tol)
    sw_stage = _stage_diagnostics(swapped, sw, tol)
    truth_target = np.sort(1.0 / base.lambdas)
    oracle_swap_lam = np.sqrt(np.asarray(sw_stage["certified_mu"], float))
    production_swap_lam = np.sort(np.asarray(sw.lambdas, float))
    oracle_target_residual = float(np.max(np.abs(oracle_swap_lam - truth_target)))
    local_prod_oracle = float(np.max(np.abs(production_swap_lam - oracle_swap_lam)))
    production_target_residual = float(
        np.max(np.abs(production_swap_lam - truth_target))
    )
    reciprocal_bound = (
        oracle_target_residual
        + local_prod_oracle
        + 64.0 * _EPS * max(float(np.max(np.abs(truth_target))), 1.0)
    )
    results.append(
        {
            "name": "parent_product_swap_inverse_spectrum",
            "lambda_residual": production_target_residual,
            "oracle_lambda_residual": oracle_target_residual,
            "reciprocal_local_production_oracle_error": local_prod_oracle,
            "reciprocal_spectrum_decomposition_bound": reciprocal_bound,
            "reciprocal_spectrum_decomposition_ratio": _safe_ratio(
                production_target_residual, reciprocal_bound
            ),
            "production_certified_mu_residual": sw_stage[
                "production_certified_mu_residual"
            ],
            "production_matches_certified": sw_stage["production_matches_certified"],
            "truth_target": truth_target.tolist(),
            "oracle_target": oracle_swap_lam.tolist(),
            "production": production_swap_lam.tolist(),
            "stage_after": _public_stage(sw_stage),
        }
    )
    return results
