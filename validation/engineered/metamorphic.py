from __future__ import annotations

import numpy as np

from cualni_cryst.numerics import DEFAULT_NUMERICAL_POLICY

from .budget import error_budget
from .compare import projective_family_distance, relative_matrix_residual
from .physical import crystal_rebasis, common_length_scale, reciprocal_swap
from .production import evaluate


def _direct_F(case):
    left = case.B_product @ case.C_m_from_a
    return np.linalg.solve(case.B_parent.T, left.T).T


def _direct_lambdas(case):
    return np.sort(np.linalg.svd(_direct_F(case), compute_uv=False))


def _classify_q(q, tol):
    q = np.sort(np.asarray(q,float))
    zero = np.abs(q) <= tol
    n = int(np.sum(zero))
    if n == 3:
        return True, 3
    if n == 2:
        return True, 2
    if n == 1:
        others = q[~zero]
        ok = bool(others[0]*others[1] < 0.0)
        return ok, (1 if ok else 0)
    return False, 0


def _physical_plane_normals(case, planes):
    out = []
    for p in planes:
        n = np.linalg.solve(case.B_parent.T, np.asarray(p, float))
        n /= np.linalg.norm(n)
        out.append(n)
    return out


def _stage_diagnostics(case, prod, tol):
    direct_lam = _direct_lambdas(case)
    generalized_q = np.sort(prod.mu - 1.0)
    whitened_q = np.sort(np.asarray(prod.ct_analysis.eigenvalues,float))
    direct_q = np.sort(direct_lam**2 - 1.0)
    direct_class = _classify_q(direct_q,tol)
    generalized_class = _classify_q(generalized_q,tol)
    whitened_class = (
        bool(prod.ct_analysis.exact_compatible),
        int(prod.ct_analysis.degeneracy_order),
    )
    return {
        "direct_cartesian_lambdas": direct_lam.tolist(),
        "direct_q": direct_q.tolist(),
        "generalized_q": generalized_q.tolist(),
        "whitened_q": whitened_q.tolist(),
        "direct_class": list(direct_class),
        "generalized_class": list(generalized_class),
        "whitened_class": list(whitened_class),
        "direct_lambda2_residual": float(abs(direct_lam[1]-1.0)),
        "generalized_lambda2_residual": float(abs(prod.lambdas[1]-1.0)),
        "whitened_nearest_zero": float(prod.ct_analysis.nearest_zero_residual),
        "generalized_vs_direct_lambda_max": float(np.max(np.abs(prod.lambdas-direct_lam))),
        "whitened_vs_generalized_q_max": float(np.max(np.abs(whitened_q-generalized_q))),
    }


def evaluate_metamorphic_family(case, index: int) -> list[dict]:
    tol = DEFAULT_NUMERICAL_POLICY.exact_eigenvalue
    base = evaluate(case, tol=tol)
    base_stage = _stage_diagnostics(case,base,tol)
    results = []

    rebased = crystal_rebasis(case, "metamorphic-rebasis", index)
    r = evaluate(rebased, tol=tol)
    re_stage = _stage_diagnostics(rebased,r,tol)

    Pa = np.linalg.solve(case.B_parent, rebased.B_parent)
    Pai = np.linalg.inv(Pa)

    base_planes = _physical_plane_normals(case, base.ct_planes_crystal)
    rebase_planes = _physical_plane_normals(rebased, r.ct_planes_crystal)
    plane_distance = projective_family_distance(base_planes, rebase_planes)

    results.append({
        "name": "independent_parent_product_SL3",
        "lambda_residual": float(np.max(np.abs(r.lambdas - base.lambdas))),
        "direct_cartesian_lambda_residual": float(
            np.max(np.abs(np.asarray(re_stage["direct_cartesian_lambdas"])-
                          np.asarray(base_stage["direct_cartesian_lambdas"])))
        ),
        "cmc_covariance_residual": relative_matrix_residual(
            r.cmc_dimensional, Pa.T @ base.cmc_dimensional @ Pa,
        ),
        "smc_covariance_residual": relative_matrix_residual(
            r.smc_dimensional, Pai @ base.smc_dimensional @ Pai.T,
        ),
        "physical_habit_family_distance": plane_distance,
        "classification_same": bool(
            r.ct_analysis.exact_compatible == base.ct_analysis.exact_compatible
            and r.ct_analysis.degeneracy_order == base.ct_analysis.degeneracy_order
        ),
        "direct_classification_same": base_stage["direct_class"] == re_stage["direct_class"],
        "generalized_classification_same": (
            base_stage["generalized_class"] == re_stage["generalized_class"]
        ),
        "stage_before": base_stage,
        "stage_after": re_stage,
        "default_exact_tolerance": tol,
        "base_budget": error_budget(case).to_dict(),
        "transformed_budget": error_budget(rebased).to_dict(),
        "condition_parent_before": case.condition_parent_basis,
        "condition_parent_after": rebased.condition_parent_basis,
        "condition_product_before": case.condition_product_basis,
        "condition_product_after": rebased.condition_product_basis,
        "condition_C_before": case.condition_correspondence,
        "condition_C_after": rebased.condition_correspondence,
    })

    for s in (1e-9,1e-6,1e-3,1.0,1e3,1e6,1e9):
        scaled = common_length_scale(case,s)
        q = evaluate(scaled,tol=tol)
        stage = _stage_diagnostics(scaled,q,tol)
        results.append({
            "name": f"common_length_scale_{s:g}",
            "lambda_residual": float(np.max(np.abs(q.lambdas-base.lambdas))),
            "classification_same": bool(
                q.ct_analysis.exact_compatible == base.ct_analysis.exact_compatible
                and q.ct_analysis.degeneracy_order == base.ct_analysis.degeneracy_order
            ),
            "direct_classification_same": stage["direct_class"] == base_stage["direct_class"],
            "generalized_classification_same": (
                stage["generalized_class"] == base_stage["generalized_class"]
            ),
            "cmc_dimensional_law": relative_matrix_residual(
                q.cmc_dimensional,(s*s)*base.cmc_dimensional
            ),
            "smc_dimensional_law": relative_matrix_residual(
                q.smc_dimensional,base.smc_dimensional/(s*s)
            ),
            "stage_before": base_stage,
            "stage_after": stage,
        })

    swapped=reciprocal_swap(case)
    sw=evaluate(swapped,tol=tol)
    target=np.sort(1.0/base.lambdas)
    results.append({
        "name":"parent_product_swap_inverse_spectrum",
        "lambda_residual":float(np.max(np.abs(sw.lambdas-target))),
        "classification_same":None,
        "truth_target":target.tolist(),
        "production":sw.lambdas.tolist(),
    })
    return results
