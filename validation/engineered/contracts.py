from __future__ import annotations

import numpy as np

from cualni_cryst.numerics import DEFAULT_NUMERICAL_POLICY

from .budget import error_budget
from .compare import projective_family_distance, relative_matrix_residual
from .exact_oracle import physical_ball_james_habit_normals
from .model import ContractResult, FailureClass, CaseEvaluation, CrystalEncoding
from .production import evaluate


def _symmetric_frame_rotation(case: CrystalEncoding) -> np.ndarray:
    w, V = np.linalg.eigh(case.M_parent)
    S = (V * np.sqrt(w)) @ V.T
    Q = S @ np.linalg.inv(case.B_parent)
    return Q


def _ct_planes_to_symmetric_normals(case: CrystalEncoding, planes: tuple[np.ndarray, ...]) -> list[np.ndarray]:
    w, V = np.linalg.eigh(case.M_parent)
    S = (V * np.sqrt(w)) @ V.T
    out = []
    for p in planes:
        g = np.linalg.solve(S, np.asarray(p, float))
        g /= np.linalg.norm(g)
        out.append(g)
    return out


def _direct_encoded_F(case: CrystalEncoding) -> np.ndarray:
    # F B_A = B_M C  =>  F = B_M C B_A^-1.
    # solve(B_A.T, (...).T).T avoids explicitly inverting B_A.
    left = case.B_product @ case.C_m_from_a
    return np.linalg.solve(case.B_parent.T, left.T).T


def evaluate_case(
    case: CrystalEncoding,
    *,
    tol: float = DEFAULT_NUMERICAL_POLICY.exact_eigenvalue,
) -> CaseEvaluation:
    budget = error_budget(case)
    prod = evaluate(case, tol=tol)
    contracts: list[ContractResult] = []

    # Stage-0 independent physical encoding check.
    F_encoded = _direct_encoded_F(case)
    direct_lam = np.sort(np.linalg.svd(F_encoded, compute_uv=False))
    intended_res = float(np.max(np.abs(direct_lam - case.truth.lambdas)))
    contracts.append(ContractResult(
        "encoded_cartesian_kinematics_vs_intended_truth",
        intended_res <= max(budget.allowed, tol),
        intended_res, max(budget.allowed, tol),
        FailureClass.NUMERICAL_BACKWARD_ERROR,
        {
            "intended": case.truth.lambdas.tolist(),
            "encoded_cartesian": direct_lam.tolist(),
        },
    ))

    # Stage-1 production generalized metric spectrum.
    lambda_res = float(np.max(np.abs(np.sort(prod.lambdas) - direct_lam)))
    contracts.append(ContractResult(
        "generalized_metric_stretches_vs_direct_encoded_cartesian",
        lambda_res <= max(budget.allowed, tol),
        lambda_res, max(budget.allowed, tol),
        FailureClass.ORACLE_MISMATCH,
        {
            "direct_encoded_cartesian": direct_lam.tolist(),
            "production": prod.lambdas.tolist(),
        },
    ))

    expected_G = prod.stretch @ prod.stretch
    metric_res = relative_matrix_residual(prod.normalized_metric, expected_G)
    contracts.append(ContractResult(
        "normalized_metric_equals_U2",
        metric_res <= budget.allowed,
        metric_res, budget.allowed,
        FailureClass.THEORY_EQUIVALENCE_VIOLATION,
        {},
    ))

    # Expected discrete classification is determined from the *encoded* floating
    # problem, not blindly from the ideal generator.  This prevents the
    # validator itself from calling a representation exact after its own
    # roundoff moved lambda2 outside the application's declared tolerance.
    q_direct = direct_lam**2 - 1.0
    zero = np.abs(q_direct) <= tol
    if int(np.sum(zero)) == 3:
        expected_exact, expected_order = True, 3
    elif int(np.sum(zero)) == 2:
        expected_exact, expected_order = True, 2
    elif int(np.sum(zero)) == 1:
        others = q_direct[~zero]
        expected_exact = bool(others[0] * others[1] < 0.0)
        expected_order = 1 if expected_exact else 0
    else:
        expected_exact, expected_order = False, 0

    class_ok = bool(prod.ct_analysis.exact_compatible == expected_exact)
    order_ok = bool(prod.ct_analysis.degeneracy_order == expected_order)
    contracts.append(ContractResult(
        "ct_classification_vs_direct_encoded_cartesian_problem",
        class_ok and order_ok,
        0.0 if (class_ok and order_ok) else 1.0, 0.0,
        FailureClass.THEORY_EQUIVALENCE_VIOLATION,
        {
            "expected_exact_from_encoded_F": expected_exact,
            "got_exact": bool(prod.ct_analysis.exact_compatible),
            "expected_order_from_encoded_F": expected_order,
            "got_order": int(prod.ct_analysis.degeneracy_order),
            "direct_q": q_direct.tolist(),
            "ct_eigenvalues": np.asarray(prod.ct_analysis.eigenvalues).tolist(),
            "tol": tol,
        },
    ))

    if expected_order == 1:
        Qframe = _symmetric_frame_rotation(case)
        oracle_phys = physical_ball_james_habit_normals(
            # Use the encoded physical U, not the ideal generator U.
            # Polar U from F_encoded is constructed independently here.
            (lambda F: (
                (lambda w,V: (V*np.sqrt(w))@V.T)(*np.linalg.eigh(F.T@F))
            ))(F_encoded)
        )
        oracle_hat = [Qframe @ n for n in oracle_phys]
        ct_hat = _ct_planes_to_symmetric_normals(case, prod.ct_planes_crystal)
        d_ct = projective_family_distance(ct_hat, oracle_hat)
        contracts.append(ContractResult(
            "ct_habit_planes_vs_independent_encoded_cartesian_ball_james_oracle",
            d_ct <= max(25.0 * budget.allowed, 2e-7),
            d_ct, max(25.0 * budget.allowed, 2e-7),
            FailureClass.THEORY_EQUIVALENCE_VIOLATION,
            {"ct_count": len(ct_hat), "oracle_count": len(oracle_hat)},
        ))

        bj_hat = [np.asarray(sol.n, float) for sol in prod.bj_solutions]
        d_bj = projective_family_distance(bj_hat, oracle_hat)
        contracts.append(ContractResult(
            "production_ball_james_vs_independent_encoded_cartesian_oracle",
            d_bj <= max(25.0 * budget.allowed, 2e-7),
            d_bj, max(25.0 * budget.allowed, 2e-7),
            FailureClass.ORACLE_MISMATCH,
            {"solution_count": len(bj_hat)},
        ))

        rank_res = max([float(sol.residual) for sol in prod.bj_solutions], default=float("inf"))
        contracts.append(ContractResult(
            "analytical_rank_one_residual",
            rank_res <= max(50.0 * budget.allowed, 5e-7),
            rank_res, max(50.0 * budget.allowed, 5e-7),
            FailureClass.NUMERICAL_BACKWARD_ERROR,
            {},
        ))

    diagnostics = {
        "budget": budget.to_dict(),
        "conditions": {
            "B_parent": case.condition_parent_basis,
            "B_product": case.condition_product_basis,
            "C": case.condition_correspondence,
            "metrics": case.condition_metric_pair,
        },
        "intended_lambdas": case.truth.lambdas.tolist(),
        "direct_encoded_cartesian_lambdas": direct_lam.tolist(),
        "production_lambdas": prod.lambdas.tolist(),
        "ct_eigenvalues": np.asarray(prod.ct_analysis.eigenvalues).tolist(),
        "tol": tol,
    }
    return CaseEvaluation(case.case_id, tuple(contracts), diagnostics)
