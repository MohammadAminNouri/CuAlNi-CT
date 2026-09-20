import numpy as np

from cualni_cryst.cofactor import evaluate_cofactor_conditions


def test_cc2_cc3_are_invariant_to_rank_one_factor_rescaling():
    U = np.diag([0.9, 1.0, 1.1])
    a = np.array([0.1, 0.2, 0.3])
    n = np.array([0.4, -0.5, 0.6])

    r1 = evaluate_cofactor_conditions(U, a, n)
    kappa = 7.3
    r2 = evaluate_cofactor_conditions(U, kappa * a, n / kappa)

    assert np.isclose(r1.cc2_residual, r2.cc2_residual, atol=1e-12, rtol=1e-12)
    assert np.isclose(r1.cc3_margin, r2.cc3_margin, atol=1e-12, rtol=1e-12)
