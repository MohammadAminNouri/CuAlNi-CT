from __future__ import annotations

import numpy as np

from cualni_cryst.ball_james import mallard_law_twins
from validation.engineered.exact_oracle import (
    independent_cofactor_values,
    independent_rank_one_residual,
)
from validation.engineered.production import cofactor, ptmc_roots, numerical_rank_one
from validation.engineered.ptmc_oracle import independent_ptmc_roots
from validation.engineered.physical import random_rotation
from validation.engineered.rng import rng


def test_cofactor_raw_expressions_match_independent_formula_for_arbitrary_inputs():
    for i in range(300):
        g = rng("cofactor-arbitrary", i)
        Q = random_rotation(g)
        lam = g.uniform(0.65, 1.45, 3)
        U = Q @ np.diag(lam) @ Q.T
        a = g.normal(size=3)
        n = g.normal(size=3)
        independent = independent_cofactor_values(U,a,n)
        got = cofactor(U,a,n)
        np.testing.assert_allclose(
            [got.cc1_residual, got.cc2_residual, got.cc3_margin],
            independent,
            atol=2e-10, rtol=2e-10,
        )


def test_each_random_mallard_branch_is_independently_rank_one_and_optimizer_finds_existence():
    for i in range(30):
        g = rng("mallard-rankone", i)
        Q0 = random_rotation(g)
        U = Q0 @ np.diag(g.uniform(0.72,1.38,3)) @ Q0.T
        e = g.normal(size=3); e /= np.linalg.norm(e)
        Q2 = -np.eye(3) + 2*np.outer(e,e)
        V = Q2 @ U @ Q2
        twins = mallard_law_twins(U,V,e)
        assert twins
        for twin in twins:
            D = twin.rotation @ U - V
            assert independent_rank_one_residual(D) < 2e-7, (i,twin.kind)
            np.testing.assert_allclose(
                D, np.outer(twin.a,twin.n), atol=3e-7, rtol=3e-7
            )
        numerical = numerical_rank_one(U,V)
        assert numerical.residual < 2e-5, (i,numerical.residual)


def test_ptmc_production_roots_agree_with_scan_bracket_oracle_when_roots_exist():
    compared = 0
    for i in range(400):
        if compared >= 20:
            break
        g = rng("ptmc-differential", i)
        Q0 = random_rotation(g)
        U = Q0 @ np.diag(np.sort([
            g.uniform(0.7,0.93),
            g.uniform(0.85,1.08),
            g.uniform(1.08,1.35),
        ])) @ Q0.T
        e = g.normal(size=3); e /= np.linalg.norm(e)
        Q2 = -np.eye(3) + 2*np.outer(e,e)
        V = Q2 @ U @ Q2
        for twin in mallard_law_twins(U,V,e):
            independent = independent_ptmc_roots(U,twin.a,twin.n)
            if not independent:
                continue
            production = ptmc_roots(U,twin.a,twin.n)
            for r in production:
                assert any(abs(r-q) < 2e-4 for q in independent), (i,twin.kind,r,independent)
            for q in independent:
                assert any(abs(q-r) < 2e-4 for r in production), (i,twin.kind,q,production)
            compared += 1
            break
    assert compared >= 8
