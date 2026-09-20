import numpy as np
import sympy as sp

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.ct import analyze_cmc, habit_planes_from_cmc

C_ID = Correspondence(sp.eye(3), label="identity synthetic regression")
M_A = np.eye(3)


def test_cmc_first_order_degeneracy_two_planes():
    M_M = np.diag([1.0, 4.0, 0.25])
    ana = analyze_cmc(M_A, M_M, C_ID, tol=1e-12)
    assert ana.exact_compatible
    assert ana.degeneracy_order == 1
    assert ana.inertia == (1, 1, 1)
    assert len(habit_planes_from_cmc(M_A, M_M, C_ID, tol=1e-12)) == 2


def test_cmc_second_order_degeneracy_one_plane():
    M_M = np.diag([1.0, 1.0, 4.0])
    ana = analyze_cmc(M_A, M_M, C_ID, tol=1e-12)
    assert ana.exact_compatible
    assert ana.degeneracy_order == 2
    assert ana.inertia == (0, 2, 1)
    assert len(habit_planes_from_cmc(M_A, M_M, C_ID, tol=1e-12)) == 1


def test_cmc_third_order_degeneracy_all_space():
    ana = analyze_cmc(M_A, M_A, C_ID, tol=1e-12)
    assert ana.exact_compatible
    assert ana.degeneracy_order == 3
    assert ana.inertia == (0, 3, 0)
    assert habit_planes_from_cmc(M_A, M_A, C_ID, tol=1e-12) == []


def test_zero_eigenvalue_with_same_sign_remainder_is_not_ips():
    M_M = np.diag([1.0, 4.0, 9.0])
    ana = analyze_cmc(M_A, M_M, C_ID, tol=1e-12)
    assert not ana.exact_compatible
    assert ana.degeneracy_order == 0
    assert ana.inertia == (0, 1, 2)
