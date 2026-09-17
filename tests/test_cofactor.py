import numpy as np

from cualni_cryst.cofactor import cofactor_matrix, evaluate_cofactor_conditions


def test_cofactor_matrix_identity_property():
    A = np.array([[2.,1.,0.],[0.,3.,1.],[1.,0.,4.]])
    cof = cofactor_matrix(A)
    assert np.allclose(A.T @ cof, np.linalg.det(A) * np.eye(3))


def test_cc1_residual():
    U = np.diag([0.9,1.0,1.1])
    a = np.array([1.,0.,0.])
    n = np.array([0.,1.,0.])
    out = evaluate_cofactor_conditions(U, a, n)
    assert abs(out.cc1_residual) < 1e-12
