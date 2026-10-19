"""Exact passive rebasing: full metric, symmetry and physical tensor identities."""
import numpy as np
import sympy as sp
import pytest
from twin_app.p0_correspondence_rebasis import rebase_metric_correspondence


def test_nontrivial_unimodular_shear_and_mixed_rational_correspondence():
    G=np.array([[8.3,.23,.19],[.23,10.3,.57],[.19,.57,11.6]])
    C=sp.Matrix([[1,sp.Rational(1,3),0],[0,1,sp.Rational(1,5)],[0,0,1]])
    P=sp.Matrix([[1,1,0],[0,1,1],[0,0,1]])
    G2,C2,H2,audit=rebase_metric_correspondence(G,C,P,(sp.eye(3),-sp.eye(3)))
    assert audit.determinant==1 and audit.metric_residual<1e-13
    assert C2!=C and not np.allclose(G2,G)
    assert sp.simplify(C2-P.inv()*C)==sp.zeros(3)
    assert all(sp.simplify(h-P.inv()*g*P)==sp.zeros(3) for h,g in zip(H2,(sp.eye(3),-sp.eye(3))))
    # Direct basis and reciprocal-plane covariance: u'=P^-1 u, h'=P^T h.
    u=sp.Matrix([2,5,7]); h=sp.Matrix([1,3,8]); assert (P.T*h).dot(P.inv()*u)==h.dot(u)


def test_reject_nonunimodular_or_noninteger_basis():
    G=np.eye(3)
    with pytest.raises(ValueError,match='GL'):
        rebase_metric_correspondence(G,sp.eye(3),sp.diag(2,1,1),(sp.eye(3),))
    with pytest.raises(ValueError,match='INTEGER'):
        rebase_metric_correspondence(G,sp.eye(3),sp.diag(sp.Rational(1,2),1,1),(sp.eye(3),))


def test_reject_non_metric_preserving_product_operator():
    G=np.diag([1.,2.,3.])
    with pytest.raises(ValueError,match='violates'):
        rebase_metric_correspondence(G,sp.eye(3),sp.eye(3),(
            sp.Matrix([[0,1,0],[1,0,0],[0,0,1]]),
        ))
