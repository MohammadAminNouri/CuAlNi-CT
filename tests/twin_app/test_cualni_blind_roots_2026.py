"""Independent randomized rank-one/habit-fraction falsification.

Generate eigenvalue-crossing problems from first principles; compute the
compatibility-root reference by an independently bracketed determinant,
then ask the backend's high-precision root oracle to find ALL fractions.
NO CuAlNi twin/habit answers, test-case root arrays or benchmark values
are injected into the scientific solver.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq
from twin_app.p1_numerics import root_oracle


def make_case(seed):
    rng=np.random.default_rng(seed)
    # Physical compatibility mathematics is valid irrespective of which
    # martensite provided the stretch; these are controlled synthetic tests.
    short=rng.uniform(.66,.85)
    middle=rng.uniform(.88,.97)
    long=rng.uniform(1.12,1.38)
    increment=rng.uniform(1-short+.07,.75)
    Z=rng.normal(size=(3,3))
    Q,R=np.linalg.qr(Z)
    Q=Q@np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q)<0:Q[:,0]*=-1
    U=Q@np.diag([short,middle,long])@Q.T
    a=increment*Q[:,0]
    n=Q[:,0]
    return U,a,n


def det_f(U,a,n,fraction):
    F=U+fraction*np.outer(a,n)
    return float(np.linalg.det(F.T@F-np.eye(3)))


@pytest.mark.parametrize("seed",range(20))
def test_independent_bracketed_fraction_matches_complete_oracle(seed):
    U,a,n=make_case(1000+seed)
    g0=det_f(U,a,n,0)
    g1=det_f(U,a,n,1)
    assert g0*g1<0
    root=brentq(lambda x:det_f(U,a,n,x),0.,1.,xtol=1e-13)
    sv=np.linalg.svd(U+root*np.outer(a,n),compute_uv=False)
    assert abs(sv[1]-1)<1e-10
    oracle=root_oracle(U,a,n,[root])
    assert oracle["status"]=="PASS_INTERIOR_ROOT_COVERAGE"
    assert len(oracle["roots"])==1
    assert abs(oracle["roots"][0]-root)<1e-8
    assert not oracle["unmatched_oracle"] and not oracle["unmatched_reported"]


@pytest.mark.parametrize("seed",range(5))
def test_rank_one_factorization_scale_invariance(seed):
    U,a,n=make_case(2000+seed)
    original=root_oracle(U,a,n)
    for scale in (1e-4,1e-2,7.,1e4):
        modified=root_oracle(U,scale*a,n/scale)
        assert len(modified["roots"])==len(original["roots"])
        assert np.allclose(modified["roots"],original["roots"],rtol=0,atol=5e-8)


@pytest.mark.parametrize("seed",range(5))
def test_unreported_compatibility_is_not_marked_passed(seed):
    U,a,n=make_case(3000+seed)
    missing=root_oracle(U,a,n,())
    assert missing["status"]=="MISMATCH"
    assert len(missing["unmatched_oracle"])==1
    wrong=root_oracle(U,a,n,[0.])
    assert wrong["status"]=="MISMATCH"
    assert wrong["unmatched_oracle"] and wrong["unmatched_reported"]


def test_near_endpoint_root_is_not_silently_certified_as_generic_interior():
    U=np.diag((.8,.9,1.2))
    a=np.array([.2/(1-2e-8),0,0])
    n=np.array([1.,0.,0.])
    oracle=root_oracle(U,a,n)
    assert oracle["status"] in ("NEEDS_BOUNDARY_INVESTIGATION","MISMATCH")
    assert oracle.get("status")!="PASS_INTERIOR_ROOT_COVERAGE"


def test_identically_zero_rank_one_polynomial_is_indeterminate_not_passed():
    U=np.eye(3)
    oracle=root_oracle(U,[.2,0,0],[0,1,0],())
    assert oracle["status"]=="INCONCLUSIVE_CONTINUUM_OR_NEAR_ZERO_POLYNOMIAL"
