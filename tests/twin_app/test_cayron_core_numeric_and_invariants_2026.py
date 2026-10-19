"""Independent Cayron metric/basis and IEEE float-boundary contracts."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.ct import analyze_cmc, cmc, habit_planes_from_cmc, smc
from cualni_cryst.stretch import stretch_from_metrics, metric_native_stretch_spectrum
from cualni_cryst.ball_james_adapter import metric_native_stretch_spectrum as bj_spectrum
from twin_app.numeric_domain import validate_solver_correspondence

ROOT = Path(__file__).resolve().parents[2]


def _gram(cell):
    a,b,c = (float(cell[k]) for k in ("a","b","c"))
    alpha,beta,gamma = (math.radians(float(cell[k])) for k in ("alpha","beta","gamma"))
    return np.array([
        [a*a,a*b*math.cos(gamma),a*c*math.cos(beta)],
        [a*b*math.cos(gamma),b*b,b*c*math.cos(alpha)],
        [a*c*math.cos(beta),b*c*math.cos(alpha),c*c]],dtype=float)


def test_independent_exact_parent_and_product_basis_changes_preserve_C_metric():
    Ma=sp.Matrix([[9,1,0],[1,10,2],[0,2,12]])
    Mm=sp.Matrix([[11,0,sp.Rational(1,3)],[0,8,2],[sp.Rational(1,3),2,10]])
    C=sp.Matrix([[0,1,sp.Rational(1,2)],[1,0,0],[0,sp.Rational(1,3),1]])
    P=sp.Matrix([[1,1,0],[0,1,1],[0,0,1]])
    Q=sp.Matrix([[1,1,0],[0,1,-1],[0,0,1]])
    assert abs(P.det())==abs(Q.det())==1
    Cp=Q.inv()*C*P
    assert sp.simplify(Cp.T*(Q.T*Mm*Q)*Cp-P.T*(C.T*Mm*C)*P)==sp.zeros(3)
    u=sp.Matrix([2,3,5]); h=sp.Matrix([7,11,13])
    assert (P.T*h).dot(P.inv()*u)==h.dot(u)
    assert Cp*(P.inv()*u)==Q.inv()*C*u
    assert sp.simplify(Cp.inv().T*(P.T*h)-Q.T*C.inv().T*h)==sp.zeros(3)


@pytest.mark.parametrize("file",[
    "bhattacharya_niti_twin_habit_v1.json",
    "bhattacharya_cualni_orthorhombic_habit_v1.json",
])
def test_two_independent_metric_stretch_routes_and_volume_with_passive_rebasing(file):
    inputs=json.loads((ROOT/"data"/"benchmarks"/file).read_text())["input"]
    Ma,Mm=_gram(inputs["parent"]),_gram(inputs["product"])
    C=Correspondence(sp.Matrix(inputs["C_M_from_A"]))
    U=stretch_from_metrics(Ma,Mm,C)
    lambdas=np.linalg.eigvalsh(U)
    m=metric_native_stretch_spectrum(Ma,Mm,C).lambdas
    b=bj_spectrum(Ma,Mm,C.to_float64_checked()).lambdas
    np.testing.assert_allclose(lambdas,m,rtol=2e-8,atol=2e-9)
    np.testing.assert_allclose(lambdas,b,rtol=2e-8,atol=2e-9)
    volume=abs(float(C.C_M_from_A.det()))*math.sqrt(np.linalg.det(Mm)/np.linalg.det(Ma))
    assert np.isclose(np.prod(lambdas),volume,rtol=2e-8)
    P=sp.Matrix([[1,1,0],[0,1,0],[0,0,1]])
    Q=sp.Matrix([[1,0,0],[0,1,1],[0,0,1]])
    pf,qf=np.asarray(P,dtype=float),np.asarray(Q,dtype=float)
    Cp=Correspondence(Q.inv()*C.C_M_from_A*P)
    rebased=metric_native_stretch_spectrum(pf.T@Ma@pf,qf.T@Mm@qf,Cp).lambdas
    np.testing.assert_allclose(lambdas,rebased,rtol=2e-8,atol=2e-9)


def test_cayron_cmc_degenerate_quadratic_factors_into_actual_null_planes():
    Ma=np.eye(3); Mm=np.diag((1.3,1.,0.8)); C=Correspondence(sp.eye(3))
    info=analyze_cmc(Ma,Mm,C,tol=1e-10)
    assert info.exact_compatible and info.degeneracy_order==1
    planes=habit_planes_from_cmc(Ma,Mm,C,tol=1e-10)
    assert len(planes)==2
    Delta=cmc(Ma,Mm,C)
    for h in planes:
        n=np.asarray(h,dtype=float).reshape(3)
        u=np.array([n[2],0.,-n[0]])
        v=np.array([0.,1.,0.])
        assert np.linalg.norm(u)>0
        for x,y in ((u,u),(u,v),(v,v)):
            assert abs(float(x@Delta@y))<2e-9


def test_ct_no_exact_habit_without_cmc_degeneracy():
    assert habit_planes_from_cmc(np.eye(3),np.diag((1.3,1.1,0.8)),
                                 Correspondence(sp.eye(3)),tol=1e-10)==[]


def test_smc_is_independently_equal_to_inverse_metric_difference():
    Ma=np.array([[2.,.2,0],[.2,1.4,.15],[0,.15,1.8]])
    Mm=np.array([[1.4,.1,.2],[.1,1.7,.1],[.2,.1,1.6]])
    C=Correspondence(sp.Matrix([[1,sp.Rational(1,2),0],[0,1,0],[0,0,1]]))
    D=C.to_float64_checked()
    np.testing.assert_allclose(smc(Ma,Mm,C),np.linalg.inv(Ma)-np.linalg.inv(D.T@Mm@D),
                               rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize("matrix",[
    sp.eye(3)+sp.Rational(1,10**400)*sp.Matrix([[0,1,0],[0,0,0],[0,0,0]]),
    sp.diag(sp.Rational(1,10**400),1,1),
    sp.diag(sp.Integer(10)**400,1,1),
])
def test_every_core_entrypoint_rejects_silently_truncated_rational_C(matrix):
    C=Correspondence(matrix)
    for fn in (
        C.to_float64_checked,
        lambda: validate_solver_correspondence(matrix),
        lambda: C.pullback_product_metric(np.eye(3)),
        lambda: stretch_from_metrics(np.eye(3),np.eye(3),C),
        lambda: cmc(np.eye(3),np.eye(3),C),
    ):
        with pytest.raises(ValueError,match="binary64|underflows|overflows"):
            fn()

def test_nonzero_subnormal_fraction_with_large_relative_rounding_loss_fails_closed():
    C=Correspondence(sp.eye(3) + sp.Rational(1,10**320)*sp.Matrix([[0,1,0],[0,0,0],[0,0,0]]))
    # 1e-320 survives float conversion, yet is rounded by ~1.1e-5 relative.
    with pytest.raises(ValueError,match="loses too much precision"):
        C.to_float64_checked()
    with pytest.raises(ValueError,match="loses too much precision"):
        stretch_from_metrics(np.eye(3),np.eye(3),C)
    assert np.isclose(
        Correspondence(sp.diag(sp.Rational(1,3),1,1)).to_float64_checked()[0,0],1/3
    )
