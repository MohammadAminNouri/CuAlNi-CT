from __future__ import annotations

"""Paper-direct CMC/SMC audit, not the backend CT calculator.

Expected values are transcribed from Cayron, Acta Materialia 316 (2026)
Table 3 and Fig.7; from Crystals 12 (2022) 130 Sec. 5.1/Table 2.
"""
import ast
import math
from pathlib import Path

import numpy as np
import pytest
from twin_app.paper_methodology_2022_2026 import cmc_smc_paper_audit


C=np.array([[0.,0.,1.],[.5,.5,0.],[-.5,.5,0.]])
A=np.eye(3)


def metric(a,b,c,beta):
    v=math.cos(math.radians(beta))
    return np.array([[a*a,0.,a*c*v],[0.,b*b,0.],[a*c*v,0.,c*c]])


def canonical(p):
    return np.asarray(p,float)/float(p[0])


def test_acta_2026_table3_independent_CMC_SMC_and_two_habits():
    M=metric(2.898/3.01,math.sqrt(2.),4.646/3.01,97.78)
    res=cmc_smc_paper_audit(A,M,C,tolerance=1e-9)
    assert res["degeneracy_order"]==1
    assert res["degeneracy_class"]=="FIRST_ORDER_DOUBLE_PLANE"
    assert res["cmc_inertia"]=={"negative":1,"near_zero":1,"positive":1}
    assert res["theoretical_habit_plane_count"]==2
    assert res["individual_AM_compatible"] is True
    assert res["full_AMM_supercompatibility_certified"] is False
    assert abs(res["lambda2_minus_one"]) < 1e-12
    habit=sorted(res["habit_planes"],
                 key=lambda x:canonical(x["habit_plane_parent_reciprocal"])[2],
                 reverse=True)
    np.testing.assert_allclose(
        canonical(habit[0]["habit_plane_parent_reciprocal"]),
        [1.,-1.,2.41966],atol=1.1e-5,rtol=0)
    np.testing.assert_allclose(
        canonical(habit[1]["habit_plane_parent_reciprocal"]),
        [1.,-1.,-.31568],atol=2.e-6,rtol=0)
    np.testing.assert_allclose(
        np.asarray(habit[0]["IPS_shear_parent_direct"])/
        habit[0]["habit_plane_parent_reciprocal"][0],
        [.36938,-.36938,-.05378],atol=1.e-5,rtol=0)


def test_paper_eq32_eq41_independently_derived_not_inverse_of_each_other():
    M=metric(2.898/3.01,math.sqrt(2.),4.646/3.01,97.78)
    got=cmc_smc_paper_audit(A,M,C)
    K=np.asarray(got["CMC_matrix"])
    S=np.asarray(got["SMC_matrix"])
    ci=np.linalg.inv(C)
    np.testing.assert_allclose(K,C.T@M@C-A,atol=1e-13)
    np.testing.assert_allclose(S,np.linalg.inv(A)-
                               ci@np.linalg.inv(M)@ci.T,atol=1e-13)
    assert not np.allclose(K@S,np.eye(3),atol=1e-8)


def test_paper_eq36_and_eq37_do_not_invent_habits_for_null_lines():
    no_zero=cmc_smc_paper_audit(A,np.diag([1.1**2,1.2**2,.95**2]),
                               np.eye(3))
    assert not no_zero["individual_AM_compatible"]
    assert not no_zero["habit_planes"]
    line=cmc_smc_paper_audit(A,np.diag([1.,1.1**2,1.2**2]),
                            np.eye(3))
    assert line["cmc_inertia"]=={"negative":0,"near_zero":1,"positive":2}
    assert line["degeneracy_class"]=="DEGENERATE_SEMIDEFINITE_QUADRATIC"
    assert line["individual_AM_compatible"] is False
    assert line["computed_habit_plane_count"]==0
    second=cmc_smc_paper_audit(A,np.diag([1.,1.,1.1**2]),np.eye(3))
    assert second["degeneracy_class"]=="SECOND_ORDER_SINGLE_PLANE"
    assert second["computed_habit_plane_count"]==1
    third=cmc_smc_paper_audit(A,A,np.eye(3))
    assert third["degeneracy_class"]=="THIRD_ORDER_IDENTITY_METRIC"
    assert third["theoretical_habit_plane_count"]=="nonunique_continuum"
    assert not third["habit_planes"]


def test_coherency_decisions_invariant_under_common_unit_conversion():
    M=metric(2.898/3.01,math.sqrt(2.),4.646/3.01,97.78)
    baseline=cmc_smc_paper_audit(A,M,C)
    for mult in (1.e-6,1.e+6):
        v=cmc_smc_paper_audit(A*mult**2,M*mult**2,C)
        np.testing.assert_allclose(v["principal_stretches"],
                                   baseline["principal_stretches"],atol=2e-12)
        assert v["degeneracy_class"]==baseline["degeneracy_class"]
        assert v["computed_habit_plane_count"]==2


def test_readonly_full_methodology_panel_has_paper_specific_outputs():
    root=Path(__file__).resolve().parents[2]
    module=(root/"twin_app"/"paper_methodology_2022_2026.py").read_text()
    ast.parse(module)
    for q in ("CMC_matrix","SMC_matrix","cmc_inertia","habit_planes",
              "source_doi","full_AMM_supercompatibility_certified"):
        assert q in module
