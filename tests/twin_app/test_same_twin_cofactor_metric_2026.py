from __future__ import annotations
"""Source-independent 2026 CCI/CCII cofactor-equivalent formula tests."""
import math
import numpy as np
import pytest
from twin_app.same_twin_cofactor_metric_2026 import cayron_generator_cc2_equivalent

C=np.array(((0.,0.,1.),(.5,.5,0.),(-.5,.5,0.)))


def metric(a,b,c,beta_deg):
    q=math.cos(math.radians(beta_deg))
    return np.array(((a*a,0.,a*c*q),(0.,b*b,0.),(a*c*q,0.,c*c)))


def test_published_o2_type_i_formula_fails_CC2_but_source_derived_correction_passes():
    beta=98.
    c=b=math.sqrt(2)
    a_printed=c/math.sin(math.radians(beta))
    a_equation_derived=1/math.sin(math.radians(beta))
    original=cayron_generator_cc2_equivalent(
        np.eye(3),metric(a_printed,b,c,beta),C,"I",(0,0,1))
    corrected=cayron_generator_cc2_equivalent(
        np.eye(3),metric(a_equation_derived,b,c,beta),C,"I",(0,0,1))
    assert original["squared_norm"]==pytest.approx(.5,abs=1e-12)
    assert original["squared_norm_minus_one"]==pytest.approx(-.5,abs=1e-12)
    assert corrected["squared_norm"]==pytest.approx(1.,abs=1e-12)
    assert corrected["absolute_squared_norm_gap"]<1e-12
    assert corrected["CC3_verified"] is False
    assert corrected["BJ_physical_branch_assignment_verified"] is False


def test_o2_type_ii_analytic_metric_has_unit_U_axis_not_inverse_U():
    beta=98.
    a=1.
    b=math.sqrt(2)
    c=math.sqrt(2)/math.sin(math.radians(beta))
    report=cayron_generator_cc2_equivalent(
        np.eye(3),metric(a,b,c,beta),C,"II",(0,0,1))
    assert report["squared_norm"]==pytest.approx(1.,abs=1e-12)
    assert report["criterion"].startswith("Type-II")
    assert abs(report["lambda2_minus_one"])<1e-12


def test_cc2_equivalent_invariant_under_metric_units_and_element_scale():
    beta=97.78
    M=metric(2.898/3.01,math.sqrt(2),4.646/3.01,beta)
    base=cayron_generator_cc2_equivalent(np.eye(3),M,C,"I",(0,0,1))
    for scale in (1e-4,1e4):
        compare=cayron_generator_cc2_equivalent(
            np.eye(3)*scale**2,M*scale**2,C,"I",(0,0,7))
        assert compare["squared_norm"]==pytest.approx(
            base["squared_norm"],abs=1e-10)


def test_wrong_crystal_family_or_singular_correspondence_fails_closed():
    with pytest.raises(ValueError):
        cayron_generator_cc2_equivalent(np.eye(3),np.eye(3),C,"III",(0,0,1))
    with pytest.raises(ValueError):
        cayron_generator_cc2_equivalent(np.eye(3),np.eye(3),
                                        np.zeros((3,3)),"I",(0,0,1))
