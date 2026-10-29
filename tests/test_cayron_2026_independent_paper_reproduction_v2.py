from __future__ import annotations
"""Blind, independent Cayron-2026 source-value acceptance criteria."""
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.ct import habit_planes_from_cmc, ips_shear_from_habit_plane
from cualni_cryst.lattice import Lattice
import sympy as sp

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/"validation"/"independent"/"reproduce_cayron2026_paper_values_v2.py"


@pytest.fixture(scope="module")
def independently_calculated():
    out=subprocess.run(
        [sys.executable,str(AUDIT)],
        cwd=ROOT,check=True,text=True,capture_output=True,timeout=45,
    )
    assert "cualni_cryst" not in AUDIT.read_text().replace(
        "no import from twin_app or\n", ""
    ).split("\nimport json")[1], "Oracle must not depend on app solver"
    return json.loads(out.stdout)


def test_published_table2_not_read_from_project_solver(independently_calculated):
    d=independently_calculated["table_2"]
    assert d["C1_equality_gap"]==pytest.approx(.137364,abs=1e-6)
    assert d["C1_inequality_margin"]==pytest.approx(.068402,abs=1e-6)


def test_published_table3_stretch_habit_shear_epsilon_and_angle(independently_calculated):
    t=independently_calculated["table_3"]
    assert t["lambda"][1]==pytest.approx(1,abs=2e-12)
    np.testing.assert_allclose(t["m_plus"]["miller"],
                               [1.,-1.,2.41966],atol=1.1e-5,rtol=0)
    np.testing.assert_allclose(t["m_minus"]["miller"],
                               [1.,-1.,-.31568],atol=2e-6,rtol=0)
    np.testing.assert_allclose(t["m_plus"]["paper_scaled_IPS_shear"],
                               [.36938,-.36938,-.05378],atol=1e-5,rtol=0)
    np.testing.assert_allclose(t["m_minus"]["paper_scaled_IPS_shear"],
                               [.115568,-.115568,.216812],atol=2e-6,rtol=0)
    assert t["O2_type_I_twin_shear"]==pytest.approx(.27325,abs=1e-5)
    assert t["m_plus"]["epsilon"]==pytest.approx(.215,abs=5e-4)
    assert t["m_plus"]["angle_deg"]==pytest.approx(5.9,abs=.06)
    assert t["m_plus"]["epsilon"]<t["m_minus"]["epsilon"]


def test_solver_reproduces_oracle_habit_planes_without_using_expected_for_generation(
    independently_calculated
):
    p=independently_calculated["normalized_source_input"]
    A=Lattice.cubic(1.)
    M=Lattice.monoclinic_unique_b(p["a"],p["b"],p["c"],p["beta_deg"])
    C=Correspondence(sp.Matrix([
        [0,0,1],[sp.Rational(1,2),sp.Rational(1,2),0],
        [-sp.Rational(1,2),sp.Rational(1,2),0]
    ]))
    predicted=habit_planes_from_cmc(A.metric(),M.metric(),C,tol=1e-9)
    assert len(predicted)==2
    roots=sorted(
        ((np.asarray(x,float)/float(x[0])).tolist(),x)
        for x in predicted
    )
    expected=sorted([
        independently_calculated["table_3"]["m_plus"]["miller"],
        independently_calculated["table_3"]["m_minus"]["miller"],
    ])
    np.testing.assert_allclose([x[0] for x in roots],expected,atol=2e-8,rtol=0)
    for (m,predicted_plane),m_expected in zip(roots,expected):
        # Match the paper's non-unit h=1 plane normalization, not a
        # crystallographic direct-vector normalization.
        d=ips_shear_from_habit_plane(predicted_plane,A.metric(),M.metric(),C)
        paper_scale=d/float(predicted_plane[0])
        chosen=(independently_calculated["table_3"]["m_plus"]
                if m_expected[2]>0 else independently_calculated["table_3"]["m_minus"])
        np.testing.assert_allclose(paper_scale,chosen["paper_scaled_IPS_shear"],
                                   atol=2e-8,rtol=0)


def test_published_o2_type_i_inconsistency_is_preserved(independently_calculated):
    original=independently_calculated["O2_type_I"]["printed_relation"]
    derived=independently_calculated["O2_type_I"][
        "equation_derived_relation_not_published"]
    assert original["SMC_dz"]==pytest.approx(.5,abs=2e-12)
    assert original["type_I_CCI_squared"]==pytest.approx(.5,abs=2e-12)
    assert derived["SMC_dz"]==pytest.approx(0.,abs=2e-12)
    assert derived["type_I_CCI_squared"]==pytest.approx(1.,abs=2e-12)
    assert original["a"]/derived["a"]==pytest.approx(math.sqrt(2),abs=2e-12)


def test_published_o2_ii_and_o4_analytic_positive_controls(independently_calculated):
    o2=independently_calculated["O2_type_II"]
    assert o2["CCI_squared"]==pytest.approx(1.,abs=2e-12)
    assert o2["SMC_shear_vector"][:2]==pytest.approx([0.,0.],abs=2e-12)
    o4=independently_calculated["O4_appendix_C"]
    assert o4["a"]==pytest.approx(.8825,abs=2e-5)
    assert o4["c"]==pytest.approx(1.6182,abs=4e-5)
    assert o4["b"]==pytest.approx(math.sqrt(2),abs=2e-12)
