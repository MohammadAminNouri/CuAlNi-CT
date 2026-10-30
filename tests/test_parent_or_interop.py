from __future__ import annotations

from math import erfc

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cualni_cryst.parent_or_interop import (
    frozen_or_boundary_audit, mtex_or_contract, proper_cubic_parent_operations,
)


def test_mt_ex_or_direction_exact_transpose_and_unit_quaternion():
    Q=Rotation.from_euler("xyz",[10,-21,32],degrees=True).as_matrix()
    a=mtex_or_contract(Q)
    assert np.allclose(np.asarray(a["mtex_p2c_daughter_from_parent"]),Q.T)
    assert np.allclose(np.asarray(a["Q_parent_from_daughter"]) @
                       np.asarray(a["mtex_p2c_daughter_from_parent"]),np.eye(3))
    assert np.isclose(np.linalg.norm(a["quaternion_xyzw_parent_from_daughter"]),1)


def test_full_symmetry_operator_and_mtex_style_gaussian_weights():
    parent=proper_cubic_parent_operations()
    assert len(parent)==24
    Q=Rotation.from_euler("xyz",[11,-23,19],degrees=True).as_matrix()
    gA=Rotation.from_euler("xyz",[13,29,-15],degrees=True).as_matrix()
    SM=Rotation.from_euler("y",180,degrees=True).as_matrix()
    g1=gA @ Q
    g2=gA @ parent[13] @ Q @ SM
    pair=np.asarray([[g1,g2], [g2,g1]])
    res=frozen_or_boundary_audit(pair,Q,parent,[np.eye(3),SM])
    assert res["n_boundary_pairs"]==2
    assert res["parent_symmetry_operations"]==24
    assert res["daughter_symmetry_operations"]==2
    assert res["median_operator_misfit_deg"]<1e-4
    assert np.allclose(res["mtex_style_probability"],
                       [0.9999987857662636]*2,atol=1e-7)
    assert res["fraction_under_3deg"]==1


def test_incorrect_or_not_marked_physically_validated():
    parent=proper_cubic_parent_operations()
    Q=Rotation.from_euler("xyz",[12,18,-9],degrees=True).as_matrix()
    g=np.eye(3)
    pairs=np.asarray([[g, g@Q.T@parent[5]@Q]])
    res=frozen_or_boundary_audit(pairs,Q,parent,[np.eye(3)])
    assert "not" in res["scientific_status"].lower()


def test_reject_improper_or_and_nonrotations():
    parent=proper_cubic_parent_operations()
    with pytest.raises(ValueError):
        mtex_or_contract(np.diag([1,1,-1]))
    with pytest.raises(ValueError):
        frozen_or_boundary_audit(np.stack([[np.eye(3),np.eye(3)]]),
                                 np.diag([1,1,-1]),parent,[np.eye(3)])


def test_mtex_exact_probability_reference_at_transition_and_tails():
    # Independent special-function reference to MTEX calcVariantGraph.m:
    # 1 - 0.5*(1 + erf(2*(misfit-threshold)/tolerance)).
    # Identity groups remove variant/symmetry ambiguity for this test.
    angles = [0.0, 1.0, 2.5, 4.0, 6.0]
    pairs = np.asarray([
        [np.eye(3), Rotation.from_euler('z', angle, degrees=True).as_matrix()]
        for angle in angles
    ])
    out = frozen_or_boundary_audit(
        pairs, np.eye(3), (np.eye(3),), (np.eye(3),),
        threshold_deg=2.5, transition_width_deg=1.5,
    )
    assert np.allclose(out['operator_misfit_deg'], angles, atol=1e-6)
    reference = [0.5*erfc(2*(angle-2.5)/1.5) for angle in angles]
    assert np.allclose(out['mtex_style_probability'], reference, atol=1e-12)
    assert out['mtex_style_probability'][2] == pytest.approx(0.5)
    assert out['mtex_style_probability'][3] == pytest.approx(0.0023388674905236327)
    assert 'not MTEX graph' in out['mtex_probability_scope']


@pytest.mark.parametrize('threshold,width', [
    (float('nan'), 1.5), (2.5, float('nan')),
    (float('inf'), 1.5), (2.5, float('inf')),
])
def test_mtex_probability_requires_finite_parameters(threshold, width):
    pair = np.asarray([[np.eye(3), np.eye(3)]])
    with pytest.raises(ValueError, match='threshold/tolerance'):
        frozen_or_boundary_audit(pair, np.eye(3), (np.eye(3),),
                                 (np.eye(3),), threshold_deg=threshold,
                                 transition_width_deg=width)
