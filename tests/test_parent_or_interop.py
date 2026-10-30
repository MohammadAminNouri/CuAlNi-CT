from __future__ import annotations

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
                       [0.9522096477271853]*2,atol=1e-4)
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
