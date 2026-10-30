from __future__ import annotations

from io import StringIO
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from cualni_cryst.parent_or_interop import (
    proper_cubic_parent_operations, frozen_or_boundary_audit,
)
from cualni_cryst.parent_grain_table_replay import replay_grain_boundary_tables


def synthetic_tables():
    p=proper_cubic_parent_operations()
    c=(np.eye(3),Rotation.from_euler("y",180,degrees=True).as_matrix())
    Q=Rotation.from_euler("xyz",[9,23,-13],degrees=True).as_matrix()
    gA=Rotation.from_euler("xyz",[31,19,8],degrees=True).as_matrix()
    g=np.asarray([gA @ p[k] @ Q @ c[k%2] for k in (0,3,7,10)])
    q=Rotation.from_matrix(g).as_quat()
    grain=pd.DataFrame(q,columns=["qx","qy","qz","qw"])
    boundary=pd.DataFrame({
        "row_index_A_zero_based":[0,1,2],
        "row_index_B_zero_based":[1,2,3],
        "boundary_contacts":[2,5,7],
    })
    direct=frozen_or_boundary_audit(
        g[np.asarray([[0,1],[1,2],[2,3]])],Q,p,c
    )
    boundary["frozen_OR_operator_misfit_degrees"]=direct["operator_misfit_deg"]
    return (
        grain.to_csv(index=False).encode(),
        boundary.to_csv(index=False).encode(),Q,tuple(p),tuple(c),
    )


def test_portable_grain_boundary_replay_preserves_frozen_model():
    g,b,Q,p,c=synthetic_tables()
    result=replay_grain_boundary_tables(g,b,Q,p,c)
    assert result["n_daughter_grain_regions"]==4
    assert result["n_adjacency_edges"]==3
    assert result["n_total_boundary_contacts"]==14
    assert result["median_operator_misfit_deg"]<1e-4
    assert result["historical_operator_residual_max_abs_difference_deg"]<1e-6
    assert result["physical_or_certified"] is False
    assert result["mtex_graph_mcl_executed"] is False


def test_boundary_replay_rejects_repeated_and_invalid_edges():
    g,b,Q,p,c=synthetic_tables()
    frame=pd.read_csv(StringIO(b.decode()))
    frame=pd.concat([frame,frame.iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError,match="Duplicated"):
        replay_grain_boundary_tables(g,frame.to_csv(index=False).encode(),Q,p,c)
    frame=pd.read_csv(StringIO(b.decode()))
    frame.loc[0,"row_index_B_zero_based"]=99
    with pytest.raises(ValueError,match="Out-of-range"):
        replay_grain_boundary_tables(g,frame.to_csv(index=False).encode(),Q,p,c)


def test_incorrect_or_not_accepted_by_boundary_fit():
    g,b,Q,p,c=synthetic_tables()
    wrong=Q@Rotation.from_euler("xyz",[17,26,11],degrees=True).as_matrix()
    good=replay_grain_boundary_tables(g,b,Q,p,c)
    bad=replay_grain_boundary_tables(g,b,wrong,p,c)
    assert good["median_operator_misfit_deg"] < bad["median_operator_misfit_deg"]
