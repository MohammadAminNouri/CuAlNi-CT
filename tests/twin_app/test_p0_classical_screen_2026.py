"""No numerical literature answers hardcoded: verify the actual necessary symmetry."""
import itertools
import numpy as np
import pytest
from twin_app.p0_classification import screen_parent_twofold


def cubic_operations():
    out=[]
    for ordering in itertools.permutations(range(3)):
        for signs in itertools.product((-1,1),repeat=3):
            R=np.zeros((3,3))
            for column,row in enumerate(ordering):R[row,column]=signs[column]
            out.append(R)
    return out


def test_quarter_turn_can_link_stretches_without_conventional_parent_twofold():
    U=np.array([[1.07,.011,.023],[.011,.94,.017],[.023,.017,1.12]])
    Q=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    U2=Q@U@Q.T
    proof=screen_parent_twofold(U,U2,parent_metric=np.eye(3),parent_group=cubic_operations())
    assert proof.operation_count==48
    assert proof.proper_twofold_count==9
    assert proof.conventional_twofold_excluded
    assert not proof.mapping_twofold_indices


def test_half_turn_is_not_falsely_labeled_nonconventional():
    U=np.array([[1.07,.011,.023],[.011,.94,.017],[.023,.017,1.12]])
    Q=np.diag([-1.,-1.,1.])
    proof=screen_parent_twofold(U,Q@U@Q.T,parent_metric=np.eye(3),parent_group=cubic_operations())
    assert proof.mapping_twofold_indices
    assert not proof.conventional_twofold_excluded


def test_wrong_metric_symmetry_fails_closed():
    U=np.diag([1.01,.98,1.05])
    with pytest.raises(ValueError,match='metric'):
        screen_parent_twofold(U,U,parent_metric=np.diag([2.,3.,4.]),parent_group=cubic_operations())


def test_empty_or_invalid_group_fails_closed():
    with pytest.raises(ValueError):
        screen_parent_twofold(np.eye(3),np.eye(3),parent_metric=np.eye(3),parent_group=[])
