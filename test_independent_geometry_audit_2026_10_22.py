"""Local mathematical self-tests (no production application required)."""
import numpy as np
import mpmath as mp
from independent_geometry_audit_2026_10_22 import (
    mx, branches, coefficient_roots, match_set, selftest,
)


def test_independent_rank_one_and_habit_oracle():
    selftest()


def test_interior_fraction_root_from_new_oracle():
    with mp.workdps(85):
        F=mx(np.diag([.8,.9,1.2]))
        D=mx(np.outer([1.,0.,0.],[0.,1.,0.]))
        kind,roots,disc=coefficient_roots(F,D)
        assert kind=='ROOTS' and len(roots)==1
        assert abs(float(roots[0])-0.2615339366124404)<1e-12
        status,habit,_=branches(mp.eye(3),F+roots[0]*D)
        assert status=='TWO_BRANCHES' and len(habit)==2


def test_incompatible_no_rank_one():
    with mp.workdps(85):
        status,twins,_=branches(mp.eye(3),mx(np.diag([.8,.9,1.2])))
        assert status=='INCOMPATIBLE' and not twins


def test_matching_is_one_to_one():
    with mp.workdps(85):
        A=mx(np.outer([1.,0,0],[0,1.,0])); B=mx(np.outer([0,1.,0],[1,0.,0]))
        assert match_set([A,B],[B,A],1e-12)['classification']=='MATCH'
        assert match_set([A,B],[A,A],1e-12)['classification']=='MISMATCH'
        assert match_set([A,B],[A],1e-12)['classification']=='MISMATCH'
