import sympy as sp
from cualni_cryst.correspondence import Correspondence
from cualni_cryst.group_theory import correspondence_groupoid
from cualni_cryst.symmetry import cubic_full_m3m, monoclinic_2_over_m_unique_b


def test_identity_toy_partition():
    C = Correspondence(sp.eye(3))
    res = correspondence_groupoid(cubic_full_m3m(), monoclinic_2_over_m_unique_b(), C)
    assert len(res.subgroup) == 4
    assert len(res.variants) == 12
    assert sum(len(c) for c in res.variants) == 48
    assert sum(len(d) for d in res.operators) == 48
