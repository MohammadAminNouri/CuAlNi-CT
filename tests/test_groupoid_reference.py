from cualni_cryst.cualni_models import do3_to_6m_branch, do3_to_2h_branch
from cualni_cryst.group_theory import correspondence_groupoid


def test_6m_groupoid_exact_counts_and_burnside():
    b=do3_to_6m_branch();g=correspondence_groupoid(list(b.parent_point_group),list(b.product_point_group),b.correspondence)
    assert len(g.subgroup)==4
    assert g.n_variants==12
    assert g.n_operators==8
    assert sorted(map(len,g.operators))==[4,4,4,4,8,8,8,8]
    assert g.burnside_count==8


def test_2h_groupoid_exact_counts_and_burnside():
    b=do3_to_2h_branch();g=correspondence_groupoid(list(b.parent_point_group),list(b.product_point_group),b.correspondence)
    assert len(g.subgroup)==8
    assert g.n_variants==6
    assert g.n_operators==3
    assert sorted(map(len,g.operators))==[8,8,32]
    assert g.burnside_count==3
