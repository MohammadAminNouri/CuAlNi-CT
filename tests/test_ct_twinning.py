import numpy as np
from cualni_cryst.cualni_models import do3_to_6m_branch, james_hane_6m_example_lattices
from cualni_cryst.group_theory import correspondence_groupoid
from cualni_cryst.twinning_ct import twins_from_operator


def test_ct_twin_predictions_are_metric_normalized():
    A,M=james_hane_6m_example_lattices();b=do3_to_6m_branch();g=correspondence_groupoid(list(b.parent_point_group),list(b.product_point_group),b.correspondence)
    found=0
    for op in g.operators:
        for tw in twins_from_operator(op,A.metric(),M.metric(),b.correspondence):
            found+=1
            assert tw.shear >= 0
            assert np.isfinite(tw.shear)
            assert abs(tw.direction_m @ M.metric() @ tw.direction_m - 1) < 1e-8
            assert abs(tw.plane_m @ np.linalg.inv(M.metric()) @ tw.plane_m - 1) < 1e-8
    assert found > 0
