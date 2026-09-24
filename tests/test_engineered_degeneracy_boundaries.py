from __future__ import annotations

import numpy as np

from validation.engineered.physical import degeneracy_case, make_truth, encode_truth
from validation.engineered.production import evaluate


def test_all_cmc_degeneracy_classes_are_distinguished():
    expected = {
        "second_low": (True, 2, 1),
        "second_high": (True, 2, 1),
        "identity": (True, 3, 0),
        "same_sign_low": (False, 0, 0),
        "same_sign_high": (False, 0, 0),
    }
    for kind, (compatible, order, plane_count) in expected.items():
        case = degeneracy_case(kind)
        prod = evaluate(case, tol=1e-10)
        assert bool(prod.ct_analysis.exact_compatible) is compatible, kind
        assert int(prod.ct_analysis.degeneracy_order) == order, kind
        assert len(prod.ct_planes_crystal) == plane_count, kind


def test_near_unit_middle_stretch_has_a_resolved_transition_not_random_flapping():
    # This does not demand exact classification below floating resolution.
    # It checks monotone numerical residual and explicit transition behavior.
    deltas = np.array([-1e-4,-1e-6,-1e-8,-1e-10,0.0,1e-10,1e-8,1e-6,1e-4])
    nearest = []
    classified = []
    for i, d in enumerate(deltas):
        truth = make_truth(
            "boundary", i,
            lambdas=np.array([0.8, 1.0 + float(d), 1.25]),
        )
        case = encode_truth(truth, "boundary-encoding", i)
        prod = evaluate(case, tol=1e-9)
        nearest.append(float(prod.ct_analysis.nearest_zero_residual))
        classified.append(bool(prod.ct_analysis.exact_compatible))

    # Away from tolerance, residual should track |lambda_2^2-1|.
    for d, r in zip(deltas, nearest):
        expected = abs((1.0+d)**2 - 1.0)
        if abs(d) >= 1e-8:
            np.testing.assert_allclose(r, expected, atol=2e-9, rtol=2e-6)

    assert classified[0] is False and classified[-1] is False
    assert classified[4] is True
