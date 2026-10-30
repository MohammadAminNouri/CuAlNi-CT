from __future__ import annotations

import numpy as np
import pytest

from cualni_cryst.niti_benchmark_runner import evaluate_niti_benchmarks


def test_niti_published_AQ_and_2026_CT_are_independently_reproduced():
    result = evaluate_niti_benchmarks()
    assert result["synthetic_generated_EBSD"] is True
    assert result["experimental_niti_parent_measured"] is False
    assert result["physical_cualni_parent_certified"] is False
    a = result["niti_OR_AQ_2020"]
    assert a["daughter_variants"]==12
    assert a["synthetic_noisy_daughter_count"]==12
    assert a["plane_error_deg"]<1e-4
    assert a["direction_error_deg"]<1e-4
    assert a["anchor_error_deg"]<1e-4
    assert a["blind_parent_consensus_error_modulo_cubic_deg"] < .5
    assert a["blind_parent_inlier_fraction"] >= .9
    assert a["anchored_forward_median_deg"] < .5
    assert a["anchored_forward_within_1deg"] > .95
    assert a["wrong_published_C_within_1deg"] < .4
    assert a["wrong_anchor_misorientation_deg"] > 5
    c = result["niti_CT_2026"]
    assert c["correspondence_variants"] == 12
    assert c["intercorrespondence_operator_classes"] == 7
    assert c["principal_stretches"] == pytest.approx(
        [0.930026279,0.965048059,1.119484928], rel=2e-8
    )
