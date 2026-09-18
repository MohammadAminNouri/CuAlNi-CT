import json

import numpy as np

from cualni_cryst.compatibility_atlas import (
    james_hane_benchmark_projection,
)
from cualni_cryst.ct import smc
from cualni_cryst.cualni_models import (
    do3_to_6m_branch,
    james_hane_6m_example_lattices,
)
from cualni_cryst.stretch import stretch_from_metrics


def _projection():
    return james_hane_benchmark_projection()


def test_source_rounded_benchmark_is_not_silently_promoted_to_exact():
    result = _projection()
    observed = result.observed

    assert observed.state.status == "SOURCE_MEASURED_ROUNDED_BENCHMARK"
    assert not observed.state.exact_compatible
    assert observed.state.n_exact_habit_planes == 0
    assert observed.state.nearest_cmc_residual > 0.0
    assert observed.twin_systems == ()


def test_exact_beta_projection_is_explicitly_hypothetical_and_exact():
    result = _projection()
    projected = result.projected

    assert projected.state.status == "HYPOTHETICAL_TEST"
    assert projected.state.exact_compatible
    assert projected.state.degeneracy_order == 1
    assert projected.state.n_exact_habit_planes == 2
    assert abs(projected.state.lambda2_residual) < 1e-8
    assert abs(result.beta_shift_deg) > 0.0
    assert min(
        abs(result.projected_beta_deg - x)
        for x in result.exact_beta_candidates_deg
    ) < 1e-12


def test_cmc_habit_planes_match_independent_ball_james_single_variant_solutions():
    result = _projection()
    cross = result.projected.am_habit_crosscheck

    assert len(cross) == 2
    assert {x.ball_james_branch for x in cross} == {-1, 1}
    assert max(x.plane_angle_deg for x in cross) < 1e-7
    assert max(x.ball_james_rank_one_residual for x in cross) < 1e-10


def test_cubic_parent_smc_bridge_to_stretch_theory():
    A, M = james_hane_6m_example_lattices()
    result = _projection()
    beta = result.projected_beta_deg

    from cualni_cryst.lattice import Lattice

    Mp = Lattice.monoclinic_unique_b(M.a, M.b, M.c, beta)
    branch = do3_to_6m_branch()
    U = stretch_from_metrics(A.metric(), Mp.metric(), branch.correspondence)

    left = smc(A.metric(), Mp.metric(), branch.correspondence)
    right = (np.eye(3) - np.linalg.inv(U @ U)) / (A.a**2)

    assert np.allclose(left, right, atol=1e-10, rtol=1e-10)


def test_exact_projection_tests_every_exact_twin_relation_in_both_branches():
    rows = _projection().projected.twin_systems

    # Six exact reference-to-target M/M relations from the frozen twin atlas,
    # with Type-I and Type-II tested independently for each relation.
    assert len(rows) == 12
    assert {x.twin_kind for x in rows} == {"I", "II"}
    assert all(np.isfinite(x.supercompatibility_residual) for x in rows)
    assert all(x.supercompatibility_residual >= 0.0 for x in rows)
    assert all(np.isfinite(x.shear_direction_angle_deg) for x in rows)
    assert all(abs(x.cc1_residual) < 1e-8 for x in rows)


def test_cofactor_and_ptmc_are_independent_reported_quantities():
    rows = _projection().projected.twin_systems

    # Do not assume the exact-beta projection is fully supercompatible.
    # Instead verify that every row exposes the independent residuals/results.
    for row in rows:
        assert np.isfinite(row.cc2_residual)
        assert np.isfinite(row.cc2_simplified)
        assert np.isfinite(row.cc3_margin)
        assert all(-1e-12 <= f <= 1.0 + 1e-12 for f in row.ptmc_volume_fractions)
        if row.ptmc_volume_fractions:
            assert row.ptmc_max_middle_stretch_residual < 1e-6


def test_backend_result_is_json_serializable_for_future_interactive_frontend():
    result = _projection().projected
    payload = result.to_dict()
    encoded = json.dumps(payload)

    assert '"state"' in encoded
    assert '"twin_systems"' in encoded
    assert payload["state"]["status"] == "HYPOTHETICAL_TEST"
