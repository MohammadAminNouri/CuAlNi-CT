"""Independent PTMC habit tensor reproduction must not alter underlying PTMC."""

from types import SimpleNamespace as NS

import numpy as np

from twin_app.theory_evidence_ui import ptmc_habit_evidence


def sample_solution():
    deformation = np.diag([1.1, 1.0, 1.0])
    return NS(
        pre_shape_deformation=deformation,
        habit_rotation=np.eye(3),
        macroscopic_deformation=deformation.copy(),
        rank_one_vector=np.array([.1, 0., 0.]),
        habit_normal_parent_cartesian=np.array([1., 0., 0.]),
        dilatational_factor=1.,
        other_variant_volume_fraction=.25,
        true_invariant_plane=True,
        base_variant_index=0,
        other_variant_index=1,
        twin_branch=0,
        habit_branch=1,
        lis_type="twin",
    )


def test_exact_rank_one_habit_is_reproduced_independently():
    s = sample_solution()
    result = ptmc_habit_evidence(NS(ptmc_report=NS(solutions=(s,))))
    assert len(result) == 1
    assert result[0]["Audit status"] == "REPRODUCED"
    assert result[0]["Independent A/M rank-one tensor residual"] < 1e-13
    assert result[0]["Independent middle-stretch gap"] < 1e-13


def test_non_rank_one_habit_cannot_be_silently_certified():
    s = sample_solution()
    s.rank_one_vector = np.array([.2, 0., 0.])
    result = ptmc_habit_evidence(NS(ptmc_report=NS(solutions=(s,))))
    assert result[0]["Audit status"] == "REVIEW_REQUIRED"


def test_nonphysical_laminate_fraction_cannot_be_certified():
    s = sample_solution()
    s.other_variant_volume_fraction = 1.2
    result = ptmc_habit_evidence(NS(ptmc_report=NS(solutions=(s,))))
    assert result[0]["Audit status"] == "REVIEW_REQUIRED"


def test_no_ptmc_report_is_not_failure_or_pass():
    assert ptmc_habit_evidence(NS(ptmc_report=None)) == []
