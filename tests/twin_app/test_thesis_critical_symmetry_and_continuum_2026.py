"""Independent negative controls for thesis-threatening false twin claims.

No alloy-specific expected shears, habit vectors, or reference outputs are used.
"""
from __future__ import annotations

import numpy as np
import pytest

from twin_app.p0_classification import screen_parent_twofold
from cualni_cryst.ptmc_adapter import solve_affine_ptmc_parameters


def _generic_stretch():
    return np.array([
        [1.07, 0.011, 0.023],
        [0.011, 0.94, 0.017],
        [0.023, 0.017, 1.12],
    ], dtype=float)


def test_mirror_only_parent_does_not_falsely_prove_nonconventional_type():
    mirror = np.diag((-1.0, 1.0, 1.0))
    U = _generic_stretch()
    other = mirror @ U @ mirror.T
    proof = screen_parent_twofold(
        U, other, parent_metric=np.eye(3),
        parent_group=(np.eye(3), mirror),
    )
    assert proof.proper_twofold_count == 0
    assert proof.mapping_twofold_indices == ()
    assert proof.parent_mirror_count == 1
    assert proof.matching_mirror_indices == (1,)
    assert not proof.conventional_symmetry_excluded
    assert not proof.conventional_twofold_excluded


def test_no_mirror_or_twofold_remains_screened_conservatively():
    U = _generic_stretch()
    mirror = np.diag((-1.0, 1.0, 1.0))
    proof = screen_parent_twofold(
        U, mirror @ U @ mirror.T,
        parent_metric=np.eye(3),
        parent_group=(np.eye(3),),
    )
    assert proof.conventional_symmetry_excluded


@pytest.mark.parametrize("factor", [1e-10, 1e-12])
def test_tiny_full_rank_increment_cannot_pass_ptmc_rank_one_gate(factor):
    with pytest.raises(ValueError, match="numerical rank one"):
        solve_affine_ptmc_parameters(
            np.eye(3), factor * np.eye(3),
            domain=(0.0, 1.0),
        )


def test_tiny_actual_rank_one_increment_is_not_mistaken_for_full_rank():
    increment = np.zeros((3, 3))
    increment[0, 1] = 1e-10
    result = solve_affine_ptmc_parameters(np.eye(3), increment, domain=(0., 1.))
    assert result.continuum


@pytest.mark.parametrize("domain", [(0., 1.), None])
def test_zero_determinant_polynomial_does_not_imply_middle_stretch(domain):
    # The THIRD singular value is identically one, not the middle value.
    # det(F.T F-I)=0 for all x, but lambda_2=0.6 near x=0.
    F0 = np.diag((0.5, 0.6, 1.0))
    A = np.diag((0.1, 0.0, 0.0))
    with pytest.raises(ValueError, match="middle stretch"):
        solve_affine_ptmc_parameters(F0, A, domain=domain)


def test_true_rank_one_simple_shear_continuum_survives_independent_check():
    A = np.zeros((3, 3))
    A[0, 1] = 0.2
    for domain in ((0.0, 1.0), None):
        result = solve_affine_ptmc_parameters(np.eye(3), A, domain=domain)
        assert result.continuum
        assert not result.discrete_roots
