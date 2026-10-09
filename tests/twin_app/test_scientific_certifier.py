from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace as Record

import numpy as np
import pytest

from twin_app.scientific_certifier import ScientificCertificationError, certify_ptmc_twinning


def _exact_case():
    """Construct a complete synthetic twin + habit system independently.

    U1,U2 are symmetry-related positive stretches with equal determinants;
    2D Mallard rank-one connection is embedded in 3D. Habit branches are then
    built from the middle-eigenvalue-one compatibility proposition.
    """
    p, q, frac = .9, 1.1, .30
    Ubase = np.diag([p, q, 1.0]); Uother = np.diag([q, p, 1.0])
    cos_theta = 2*p*q/(p*p+q*q)
    sin_theta = np.sqrt(1.0-cos_theta*cos_theta)
    Rt = np.eye(3)
    Rt[:2,:2] = [[cos_theta, -sin_theta], [sin_theta, cos_theta]]
    D = Rt @ Uother - Ubase
    us, sing, vt = np.linalg.svd(D)
    a = us[:,0]*sing[0]; n = vt[0,:]
    assert np.linalg.norm(D - np.outer(a,n)) < 1e-12

    F = Ubase + frac*np.outer(a,n)
    mu, e = np.linalg.eigh(F.T@F)
    m1,m2,m3 = mu
    assert m1 < 1.0 and abs(m2-1) < 1e-12 and m3 > 1.0
    den = np.sqrt(m3-m1)
    shape = (np.sqrt(m3*(1-m1))*e[:,0] + np.sqrt(m1*(m3-1))*e[:,2])/den
    normal = (np.sqrt(m3)-np.sqrt(m1))/den * (-np.sqrt(1-m1)*e[:,0]+np.sqrt(m3-1)*e[:,2])
    Rh = (np.eye(3) + np.outer(shape,normal)) @ np.linalg.inv(F)
    normal_norm = np.linalg.norm(normal)
    b = shape * normal_norm
    m = normal/normal_norm
    assert np.linalg.norm(Rh.T@Rh-np.eye(3))<1e-12
    assert np.linalg.norm(Rh@F-np.eye(3)-np.outer(b,m))<1e-12

    return Record(
        variants=(Record(index=0,U=Ubase),Record(index=1,U=Uother)),
        twin_relations=(Record(base_variant_index=0,other_variant_index=1,branch=1,
                               rotation_hat=Rt,a=a,n_reference=n),),
        solutions=(Record(base_variant_index=0,other_variant_index=1,twin_branch=1,
                          true_invariant_plane=True,other_variant_volume_fraction=frac,
                          base_variant_volume_fraction=1-frac,pre_shape_deformation=F,
                          habit_rotation=Rh,rank_one_vector=b,
                          habit_normal_parent_cartesian=m),),
    )


def test_independently_constructed_exact_twin_and_habit_pass():
    result=certify_ptmc_twinning(_exact_case())
    assert result.twin_relations_checked==1
    assert result.habit_solutions_checked==1
    assert result.maximum_twin_tensor_residual<1e-12
    assert result.maximum_habit_tensor_residual<1e-12
    assert result.maximum_middle_stretch_residual<1e-12


@pytest.mark.parametrize('fault',[
    'wrong_twin_rotation', 'mismatched_laminate', 'wrong_shape_magnitude',
    'bad_habit_rotation', 'non_unit_habit_normal', 'wrong_branch',
    'invalid_lambda', 'invalid_volume_balance',
])
def test_corrupt_data_is_rejected_instead_of_rendered(fault):
    case=deepcopy(_exact_case())
    twin=case.twin_relations[0]
    habit=case.solutions[0]
    if fault == 'wrong_twin_rotation': twin.rotation_hat[0,0]+=.02
    elif fault == 'mismatched_laminate': habit.pre_shape_deformation[0,0]+=.02
    elif fault == 'wrong_shape_magnitude': habit.rank_one_vector*=2
    elif fault == 'bad_habit_rotation': habit.habit_rotation[0,0]+=.01
    elif fault == 'non_unit_habit_normal': habit.habit_normal_parent_cartesian*=2
    elif fault == 'wrong_branch': habit.twin_branch=-1
    elif fault == 'invalid_lambda': habit.other_variant_volume_fraction=1.2
    elif fault == 'invalid_volume_balance': habit.base_variant_volume_fraction=.75
    with pytest.raises(ScientificCertificationError):
        certify_ptmc_twinning(case)


def test_no_habit_is_valid_if_no_exact_root_exists():
    case=_exact_case()
    case.solutions=()
    result=certify_ptmc_twinning(case)
    assert result.habit_solutions_checked==0
    assert result.twin_relations_checked==1


def test_certification_is_invariant_under_common_proper_frame_rotation():
    case = _exact_case()
    angle = 0.41
    R = np.array([[np.cos(angle), 0., np.sin(angle)],
                  [0., 1., 0.],
                  [-np.sin(angle), 0., np.cos(angle)]])
    for variant in case.variants:
        variant.U = R @ variant.U @ R.T
    twin = case.twin_relations[0]
    twin.rotation_hat = R @ twin.rotation_hat @ R.T
    twin.a = R @ twin.a
    twin.n_reference = R @ twin.n_reference
    habit = case.solutions[0]
    habit.pre_shape_deformation = R @ habit.pre_shape_deformation @ R.T
    habit.habit_rotation = R @ habit.habit_rotation @ R.T
    habit.rank_one_vector = R @ habit.rank_one_vector
    habit.habit_normal_parent_cartesian = R @ habit.habit_normal_parent_cartesian
    summary = certify_ptmc_twinning(case)
    assert summary.maximum_twin_tensor_residual < 1e-12
    assert summary.maximum_habit_tensor_residual < 1e-12


def test_habit_rank_one_is_invariant_under_joint_sign_reversal():
    case = _exact_case()
    habit = case.solutions[0]
    habit.rank_one_vector *= -1
    habit.habit_normal_parent_cartesian *= -1
    assert certify_ptmc_twinning(case).habit_solutions_checked == 1


def test_variant_volume_mismatch_is_rejected_even_if_rank_one_is_fake():
    case = _exact_case()
    case.variants[1].U[0, 0] += 0.04
    with pytest.raises(ScientificCertificationError):
        certify_ptmc_twinning(case)
