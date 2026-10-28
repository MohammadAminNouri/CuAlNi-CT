"""P0/P1 mathematical falsification with no experimental inputs or fitted results."""

from types import SimpleNamespace as NS
import numpy as np
import pytest

from cualni_cryst.point_groups import point_group_operations
from cualni_cryst.theory_unified import PredictionKind
from twin_app.theory_p0p1_certificate import (
    passive_basis_covariance, independent_shear_shear_epsilon,
    certify_existing_theory, _group_audit,
    PARENT_REBASE, PRODUCT_REBASE,
)


def published_orthorhombic_case():
    # Only published Cu–Al–Ni rounded INPUTS, not benchmark habit answers.
    # data/benchmarks/bhattacharya_cualni_orthorhombic_habit_v1.json
    a = 5.836
    product_axes = (4.382116368337481, 5.3562808, 4.222001371547299)
    Ma = np.eye(3) * (a * a)
    Mm = np.diag(np.square(product_axes))
    C = np.array([[0., 1., 1.], [1., 0., 0.], [0., 1., -1.]])
    ga = tuple(np.asarray(g, float) for g in point_group_operations("m-3m"))
    gm = tuple(np.asarray(g, float) for g in point_group_operations("mmm"))
    return Ma, Mm, C, ga, gm


def test_p0_two_independent_crystal_basis_changes_for_published_cualni_input():
    Ma, Mm, C, ga, gm = published_orthorhombic_case()
    audit = passive_basis_covariance(Ma, Mm, C, ga, gm, tolerance=1e-7)
    assert audit["pass"]
    assert audit["parent_declared_group"]["count"] == len(ga)
    assert audit["product_declared_group"]["count"] == len(gm)
    for key in ("parent_only", "product_only", "both_independently"):
        assert audit[key]["pass"], audit[key]
        assert audit[key]["maximum_residual"] < 1e-10
    assert audit["actual_structure_symmetry_verified"] is False
    assert audit["experimental_correspondence_verified"] is False


@pytest.mark.parametrize("parent_change,product_change", [
    (PARENT_REBASE, np.eye(3)), (np.eye(3), PRODUCT_REBASE),
    (PARENT_REBASE, PRODUCT_REBASE),
])
def test_p0_direct_reciprocal_incidence_is_exact_under_passive_changes(
        parent_change, product_change):
    Ma, Mm, C, _, _ = published_orthorhombic_case()
    pi, qi = np.linalg.inv(parent_change), np.linalg.inv(product_change)
    ur = pi @ np.array([1., 2., -1.])
    hr = parent_change.T @ np.array([2., -1., 0.])
    Cr = qi @ C @ parent_change
    assert hr @ ur == pytest.approx(0., abs=1e-14)
    assert np.linalg.solve(Cr.T, hr) @ (Cr @ ur) == pytest.approx(0., abs=1e-14)
    assert np.linalg.det(Cr) != 0
    assert np.linalg.norm(
        Cr.T @ product_change.T @ Mm @ product_change @ Cr -
        parent_change.T @ C.T @ Mm @ C @ parent_change
    ) < 1e-11


def test_p0_correctly_rejects_a_metric_violating_declared_operation():
    Ma, Mm, C, ga, gm = published_orthorhombic_case()
    # 90-degree exchange of axes is not orthorhombic daughter symmetry
    wrong = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    report = passive_basis_covariance(Ma, Mm, C, ga, [np.eye(3), wrong])
    assert report["pass"] is False
    assert report["product_declared_group"]["computed_group_pass"] is False


def test_p0_metric_preservation_never_proves_actual_atomic_symmetry():
    # Even inversion preserves a cubic metric. A noncentrosymmetric atomic
    # structure need NOT possess inversion (Cayron 2019 corrigendum).
    s = _group_audit([np.eye(3), -np.eye(3)], np.eye(3), tolerance=1e-8)
    assert s["computed_group_pass"] is True
    assert s["actual_atomic_structure_symmetry_verified"] is False
    assert "need not" in s["external_evidence_required"]


def test_p0_group_closure_fails_when_rotation_generator_not_closed():
    Q = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    out = _group_audit([np.eye(3), Q], np.eye(3), tolerance=1e-8)
    assert out["group_closed"] is False
    assert out["computed_group_pass"] is False


def test_p1_cayron_eq49_is_basis_invariant_under_independent_rebasing():
    M = np.diag([3., 5., 7.])
    m = np.array([1., -2., 1.])
    d = np.array([.05, .09, -.07])
    n = np.array([-2., 1., 2.])
    eta = np.array([2., 2., 1.])
    magnitude = .29
    baseline = independent_shear_shear_epsilon(M, m, d, n, eta, magnitude)
    A = PARENT_REBASE
    rebased = independent_shear_shear_epsilon(
        A.T @ M @ A, A.T @ m, np.linalg.inv(A) @ d,
        A.T @ n, np.linalg.inv(A) @ eta, magnitude)
    assert rebased == pytest.approx(baseline, rel=2e-14, abs=1e-14)
    assert baseline > 0


def test_p1_cayron_eq49_does_not_accept_zero_twin_shear():
    with pytest.raises(ValueError, match="nonzero"):
        independent_shear_shear_epsilon(
            np.eye(3), [1, 0, 0], [0.1, 0, 0],
            [0, 1, 0], [0, 0, 1], 0.)


def minimal_project_and_row(*, corrupted=False):
    Ma, Mm, C, _, _ = published_orthorhombic_case()
    parent = NS(lattice=NS(metric=lambda: Ma),
                symmetry_matrices=lambda: (np.eye(3), -np.eye(3)))
    product = NS(lattice=NS(metric=lambda: Mm),
                 symmetry_matrices=lambda: (np.eye(3), -np.eye(3)))
    phases = {"A": parent, "M": product}
    tr = NS(parent_phase_id="A", product_phase_id="M",
            correspondence=NS(to_float64_checked=lambda: C))
    project = NS(transformation=lambda name: tr, phase=lambda name: phases[name])
    exact = independent_shear_shear_epsilon(
        Ma, [1., 0., 1.], [0., 0.1, .2],
        [1., 1., 0.], [0., 0., 1.], .21,
    )
    row = NS(
        prediction_kind=PredictionKind.CT_SUPERCOMPATIBILITY,
        row_id="ct_super_0", habit_plane_parent_crystal=[1., 0., 1.],
        shape_vector_parent_crystal=[0., 0.1, .2],
        twin_plane_parent_crystal=[1., 1., 0.],
        twin_direction_parent_crystal=[0., 0., 1.],
        shear_magnitude=.21,
        residuals={"ct_supercompatibility_dimensionless":
                   exact + (.11 if corrupted else 0.)})
    bj = NS(variants=(), martensite_twin_branches=())
    unified = NS(rows=(row,), ball_james_report=bj, ptmc_report=None)
    return project, unified


def test_p0p1_positive_synthetic_nontrivial_full_certificate():
    project, unified = minimal_project_and_row()
    cert = certify_existing_theory(project, "A_to_M", unified)
    assert cert["P0_computational_pass"]
    assert cert["P1_computational_pass"]
    assert cert["P1_shear_shear"]["audited_pairs"] == 1
    assert cert["P1_shear_shear"]["max_independent_error"] < 1e-12
    assert cert["P1_Ball_James"]["audited_branches"] == 0
    assert cert["P1_PTMC"]["audited_habits"] == 0


def test_p1_tampered_reported_shear_shear_is_detected():
    project, unified = minimal_project_and_row(corrupted=True)
    cert = certify_existing_theory(project, "A_to_M", unified)
    assert cert["P0_computational_pass"]
    assert cert["P1_computational_pass"] is False
    assert cert["P1_shear_shear"]["max_independent_error"] > .1
