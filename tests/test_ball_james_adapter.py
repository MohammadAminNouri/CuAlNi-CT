from __future__ import annotations

import ast
import inspect

import numpy as np

import cualni_cryst.ball_james_adapter as ball_james_adapter_module
from cualni_cryst.ball_james_adapter import (
    BallJamesAdapter,
    BallJamesCrystalInput,
    analytical_rank_one_connections,
    evaluate_cofactor_conditions,
    mallard_law_twins,
    metric_native_stretch_spectrum,
    parent_symmetry_rotations,
    stretch_from_crystallographic_state,
)
from cualni_cryst.cualni_models import do3_to_6m_branch, james_hane_6m_example_lattices
from cualni_cryst.project_state import james_hane_6m_reference_project


def test_module_has_no_ct_or_ptmc_imports():
    tree = ast.parse(inspect.getsource(ball_james_adapter_module))
    forbidden = {"cualni_cryst.ct", "cualni_cryst.ptmc", "cualni_cryst.ct_orientation"}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert forbidden.isdisjoint(imported)


def test_proposition_one_returns_two_exact_single_variant_habit_branches():
    U = np.diag([0.9, 1.0, 1.1])
    branches = analytical_rank_one_connections(U, np.eye(3))
    assert len(branches) == 2
    assert {branch.branch for branch in branches} == {-1, 1}
    assert max(branch.maximum_residual for branch in branches) < 1.0e-12
    for branch in branches:
        assert np.isclose(np.linalg.norm(branch.n), 1.0)
        assert np.linalg.det(branch.rotation) > 0.0


def test_proposition_one_refuses_noncompatible_middle_eigenvalue():
    U = np.diag([0.9, 1.02, 1.1])
    assert analytical_rank_one_connections(U, np.eye(3)) == ()


def test_mallard_law_independently_matches_general_rank_one_solver():
    U_base = np.diag([0.9, 1.0, 1.1])
    axis = np.array([1.0, 0.0, 1.0])
    axis /= np.linalg.norm(axis)
    Q = 2.0 * np.outer(axis, axis) - np.eye(3)
    U_other = Q @ U_base @ Q.T

    general = analytical_rank_one_connections(U_other, U_base)
    mallard = mallard_law_twins(U_base, U_other, Q)
    assert len(general) == 2
    assert {kind for kind, _, _ in mallard} == {"I", "II"}

    for _, mallard_branch, mallard_shear in mallard:
        best = min(
            min(
                np.linalg.norm(branch.rotation - mallard_branch.rotation, ord="fro"),
                np.linalg.norm(
                    np.outer(branch.a, branch.n)
                    - np.outer(mallard_branch.a, mallard_branch.n),
                    ord="fro",
                ),
            )
            for branch in general
        )
        assert best < 1.0e-10
        assert mallard_branch.maximum_residual < 1.0e-10
        assert mallard_shear > 0.0


def test_cofactor_margins_are_invariant_under_rank_one_rescaling():
    U = np.diag([0.9, 1.0, 1.1])
    axis = np.array([1.0, 0.0, 1.0])
    axis /= np.linalg.norm(axis)
    Q = 2.0 * np.outer(axis, axis) - np.eye(3)
    U_other = Q @ U @ Q.T
    branch = analytical_rank_one_connections(U_other, U)[0]

    first = evaluate_cofactor_conditions(U, branch.a, branch.n, fraction_samples=21)
    scale = 7.25
    second = evaluate_cofactor_conditions(
        U,
        scale * branch.a,
        branch.n / scale,
        fraction_samples=21,
    )

    assert np.isclose(first.cc1_lambda2_minus_one, second.cc1_lambda2_minus_one)
    assert np.isclose(first.cc2_value, second.cc2_value)
    assert np.isclose(first.cc3_margin, second.cc3_margin)
    assert np.isclose(
        first.sampled_all_fraction_max_lambda2_residual,
        second.sampled_all_fraction_max_lambda2_residual,
    )


def test_metric_native_and_whitened_stretch_spectra_agree_for_oblique_cells():
    M_a = np.array(
        [
            [4.0, 0.2, 0.1],
            [0.2, 5.0, 0.3],
            [0.1, 0.3, 6.0],
        ]
    )
    M_m = np.array(
        [
            [4.2, 0.1, 0.2],
            [0.1, 4.8, 0.1],
            [0.2, 0.1, 6.2],
        ]
    )
    C = np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    U, residual = stretch_from_crystallographic_state(M_a, M_m, C)
    spectrum = metric_native_stretch_spectrum(M_a, M_m, C)
    lambdas = np.sort(np.linalg.eigvalsh(U))

    assert residual < 1.0e-12
    assert np.max(np.abs(lambdas - spectrum.lambdas)) < 1.0e-12
    assert spectrum.eigen_equation_residual < 1.0e-12
    assert spectrum.metric_orthonormality_residual < 1.0e-12


def test_noncubic_metric_symmetry_is_converted_to_same_whitened_frame():
    M = np.array(
        [
            [4.0, 0.2, 0.1],
            [0.2, 5.0, 0.3],
            [0.1, 0.3, 6.0],
        ]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    B = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    B_inv = np.linalg.inv(B)
    axis = np.array([1.0, 0.0, 1.0])
    axis /= np.linalg.norm(axis)
    Q = 2.0 * np.outer(axis, axis) - np.eye(3)
    g = B_inv @ Q @ B

    rotations = parent_symmetry_rotations(M, [np.eye(3), g])
    assert len(rotations) == 2
    assert max(np.linalg.norm(Qi.T @ Qi - np.eye(3)) for _, Qi in rotations) < 1.0e-12


def test_james_hane_project_generates_independent_12_variant_family_and_exact_twins():
    project = james_hane_6m_reference_project()
    report = BallJamesAdapter(project, "do3_to_6m_reference").analyze(
        fraction_samples=21
    )

    assert report.full_parent_symmetry_order == 48
    assert report.proper_parent_symmetry_order == 24
    assert len(report.variants) == 12
    assert report.audit.pulled_back_metric_residual < 1.0e-10
    assert report.audit.metric_native_spectrum_residual < 1.0e-10
    assert report.audit.maximum_parent_symmetry_so3_residual < 1.0e-10
    assert report.audit.maximum_variant_spectrum_residual < 1.0e-10
    assert report.martensite_twin_branches
    assert report.audit.maximum_martensite_rank_one_residual < 1.0e-8
    assert any(branch.mallard_matches for branch in report.martensite_twin_branches)

    # Rounded measured lattice parameters need not satisfy exact A/M lambda2=1.
    for branch in report.austenite_martensite_branches:
        assert branch.rank_one_residual < 1.0e-8

    payload = report.to_dict()
    assert payload["transformation_id"] == "do3_to_6m_reference"
    assert len(payload["variants"]) == 12


def test_ptclab_style_crystal_input_matches_project_bound_adapter():
    parent, product = james_hane_6m_example_lattices()
    branch = do3_to_6m_branch()
    simple_report = BallJamesCrystalInput(
        parent_lattice=parent,
        product_lattice=product,
        correspondence=branch.correspondence,
        parent_point_group="m-3m",
        product_point_group="2/m",
        transformation_id="manual_reference",
    ).analyze(fraction_samples=21)

    project = james_hane_6m_reference_project()
    project_report = BallJamesAdapter(project, "do3_to_6m_reference").analyze(
        fraction_samples=21
    )

    assert len(simple_report.variants) == len(project_report.variants) == 12
    assert simple_report.full_parent_symmetry_order == 48
    assert simple_report.proper_parent_symmetry_order == 24
    assert simple_report.full_product_symmetry_order == 4
    assert simple_report.proper_product_symmetry_order == 2
    assert np.allclose(
        simple_report.metric_spectrum.lambdas, project_report.metric_spectrum.lambdas
    )
    assert np.allclose(simple_report.U0, project_report.U0)


def test_ptclab_style_input_rejects_incompatible_point_group_and_cell():
    parent, product = james_hane_6m_example_lattices()
    branch = do3_to_6m_branch()
    bad = BallJamesCrystalInput(
        parent_lattice=parent,
        product_lattice=product,
        correspondence=branch.correspondence,
        parent_point_group="m-3m",
        product_point_group="m-3m",
    )
    try:
        bad.analyze(fraction_samples=5)
    except ValueError as exc:
        assert "incompatible" in str(exc)
    else:
        raise AssertionError("Incompatible product symmetry must fail loudly")
