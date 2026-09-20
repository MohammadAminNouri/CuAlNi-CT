from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import cualni_cryst.ptmc_adapter as ptmc_adapter_module
from cualni_cryst.lattice import Lattice
from cualni_cryst.ptmc_adapter import (
    PTMCCrystalInput,
    PTMCSlipSystemInput,
    PTMCTwinPlaneInput,
    analyze_ptmc_twinning,
    solve_affine_ptmc_parameters,
)


def test_ptmc_adapter_does_not_import_ct_or_ball_james_adapter():
    source = Path(ptmc_adapter_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden_suffixes = {
        "ct",
        "ct_orientation",
        "ct_weak_orientation",
        "twinning_ct",
        "ball_james_adapter",
        "compatibility_atlas",
    }
    assert not {
        name
        for name in imported
        if any(
            name == suffix or name.endswith(f".{suffix}")
            for suffix in forbidden_suffixes
        )
    }


def test_affine_parameter_solver_recovers_both_discrete_roots():
    F0 = np.diag([0.8, 0.9, 1.2])
    increment = np.outer([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    result = solve_affine_ptmc_parameters(F0, increment)

    expected = 0.2615339366124404
    assert not result.continuum
    assert len(result.discrete_roots) == 2
    assert result.discrete_roots[0] == pytest.approx(-expected, abs=1.0e-12)
    assert result.discrete_roots[1] == pytest.approx(expected, abs=1.0e-12)
    assert result.polynomial_verification_residual < 1.0e-12


def test_affine_parameter_solver_reports_exact_continuum_without_sampling():
    F0 = np.diag([0.8, 1.2, 1.0])
    increment = np.outer([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    result = solve_affine_ptmc_parameters(
        F0,
        increment,
        domain=(0.0, 1.0),
    )

    assert result.continuum
    assert result.discrete_roots == ()
    assert result.domain == (0.0, 1.0)
    assert max(abs(x) for x in result.polynomial_coefficients_ascending) < 1.0e-14


def test_ptclab_style_slip_input_gives_all_four_discrete_branches():
    parent = Lattice.cubic(1.0, length_unit="angstrom")
    product = Lattice.cubic(1.0, length_unit="angstrom")
    # This synthetic correspondence produces U=diag(0.8,0.9,1.2).
    correspondence = np.diag([0.8, 0.9, 1.2])
    model = PTMCCrystalInput(
        parent_lattice=parent,
        product_lattice=product,
        correspondence=correspondence,
        parent_point_group="1",
        product_point_group="1",
    )
    slip = PTMCSlipSystemInput(
        plane_product=(0.0, 1.0, 0.0),
        direction_product=(1.0, 0.0, 0.0),
    )

    report = model.analyze_slip(slip)

    assert len(report.variants) == 1
    assert report.full_parent_symmetry_order == 1
    assert report.proper_parent_symmetry_order == 1
    assert len(report.parameter_roots) == 1
    roots = report.parameter_roots[0].discrete_roots
    assert len(roots) == 2
    assert roots[0] == pytest.approx(-0.2905932629027115, abs=1.0e-12)
    assert roots[1] == pytest.approx(0.2905932629027115, abs=1.0e-12)
    # Two slip-shear roots and two habit-plane branches for each root.
    assert len(report.solutions) == 4
    assert not report.continuous_families
    assert report.audit.maximum_residual < 1.0e-8
    for solution in report.solutions:
        assert solution.lis_type == "slip"
        assert solution.true_invariant_plane
        assert solution.rank_one_residual < 1.0e-10
        assert solution.middle_stretch_residual < 1.0e-10
        assert abs(solution.rotation_determinant - 1.0) < 1.0e-10
        assert solution.or_base_residual < 1.0e-10
        assert np.allclose(
            solution.macroscopic_deformation - np.eye(3),
            np.outer(
                solution.rank_one_vector,
                solution.habit_normal_parent_cartesian,
            ),
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        assert solution.invariant_line is not None
        assert solution.invariant_line.deformation_residual < 1.0e-10


def test_slip_direction_must_lie_in_product_slip_plane():
    with pytest.raises(ValueError, match="must lie in"):
        PTMCSlipSystemInput(
            plane_product=(0.0, 1.0, 0.0),
            direction_product=(0.0, 1.0, 0.0),
        )


def _two_variant_synthetic_report():
    lattice = Lattice.cubic(1.0, length_unit="angstrom")
    # Proper 180-degree rotation that exchanges x and y and reverses z.
    twofold = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    return analyze_ptmc_twinning(
        lattice,
        lattice,
        np.diag([0.8, 1.1, 1.2]),
        (np.eye(3), twofold),
        (np.eye(3),),
        base_variant_index=0,
    )


def test_twinning_ptmc_enumerates_both_twin_and_habit_branches():
    report = _two_variant_synthetic_report()

    assert len(report.variants) == 2
    assert len(report.twin_relations) == 2
    assert {item.branch for item in report.twin_relations} == {-1, 1}
    for roots in report.parameter_roots:
        assert roots.discrete_roots == pytest.approx(
            (0.3135474780640175, 0.6864525219359833),
            abs=1.0e-12,
        )
        assert not roots.continuum
        assert roots.polynomial_verification_residual < 1.0e-12

    # 2 twin branches x 2 fractions x 2 habit branches.
    assert len(report.solutions) == 8
    assert report.audit.maximum_residual < 1.0e-8
    for solution in report.solutions:
        assert solution.lis_type == "twin"
        assert solution.base_variant_volume_fraction is not None
        assert solution.other_variant_volume_fraction is not None
        assert (
            solution.base_variant_volume_fraction
            + solution.other_variant_volume_fraction
            == pytest.approx(1.0, abs=1.0e-12)
        )
        assert solution.or_parent_from_product_other_symmetric is not None
        assert solution.or_parent_from_product_other_ptclab is not None


def test_twin_plane_input_finds_shear_direction_from_exact_twin_family():
    report = _two_variant_synthetic_report()
    relation = report.twin_relations[0]
    lattice = Lattice.cubic(1.0, length_unit="angstrom")
    twofold = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    selected = analyze_ptmc_twinning(
        lattice,
        lattice,
        np.diag([0.8, 1.1, 1.2]),
        (np.eye(3), twofold),
        (np.eye(3),),
        twin_plane=PTMCTwinPlaneInput(
            plane_product=tuple(relation.twin_plane_product_crystal_base),
            base_variant_index=0,
            plane_tolerance_deg=1.0e-6,
        ),
    )

    assert selected.twin_relations
    assert all(item.base_variant_index == 0 for item in selected.twin_relations)
    assert all(
        np.isfinite(item.twin_shear_magnitude) and item.twin_shear_magnitude > 0.0
        for item in selected.twin_relations
    )
    assert selected.solutions


def test_dilatational_factor_is_explicit_and_not_called_true_invariant_plane():
    parent = Lattice.cubic(1.0, length_unit="angstrom")
    product = Lattice.cubic(1.0, length_unit="angstrom")
    model = PTMCCrystalInput(
        parent_lattice=parent,
        product_lattice=product,
        correspondence=np.diag([0.8, 0.9, 1.2]),
        parent_point_group="1",
        product_point_group="1",
    )
    report = model.analyze_slip(
        PTMCSlipSystemInput(
            plane_product=(0.0, 1.0, 0.0),
            direction_product=(1.0, 0.0, 0.0),
        ),
        dilatational_factor=1.01,
    )

    assert report.dilatational_factor == pytest.approx(1.01)
    assert all(not solution.true_invariant_plane for solution in report.solutions)


def test_report_is_json_ready():
    import json

    report = _two_variant_synthetic_report()
    payload = report.to_dict()
    encoded = json.dumps(payload)

    assert '"lis_type": "twin"' in encoded
    assert "solutions" in payload
    assert "audit" in payload
