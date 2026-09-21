from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.cualni_models import (
    do3_to_6m_branch,
    james_hane_6m_example_lattices,
)
from cualni_cryst.lattice import Lattice
from cualni_cryst.physical_validation import (
    cross_validate_ct_mallard_ball_james,
    validate_ct_basis_covariance,
)
from cualni_cryst.point_groups import point_group_operations


BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks"


def _expr(value: object) -> sp.Expr:
    if isinstance(value, (int, float)):
        return sp.Rational(str(value))
    return sp.sympify(str(value), locals={"sqrt": sp.sqrt})


def _matrix(rows) -> sp.Matrix:
    return sp.Matrix([[_expr(item) for item in row] for row in rows])


def _lattice(spec: dict) -> Lattice:
    kind = spec["kind"]
    if kind == "cubic":
        return Lattice.cubic(spec["a_angstrom"], length_unit="angstrom")
    if kind == "orthorhombic":
        return Lattice.orthorhombic(
            spec["a_angstrom"],
            spec["b_angstrom"],
            spec["c_angstrom"],
            length_unit="angstrom",
        )
    if kind == "monoclinic_unique_b":
        return Lattice.monoclinic_unique_b(
            spec["a_angstrom"],
            spec["b_angstrom"],
            spec["c_angstrom"],
            spec["beta_deg"],
            length_unit="angstrom",
        )
    raise AssertionError(f"Unsupported benchmark lattice kind {kind!r}")


def _benchmark_state(filename: str):
    manifest = json.loads((BENCHMARK_DIR / filename).read_text())
    inp = manifest["input"]
    A = _lattice(inp["parent"]["lattice"])
    M = _lattice(inp["product"]["lattice"])
    parent = point_group_operations(inp["parent"]["point_group"])
    product = point_group_operations(inp["product"]["point_group"])
    C = Correspondence(_matrix(inp["package_C_M_from_A"]))
    return A.metric(), M.metric(), parent, product, C


def _random_unimodular(rng: random.Random, steps: int = 6) -> sp.Matrix:
    """Small exact GL(3,Z) basis change built from elementary operations."""

    P = sp.eye(3)
    for _ in range(steps):
        action = rng.randrange(3)
        if action == 0:
            i, j = rng.sample(range(3), 2)
            E = sp.eye(3)
            E.col_swap(i, j)
        elif action == 1:
            i = rng.randrange(3)
            E = sp.eye(3)
            E[i, i] = -1
        else:
            i, j = rng.sample(range(3), 2)
            k = rng.choice((-1, 1))
            E = sp.eye(3)
            # New basis vector i gets +/- one copy of basis vector j.
            E[j, i] = k
        P = sp.simplify(P * E)

    assert abs(int(P.det())) == 1
    # Avoid extreme condition numbers in a numerical validation whose purpose
    # is convention invariance, not arbitrary-precision stress testing.
    assert np.linalg.cond(np.asarray(P, dtype=float)) < 50.0
    return P


@pytest.mark.parametrize(
    "filename",
    [
        "chen_2000_blind_2h_v1.json",
        "cayron_2022_niti_ct_v1.json",
    ],
)
def test_complete_ct_objects_are_covariant_under_exact_random_basis_changes(filename):
    M_a, M_m, parent, product, C = _benchmark_state(filename)
    rng = random.Random(20260921 + sum(map(ord, filename)))

    for _ in range(6):
        # Regenerate if a rare combination becomes unnecessarily ill-conditioned.
        for _attempt in range(50):
            try:
                P_A = _random_unimodular(rng)
                P_M = _random_unimodular(rng)
                break
            except AssertionError:
                continue
        else:
            raise AssertionError("Could not generate a well-conditioned GL(3,Z) pair")

        report = validate_ct_basis_covariance(
            M_a, M_m, parent, product, C, P_A, P_M
        )

        assert report.subgroup_exact, report
        assert report.variant_partition_exact, report
        assert report.operator_partition_exact, report
        assert report.adjacency_exact, report
        assert report.twin_provenance_exact, report
        assert report.classification_exact, report
        assert report.max_geometry_residual < 2.0e-9, report
        assert report.max_shear_relative_residual < 2.0e-10, report
        assert report.max_intercorrespondence_residual < 2.0e-10, report
        assert report.success, report


@pytest.mark.parametrize(
    "filename",
    [
        "chen_2000_blind_2h_v1.json",
        "cayron_2022_niti_ct_v1.json",
    ],
)
def test_ct_mallard_and_generic_ball_james_agree_independently(filename):
    M_a, M_m, parent, product, C = _benchmark_state(filename)

    report = cross_validate_ct_mallard_ball_james(
        M_a, M_m, parent, product, C
    )

    assert report.relation_count > 0, report
    assert report.ct_coverage_complete, report
    assert report.classification_exact, report
    assert report.max_ct_geometry_angle_deg < 2.0e-6, report
    assert report.max_ball_james_geometry_angle_deg < 2.0e-6, report
    assert report.max_shear_relative_residual < 2.0e-9, report
    assert report.max_rank_one_residual < 2.0e-9, report
    assert report.max_rotation_residual < 2.0e-8, report
    assert report.success, report


def test_generic_theory_validator_reproduces_existing_cualni_6m_crosslock():
    A, M = james_hane_6m_example_lattices()
    branch = do3_to_6m_branch()

    report = cross_validate_ct_mallard_ball_james(
        A.metric(),
        M.metric(),
        branch.parent_point_group,
        branch.product_point_group,
        branch.correspondence,
    )

    # This is deliberately not fitted to operator numbering. The existing 6M
    # atlas is a separate independent regression; here we test the generic
    # physical-equivalence machinery on the same scientific state.
    assert report.relation_count > 0, report
    assert report.ct_coverage_complete, report
    assert report.classification_exact, report
    assert report.max_ct_geometry_angle_deg < 2.0e-6, report
    assert report.max_ball_james_geometry_angle_deg < 2.0e-6, report
    assert report.max_shear_relative_residual < 2.0e-9, report
    assert report.max_rank_one_residual < 2.0e-9, report
    assert report.max_rotation_residual < 2.0e-8, report
    assert report.success, report
