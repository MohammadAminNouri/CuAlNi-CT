from __future__ import annotations

import math
import random

import numpy as np
import pytest
import sympy as sp

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.ct import (
    analyze_cmc,
    approximate_cmc_habit_planes,
    habit_planes_from_cmc,
)
from cualni_cryst.group_theory import correspondence_groupoid
from cualni_cryst.lattice import Lattice
from cualni_cryst.point_groups import (
    metric_preservation_residual,
    point_group_operations,
)
from cualni_cryst.twinning_ct import (
    classify_parent_order_two_isometry,
    twins_from_operator,
)


def _matrix(rows) -> sp.Matrix:
    return sp.Matrix(rows)


def _run_case(
    parent_lattice: Lattice,
    product_lattice: Lattice,
    parent_point_group: str,
    product_point_group: str,
    C_rows,
):
    M_a = parent_lattice.metric()
    M_m = product_lattice.metric()
    parent = point_group_operations(parent_point_group)
    product = point_group_operations(product_point_group)
    correspondence = Correspondence(_matrix(C_rows))

    assert metric_preservation_residual(parent, M_a) < 1.0e-11
    assert metric_preservation_residual(product, M_m) < 1.0e-11

    groupoid = correspondence_groupoid(
        list(parent), list(product), correspondence
    )
    assert len(parent) % len(groupoid.subgroup) == 0
    assert groupoid.n_variants == len(parent) // len(groupoid.subgroup)
    assert sum(len(item) for item in groupoid.variants) == len(parent)
    assert sum(len(item) for item in groupoid.operators) == len(parent)
    assert groupoid.burnside_count == groupoid.n_operators
    assert len(groupoid.adjacency) == groupoid.n_variants
    assert all(len(row) == groupoid.n_variants for row in groupoid.adjacency)

    cmc = analyze_cmc(M_a, M_m, correspondence)
    assert np.all(np.isfinite(cmc.eigenvalues))
    assert np.isfinite(cmc.nearest_zero_residual)

    twins = [
        twin
        for operator in groupoid.operators
        for twin in twins_from_operator(operator, M_a, M_m, correspondence)
    ]
    for twin in twins:
        assert twin.kind in {"I", "II"}
        assert twin.classification in {"type_I", "type_II", "compound"}
        assert twin.representations in {("I",), ("II",), ("I", "II")}
        assert np.isfinite(twin.shear)
        assert twin.shear >= 0.0
        assert abs(float(twin.direction_m @ M_m @ twin.direction_m) - 1.0) < 1.0e-8
        assert (
            abs(
                float(
                    twin.plane_m
                    @ np.linalg.inv(M_m)
                    @ twin.plane_m
                )
                - 1.0
            )
            < 1.0e-8
        )
        assert abs(float(twin.plane_m @ twin.direction_m)) < 1.0e-8

    return groupoid, cmc, twins


@pytest.mark.parametrize(
    (
        "label",
        "parent_lattice",
        "product_lattice",
        "parent_pg",
        "product_pg",
        "C",
        "minimum_twins",
    ),
    [
        (
            "cubic_to_tetragonal",
            Lattice.cubic(3.17),
            Lattice(3.02, 3.02, 4.73),
            "m-3m",
            "4/mmm",
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            1,
        ),
        (
            "cubic_to_orthorhombic",
            Lattice.cubic(3.11),
            Lattice.orthorhombic(2.70, 3.60, 4.50),
            "m-3m",
            "mmm",
            [[0, 1, 1], [1, 0, 0], [0, 1, -1]],
            1,
        ),
        (
            "hexagonal_to_orthorhombic",
            Lattice(2.95, 2.95, 4.68, 90.0, 90.0, 120.0),
            Lattice.orthorhombic(2.80, 3.10, 4.40),
            "6/mmm",
            "mmm",
            [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
            1,
        ),
        (
            "monoclinic_to_monoclinic",
            Lattice.monoclinic_unique_b(3.10, 4.20, 5.30, 104.0),
            Lattice.monoclinic_unique_b(3.40, 4.00, 5.10, 98.0),
            "2/m",
            "2/m",
            [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
            1,
        ),
        (
            "monoclinic_to_triclinic",
            Lattice.monoclinic_unique_b(3.10, 4.20, 5.30, 104.0),
            Lattice(3.30, 4.10, 5.00, 82.0, 101.0, 77.0),
            "2/m",
            "-1",
            [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
            1,
        ),
        (
            "triclinic_to_triclinic_legitimate_zero_twin_state",
            Lattice(3.20, 4.10, 5.00, 79.0, 96.0, 73.0),
            Lattice(3.00, 4.40, 4.90, 83.0, 92.0, 71.0),
            "-1",
            "1",
            [[1, 1, 0], [0, 1, 1], [0, 0, 1]],
            0,
        ),
    ],
)
def test_deterministic_cross_crystal_systems(
    label,
    parent_lattice,
    product_lattice,
    parent_pg,
    product_pg,
    C,
    minimum_twins,
):
    groupoid, _, twins = _run_case(
        parent_lattice,
        product_lattice,
        parent_pg,
        product_pg,
        C,
    )
    assert groupoid.n_variants >= 1, label
    assert groupoid.n_operators >= 1, label
    if minimum_twins:
        assert len(twins) >= minimum_twins, label
    else:
        assert twins == [], label


def _transform_group(group, P: sp.Matrix):
    Pi = P.inv()
    return [sp.simplify(Pi * sp.Matrix(g) * P) for g in group]


def _twin_signature(twins):
    return sorted(
        (
            twin.kind,
            twin.classification,
            round(float(twin.shear), 11),
        )
        for twin in twins
    )


def test_unimodular_basis_change_covariance_on_niti():
    """Physical topology/shear must not depend on a conventional basis choice."""

    A = Lattice.cubic(3.015)
    M = Lattice.monoclinic_unique_b(2.889, 4.120, 4.622, 96.8)
    C = Correspondence(
        sp.Matrix(
            [
                [0, 0, 1],
                [sp.Rational(1, 2), sp.Rational(1, 2), 0],
                [-sp.Rational(1, 2), sp.Rational(1, 2), 0],
            ]
        )
    )
    G_a = list(point_group_operations("m-3m"))
    G_m = list(point_group_operations("2/m"))

    base_groupoid = correspondence_groupoid(G_a, G_m, C)
    base_twins = [
        twin
        for operator in base_groupoid.operators
        for twin in twins_from_operator(operator, A.metric(), M.metric(), C)
    ]
    base_cmc = analyze_cmc(A.metric(), M.metric(), C)

    P_a = sp.Matrix([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    P_m = sp.Matrix([[1, 0, 0], [0, 1, 1], [0, 0, 1]])

    M_a_prime = (
        np.asarray(P_a, dtype=float).T
        @ A.metric()
        @ np.asarray(P_a, dtype=float)
    )
    M_m_prime = (
        np.asarray(P_m, dtype=float).T
        @ M.metric()
        @ np.asarray(P_m, dtype=float)
    )
    C_prime = Correspondence(sp.simplify(P_m.inv() * C.C_M_from_A * P_a))
    G_a_prime = _transform_group(G_a, P_a)
    G_m_prime = _transform_group(G_m, P_m)

    transformed_groupoid = correspondence_groupoid(
        G_a_prime, G_m_prime, C_prime
    )
    transformed_twins = [
        twin
        for operator in transformed_groupoid.operators
        for twin in twins_from_operator(
            operator, M_a_prime, M_m_prime, C_prime
        )
    ]
    transformed_cmc = analyze_cmc(
        M_a_prime, M_m_prime, C_prime
    )

    assert len(base_groupoid.subgroup) == len(transformed_groupoid.subgroup)
    assert base_groupoid.n_variants == transformed_groupoid.n_variants
    assert base_groupoid.n_operators == transformed_groupoid.n_operators
    assert _twin_signature(base_twins) == _twin_signature(transformed_twins)
    assert np.allclose(
        np.sort(base_cmc.eigenvalues),
        np.sort(transformed_cmc.eigenvalues),
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_cmc_degeneracy_and_pathological_boundaries():
    I = Correspondence(sp.eye(3))
    parent = np.eye(3)

    third = analyze_cmc(parent, np.eye(3), I, tol=1.0e-10)
    assert third.exact_compatible
    assert third.degeneracy_order == 3
    assert third.inertia == (0, 3, 0)

    second_metric = np.diag([1.0, 1.0, 1.21])
    second = analyze_cmc(parent, second_metric, I, tol=1.0e-10)
    assert second.exact_compatible
    assert second.degeneracy_order == 2
    assert len(habit_planes_from_cmc(parent, second_metric, I, tol=1.0e-10)) == 1

    first_metric = np.diag([0.81, 1.0, 1.21])
    first = analyze_cmc(parent, first_metric, I, tol=1.0e-10)
    assert first.exact_compatible
    assert first.degeneracy_order == 1
    assert first.inertia == (1, 1, 1)
    assert len(habit_planes_from_cmc(parent, first_metric, I, tol=1.0e-10)) == 2

    near_metric = np.diag([0.81, 1.0 + 1.0e-7, 1.21])
    near = analyze_cmc(parent, near_metric, I, tol=1.0e-9)
    assert not near.exact_compatible
    approximate = approximate_cmc_habit_planes(parent, near_metric, I)
    assert 0.0 < approximate.residual < 2.0e-7
    assert approximate.admissible_signature
    assert len(approximate.candidate_planes) == 2

    incompatible_metric = np.diag([0.81, 0.90, 1.21])
    incompatible = analyze_cmc(
        parent, incompatible_metric, I, tol=1.0e-10
    )
    assert not incompatible.exact_compatible
    assert incompatible.degeneracy_order == 0


def test_invalid_inputs_fail_loudly():
    with pytest.raises(ValueError, match="Correspondence must be an invertible"):
        Correspondence(sp.diag(1, 1, 0))

    with pytest.raises(ValueError, match="Lattice lengths must be positive"):
        Lattice(-1.0, 2.0, 3.0)

    # A tetragonal fourfold is not an isometry of this a != b orthorhombic metric.
    R4 = sp.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="does not preserve"):
        classify_parent_order_two_isometry(
            R4,
            np.diag([2.0, 3.0, 4.0]),
        )


def _random_lattice_for_point_group(
    point_group: str, rng: random.Random
) -> Lattice:
    if point_group == "m-3m":
        return Lattice.cubic(rng.uniform(2.5, 5.5))
    if point_group == "4/mmm":
        a = rng.uniform(2.5, 4.5)
        return Lattice(a, a, rng.uniform(3.0, 6.0))
    if point_group == "mmm":
        return Lattice.orthorhombic(
            rng.uniform(2.5, 4.5),
            rng.uniform(3.0, 5.0),
            rng.uniform(3.5, 6.0),
        )
    if point_group == "6/mmm":
        a = rng.uniform(2.4, 3.8)
        return Lattice(
            a, a, rng.uniform(3.5, 6.0), 90.0, 90.0, 120.0
        )
    if point_group == "2/m":
        return Lattice.monoclinic_unique_b(
            rng.uniform(2.5, 4.5),
            rng.uniform(3.0, 5.0),
            rng.uniform(3.5, 6.0),
            rng.uniform(94.0, 108.0),
        )
    if point_group in {"-1", "1"}:
        return Lattice(
            rng.uniform(2.5, 4.5),
            rng.uniform(3.0, 5.0),
            rng.uniform(3.5, 6.0),
            rng.uniform(82.0, 98.0),
            rng.uniform(84.0, 104.0),
            rng.uniform(78.0, 102.0),
        )
    raise AssertionError(point_group)


def test_seeded_multi_system_random_campaign():
    """Synthetic stress test: invariants only, never material predictions."""

    rng = random.Random(20260921)
    systems = [
        ("m-3m", "4/mmm"),
        ("m-3m", "mmm"),
        ("6/mmm", "mmm"),
        ("2/m", "2/m"),
        ("2/m", "-1"),
        ("-1", "1"),
    ]
    correspondences = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 1], [0, 1, 0], [0, 0, 1]],
        [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        [[1, 1, 0], [0, 1, 1], [0, 0, 1]],
    ]

    for trial in range(24):
        parent_pg, product_pg = systems[trial % len(systems)]
        parent_lattice = _random_lattice_for_point_group(parent_pg, rng)
        product_lattice = _random_lattice_for_point_group(product_pg, rng)
        C = correspondences[rng.randrange(len(correspondences))]

        groupoid, cmc, twins = _run_case(
            parent_lattice,
            product_lattice,
            parent_pg,
            product_pg,
            C,
        )

        assert groupoid.n_variants >= 1
        assert groupoid.n_operators >= 1
        assert cmc.inertia[0] + cmc.inertia[1] + cmc.inertia[2] == 3
        assert all(math.isfinite(twin.shear) for twin in twins)
