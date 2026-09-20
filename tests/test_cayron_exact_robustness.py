from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.cualni_models import (
    do3_to_6m_branch,
    james_hane_6m_example_lattices,
)
from cualni_cryst.group_theory import correspondence_groupoid
from cualni_cryst.lattice import Lattice
from cualni_cryst.twinning_ct import twins_from_operator
from cualni_cryst.weak_twins import (
    BravaisNodeBasis,
    evaluate_axial_weak_twin,
    generalized_shear,
)


def _hex_metric(a: float, c: float) -> np.ndarray:
    return Lattice(a, a, c, 90.0, 90.0, 120.0).metric()


def test_case_270_exact_plane_mapping_survives_ill_conditioned_float_boundary():
    metric = _hex_metric(3.6180986492765816, 9.836369332506465)
    C = sp.Matrix(
        [
            [sp.Rational(4051, 4), sp.Rational(29, 4), sp.Rational(1699, 4)],
            [-5354, -sp.Rational(449, 12), -sp.Rational(26971, 12)],
            [-sp.Rational(9275, 4), -sp.Rational(50, 3), -sp.Rational(2917, 3)],
        ]
    )
    u = sp.Matrix([-24, 127, 55])
    p1 = sp.Matrix([51, 72, -144])
    p2 = sp.Matrix([sp.Rational(159, 4), sp.Rational(273, 4), -sp.Rational(561, 4)])

    assert C.det() == 1
    assert C * u == u
    assert C.inv().T * p1 == p2
    assert np.linalg.cond(np.asarray(C, dtype=float)) > 1.0e7

    result = evaluate_axial_weak_twin(
        metric,
        C,
        u,
        p1,
        p2,
        node_basis=BravaisNodeBasis.primitive_conventional(),
    )

    assert result.selected.residuals.maximum < 1.0e-8
    T = np.asarray(result.selected.T_2_from_1)
    F = np.asarray(result.selected.distortion_F1)
    Cf = np.asarray(C, dtype=float)
    assert np.linalg.norm(T @ F - Cf) / np.linalg.norm(Cf) < 1.0e-10


def test_exact_projective_plane_scaling_is_accepted():
    metric = _hex_metric(3.21, 5.21)
    C = sp.Matrix(
        [
            [1, -sp.Rational(1, 2), 1],
            [0, 0, 2],
            [0, sp.Rational(1, 2), 0],
        ]
    )
    result = evaluate_axial_weak_twin(
        metric,
        C,
        [1, 0, 0],
        [0, 0, 1],
        [0, -sp.Rational(7, 3), 0],
        node_basis=BravaisNodeBasis.primitive_conventional(),
    )
    assert result.generalized_twin_index == 2


def test_tiny_exact_wrong_plane_is_rejected_not_hidden_by_tolerance():
    metric = _hex_metric(3.21, 5.21)
    C = sp.Matrix(
        [
            [1, -sp.Rational(1, 2), 1],
            [0, 0, 2],
            [0, sp.Rational(1, 2), 0],
        ]
    )
    wrong = sp.Matrix([0, 1, sp.Rational(1, 10**12)])
    with pytest.raises(ValueError, match="not exactly related"):
        evaluate_axial_weak_twin(
            metric,
            C,
            [1, 0, 0],
            [0, 0, 1],
            wrong,
            node_basis=BravaisNodeBasis.primitive_conventional(),
        )


def test_equal_volume_requirement_is_exact():
    metric = np.eye(3)
    C = sp.diag(2, 1, 1)
    with pytest.raises(ValueError, match=r"must equal 1 exactly"):
        evaluate_axial_weak_twin(
            metric,
            C,
            [0, 1, 0],
            [1, 0, 0],
            [sp.Rational(1, 2), 0, 0],
            node_basis=BravaisNodeBasis.primitive_conventional(),
        )


def test_cayron_eq5_returns_simple_shear_amplitude_in_nonorthogonal_metric():
    metric = _hex_metric(3.21, 5.21)
    metric_inv = np.linalg.inv(metric)

    d = np.array([1.0, 0.0, 0.0])
    p = np.array([0.0, 1.0, 0.0])
    assert p @ d == 0.0

    d /= math.sqrt(float(d @ metric @ d))
    p /= math.sqrt(float(p @ metric_inv @ p))
    shear = 0.173
    F = np.eye(3) + shear * np.outer(d, p)

    assert generalized_shear(metric, F) == pytest.approx(shear, abs=2.0e-13)

    eps2 = float(np.trace(metric @ F @ metric_inv @ F.T) - 3.0)
    assert math.sqrt(max(0.0, eps2)) == pytest.approx(shear, abs=2.0e-12)


def test_type_i_ii_intercorrespondence_is_exact_correspondence_conjugation():
    A, M = james_hane_6m_example_lattices()
    branch = do3_to_6m_branch()
    groupoid = correspondence_groupoid(
        list(branch.parent_point_group),
        list(branch.product_point_group),
        branch.correspondence,
    )

    found = 0
    for operator in groupoid.operators:
        for twin in twins_from_operator(
            operator,
            A.metric(),
            M.metric(),
            branch.correspondence,
        ):
            exact = branch.correspondence.intercorrespondence_from_parent_symmetry(
                twin.parent_symmetry
            )
            assert np.allclose(
                twin.intercorrespondence,
                np.asarray(exact, dtype=float),
                atol=1.0e-12,
                rtol=1.0e-12,
            )
            found += 1
    assert found > 0


def test_correspondence_plane_map_is_exact_reciprocal_dual():
    C = Correspondence(
        sp.Matrix(
            [
                [1, sp.Rational(1, 3), 0],
                [0, 1, sp.Rational(1, 2)],
                [0, 0, 1],
            ]
        )
    )
    p = sp.Matrix([2, -3, 5])
    mapped = C.map_plane_A_to_M(p)
    assert sp.simplify(C.C_M_from_A.T * mapped - p) == sp.zeros(3, 1)
