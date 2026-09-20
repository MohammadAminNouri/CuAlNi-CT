from __future__ import annotations

import math

import numpy as np
import sympy as sp

from cualni_cryst.weak_twins import (
    BravaisNodeBasis,
    enumerate_ct_constrained_weak_planes,
    evaluate_axial_weak_twin,
    generalized_twin_index,
)


def hcp_metric(a: float = 3.21, c: float = 5.21) -> np.ndarray:
    return np.array(
        [
            [a * a, -0.5 * a * a, 0.0],
            [-0.5 * a * a, a * a, 0.0],
            [0.0, 0.0, c * c],
        ]
    )


def test_cayron_2022_mg_a_axis_weak_twin_1():
    correspondence = sp.Matrix(
        [
            [1, -sp.Rational(1, 2), 1],
            [0, 0, 2],
            [0, sp.Rational(1, 2), 0],
        ]
    )
    result = evaluate_axial_weak_twin(
        hcp_metric(),
        correspondence,
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        node_basis=BravaisNodeBasis.primitive_conventional(),
    )
    assert result.generalized_twin_index == 2
    assert math.isclose(result.generalized_strain, 0.1300860007, abs_tol=2e-9)
    assert math.isclose(result.selected.generalized_shear, 0.0920332215, abs_tol=2e-9)
    assert result.selected.determinant < 0.0
    assert result.selected.residuals.maximum < 1e-10


def test_cayron_2022_mg_a_axis_weak_twin_2():
    correspondence = sp.Matrix(
        [
            [1, -sp.Rational(1, 4), sp.Rational(3, 4)],
            [0, sp.Rational(1, 2), sp.Rational(3, 2)],
            [0, -sp.Rational(1, 2), sp.Rational(1, 2)],
        ]
    )
    result = evaluate_axial_weak_twin(
        hcp_metric(),
        correspondence,
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        node_basis=BravaisNodeBasis.primitive_conventional(),
    )
    assert result.generalized_twin_index == 4
    assert math.isclose(result.generalized_strain, 0.1367030152, abs_tol=2e-9)
    assert math.isclose(result.selected.generalized_shear, 0.1068698448, abs_tol=2e-9)
    assert result.selected.determinant > 0.0
    assert result.selected.residuals.maximum < 1e-10


def test_exact_generalized_twin_index_uses_smith_normal_form():
    c1 = sp.Matrix([[1, -sp.Rational(1, 2), 1], [0, 0, 2], [0, sp.Rational(1, 2), 0]])
    c2 = sp.Matrix(
        [
            [1, -sp.Rational(1, 4), sp.Rational(3, 4)],
            [0, sp.Rational(1, 2), sp.Rational(3, 2)],
            [0, -sp.Rational(1, 2), sp.Rational(1, 2)],
        ]
    )
    assert generalized_twin_index(c1) == 2
    assert generalized_twin_index(c2) == 4


def test_c_centered_direct_reciprocal_duality():
    node_basis = BravaisNodeBasis.c_centered_unique_b()
    u_p = sp.Matrix([2, -1, 3])
    p_c = sp.Matrix([1, 2, -1])
    u_c = node_basis.direct_to_conventional(u_p)
    p_p = node_basis.plane_to_primitive(p_c)
    assert (p_c.T * u_c)[0] == (p_p.T * u_p)[0]
    assert node_basis.determinant == sp.Rational(1, 2)


def test_ct_constrained_search_recovers_published_basal_prismatic_pair():
    correspondence = sp.Matrix(
        [
            [1, -sp.Rational(1, 2), 1],
            [0, 0, 2],
            [0, sp.Rational(1, 2), 0],
        ]
    )
    results = enumerate_ct_constrained_weak_planes(
        hcp_metric(),
        correspondence,
        [1, 0, 0],
        node_basis=BravaisNodeBasis.primitive_conventional(),
        max_plane_index=2,
        maximum_generalized_shear=0.2,
    )
    pairs = {(item.plane1_primitive, item.plane2_primitive) for item in results}
    assert ((0, 0, 1), (0, 1, 0)) in pairs


def test_noninvariant_axis_fails_loudly():
    correspondence = sp.Matrix(
        [
            [1, -sp.Rational(1, 2), 1],
            [0, 0, 2],
            [0, sp.Rational(1, 2), 0],
        ]
    )
    try:
        evaluate_axial_weak_twin(
            hcp_metric(),
            correspondence,
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            node_basis=BravaisNodeBasis.primitive_conventional(),
        )
    except ValueError as exc:
        assert "not invariant" in str(exc)
    else:
        raise AssertionError("non-invariant axis was silently accepted")
