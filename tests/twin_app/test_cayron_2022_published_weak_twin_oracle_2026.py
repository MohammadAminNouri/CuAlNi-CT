"""Independent PRIMARY-SOURCE NiTi weak-twin oracle: Cayron, Crystals 12 (2022), 130.

The published numbers are asserted only AFTER calling the implementation.
Do not substitute the published values as runtime model outputs.
Reference: DOI 10.3390/cryst12020130, Section 5 and Tables 3/5.
2019 matrix conventions: DOI 10.1107/S205327331900038X (with corrigendum).
"""
from __future__ import annotations

import itertools
import math
import numpy as np
import pytest
import sympy as sp

from cualni_cryst.weak_twins import (
    BravaisNodeBasis, evaluate_axial_weak_twin,
    generalized_twin_index, generalized_strain,
)
from twin_app.cayron_correspondence import calculate_from_paired_vectors
from twin_app.input_logic import exact_matrix


# Published B19' NiTi inter-correspondence for the [011] weak-twin operator.
# This is NOT the parent->martensite transformation correspondence.
PUBLISHED_C_INT = sp.Matrix([
    [0, -1, 1],
    [sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(1, 2)],
    [-sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(1, 2)],
])
AXIS = sp.Matrix([0, 1, 1])
PLANE_1 = sp.Matrix([1, -3, 3])
PLANE_2 = sp.Matrix([3, 1, -1])


def _published_metric(beta_deg=96.8):
    a, b, c = 2.889, 4.120, 4.622
    cross = a * c * math.cos(math.radians(beta_deg))
    return np.array([[a*a, 0.0, cross], [0.0, b*b, 0.0],
                     [cross, 0.0, c*c]], dtype=float)


def _vectors_as_columns(M):
    return [[str(M[i, j]) for i in range(3)] for j in range(3)]


def test_cayron_2022_exact_supercell_directions_and_reciprocal_planes():
    # Cayron's printed inter-correspondence matrix is the authoritative input.
    # Applying that matrix gives [-1, 1, -2] for [3, 0, -1]; a prose
    # transcription showing [-1, -1, -2] is inconsistent with this matrix.
    # Do not change the expected result to make it match that printed sign.
    C = PUBLISHED_C_INT
    assert C.det() == 1
    assert C * AXIS == AXIS
    assert C * sp.Matrix([3, 1, 0]) == sp.Matrix([-1, 2, -1])
    assert C * sp.Matrix([3, 0, -1]) == sp.Matrix([-1, 1, -2])
    assert 2 * C * sp.Matrix([1, 0, 0]) == sp.Matrix([0, 1, -1])
    assert C.inv().T * PLANE_1 == PLANE_2
    assert (PLANE_1.T * AXIS)[0] == (PLANE_2.T * AXIS)[0] == 0

    # Actual Cayron paired-vector input adapter, non-identity source supercell.
    source = sp.Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    destination = C * source
    result = calculate_from_paired_vectors(
        _vectors_as_columns(source), _vectors_as_columns(destination),
        direction="A_TO_M",
    )
    assert exact_matrix(result.canonical_M_from_A) == C
    # The same exact operation entered backwards MUST return canonical C.
    reverse = calculate_from_paired_vectors(
        _vectors_as_columns(destination), _vectors_as_columns(source),
        direction="M_TO_A",
    )
    assert exact_matrix(reverse.canonical_M_from_A) == C


def test_cayron_2022_published_weak_twin_index_strain_and_shear():
    M = _published_metric()
    result = evaluate_axial_weak_twin(
        M, PUBLISHED_C_INT, AXIS, PLANE_1, PLANE_2,
        node_basis=BravaisNodeBasis.primitive_conventional(),
    )
    # Printed results for Cayron's selected axial weak plane (1 -3 3)//(3 1 -1).
    assert result.generalized_twin_index == 2
    assert generalized_twin_index(PUBLISHED_C_INT) == 2
    assert result.generalized_strain == pytest.approx(0.2912, abs=0.00012)
    assert generalized_strain(M, PUBLISHED_C_INT) == pytest.approx(0.2912534777123, abs=1e-10)
    assert result.selected.generalized_shear == pytest.approx(0.2911, abs=0.00012)
    assert result.selected.generalized_shear == pytest.approx(0.2911398202403, abs=1e-10)
    assert result.selected.determinant == pytest.approx(1.0, abs=1e-10)
    assert result.selected.residuals.maximum < 1e-9
    assert result.axis_primitive == (0, 1, 1)
    assert result.plane1_primitive == (1, -3, 3)
    assert result.plane2_primitive == (3, 1, -1)


def test_cayron_2022_wrong_plane_and_wrong_correspondence_fail_closed():
    M = _published_metric()
    kwargs = dict(node_basis=BravaisNodeBasis.primitive_conventional())
    with pytest.raises(ValueError, match="weak plane|projectively"):
        evaluate_axial_weak_twin(M, PUBLISHED_C_INT, AXIS, PLANE_1,
                                (3, 1, 1), **kwargs)
    with pytest.raises(ValueError, match="equal volume"):
        evaluate_axial_weak_twin(M, PUBLISHED_C_INT + sp.eye(3),
                                AXIS, PLANE_1, PLANE_2, **kwargs)


def test_cayron_rational_elements_metric_independent_but_shear_is_not():
    # This is a numerical counterexample to confusing a rational correspondence
    # invariant with the metric-dependent generalized shear.
    kwargs = dict(node_basis=BravaisNodeBasis.primitive_conventional())
    reference = evaluate_axial_weak_twin(
        _published_metric(), PUBLISHED_C_INT, AXIS, PLANE_1, PLANE_2, **kwargs
    )
    altered = evaluate_axial_weak_twin(
        _published_metric(98.3), PUBLISHED_C_INT, AXIS, PLANE_1, PLANE_2, **kwargs
    )
    assert altered.correspondence_primitive_exact == reference.correspondence_primitive_exact
    assert altered.plane1_primitive == reference.plane1_primitive
    assert altered.plane2_primitive == reference.plane2_primitive
    assert abs(altered.selected.generalized_shear - reference.selected.generalized_shear) > 0.025
