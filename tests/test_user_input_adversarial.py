import math

import numpy as np
import pytest
import sympy as sp

from cualni_cryst.user_workflow import (
    BasisVectorMapping,
    CorrespondenceInput,
    CrystalInput,
    exact_matrix3,
    exact_vector3,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("[1 0 -1]", [1, 0, -1]),
        ("(1, 2, 3)", [1, 2, 3]),
        (["1/2", "-3/2", 2], [sp.Rational(1,2), sp.Rational(-3,2), 2]),
        (np.array([1, -2, 4]), [1, -2, 4]),
    ],
)
def test_user_index_parser_accepts_common_crystallographic_forms(value, expected):
    assert exact_vector3(value) == sp.Matrix(expected)


@pytest.mark.parametrize(
    "bad",
    ["[]", "[1 2]", "[1 2 3 4]", [0, 0, 0], "", [1, math.inf, 0]],
)
def test_user_index_parser_rejects_bad_vectors(bad):
    with pytest.raises((TypeError, ValueError)):
        exact_vector3(bad)


def test_decimal_is_preserved_not_silently_rationalized_to_one_third():
    M = exact_matrix3([[1.0, 0.333333, 0], [0, 1, 0], [0, 0, 1]])
    assert M[0, 1] == sp.Rational(333333, 1000000)
    assert M[0, 1] != sp.Rational(1, 3)


@pytest.mark.parametrize(
    "matrix",
    [
        [[1,0,0],[0,0,0],[0,0,1]],
        [[1,0],[0,1]],
        [[1,0,0],[0,float("nan"),0],[0,0,1]],
        [[1,0,0],[0,float("inf"),0],[0,0,1]],
    ],
)
def test_correspondence_matrix_rejects_singular_wrong_shape_or_nonfinite(matrix):
    with pytest.raises((TypeError, ValueError)):
        CorrespondenceInput.from_matrix(
            matrix, matrix_maps="parent_to_product"
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(label="bad", point_group="1", a=0.0),
        dict(label="bad", point_group="1", a=-1.0),
        dict(label="bad", point_group="1", a=1.0, alpha_deg=0.0),
        dict(label="bad", point_group="1", a=1.0, beta_deg=180.0),
        dict(label="bad", point_group="not-a-group", a=1.0),
    ],
)
def test_crystal_input_rejects_nonphysical_or_unknown_cell_definition(kwargs):
    with pytest.raises((ValueError, KeyError)):
        CrystalInput(**kwargs)


def test_general_triclinic_cell_is_accepted_without_user_metric_tensor():
    c = CrystalInput(
        "triclinic",
        "1",
        a=4.2,
        b=5.1,
        c=6.3,
        alpha_deg=82.0,
        beta_deg=101.0,
        gamma_deg=74.0,
    )
    eig = np.linalg.eigvalsh(c.lattice.metric())
    assert np.all(eig > 0.0)


def test_matrix_direction_enum_rejects_typo_instead_of_guessing():
    with pytest.raises(ValueError, match="parent_to_product"):
        CorrespondenceInput.from_matrix(
            np.eye(3).tolist(),
            matrix_maps="parent-to-product-ish",
        )


def test_basis_mapping_scaling_is_not_treated_as_parallelism():
    a = CorrespondenceInput.from_basis_mappings(
        [
            BasisVectorMapping((1,0,0), (1,0,0)),
            BasisVectorMapping((0,1,0), (0,1,0)),
            BasisVectorMapping((0,0,1), (0,0,1)),
        ]
    )
    b = CorrespondenceInput.from_basis_mappings(
        [
            BasisVectorMapping((1,0,0), (2,0,0)),
            BasisVectorMapping((0,1,0), (0,1,0)),
            BasisVectorMapping((0,0,1), (0,0,1)),
        ]
    )
    assert a.C_product_from_parent != b.C_product_from_parent
    assert sp.simplify(b.C_product_from_parent.det() - 2) == 0


def test_payload_rejects_unknown_or_answer_bearing_fields_instead_of_ignoring_them():
    with pytest.raises(ValueError, match="Unknown correspondence input fields"):
        CorrespondenceInput.from_user_payload(
            {
                "matrix": [[1,0,0],[0,1,0],[0,0,1]],
                "matrix_maps": "parent_to_product",
                "expected_plane": [1, 2, 1],
            }
        )
