import numpy as np
import sympy as sp

from cualni_cryst.weak_operator_engine import (
    audit_parent_symmetry_element,
    exact_matrix_order,
    generalized_twin_lattice_audit,
)


def cubic_metric():
    return np.eye(3)


def hex_metric():
    return np.array(
        [
            [1.0, -0.5, 0.0],
            [-0.5, 1.0, 0.0],
            [0.0, 0.0, 2.25],
        ]
    )


def test_exact_finite_orders_cover_crystallographic_rotation_orders():
    identity = sp.eye(3)
    twofold_z = sp.diag(-1, -1, 1)
    threefold_xyz = sp.Matrix([[0,0,1],[1,0,0],[0,1,0]])
    fourfold_z = sp.Matrix([[0,-1,0],[1,0,0],[0,0,1]])
    sixfold_hex = sp.Matrix([[1,-1,0],[1,0,0],[0,0,1]])

    assert exact_matrix_order(identity) == 1
    assert exact_matrix_order(twofold_z) == 2
    assert exact_matrix_order(threefold_xyz) == 3
    assert exact_matrix_order(fourfold_z) == 4
    assert exact_matrix_order(sixfold_hex) == 6


def test_route_classification_distinguishes_classical_higher_order_and_improper():
    mirror_x = sp.diag(-1, 1, 1)
    twofold_z = sp.diag(-1, -1, 1)
    threefold_xyz = sp.Matrix([[0,0,1],[1,0,0],[0,1,0]])
    fourfold_z = sp.Matrix([[0,-1,0],[1,0,0],[0,0,1]])
    improper_fourfold = -fourfold_z

    assert (
        audit_parent_symmetry_element(
            mirror_x, cubic_metric()
        ).route == "type_I_reflection"
    )
    assert (
        audit_parent_symmetry_element(
            twofold_z, cubic_metric()
        ).route == "type_II_twofold"
    )
    assert (
        audit_parent_symmetry_element(
            threefold_xyz, cubic_metric()
        ).route == "axial_weak_rotation"
    )
    assert (
        audit_parent_symmetry_element(
            fourfold_z, cubic_metric()
        ).route == "axial_weak_rotation"
    )
    assert (
        audit_parent_symmetry_element(
            improper_fourfold, cubic_metric()
        ).route == "improper_higher_order"
    )


def test_hexagonal_sixfold_rotation_requires_and_preserves_hexagonal_metric():
    sixfold_hex = sp.Matrix([[1,-1,0],[1,0,0],[0,0,1]])
    audit = audit_parent_symmetry_element(sixfold_hex, hex_metric())
    assert audit.order == 6
    assert audit.route == "axial_weak_rotation"
    assert audit.axis_parent == (0, 0, 1)
    assert audit.metric_preservation_residual < 1e-12


def test_symmetry_element_is_rejected_if_it_does_not_preserve_supplied_metric():
    fourfold_z = sp.Matrix([[0,-1,0],[1,0,0],[0,0,1]])
    orthorhombic = np.diag([1.0, 4.0, 9.0])
    try:
        audit_parent_symmetry_element(fourfold_z, orthorhombic)
    except ValueError as exc:
        assert "does not preserve" in str(exc)
    else:
        raise AssertionError("non-symmetry operation was silently accepted")


def test_generalized_twin_index_is_invariant_under_unimodular_basis_conjugation():
    C = sp.Matrix(
        [
            [1, sp.Rational(1,2), 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    baseline = generalized_twin_lattice_audit(C)
    assert baseline.equal_volume
    assert baseline.q_g is not None

    transforms = [
        sp.Matrix([[1,1,0],[0,1,0],[0,0,1]]),
        sp.Matrix([[0,1,0],[1,0,0],[0,0,1]]),
        sp.diag(-1,1,1),
    ]
    for P in transforms:
        transformed = sp.simplify(P.inv() * C * P)
        audit = generalized_twin_lattice_audit(transformed)
        assert audit.equal_volume
        assert audit.q_g == baseline.q_g
        assert audit.domain.index == baseline.domain.index
        assert audit.codomain.index == baseline.codomain.index


def test_non_equal_volume_map_is_not_mislabeled_as_generalized_twin_index():
    audit = generalized_twin_lattice_audit(sp.diag(2,1,1))
    assert not audit.equal_volume
    assert audit.q_g is None
    assert audit.domain.index >= 1
    assert audit.codomain.index >= 1
