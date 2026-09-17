import numpy as np
import sympy as sp

from cualni_cryst.cualni_models import (
    DO3_TO_6M,
    do3_to_6m_branch,
    james_hane_6m_example_lattices,
)
from cualni_cryst.group_theory import correspondence_groupoid, validate_group
from cualni_cryst.james_hane import cube_edge_6m_variants
from cualni_cryst.reference_6m import (
    A6M_IN_A,
    B6M_IN_A,
    C6M_IN_A,
    C_EXPLORATORY_ALT_M_FROM_A,
    C_REF_A_FROM_M,
    C_REF_M_FROM_A,
    DO3_6M_BASIS_CONVENTION,
    daughter_basis_in_parent_coordinates,
    general_monoclinic_metric_symbolic,
    parent_axis_permutation_relating_reference_to_alt,
    pulled_back_reference_metric_symbolic,
    reference_parent_translation_lengths,
    unique_b_6m_metric_symbolic,
)
from cualni_cryst.stretch import (
    generate_stretch_variants,
    optimal_match_matrix_family,
    stretch_from_metrics,
)
from cualni_cryst.symmetry import cubic_full_m3m, cubic_proper_rotations


def test_reference_basis_is_explicit_and_invertible():
    expected_basis = sp.Matrix.hstack(A6M_IN_A, B6M_IN_A, C6M_IN_A)
    assert daughter_basis_in_parent_coordinates() == expected_basis
    assert C_REF_A_FROM_M == expected_basis
    assert C_REF_M_FROM_A == sp.simplify(expected_basis.inv())
    assert sp.simplify(C_REF_M_FROM_A * C_REF_A_FROM_M - sp.eye(3)) == sp.zeros(3)
    assert DO3_TO_6M.C_M_from_A == C_REF_M_FROM_A


def test_reference_matrix_has_declared_basis_meaning():
    e1, e2, e3 = sp.eye(3).columnspace()
    assert DO3_TO_6M.map_direction_M_to_A(e1) == A6M_IN_A
    assert DO3_TO_6M.map_direction_M_to_A(e2) == B6M_IN_A
    assert DO3_TO_6M.map_direction_M_to_A(e3) == C6M_IN_A


def test_james_hane_stretch_scaling_follows_reference_parent_translations():
    a0, a, b, c = sp.symbols("a0 a b c", positive=True)
    La, Lb, Lc = reference_parent_translation_lengths(a0)
    assert sp.simplify(a / La - sp.sqrt(2) * a / a0) == 0
    assert sp.simplify(b / Lb - b / a0) == 0
    assert sp.simplify(c / Lc - sp.sqrt(2) * c / (3 * a0)) == 0


def test_basis_metadata_locks_monoclinic_angle_meaning():
    meta = DO3_6M_BASIS_CONVENTION
    assert meta.unique_axis == "b_6M"
    assert meta.non_right_angle_between == ("a_6M", "c_6M")
    assert meta.internal_angle_symbol == "beta"


def test_general_metric_reduces_exactly_to_unique_b_metric():
    a, b, c, beta = sp.symbols("a b c beta", positive=True, real=True)
    general = general_monoclinic_metric_symbolic(
        a, b, c, sp.pi / 2, beta, sp.pi / 2
    )
    unique_b = unique_b_6m_metric_symbolic(a, b, c, beta)
    assert sp.simplify(general - unique_b) == sp.zeros(3)
    assert unique_b == sp.Matrix(
        [
            [a**2, 0, a * c * sp.cos(beta)],
            [0, b**2, 0],
            [a * c * sp.cos(beta), 0, c**2],
        ]
    )


def test_reference_pulled_metric_exact_structure():
    a, b, c, beta = sp.symbols("a b c beta", positive=True, real=True)
    G = pulled_back_reference_metric_symbolic(a, b, c, beta)
    expected = sp.Matrix(
        [
            [b**2, 0, 0],
            [
                0,
                a**2 + sp.Rational(2, 3) * a * c * sp.cos(beta) + c**2 / 9,
                a**2 - c**2 / 9,
            ],
            [
                0,
                a**2 - c**2 / 9,
                a**2 - sp.Rational(2, 3) * a * c * sp.cos(beta) + c**2 / 9,
            ],
        ]
    )
    assert sp.simplify(G - expected) == sp.zeros(3)


def test_legacy_exploratory_matrix_is_parent_axis_permutation_equivalent():
    P = parent_axis_permutation_relating_reference_to_alt()
    assert sp.simplify(P.T * P - sp.eye(3)) == sp.zeros(3)
    assert P.det() == 1
    assert P in cubic_proper_rotations()
    assert sp.simplify(C_REF_M_FROM_A * P - C_EXPLORATORY_ALT_M_FROM_A) == sp.zeros(3)


def test_monoclinic_2_over_m_preserves_symbolic_6m_metric():
    a, b, c, beta = sp.symbols("a b c beta", positive=True, real=True)
    M = unique_b_6m_metric_symbolic(a, b, c, beta)
    branch = do3_to_6m_branch()
    validate_group(list(branch.product_point_group))
    for g in branch.product_point_group:
        assert sp.simplify(g.T * M * g - M) == sp.zeros(3)


def test_discrete_groupoid_is_derived_and_exact():
    branch = do3_to_6m_branch()
    validate_group(list(branch.parent_point_group))
    assert len(cubic_full_m3m()) == 48
    assert len(cubic_proper_rotations()) == 24

    result = correspondence_groupoid(
        list(branch.parent_point_group),
        list(branch.product_point_group),
        branch.correspondence,
    )
    assert len(result.subgroup) == 4
    assert result.n_variants == 12
    assert result.n_operators == 8
    assert sorted(len(x) for x in result.operators) == [4, 4, 4, 4, 8, 8, 8, 8]
    assert result.burnside_count == result.n_operators
    assert sum(len(x) for x in result.variants) == 48
    assert sum(len(x) for x in result.operators) == 48


def test_optimal_james_hane_family_match_is_bijective_and_roundoff_scale():
    A, M = james_hane_6m_example_lattices()
    U = stretch_from_metrics(A.metric(), M.metric(), DO3_TO_6M)
    derived = generate_stretch_variants(
        U, [np.array(q, float) for q in cubic_proper_rotations()]
    )
    reference = cube_edge_6m_variants(A.a, M.a, M.b, M.c, M.beta_deg)
    match = optimal_match_matrix_family(derived, reference, tol=1e-12)

    assert len(derived) == 12
    assert len(reference) == 12
    assert match.success
    assert len(match.mapping) == 12
    assert len({i for i, _ in match.mapping}) == 12
    assert len({j for _, j in match.mapping}) == 12
    assert match.maximum_residual < 1e-12
    assert match.rms_residual < 1e-12
