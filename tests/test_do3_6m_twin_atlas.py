import numpy as np

from cualni_cryst.cualni_models import (
    do3_to_6m_branch,
    james_hane_6m_example_lattices,
)
from cualni_cryst.twin_compare import build_twin_atlas


def _atlas():
    A, M = james_hane_6m_example_lattices()
    return build_twin_atlas(do3_to_6m_branch(), A.metric(), M.metric())


def test_operator_classification_for_truth_locked_6m_reference():
    atlas = _atlas()
    classification = {
        op.operator_index: op.classification for op in atlas.operators
    }
    assert classification == {
        0: "neutral",
        1: "compound",
        2: "compound",
        3: "weak_candidate_non_twofold",
        4: "type_I_II",
        5: "weak_candidate_non_twofold",
        6: "weak_candidate_non_twofold",
        7: "type_I_II",
    }


def test_exact_relation_multiplicities_and_compound_definition():
    atlas = _atlas()
    relations_per_operator = {
        op.operator_index: len(op.relations) for op in atlas.operators
    }
    assert relations_per_operator == {
        0: 0,
        1: 1,
        2: 1,
        3: 0,
        4: 2,
        5: 0,
        6: 0,
        7: 2,
    }
    assert atlas.n_exact_relations == 6

    compound = {
        op.operator_index: [rel.compound for rel in op.relations]
        for op in atlas.operators
        if op.relations
    }
    assert compound[1] == [True]
    assert compound[2] == [True]
    assert compound[4] == [False, False]
    assert compound[7] == [False, False]

    # Compound is detected by multiple distinct twofold generators of the SAME
    # stretch pair, not merely by seeing a reflection and a rotation in a double coset.
    assert len(atlas.operators[1].relations[0].generator_axes) == 2
    assert len(atlas.operators[2].relations[0].generator_axes) == 2
    assert all(len(r.generator_axes) == 1 for r in atlas.operators[4].relations)
    assert all(len(r.generator_axes) == 1 for r in atlas.operators[7].relations)


def test_ct_and_mallard_agree_for_every_exact_relation():
    atlas = _atlas()
    assert atlas.max_relative_shear_residual < 1e-11
    assert atlas.max_geometry_angle_deg < 1e-8
    assert atlas.max_rank_one_residual < 1e-11

    for op in atlas.operators:
        for rel in op.relations:
            assert rel.ct_type_i.shear > 0.0
            assert rel.ct_type_ii.shear > 0.0
            assert rel.ct_type_i_vs_ii_shear_abs < 1e-11

            assert rel.type_i_shear_rel_residual < 1e-11
            assert rel.type_ii_shear_rel_residual < 1e-11

            # Type-I: CT K1 reference plane == Mallard reference rank-one normal.
            assert rel.type_i_plane_angle_deg < 1e-8
            # CT eta1 is stored in reference coordinates and must be pushed by Uj
            # before comparison with Ball-James current/deformed shear vector a.
            assert rel.type_i_direction_angle_deg < 1e-8

            # Type-II: same coordinate logic, with CT K2 and eta2.
            assert rel.type_ii_plane_angle_deg < 1e-8
            assert rel.type_ii_direction_angle_deg < 1e-8

            assert rel.bj_type_i.residual < 1e-11
            assert rel.bj_type_ii.residual < 1e-11
            assert rel.bj_type_i_orthogonality_residual < 1e-11
            assert rel.bj_type_ii_orthogonality_residual < 1e-11
            assert rel.bj_type_i_determinant_residual < 1e-11
            assert rel.bj_type_ii_determinant_residual < 1e-11


def test_benchmark_operator_shear_values_are_stable_regression_values():
    """Regression values from source-rounded James-Hane Table-4 lattice data.

    These are NOT universal Cu-Al-Ni constants. They are computation-derived
    numbers for the single literature benchmark and are kept only to catch
    accidental convention changes.
    """

    atlas = _atlas()
    shear_by_operator = {
        op.operator_index: op.relations[0].shear
        for op in atlas.operators
        if op.relations
    }
    expected = {
        1: 0.07709383077395,
        2: 0.19892147651004,
        4: 0.18908990849744,
        7: 0.38548856778794,
    }
    assert shear_by_operator.keys() == expected.keys()
    for k, value in expected.items():
        assert np.isclose(shear_by_operator[k], value, rtol=0.0, atol=5e-13)


def test_non_twofold_operators_are_not_mislabeled_as_exact_twins():
    atlas = _atlas()
    for k in (3, 5, 6):
        op = atlas.operators[k]
        assert op.classification == "weak_candidate_non_twofold"
        assert not op.relations
        assert all(order != 2 for order in op.proper_rotation_orders)
