import numpy as np

from cualni_cryst.cualni_models import (
    do3_to_2h_branch,
    do3_to_6m_branch,
    james_hane_6m_example_lattices,
)
from cualni_cryst.group_theory import correspondence_groupoid
from cualni_cryst.lattice import Lattice
from cualni_cryst.twinning_ct import twins_from_operator


def test_ct_twin_predictions_are_metric_normalized():
    A,M=james_hane_6m_example_lattices();b=do3_to_6m_branch();g=correspondence_groupoid(list(b.parent_point_group),list(b.product_point_group),b.correspondence)
    found=0
    for op in g.operators:
        for tw in twins_from_operator(op,A.metric(),M.metric(),b.correspondence):
            found+=1
            assert tw.shear >= 0
            assert np.isfinite(tw.shear)
            assert abs(tw.direction_m @ M.metric() @ tw.direction_m - 1) < 1e-8
            assert abs(tw.plane_m @ np.linalg.inv(M.metric()) @ tw.plane_m - 1) < 1e-8
    assert found > 0


def _projectively_parallel(vector, target, tol=1.0e-8):
    v = np.asarray(vector, dtype=float).reshape(3)
    t = np.asarray(target, dtype=float).reshape(3)
    return np.linalg.norm(np.cross(v, t)) <= tol * np.linalg.norm(v) * np.linalg.norm(t)


def test_raw_ct_twin_classification_preserves_construction_route():
    A, M = james_hane_6m_example_lattices()
    branch = do3_to_6m_branch()
    groupoid = correspondence_groupoid(
        list(branch.parent_point_group),
        list(branch.product_point_group),
        branch.correspondence,
    )
    twins = [
        twin
        for operator in groupoid.operators
        for twin in twins_from_operator(
            operator, A.metric(), M.metric(), branch.correspondence
        )
    ]
    assert twins
    assert all(twin.kind in {"I", "II"} for twin in twins)
    assert all(
        twin.classification in {"type_I", "type_II", "compound"}
        for twin in twins
    )
    assert any(twin.compound for twin in twins)
    assert all(
        twin.representations == ("I", "II")
        for twin in twins if twin.compound
    )


def test_blind_2h_ct_recovers_chen_101_as_compound_without_using_chen_output():
    # Independent Landa et al. 2H metric. Chen EBSD twin labels are not input.
    A = Lattice.cubic(5.835, length_unit="angstrom")
    M = Lattice.orthorhombic(4.389, 5.342, 4.224, length_unit="angstrom")
    branch = do3_to_2h_branch()
    groupoid = correspondence_groupoid(
        list(branch.parent_point_group),
        list(branch.product_point_group),
        branch.correspondence,
    )

    assert len(groupoid.subgroup) == 8
    assert groupoid.n_variants == 6
    assert groupoid.n_operators == 3

    twins = [
        twin
        for operator in groupoid.operators
        for twin in twins_from_operator(
            operator, A.metric(), M.metric(), branch.correspondence
        )
    ]

    family_101 = [
        twin for twin in twins
        if _projectively_parallel(twin.plane_m, [1.0, 0.0, 1.0])
    ]
    assert family_101
    assert {twin.kind for twin in family_101} == {"I", "II"}
    assert all(twin.compound for twin in family_101)
    assert all(twin.representations == ("I", "II") for twin in family_101)
    assert all(
        _projectively_parallel(twin.direction_m, [1.0, 0.0, -1.0])
        for twin in family_101
    )

    family_121 = [
        twin for twin in twins
        if _projectively_parallel(twin.plane_m, [1.0, 2.0, 1.0])
    ]
    assert family_121
    assert any(twin.kind == "I" and not twin.compound for twin in family_121)
    type_i_121 = next(
        twin for twin in family_121
        if twin.kind == "I" and not twin.compound
    )
    # Regression for the Landa 2007 lattice constants used in THIS test.
    # This is not the 0.2606669 value from the separate blind Chen run.
    assert np.isclose(
        type_i_121.shear,
        0.2682091893754589,
        atol=5.0e-12,
        rtol=0.0,
    )

