
import numpy as np

from cualni_cryst.ptmc_double_shear import (
    DoubleShearComposition,
    DoubleShearParameterSemantics,
    DoubleShearSystem,
    DoubleShearSystemSource,
    solve_double_shear_ptmc,
)


def test_additive_double_shear_is_root_complete_for_fixed_second_parameter():
    U = np.diag([0.8, 0.9, 1.2])
    first = DoubleShearSystem(
        a=(0.0, 0.1, 0.0),
        n=(0.0, 1.0, 0.0),
        label="synthetic first rank-one branch",
    )
    second = DoubleShearSystem(
        a=(0.1, 0.0, 0.0),
        n=(1.0, 0.0, 0.0),
        label="synthetic second rank-one branch",
    )
    report = solve_double_shear_ptmc(
        U,
        first,
        second,
        fixed_second_parameter=0.0,
        composition=DoubleShearComposition.ADDITIVE_LAMINATE,
        tolerance=1e-8,
    )
    assert report.continuum is None
    assert report.solutions
    target = min(report.solutions, key=lambda item: abs(item.first_parameter - 1.0))
    assert abs(target.first_parameter - 1.0) < 1e-8
    assert target.middle_stretch_residual < 1e-8
    assert target.habit_connections
    for habit in target.habit_connections:
        assert abs(np.linalg.norm(habit.habit_normal_parent_cartesian) - 1.0) < 1e-12
        assert abs(
            habit.shape_vector_magnitude
            - np.linalg.norm(habit.shape_vector_parent_cartesian)
        ) < 1e-12


def test_sequential_composition_has_the_same_single_shear_limit_when_second_is_zero():
    U = np.diag([0.8, 0.9, 1.2])
    first = DoubleShearSystem(
        a=(0.0, 0.1, 0.0),
        n=(0.0, 1.0, 0.0),
        label="first",
    )
    second = DoubleShearSystem(
        a=(0.1, 0.0, 0.0),
        n=(1.0, 0.0, 0.0),
        label="second",
    )
    additive = solve_double_shear_ptmc(
        U,
        first,
        second,
        fixed_second_parameter=0.0,
        composition=DoubleShearComposition.ADDITIVE_LAMINATE,
        tolerance=1e-8,
    )
    sequential = solve_double_shear_ptmc(
        U,
        first,
        second,
        fixed_second_parameter=0.0,
        composition=DoubleShearComposition.SEQUENTIAL_SIMPLE_SHEAR,
        tolerance=1e-8,
    )
    assert len(additive.solutions) == len(sequential.solutions)
    assert np.allclose(
        [item.first_parameter for item in additive.solutions],
        [item.first_parameter for item in sequential.solutions],
        atol=1e-10,
    )
    for first_solution, second_solution in zip(
        additive.solutions, sequential.solutions, strict=True
    ):
        assert np.allclose(
            first_solution.pre_shape_deformation,
            second_solution.pre_shape_deformation,
            atol=1e-12,
        )


def test_three_variant_fraction_semantics_requires_explicit_common_base_provenance():
    U = np.diag([0.8, 0.9, 1.2])
    first = DoubleShearSystem(
        a=(0.0, 0.1, 0.0),
        n=(0.0, 1.0, 0.0),
        label="unprovenanced first",
    )
    second = DoubleShearSystem(
        a=(0.1, 0.0, 0.0),
        n=(1.0, 0.0, 0.0),
        label="unprovenanced second",
    )
    import pytest
    with pytest.raises(ValueError, match="variant_rank_one_increment"):
        solve_double_shear_ptmc(
            U,
            first,
            second,
            fixed_second_parameter=0.1,
            composition=DoubleShearComposition.ADDITIVE_LAMINATE,
            parameter_semantics=DoubleShearParameterSemantics.THREE_VARIANT_FRACTIONS,
        )


def test_three_variant_fraction_semantics_accepts_audited_common_base_relations():
    U = np.diag([0.8, 0.9, 1.2])
    first = DoubleShearSystem(
        a=(0.0, 0.1, 0.0),
        n=(0.0, 1.0, 0.0),
        label="variant 0 to 1",
        source=DoubleShearSystemSource.VARIANT_RANK_ONE_INCREMENT,
        base_variant_index=0,
        other_variant_index=1,
        source_rank_one_residual=1e-12,
    )
    second = DoubleShearSystem(
        a=(0.1, 0.0, 0.0),
        n=(1.0, 0.0, 0.0),
        label="variant 0 to 2",
        source=DoubleShearSystemSource.VARIANT_RANK_ONE_INCREMENT,
        base_variant_index=0,
        other_variant_index=2,
        source_rank_one_residual=1e-12,
    )
    report = solve_double_shear_ptmc(
        U,
        first,
        second,
        fixed_second_parameter=0.1,
        composition=DoubleShearComposition.ADDITIVE_LAMINATE,
        parameter_semantics=DoubleShearParameterSemantics.THREE_VARIANT_FRACTIONS,
    )
    assert report.parameter_semantics is DoubleShearParameterSemantics.THREE_VARIANT_FRACTIONS
