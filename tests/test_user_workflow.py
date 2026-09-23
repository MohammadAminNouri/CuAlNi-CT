import numpy as np
import pytest
import sympy as sp

from cualni_cryst.user_workflow import (
    BasisVectorMapping,
    CorrespondenceInput,
    CrystalInput,
    MatrixMapping,
    OrientationParallelisms,
    TransformationInput,
)


C_PAPER_NITI = [[0, 1, -1], [0, 1, 1], [1, 0, 0]]
C_INTERNAL_NITI = sp.Matrix(C_PAPER_NITI).inv()


def niti_input() -> TransformationInput:
    parent = CrystalInput.cubic("B2", "m-3m", 3.015)
    product = CrystalInput.monoclinic_unique_b(
        "B19-prime", "2/m", 2.889, 4.120, 4.622, 96.8
    )
    corr = CorrespondenceInput.from_matrix(
        C_PAPER_NITI,
        matrix_maps=MatrixMapping.PRODUCT_TO_PARENT,
        verification_mappings=[
            BasisVectorMapping((1, 1, 0), (0, 1, 0)),
            BasisVectorMapping((0, 0, 1), (1, 0, 0)),
            BasisVectorMapping((-1, 1, 0), (0, 0, 1)),
        ],
    )
    return TransformationInput(parent, product, corr)


def test_product_to_parent_matrix_compiles_to_repository_parent_to_product_convention():
    compiled = niti_input().compile()
    C = compiled.project.transformation("A_to_M").correspondence.C_M_from_A
    assert sp.simplify(C - C_INTERNAL_NITI) == sp.zeros(3)


def test_same_physics_from_explicit_inverse_matrix_direction():
    original = niti_input()
    parent_to_product = CorrespondenceInput.from_matrix(
        C_INTERNAL_NITI.tolist(), matrix_maps="parent_to_product"
    )
    equivalent = TransformationInput(
        original.parent, original.product, parent_to_product
    )
    C1 = original.compile().project.transformation("A_to_M").correspondence.C_M_from_A
    C2 = equivalent.compile().project.transformation("A_to_M").correspondence.C_M_from_A
    assert sp.simplify(C1 - C2) == sp.zeros(3)


def test_three_basis_mappings_derive_same_exact_correspondence():
    mappings = [
        BasisVectorMapping((1, 1, 0), (0, 1, 0)),
        BasisVectorMapping((0, 0, 1), (1, 0, 0)),
        BasisVectorMapping((-1, 1, 0), (0, 0, 1)),
    ]
    derived = CorrespondenceInput.from_basis_mappings(mappings)
    assert sp.simplify(
        derived.C_product_from_parent - C_INTERNAL_NITI
    ) == sp.zeros(3)
    assert derived.audit().passed


def test_bare_matrix_is_rejected_at_ui_boundary():
    with pytest.raises(ValueError, match="ambiguous"):
        CorrespondenceInput.from_user_payload({"matrix": C_PAPER_NITI})


def test_both_matrix_and_basis_mappings_are_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        CorrespondenceInput.from_user_payload(
            {
                "matrix": C_PAPER_NITI,
                "matrix_maps": "product_to_parent",
                "basis_mappings": [
                    {"parent": [1, 0, 0], "product": [1, 0, 0]},
                    {"parent": [0, 1, 0], "product": [0, 1, 0]},
                    {"parent": [0, 0, 1], "product": [0, 0, 1]},
                ],
            }
        )


def test_dependent_basis_mappings_are_rejected():
    with pytest.raises(ValueError, match="linearly dependent"):
        CorrespondenceInput.from_basis_mappings(
            [
                BasisVectorMapping((1, 0, 0), (1, 0, 0)),
                BasisVectorMapping((2, 0, 0), (0, 1, 0)),
                BasisVectorMapping((0, 0, 1), (0, 0, 1)),
            ]
        )


def test_contradictory_verification_mapping_is_rejected_before_theory_runs():
    corr = CorrespondenceInput.from_matrix(
        C_PAPER_NITI,
        matrix_maps="product_to_parent",
        verification_mappings=[
            BasisVectorMapping((1, 1, 0), (1, 0, 0)),
        ],
    )
    assert not corr.audit().passed
    with pytest.raises(ValueError, match="Invalid physical correspondence"):
        TransformationInput(
            CrystalInput.cubic("B2", "m-3m", 3.015),
            CrystalInput.monoclinic_unique_b(
                "B19-prime", "2/m", 2.889, 4.120, 4.622, 96.8
            ),
            corr,
        ).compile()


def test_normal_user_never_selects_cartesian_frame_but_all_conventions_are_crosschecked():
    compiled = niti_input().compile()
    audit = compiled.representation_audit()
    assert audit.convention_pairs_checked == 9
    assert audit.maximum_internal_parity_residual < 1e-9
    assert audit.maximum_principal_stretch_disagreement < 1e-9
    assert audit.passed


def test_user_analysis_returns_crystallographic_summary_not_metric_tensor():
    result = niti_input().compile().analyze().to_dict()
    assert result["subgroup_order"] == 4
    assert result["variant_count"] == 12
    assert result["operator_count"] == 7
    assert len(result["principal_stretches"]) == 3
    assert result["representation_consistent"] is True
    assert "metric_tensor" not in result
    assert "cartesian_matrix" not in result


def test_parallelism_facade_recovers_ks_topology_without_cartesian_input():
    transform = TransformationInput(
        CrystalInput.cubic("gamma", "m-3m", 1.0),
        CrystalInput.cubic("alpha", "m-3m", 1.0),
        CorrespondenceInput.from_matrix(
            np.eye(3, dtype=int).tolist(),
            matrix_maps="parent_to_product",
        ),
    ).compile()
    result = transform.analyze_orientation(
        OrientationParallelisms(
            parent_plane="(1 1 1)",
            product_plane="(1 1 0)",
            parent_direction="[1 -1 0]",
            product_direction="[1 -1 1]",
        )
    )
    assert result.candidate_count > 0
    assert {c.intersection_order for c in result.candidates} == {2}
    assert {c.variant_count for c in result.candidates} == {24}
    assert {c.operator_count for c in result.candidates} == {24}


def test_orientation_reversing_correspondence_is_rejected_at_normal_user_boundary():
    corr = CorrespondenceInput.from_matrix(
        [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
        matrix_maps="parent_to_product",
    )
    audit = corr.audit()
    assert audit.determinant_exact == "-1"
    assert audit.orientation_preserving is False
    assert audit.passed is False

    with pytest.raises(
        ValueError,
        match=r"det\(C_M_from_A\) must be positive",
    ):
        TransformationInput(
            CrystalInput.cubic("A", "m-3m", 3.0),
            CrystalInput.cubic("M", "m-3m", 3.0),
            corr,
        ).compile()
