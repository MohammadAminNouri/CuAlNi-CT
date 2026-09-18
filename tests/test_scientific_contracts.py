import json

import numpy as np

from cualni_cryst.compatibility_atlas import james_hane_benchmark_projection
from cualni_cryst.cualni_models import (
    do3_to_6m_branch,
    james_hane_6m_example_lattices,
)
from cualni_cryst.lattice import Lattice
from cualni_cryst.numerics import (
    DEFAULT_NUMERICAL_POLICY,
    NumericalPolicy,
    ResidualClass,
    classify_residual,
)
from cualni_cryst.scientific_contracts import audit_core_theory_consistency


def test_numerical_policy_rejects_nonpositive_tolerances():
    try:
        NumericalPolicy(algebraic=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero algebraic tolerance must be rejected")


def test_residual_classification_always_depends_on_explicit_tolerance():
    assert classify_residual(1e-12, 1e-10) is ResidualClass.PASS
    assert classify_residual(2e-10, 1e-10) is ResidualClass.NEAR
    assert classify_residual(2e-8, 1e-10) is ResidualClass.FAIL


def test_source_rounded_do3_6m_state_passes_implementation_contracts():
    A, M = james_hane_6m_example_lattices()
    branch = do3_to_6m_branch()

    audit = audit_core_theory_consistency(A, M, branch.correspondence)

    audit.assert_passed()
    assert audit.representation_pairs_checked == 9
    assert audit.maximum_algebraic_residual <= DEFAULT_NUMERICAL_POLICY.algebraic
    assert (
        audit.representation_max_residual
        <= DEFAULT_NUMERICAL_POLICY.representation
    )

    # The rounded literature benchmark is not silently called exact.
    assert audit.lambda2_residual > DEFAULT_NUMERICAL_POLICY.exact_eigenvalue
    assert audit.cmc_nearest_zero_residual > 0.0
    assert audit.exact_compatibility_agreement


def test_exact_beta_projection_passes_same_contracts_and_exact_classification():
    result = james_hane_benchmark_projection()
    A, M_source = james_hane_6m_example_lattices()

    M_exact = Lattice.monoclinic_unique_b(
        M_source.a,
        M_source.b,
        M_source.c,
        result.projected_beta_deg,
        label="HYPOTHETICAL_TEST exact-compatible 6M",
    )
    branch = do3_to_6m_branch()

    audit = audit_core_theory_consistency(A, M_exact, branch.correspondence)

    audit.assert_passed()
    assert audit.lambda2_residual <= DEFAULT_NUMERICAL_POLICY.exact_eigenvalue
    assert audit.cmc_nearest_zero_residual <= DEFAULT_NUMERICAL_POLICY.exact_eigenvalue
    assert audit.exact_compatibility_agreement


def test_contracts_are_not_specific_to_cubic_parent():
    # Generic non-cubic positive-definite lattices + identity correspondence.
    # This test guards the general metric identities themselves.
    import sympy as sp

    from cualni_cryst.correspondence import Correspondence

    A = Lattice(
        4.2,
        5.1,
        6.3,
        alpha_deg=82.0,
        beta_deg=101.0,
        gamma_deg=74.0,
        label="generic parent",
    )
    M = Lattice(
        4.3,
        5.0,
        6.4,
        alpha_deg=83.0,
        beta_deg=99.0,
        gamma_deg=75.0,
        label="generic product",
    )
    C = Correspondence(sp.eye(3), label="identity generic test")

    audit = audit_core_theory_consistency(A, M, C)

    audit.assert_passed()
    assert audit.maximum_algebraic_residual < 1e-9


def test_audit_payload_is_json_serializable_for_future_reliability_panel():
    A, M = james_hane_6m_example_lattices()
    branch = do3_to_6m_branch()

    payload = audit_core_theory_consistency(
        A,
        M,
        branch.correspondence,
    ).to_dict()

    encoded = json.dumps(payload)
    assert '"passed": true' in encoded
    assert '"policy"' in encoded


def test_cmc_eigenvalues_follow_lambda_squared_minus_one_directly():
    A, M = james_hane_6m_example_lattices()
    branch = do3_to_6m_branch()

    audit = audit_core_theory_consistency(A, M, branch.correspondence)

    assert np.isfinite(audit.cmc_eigenvalue_vs_lambda_residual)
    assert audit.cmc_eigenvalue_vs_lambda_residual < 1e-10
