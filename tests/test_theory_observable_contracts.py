import numpy as np
import sympy as sp
import pytest

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.ct import smc
from cualni_cryst.lattice import Lattice, metric_sqrt, normalize_plane
from cualni_cryst.theory_observable_contracts import (
    ShapeVectorRole,
    audit_smc_rank_one_bridge,
    canonical_shape_vector_role,
    infer_native_shape_vector_role,
    rank_one_shape_from_smc_cartesian,
    smc_vector_from_rank_one_shape_cartesian,
    smc_vector_from_stretch_cartesian,
)


def _exact_rank_one_state():
    n = np.array([1.0, -2.0, 0.5], dtype=float)
    n /= np.linalg.norm(n)
    b = np.array([0.075, 0.021, -0.034], dtype=float)
    F = np.eye(3) + np.outer(b, n)
    C = F.T @ F
    w, V = np.linalg.eigh(C)
    U = (V * np.sqrt(w)) @ V.T
    return n, b, F, U


def test_shape_vector_role_is_canonical_and_fail_loud():
    assert canonical_shape_vector_role(ShapeVectorRole.CT_SMC_IPS_D) == "ct_smc_ips_d"
    assert canonical_shape_vector_role("rank_one_parent_identity_b") == "rank_one_parent_identity_b"
    assert canonical_shape_vector_role(None) is None
    with pytest.raises(ValueError):
        canonical_shape_vector_role("shape-ish-vector")


def test_bridge_is_crosslocked_to_direct_cauchy_green_not_only_roundtrip():
    n, b, F, U = _exact_rank_one_state()
    d_direct = smc_vector_from_stretch_cartesian(U, n)
    J = float(np.linalg.det(U))
    d_from_b = smc_vector_from_rank_one_shape_cartesian(b, n, J)
    assert np.allclose(d_from_b, d_direct, atol=3e-14, rtol=3e-13)

    # The original bug would identify d with b.  This deliberately kills that
    # semantic mutation on a nontrivial exact IPS.
    assert np.linalg.norm(d_direct - b) > 1e-4

    recovered = rank_one_shape_from_smc_cartesian(d_direct, n, J)
    assert np.allclose(recovered, b, atol=3e-14, rtol=3e-13)

    audit = audit_smc_rank_one_bridge(d_direct, n, U)
    assert audit.source_smc_relative_residual < 1e-12
    assert audit.roundtrip_relative_residual < 1e-12
    assert audit.cauchy_green_relative_residual < 1e-12
    assert audit.rotation_orthogonality_residual < 1e-12
    assert audit.rotation_determinant_residual < 1e-12

    R = F @ np.linalg.inv(U)
    assert np.linalg.norm(R.T @ R - np.eye(3), ord="fro") < 1e-12
    assert abs(np.linalg.det(R) - 1.0) < 1e-12


def test_bridge_respects_projective_rank_one_gauge():
    n, b, _, U = _exact_rank_one_state()
    d = smc_vector_from_stretch_cartesian(U, n)
    J = float(np.linalg.det(U))
    b1 = rank_one_shape_from_smc_cartesian(d, n, J)
    b2 = rank_one_shape_from_smc_cartesian(-d, -n, J)
    assert np.allclose(np.outer(b1, n), np.outer(b2, -n), atol=3e-14, rtol=3e-13)


def test_ct_smc_physicalization_is_correct_for_oblique_parent_metric():
    # This checks the actual crystallographic SMC implementation, units and
    # direct/reciprocal basis transformations for a non-cubic parent metric.
    parent = Lattice.monoclinic_unique_b(4.3, 5.1, 6.4, 103.0)
    M_a = parent.metric()
    B = metric_sqrt(M_a)

    n, b, _, U = _exact_rank_one_state()
    pulled_product_metric = B @ (U @ U) @ B
    correspondence = Correspondence(sp.eye(3), label="identity_test")

    # p=B n is the reciprocal covector whose physical normal in the symmetric
    # metric frame is n; it is already unit in the reciprocal metric.
    p_a = normalize_plane(B @ n, M_a)
    d_crystal = smc(M_a, pulled_product_metric, correspondence) @ p_a
    d_cartesian_from_ct = B @ d_crystal
    d_cartesian_direct = smc_vector_from_stretch_cartesian(U, n)

    assert np.allclose(
        d_cartesian_from_ct,
        d_cartesian_direct,
        atol=5e-12,
        rtol=5e-12,
    )

    audit = audit_smc_rank_one_bridge(d_cartesian_from_ct, n, U)
    assert np.allclose(audit.rank_one_shape_cartesian, b, atol=5e-12, rtol=5e-12)
    assert audit.maximum_residual < 5e-11


def test_bridge_rejects_nonunit_normal_nonphysical_volume_and_nonsymmetric_stretch():
    with pytest.raises(ValueError, match="unit Cartesian habit normal"):
        rank_one_shape_from_smc_cartesian([0.1, 0.0, 0.0], [2.0, 0.0, 0.0], 1.1)
    with pytest.raises(ValueError, match="strictly positive"):
        rank_one_shape_from_smc_cartesian([0.1, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0)
    with pytest.raises(ValueError, match="symmetric"):
        smc_vector_from_stretch_cartesian(
            np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            [1.0, 0.0, 0.0],
        )


def test_native_role_inference_is_contract_based_and_fail_closed():
    assert infer_native_shape_vector_role(
        theory="cayron_ct", prediction_kind="ct_am_habit", exact=True, metadata={}
    ) is ShapeVectorRole.CT_SMC_IPS_D
    assert infer_native_shape_vector_role(
        theory="ball_james", prediction_kind="ball_james_am", exact=True, metadata={}
    ) is ShapeVectorRole.RANK_ONE_PARENT_IDENTITY_B
    assert infer_native_shape_vector_role(
        theory="ptmc", prediction_kind="ptmc_habit", exact=True,
        metadata={"dilatational_factor": 1.0, "true_invariant_plane": True},
    ) is ShapeVectorRole.RANK_ONE_PARENT_IDENTITY_B
    assert infer_native_shape_vector_role(
        theory="ptmc", prediction_kind="ptmc_habit", exact=False,
        metadata={"dilatational_factor": 1.01, "true_invariant_plane": False},
    ) is ShapeVectorRole.RANK_ONE_DILATED_IDENTITY_B
    assert infer_native_shape_vector_role(
        theory="experiment", prediction_kind="experiment", exact=None, metadata={}
    ) is ShapeVectorRole.UNSPECIFIED

    with pytest.raises(ValueError, match="inconsistent"):
        infer_native_shape_vector_role(
            theory="ptmc", prediction_kind="ptmc_habit", exact=True,
            metadata={"dilatational_factor": 1.0, "true_invariant_plane": False},
        )
    with pytest.raises(ValueError, match="dilatational_factor != 1"):
        infer_native_shape_vector_role(
            theory="ptmc", prediction_kind="ptmc_habit", exact=True,
            metadata={"dilatational_factor": 1.01, "true_invariant_plane": True},
        )
