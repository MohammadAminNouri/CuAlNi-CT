from __future__ import annotations

"""Second-layer CT covariance tests.

These are intentionally NOT repeats of Cartesian/metric representation parity.
They test the *coupled* CMC -> habit-plane -> SMC -> IPS -> shear/shear pipeline
under independent exact changes of parent and product crystallographic bases,
plus dimensional unit scaling.

Coordinate convention:
    u_M = C_M_from_A u_A

For basis changes u_A(old)=P_A u_A(new), u_M(old)=P_M u_M(new):
    M_A' = P_A^T M_A P_A
    M_M' = P_M^T M_M P_M
    C'   = P_M^-1 C P_A
    CMC' = P_A^T CMC P_A
    SMC' = P_A^-1 SMC P_A^-T
"""

import numpy as np
import sympy as sp

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.ct import (
    analyze_cmc,
    cmc,
    ct_supercompatibility_residual,
    habit_planes_from_cmc,
    ips_shear_from_habit_plane,
    normalized_cmc,
    smc,
)
from cualni_cryst.lattice import (
    Lattice,
    metric_norm,
    normalize_direct,
    normalize_plane,
)


def _transform_metric(M: np.ndarray, P: sp.Matrix) -> np.ndarray:
    Pf = np.asarray(P, dtype=float)
    return Pf.T @ np.asarray(M, dtype=float) @ Pf


def _transform_correspondence(
    C: Correspondence, P_A: sp.Matrix, P_M: sp.Matrix
) -> Correspondence:
    Cp = sp.simplify(P_M.inv() * C.C_M_from_A * P_A)
    return Correspondence(Cp, label="independent-basis transform")


def _plane_projective_residual(
    p: np.ndarray,
    q: np.ndarray,
    M: np.ndarray,
) -> float:
    """Stable distance between two unoriented reciprocal planes.

    `sqrt(1-cos^2)` is ill-conditioned for nearly coincident planes because
    cos(theta) is within machine precision of 1. Instead convert each
    metric-normalized covector to its unit direct-space physical normal,
    sign-align the two projective normals, and take the metric norm of their
    difference.
    """

    p_unit = normalize_plane(np.asarray(p, float).reshape(3), M)
    q_unit = normalize_plane(np.asarray(q, float).reshape(3), M)
    M_inv = np.linalg.inv(np.asarray(M, float))

    n_p = M_inv @ p_unit
    n_q = M_inv @ q_unit

    sign = 1.0 if float(n_p @ M @ n_q) >= 0.0 else -1.0
    return metric_norm(n_p - sign * n_q, M)


def _direct_relative_residual(
    lhs: np.ndarray, rhs: np.ndarray, M: np.ndarray
) -> float:
    delta = np.asarray(lhs, float).reshape(3) - np.asarray(rhs, float).reshape(3)
    scale = max(
        metric_norm(np.asarray(lhs, float).reshape(3), M),
        metric_norm(np.asarray(rhs, float).reshape(3), M),
        1.0,
    )
    return metric_norm(delta, M) / scale


def _synthetic_first_order_case():
    # Non-orthogonal parent: this is deliberately not a cubic convenience case.
    A = Lattice(1.3, 1.7, 2.1, 78.0, 101.0, 112.0)
    M_A = A.metric()

    # C=I and an exactly singular indefinite CMC in the original crystal basis.
    # The perturbation is small enough that M_M remains SPD.
    D = np.diag([-0.25, 0.0, 0.35])
    M_M = M_A + D
    assert np.min(np.linalg.eigvalsh(M_M)) > 0.0
    C = Correspondence(sp.eye(3))
    return M_A, M_M, C


def _match_plane(
    expected: np.ndarray, candidates: list[np.ndarray], M: np.ndarray
) -> np.ndarray:
    residuals = [
        _plane_projective_residual(expected, candidate, M)
        for candidate in candidates
    ]
    idx = int(np.argmin(residuals))
    assert residuals[idx] < 1.0e-10, residuals
    return np.asarray(candidates[idx], float)


def test_cmc_smc_habit_and_ips_are_covariant_under_independent_phase_basis_changes():
    M_A, M_M, C = _synthetic_first_order_case()

    P_A = sp.Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    P_M = sp.Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    assert P_A.det() == 1
    assert P_M.det() == 1

    M_A2 = _transform_metric(M_A, P_A)
    M_M2 = _transform_metric(M_M, P_M)
    C2 = _transform_correspondence(C, P_A, P_M)

    CMC1 = cmc(M_A, M_M, C)
    CMC2 = cmc(M_A2, M_M2, C2)
    SMC1 = smc(M_A, M_M, C)
    SMC2 = smc(M_A2, M_M2, C2)

    PA = np.asarray(P_A, float)
    assert np.allclose(CMC2, PA.T @ CMC1 @ PA, atol=2e-11, rtol=2e-11)
    assert np.allclose(
        SMC2,
        np.linalg.inv(PA) @ SMC1 @ np.linalg.inv(PA).T,
        atol=2e-11,
        rtol=2e-11,
    )

    a1 = analyze_cmc(M_A, M_M, C, tol=1e-9)
    a2 = analyze_cmc(M_A2, M_M2, C2, tol=1e-9)
    assert a1.exact_compatible and a2.exact_compatible
    assert a1.degeneracy_order == a2.degeneracy_order == 1
    assert a1.inertia == a2.inertia
    assert np.allclose(
        np.sort(np.linalg.eigvalsh(normalized_cmc(M_A, M_M, C))),
        np.sort(np.linalg.eigvalsh(normalized_cmc(M_A2, M_M2, C2))),
        atol=2e-10,
        rtol=2e-10,
    )

    planes1 = habit_planes_from_cmc(M_A, M_M, C, tol=1e-9)
    planes2 = habit_planes_from_cmc(M_A2, M_M2, C2, tol=1e-9)
    assert len(planes1) == len(planes2) == 2

    PA_inv = np.linalg.inv(PA)
    for p1 in planes1:
        expected_p2 = PA.T @ np.asarray(p1, float)
        p2 = _match_plane(expected_p2, planes2, M_A2)

        d1 = ips_shear_from_habit_plane(p1, M_A, M_M, C)
        d2 = ips_shear_from_habit_plane(p2, M_A2, M_M2, C2)
        expected_d2 = PA_inv @ d1

        # Plane sign is projective; d changes sign with the plane. Align once.
        if _direct_relative_residual(d2, expected_d2, M_A2) > _direct_relative_residual(
            d2, -expected_d2, M_A2
        ):
            expected_d2 = -expected_d2
        assert _direct_relative_residual(d2, expected_d2, M_A2) < 3e-9


def test_entire_shear_shear_residual_is_basis_invariant_not_just_cmc():
    M_A, M_M, C = _synthetic_first_order_case()
    plane = habit_planes_from_cmc(M_A, M_M, C, tol=1e-9)[0]
    d = ips_shear_from_habit_plane(plane, M_A, M_M, C)

    # A deliberately generic admissible trial twin geometry. It need not be
    # supercompatible; the scalar mismatch itself must be a coordinate invariant.
    twin_plane = np.array([1.0, 0.0, -1.0])
    twin_direction = np.array([0.0, 1.0, 0.0])
    twin_shear = 0.2718281828459045

    r1 = ct_supercompatibility_residual(
        plane, d, twin_plane, twin_direction, twin_shear, M_A
    )

    P_A = sp.Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    P_M = sp.Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    M_A2 = _transform_metric(M_A, P_A)
    M_M2 = _transform_metric(M_M, P_M)
    C2 = _transform_correspondence(C, P_A, P_M)

    PA = np.asarray(P_A, float)
    plane2_expected = PA.T @ plane
    plane2 = _match_plane(
        plane2_expected,
        habit_planes_from_cmc(M_A2, M_M2, C2, tol=1e-9),
        M_A2,
    )
    d2 = ips_shear_from_habit_plane(plane2, M_A2, M_M2, C2)
    twin_plane2 = PA.T @ twin_plane
    twin_direction2 = np.linalg.inv(PA) @ twin_direction

    r2 = ct_supercompatibility_residual(
        plane2, d2, twin_plane2, twin_direction2, twin_shear, M_A2
    )
    assert abs(r1 - r2) < 3e-9


def test_projective_plane_residual_has_no_near_parallel_sqrt_cancellation_floor():
    M_A, M_M, C = _synthetic_first_order_case()
    reference = habit_planes_from_cmc(M_A, M_M, C, tol=1e-9)[0]

    for scale in (1.0e-6, 1.0, 1.0e6):
        factor = scale * scale
        planes = habit_planes_from_cmc(
            factor * M_A,
            factor * M_M,
            C,
            tol=1e-8,
        )
        residuals = [
            _plane_projective_residual(reference, p, factor * M_A)
            for p in planes
        ]
        assert min(residuals) < 1.0e-10, (scale, residuals)


def test_length_unit_scaling_laws_are_explicit_and_dimensionally_correct():
    M_A, M_M, C = _synthetic_first_order_case()
    CMC0 = cmc(M_A, M_M, C)
    SMC0 = smc(M_A, M_M, C)
    N0 = normalized_cmc(M_A, M_M, C)
    plane0 = habit_planes_from_cmc(M_A, M_M, C, tol=1e-9)[0]
    d0 = ips_shear_from_habit_plane(plane0, M_A, M_M, C)

    for scale in (1.0e-6, 1.0e6):
        factor = scale * scale
        MA = factor * M_A
        MM = factor * M_M

        assert np.allclose(cmc(MA, MM, C), factor * CMC0, atol=1e-8 * max(factor, 1.0), rtol=2e-10)
        assert np.allclose(smc(MA, MM, C), SMC0 / factor, atol=1e-8 / max(factor, 1.0), rtol=2e-10)
        assert np.allclose(normalized_cmc(MA, MM, C), N0, atol=3e-9, rtol=3e-9)

        planes = habit_planes_from_cmc(MA, MM, C, tol=1e-8)
        p = _match_plane(plane0, planes, MA)
        d = ips_shear_from_habit_plane(p, MA, MM, C)

        # Coordinate components of a unit-aware direct shear scale as 1/scale.
        expected = d0 / scale
        if _direct_relative_residual(d, expected, MA) > _direct_relative_residual(
            d, -expected, MA
        ):
            expected = -expected
        assert _direct_relative_residual(d, expected, MA) < 5e-8
