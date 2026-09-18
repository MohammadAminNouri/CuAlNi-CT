import numpy as np

from cualni_cryst.cualni_models import (
    do3_to_6m_branch,
    james_hane_6m_example_lattices,
)
from cualni_cryst.lattice import Lattice
from cualni_cryst.representation import (
    CartesianConvention,
    CartesianFrame,
    RepresentationBridge,
    basis_matrix,
    compare_convention_pairs,
    frame_rotation,
)
from cualni_cryst.stretch import principal_stretches, stretch_from_metrics


def general_lattice() -> Lattice:
    return Lattice(
        a=4.2,
        b=5.1,
        c=6.3,
        alpha_deg=82.0,
        beta_deg=101.0,
        gamma_deg=74.0,
        label="general triclinic test cell",
    )


def test_every_cartesian_convention_reproduces_same_metric():
    lattice = general_lattice()
    M = lattice.metric()

    for convention in CartesianConvention:
        B = basis_matrix(lattice, convention)
        assert np.linalg.det(B) > 0.0
        assert np.allclose(B.T @ B, M, atol=1e-10, rtol=1e-10)


def test_ptclab_convention_is_x_parallel_a_and_c_in_xz():
    lattice = general_lattice()
    B = basis_matrix(lattice, CartesianConvention.PTCLAB_A_X_C_XZ)

    # First basis vector is exactly along +x.
    assert np.allclose(B[:, 0], [lattice.a, 0.0, 0.0], atol=1e-12)

    # Third basis vector lies in xz, exactly as PTCLab manual section 2.2 states.
    assert abs(B[1, 2]) < 1e-12

    # The a-c angle remains beta.
    cos_beta = np.dot(B[:, 0], B[:, 2]) / (
        np.linalg.norm(B[:, 0]) * np.linalg.norm(B[:, 2])
    )
    assert np.isclose(cos_beta, np.cos(np.deg2rad(lattice.beta_deg)), atol=1e-12)


def test_symmetric_metric_frame_is_symmetric_positive_and_reconstructs_metric():
    lattice = general_lattice()
    B = basis_matrix(lattice, CartesianConvention.SYMMETRIC_METRIC)

    assert np.allclose(B, B.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(B) > 0.0)
    assert np.allclose(B.T @ B, lattice.metric(), atol=1e-10, rtol=1e-10)


def test_direct_reciprocal_and_incidence_parity():
    lattice = general_lattice()
    M = lattice.metric()
    u = np.array([2.0, -1.0, 3.0])
    p = np.array([1.0, 4.0, -2.0])

    for convention in CartesianConvention:
        frame = CartesianFrame(lattice, convention)

        v = frame.direct_to_cartesian(u)
        g = frame.plane_to_cartesian(p)

        assert np.isclose(v @ v, u @ M @ u, atol=1e-10, rtol=1e-10)
        assert np.isclose(
            g @ g,
            p @ np.linalg.inv(M) @ p,
            atol=1e-10,
            rtol=1e-10,
        )
        assert np.isclose(g @ v, p @ u, atol=1e-10, rtol=1e-10)

        assert np.allclose(frame.direct_from_cartesian(v), u, atol=1e-10)
        assert np.allclose(frame.plane_from_cartesian(g), p, atol=1e-10)


def test_valid_embeddings_of_same_metric_differ_only_by_proper_rotation():
    lattice = general_lattice()
    frames = [CartesianFrame(lattice, c) for c in CartesianConvention]

    for source in frames:
        for target in frames:
            Q = frame_rotation(source, target)
            assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-10)
            assert np.isclose(np.linalg.det(Q), 1.0, atol=1e-10)


def test_do3_6m_correspondence_maps_directions_and_planes_identically_two_ways():
    A, M = james_hane_6m_example_lattices()
    branch = do3_to_6m_branch()

    for parent_convention in CartesianConvention:
        for product_convention in CartesianConvention:
            bridge = RepresentationBridge(
                A,
                M,
                branch.correspondence,
                parent_convention,
                product_convention,
            )

            d_crystal, d_F = bridge.map_parent_direction_two_ways([1, 2, -1])
            p_crystal, p_F = bridge.map_parent_plane_two_ways([2, -1, 1])

            assert np.allclose(d_crystal, d_F, atol=1e-10, rtol=1e-10)
            assert np.allclose(p_crystal, p_F, atol=1e-10, rtol=1e-10)


def test_cartesian_right_cauchy_green_is_metric_congruence():
    A, M = james_hane_6m_example_lattices()
    branch = do3_to_6m_branch()

    for parent_convention in CartesianConvention:
        for product_convention in CartesianConvention:
            bridge = RepresentationBridge(
                A,
                M,
                branch.correspondence,
                parent_convention,
                product_convention,
            )
            assert np.allclose(
                bridge.right_cauchy_green_cartesian(),
                bridge.right_cauchy_green_from_metric(),
                atol=1e-10,
                rtol=1e-10,
            )


def test_principal_stretches_are_invariant_under_cartesian_convention():
    A, M = james_hane_6m_example_lattices()
    branch = do3_to_6m_branch()

    values = compare_convention_pairs(A, M, branch.correspondence)
    reference = next(iter(values.values()))
    for lambdas in values.values():
        assert np.allclose(lambdas, reference, atol=1e-10, rtol=1e-10)

    U_metric = stretch_from_metrics(A.metric(), M.metric(), branch.correspondence)
    metric_lambdas, _ = principal_stretches(U_metric)
    assert np.allclose(reference, metric_lambdas, atol=1e-10, rtol=1e-10)


def test_full_parity_audit_passes_for_all_do3_6m_convention_pairs():
    A, M = james_hane_6m_example_lattices()
    branch = do3_to_6m_branch()

    for parent_convention in CartesianConvention:
        for product_convention in CartesianConvention:
            bridge = RepresentationBridge(
                A,
                M,
                branch.correspondence,
                parent_convention,
                product_convention,
            )
            audit = bridge.audit()
            audit.assert_within(1e-9)
