import math

import numpy as np
import pytest
import sympy as sp

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.lattice import Lattice
from cualni_cryst.representation import CartesianConvention, RepresentationBridge


def _relative(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(
        np.linalg.norm(a - b)
        / max(np.linalg.norm(a), np.linalg.norm(b), 1.0)
    )


def _all_convention_spectra(parent, product, corr):
    spectra = []
    residuals = []
    for ca in CartesianConvention:
        for cm in CartesianConvention:
            bridge = RepresentationBridge(parent, product, corr, ca, cm)
            audit = bridge.audit()
            audit.assert_within(1e-9)
            residuals.append(audit.maximum_residual)
            spectra.append(bridge.principal_stretches())
    return spectra, residuals


def _cell_from_basis(B):
    vectors = [B[:, i] for i in range(3)]
    lengths = [float(np.linalg.norm(v)) for v in vectors]

    def angle(i, j):
        c = float(
            np.dot(vectors[i], vectors[j])
            / (lengths[i] * lengths[j])
        )
        c = min(1.0, max(-1.0, c))
        return math.degrees(math.acos(c))

    return Lattice(
        a=lengths[0],
        b=lengths[1],
        c=lengths[2],
        alpha_deg=angle(1, 2),
        beta_deg=angle(0, 2),
        gamma_deg=angle(0, 1),
    )


def _random_valid_lattice(rng):
    diag = rng.uniform(0.5, 8.0, size=3)
    B = np.array(
        [
            [diag[0], rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)],
            [0.0, diag[1], rng.uniform(-1.5, 1.5)],
            [0.0, 0.0, diag[2]],
        ]
    )
    return _cell_from_basis(B)


def _random_proper_unimodular(rng):
    """Return a seeded random SL(3,Z) map for a physical deformation test."""

    P = sp.eye(3)
    for _ in range(8):
        op = int(rng.integers(0, 3))
        i, j = [int(x) for x in rng.choice(3, size=2, replace=False)]
        if op == 0:
            P.row_swap(i, j)
        elif op == 1:
            P.row_op(i, lambda value, _: -value)
        else:
            k = int(rng.choice([-2, -1, 1, 2]))
            source = [P[i, col] + k * P[j, col] for col in range(3)]
            for col, value in enumerate(source):
                P[i, col] = value

    det = int(P.det())
    assert det in (-1, 1)
    if det < 0:
        P.row_op(0, lambda value, _: -value)

    assert P.det() == 1
    return P


def test_all_cartesian_conventions_on_all_crystal_system_shapes():
    cells = [
        Lattice(a=4.0, b=4.0, c=4.0, alpha_deg=90, beta_deg=90, gamma_deg=90),
        Lattice(a=4.0, b=4.0, c=6.0, alpha_deg=90, beta_deg=90, gamma_deg=90),
        Lattice(a=4.0, b=5.0, c=6.0, alpha_deg=90, beta_deg=90, gamma_deg=90),
        Lattice(a=3.0, b=3.0, c=5.0, alpha_deg=90, beta_deg=90, gamma_deg=120),
        Lattice(a=3.5, b=4.2, c=5.1, alpha_deg=90, beta_deg=103.0, gamma_deg=90),
        Lattice(a=4.2, b=5.1, c=6.3, alpha_deg=82.0, beta_deg=101.0, gamma_deg=74.0),
    ]
    correspondences = [
        Correspondence(sp.eye(3)),
        Correspondence(sp.Matrix([[1, 1, 0], [0, 1, 0], [0, 0, 1]])),
        Correspondence(
            sp.Matrix(
                [
                    [0, 0, 1],
                    [sp.Rational(1,2), sp.Rational(1,2), 0],
                    [-sp.Rational(1,2), sp.Rational(1,2), 0],
                ]
            )
        ),
    ]

    for parent in cells:
        for product in cells:
            for corr in correspondences:
                spectra, residuals = _all_convention_spectra(
                    parent, product, corr
                )
                ref = spectra[0]
                assert max(residuals) < 1e-9
                assert max(_relative(s, ref) for s in spectra) < 1e-9


def test_seeded_random_valid_cells_survive_representation_crosscheck():
    rng = np.random.default_rng(20260922)
    for _ in range(32):
        parent = _random_valid_lattice(rng)
        product = _random_valid_lattice(rng)
        P = _random_proper_unimodular(rng)
        corr = Correspondence(P)
        spectra, residuals = _all_convention_spectra(parent, product, corr)
        ref = spectra[0]
        assert max(residuals) < 2e-9
        assert max(_relative(s, ref) for s in spectra) < 2e-9


def test_common_length_scale_does_not_change_dimensionless_stretches():
    parent = Lattice(
        a=4.2, b=5.1, c=6.3,
        alpha_deg=82.0, beta_deg=101.0, gamma_deg=74.0
    )
    product = Lattice(
        a=4.8, b=5.4, c=6.0,
        alpha_deg=88.0, beta_deg=97.0, gamma_deg=78.0
    )
    corr = Correspondence(
        sp.Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    )
    reference = RepresentationBridge(
        parent,
        product,
        corr,
        CartesianConvention.SYMMETRIC_METRIC,
        CartesianConvention.SYMMETRIC_METRIC,
    ).principal_stretches()

    for scale in (1e-6, 1e-3, 1.0, 1e3, 1e6):
        A = Lattice(
            a=parent.a*scale, b=parent.b*scale, c=parent.c*scale,
            alpha_deg=parent.alpha_deg,
            beta_deg=parent.beta_deg,
            gamma_deg=parent.gamma_deg,
        )
        M = Lattice(
            a=product.a*scale, b=product.b*scale, c=product.c*scale,
            alpha_deg=product.alpha_deg,
            beta_deg=product.beta_deg,
            gamma_deg=product.gamma_deg,
        )
        got = RepresentationBridge(
            A,
            M,
            corr,
            CartesianConvention.PTCLAB_A_X_C_XZ,
            CartesianConvention.LEGACY_A_X_B_XY,
        ).principal_stretches()
        assert np.allclose(
            got, reference, atol=2e-10, rtol=2e-10
        )


def test_direct_and_reciprocal_mapping_preserve_incidence_for_random_exact_correspondences():
    rng = np.random.default_rng(771)
    for _ in range(64):
        P = _random_proper_unimodular(rng)
        corr = Correspondence(P)
        u = sp.Matrix(
            [int(x) for x in rng.integers(-4, 5, size=3)]
        )
        w = sp.Matrix(
            [int(x) for x in rng.integers(-4, 5, size=3)]
        )
        if (
            u == sp.zeros(3, 1)
            or w == sp.zeros(3, 1)
            or u.cross(w) == sp.zeros(3, 1)
        ):
            continue
        p = u.cross(w)
        assert sp.simplify((p.T * u)[0]) == 0

        um = corr.map_direction_A_to_M(u)
        pm = corr.map_plane_A_to_M(p)
        assert sp.simplify((pm.T * um)[0]) == 0
        assert corr.map_direction_M_to_A(um) == u
        assert corr.map_plane_M_to_A(pm) == p


def test_orientation_reversing_map_is_explicitly_rejected_by_physical_bridge():
    parent = Lattice(
        a=3.0, b=3.0, c=3.0,
        alpha_deg=90.0, beta_deg=90.0, gamma_deg=90.0,
    )
    product = Lattice(
        a=3.1, b=3.1, c=3.1,
        alpha_deg=90.0, beta_deg=90.0, gamma_deg=90.0,
    )
    corr = Correspondence(sp.diag(-1, 1, 1))

    bridge = RepresentationBridge(
        parent,
        product,
        corr,
        CartesianConvention.SYMMETRIC_METRIC,
        CartesianConvention.SYMMETRIC_METRIC,
    )

    with pytest.raises(
        ValueError,
        match=r"must preserve handedness",
    ):
        bridge.audit()
