"""Blind numerical reproduction and engine cross-check of Acta Materialia 270 (2024) 119870.

The paper's four model inputs appear ONLY in _solve_contacts; all published
outputs are restricted to expected-value assertions. Neither output matrix,
atomic position, nor volume ratio is injected into an engine calculation.
This is a benchmark of the existing lattice/metric engine, NOT an implementation
of the hard-sphere model inside the Streamlit interface.
"""

from math import cos, radians, sin, sqrt
import numpy as np
import sympy as sp

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.stretch import stretch_from_metrics, metric_native_stretch_spectrum
from twin_app.cayron_correspondence import calculate_from_paired_vectors
from twin_app.input_logic import exact_matrix


def _solve_contacts():
    # Published model inputs: Ti/Ni atomic diameters, final monoclinic angle,
    # and parent B2 lattice parameter. Published outputs are NOT inputs.
    d_ti, d_ni, beta, a_b2 = 2.9, 2.27, radians(97.9), 3.01
    k = d_ni * (d_ni + 2 * d_ti)
    c = sqrt(k) * sin(beta) - d_ti * cos(beta)  # Paper Eq. (2)
    A = np.array([d_ti * sin(beta), 0., d_ti * cos(beta)])
    E = np.array([
        (-sqrt(k) * cos(beta) + d_ti * sin(beta)) / 2.,
        0.,
        (d_ti * cos(beta) + sqrt(k) * sin(beta)) / 2.,
    ])  # Paper Eq. (3)
    C_position = np.array([0., 0., c])
    d_ti_ni = (d_ti + d_ni) / 2.
    # Three independent sphere-contact equations, paper Eq. (6).
    # Subtract their squares to eliminate the quadratic terms in x and z.
    lhs = np.array([
        [A[0], A[2]],
        [(E - C_position)[0], (E - C_position)[2]],
    ])
    rhs = np.array([
        np.dot(E, A) - d_ti**2 / 2,
        (d_ti**2 - d_ti_ni**2 - np.dot(C_position, C_position)
         + np.dot(E, E)) / 2,
    ])
    x, z = np.linalg.solve(lhs, rhs)
    y2 = d_ti_ni**2 - (x - E[0])**2 - (z - E[2])**2
    assert y2 > 0, "No physical second-layer position"
    G = np.array([x, sqrt(y2), z])
    b = 2. * G[1]  # Paper Eq. (7), positive-b branch
    atom_B = G + A - E
    S_m = np.array([
        [d_ti * sin(beta), 0., 0.],
        [0., b, 0.],
        [d_ti * cos(beta), 0., c],
    ])
    S_a = a_b2 * np.array([
        [0., 0., 1.],
        [1/sqrt(2), 1/sqrt(2), 0.],
        [-1/sqrt(2), 1/sqrt(2), 0.],
    ])
    # Paper Eq. (9) orientation/basis mapping and Eq. (10) distortion:
    # T and F here are NOT correspondence C.
    T = np.linalg.solve(S_a, S_m)
    def fractions(v):
        return np.linalg.solve(S_m, v)
    return {
        "a_b2": a_b2,
        "b": b, "c": c, "T": T,
        "parent_metric": S_a.T @ S_a,
        "product_metric": S_m.T @ S_m,
        "ni1": fractions(E), "ti2": fractions(G), "ni2": fractions(atom_B),
        "contact_residuals": (
            np.linalg.norm(G - E) - d_ti_ni,
            np.linalg.norm(G - C_position) - d_ti,
            np.linalg.norm(G - E + A) - d_ti_ni,
        ),
    }


def _independently_entered_correspondence():
    # Paper's paired directions: parent [001], [110], [-110] become product
    # [100], [010], [001], respectively. Form C=Q*P^-1 with the ACTUAL
    # code path used by both methods of the Streamlit correspondence editor.
    parent_vectors = (("0", "0", "1"), ("1", "1", "0"), ("-1", "1", "0"))
    product_vectors = (("1", "0", "0"), ("0", "1", "0"), ("0", "0", "1"))
    fwd = calculate_from_paired_vectors(
        parent_vectors, product_vectors, direction="A_TO_M"
    )
    rev = calculate_from_paired_vectors(
        product_vectors, parent_vectors, direction="M_TO_A"
    )
    C = exact_matrix(fwd.canonical_M_from_A)
    assert exact_matrix(rev.canonical_M_from_A) == C
    assert exact_matrix(rev.source_to_destination) == C.inv()
    # The matrix printed as Eq. (1) in the paper is parent coordinates
    # from product basis vectors; it is C^{-1} under OUR canonical convention.
    published_eq_1 = sp.Matrix([[0, 1, -1], [0, 1, 1], [1, 0, 0]])
    assert C.inv() == published_eq_1
    return Correspondence(C)


def test_hard_sphere_atomic_contacts_reproduce_published_numbers_without_fitting():
    result = _solve_contacts()
    assert max(abs(float(z)) for z in result["contact_residuals"]) < 2e-12
    # Expected numbers enter only HERE, never the computational path.
    assert abs(result["b"] - 4.147) < 5e-4
    assert abs(result["c"] - 4.638) < 5e-4
    np.testing.assert_allclose(result["ni1"], (0.6024, 0., 0.4658), atol=5e-5, rtol=0.)
    np.testing.assert_allclose(result["ti2"], (0.1277, .5, 0.5811), atol=5e-5, rtol=0.)
    np.testing.assert_allclose(result["ni2"], (0.5253, .5, 0.1152), atol=5e-5, rtol=0.)


def test_blind_2024_correspondence_and_metric_stretch_agree_with_f_and_volume():
    source = _solve_contacts()
    correspondence = _independently_entered_correspondence()
    C = np.asarray(correspondence.C_M_from_A, dtype=float)
    # Independently calculate paper Eq (10), using derived T and editor C.
    F = source["T"] @ C
    np.testing.assert_allclose(
        F,
        [[1.0319, -.0577, .0936], [-.0577, 1.0319, -.0936],
         [0., 0., .9543]],
        atol=5e-5, rtol=0.,
    )
    assert abs(np.linalg.det(F) - 1.0130) < 5e-5
    U = stretch_from_metrics(
        source["parent_metric"], source["product_metric"], correspondence
    )
    spectrum = metric_native_stretch_spectrum(
        source["parent_metric"], source["product_metric"], correspondence
    )
    # F and U differ by a proper rotation but MUST have identical U^2.
    np.testing.assert_allclose(U.T @ U, F.T @ F, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(
        np.sort(np.linalg.eigvalsh(U)), spectrum.lambdas, atol=1e-9, rtol=1e-9
    )
    assert abs(np.linalg.det(U) - np.linalg.det(F)) < 1e-9
    # Direct and reciprocal mapping must be consistent without using F as C.
    parent_a = sp.Matrix([0, 0, 1])
    product_a = correspondence.map_direction_A_to_M(parent_a)
    assert product_a == sp.Matrix([1, 0, 0])
    parent_plane = sp.Matrix([1, 1, 0])
    product_plane = correspondence.map_plane_A_to_M(parent_plane)
    assert sp.simplify(product_plane.dot(product_a)) == 0 if parent_plane.dot(parent_a) == 0 else True
