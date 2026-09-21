from __future__ import annotations

import inspect
import itertools

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cualni_cryst.ebsd_blind_discovery import (
    BlindORSettings,
    blind_observable_embedding_distance_deg,
    discover_orientation_relationship_blind,
)
from cualni_cryst.ebsd_map import EBSDPhase
from cualni_cryst.ebsd_theory_bridge import (
    build_theory_library,
    orientation_relationship_distance_deg,
)
from cualni_cryst.lattice import Lattice


def _phase(phase_id, name, lattice, point_group):
    return EBSDPhase.from_point_group(
        phase_id, name, lattice, point_group
    )


def _hidden_case(
    parent,
    product,
    *,
    seed,
    n_variants,
    noise_deg,
):
    rng = np.random.default_rng(seed)
    true_or = Rotation.random(random_state=rng).as_matrix()
    theory = build_theory_library(parent, product, true_or)
    count = min(n_variants, theory.n_variants)
    if count < 3:
        raise RuntimeError("test phase pair generated too few variants")

    parent_g = Rotation.random(random_state=rng).as_matrix()
    indices = np.linspace(
        0, theory.n_variants - 1, count, dtype=int
    )
    orientations = []
    for index in indices:
        exact = (
            parent_g
            @ theory.variant_set.variants[int(index)].R_parent_from_product
        )
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        amplitude = rng.normal(scale=np.deg2rad(noise_deg))
        noise = Rotation.from_rotvec(amplitude * axis).as_matrix()
        orientations.append(noise @ exact)
    orientations = np.asarray(orientations)
    adjacency = tuple(itertools.combinations(range(count), 2))
    return true_or, orientations, adjacency


def test_public_blind_solver_signature_contains_no_or_or_expected_answer():
    signature = inspect.signature(
        discover_orientation_relationship_blind
    )
    forbidden = {
        "initial_or",
        "expected_or",
        "true_or",
        "orientation_relationship",
        "reference_or",
    }
    assert forbidden.isdisjoint(signature.parameters)


def test_blind_solver_refuses_too_little_evidence_instead_of_guessing():
    parent = _phase(
        1, "A", Lattice(3.1, 3.1, 5.0), "4/mmm"
    )
    product = _phase(
        2, "M", Lattice.orthorhombic(3.0, 4.0, 5.0), "mmm"
    )
    orientations = np.asarray(
        [
            Rotation.from_euler("x", 10, degrees=True).as_matrix(),
            Rotation.from_euler("y", 20, degrees=True).as_matrix(),
        ]
    )
    result = discover_orientation_relationship_blind(
        orientations,
        [(0, 1)],
        parent,
        product,
        settings=BlindORSettings(
            global_samples=64,
            coarse_keep=8,
            exact_keep=4,
            minimum_boundaries=4,
        ),
    )
    assert result.status == "insufficient_evidence"
    assert result.best is None



@pytest.mark.parametrize(
    "parent, product, normalizer, minimum_physical_distance_deg",
    [
        (
            _phase(1, "A", Lattice(3.1, 3.1, 5.0), "4/mmm"),
            _phase(2, "M", Lattice.orthorhombic(3.0, 4.0, 5.0), "mmm"),
            Rotation.from_euler("z", 45.0, degrees=True).as_matrix(),
            10.0,
        ),
        (
            _phase(
                1,
                "A",
                Lattice(3.2, 3.2, 5.1, 90, 90, 120),
                "6/mmm",
            ),
            _phase(2, "M", Lattice.orthorhombic(3.0, 4.2, 5.3), "mmm"),
            Rotation.from_euler("z", 30.0, degrees=True).as_matrix(),
            5.0,
        ),
        (
            _phase(1, "A", Lattice.orthorhombic(3.0, 4.0, 5.0), "mmm"),
            _phase(
                2,
                "M",
                Lattice.monoclinic_unique_b(3.1, 4.2, 5.4, 103.0),
                "2/m",
            ),
            Rotation.from_rotvec(
                np.deg2rad(120.0)
                * np.array([1.0, 1.0, 1.0])
                / np.sqrt(3.0)
            ).as_matrix(),
            10.0,
        ),
    ],
)
def test_blind_inverse_uses_parent_normalizer_observable_quotient(
    parent,
    product,
    normalizer,
    minimum_physical_distance_deg,
):
    base = Rotation.from_euler(
        "xyz", [17.0, -23.0, 31.0], degrees=True
    ).as_matrix()
    gauge_shifted = normalizer @ base

    # These can be very different physical OR representatives under the
    # ordinary parent/product symmetry quotient...
    physical = orientation_relationship_distance_deg(
        gauge_shifted,
        base,
        parent.proper_symmetry_cartesian,
        product.proper_symmetry_cartesian,
    )
    assert physical > minimum_physical_distance_deg

    # ...while being *exactly the same observable child/child subgroup
    # embedding*.  A blind inverse solver must not call that a failure.
    observable = blind_observable_embedding_distance_deg(
        gauge_shifted,
        base,
        parent.proper_symmetry_cartesian,
        product.proper_symmetry_cartesian,
    )
    assert observable < 5.0e-6

def test_hidden_tetragonal_to_orthorhombic_or_is_recovered_without_or_input():
    parent = _phase(
        1, "A", Lattice(3.1, 3.1, 5.0), "4/mmm"
    )
    product = _phase(
        2, "M", Lattice.orthorhombic(3.0, 4.0, 5.0), "mmm"
    )
    true_or, orientations, adjacency = _hidden_case(
        parent,
        product,
        seed=151,
        n_variants=7,
        noise_deg=0.15,
    )
    result = discover_orientation_relationship_blind(
        orientations,
        adjacency,
        parent,
        product,
        settings=BlindORSettings(
            global_samples=768,
            coarse_keep=32,
            exact_keep=10,
            local_radius_deg=30.0,
            local_max_iterations=70,
            acceptance_median_deg=1.0,
            support_threshold_deg=2.0,
            minimum_support_fraction=0.70,
            ambiguity_median_gap_deg=0.20,
            sobol_seed=911,
        ),
    )
    assert result.best is not None
    observable_distance = blind_observable_embedding_distance_deg(
        result.best.R_parent_from_product,
        true_or,
        parent.proper_symmetry_cartesian,
        product.proper_symmetry_cartesian,
    )
    assert observable_distance < 1.0
    assert result.status != "inconsistent"
    assert result.best.median_residual_deg < 0.75


def test_hidden_hexagonal_to_orthorhombic_or_is_recovered_without_or_input():
    parent = _phase(
        1,
        "A",
        Lattice(3.2, 3.2, 5.1, 90, 90, 120),
        "6/mmm",
    )
    product = _phase(
        2, "M", Lattice.orthorhombic(3.0, 4.2, 5.3), "mmm"
    )
    true_or, orientations, adjacency = _hidden_case(
        parent,
        product,
        seed=811,
        n_variants=9,
        noise_deg=0.20,
    )
    result = discover_orientation_relationship_blind(
        orientations,
        adjacency,
        parent,
        product,
        settings=BlindORSettings(
            global_samples=1024,
            coarse_keep=36,
            exact_keep=12,
            local_radius_deg=30.0,
            local_max_iterations=80,
            acceptance_median_deg=1.2,
            support_threshold_deg=2.5,
            minimum_support_fraction=0.70,
            sobol_seed=191,
        ),
    )
    assert result.best is not None
    observable_distance = blind_observable_embedding_distance_deg(
        result.best.R_parent_from_product,
        true_or,
        parent.proper_symmetry_cartesian,
        product.proper_symmetry_cartesian,
    )
    assert observable_distance < 1.2
    assert result.status != "inconsistent"


def test_hidden_cubic_to_tetragonal_case_survives_noise_without_answer_seed():
    parent = _phase(
        1, "A", Lattice.cubic(3.0), "m-3m"
    )
    product = _phase(
        2, "M", Lattice(3.0, 3.0, 4.7), "4/mmm"
    )
    true_or, orientations, adjacency = _hidden_case(
        parent,
        product,
        seed=404,
        n_variants=10,
        noise_deg=0.20,
    )
    result = discover_orientation_relationship_blind(
        orientations,
        adjacency,
        parent,
        product,
        settings=BlindORSettings(
            global_samples=1536,
            coarse_boundary_limit=20,
            coarse_keep=40,
            exact_keep=12,
            local_radius_deg=25.0,
            local_max_iterations=75,
            acceptance_median_deg=1.3,
            support_threshold_deg=2.5,
            minimum_support_fraction=0.70,
            sobol_seed=707,
        ),
    )
    assert result.best is not None
    observable_distance = blind_observable_embedding_distance_deg(
        result.best.R_parent_from_product,
        true_or,
        parent.proper_symmetry_cartesian,
        product.proper_symmetry_cartesian,
    )
    assert observable_distance < 1.5
    assert result.status == "identified_observable_class"


def test_hidden_orthorhombic_to_monoclinic_low_symmetry_case():
    parent = _phase(
        1, "A", Lattice.orthorhombic(3.0, 4.0, 5.0), "mmm"
    )
    product = _phase(
        2,
        "M",
        Lattice.monoclinic_unique_b(3.1, 4.2, 5.4, 103.0),
        "2/m",
    )
    true_or, orientations, adjacency = _hidden_case(
        parent,
        product,
        seed=909,
        n_variants=4,
        noise_deg=0.10,
    )
    result = discover_orientation_relationship_blind(
        orientations,
        adjacency,
        parent,
        product,
        settings=BlindORSettings(
            global_samples=1536,
            coarse_keep=40,
            exact_keep=14,
            local_radius_deg=30.0,
            local_max_iterations=90,
            minimum_boundaries=4,
            minimum_distinct_operator_classes=2,
            acceptance_median_deg=1.0,
            support_threshold_deg=2.0,
            minimum_support_fraction=0.65,
            sobol_seed=313,
        ),
    )
    assert result.best is not None
    observable_distance = blind_observable_embedding_distance_deg(
        result.best.R_parent_from_product,
        true_or,
        parent.proper_symmetry_cartesian,
        product.proper_symmetry_cartesian,
    )
    assert observable_distance < 1.5
    assert result.status != "inconsistent"
