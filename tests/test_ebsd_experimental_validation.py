from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cualni_cryst.ebsd_adversarial import (
    add_isotropic_orientation_noise,
    apply_crystal_frame_offset,
    apply_global_sample_rotation,
    inject_random_orientation_outliers,
    transpose_orientations,
)
from cualni_cryst.ebsd_analysis import (
    GrainSegmentation,
    build_neighbor_graph,
    minimum_disorientation,
)
from cualni_cryst.ebsd_map import EBSDMap, EBSDPhase
from cualni_cryst.ebsd_reconstruction import classify_boundary_operator
from cualni_cryst.ebsd_theory_bridge import (
    build_or_variant_set,
    build_theory_library,
    orientation_relationship_distance_deg,
)
from cualni_cryst.ebsd_trace_validation import (
    TwinPlaneHypothesis,
    estimate_boundary_trace,
    rank_trace_hypotheses,
)
from cualni_cryst.ebsd_validation import (
    PreparedBoundaryOperatorKernel,
    PreparedORVariantEnumerator,
    PreparedParentCompatibilityKernel,
    adjusted_rand_index,
    pair_operators_from_variants,
    parent_pair_compatibility_deg,
    rank_orientation_hypotheses,
    reconstruct_variant_graph_candidates,
    refine_orientation_relationship_from_child_boundaries,
    score_theory_consistency,
    sweep_segmentation_thresholds,
)
from cualni_cryst.lattice import Lattice
from cualni_cryst.orientation_kernel import OrientationKernel
from cualni_cryst.point_groups import (
    point_group_definitions,
    point_group_operations,
)


BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "benchmarks"
    / "ebsd_adversarial_campaign_v1.json"
)


def _phase(
    phase_id: int,
    name: str,
    lattice: Lattice,
    point_group: str,
) -> EBSDPhase:
    return EBSDPhase.from_point_group(
        phase_id, name, lattice, point_group
    )


def _generic_or() -> np.ndarray:
    return Rotation.from_euler(
        "ZXZ", [23.0, 37.0, 11.0], degrees=True
    ).as_matrix()


def _generic_phases():
    parent = _phase(
        1,
        "parent",
        Lattice(3.1, 3.1, 5.0),
        "4/mmm",
    )
    product = _phase(
        2,
        "product",
        Lattice.orthorhombic(3.0, 4.0, 5.0),
        "mmm",
    )
    return parent, product


def _synthetic_child_grains(n: int = 6):
    parent, product = _generic_phases()
    theory = build_theory_library(parent, product, _generic_or())
    assert theory.n_variants >= n
    parent_g = Rotation.from_euler(
        "xyz", [11.0, -7.0, 23.0], degrees=True
    ).as_matrix()
    orientations = np.asarray(
        [
            parent_g @ theory.variant_set.variants[i].R_parent_from_product
            for i in range(n)
        ],
        dtype=float,
    )
    adjacency = tuple(itertools.combinations(range(n), 2))
    return parent, product, theory, parent_g, orientations, adjacency


def test_adversarial_manifest_is_present_and_self_hashed():
    payload = json.loads(BENCHMARK.read_text())
    stored = payload.pop("sha256_without_sha_field")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    import hashlib
    assert hashlib.sha256(canonical.encode()).hexdigest() == stored
    assert payload["seed"] == 20260921


def test_or_variant_builder_crosschecks_metric_topology_for_all_32_point_groups():
    definitions = point_group_definitions()
    family_lattice = {
        "triclinic": Lattice(3.0, 4.0, 5.0, 70, 80, 75),
        "monoclinic": Lattice.monoclinic_unique_b(3.0, 4.0, 5.0, 105),
        "orthorhombic": Lattice.orthorhombic(3.0, 4.0, 5.0),
        "tetragonal": Lattice(3.0, 3.0, 5.0),
        "trigonal": Lattice(3.0, 3.0, 5.0, 90, 90, 120),
        "hexagonal": Lattice(3.0, 3.0, 5.0, 90, 90, 120),
        "cubic": Lattice.cubic(3.0),
    }
    base = Rotation.from_euler(
        "xyz", [13.7, 21.3, -17.2], degrees=True
    ).as_matrix()

    for index, definition in enumerate(definitions, start=1):
        lattice = family_lattice[definition.crystal_family]
        phase = _phase(index, definition.symbol, lattice, definition.symbol)
        variants = build_or_variant_set(
            phase,
            phase,
            base,
            crosscheck_topology=True,
        )
        prepared = PreparedORVariantEnumerator.prepare(phase, phase)
        fast_variants = prepared.variants(base)

        kernel = OrientationKernel(
            lattice.metric(),
            lattice.metric(),
            point_group_operations(definition.symbol),
            point_group_operations(definition.symbol),
        )
        topology = kernel.topology(base)
        assert variants.n_variants == topology.proper_variant_count
        assert variants.topology_expected_variant_count == topology.proper_variant_count
        assert len(fast_variants) == variants.n_variants


def test_prepared_operator_kernel_is_equivalent_to_frozen_slow_reference():
    parent, product, theory, _, orientations, adjacency = _synthetic_child_grains(6)
    noisy = add_isotropic_orientation_noise(
        orientations, 0.23, seed=7001
    )
    prepared = PreparedBoundaryOperatorKernel.prepare(
        noisy,
        adjacency,
        product.proper_symmetry_cartesian,
    )
    fast = prepared.best_class_residuals_deg(theory.boundary_operators)

    slow = []
    for a, b in adjacency:
        slow.append(
            classify_boundary_operator(
                noisy[a],
                noisy[b],
                theory.boundary_operators,
                product.proper_symmetry_cartesian,
                maximum_residual_deg=180.0,
                minimum_margin_deg=0.0,
            ).best_residual_deg
        )
    assert fast == pytest.approx(np.asarray(slow), abs=3.0e-8)

    # The OR objective may bypass class construction and use every variant pair.
    pair_fast = prepared.pair_operator_residuals_deg(
        pair_operators_from_variants(theory.variant_set.variants)
    )
    assert pair_fast == pytest.approx(np.asarray(slow), abs=3.0e-8)


def test_prepared_parent_compatibility_is_equivalent_to_reference_definition():
    parent, product, theory, _, orientations, adjacency = _synthetic_child_grains(5)
    prepared = PreparedParentCompatibilityKernel.prepare(
        orientations,
        theory.variant_set.variants,
        parent.proper_symmetry_cartesian,
    )
    fast = prepared.residuals_deg(adjacency)
    slow = np.asarray(
        [
            parent_pair_compatibility_deg(
                orientations[a],
                orientations[b],
                theory.variant_set.variants,
                parent.proper_symmetry_cartesian,
            )
            for a, b in adjacency
        ],
        dtype=float,
    )
    assert fast == pytest.approx(slow, abs=3.0e-8)


def test_global_sample_rotation_is_exactly_nonidentifiable_internally():
    parent, product, theory, _, orientations, adjacency = _synthetic_child_grains()
    base = score_theory_consistency(
        orientations,
        adjacency,
        theory,
        parent.proper_symmetry_cartesian,
        product.proper_symmetry_cartesian,
    )
    global_rotation = Rotation.from_rotvec(
        np.deg2rad(37.0)
        * np.array([0.3, 1.0, -0.2])
        / np.linalg.norm([0.3, 1.0, -0.2])
    ).as_matrix()
    rotated = apply_global_sample_rotation(
        orientations, global_rotation
    )
    moved = score_theory_consistency(
        rotated,
        adjacency,
        theory,
        parent.proper_symmetry_cartesian,
        product.proper_symmetry_cartesian,
    )
    assert moved.combined_robust_score_deg == pytest.approx(
        base.combined_robust_score_deg, abs=2.0e-8
    )

    ranking = rank_orientation_hypotheses(
        {"raw": orientations, "global_rotated": rotated},
        adjacency,
        theory,
        parent.proper_symmetry_cartesian,
        product.proper_symmetry_cartesian,
        tie_tolerance_deg=3.0e-8,
    )
    assert not ranking.internally_identifiable
    assert set(ranking.best_labels) == {"raw", "global_rotated"}


def test_wrong_crystal_frame_and_active_passive_are_detected_against_fixed_theory():
    parent, product, theory, _, orientations, adjacency = _synthetic_child_grains()
    axis = np.array([1.0, -2.0, 0.5])
    axis /= np.linalg.norm(axis)
    offset = Rotation.from_rotvec(np.deg2rad(17.0) * axis).as_matrix()
    crystal_wrong = apply_crystal_frame_offset(orientations, offset)
    passive_wrong = transpose_orientations(orientations)

    ranking = rank_orientation_hypotheses(
        {
            "correct": orientations,
            "wrong_crystal_frame": crystal_wrong,
            "wrong_active_passive": passive_wrong,
        },
        adjacency,
        theory,
        parent.proper_symmetry_cartesian,
        product.proper_symmetry_cartesian,
        minimum_identifiable_gap_deg=0.05,
    )
    assert ranking.best_labels == ("correct",)
    assert ranking.internally_identifiable
    by_label = {item.label: item.score for item in ranking.results}
    assert by_label["correct"].combined_robust_score_deg < 1.0e-5
    assert (
        by_label["wrong_crystal_frame"].combined_robust_score_deg
        > by_label["correct"].combined_robust_score_deg + 0.1
    )
    assert (
        by_label["wrong_active_passive"].combined_robust_score_deg
        > by_label["correct"].combined_robust_score_deg + 0.1
    )


def test_or_refinement_recovers_a_perturbed_generic_or_from_child_boundaries():
    parent, product, true_theory, _, orientations, adjacency = _synthetic_child_grains(5)
    axis = np.array([1.0, 2.0, -1.0])
    axis /= np.linalg.norm(axis)
    error = Rotation.from_rotvec(np.deg2rad(2.0) * axis).as_matrix()
    initial = error @ true_theory.variant_set.base_R_parent_from_product

    result = refine_orientation_relationship_from_child_boundaries(
        orientations,
        adjacency,
        parent,
        product,
        initial,
        maximum_correction_deg=4.0,
        trim_fraction=1.0,
        huber_delta_deg=1.0,
        minimum_improvement_deg2=1.0e-5,
        maximum_iterations=80,
    )
    initial_distance = orientation_relationship_distance_deg(
        initial,
        true_theory.variant_set.base_R_parent_from_product,
        parent.proper_symmetry_cartesian,
        product.proper_symmetry_cartesian,
    )
    fitted_distance = orientation_relationship_distance_deg(
        result.fitted_R_parent_from_product,
        true_theory.variant_set.base_R_parent_from_product,
        parent.proper_symmetry_cartesian,
        product.proper_symmetry_cartesian,
    )
    assert fitted_distance < initial_distance
    assert fitted_distance < 0.35
    assert result.objective_fitted < result.objective_initial
    assert result.accepted_improvement
    assert result.n_starts == 7
    assert result.objective_evaluations > 0


def test_variant_graph_keeps_clean_parent_domain_and_rejects_random_outlier():
    parent, product, theory, _, orientations, _ = _synthetic_child_grains(5)
    random_outlier = Rotation.random(1, random_state=123).as_matrix()[0]
    values = np.concatenate((orientations, random_outlier[None, :, :]), axis=0)
    adjacency = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))

    report = reconstruct_variant_graph_candidates(
        values,
        adjacency,
        theory.variant_set.variants,
        product.proper_symmetry_cartesian,
        parent.proper_symmetry_cartesian,
        link_tolerance_deg=1.0,
        reconstruction_tolerance_deg=1.0,
        minimum_grains=3,
    )
    assert report.domain_candidates
    assert any(
        set(domain.grain_indices) == {0, 1, 2, 3, 4}
        for domain in report.domain_candidates
    )
    assert 5 in report.unassigned_grains or all(
        5 not in domain.grain_indices for domain in report.domain_candidates
    )


def test_noise_and_outliers_do_not_force_false_exactness():
    parent, product, theory, _, orientations, adjacency = _synthetic_child_grains(6)
    noisy = add_isotropic_orientation_noise(
        orientations, 0.35, seed=20260921
    )
    contaminated, mask = inject_random_orientation_outliers(
        noisy, 1.0 / 6.0, seed=44
    )
    assert np.count_nonzero(mask) == 1

    clean_score = score_theory_consistency(
        noisy,
        adjacency,
        theory,
        parent.proper_symmetry_cartesian,
        product.proper_symmetry_cartesian,
    )
    dirty_score = score_theory_consistency(
        contaminated,
        adjacency,
        theory,
        parent.proper_symmetry_cartesian,
        product.proper_symmetry_cartesian,
    )
    assert clean_score.operator_median_deg < 2.0
    assert dirty_score.operator_p90_deg > clean_score.operator_p90_deg


def test_segmentation_sweep_reports_plateau_and_then_merge_without_picking_truth():
    phase = _phase(1, "cubic", Lattice.cubic(3.0), "m-3m")
    orientations = np.asarray(
        [
            Rotation.from_euler("z", angle, degrees=True).as_matrix()
            for angle in (0.0, 0.3, 0.6, 12.0, 12.3, 12.6)
        ]
    )
    data = EBSDMap(
        orientations=orientations,
        phase_id=np.ones(6, dtype=int),
        indexed=np.ones(6, dtype=bool),
        x=np.arange(6, dtype=float),
        y=np.zeros(6),
        z=np.zeros(6),
    )
    graph = build_neighbor_graph(data, radius=1.01)
    sweep = sweep_segmentation_thresholds(
        data,
        {1: phase},
        [1.0, 3.0, 5.0, 15.0],
        neighbor_graph=graph,
    )
    assert [item.n_grains for item in sweep.entries[:3]] == [2, 2, 2]
    assert sweep.entries[-1].n_grains == 1
    assert sweep.consecutive_adjusted_rand[:2] == pytest.approx((1.0, 1.0))
    assert sweep.consecutive_adjusted_rand[-1] < 1.0
    assert adjusted_rand_index(
        sweep.entries[0].labels, sweep.entries[0].labels
    ) == pytest.approx(1.0)


def test_boundary_trace_uses_surface_geometry_and_both_crystal_sides():
    phase = _phase(
        1,
        "orthorhombic",
        Lattice.orthorhombic(3.0, 4.0, 5.0),
        "mmm",
    )
    coords = [
        (0.0, 0.0), (0.0, 1.0), (0.0, 2.0),
        (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
    ]
    data = EBSDMap(
        orientations=np.repeat(np.eye(3)[None, :, :], 6, axis=0),
        phase_id=np.ones(6, dtype=int),
        indexed=np.ones(6, dtype=bool),
        x=np.asarray([item[0] for item in coords]),
        y=np.asarray([item[1] for item in coords]),
        z=np.zeros(6),
    )
    graph = build_neighbor_graph(data, radius=1.01)
    segmentation = GrainSegmentation(
        grain_id=np.array([0, 0, 0, 1, 1, 1], dtype=int),
        n_grains=2,
        threshold_deg=1.0,
        neighbor_graph=graph,
    )
    estimate = estimate_boundary_trace(
        data,
        segmentation,
        0,
        1,
        surface_normal_sample=np.array([0.0, 0.0, 1.0]),
    )
    assert estimate.linearity > 0.999999
    hypothesis = TwinPlaneHypothesis(
        "x-normal plane",
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    result = rank_trace_hypotheses(
        estimate,
        np.eye(3),
        np.eye(3),
        phase,
        [hypothesis],
    )[0]
    assert result.maximum_trace_residual_deg < 1.0e-9
    assert result.plane_normal_coherence_deg < 1.0e-9


def test_curved_or_underresolved_trace_is_not_silently_called_exact():
    phase = _phase(1, "cubic", Lattice.cubic(3.0), "m-3m")
    coords = np.array(
        [
            [0, 0, 0], [1, 0, 0], [2, 0, 0],
            [0, 1, 0], [1, 1, 0], [2, 1, 0],
            [0, 2, 0], [1, 2, 0], [2, 2, 0],
        ],
        dtype=float,
    )
    labels = np.array([0, 0, 1, 0, 1, 1, 1, 1, 1], dtype=int)
    data = EBSDMap(
        orientations=np.repeat(np.eye(3)[None, :, :], 9, axis=0),
        phase_id=np.ones(9, dtype=int),
        indexed=np.ones(9, dtype=bool),
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
    )
    graph = build_neighbor_graph(data, radius=1.01)
    segmentation = GrainSegmentation(labels, 2, 1.0, graph)
    estimate = estimate_boundary_trace(
        data,
        segmentation,
        0,
        1,
        surface_normal_sample=np.array([0.0, 0.0, 1.0]),
    )
    hypothesis = TwinPlaneHypothesis(
        "candidate",
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    if estimate.linearity < 0.95:
        with pytest.raises(ValueError, match="not sufficiently line-like"):
            rank_trace_hypotheses(
                estimate,
                np.eye(3),
                np.eye(3),
                phase,
                [hypothesis],
                minimum_linearity=0.95,
            )
