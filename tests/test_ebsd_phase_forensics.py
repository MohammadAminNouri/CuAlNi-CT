from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from cualni_cryst.ebsd_analysis import (
    build_neighbor_graph,
    segment_grains,
)
from cualni_cryst.ebsd_map import EBSDMap, EBSDPhase
from cualni_cryst.ebsd_phase_forensics import (
    PhaseForensicsSettings,
    assert_component_segmentation_parity,
    assert_vectorized_disorientation_parity,
    assert_vectorized_mean_parity,
    discover_interphase_relation,
    exact_same_phase_edge_angles,
    interface_units,
    orientation_components,
    relation_quotient_distance_deg,
    run_phase_forensics,
    spatial_components,
    component_mean_orientations,
)
from cualni_cryst.lattice import Lattice


def _phase(phase_id):
    return EBSDPhase.from_point_group(
        phase_id,
        f"anonymous_{phase_id}",
        Lattice.cubic(3.0 + 0.1 * phase_id),
        "m-3m",
    )


def _grid_map():
    phase_a = _phase(1)
    phase_b = _phase(2)
    phases = {1: phase_a, 2: phase_b}

    nx = 12
    ny = 10
    x, y = np.meshgrid(
        np.arange(nx, dtype=float),
        np.arange(ny, dtype=float),
        indexing="xy",
    )
    x = x.ravel()
    y = y.ravel()
    n = len(x)

    phase = np.ones(n, dtype=int)
    # Isolated minority pixels: deliberately impossible to survive min_points=5.
    minority_xy = {(2, 2), (5, 2), (8, 2), (2, 6), (5, 6), (8, 6)}
    for index, (xi, yi) in enumerate(zip(x.astype(int), y.astype(int))):
        if (xi, yi) in minority_xy:
            phase[index] = 2

    g_a = Rotation.from_euler(
        "xyz", [13.0, -17.0, 29.0], degrees=True
    ).as_matrix()
    hidden_relation = Rotation.from_euler(
        "ZXZ", [31.0, 18.0, 7.0], degrees=True
    ).as_matrix()
    g_b = g_a @ hidden_relation

    orientations = np.empty((n, 3, 3), dtype=float)
    rng = np.random.default_rng(44)
    for index in range(n):
        base = g_a if phase[index] == 1 else g_b
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = np.deg2rad(rng.normal(scale=0.12))
        orientations[index] = (
            Rotation.from_rotvec(angle * axis).as_matrix() @ base
        )

    data = EBSDMap(
        orientations=orientations,
        phase_id=phase,
        indexed=np.ones(n, dtype=bool),
        x=x,
        y=y,
        z=np.zeros(n, dtype=float),
        quality={
            "GENERIC_QUALITY": np.linspace(0.0, 1.0, n),
        },
        metadata={"XSTEP": 1.0, "YSTEP": 1.0},
    )
    return data, phases, hidden_relation


def test_fast_disorientation_and_component_partition_crosslock_frozen_backend():
    data, phases, _ = _grid_map()
    graph = build_neighbor_graph(data, radius=1.01)

    assert_vectorized_disorientation_parity(
        data,
        phases,
        graph,
        maximum_edges_per_phase=40,
    )
    edge_angles = exact_same_phase_edge_angles(data, phases, graph)
    for threshold in (1.0, 2.0, 5.0):
        assert_component_segmentation_parity(
            data,
            phases,
            graph,
            edge_angles,
            threshold_deg=threshold,
        )


def test_vectorized_mean_crosslocks_frozen_mean():
    data, phases, _ = _grid_map()
    values = data.orientations[data.phase_id == 1][:50]
    assert_vectorized_mean_parity(
        values,
        phases[1].proper_symmetry_cartesian,
    )


def test_forensics_detects_phase_extinction_before_minimum_grain_filter():
    data, phases, _ = _grid_map()
    graph = build_neighbor_graph(data, radius=1.01)

    conventional = segment_grains(
        data,
        phases,
        threshold_deg=5.0,
        neighbor_graph=graph,
        minimum_points=5,
    )
    retained_phase_2 = np.any(
        (conventional.grain_id >= 0) & (data.phase_id == 2)
    )
    assert not retained_phase_2

    settings = PhaseForensicsSettings(
        orientation_thresholds_deg=(2.0, 5.0),
        minimum_component_sizes=(1, 2, 3, 5),
        relation_minimum_interface_units=4,
        relation_seed_cap=32,
        relation_refine_seeds=3,
        null_permutations=32,
        random_seed=9,
    )
    report = run_phase_forensics(
        data,
        phases,
        graph,
        settings=settings,
    )
    phase_2 = next(
        item for item in report.phase_survival
        if item.phase_id == 2
    )
    assert phase_2.raw_indexed_pixels == 6
    assert phase_2.raw_spatial_components == 6
    assert phase_2.raw_components_ge_size[5] == 0
    assert any("erase this phase" in warning for warning in report.warnings)

    for threshold in report.thresholds:
        p2 = next(
            item for item in threshold.phase_survival
            if item.phase_id == 2
        )
        assert p2.orientation_components == 6
        assert p2.components_ge_size[1] == 6
        assert p2.components_ge_size[5] == 0


def test_direct_interface_route_recovers_relation_despite_zero_min5_minority_grains():
    data, phases, hidden_relation = _grid_map()
    graph = build_neighbor_graph(data, radius=1.01)
    edge_angles = exact_same_phase_edge_angles(data, phases, graph)
    segmentation = orientation_components(
        data,
        graph,
        edge_angles,
        threshold_deg=2.0,
    )
    means = component_mean_orientations(
        data,
        phases,
        segmentation,
    )
    units = interface_units(
        data,
        graph,
        segmentation,
        means,
    )
    pair_units = tuple(
        unit for unit in units
        if unit.phase_a == 1 and unit.phase_b == 2
    )
    assert len(pair_units) >= 4

    result = discover_interphase_relation(
        pair_units,
        phases[1],
        phases[2],
        settings=PhaseForensicsSettings(
            orientation_thresholds_deg=(2.0,),
            minimum_component_sizes=(1, 5),
            relation_minimum_interface_units=4,
            relation_seed_cap=32,
            relation_refine_seeds=3,
            null_permutations=32,
            null_significance_level=0.20,
            random_seed=12,
        ),
    )
    assert result.best is not None
    distance = relation_quotient_distance_deg(
        result.best.R_phase_a_from_phase_b,
        hidden_relation,
        phases[1].proper_symmetry_cartesian,
        phases[2].proper_symmetry_cartesian,
    )
    assert distance < 1.0
    assert result.best.holdout_median_deg is not None
    assert result.best.holdout_median_deg < 1.0


def test_unrelated_interface_orientations_do_not_get_perfect_holdout_support():
    data, phases, _ = _grid_map()
    rng = np.random.default_rng(991)
    orientations = data.orientations.copy()
    minority = np.flatnonzero(data.phase_id == 2)
    orientations[minority] = Rotation.random(
        len(minority),
        random_state=rng,
    ).as_matrix()
    unrelated = EBSDMap(
        orientations=orientations,
        phase_id=data.phase_id,
        indexed=data.indexed,
        x=data.x,
        y=data.y,
        z=data.z,
        quality=data.quality,
        metadata=data.metadata,
    )
    graph = build_neighbor_graph(unrelated, radius=1.01)
    edge_angles = exact_same_phase_edge_angles(
        unrelated, phases, graph
    )
    segmentation = orientation_components(
        unrelated,
        graph,
        edge_angles,
        threshold_deg=2.0,
    )
    means = component_mean_orientations(
        unrelated,
        phases,
        segmentation,
    )
    units = interface_units(
        unrelated,
        graph,
        segmentation,
        means,
    )
    pair_units = tuple(
        unit for unit in units
        if unit.phase_a == 1 and unit.phase_b == 2
    )
    result = discover_interphase_relation(
        pair_units,
        phases[1],
        phases[2],
        settings=PhaseForensicsSettings(
            orientation_thresholds_deg=(2.0,),
            minimum_component_sizes=(1, 5),
            relation_minimum_interface_units=4,
            relation_seed_cap=32,
            relation_refine_seeds=3,
            null_permutations=32,
            random_seed=33,
        ),
    )
    if result.best is not None:
        assert (
            result.best.holdout_support_fraction is None
            or result.best.holdout_support_fraction < 1.0
            or result.best.holdout_median_deg > 0.5
        )
