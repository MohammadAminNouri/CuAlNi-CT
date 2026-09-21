from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np
from scipy.spatial.transform import Rotation

from cualni_cryst.ebsd_analysis import (
    build_neighbor_graph,
    grain_statistics,
    segment_grains,
)
from cualni_cryst.ebsd_map import EBSDMap, EBSDPhase
from cualni_cryst.ebsd_phase_forensics import (
    PhaseForensicsSettings,
    assess_grain_route_reliability,
    compact_phase_forensics_report,
    phase_forensics_array_payload,
    run_phase_forensics,
)
from cualni_cryst.lattice import Lattice


def _phase(phase_id: int) -> EBSDPhase:
    return EBSDPhase.from_point_group(
        phase_id,
        f"anonymous_{phase_id}",
        Lattice.cubic(3.0 + 0.1 * phase_id),
        "m-3m",
    )


def _singleton_dominated_map():
    phases = {1: _phase(1), 2: _phase(2)}
    nx, ny = 20, 14
    xg, yg = np.meshgrid(
        np.arange(nx, dtype=float),
        np.arange(ny, dtype=float),
        indexing="xy",
    )
    x = xg.ravel()
    y = yg.ravel()
    n = len(x)

    phase = np.ones(n, dtype=int)
    minority = {
        (2, 2), (5, 2), (8, 2), (11, 2), (14, 2), (17, 2),
        (2, 6), (5, 6), (8, 6), (11, 6), (14, 6), (17, 6),
        (2, 10), (5, 10), (8, 10), (11, 10), (14, 10), (17, 10),
    }
    for index, (xi, yi) in enumerate(zip(x.astype(int), y.astype(int))):
        if (xi, yi) in minority:
            phase[index] = 2

    base_a = Rotation.from_euler(
        "xyz", [11.0, -9.0, 21.0], degrees=True
    ).as_matrix()
    relation = Rotation.from_euler(
        "ZXZ", [31.0, 18.0, 7.0], degrees=True
    ).as_matrix()
    base_b = base_a @ relation

    rng = np.random.default_rng(404)
    orientations = np.empty((n, 3, 3), dtype=float)
    for index in range(n):
        base = base_a if phase[index] == 1 else base_b
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = np.deg2rad(rng.normal(scale=0.08))
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
        quality={},
        metadata={"XSTEP": 1.0, "YSTEP": 1.0},
    )
    return data, phases


def _settings() -> PhaseForensicsSettings:
    return PhaseForensicsSettings(
        orientation_thresholds_deg=(2.0,),
        minimum_component_sizes=(1, 2, 3, 5),
        relation_minimum_interface_units=4,
        relation_seed_cap=24,
        relation_refine_seeds=2,
        null_permutations=32,
        reliability_component_min_points=2,
        reliability_minimum_spatial_supported_pixel_fraction=0.20,
        reliability_minimum_orientation_supported_pixel_fraction=0.20,
        reliability_minimum_supported_components=4,
        random_seed=77,
    )


def test_singleton_dominated_phase_is_rejected_before_interphase_optimizer():
    data, phases = _singleton_dominated_map()
    graph = build_neighbor_graph(data, radius=1.01)
    report = run_phase_forensics(
        data,
        phases,
        graph,
        settings=_settings(),
    )

    threshold = report.thresholds[0]
    phase2 = next(
        gate for gate in threshold.phase_reliability
        if gate.phase_id == 2
    )
    assert not phase2.interface_inference_allowed
    assert (
        "spatial_supported_pixel_fraction_below_minimum"
        in phase2.reason_codes
    )
    assert (
        "orientation_supported_pixel_fraction_below_minimum"
        in phase2.reason_codes
    )

    assert len(threshold.direct_relations) == 1
    relation = threshold.direct_relations[0]
    assert relation.status == "rejected_by_phase_reliability_gate"
    assert relation.best is None
    assert relation.n_fit_units == 0
    assert relation.n_holdout_units == 0


def test_grain_route_gate_is_route_specific_and_does_not_require_other_phase():
    data, phases = _singleton_dominated_map()
    graph = build_neighbor_graph(data, radius=1.01)
    segmentation = segment_grains(
        data,
        phases,
        threshold_deg=2.0,
        neighbor_graph=graph,
        minimum_points=5,
    )
    grains = grain_statistics(data, segmentation, phases)

    phase1 = assess_grain_route_reliability(
        data,
        segmentation,
        grains,
        phase_id=1,
        minimum_required_grains=1,
    )
    phase2 = assess_grain_route_reliability(
        data,
        segmentation,
        grains,
        phase_id=2,
        minimum_required_grains=1,
    )

    assert phase1.allowed
    assert phase1.retained_grains >= 1
    assert not phase2.allowed
    assert phase2.indexed_pixels > 0
    assert phase2.retained_grains == 0
    assert "no_retained_grains" in phase2.reason_codes


def test_compact_report_never_contains_point_or_component_label_arrays():
    data, phases = _singleton_dominated_map()
    graph = build_neighbor_graph(data, radius=1.01)
    report = run_phase_forensics(
        data,
        phases,
        graph,
        settings=_settings(),
    )

    compact = compact_phase_forensics_report(report)
    encoded = json.dumps(
        compact,
        default=lambda value: (
            value.tolist()
            if isinstance(value, np.ndarray)
            else asdict(value)
        ),
        sort_keys=True,
    )

    assert '"component_id"' not in encoded
    assert '"component_sizes"' not in encoded
    assert '"component_phase_id"' not in encoded

    threshold = compact["thresholds"][0]
    assert set(threshold["segmentation"]) == {
        "n_components",
        "threshold_deg",
    }

    arrays = phase_forensics_array_payload(report)
    assert arrays
    assert any(name.endswith("_component_id") for name in arrays)
    assert any(name.endswith("_component_sizes") for name in arrays)
    assert any(name.endswith("_component_phase_id") for name in arrays)


def test_reliability_support_size_is_internal_not_coupled_to_reporting_grid():
    data, phases = _singleton_dominated_map()
    graph = build_neighbor_graph(data, radius=1.01)

    settings = PhaseForensicsSettings(
        orientation_thresholds_deg=(2.0,),
        minimum_component_sizes=(1, 3, 5),
        reliability_component_min_points=2,
        relation_minimum_interface_units=4,
        relation_seed_cap=24,
        relation_refine_seeds=2,
        null_permutations=32,
        reliability_minimum_spatial_supported_pixel_fraction=0.20,
        reliability_minimum_orientation_supported_pixel_fraction=0.20,
        reliability_minimum_supported_components=4,
        random_seed=88,
    )
    report = run_phase_forensics(
        data,
        phases,
        graph,
        settings=settings,
    )

    gate_phase_1 = next(
        item
        for item in report.thresholds[0].phase_reliability
        if item.phase_id == 1
    )
    assert gate_phase_1.minimum_component_points == 2
    assert gate_phase_1.raw_supported_pixel_fraction > 0.90
    assert gate_phase_1.orientation_supported_pixel_fraction > 0.90
