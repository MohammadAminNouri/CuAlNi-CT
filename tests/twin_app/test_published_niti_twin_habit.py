from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "src" / "cualni_cryst").exists():
    pytest.skip(
        "repository backend is not present in this extracted package",
        allow_module_level=True,
    )

from app.application import (
    CalculationRequest,
    LatticeInput,
    PhaseInput,
    TransformationInput,
)
from cualni_cryst.point_groups import point_group_operations
from twin_app.scientific_engine import build_twin_family_report


LEGACY_BENCHMARK = (
    ROOT / "data" / "benchmarks" / "bhattacharya_niti_twin_habit_v1.json"
)
PHYSICAL_HABIT_BENCHMARK = (
    ROOT / "data" / "benchmarks" / "otsuka_ren_2005_niti_typeii_habit_v1.json"
)


def _payload(data: dict) -> dict[str, object]:
    inp = data["input"]
    parent = inp["parent"]
    product = inp["product"]
    request = CalculationRequest(
        project_id="published_niti_validation",
        title="Published NiTi twin/habit validation",
        parent=PhaseInput(
            phase_id="A",
            label="Parent",
            point_group=parent["point_group"],
            lattice=LatticeInput(
                parent["a"],
                parent["b"],
                parent["c"],
                parent["alpha"],
                parent["beta"],
                parent["gamma"],
                "angstrom",
            ),
        ),
        product=PhaseInput(
            phase_id="M",
            label="Product",
            point_group=product["point_group"],
            lattice=LatticeInput(
                product["a"],
                product["b"],
                product["c"],
                product["alpha"],
                product["beta"],
                product["gamma"],
                "angstrom",
            ),
        ),
        transformation=TransformationInput.from_rows(
            "A_to_M",
            "A",
            "M",
            inp["C_M_from_A"],
        ),
    )
    return request.to_project_payload()


def _all_systems(report):
    return [
        system
        for family in report.families
        for system in family.classical_systems
    ]


def _all_constructions(report):
    return [
        construction
        for family in report.families
        for pair in family.pair_records
        for construction in pair.constructions
    ]


def _matches_fraction(value: float, target: float, tol: float) -> bool:
    return min(abs(value - target), abs((1.0 - value) - target)) <= tol


def _outer_residual_up_to_cubic_symmetry(
    pred_b,
    pred_m,
    target_b,
    target_m,
) -> float:
    predicted = np.outer(
        np.asarray(pred_b, float),
        np.asarray(pred_m, float),
    )
    target = np.outer(
        np.asarray(target_b, float),
        np.asarray(target_m, float),
    )
    best = float("inf")
    for operation in point_group_operations("m-3m"):
        Q = np.asarray(operation, dtype=float)
        if np.linalg.det(Q) <= 0.0:
            continue
        transformed = Q @ target @ Q.T
        scale = max(
            np.linalg.norm(predicted),
            np.linalg.norm(transformed),
            1.0e-12,
        )
        best = min(
            best,
            float(np.linalg.norm(predicted - transformed) / scale),
        )
    return best


@pytest.fixture(scope="module")
def published_case():
    legacy = json.loads(LEGACY_BENCHMARK.read_text(encoding="utf-8"))
    physical = json.loads(
        PHYSICAL_HABIT_BENCHMARK.read_text(encoding="utf-8")
    )
    assert legacy["input"] == physical["input"]
    report = build_twin_family_report(_payload(legacy), "A_to_M")
    return legacy, physical, report


def test_published_twin_shear_families_are_recovered_from_inputs_only(
    published_case,
):
    data, _, report = published_case
    systems = _all_systems(report)
    constructions = _all_constructions(report)
    assert report.audit.topology_variant_count == 12

    for expected in data["expected"]["twin_shear_families"]:
        if expected["classification"] == "Compound":
            matches = [
                item
                for item in constructions
                if item.classification == "Compound"
                and abs(item.shear_magnitude - expected["shear"])
                <= expected["abs_tol"]
            ]
        else:
            matches = [
                system
                for system in systems
                if system.classification == expected["classification"]
                and abs(system.shear_magnitude - expected["shear"])
                <= expected["abs_tol"]
            ]
        assert matches, expected


def test_published_habit_fractions_are_recovered_and_noninterface_modes_remain_empty(
    published_case,
):
    data, _, report = published_case
    constructions = _all_constructions(report)

    for expected in data["expected"]["habit_fraction_targets"]:
        matches = []
        for item in constructions:
            if item.classification != expected["classification"]:
                continue
            if abs(item.shear_magnitude - expected["shear"]) > 0.001:
                continue
            if any(
                _matches_fraction(
                    solution.other_variant_volume_fraction,
                    expected["lambda"],
                    expected["lambda_abs_tol"],
                )
                for solution in item.habit_solutions
            ):
                matches.append(item)
        assert matches, expected

    for expected in data["expected"]["no_exact_interface_shears"]:
        candidates = [
            item
            for item in constructions
            if abs(item.shear_magnitude - expected["shear"])
            <= expected["abs_tol"]
        ]
        assert candidates, expected
        assert all(
            not item.habit_solutions and not item.continuum_fraction
            for item in candidates
        )


def test_published_typeii_physical_habit_is_recovered_with_source_native_semantics(
    published_case,
):
    _, physical, report = published_case
    expected = physical["expected"]
    constructions = _all_constructions(report)

    direction = np.asarray(
        expected["shape_direction_parent_cartesian"],
        dtype=float,
    )
    direction_norm = float(np.linalg.norm(direction))
    assert abs(direction_norm - 1.0) < 5.0e-5
    direction /= direction_norm

    target_b = float(expected["shape_magnitude"]) * direction
    target_m = np.asarray(
        expected["habit_normal_parent_cartesian"],
        dtype=float,
    )
    assert abs(float(np.linalg.norm(target_m)) - 1.0) < 5.0e-5

    candidates = [
        item
        for item in constructions
        if item.classification == expected["classification"]
        and abs(item.shear_magnitude - expected["twin_shear"])
        <= expected["twin_shear_abs_tol"]
    ]
    assert candidates, expected

    best_outer = float("inf")
    best_magnitude = float("inf")
    best_solution = None

    for item in candidates:
        for solution in item.habit_solutions:
            if not _matches_fraction(
                solution.other_variant_volume_fraction,
                expected["lambda"],
                expected["lambda_abs_tol"],
            ):
                continue

            outer = _outer_residual_up_to_cubic_symmetry(
                solution.shape_vector_parent_cartesian,
                solution.habit_normal_parent_cartesian,
                target_b,
                target_m,
            )
            magnitude = abs(
                float(np.linalg.norm(solution.shape_vector_parent_cartesian))
                - float(expected["shape_magnitude"])
            )
            if (outer, magnitude) < (best_outer, best_magnitude):
                best_outer = outer
                best_magnitude = magnitude
                best_solution = solution

    assert best_solution is not None, expected
    assert best_outer < expected["outer_product_relative_tol"], (
        expected,
        best_outer,
    )
    assert best_magnitude <= expected["shape_magnitude_abs_tol"], (
        expected,
        best_magnitude,
    )

    assert best_solution.rank_one_residual < 1.0e-8
    assert best_solution.rotation_residual < 1.0e-8
    assert best_solution.middle_stretch_residual < 1.0e-8
    assert best_solution.frame_plane_residual < 1.0e-9
    assert best_solution.frame_shape_vector_residual < 1.0e-9


def test_compound_pair_crosslock_uses_both_discrete_routes(published_case):
    data, _, report = published_case
    expected = next(
        item
        for item in data["expected"]["twin_shear_families"]
        if item["classification"] == "Compound"
    )
    found = []
    for family in report.families:
        systems = {
            system.system_id: system
            for system in family.classical_systems
        }
        for pair in family.pair_records:
            for construction in pair.constructions:
                if (
                    construction.classification == "Compound"
                    and abs(
                        construction.shear_magnitude - expected["shear"]
                    )
                    <= expected["abs_tol"]
                ):
                    routes = {
                        rep.route
                        for system_id in construction.classical_system_ids
                        for rep in systems[system_id].representations
                    }
                    found.append((construction, routes))

    assert found
    assert all(routes == {"I", "II"} for _, routes in found)
    assert all(
        item.discrete_plane_angle_deg is not None
        for item, _ in found
    )
    assert all(
        item.discrete_direction_angle_deg is not None
        for item, _ in found
    )
