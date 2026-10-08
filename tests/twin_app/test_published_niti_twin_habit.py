from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "src" / "cualni_cryst").exists():
    pytest.skip("repository backend is not present in this extracted package", allow_module_level=True)

from app.application import CalculationRequest, LatticeInput, PhaseInput, TransformationInput
from cualni_cryst.point_groups import point_group_operations
from twin_app.scientific_engine import build_twin_family_report


BENCHMARK = ROOT / "data" / "benchmarks" / "bhattacharya_niti_twin_habit_v1.json"


def _projective_residual(a, b) -> float:
    u = np.asarray(a, dtype=float)
    v = np.asarray(b, dtype=float)
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    return float(min(np.linalg.norm(u - v), np.linalg.norm(u + v)))


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
                parent["a"], parent["b"], parent["c"],
                parent["alpha"], parent["beta"], parent["gamma"], "angstrom"
            ),
        ),
        product=PhaseInput(
            phase_id="M",
            label="Product",
            point_group=product["point_group"],
            lattice=LatticeInput(
                product["a"], product["b"], product["c"],
                product["alpha"], product["beta"], product["gamma"], "angstrom"
            ),
        ),
        transformation=TransformationInput.from_rows(
            "A_to_M", "A", "M", inp["C_M_from_A"]
        ),
    )
    return request.to_project_payload()


def _all_systems(report):
    return [system for family in report.families for system in family.classical_systems]


def _all_constructions(report):
    return [
        construction
        for family in report.families
        for pair in family.pair_records
        for construction in pair.constructions
    ]


def _matches_fraction(value: float, target: float, tol: float) -> bool:
    return min(abs(value - target), abs((1.0 - value) - target)) <= tol


def _outer_residual_up_to_cubic_symmetry(pred_b, pred_m, target_b, target_m) -> float:
    predicted = np.outer(np.asarray(pred_b, float), np.asarray(pred_m, float))
    target = np.outer(np.asarray(target_b, float), np.asarray(target_m, float))
    best = float("inf")
    for operation in point_group_operations("m-3m"):
        Q = np.asarray(operation, dtype=float)
        if np.linalg.det(Q) <= 0.0:
            continue
        transformed = Q @ target @ Q.T
        scale = max(np.linalg.norm(predicted), np.linalg.norm(transformed), 1.0e-12)
        best = min(best, float(np.linalg.norm(predicted - transformed) / scale))
    return best


@pytest.fixture(scope="module")
def published_case():
    data = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    report = build_twin_family_report(_payload(data), "A_to_M")
    return data, report


def test_published_twin_shear_families_are_recovered_from_inputs_only(published_case):
    data, report = published_case
    systems = _all_systems(report)
    constructions = _all_constructions(report)
    assert report.audit.topology_variant_count == 12

    for expected in data["expected"]["twin_shear_families"]:
        if expected["classification"] == "Compound":
            matches = [
                item for item in constructions
                if item.classification == "Compound"
                and abs(item.shear_magnitude - expected["shear"]) <= expected["abs_tol"]
            ]
        else:
            matches = [
                system for system in systems
                if system.classification == expected["classification"]
                and abs(system.shear_magnitude - expected["shear"]) <= expected["abs_tol"]
            ]
        assert matches, expected


def test_published_habit_fractions_are_recovered_and_noninterface_modes_remain_empty(published_case):
    data, report = published_case
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
            item for item in constructions
            if abs(item.shear_magnitude - expected["shear"]) <= expected["abs_tol"]
        ]
        assert candidates, expected
        assert all(not item.habit_solutions and not item.continuum_fraction for item in candidates)


def test_published_habit_outer_products_are_recovered_up_to_cubic_symmetry(published_case):
    data, report = published_case
    constructions = _all_constructions(report)

    for expected in data["expected"]["habit_outer_products"]:
        candidates = [
            item for item in constructions
            if item.classification == expected["classification"]
            and abs(item.shear_magnitude - expected["shear"]) <= 0.001
        ]
        assert candidates, expected
        for branch in expected["branches"]:
            best = float("inf")
            for item in candidates:
                for solution in item.habit_solutions:
                    if not _matches_fraction(
                        solution.other_variant_volume_fraction,
                        expected["lambda"],
                        0.003,
                    ):
                        continue
                    best = min(
                        best,
                        _outer_residual_up_to_cubic_symmetry(
                            solution.shape_vector_parent_cartesian,
                            solution.habit_normal_parent_cartesian,
                            branch["b"],
                            branch["m"],
                        ),
                    )
            assert best < 0.04, (expected, branch, best)


def test_compound_pair_crosslock_uses_both_discrete_routes(published_case):
    data, report = published_case
    expected = next(
        item for item in data["expected"]["twin_shear_families"]
        if item["classification"] == "Compound"
    )
    found = []
    for family in report.families:
        systems = {system.system_id: system for system in family.classical_systems}
        for pair in family.pair_records:
            for construction in pair.constructions:
                if (
                    construction.classification == "Compound"
                    and abs(construction.shear_magnitude - expected["shear"]) <= expected["abs_tol"]
                ):
                    routes = {
                        rep.route
                        for system_id in construction.classical_system_ids
                        for rep in systems[system_id].representations
                    }
                    found.append((construction, routes))
    assert found
    assert all(routes == {"I", "II"} for _, routes in found)
    assert all(item.discrete_plane_angle_deg is not None for item, _ in found)
    assert all(item.discrete_direction_angle_deg is not None for item, _ in found)
