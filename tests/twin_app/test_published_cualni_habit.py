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


BENCHMARK = ROOT / "data" / "benchmarks" / "bhattacharya_cualni_orthorhombic_habit_v1.json"


def _payload(data: dict) -> dict[str, object]:
    inp = data["input"]
    parent = inp["parent"]
    product = inp["product"]
    request = CalculationRequest(
        project_id="published_cualni_habit_validation",
        title="Published CuAlNi orthorhombic twin/habit validation",
        parent=PhaseInput(
            phase_id="A",
            label="Parent",
            point_group=parent["point_group"],
            lattice=LatticeInput(
                parent["a"], parent["b"], parent["c"],
                parent["alpha"], parent["beta"], parent["gamma"], "angstrom",
            ),
        ),
        product=PhaseInput(
            phase_id="M",
            label="Product",
            point_group=product["point_group"],
            lattice=LatticeInput(
                product["a"], product["b"], product["c"],
                product["alpha"], product["beta"], product["gamma"], "angstrom",
            ),
        ),
        transformation=TransformationInput.from_rows(
            "A_to_M", "A", "M", inp["C_M_from_A"]
        ),
    )
    return request.to_project_payload()


def _all_constructions(report):
    return [
        construction
        for family in report.families
        for pair in family.pair_records
        for construction in pair.constructions
    ]


def _fraction_matches(value: float, target: float, tolerance: float) -> bool:
    return min(abs(value - target), abs((1.0 - value) - target)) <= tolerance


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


def test_non_niti_cualni_root_topology_is_recovered_from_input_only(published_case):
    data, report = published_case
    assert report.audit.topology_variant_count == data["expected"]["topology_variant_count"]
    assert report.audit.operator_count == data["expected"]["operator_count"]
    assert report.audit.stretch_variant_count == 6


def test_non_niti_cualni_published_type_i_and_type_ii_habits_are_recovered(published_case):
    data, report = published_case
    constructions = _all_constructions(report)
    tolerance = data["expected"]["outer_product_relative_tol"]

    for mode in data["expected"]["habit_modes"]:
        candidates = [
            construction
            for construction in constructions
            if construction.classification == mode["classification"]
            and any(
                _fraction_matches(
                    solution.other_variant_volume_fraction,
                    mode["lambda"],
                    mode["lambda_abs_tol"],
                )
                for solution in construction.habit_solutions
            )
        ]
        assert candidates, mode

        for target in mode["branches"]:
            best = float("inf")
            best_solution = None
            for construction in candidates:
                for solution in construction.habit_solutions:
                    if not _fraction_matches(
                        solution.other_variant_volume_fraction,
                        mode["lambda"],
                        mode["lambda_abs_tol"],
                    ):
                        continue
                    residual = _outer_residual_up_to_cubic_symmetry(
                        solution.shape_vector_parent_cartesian,
                        solution.habit_normal_parent_cartesian,
                        target["b"],
                        target["m"],
                    )
                    if residual < best:
                        best = residual
                        best_solution = solution
            assert best < tolerance, (mode, target, best)
            assert best_solution is not None
            assert best_solution.rank_one_residual < 1.0e-8
            assert best_solution.rotation_residual < 1.0e-8
            assert best_solution.middle_stretch_residual < 1.0e-8
            assert best_solution.frame_plane_residual < 1.0e-9
            assert best_solution.frame_shape_vector_residual < 1.0e-9
