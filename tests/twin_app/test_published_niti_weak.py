from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "src" / "cualni_cryst").exists():
    pytest.skip("repository backend is not present in this extracted package", allow_module_level=True)

from app.application import CalculationRequest, LatticeInput, PhaseInput, TransformationInput
from cualni_cryst.weak_twins import BravaisNodeBasis
from twin_app.scientific_engine import build_twin_family_report


BENCHMARK = ROOT / "data" / "benchmarks" / "niti_higher_order_weak_v1.json"


def _payload(data: dict) -> dict[str, object]:
    inp = data["input"]
    p = inp["parent"]
    m = inp["product"]
    request = CalculationRequest(
        project_id="weak_validation",
        title="Weak-family validation",
        parent=PhaseInput(
            phase_id="A", label="Parent", point_group=p["point_group"],
            lattice=LatticeInput(p["a"], p["a"], p["a"], 90, 90, 90, "angstrom"),
        ),
        product=PhaseInput(
            phase_id="M", label="Product", point_group=m["point_group"],
            lattice=LatticeInput(m["a"], m["b"], m["c"], 90, m["beta"], 90, "angstrom"),
        ),
        transformation=TransformationInput.from_rows(
            "A_to_M", "A", "M", inp["C_M_from_A"]
        ),
    )
    return request.to_project_payload()


def test_higher_order_family_is_not_mislabelled_classical_and_published_weak_plane_is_found():
    data = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    expected = data["expected"]
    report = build_twin_family_report(
        _payload(data),
        "A_to_M",
        product_node_basis=BravaisNodeBasis.primitive_conventional(),
        weak_max_plane_index=data["input"]["max_plane_index"],
    )
    weak_families = [family for family in report.families if family.route == "axial_weak"]
    assert weak_families
    for family in weak_families:
        assert not family.classical_systems

    matches = [
        candidate
        for family in weak_families
        for candidate in family.weak_candidates
        if candidate.parent_rotation_order == expected["parent_rotation_order"]
        and candidate.parent_axis == tuple(expected["parent_axis"])
        and candidate.product_axis == tuple(expected["product_axis"])
        and candidate.plane1_primitive == tuple(expected["plane1"])
        and candidate.plane2_primitive == tuple(expected["plane2"])
    ]
    assert matches
    candidate = min(matches, key=lambda item: item.intraplanar_distortion)
    assert candidate.generalized_twin_index == expected["generalized_twin_index"]
    assert candidate.generalized_shear == pytest.approx(
        expected["generalized_shear"], abs=expected["generalized_shear_abs_tol"]
    )
    assert candidate.generalized_strain == pytest.approx(
        expected["generalized_strain"], abs=expected["generalized_strain_abs_tol"]
    )
