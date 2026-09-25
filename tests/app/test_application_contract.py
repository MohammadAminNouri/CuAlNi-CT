from __future__ import annotations

import json

import numpy as np
import pytest

from app.application import (
    CalculationRequest,
    LatticeInput,
    PhaseInput,
    TransformationInput,
    calculate_payload,
    calculate_request,
    point_group_options,
)
from app.errors import ApplicationError
from cualni_cryst.calculation_service import CalculationKind, CalculationService
from cualni_cryst.project_io import project_from_dict


def _phase(phase_id: str, *, shifted: bool = False) -> PhaseInput:
    if shifted:
        lattice = LatticeInput(5.2, 5.4, 6.1, 81.0, 91.0, 99.0)
    else:
        lattice = LatticeInput(5.0, 5.5, 6.0, 80.0, 90.0, 100.0)
    return PhaseInput(
        phase_id=phase_id,
        label=phase_id,
        lattice=lattice,
        point_group="1",
        cell_representation="triclinic user cell",
    )


def _identity_request() -> CalculationRequest:
    return CalculationRequest(
        project_id="phase1_contract",
        title="Phase 1 exact identity contract",
        parent=_phase("A"),
        product=_phase("M"),
        transformation=TransformationInput.from_rows(
            "A_to_M",
            "A",
            "M",
            (("1", "0", "0"), ("0", "1", "0"), ("0", "0", "1")),
        ),
    )


def test_all_32_conventional_point_groups_are_exposed() -> None:
    rows = point_group_options()
    assert len(rows) == 32
    assert {row["symbol"] for row in rows} >= {"1", "2/m", "m-3m", "6/mmm"}


def test_exact_fraction_survives_application_boundary() -> None:
    request = CalculationRequest(
        project_id="fraction_contract",
        title="Exact fractional correspondence",
        parent=_phase("A"),
        product=_phase("M", shifted=True),
        transformation=TransformationInput.from_rows(
            "A_to_M",
            "A",
            "M",
            (("1", "0", "0"), ("0", "1/2", "1/2"), ("0", "-1/2", "1/2")),
        ),
    )
    payload = request.to_project_payload()
    loaded = project_from_dict(payload)
    C = loaded.project.transformation("A_to_M").correspondence.C_M_from_A
    assert str(C[1, 1]) == "1/2"
    assert str(C[2, 1]) == "-1/2"


def test_adapter_matches_direct_calculation_service() -> None:
    request = _identity_request()
    response = calculate_request(request)

    loaded = project_from_dict(request.to_project_payload())
    direct = CalculationService(loaded.project).compute(
        CalculationKind.TRANSFORMATION_BUNDLE,
        "A_to_M",
    )

    assert response.result["summary"]["ct_exact_compatible"] is True
    assert response.result["summary"]["degeneracy_order"] == 3
    np.testing.assert_allclose(
        response.result["metric"]["cmc_dimensional"],
        direct.metric.cmc_dimensional,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        response.result["metric"]["principal_stretches"],
        direct.metric.principal_stretches,
        rtol=0.0,
        atol=0.0,
    )


def test_arbitrary_triclinic_fractional_project_is_strict_json() -> None:
    request = CalculationRequest(
        project_id="generic_triclinic",
        title="Generic triclinic fractional test",
        parent=_phase("A"),
        product=_phase("M", shifted=True),
        transformation=TransformationInput.from_rows(
            "A_to_M",
            "A",
            "M",
            (("1", "0", "0"), ("0", "1/2", "1/2"), ("0", "-1/2", "1/2")),
        ),
    )
    response = calculate_request(request)
    encoded = response.to_json()
    decoded = json.loads(encoded)
    assert decoded["transformation_id"] == "A_to_M"
    assert decoded["result"]["metric"]["parent_metric"]


def test_project_save_reload_recalculates_same_metric_result() -> None:
    first = calculate_request(_identity_request())
    reloaded = json.loads(first.project_json())
    second = calculate_payload(reloaded)
    assert first.result["metric"] == second.result["metric"]
    assert first.result["summary"] == second.result["summary"]


def test_singular_correspondence_is_rejected_without_fallback() -> None:
    request = CalculationRequest(
        project_id="bad_C",
        title="Bad C",
        parent=_phase("A"),
        product=_phase("M"),
        transformation=TransformationInput.from_rows(
            "A_to_M",
            "A",
            "M",
            (("1", "0", "0"), ("0", "0", "0"), ("0", "0", "1")),
        ),
    )
    with pytest.raises(ApplicationError) as caught:
        calculate_request(request)
    assert caught.value.info.code == "INVALID_INPUT"
    assert "invertible" in caught.value.info.message.lower() or "singular" in caught.value.info.message.lower()


def test_nonfinite_lattice_is_rejected_at_ui_contract() -> None:
    request = CalculationRequest(
        project_id="bad_metric",
        title="Bad metric",
        parent=PhaseInput("A", "A", LatticeInput(float("nan"), 5.0, 5.0), point_group="1"),
        product=_phase("M"),
        transformation=TransformationInput.from_rows(
            "A_to_M",
            "A",
            "M",
            (("1", "0", "0"), ("0", "1", "0"), ("0", "0", "1")),
        ),
    )
    with pytest.raises(ApplicationError) as caught:
        calculate_request(request)
    assert caught.value.info.code == "INVALID_INPUT"
    assert "finite" in caught.value.info.message.lower()
