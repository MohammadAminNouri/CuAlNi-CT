"""Independent reproductions of unsafe C float conversion and provenance overclaim."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import sympy as sp

from app.application import CalculationRequest, PhaseInput, LatticeInput, TransformationInput
from twin_app.numeric_domain import validate_solver_correspondence
from twin_app.input_logic import exact_matrix
from cualni_cryst.ball_james_adapter import _validated_invertible as bj_validated
from cualni_cryst.ptmc_adapter import _validated_invertible as ptmc_validated

NITI = (("0","0","1"),("1/2","1/2","0"),("-1/2","1/2","0"))


def test_unverified_user_inputs_do_not_claim_to_be_measured():
    a = PhaseInput("A", "austenite", LatticeInput(3.015,3.015,3.015,90,90,90,"angstrom"), "m-3m")
    m = PhaseInput("M", "martensite", LatticeInput(2.889,4.120,4.622,90,96.8,90,"angstrom"), "2/m")
    t = TransformationInput.from_rows("AM","A","M",NITI)
    assert a.to_payload()["provenance"]["status"] == "UNVERIFIED"
    assert m.to_payload()["provenance"]["status"] == "UNVERIFIED"
    assert t.to_payload()["provenance"]["status"] == "UNVERIFIED"
    payload = CalculationRequest("p","t",a,m,t).to_project_payload()
    assert all(phase["provenance"]["status"] == "UNVERIFIED" for phase in payload["phases"])
    assert payload["transformations"][0]["provenance"]["status"] == "UNVERIFIED"


def test_niti_and_cualni_published_inputs_are_in_numerically_supported_range():
    root = Path(__file__).resolve().parents[2]
    for filename in ("bhattacharya_niti_twin_habit_v1.json","bhattacharya_cualni_orthorhombic_habit_v1.json"):
        data = json.loads((root/"data"/"benchmarks"/filename).read_text())["input"]
        arr = validate_solver_correspondence(data["C_M_from_A"])
        assert np.linalg.cond(arr) < 100


@pytest.mark.parametrize("entry", [
    "1/"+str(10**400),
    str(10**400),
    "1/100000000000000",
])
def test_exactly_invertible_but_numerically_unsafe_matrices_are_rejected(entry):
    rows = ((entry,"0","0"),("0","1","0"),("0","0","1"))
    assert exact_matrix(rows).det() != 0
    with pytest.raises(ValueError, match="binary64|underflows|ill-conditioned"):
        validate_solver_correspondence(rows)


def test_sympy_matrix_input_is_checked_without_rounding():
    C = sp.diag(sp.Rational(1,10**400), 1, 1)
    assert C.det() != 0
    with pytest.raises(ValueError, match="underflows"):
        validate_solver_correspondence(C)


@pytest.mark.parametrize("validator", [bj_validated, ptmc_validated])
def test_independent_numerical_adapters_use_scale_relative_invertibility(validator):
    scaled = np.eye(3)*1.0e-6
    assert np.linalg.cond(scaled) == 1.0
    np.testing.assert_array_equal(validator(scaled,name="small but nonsingular"),scaled)
    with pytest.raises(ValueError,match="ill-conditioned"):
        validator(np.diag((1.0e-14,1.0,1.0)),name="badly conditioned")


def test_reject_nonfinite_maximum_condition():
    with pytest.raises(ValueError):
        validate_solver_correspondence(sp.eye(3),maximum_condition=float("nan"))
