"""Source-semantics and 6M basis-gauge regression, NOT experimental 18R proof."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from twin_app.p1_cualni import (
    CELLS, CrystalInputError, GAMMA_TO_UNIQUE_B, gram,
    reduced_gamma_to_unique_b, transport_reduced_gamma_correspondence,
)

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data" / "benchmarks"


def test_legacy_niti_vector_is_ambiguous_unit_direction_not_shape_magnitude():
    """Do not misrepresent a locked legacy unit vector as physical b."""
    old = json.loads((BENCH / "bhattacharya_niti_twin_habit_v1.json").read_text())
    audit = old["reference_vector_semantics_audit_2026"]
    assert audit["status"] == "UNRESOLVED_LEGACY_VECTOR_NORMALIZATION"
    assert audit["primary_table_checked"] is False
    assert audit["source_correction_claimed"] is False
    candidate = next(
        item for item in old["expected"]["habit_outer_products"]
        if item["classification"] == "Type II"
        and abs(item["shear"] - 0.2804) < 0.0001
    )
    unit_like = np.asarray(candidate["branches"][1]["b"], dtype=float)
    assert np.linalg.norm(unit_like) == pytest.approx(1.0, abs=1e-4)
    assert np.linalg.norm(np.asarray(candidate["branches"][0]["b"])) < 0.2

    physical = json.loads((BENCH / "otsuka_ren_2005_niti_typeii_habit_v1.json").read_text())
    assert physical["input"] == old["input"]
    magnitude = float(physical["expected"]["shape_magnitude"])
    direction = np.asarray(physical["expected"]["shape_direction_parent_cartesian"], dtype=float)
    normal = np.asarray(physical["expected"]["habit_normal_parent_cartesian"], dtype=float)
    assert magnitude == pytest.approx(0.13078, abs=1e-6)
    assert np.linalg.norm(direction) == pytest.approx(1, abs=5e-5)
    assert np.linalg.norm(normal) == pytest.approx(1, abs=5e-5)
    physical_b = magnitude * direction / np.linalg.norm(direction)
    # b in F=I+b⊗m has a physical magnitude; unit direction alone is not b.
    assert np.linalg.norm(physical_b) == pytest.approx(magnitude, abs=1e-12)
    assert np.linalg.norm(np.outer(physical_b, normal / np.linalg.norm(normal))) == pytest.approx(
        magnitude, abs=1e-12
    )
    assert abs(np.linalg.norm(unit_like) - magnitude) > 0.8


def test_rebased_reduced_6m_metric_and_exact_correspondence_are_consistent():
    """A passive daughter-cell reindexing must leave physical strain invariant."""
    old = CELLS["CuAlNi_6M_James_Hane_2000"].lattice
    rebased = reduced_gamma_to_unique_b(old)
    P = GAMMA_TO_UNIQUE_B
    assert P.det() == 1
    C_old = sp.Matrix([[1, sp.Rational(1, 2), 0],
                       [0, 1, sp.Rational(1, 3)],
                       [0, 0, 1]])
    new_rows = transport_reduced_gamma_correspondence(C_old.tolist())
    C_new = sp.Matrix([[sp.Rational(entry) for entry in row] for row in new_rows])
    assert P * C_new == C_old
    G_old = gram(old)
    G_new = gram(rebased["new_cell"])
    P_float = np.asarray(P, dtype=float)
    np.testing.assert_allclose(G_new, P_float.T @ G_old @ P_float, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(C_old, float).T @ G_old @ np.asarray(C_old, float),
        np.asarray(C_new, float).T @ G_new @ np.asarray(C_new, float),
        rtol=1e-12, atol=1e-10,
    )
    # Reciprocal and direct planes retain physical incidence.
    u_old = sp.Matrix([3, 2, -1])
    h_old = sp.Matrix([2, -3, 0])
    u_new = P.inv() * u_old
    h_new = P.T * h_old
    assert (h_new.T * u_new)[0] == (h_old.T * u_old)[0]


@pytest.mark.parametrize("invalid", [
    None,
    ((1, 0, 0), (0, 1, 0)),
    ((1, 0, 0), (0, 0, 0), (0, 0, 1)),
    (("bad", 0, 0), (0, 1, 0), (0, 0, 1)),
])
def test_setting_transport_requires_existing_nonsingular_exact_correspondence(invalid):
    with pytest.raises(CrystalInputError, match="correspondence|3x3|metrics"):
        transport_reduced_gamma_correspondence(invalid)


def test_6m_gamma_rebase_never_asserts_separately_measured_18r_identity():
    result = reduced_gamma_to_unique_b(CELLS["CuAlNi_6M_James_Hane_2000"].lattice)
    assert "not a proof" in result["warning"]
    assert result["volume_ratio"] == pytest.approx(1.0, abs=1e-12)
