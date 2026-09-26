from __future__ import annotations

import numpy as np
import pytest

from app.errors import ApplicationError
from app.ui_components import constrain_cell
from app.workbench import (
    calpad_normal_conversion,
    manual_ptmc_cofactor_analysis,
    map_correspondence_object,
    orientation_from_matrix,
    reconstruct_sample_orientations,
)


def _phase(
    phase_id: str,
    *,
    a: float = 5.0,
    b: float = 5.5,
    c: float = 6.0,
    alpha: float = 80.0,
    beta: float = 90.0,
    gamma: float = 100.0,
) -> dict[str, object]:
    return {
        "phase_id": phase_id,
        "label": phase_id,
        "physical_phase": phase_id,
        "cell_representation": "triclinic user cell",
        "basis_id": f"{phase_id}_basis",
        "cell": {
            "a": a,
            "b": b,
            "c": c,
            "alpha_deg": alpha,
            "beta_deg": beta,
            "gamma_deg": gamma,
            "length_unit": "angstrom",
        },
        "point_group": "1",
        "provenance": {"status": "USER_MEASURED"},
    }


def _payload(C: list[list[object]] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "project_id": "phase2_contract",
        "title": "Phase 2 workstation contract",
        "phases": [_phase("A"), _phase("M", a=5.2, b=5.4, c=6.1, alpha=81.0, beta=91.0, gamma=99.0)],
        "transformations": [
            {
                "transformation_id": "A_to_M",
                "label": "A to M",
                "parent_phase_id": "A",
                "product_phase_id": "M",
                "correspondence_M_from_A": C or [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]],
            }
        ],
        "orientations": [],
        "sources": [],
        "representation_links": [],
    }


def _identity_metric_payload() -> dict[str, object]:
    phase_A = _phase("A")
    phase_M = _phase("M")
    return {
        "schema_version": 1,
        "project_id": "identity_metric",
        "title": "Identity metric",
        "phases": [phase_A, phase_M],
        "transformations": [
            {
                "transformation_id": "A_to_M",
                "parent_phase_id": "A",
                "product_phase_id": "M",
                "correspondence_M_from_A": [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]],
            }
        ],
        "orientations": [],
        "sources": [],
        "representation_links": [],
    }


def test_smart_cell_constraints_are_conventional_and_explicit() -> None:
    cubic = constrain_cell("cubic", a=3.6, b=8, c=9, alpha=70, beta=80, gamma=100)
    assert (cubic.a, cubic.b, cubic.c) == (3.6, 3.6, 3.6)
    assert (cubic.alpha, cubic.beta, cubic.gamma) == (90.0, 90.0, 90.0)

    monoclinic = constrain_cell("monoclinic", a=4, b=5, c=6, alpha=77, beta=103, gamma=88)
    assert (monoclinic.alpha, monoclinic.beta, monoclinic.gamma) == (90.0, 103, 90.0)

    hexagonal = constrain_cell("hexagonal", a=2.9, b=8, c=4.7, alpha=70, beta=80, gamma=90)
    assert (hexagonal.a, hexagonal.b, hexagonal.c) == (2.9, 2.9, 4.7)
    assert hexagonal.gamma == 120.0

    triclinic = constrain_cell("triclinic", a=4, b=5, c=6, alpha=77, beta=103, gamma=88)
    assert (triclinic.alpha, triclinic.beta, triclinic.gamma) == (77, 103, 88)


def test_correspondence_object_mapping_is_bidirectional_and_exact() -> None:
    payload = _payload([["2", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]])
    forward = map_correspondence_object(payload, "[1 0 0]", source_phase="parent")
    assert forward["mapped_exact_coefficients"] == ["2", "0", "0"]

    reverse = map_correspondence_object(payload, "[2 0 0]", source_phase="product")
    assert reverse["mapped_exact_coefficients"] == ["1", "0", "0"]

    plane = map_correspondence_object(payload, "(1 0 0)", source_phase="parent")
    assert plane["mapped_exact_coefficients"] == ["1/2", "0", "0"]


def test_orientation_reconstruction_roundtrip_parent_and_product() -> None:
    payload = _payload()
    R_A_from_M = [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    analysis = orientation_from_matrix(payload, R_A_from_M)
    assert len(analysis["variants"]) == 1  # point group 1 / point group 1

    identity = np.eye(3).tolist()
    product_candidates = reconstruct_sample_orientations(
        payload,
        analysis,
        identity,
        observed_phase="parent",
    )
    g_product = np.asarray(product_candidates["candidates"][0]["g_sample_from_reconstructed"], dtype=float)
    np.testing.assert_allclose(g_product, np.asarray(R_A_from_M), atol=1e-12, rtol=0.0)

    parent_candidates = reconstruct_sample_orientations(
        payload,
        analysis,
        g_product.tolist(),
        observed_phase="product",
    )
    g_parent = np.asarray(parent_candidates["candidates"][0]["g_sample_from_reconstructed"], dtype=float)
    np.testing.assert_allclose(g_parent, np.eye(3), atol=1e-12, rtol=0.0)


def test_orientation_report_exposes_forward_and_reverse_matrices() -> None:
    analysis = orientation_from_matrix(_payload(), np.eye(3).tolist())
    variant = analysis["variants"][0]
    # Identity OR can acquire machine-zero entries (~1e-17) when it is
    # re-expressed through two explicit Cartesian lattice frames. That is
    # representation roundoff, not a crystallographic discrepancy. Keep this
    # test much tighter than the project's scientific tolerances while avoiding
    # an invalid bit-for-bit floating-point requirement.
    forward = np.asarray(variant["R_parent_from_product"], dtype=float)
    reverse = np.asarray(variant["R_product_from_parent"], dtype=float)
    np.testing.assert_allclose(forward, np.eye(3), atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(reverse, np.eye(3), atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(reverse, forward.T, atol=1e-14, rtol=0.0)
    assert "x_parent" in analysis["convention"]


def test_calpad_normal_conversion_roundtrip_is_metric_correct() -> None:
    report = calpad_normal_conversion(_payload(), "A", "(1 0 0)", max_index=8)
    assert report["target_kind"] == "direction"
    assert report["roundtrip_projective_residual"] < 1e-12
    assert report["nearest_low_index"]["search_max_index"] == 8


def test_manual_ptmc_route_reports_backend_domain_result_without_fabrication() -> None:
    result = manual_ptmc_cofactor_analysis(
        _identity_metric_payload(),
        0,
        [0.1, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    )
    assert "cofactor" in result
    assert "ptmc" in result
    # For a degenerate identity stretch, the backend may report a continuum
    # rather than discrete roots. The application must preserve that fact.
    if not result["ptmc"]:
        assert isinstance(result["ptmc_note"], str)


def test_reconstruction_rejects_non_rotation_observation() -> None:
    payload = _payload()
    analysis = orientation_from_matrix(payload, np.eye(3).tolist())
    bad = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ApplicationError):
        reconstruct_sample_orientations(payload, analysis, bad, observed_phase="parent")
