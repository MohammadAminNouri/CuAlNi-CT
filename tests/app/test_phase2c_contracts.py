from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from app.phase2_archive import (
    ResearchArchiveError,
    build_research_archive,
    exact_project_payload,
    load_research_archive,
    serialize_research_archive,
)
from app.phase2_ebsd_guard import EBSDPreflightError, validate_ebsd_run_config


def _exact_project() -> dict:
    return {
        "schema_version": 1,
        "project_id": "frozen_phase2c_archive",
        "title": "Exact correspondence archive check",
        "phases": [
            {"phase_id": "A", "label": "A"},
            {"phase_id": "M", "label": "M"},
        ],
        "transformations": [
            {
                "transformation_id": "A_to_M",
                "parent_phase_id": "A",
                "product_phase_id": "M",
                "correspondence": [
                    ["0", "1", "1"],
                    ["1", "0", "0"],
                    ["0", "1/3", "-1/3"],
                ],
            }
        ],
    }


def _valid_ebsd_config() -> dict:
    phase = lambda pid, name: {
        "id": pid,
        "name": name,
        "point_group": "m-3m" if pid == 1 else "2/m",
        "lattice": {
            "a": 5.8 if pid == 1 else 4.4,
            "b": 5.8 if pid == 1 else 5.3,
            "c": 5.8 if pid == 1 else 13.0,
            "alpha_deg": 90.0,
            "beta_deg": 90.0 if pid == 1 else 100.0,
            "gamma_deg": 90.0,
            "length_unit": "angstrom",
        },
    }
    return {
        "schema_version": 1,
        "input": {
            "path": "map.csv",
            "format": "csv",
            "convention": {
                "scipy_sequence": "ZXZ",
                "angle_unit": "degree",
                "raw_matrix_direction": "crystal_to_sample",
                "sample_correction": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "crystal_correction": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            "schema": {
                "phase": "phase",
                "x": "x",
                "y": "y",
                "z": None,
                "indexed": "indexed",
                "euler": ["phi1", "Phi", "phi2"],
                "quaternion": None,
                "matrix": None,
                "quality": {},
            },
        },
        "phases": [phase(1, "parent"), phase(2, "product")],
        "parent_phase_id": 1,
        "product_phase_id": 2,
        "orientation_relationship": {
            "mode": "matrix",
            "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "matrix_direction": "parent_from_product",
        },
        "quality_filters": [],
        "segmentation": {
            "main_threshold_deg": 5.0,
            "sweep_thresholds_deg": [2, 3, 4, 5, 6, 8],
            "minimum_grain_points": 5,
            "neighbor_radius_factor": 1.15,
            "kam_max_neighbor_misorientation_deg": 5.0,
        },
        "or_refinement": {
            "enabled": True,
            "use_if_accepted": False,
            "maximum_correction_deg": 5.0,
            "trim_fraction": 0.65,
            "minimum_improvement_deg2": 0.02,
        },
        "parent_reconstruction": {
            "enabled": True,
            "link_tolerance_deg": 3.0,
            "reconstruction_tolerance_deg": 3.0,
            "minimum_grains": 2,
        },
        "boundary_classification": {
            "maximum_residual_deg": 3.0,
            "minimum_margin_deg": 0.5,
        },
        "trace_validation": {"enabled": False},
        "output": {"directory": "results", "run_name": "test", "overwrite": False},
    }


def test_archive_roundtrip_preserves_exact_rational_strings_and_machine_statuses():
    project = _exact_project()
    archive = build_research_archive(
        project_payload=project,
        calculated_draft_signature="sig-123",
        transformation_result={"ct_exact": False, "supercompatibility": None},
        unified_theory={
            "rows": [
                {
                    "exact": False,
                    "provenance": "Cayron CMC/SMC",
                    "notes": "nearest-degeneracy diagnostic",
                }
            ]
        },
    )
    text = serialize_research_archive(archive)
    content = load_research_archive(text)
    recovered = exact_project_payload(text)

    C = recovered["transformations"][0]["correspondence"]
    assert C[2][1] == "1/3"
    assert C[2][2] == "-1/3"
    assert content["transformation_result"]["ct_exact"] is False
    assert content["transformation_result"]["supercompatibility"] is None
    assert content["research_outputs"]["unified_theory"]["rows"][0]["provenance"] == "Cayron CMC/SMC"
    assert content["scientific_contract"]["archive_restores_calculated_state_automatically"] is False


def test_archive_serialization_is_deterministic():
    kwargs = dict(
        project_payload=_exact_project(),
        calculated_draft_signature="same",
        workbench_analyses={"z": 1, "a": [3, 2, 1]},
    )
    first = serialize_research_archive(build_research_archive(**kwargs))
    second = serialize_research_archive(build_research_archive(**kwargs))
    assert first == second


def test_archive_rejects_tampering():
    archive = build_research_archive(
        project_payload=_exact_project(), calculated_draft_signature="sig"
    )
    tampered = deepcopy(archive)
    tampered["content"]["project_payload"]["title"] = "tampered"
    with pytest.raises(ResearchArchiveError, match="integrity check failed"):
        load_research_archive(tampered)



def test_archive_serializes_nested_ebsd_dataframes_without_losing_rows():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame([{"phase": 1, "residual": 0.1}, {"phase": 2, "residual": 0.2}])
    archive = build_research_archive(
        project_payload=_exact_project(),
        calculated_draft_signature="sig-df",
        ebsd_result={"tables": {"boundaries.csv": frame}},
    )
    content = load_research_archive(serialize_research_archive(archive))
    rows = content["research_outputs"]["ebsd"]["tables"]["boundaries.csv"]
    assert rows == [{"phase": 1, "residual": 0.1}, {"phase": 2, "residual": 0.2}]

def test_valid_explicit_ebsd_configuration_passes_preflight():
    report = validate_ebsd_run_config(_valid_ebsd_config())
    assert report.status == "ready"
    assert any("phase mapping" in item for item in report.checks)
    assert any("ambiguity margin" in item for item in report.checks)
    assert any("surface normal is inferred" in item for item in report.checks)


def test_advanced_ebsd_format_cannot_run_without_explicit_schema_and_convention():
    config = _valid_ebsd_config()
    del config["input"]["schema"]
    with pytest.raises(EBSDPreflightError, match="input.schema"):
        validate_ebsd_run_config(config)


def test_ebsd_phase_identity_cannot_be_ambiguous():
    config = _valid_ebsd_config()
    config["product_phase_id"] = 1
    with pytest.raises(EBSDPreflightError, match="must be distinct"):
        validate_ebsd_run_config(config)


def test_ebsd_refined_or_cannot_be_used_when_refinement_is_disabled():
    config = _valid_ebsd_config()
    config["or_refinement"]["enabled"] = False
    config["or_refinement"]["use_if_accepted"] = True
    with pytest.raises(EBSDPreflightError, match="cannot be used"):
        validate_ebsd_run_config(config)


def test_ebsd_trace_analysis_requires_surface_normal_and_explicit_hypothesis():
    config = _valid_ebsd_config()
    config["trace_validation"] = {
        "enabled": True,
        "surface_normal_sample": [0, 0, 0],
        "minimum_linearity": 0.9,
        "hypotheses": [
            {
                "label": "candidate",
                "plane_side1_crystal": [1, 0, 0],
                "plane_side2_crystal": [0, 1, 0],
            }
        ],
    }
    with pytest.raises(EBSDPreflightError, match="surface_normal_sample must be non-zero"):
        validate_ebsd_run_config(config)


def test_ebsd_initial_or_must_be_proper_rotation():
    config = _valid_ebsd_config()
    config["orientation_relationship"]["matrix"] = [[1, 0, 0], [0, 1, 0], [0, 0, 2]]
    with pytest.raises(EBSDPreflightError, match="not a proper rotation"):
        validate_ebsd_run_config(config)


def test_phase2c_live_entrypoints_are_final_v10_and_guard_is_wired():
    root = Path(__file__).resolve().parents[2]
    entry = (root / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    research_page = (root / "app" / "pages" / "2_CT_Equivalence_Lab.py").read_text(encoding="utf-8")
    research_v9 = (root / "app" / "research_workspaces_v9.py").read_text(encoding="utf-8")
    research_v10 = (root / "app" / "research_workspaces_v10.py").read_text(encoding="utf-8")
    workstation_v9 = (root / "app" / "streamlit_workstation_v9.py").read_text(encoding="utf-8")

    assert "streamlit_workstation_v9" in entry
    assert "research_workspaces_v10" in research_page
    assert "research_workspaces_v9" in research_v10
    assert "Cayron-CT" in research_v10
    assert "validate_ebsd_config_file(config_path)" in research_v9
    assert "Complete reproducibility archive" in workstation_v9
    assert "Import never activates a calculated result automatically" in workstation_v9
