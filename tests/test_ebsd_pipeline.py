from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from cualni_cryst.ebsd_map import EBSDPhase
from cualni_cryst.ebsd_pipeline import (
    PipelineConfigError,
    load_config,
    orientation_convention_from_config,
    resolve_orientation_relationship,
    run_pipeline,
    validate_config_only,
)
from cualni_cryst.ebsd_theory_bridge import build_theory_library
from cualni_cryst.lattice import Lattice


def _phases():
    parent = EBSDPhase.from_point_group(
        1,
        "parent",
        Lattice(3.1, 3.1, 5.0),
        "4/mmm",
    )
    product = EBSDPhase.from_point_group(
        2,
        "product",
        Lattice.orthorhombic(3.0, 4.0, 5.0),
        "mmm",
    )
    return parent, product


def _base_or():
    return Rotation.from_euler(
        "ZXZ", [23.0, 37.0, 11.0], degrees=True
    ).as_matrix()


def _phase_config(phase_id, name, point_group, lattice):
    return {
        "id": phase_id,
        "name": name,
        "point_group": point_group,
        "lattice": lattice,
    }


def _write_synthetic_csv(tmp_path: Path):
    parent, product = _phases()
    theory = build_theory_library(parent, product, _base_or())
    assert theory.n_variants >= 3

    parent_g = Rotation.from_euler(
        "xyz", [11.0, -7.0, 23.0], degrees=True
    ).as_matrix()
    grain_orientations = [
        parent_g
        @ theory.variant_set.variants[index].R_parent_from_product
        for index in range(3)
    ]

    rows = []
    # Three vertical product grains, each 2 columns x 3 rows.
    for grain in range(3):
        euler = Rotation.from_matrix(
            grain_orientations[grain]
        ).as_euler("ZXZ", degrees=True)
        for local_x in range(2):
            for y in range(3):
                rows.append(
                    {
                        "phase": 2,
                        "x": float(2 * grain + local_x),
                        "y": float(y),
                        "phi1": euler[0],
                        "Phi": euler[1],
                        "phi2": euler[2],
                        "quality": 1.0,
                    }
                )
    path = tmp_path / "synthetic.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _config(tmp_path: Path, input_path: Path):
    return {
        "schema_version": 1,
        "input": {
            "path": str(input_path),
            "format": "csv",
            "convention": {
                "scipy_sequence": "ZXZ",
                "angle_unit": "degree",
                "raw_matrix_direction": "crystal_to_sample",
                "label": "synthetic test",
            },
            "schema": {
                "phase": "phase",
                "x": "x",
                "y": "y",
                "euler": ["phi1", "Phi", "phi2"],
                "quality": {"score": "quality"},
            },
        },
        "phases": [
            _phase_config(
                1,
                "parent",
                "4/mmm",
                {
                    "a": 3.1,
                    "b": 3.1,
                    "c": 5.0,
                },
            ),
            _phase_config(
                2,
                "product",
                "mmm",
                {
                    "a": 3.0,
                    "b": 4.0,
                    "c": 5.0,
                },
            ),
        ],
        "parent_phase_id": 1,
        "product_phase_id": 2,
        "orientation_relationship": {
            "mode": "matrix",
            "matrix": _base_or().tolist(),
            "matrix_direction": "parent_from_product",
        },
        "quality_filters": [
            {"field": "score", "op": ">=", "value": 0.5}
        ],
        "segmentation": {
            "main_threshold_deg": 2.0,
            "sweep_thresholds_deg": [1.0, 2.0, 4.0, 8.0],
            "minimum_grain_points": 2,
            "neighbor_radius": 1.01,
            "kam_max_neighbor_misorientation_deg": 5.0,
        },
        "theory": {
            "crosscheck_topology": True,
        },
        "or_refinement": {
            "enabled": False,
        },
        "parent_reconstruction": {
            "enabled": True,
            "link_tolerance_deg": 1.0,
            "reconstruction_tolerance_deg": 1.0,
            "minimum_grains": 2,
        },
        "boundary_classification": {
            "maximum_residual_deg": 1.0,
            "minimum_margin_deg": 0.1,
        },
        "trace_validation": {
            "enabled": False,
        },
        "output": {
            "directory": str(tmp_path / "results"),
            "run_name": "synthetic-run",
            "overwrite": False,
        },
    }



def test_orientation_convention_uses_valid_identity_defaults():
    convention = orientation_convention_from_config(
        {
            "scipy_sequence": "ZXZ",
            "angle_unit": "degree",
            "raw_matrix_direction": "crystal_to_sample",
        },
        required=True,
        context="input.convention",
    )
    assert convention is not None
    assert np.asarray(convention.sample_correction) == pytest.approx(
        np.eye(3), abs=0.0
    )
    assert np.asarray(convention.crystal_correction) == pytest.approx(
        np.eye(3), abs=0.0
    )


def test_orientation_convention_accepts_explicit_numpy_and_rational_matrix_values():
    convention = orientation_convention_from_config(
        {
            "sample_correction": np.eye(3),
            "crystal_correction": [
                ["1", "0", "0"],
                ["0", "1/1", "0"],
                ["0", "0", 1],
            ],
        },
        required=True,
        context="input.convention",
    )
    assert convention is not None
    assert np.asarray(convention.sample_correction) == pytest.approx(
        np.eye(3), abs=0.0
    )
    assert np.asarray(convention.crystal_correction) == pytest.approx(
        np.eye(3), abs=0.0
    )


def test_orientation_convention_rejects_non_so3_correction_as_config_error():
    with pytest.raises(
        PipelineConfigError,
        match="physically valid orientation convention",
    ):
        orientation_convention_from_config(
            {
                "sample_correction": [
                    [2, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            },
            required=True,
            context="input.convention",
        )


def test_orientation_convention_rejects_invalid_euler_sequence_statically():
    with pytest.raises(PipelineConfigError, match="Euler sequence"):
        orientation_convention_from_config(
            {"scipy_sequence": "ZZZ"},
            required=True,
            context="input.convention",
        )

def test_strict_config_rejects_unknown_root_key(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input": {},
                "phases": [],
                "parent_phase_id": 1,
                "product_phase_id": 2,
                "orientation_relationship": {},
                "segmentation": {},
                "output": {},
                "typo_key": 1,
            }
        )
    )
    with pytest.raises(PipelineConfigError, match="unknown keys"):
        load_config(path)


def test_parallelism_or_solver_can_define_identity_without_experimental_fit():
    parent = EBSDPhase.from_point_group(
        1, "A", Lattice.orthorhombic(3, 4, 5), "1"
    )
    product = EBSDPhase.from_point_group(
        2, "M", Lattice.orthorhombic(3, 4, 5), "1"
    )
    resolved = resolve_orientation_relationship(
        {
            "mode": "parallelisms",
            "parallelisms": [
                {
                    "parent": {
                        "kind": "direction",
                        "indices": [1, 0, 0],
                    },
                    "product": {
                        "kind": "direction",
                        "indices": [1, 0, 0],
                    },
                    "projective": False,
                },
                {
                    "parent": {
                        "kind": "direction",
                        "indices": [0, 1, 0],
                    },
                    "product": {
                        "kind": "direction",
                        "indices": [0, 1, 0],
                    },
                    "projective": False,
                },
            ],
        },
        parent_phase=parent,
        product_phase=product,
    )
    assert resolved.candidate_count == 1
    assert resolved.R_parent_from_product == pytest.approx(
        np.eye(3), abs=1.0e-12
    )


def test_validate_config_reports_provenance_without_running_map(tmp_path):
    input_path = _write_synthetic_csv(tmp_path)
    config = _config(tmp_path, input_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    report = validate_config_only(config_path)
    assert report["input_exists"]
    assert report["or_source_mode"] == "matrix"
    assert report["or_candidate_count"] == 1
    assert len(report["config_sha256"]) == 64



def test_validate_config_catches_bad_input_schema_without_reading_map(tmp_path):
    missing_map = tmp_path / "not-created.csv"
    config = _config(tmp_path, missing_map)
    config["input"]["schema"]["euler"] = ["phi1", "Phi"]
    config_path = tmp_path / "bad-schema.json"
    config_path.write_text(json.dumps(config))

    with pytest.raises(
        PipelineConfigError,
        match="exactly 3 columns",
    ):
        validate_config_only(config_path)

def test_end_to_end_real_map_pipeline_writes_atomic_reproducible_outputs(tmp_path):
    input_path = _write_synthetic_csv(tmp_path)
    config = _config(tmp_path, input_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    result = run_pipeline(config_path)
    run_dir = result.run_directory
    assert run_dir.is_dir()

    expected = {
        "summary.json",
        "resolved_config.json",
        "phases.csv",
        "grains.csv",
        "boundaries.csv",
        "parent_domains.csv",
        "grain_variant_assignments.csv",
        "trace_validation.csv",
        "segmentation_sweep.csv",
        "map_fields.npz",
        "theory.json",
    }
    assert expected.issubset({path.name for path in run_dir.iterdir()})

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["map_audit_original"]["n_points"] == 18
    assert summary["quality_filter_audit"]["retained_indexed_points"] == 18
    assert summary["segmentation"]["n_grains"] == 3
    assert (
        summary["orientation_relationship"]["initial_theory"]["n_variants"]
        >= 3
    )
    assert (
        summary["orientation_relationship"]["initial_theory"]["consistency"][
            "operator_median_deg"
        ]
        < 1.0e-5
    )
    assert summary["parent_reconstruction"]["n_domain_candidates"] >= 1
    assert summary["provenance"]["input_sha256"]

    grains = pd.read_csv(run_dir / "grains.csv")
    assert len(grains) == 3

    fields = np.load(run_dir / "map_fields.npz")
    assert fields["grain_id"].shape == (18,)
    assert fields["variant_id"].shape == (18,)


def test_output_is_not_overwritten_without_explicit_permission(tmp_path):
    input_path = _write_synthetic_csv(tmp_path)
    config = _config(tmp_path, input_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    first = run_pipeline(config_path)
    assert first.run_directory.exists()

    with pytest.raises(FileExistsError):
        run_pipeline(config_path)


def test_quality_filter_cannot_reference_missing_vendor_metric(tmp_path):
    input_path = _write_synthetic_csv(tmp_path)
    config = _config(tmp_path, input_path)
    config["quality_filters"] = [
        {"field": "DOES_NOT_EXIST", "op": ">=", "value": 0}
    ]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    with pytest.raises(PipelineConfigError, match="Available quality fields"):
        run_pipeline(config_path)
