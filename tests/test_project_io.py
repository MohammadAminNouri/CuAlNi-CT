import json
from pathlib import Path

import numpy as np
import pytest

from cualni_cryst.project_io import (
    generic_template,
    load_project,
    project_from_dict,
)


def test_perez_cerrato_2024_beta3_preset_reproduces_published_cell():
    loaded = load_project("preset:perez_cerrato_2024_beta3")
    phase = loaded.project.phase("beta3_perez2024")
    lattice = phase.lattice
    assert np.isclose(lattice.a, 13.817)
    assert np.isclose(lattice.b, 5.2856)
    assert np.isclose(lattice.c, 4.3987)
    assert np.isclose(lattice.beta_deg, 113.6)
    assert phase.point_group_symbol == "2/m"
    assert len(phase.symmetry_operators) == 4
    volume = float(np.sqrt(np.linalg.det(lattice.metric())))
    assert np.isclose(volume, 294.37424582405083, rtol=0.0, atol=1e-10)


def test_james_hane_is_only_a_loadable_preset_not_loader_global_state():
    loaded = load_project("preset:james_hane_2000")
    assert loaded.project.project_id == "james_hane_cualni_6m_reference"
    assert len(loaded.project.phases) == 2


def test_arbitrary_triclinic_and_hexagonal_user_project_loads():
    payload = {
        "schema_version": 1,
        "project_id": "generic_test",
        "title": "generic",
        "phases": [
            {
                "phase_id": "tri",
                "cell": {
                    "a": 3.1,
                    "b": 4.2,
                    "c": 5.3,
                    "alpha": 75.0,
                    "beta": 83.0,
                    "gamma": 67.0,
                    "length_unit": "angstrom",
                },
                "point_group": "-1",
            },
            {
                "phase_id": "hex",
                "cell": {
                    "a": 2.95,
                    "b": 2.95,
                    "c": 4.68,
                    "alpha": 90.0,
                    "beta": 90.0,
                    "gamma": 120.0,
                    "length_unit": "angstrom",
                },
                "point_group": "6/mmm",
            },
        ],
    }
    loaded = project_from_dict(payload)
    assert loaded.project.phase("tri").point_group_symbol == "-1"
    assert loaded.project.phase("hex").point_group_symbol == "6/mmm"
    assert len(loaded.project.phase("hex").symmetry_operators) == 24


def test_incompatible_cell_and_point_group_fails_loudly():
    payload = {
        "schema_version": 1,
        "project_id": "bad",
        "phases": [
            {
                "phase_id": "wrong_tetragonal",
                "cell": {
                    "a": 3.0,
                    "b": 4.0,
                    "c": 5.0,
                    "alpha": 90,
                    "beta": 90,
                    "gamma": 90,
                },
                "point_group": "4/mmm",
            }
        ],
    }
    with pytest.raises(AssertionError, match="SYMMETRY_METRIC_MISMATCH"):
        project_from_dict(payload)


def test_explicit_symmetry_matrices_support_nonstandard_user_setting():
    payload = {
        "schema_version": 1,
        "project_id": "explicit",
        "phases": [
            {
                "phase_id": "rhombohedral_user",
                "cell": {
                    "a": 4.0,
                    "b": 4.0,
                    "c": 4.0,
                    "alpha": 78.0,
                    "beta": 78.0,
                    "gamma": 78.0,
                },
                "point_group": "user_setting",
                "symmetry_matrices": [
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                    [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                ],
            }
        ],
    }
    loaded = project_from_dict(payload)
    assert len(loaded.project.phase("rhombohedral_user").symmetry_operators) == 3


def test_exact_fractional_correspondence_and_stored_or_load():
    payload = generic_template()
    payload["transformations"] = [
        {
            "transformation_id": "A_to_B",
            "parent_phase_id": "phase_A",
            "product_phase_id": "phase_B",
            "correspondence_M_from_A": [
                ["1/2", 0, 0],
                [0, 1, 0],
                [0, 0, 2],
            ],
        }
    ]
    payload["orientations"] = [
        {
            "orientation_id": "measured_or",
            "reference_phase_id": "phase_A",
            "moving_phase_id": "phase_B",
            "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "theory_origin": "experimental",
            "transformation_id": "A_to_B",
        }
    ]
    loaded = project_from_dict(payload)
    corr = loaded.project.transformation("A_to_B").correspondence.C_M_from_A
    assert str(corr[0, 0]) == "1/2"
    assert (
        loaded.project.orientation("measured_or").theory_origin.value == "experimental"
    )


def test_representation_link_metric_and_volume_are_audited():
    payload = {
        "schema_version": 1,
        "project_id": "representations",
        "phases": [
            {
                "phase_id": "small",
                "physical_phase": "same phase",
                "cell": {"a": 1, "b": 1, "c": 1},
                "point_group": "1",
            },
            {
                "phase_id": "supercell",
                "physical_phase": "same phase",
                "cell": {"a": 2, "b": 1, "c": 1},
                "point_group": "1",
            },
        ],
        "representation_links": [
            {
                "link_id": "small_to_supercell",
                "from_phase_id": "small",
                "to_phase_id": "supercell",
                "coordinates_from_from_to": [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
        ],
    }
    loaded = project_from_dict(payload)
    audit = loaded.representation_audits()[0]
    assert audit.passed
    assert audit.metric_relative_residual < 1e-14
    assert audit.volume_jacobian_residual < 1e-14


def test_invalid_representation_link_is_rejected():
    payload = {
        "schema_version": 1,
        "project_id": "bad_rep",
        "phases": [
            {"phase_id": "a", "cell": {"a": 1, "b": 1, "c": 1}, "point_group": "1"},
            {"phase_id": "b", "cell": {"a": 2, "b": 1, "c": 1}, "point_group": "1"},
        ],
        "representation_links": [
            {
                "link_id": "wrong",
                "from_phase_id": "a",
                "to_phase_id": "b",
                "coordinates_from_from_to": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
        ],
    }
    with pytest.raises(ValueError, match="Representation-link validation failed"):
        project_from_dict(payload)


def test_json_file_round_trip_loading(tmp_path: Path):
    path = tmp_path / "project.json"
    path.write_text(json.dumps(generic_template()), encoding="utf-8")
    loaded = load_project(path)
    assert loaded.project.project_id == "my_crystallography_project"


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda p: p.__setitem__("phases", {}), "phases"),
        (
            lambda p: p["phases"][0].__setitem__("cell", []),
            "cell",
        ),
        (
            lambda p: p["phases"][0].__setitem__("symmetry_matrices", {}),
            "symmetry_matrices",
        ),
        (lambda p: p.__setitem__("sources", {}), "sources"),
        (
            lambda p: p.__setitem__(
                "sources",
                [{"key": "x", "citation": "x", "equations": {}}],
            ),
            "equations",
        ),
        (lambda p: p.__setitem__("composition", []), "composition"),
        (
            lambda p: p.__setitem__(
                "composition",
                {
                    "basis": "weight_percent",
                    "components": "Cu=100",
                },
            ),
            "components",
        ),
        (
            lambda p: p.__setitem__("thermomechanical_state", []),
            "thermomechanical_state",
        ),
        (
            lambda p: p.__setitem__("numerical_policy", []),
            "numerical_policy",
        ),
    ],
)
def test_schema_type_errors_fail_with_typeerror(mutator, match):
    payload = generic_template()
    mutator(payload)
    with pytest.raises(TypeError, match=match):
        project_from_dict(payload)


def test_top_level_json_type_error_is_not_misreported_as_value_error(tmp_path: Path):
    path = tmp_path / "bad_top_level.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="Top-level"):
        load_project(path)


def test_boolean_is_rejected_as_exact_correspondence_scalar_type():
    payload = generic_template()
    payload["transformations"] = [
        {
            "transformation_id": "bad_bool",
            "parent_phase_id": "phase_A",
            "product_phase_id": "phase_B",
            "correspondence_M_from_A": [
                [True, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
        }
    ]
    with pytest.raises(TypeError, match="Boolean"):
        project_from_dict(payload)


def test_provenance_wrong_type_fails_loudly():
    payload = generic_template()
    payload["phases"][0]["provenance"] = []
    with pytest.raises(TypeError, match="provenance"):
        project_from_dict(payload)
