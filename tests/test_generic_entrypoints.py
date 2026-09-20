import json
import subprocess
from pathlib import Path

from cualni_cryst.project_io import generic_template


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False)


def test_calpad_accepts_non_hane_perez_preset():
    result = _run(
        "cualni-calpad",
        "--project",
        "preset:perez_cerrato_2024_beta3",
        "cell",
        "--phase",
        "beta3_perez2024",
    )
    assert result.returncode == 0, result.stderr
    assert "13.817" in result.stdout
    assert "113.6" in result.stdout
    assert "2/m" in result.stdout


def test_calpad_accepts_project_after_subcommand_too():
    result = _run(
        "cualni-calpad",
        "cell",
        "--project",
        "preset:perez_cerrato_2024_beta3",
        "--phase",
        "beta3_perez2024",
    )
    assert result.returncode == 0, result.stderr
    assert "beta3_perez2024" in result.stdout


def test_two_phase_accepts_arbitrary_json_project_and_infers_two_phase_pair(
    tmp_path: Path,
):
    payload = generic_template()
    path = tmp_path / "generic.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = _run(
        "cualni-two",
        "summary",
        "--project",
        str(path),
        "--or",
        "identity",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["reference_phase_id"] == "phase_A"
    assert data["moving_phase_id"] == "phase_B"
    assert data["theory_origin"] == "user_defined"


def test_two_phase_multi_phase_project_requires_explicit_pair_for_user_or(
    tmp_path: Path,
):
    payload = generic_template()
    payload["phases"].append(
        {
            "phase_id": "phase_C",
            "cell": {
                "a": 3.0,
                "b": 4.0,
                "c": 5.0,
                "alpha": 80,
                "beta": 85,
                "gamma": 75,
            },
            "point_group": "1",
        }
    )
    path = tmp_path / "three.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = _run(
        "cualni-two",
        "summary",
        "--project",
        str(path),
        "--or",
        "identity",
    )
    assert result.returncode != 0
    assert "reference" in (result.stderr + result.stdout).lower()


def test_project_cli_lists_all_32_point_groups():
    result = _run("cualni-project", "point-groups")
    assert result.returncode == 0, result.stderr
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    assert any("m-3m" in line for line in rows)
    assert any("-6m2" in line for line in rows)
    assert any("-43m" in line for line in rows)


def test_project_cli_validates_perez_preset():
    result = _run(
        "cualni-project",
        "validate",
        "preset:perez_cerrato_2024_beta3",
    )
    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout


def test_two_phase_accepts_project_before_subcommand(tmp_path: Path):
    payload = generic_template()
    path = tmp_path / "generic_before.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = _run(
        "cualni-two",
        "--project",
        str(path),
        "summary",
        "--or",
        "identity",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["reference_phase_id"] == "phase_A"
    assert data["moving_phase_id"] == "phase_B"


def test_stored_or_rejects_conflicting_explicit_phase_endpoint(tmp_path: Path):
    payload = generic_template()
    payload["phases"].append(
        {
            "phase_id": "phase_C",
            "cell": {
                "a": 3.1,
                "b": 4.2,
                "c": 5.3,
                "alpha": 75,
                "beta": 83,
                "gamma": 67,
            },
            "point_group": "1",
        }
    )
    payload["orientations"] = [
        {
            "orientation_id": "stored_AB",
            "reference_phase_id": "phase_A",
            "moving_phase_id": "phase_B",
            "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }
    ]
    path = tmp_path / "stored_conflict.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = _run(
        "cualni-two",
        "summary",
        "--project",
        str(path),
        "--or",
        "stored:stored_AB",
        "--reference",
        "phase_C",
    )
    assert result.returncode != 0
    assert "reference" in (result.stderr + result.stdout).lower()
    assert "phase_C" in (result.stderr + result.stdout)
