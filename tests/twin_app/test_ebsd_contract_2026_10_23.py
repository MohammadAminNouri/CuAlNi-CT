"""Fail-closed contracts without needing Streamlit during test collection."""
from pathlib import Path
from twin_app import ebsd_integration


def test_no_deceptive_physical_twin_certification():
    root = Path(__file__).resolve().parents[2]
    assert ebsd_integration.EBSDObservation.__dataclass_fields__["physical_twin_evidence"].default == "NOT_CERTIFIED_FROM_CTF_ALONE"
    code = (root / "twin_app/ebsd_ctf_ui.py").read_text()
    assert "NOT CERTIFIED" in code
    assert "tf_ctf_v1_confirm" in code
    assert "run_ctf_observation" in code


def test_original_workstation_never_rewired_to_twin_app():
    root = Path(__file__).resolve().parents[2]
    assert "streamlit_workstation_v9" in (root / "app/streamlit_app.py").read_text()
    assert "ebsd_ctf_ui" in (root / "twin_app/streamlit_app.py").read_text()
