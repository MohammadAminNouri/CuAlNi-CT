"""Real CTF path + Step-01 UI guard; no fabricated OR or phase assignment."""
from pathlib import Path

import numpy as np
import pytest

from twin_app.ebsd_integration import (
    MAX_CTF_BYTES, ctf_analysis_signature, inspect_ctf_upload,
    parse_explicit_or,
)


ROOT = Path(__file__).resolve().parents[2]
APP = (ROOT / "twin_app/streamlit_app.py").read_text(encoding="utf-8")
CTF_UI = (ROOT / "twin_app/ebsd_ctf_ui.py").read_text(encoding="utf-8")


def _ctf(phase_id=2):
    header = (
        "Channel Text File\nPrj test\nJobMode Grid\n"
        "XCells 2\nYCells 1\nXStep 1\nYStep 1\nPhases 2\n"
        "Phase X Y Bands Error Euler1 Euler2 Euler3 MAD BC BS\n"
    )
    points = [
        f"{phase_id} 0 0 8 0 12 21 33 0.2 100 200",
        f"{phase_id} 1 0 8 0 12 21 33 0.2 100 200",
    ]
    return (header + "\n".join(points) + "\n").encode("utf-8")


def test_real_ctf_preview_works_without_or_or_theoretical_result():
    preview = inspect_ctf_upload(_ctf(), filename="aztec.ctf")
    assert preview["indexed_pixels"] == 2
    assert preview["total_pixels"] == 2
    assert preview["phase_counts"] == {2: 2}
    assert preview["independent_or_available"] is False
    assert preview["phase_identification"].startswith("CTF phase IDs")
    assert len(preview["sha256"]) == 64


def test_preview_fails_closed_on_cif_size_and_invalid_path():
    for raw in (b"data_test", b"", b"Channel Text File" + b"\n" * 8):
        with pytest.raises(ValueError):
            inspect_ctf_upload(raw, filename="aztec.ctf")
    with pytest.raises(ValueError, match="35 MB"):
        inspect_ctf_upload(b"Channel Text File" + b"x" * MAX_CTF_BYTES,
                           filename="aztec.ctf")
    with pytest.raises(ValueError, match="simple"):
        inspect_ctf_upload(_ctf(), filename="../aztec.ctf")


def test_ui_input_coherent_with_source_cif_and_one_calculate_button():
    assert APP.index('st.subheader("01  /  Crystal setup")') < APP.index(
        'with st.expander("Experimental EBSD / CTF (optional)"'
    )
    assert APP.index('with st.expander("Experimental EBSD / CTF (optional)"') < APP.index(
        'st.button("Calculate physical twinning modes and habit planes"'
    )
    assert APP.index('st.button("Calculate physical twinning modes and habit planes"') < APP.index(
        'st.subheader("02  /  Calculated twinning modes")'
    )
    assert 'with st.expander("Crystal parameters and correspondence"' in APP
    assert 'st.file_uploader(f"{name.capitalize()} CIF"' in APP
    assert 'render_ctf_panel(' in APP


def test_or_is_not_generated_by_ctf_and_confirmation_is_fresh():
    assert '"I have an independently established parent-from-martensite OR"' in CTF_UI
    assert 'if not has_or:' in CTF_UI
    assert 'parse_explicit_or(matrix_txt)' in CTF_UI
    assert 'key="tf_ctf_v1_confirm_" + selected[:20]' in CTF_UI
    assert 'report is not None and not invalid and model_audit["pass"]' in CTF_UI
    assert 'not ready' in CTF_UI
    assert '"NOT CERTIFIED:' in CTF_UI
    assert "st.file_uploader(" in CTF_UI
    assert "max_upload_size=35" in CTF_UI
    assert "max_upload_size=2" in APP
    assert "st.cache_data(" in CTF_UI
    assert "independent_ctf_cartography" in CTF_UI
    assert 'Analyze indexed EBSD independently' in CTF_UI
    assert 'Download complete EBSD pipeline (.zip)' in CTF_UI
    assert "run_ctf_observation(" in CTF_UI


def test_ctf_audit_signature_changes_with_every_scientific_input():
    data = _ctf()
    f = ctf_analysis_signature
    a = f(data, "model1", "1 0 0\n0 1 0\n0 0 1", 1, 2, 3)
    assert a == f(data, "model1", "1 0 0\n0 1 0\n0 0 1", 1, 2, 3)
    changed = (
        f(data + b"\n", "model1", "1 0 0\n0 1 0\n0 0 1", 1, 2, 3),
        f(data, "model2", "1 0 0\n0 1 0\n0 0 1", 1, 2, 3),
        f(data, "model1", "0 -1 0\n1 0 0\n0 0 1", 1, 2, 3),
        f(data, "model1", "1 0 0\n0 1 0\n0 0 1", 2, 1, 3),
        f(data, "model1", "1 0 0\n0 1 0\n0 0 1", 1, 2, 4),
    )
    assert all(z != a for z in changed)
    assert len(set(changed)) == len(changed)


def test_orientation_is_proper_not_identity_by_default():
    with pytest.raises(ValueError):
        parse_explicit_or("")
    with pytest.raises(ValueError):
        parse_explicit_or("1 0 0\n0 1 0\n0 0 -1")
    np.testing.assert_allclose(
        parse_explicit_or("1 0 0\n0 1 0\n0 0 1"), np.eye(3),
    )
