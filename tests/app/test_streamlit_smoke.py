from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_workbench_starts_without_exception() -> None:
    app_path = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not at.exception
    assert any("CuAlNi-CT" in item.value for item in at.markdown)
    assert at.button
