"""Headless Streamlit startup smoke test for the deployed Twin Family Explorer."""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[2]


def test_deployed_entrypoint_renders_the_three_real_workspaces():
    app = AppTest.from_file(str(ROOT / "twin_app" / "streamlit_app.py"),
                            default_timeout=90)
    app.run()
    assert not app.exception, [str(e.message) for e in app.exception]
    assert len(app.tabs) == 3
    assert [tab.label for tab in app.tabs] == [
        "01  ·  Crystal setup",
        "02  ·  Twin family & modes",
        "03  ·  Evidence & Cu–Al–Ni",
    ]
