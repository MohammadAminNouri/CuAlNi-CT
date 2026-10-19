"""Headless deployment-compatible Streamlit 3.14 startup and navigation smoke test."""
from __future__ import annotations
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]

def test_deployed_entrypoint_renders_accessible_linear_workflow():
    app = AppTest.from_file(str(ROOT / "twin_app" / "streamlit_app.py"), default_timeout=90)
    app.run()
    assert not app.exception, [str(e.message) for e in app.exception]
    assert len(app.tabs) == 0
    headings = [h.value for h in app.subheader]
    assert "01  /  Crystal setup" in headings
    assert "02  /  Calculated twin-family genealogy" in headings
    # Default NiTi/general workflow must not display Cu-Al-Ni reference guides.
    assert not any("Cu–Al–Ni · phase/setting reference" in e.label for e in app.get("expander"))
    guide_toggle = next(c for c in app.checkbox if c.label == "Show optional material-specific references")
    guide_toggle.set_value(True)
    app.run()
    assert not app.exception, [str(e.message) for e in app.exception]
    assert any("Cu–Al–Ni · phase/setting reference" in e.label for e in app.get("expander"))
