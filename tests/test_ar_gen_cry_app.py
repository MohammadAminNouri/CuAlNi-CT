from __future__ import annotations

"""Exercise the actual standalone Streamlit script through Streamlit AppTest."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "ar_gen_cry_app.py"


def test_separate_entrypoint_has_no_twin_ui_imports():
    source = ENTRY.read_text(encoding="utf-8")
    assert "app.streamlit_workstation" not in source
    assert "app.streamlit_app" not in source
    assert "from app." not in source
    assert "ar_gen_cry_pipeline" in source


def test_streamlit_initial_page_renders_without_exceptions():
    app = AppTest.from_file(str(ENTRY), default_timeout=90).run()
    assert len(app.exception) == 0, [str(e.message) for e in app.exception]
    assert any("AR · Gen · Cry" in x.value for x in app.title)
    assert len(app.radio) >= 1


def test_gen_navigation_is_protected_before_cry():
    app = AppTest.from_file(str(ENTRY), default_timeout=90).run()
    app.radio[0].set_value("2 · Gen").run()
    assert len(app.exception) == 0, [str(e.message) for e in app.exception]
    assert len(app.warning) >= 1


def test_ar_navigation_is_protected_before_cry():
    app = AppTest.from_file(str(ENTRY), default_timeout=90).run()
    app.radio[0].set_value("3 · AR").run()
    assert len(app.exception) == 0, [str(e.message) for e in app.exception]
    assert len(app.warning) >= 1


def test_report_does_not_invent_calculations():
    app = AppTest.from_file(str(ENTRY), default_timeout=90).run()
    app.radio[0].set_value("4 · Report").run()
    assert len(app.exception) == 0, [str(e.message) for e in app.exception]
    assert len(app.warning) >= 1
