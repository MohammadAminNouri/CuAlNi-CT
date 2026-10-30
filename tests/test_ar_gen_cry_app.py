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
    _radio(app, "Workflow").set_value("2 · Gen").run()
    assert len(app.exception) == 0, [str(e.message) for e in app.exception]
    assert len(app.warning) >= 1


def test_ar_navigation_is_protected_before_cry():
    app = AppTest.from_file(str(ENTRY), default_timeout=90).run()
    _radio(app, "Workflow").set_value("3 · AR").run()
    assert len(app.exception) == 0, [str(e.message) for e in app.exception]
    assert len(app.warning) >= 1


def test_report_does_not_invent_calculations():
    app = AppTest.from_file(str(ENTRY), default_timeout=90).run()
    _radio(app, "Workflow").set_value("4 · Report").run()
    assert len(app.exception) == 0, [str(e.message) for e in app.exception]
    assert len(app.warning) >= 1


def _radio(app: AppTest, label: str):
    controls = [item for item in app.radio if item.label == label]
    assert len(controls) == 1, f"Expected one radio control labeled {label!r}"
    return controls[0]


def _click_labeled(app: AppTest, label: str) -> AppTest:
    candidates = [button for button in app.button if button.label == label]
    assert len(candidates) == 1, f"Expected one {label!r} button"
    return candidates[0].click().run()


def test_complete_standalone_cry_to_gen_to_ar_to_report_workflow():
    """Real UI interactions must preserve each stage without another app."""
    app = AppTest.from_file(str(ENTRY), default_timeout=120).run()
    assert len(app.exception) == 0
    app = _click_labeled(app, "Run Cry")
    assert len(app.exception) == 0, [str(x.message) for x in app.exception]
    assert any(metric.label == "Parent d-spacing (Å)" for metric in app.metric)

    app = _click_labeled(app, "Continue to Gen")
    assert len(app.exception) == 0
    app = _click_labeled(app, "Run Gen")
    assert len(app.exception) == 0, [str(x.message) for x in app.exception]
    assert any(metric.label == "Variants" for metric in app.metric)

    app = _click_labeled(app, "Continue to AR")
    assert len(app.exception) == 0
    app = _click_labeled(app, "Continue to Report")
    assert len(app.exception) == 0, [str(x.message) for x in app.exception]
    assert any("Report" in title.value for title in app.subheader)


def test_archive_examples_and_manual_source_are_interactive():
    app = AppTest.from_file(str(ENTRY), default_timeout=120).run()
    assert len(app.exception) == 0, [str(x.message) for x in app.exception]
    assert any("NiTi" in choice for choice in app.selectbox[0].options)
    app.selectbox[0].set_value("AuTi · B2 → B19").run()
    assert len(app.exception) == 0, [str(x.message) for x in app.exception]
    app = _click_labeled(app, "Run Cry")
    assert app.session_state["cry_result"]["parent_phase"]["a"] == 3.25
    _radio(app, "Input source").set_value("Manual").run()
    assert "cry_result" not in app.session_state
    assert len(app.exception) == 0


def test_guided_correspondence_input_and_synthetic_ebsd_walkthrough():
    app = AppTest.from_file(str(ENTRY), default_timeout=120).run()
    app = _click_labeled(app, "Run Cry")
    app = _click_labeled(app, "Continue to Gen")
    _radio(app, "Correspondence input").set_value("Three direction pairs").run()
    assert len(app.exception) == 0, [str(x.message) for x in app.exception]
    app = _click_labeled(app, "Run Gen")
    assert len(app.exception) == 0, [str(x.message) for x in app.exception]
    assert app.session_state["gen_result"]["input_method"] == "Three direction pairs"
    app = _click_labeled(app, "Continue to AR")
    _radio(app, "EBSD source").set_value("Synthetic demo").run()
    app = _click_labeled(app, "Inspect EBSD")
    assert len(app.exception) == 0, [str(x.message) for x in app.exception]
    assert app.session_state["ar_result"]["indexed_points"] == 3
    assert app.session_state["ar_result"]["input_kind"] == "synthetic_software_demo"
    app = _click_labeled(app, "Continue to Report")
    assert len(app.exception) == 0, [str(x.message) for x in app.exception]


def test_streamlit_actual_niti_literature_benchmark_control():
    """UI must execute actual NiTi benchmark, not just render an inert link."""
    app=AppTest.from_file(str(ENTRY),default_timeout=180).run()
    app=_click_labeled(app,"Run Cry")
    app=_click_labeled(app,"Continue to Gen")
    app=_click_labeled(app,"Continue to AR")
    app=_click_labeled(app,"Run independent NiTi benchmarks")
    assert len(app.exception)==0,[str(e.message) for e in app.exception]
    benchmark=app.session_state["niti_benchmark_report"]
    assert benchmark["niti_OR_AQ_2020"]["daughter_variants"]==12
    assert benchmark["niti_CT_2026"]["intercorrespondence_operator_classes"]==7
    assert benchmark["physical_cualni_parent_certified"] is False
