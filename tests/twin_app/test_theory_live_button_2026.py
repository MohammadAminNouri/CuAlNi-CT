"""Actually click the deployed theory button; startup alone cannot detect a freeze."""
from pathlib import Path
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[2]


def test_default_niti_theory_button_finishes_and_renders_numeric_results():
    app = AppTest.from_file(
        str(ROOT / "twin_app" / "streamlit_app.py"), default_timeout=160,
    )
    app.run()
    assert not app.exception, [str(e.message) for e in app.exception]

    mode = next(
        btn for btn in app.button
        if btn.label == "Calculate physical twinning modes and habit planes"
    )
    mode.click()
    app.run()
    assert not app.exception, [str(e.message) for e in app.exception]

    compare = next(
        btn for btn in app.button
        if btn.key == "tf_theory_compare_run"
    )
    compare.click()
    app.run()
    assert not app.exception, [str(e.message) for e in app.exception]
    stored = app.session_state["tf_theory_comparison_result"]
    assert len(stored[1].rows) > 0
    assert stored[1].ptmc_report is not None
    assert stored[2].mm_independent_audit is not None
    assert any(
        h.value == "04  /  Compare crystallographic theories"
        for h in app.subheader
    )
    assert any(
        "Cayron's compatibility argument" in getattr(x, "value", "")
        for x in app.markdown
    )
    assert any(
        "Equations and results" in getattr(x, "value", "")
        for x in app.markdown
    )
