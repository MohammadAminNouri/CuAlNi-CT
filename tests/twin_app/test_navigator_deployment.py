from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

from twin_app import navigator_component as navigator

ROOT = Path(__file__).resolve().parents[2]


def _model() -> dict:
    return {
        "families": [
            {"label": "Family F1", "couples": [
                {"id": "F1:M1-M2", "label": "M1 ↔ M2", "state": "exact"}]},
            {"label": "Family F2", "couples": [
                {"id": "F2:M3-M4", "label": "M3 ↔ M4", "state": "unresolved"}]},
        ]
    }


def _fake_streamlit(monkeypatch):
    recorded = {"warnings": [], "options": []}

    def selectbox(label, options, *, index, format_func, key, help):
        assert "all calculated families" in label.lower()
        recorded["options"] = list(options)
        assert all(format_func(option) for option in options)
        return options[index]

    fake = SimpleNamespace(
        warning=lambda text: recorded["warnings"].append(text),
        info=lambda text: None,
        selectbox=selectbox,
    )
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    return recorded


def test_bundled_component_directory_exists_in_repository():
    directory = ROOT / "twin_app" / "navigator_frontend"
    index = directory / "index.html"
    assert directory.is_dir(), "Missing frontend directory in GitHub upload"
    assert index.is_file(), "Missing twin_app/navigator_frontend/index.html"
    assert index.stat().st_size > 1000
    assert navigator._frontend_assets_present()


def test_frontend_declares_complete_v1_streamlit_message_protocol():
    source = (ROOT / "twin_app" / "navigator_frontend" / "index.html").read_text()
    assert "apiVersion:1" in source, "Missing V1 ready handshake"
    assert "dataType:'json'" in source, "Missing V1 value type"
    assert "streamlit:componentReady" in source
    assert "streamlit:setComponentValue" in source
    assert "streamlit:setFrameHeight" in source
    assert "streamlit:render" in source
    assert "aria-pressed" in source
    assert "focus-visible" in source
    assert "prefers-reduced-motion" in source


def test_frontend_asset_is_covered_by_repository_manifest():
    path = ROOT / "twin_app" / "navigator_frontend" / "index.html"
    manifest = (ROOT / "MANIFEST.sha256").read_text()
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        + "  twin_app/navigator_frontend/index.html"
    ) in manifest


def test_missing_assets_fail_safe_to_all_family_native_selection(monkeypatch):
    record = _fake_streamlit(monkeypatch)
    monkeypatch.setattr(navigator, "_frontend_assets_present", lambda: False)
    result = navigator.render_navigator(_model(), selected="F2:M3-M4")
    assert result == "F2:M3-M4"
    assert record["options"] == ["F1:M1-M2", "F2:M3-M4"]
    assert record["warnings"]


def test_streamlit_registration_error_does_not_hide_scientific_results(monkeypatch):
    record = _fake_streamlit(monkeypatch)
    monkeypatch.setattr(navigator, "_frontend_assets_present", lambda: True)

    def broken_component():
        raise RuntimeError("simulated StreamlitAPIException in bridge")

    monkeypatch.setattr(navigator, "_component", broken_component)
    result = navigator.render_navigator(_model(), selected="F1:M1-M2")
    assert result == "F1:M1-M2"
    assert len(record["options"]) == 2
    assert record["warnings"]


def test_invalid_frontend_value_is_not_promoted_to_scientific_selection(monkeypatch):
    monkeypatch.setattr(navigator, "_frontend_assets_present", lambda: True)
    monkeypatch.setattr(navigator, "_component", lambda: (lambda **kwargs: "UNKNOWN"))
    assert navigator.render_navigator(_model(), selected="F1:M1-M2") is None
