from __future__ import annotations

from pathlib import Path


def test_production_twin_app_contains_no_published_answer_constants_or_named_theorist():
    root = Path(__file__).resolve().parents[2] / "twin_app"
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.glob("*.py")
    ).lower()

    # Known literature outputs belong in tests only, never runtime production code.
    for token in ("0.2385", "0.2804", "0.3096", "0.1423", "0.29202", "0.27102"):
        assert token not in text

    # The standalone UI/engine uses neutral scientific terminology.
    assert "cayron" not in text
