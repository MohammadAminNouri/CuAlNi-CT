"""Primary-source traceability is scientifically scoped and stable."""

from pathlib import Path

from twin_app.theory_primary_sources import (
    REFERENCES, citation_rows, thesis_reference_notes,
)


def test_exact_primary_sources_and_correction_are_registered():
    by_key = {r["key"]: r for r in REFERENCES}
    assert len(by_key) == len(REFERENCES)
    required = ("Cayron2026", "Cayron2022", "Cayron2019",
                "Cayron2019Correction", "BallJames1987", "JamesHane2000",
                "WLR1953", "BowlesMackenzie1954", "ChenEtAl2013",
                "CuAlNi2006", "TahseenDabade2026")
    assert set(required).issubset(by_key)
    assert by_key["Cayron2019Correction"]["doi"] == "10.1107/S2053273319009276"
    assert "necessary, NOT sufficient" in by_key["Cayron2019Correction"]["boundary"]
    assert "MINUS det" in by_key["ChenEtAl2013"]["boundary"]
    assert "PLUS det" in by_key["ChenEtAl2013"]["boundary"]
    assert "compound" in by_key["TahseenDabade2026"]["boundary"].lower()


def test_every_reference_has_a_precise_role_and_assumption():
    for r in REFERENCES:
        assert r["key"] and r["citation"] and r["theory"]
        assert len(r["used_for"]) > 35
        assert len(r["boundary"]) > 45
        assert r["doi"] or r["key"] == "WLR1953"
    assert len(citation_rows()) == len(REFERENCES)


def test_scope_warnings_are_exported_to_thesis_notes():
    notes = thesis_reference_notes()
    for word in ("Cayron2026", "BallJames1987", "WLR1953",
                 "ChenEtAl2013", "Cayron2019Correction", "CuAlNi2006"):
        assert word in notes
    assert "Theorem 2" in notes
    assert "not experimental" in notes


def test_theory_panel_has_visible_sources_and_export():
    source = (Path(__file__).resolve().parents[2] /
              "twin_app" / "theory_comparison_ui.py").read_text(encoding="utf-8")
    assert "render_academic_sources()" in source
    assert "thesis_reference_notes()" in source
    assert '"academic_primary_sources": citation_rows()' in source
    assert "render_cayron_paper_comparison(unified, equivalence)" in source
