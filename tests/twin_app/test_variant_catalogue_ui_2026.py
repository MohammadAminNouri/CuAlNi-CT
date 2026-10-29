from __future__ import annotations

"""Variant catalogue: count M and U separately, retain all corresponding M_i."""
from types import SimpleNamespace as NS
from pathlib import Path
import ast
import pytest

from twin_app.variant_catalogue_ui import build_variant_catalogue


def _fake_report():
    def variant(i, stretch):
        return NS(
            variant_index=i, representative_determinant=1,
            representative_parent_symmetry=(("1","0","0"),("0","1","0"),("0","0","1")),
            correspondence_M_from_A=(("1","0","0"),("0","1","0"),("0","0","1")),
            stretch_variant_index=stretch, stretch_mapping_residual=0.,
            mapping_status="mapped",
        )
    def pair(i,j,n):
        habit=NS(habit_branch=1,other_variant_volume_fraction=.3,
                 habit_normal_parent_cartesian=(.0,.0,1.0))
        c=NS(construction_id=f"P{n}.B1",classification="Type I",
             habit_solutions=(habit,))
        return NS(variant_i=i,variant_j=j,pair_id=f"M{i}–M{j}",
                  stretch_i=1 if i<3 else 2,stretch_j=1 if j<3 else 2,
                  constructions=(c,))
    audit=NS(topology_variant_count=3,stretch_variant_count=2,
             mapped_correspondence_variant_count=3,
             collapsed_correspondence_variant_count=1,
             unmapped_correspondence_variant_count=0)
    # Pair appears twice in different presentation orbits and must count once.
    p13=pair(1,3,1)
    return NS(transformation_id="A_to_M",audit=audit,
              correspondence_variants=(variant(1,1),variant(2,1),variant(3,2)),
              families=(NS(pair_records=(p13,pair(2,3,2))),
                        NS(pair_records=(p13,))))


def test_exact_correspondence_and_stretch_counts_are_distinct():
    report=build_variant_catalogue(_fake_report())
    assert report["total_correspondence_variants"]==3
    assert report["distinct_stretch_variants"]==2
    assert report["collapsed_correspondence_variants"]==1
    assert report["mapped_correspondence_variants"]==3
    assert report["unique_unordered_correspondence_pairs"]==2
    assert report["twin_construction_records"]==2
    assert report["rows"][2]["partners"]==(1,2)
    assert report["rows"][2]["habit_solution_records"]==2
    assert report["rows"][0]["correspondence_M_from_A"]


def test_variant_count_mismatch_and_duplicate_fail_closed():
    report=_fake_report()
    report.audit.topology_variant_count=4
    with pytest.raises(ValueError,match="topology"):
        build_variant_catalogue(report)
    report=_fake_report()
    report.correspondence_variants=(
        report.correspondence_variants[0],
        report.correspondence_variants[0],
        report.correspondence_variants[2],
    )
    with pytest.raises(ValueError,match="Duplicate"):
        build_variant_catalogue(report)


def test_variant_panel_is_after_original_twins_and_before_theory_comparison():
    root=Path(__file__).resolve().parents[2]
    streamlit=(root/"twin_app"/"streamlit_app.py").read_text()
    ast.parse(streamlit)
    assert streamlit.index("render_mode_report(catalogue, report)") < (
        streamlit.index("render_variant_catalogue(report)") <
        streamlit.index("render_theory_comparison(payload")
    )
    panel=(root/"twin_app"/"variant_catalogue_ui.py").read_text()
    ast.parse(panel)
    assert "All calculated martensite variants" in panel
