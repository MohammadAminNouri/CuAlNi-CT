from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from twin_app.navigator_model import build_navigator_model, first_interpretable_couple
from twin_app.point_group_visualizer import cell_basis, element_geometry
from twin_app.report_export import make_export

ROOT = Path(__file__).resolve().parents[2]


def _report():
    def pair(i, j, *, compatible=False, unresolved=False):
        branch = NS(
            classification_status="classification cross-lock unresolved" if unresolved else "confirmed",
            habit_solutions=[NS()] if compatible else [], continuum_fraction=False,
        )
        return NS(variant_i=i, variant_j=j, constructions=(branch,),status="calculated")
    a = pair(1,2,compatible=True)
    b = pair(3,4,unresolved=True)
    c = pair(1,3)
    family_a=NS(family_id="F1", route="classical_exact", equivalent_pairs=((1,2),(3,4)),pair_records=(a,b))
    family_b=NS(family_id="F2", route="classical_exact", equivalent_pairs=((1,3),),pair_records=(c,))
    return NS(families=(family_a,family_b),parent_phase_id="A",product_phase_id="M")


def test_navigator_never_drops_families_or_couples_and_preserves_status():
    model=build_navigator_model(_report())
    assert model["summary"]["family_count"]==2
    assert model["summary"]["couple_count"]==3
    assert model["summary"]["couples_with_habit"]==1
    assert model["summary"]["unresolved_branches"]==1
    assert [n["id"] for f in model["families"] for n in f["couples"]]==[
        "F1:M1-M2","F1:M3-M4","F2:M1-M3"
    ]
    assert first_interpretable_couple(model)=="F1:M1-M2"
    assert model["families"][0]["couples"][1]["state"]=="unresolved"


def test_navigator_uses_accessible_semantics_without_network_dependencies():
    html=(ROOT/"twin_app"/"navigator_frontend"/"index.html").read_text()
    for token in (
        "aria-pressed", "aria-label", "aria-live", "focus-visible",
        "prefers-reduced-motion", "streamlit:componentReady", "streamlit:setComponentValue",
        "Show family overview", "Show all couples", "Find a couple",
    ):
        assert token in html
    for token in ("https://", "http://", "<script src=", "@xyflow/react", "elkjs"):
        assert token not in html
    assert html.count('class="shell"')==1


def test_plane_normals_use_the_reciprocal_metric_not_direct_axis():
    B=cell_basis(3.1,4.0,5.4,90.,100.,90.)
    label_axis,a=element_geometry(B,"proper 2-fold rotation",(0,0,1))
    label_plane,n=element_geometry(B,"mirror reflection",(0,0,1))
    assert "direct" in label_axis and "reciprocal" in label_plane
    assert np.isclose(np.linalg.norm(a),1.0)
    assert np.isclose(np.linalg.norm(n),1.0)
    # For a non-orthogonal metric, the [001] direct vector and (001)
    # reciprocal normal are *different* geometrical objects.
    assert abs(float(a@n)) < .999


def test_reproducible_export_includes_input_hash_and_not_reference_answers():
    class R:
        def to_dict(self):return {"families": [{"family_id":"F1"}], "audit": {"residual":1e-12}}
    x=make_export(R(),input_payload={"C":[[1,0,0],[0,1,0],[0,0,1]],"a":3.0})
    assert len(x["input_sha256"])==64
    assert x["scientific_report"]["audit"]["residual"]==1e-12
    assert "point" not in x["input_sha256"]
    assert "benchmark" not in x
    assert x["coordinate_conventions"]["K1"].startswith("product crystal")


def test_solver_is_not_modified_by_the_navigation_release():
    engine=(ROOT/"twin_app"/"scientific_engine.py").read_text()
    certifier=(ROOT/"twin_app"/"scientific_certifier.py").read_text()
    assert "certify_ptmc_twinning" in certifier
    assert "certify_ptmc_twinning" in engine
    assert "otsuka_ren_2005" not in engine


def test_ux_has_scientifically_explicit_no_solution_and_unresolved_states():
    source=(ROOT/"twin_app"/"tree_renderer.py").read_text()
    assert "No exact A/M habit plane for this twin branch" in source
    assert "Twin type not verified" in source
    assert "parent Cartesian, not Miller indices" in source
    assert "Second-variant fraction" in source
    assert "Shape-strain vector" in source
    assert "Habit-plane normal" in source
    assert "representative_habit_solutions" in source
    assert "render_navigator" in source
    assert "st.plotly_chart(" not in source
    ast.parse(source)
