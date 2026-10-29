from __future__ import annotations

"""18 unique actual Cu–Al–Ni papers: numerical reproduction and evidence gates.

One publication is not a new independent experiment every time another
publication quotes its lattice cell. Those papers are deliberately marked
BORROWED and must never be reported as full CuAlNi Eq42 benchmarks.
"""
from pathlib import Path
import ast
import math
import numpy as np
import pytest
from twin_app.cualni_paper_campaign_2026 import (
    campaign_registry,evaluate_one,run_campaign,
    _hane_shield_1999,_hane_shield_full_twin_ptmc_benchmark,_metric_evidence,
)


@pytest.fixture(scope="module")
def registry():
    return campaign_registry()


@pytest.fixture(scope="module")
def results():
    return run_campaign()


def test_at_least_ten_independently_identified_studies_with_distinct_dois(registry):
    p=registry["papers"]
    assert len(p)>=18
    assert len({x["doi"].lower() for x in p})==len(p)
    assert all(x["url"].startswith("https://") for x in p)
    assert all(not x["eligible_for_unqualified_full_Cayron_Eq42_or_BJ_CC123_validation"]
               for x in p)


@pytest.mark.parametrize("index",range(18))
def test_each_registered_paper_is_actually_evaluated_with_scope_and_provenance(
    registry,index
):
    p=registry["papers"][index]
    result=evaluate_one(p)
    assert result["doi"]==p["doi"]
    assert result["outcome"] and result["reason"]
    assert not result["paper_3d_twin_or_habit_prediction_validated"]
    assert not result["full_ct_amm_eq42_experimentally_comparable"]
    assert not result["full_bj_cc123_experimentally_comparable"]


def test_hane_shield_1999_independent_numerical_source_benchmark(results):
    record=next(x for x in results["source_records"] if x["id"]=="hane_shield_1999")
    source=record["numeric_checks"]["Hane_Shield_1999"]
    bj=record["numeric_checks"]["independent_BJ_PTMC"]
    assert record["outcome"]=="HANE_SHIELD_1999_BALL_JAMES_MALLARD_PTMC_NUMERICALLY_REPRODUCED"
    np.testing.assert_allclose(
        source["Hane_Shield_independent_stretches"],
        [1.0619,.9178,1.0230],atol=1.3e-4,rtol=0)
    assert source["distinct_stretch_U_variants"]==6
    assert source["ordered_U_pair_count"]==30
    assert source["unordered_U_pair_count"]==15
    assert source["single_variant_AM_IPS_exists"] is False
    assert source["single_variant_lambda2_minus_one"]>0.02
    assert source["volume_change_percent"]==pytest.approx(-0.3,abs=.02)
    assert bj["status"]=="NUMERIC_BALL_JAMES_TWIN_PTMC_SOURCE_VALUES_REPRODUCED"
    assert bj["compound_unordered_pairs"]==3
    assert bj["type_I_II_unordered_pairs"]==12
    assert bj["computed_twinned_habit_solution_records"]==96
    assert len(bj["all_type_I_II_twin_pair_ptmc_results"])==24
    shears_comp=[
        s["shear"] for p in bj["pair_results"]
        if p["parent_180_degree_generators"]==2
        for s in p["twin_solutions"]]
    shears_other=[
        s["shear"] for p in bj["pair_results"]
        if p["parent_180_degree_generators"]==1
        for s in p["twin_solutions"]]
    assert all(abs(v-.0747)<.0006 for v in shears_comp)
    assert all(abs(v-.261)<.0006 for v in shears_other)
    assert all(p["rank_one_relative_residual"]<5e-9
               for row in bj["pair_results"] for p in row["twin_solutions"])
    assert min(bj["representative_PTMC_type_I_II_fractions"]["I"])==pytest.approx(
        .290,abs=.002)
    assert min(bj["representative_PTMC_type_I_II_fractions"]["II"])==pytest.approx(
        .301,abs=.002)
    assert not any(bj["compound_family_PTMC_roots"].values())


def test_application_metric_matches_independently_derived_crystal_gram_matrices(
    results
):
    metric_sources=[
        ("hane_shield_1999","2H_cell"),
        ("ritter_yang_pope_laird_1979","source_2H_cell"),
        ("landa_sedlak_2007","registered_2H_cell"),
        ("perez_cerrato_2022_lpbf","published_borrowed_M18R_cell"),
        ("perez_cerrato_2022_lpbf","borrowed_2H_cell"),
        ("thin_film_2023_martensite","published_borrowed_M18R_cell"),
    ]
    byid={r["id"]:r for r in results["source_records"]}
    for paper,kind in metric_sources:
        cell=byid[paper]["numeric_checks"][kind]
        assert cell["metric_relative_error"]<2e-11
        assert cell["primitive_cell_volume_angstrom3"]>0
        assert len(cell["tested_reciprocal_d_spacings"])==6
        np.testing.assert_allclose(
            cell["calculated_metric"],cell["independent_metric"],atol=2e-10)


def test_1974_missing_18r_beta_is_not_replaced_with_90(registry):
    p=next(p for p in registry["papers"] if p["id"].startswith("otsuka_"))
    assert p["observables"]["beta_deg"] is None
    result=evaluate_one(p)
    assert result["outcome"]=="PARTIAL_METRIC__ANGLE_NOT_PUBLISHED"
    assert "primitive_cell_volume_angstrom3" not in str(result["numeric_checks"])
    with pytest.raises(ValueError,match="unknown angles"):
        _metric_evidence(p["observables"]["product_cell_A"],
                         "monoclinic_unique_b",None)


def test_2h_low_angle_source_not_silently_idealized_to_orthorhombic(registry):
    p=next(p for p in registry["papers"] if p["id"].startswith("zhang_xu"))
    a=evaluate_one(p)
    assert a["outcome"].startswith("PARTIAL_MONOCLINIC_DISTORTION")
    assert a["numeric_checks"]["measured_beta_minus_orthorhombic_deg"]==pytest.approx(
        -.57,abs=1e-10)
    assert "calculated_metric" not in str(a["numeric_checks"])


def test_borrowed_18r_cell_is_not_three_independent_measurements(results):
    entries=[
        r for r in results["source_records"]
        if r["id"] in ("perez_cerrato_2022_lpbf","thin_film_2023_martensite")
    ]
    assert len(entries)==2
    for row in entries:
        assert row["outcome"].startswith("BORROWED_REFERENCE_CELL_METRIC_PASS")
        assert not row["full_ct_amm_eq42_experimentally_comparable"]
    np.testing.assert_allclose(
        entries[0]["numeric_checks"]["published_borrowed_M18R_cell"]["calculated_metric"],
        entries[1]["numeric_checks"]["published_borrowed_M18R_cell"]["calculated_metric"],
        atol=1e-12,rtol=0,
    )


def test_old_18r_conventional_cell_not_mislabeled_as_6m_reduced(registry):
    p=next(p for p in registry["papers"] if p["id"]=="otsuka_nakamura_shimizu_1974")
    assert p["observables"]["phase"]=="18R_beta1_prime"
    assert p["observables"]["product_cell_A"][2]==38.
    assert p["observables"]["beta_deg"] is None


def test_chen_ebsd_twins_keep_distinct_indexed_plane_families(registry):
    p=next(p for p in registry["papers"] if p["id"]=="chen_gui_wang_2000")
    assert p["observables"]["AC_BD_twin_plane_family"]=="{121}_2H"
    assert p["observables"]["AD_BC_twin_plane_family"]=="{101}_2H"


def test_phase_population_or_heat_treatment_not_predicted_by_metric_only_solver(
    registry,results
):
    for id in ("sari_aksoy_2006","two_way_1999_singlecrystals",
               "neutron_pseudoelastic_2008","ritchie_et_al_2016_avalanche"):
        row=next(r for r in results["source_records"] if r["id"]==id)
        assert not row["full_ct_amm_eq42_experimentally_comparable"]
        assert not row["paper_3d_twin_or_habit_prediction_validated"]


def test_total_report_counts_are_not_false_shear_shear_successes(results):
    assert results["papers_evaluated"]==18
    assert results["unique_DOIs"]==18
    assert results["full_Cayron_AMM_or_BJ_experimental_predictions_validated"]==0
    assert results["paper_input_or_lattice_numerical_passes"]>=5
    assert len(results["source_records"])==18


def test_campaign_is_additive_to_existing_twin_ui_and_safety_guard():
    root=Path(__file__).resolve().parents[2]
    src=(root/"twin_app"/"cualni_paper_campaign_2026.py").read_text()
    ast.parse(src)
    assert "source-verified correspondence" in src
    assert "mallard_law_twins" in src
    assert "ptmc_volume_fractions" in src


def test_campaign_ui_preserves_main_ct_bj_ptmc_advisors_and_exposes_results():
    root=Path(__file__).resolve().parents[2]
    entry=(root/"twin_app"/"metric_design_ui.py").read_text()
    panel=(root/"twin_app"/"cualni_campaign_ui.py").read_text()
    ast.parse(entry)
    ast.parse(panel)
    for name in (
        "render_cualni_literature_campaign(signature)",
        "_render_joint_compatibility_design(",
        "_render_same_twin_supercompatibility_inverse(",
        "_render_full_paper_equation_audit(",
    ):
        assert name in entry
    for name in (
        "run_campaign()",
        "paper_numeric_input_reproduced",
        "full_Cayron_AMM_or_BJ_experimental_predictions_validated",
        "Export all 18 source-linked scientific results",
    ):
        assert name in panel
