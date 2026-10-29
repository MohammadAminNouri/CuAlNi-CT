from __future__ import annotations

"""Independent source-specific Cu-Al-Ni literature evidence campaign.

Does NOT infer C from a phase name, use metrics from another specimen,
or interpret a qualitative EBSD/twin report as an Eq.42 acceptance test.
The campaign deliberately separates (1) reproduction of source numbers,
(2) internal numerical/physical consistency of supplied crystal metrics,
(3) information needed for an empirical CT/BJ supercompatibility verdict.
"""
from pathlib import Path
import json
import math
from typing import Any

import numpy as np

FILE=Path(__file__).resolve().parents[1]/"data"/"benchmarks"/"cualni_18_paper_campaign_2026.json"


def campaign_registry() -> dict:
    record=json.loads(FILE.read_text(encoding="utf-8"))
    papers=record["papers"]
    if len(papers)<10:
        raise ValueError("Fewer than ten independent Cu-Al-Ni scientific papers")
    if len({p["doi"].lower() for p in papers})!=len(papers):
        raise ValueError("Duplicate DOI: a reused reference cannot count as a second paper")
    if len({p["id"] for p in papers})!=len(papers):
        raise ValueError("Duplicate paper record")
    for p in papers:
        if not p["doi"].startswith("10.") or not p["url"].startswith("https://"):
            raise ValueError("Unverified DOI or web reference")
        if p["eligible_for_unqualified_full_Cayron_Eq42_or_BJ_CC123_validation"]:
            raise ValueError(
                "No source in this campaign contains the necessary complete "
                "same-specimen frame-verified A/M, M/M and correspondence evidence"
            )
    return record


def _metric_independent(a:float,b:float,c:float,
                        alpha:float,beta:float,gamma:float) -> np.ndarray:
    """Independent 3D Gram-metric construction from lengths and angles (degrees)."""
    alpha,beta,gamma=map(math.radians,(alpha,beta,gamma))
    return np.array([
        [a*a,a*b*math.cos(gamma),a*c*math.cos(beta)],
        [a*b*math.cos(gamma),b*b,b*c*math.cos(alpha)],
        [a*c*math.cos(beta),b*c*math.cos(alpha),c*c],
    ],dtype=float)


def _metric_evidence(cell: list[float], setting: str, beta_deg:float|None=None):
    from cualni_cryst.lattice import Lattice
    a,b,c=map(float,cell)
    if setting=="orthorhombic":
        lattice=Lattice.orthorhombic(a,b,c,length_unit="angstrom")
        degrees=(90.,90.,90.)
    elif setting=="monoclinic_unique_b" and beta_deg is not None:
        lattice=Lattice.monoclinic_unique_b(
            a,b,c,float(beta_deg),length_unit="angstrom")
        degrees=(90.,float(beta_deg),90.)
    else:
        raise ValueError("Do not construct a metric from unknown angles or setting")
    reference=_metric_independent(a,b,c,*degrees)
    calculated=lattice.metric()
    relative_error=float(np.linalg.norm(reference-calculated)/np.linalg.norm(reference))
    if relative_error>2e-11:
        raise AssertionError(
            "Application metric disagrees with independent geometric Gram matrix"
        )
    if min(np.linalg.eigvalsh(calculated))<=0:
        raise AssertionError("Invalid indefinite measured crystal cell")
    expected_volume=float(np.sqrt(np.linalg.det(reference)))
    actual_volume=float(np.sqrt(np.linalg.det(calculated)))
    if not np.isclose(expected_volume,actual_volume,rtol=3e-11,atol=1e-12):
        raise AssertionError("Application metric volume disagrees with independent cell")
    # reciprocal 1/d^2 = h.T inv(M) h, no assumed indexing of source peaks
    probes=[]
    for h in ((1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(1,1,1)):
        v=np.asarray(h,float)
        reciprocal=float(v@np.linalg.solve(calculated,v))
        expected=float(v@np.linalg.solve(reference,v))
        if not np.isclose(reciprocal,expected,rtol=2e-11,atol=1e-12):
            raise AssertionError("Reciprocal metric inconsistent with independent d-spacing")
        probes.append({
            "hkl":h,"d_angstrom":float(1./np.sqrt(reciprocal))
        })
    return {
        "calculated_metric":calculated.tolist(),
        "independent_metric":reference.tolist(),
        "metric_relative_error":relative_error,
        "primitive_cell_volume_angstrom3":actual_volume,
        "tested_reciprocal_d_spacings":probes,
        "data_provenance":"SOURCE_CELL_GEOMETRY_ONLY",
        "experimental_twin_or_habit_angle_verified":False,
    }


def _hane_shield_1999(p:dict) -> dict:
    """Independent 1999 Eq.2 cubic/2H stretches and six U_i tensors.

    Reproduces the ORTHORHOMBIC six-stretch lattice family, not the
    correspondence-variant count M_i and not experimentally measured C.
    """
    o=p["observables"]
    a0=float(o["parent_cubic_A"])
    a,b,c=(float(x) for x in o["product_cell_A"])
    alpha=math.sqrt(2)*a/a0
    beta=b/a0
    gamma=math.sqrt(2)*c/a0
    independent=(alpha,beta,gamma)
    if not np.allclose(independent,o["stretches_reported"],rtol=0,atol=1.3e-4):
        raise AssertionError("Hane–Shield source-stated stretches not reproduced")
    h=(alpha+gamma)*.5
    k=(alpha-gamma)*.5
    variants=[]
    # U_i = one principal beta axis and the +/- off-diagonal alpha,gamma
    # splits in its perpendicular plane. This is explicitly the 1999
    # Eq.(2.1) six-variant cubic-to-orthorhombic stretch construction.
    for singled in range(3):
        for sign in (-1,1):
            U=np.eye(3)*h
            U[singled,singled]=beta
            i,j=[z for z in range(3) if z!=singled]
            U[i,j]=sign*k
            U[j,i]=sign*k
            ev=np.linalg.eigvalsh(U)
            if not np.allclose(np.sort(ev),np.sort(independent),atol=3e-13):
                raise AssertionError("Incorrect independent cubic-to-orthorhombic U tensor")
            variants.append(U)
    if len(variants)!=6 or any(
        np.allclose(variants[i],variants[j],atol=1e-13)
        for i in range(6) for j in range(i+1,6)
    ):
        raise AssertionError("Distinct stretch variants incorrectly collapsed")
    # The 1999 article counts 30 *ordered* pairs; unordered M/M stretch
    # pairs are 15. Its 12 compounds and 24 Type I/II solutions are not
    # raw unordered pair counts, and must not be misrepresented as such.
    n=len(variants)
    ordered=n*(n-1)
    if ordered!=o["variant_pair_count"]:
        raise AssertionError("Hane–Shield 1999 ordered pair count fails")
    volume_ratio=float(np.linalg.det(variants[0]))
    if abs(volume_ratio-1+0.003)>.001:
        raise AssertionError("Hane–Shield 1999 near-volume-preserving stretch fails")
    # lambda_2 is the middle stretch of one UNTWinned U_i.
    # The source reports that NO single-variant A/M interface is possible
    # although there are twinned-martensite habit planes.
    mid=float(sorted(independent)[1])
    if abs(mid-1.)<1e-5 or o["single_variant_AM_habit_exists"] is not False:
        raise AssertionError("Source's single-variant incompatibility lost")
    return {
        "Hane_Shield_independent_stretches":independent,
        "source_stretches":o["stretches_reported"],
        "computed_six_U_matrices":[u.tolist() for u in variants],
        "distinct_stretch_U_variants":n,
        "ordered_U_pair_count":ordered,
        "unordered_U_pair_count":n*(n-1)//2,
        "single_variant_lambda2_minus_one":mid-1.,
        "single_variant_AM_IPS_exists":False,
        "twinned_habit_prediction_in_source":True,
        "reported_twin_shears_not_reproduced_from_unpaired_correspondence":{
            "compound":o["compound_shear"],"type_I_II":o["type_I_II_shear"],
        },
        "source_predicted_twinned_habit_solutions":o[
            "twinned_habit_solution_count"
        ],
        "volume_change_percent":100*(volume_ratio-1.),
        "status":"SOURCE_NUMERICAL_STRETCH_AND_VARIANT_REPRODUCTION",
        "scope":"Reproduced orthorhombic Ball-James stretch tensors and single-variant incompatibility. The paper's PTMC twin shear and habit output require an additional source-consistent twinning solver comparison and are NOT marked reproduced.",
    }


def evaluate_one(p:dict) -> dict:
    o=p["observables"]
    out={
        "id":p["id"],"doi":p["doi"],"title":p["title"],
        "source_access":p["source_access"],
        "validation_level":p["validation_level"],
        "paper_numeric_input_reproduced":False,
        "paper_3d_twin_or_habit_prediction_validated":False,
        "full_ct_amm_eq42_experimentally_comparable":False,
        "full_bj_cc123_experimentally_comparable":False,
        "reason":p["limitation"],
        "numeric_checks":{},
    }
    if p["id"]=="hane_shield_1999":
        out["numeric_checks"]["Hane_Shield_1999"]=_hane_shield_1999(p)
        out["numeric_checks"]["2H_cell"]=_metric_evidence(
            o["product_cell_A"],"orthorhombic")
        out["paper_numeric_input_reproduced"]=True
        out["outcome"]="NUMERICAL_STRETCH_VARIANTS_PASS__FULL_TWIN_MODEL_NOT_YET_BLIND"
    elif p["id"]=="otsuka_nakamura_shimizu_1974":
        if o["beta_deg"] is not None:
            raise AssertionError("Do not invent a monoclinic angle for 1974 18R")
        out["numeric_checks"]["reported_cell_lengths_angstrom"]=o["product_cell_A"]
        out["outcome"]="PARTIAL_METRIC__ANGLE_NOT_PUBLISHED"
    elif p["id"]=="ritter_yang_pope_laird_1979":
        out["numeric_checks"]["source_2H_cell"]=_metric_evidence(
            o["product_cell_A"],"orthorhombic")
        out["paper_numeric_input_reproduced"]=True
        out["outcome"]="SOURCE_CELL_METRIC_PASS__NO_PAIRED_PARENT_C"
    elif p["id"]=="landa_sedlak_2007":
        out["numeric_checks"]["registered_2H_cell"]=_metric_evidence(
            o["product_cell_A"],"orthorhombic")
        out["paper_numeric_input_reproduced"]=True
        out["outcome"]="REPOSITORY_REGISTERED_CELL_METRIC_PASS__ORIGINAL_TABLE_NOT_RECHECKED"
    elif p["id"]=="zhang_xu_zhao_2023_microstructure":
        beta=float(o["2H_beta_deg_reported"])
        if abs(beta-90)<.01:
            raise AssertionError("Measured beta near 90 must not be rounded to 90")
        out["numeric_checks"]["measured_beta_minus_orthorhombic_deg"]=beta-90.
        out["numeric_checks"]["source_2H_cell_lengths_angstrom"]=o["2H_product_cell_A"]
        out["outcome"]="PARTIAL_MONOCLINIC_DISTORTION__OTHER_ANGLES_NOT_SOURCE_LOCKED"
    elif p["id"] in {
        "perez_cerrato_2022_lpbf","thin_film_2023_martensite"
    }:
        out["numeric_checks"]["published_borrowed_M18R_cell"]=_metric_evidence(
            o["M18R_reference_cell_A"],"monoclinic_unique_b",
            float(o["M18R_beta_deg"]))
        if "2H_reference_cell_A" in o:
            out["numeric_checks"]["borrowed_2H_cell"]=_metric_evidence(
                o["2H_reference_cell_A"],"orthorhombic")
        out["paper_numeric_input_reproduced"]=True
        out["outcome"]="BORROWED_REFERENCE_CELL_METRIC_PASS__NOT_INDEPENDENT_SPECIMEN"
    elif p["id"]=="ritchie_et_al_2016_avalanche":
        T=o["cooling_transition_K_approx"]
        if not T<o["heating_18R_to_beta_K_approx"]<o["heating_2H_to_beta_K_approx"]:
            raise AssertionError("Published two-stage transitions invalid")
        out["numeric_checks"]["reported_temperature_order_K"]=(
            T,o["heating_18R_to_beta_K_approx"],o["heating_2H_to_beta_K_approx"]
        )
        out["outcome"]="THERMAL_EVIDENCE_INTACT__OUTSIDE_LATTICE_SOLVER_DOMAIN"
    elif p["id"]=="chen_gui_wang_2000":
        if o["AC_BD_twin_plane_family"]==o["AD_BC_twin_plane_family"]:
            raise AssertionError("EBSD reports two different 2H twin-plane families")
        out["numeric_checks"]["source_twin_families"]={
            "AC_BD":o["AC_BD_twin_plane_family"],
            "AD_BC":o["AD_BC_twin_plane_family"],
        }
        out["outcome"]="SOURCE_EBSD_RELATIONS_LOCKED__3D_TWIN_PAIR_NOT_IDENTIFIED"
    elif p["id"]=="no_caillard_sanjuan_2009":
        if o["observed_habit_or_count"]!=9:
            raise AssertionError("Incorrect stated TEM habit observation count")
        out["outcome"]="NINE_HABIT_OR_OBSERVATIONS_LOCKED__3D_VALUES_UNAVAILABLE"
    elif p["id"]=="ge_vronka_2022":
        if set(o["twin_categories"])!={"Type I","Type II","compound"}:
            raise AssertionError("Missing 2H three-mode twin observation")
        out["outcome"]="THREE_TWIN_MODE_OBSERVATIONS_LOCKED__NO_DIRECT_3D_NUMERIC"
    else:
        out["outcome"]="SOURCE_EVIDENCE_SCOPE_ONLY__NO_FULL_CRYSTALLOGRAPHY"
    if (out["full_ct_amm_eq42_experimentally_comparable"] or
        out["full_bj_cc123_experimentally_comparable"]):
        raise AssertionError("Do not claim experimental full SC without matched C and 3D twin")
    return out


def run_campaign()->dict:
    registry=campaign_registry()
    results=[evaluate_one(p) for p in registry["papers"]]
    quantitative=sum(x["paper_numeric_input_reproduced"] for x in results)
    return {
        "schema":"cualni-18-paper-campaign-execution-v1",
        "papers_evaluated":len(results),
        "unique_DOIs":len({r["doi"].lower() for r in results}),
        "paper_input_or_lattice_numerical_passes":quantitative,
        "source_gate_or_incomplete_records":len(results)-quantitative,
        "full_Cayron_AMM_or_BJ_experimental_predictions_validated":0,
        "source_records":results,
        "warning":(
            "Passing a source gate is NOT a Cu-Al-Ni Eq42 or cofactor "
            "validation. Only the Hane–Shield independent stretch and "
            "variant calculation tests a numerical predicted quantity "
            "against the paper. Complete specimen-paired 3D data and "
            "a source-verified correspondence are absent in the rest."
        ),
    }


if __name__=="__main__":
    print(json.dumps(run_campaign(),indent=2,default=str))
