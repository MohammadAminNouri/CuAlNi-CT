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


def _hane_shield_full_twin_ptmc_benchmark(stretches: tuple[float,float,float], source:dict) -> dict:
    """Full *paper-specific* nonlinear-elasticity benchmark, without guessed C.

    Hane–Shield 1999's 2H problem is a six-stretch U_i Ball–James/PTMC
    problem, not a Cayron CT single-variant/shear–shear Eq42 benchmark.
    Enumerate exact 180° parent rotations and the symmetrically related
    stretch pairs. Compare independent source shears, two PTMC fractions,
    and count all twinned A/M habit solutions.
    """
    from cualni_cryst.ball_james import mallard_law_twins
    from cualni_cryst.ptmc import (
        ptmc_volume_fractions, middle_singular_value,
    )
    from cualni_cryst.ball_james import analytical_rank_one_connections

    alpha,beta,gamma=stretches
    axes=[]
    for vec in ((1,0,0),(0,1,0),(0,0,1),
                (1,1,0),(1,-1,0),(1,0,1),(1,0,-1),(0,1,1),(0,1,-1)):
        v=np.asarray(vec,float)
        axes.append(v/np.linalg.norm(v))
    Us=[]
    for beta_axis in range(3):
        for sign in (-1,1):
            U=np.eye(3)*((alpha+gamma)/2.)
            U[beta_axis,beta_axis]=beta
            i,j=(v for v in range(3) if v!=beta_axis)
            U[i,j]=U[j,i]=sign*(alpha-gamma)/2.
            Us.append(U)
    pair_records=[]
    for i in range(6):
        for j in range(i+1,6):
            generators=[]
            for axis_index,e in enumerate(axes):
                Q=2*np.outer(e,e)-np.eye(3)
                if np.linalg.norm(Us[j]-Q@Us[i]@Q,ord="fro")<1e-10:
                    generators.append((axis_index,e))
            if len(generators) not in (1,2):
                raise AssertionError(
                    "Hane–Shield 2H twin pair is not reproduced by one or "
                    "two cubic 180-degree isometries"
                )
            # Duplicate physical twinning constructions through different
            # parent twofolds for a compound family: source distinguishes 3
            # unique unordered compound pairs from 12 noncompound pairs.
            representative=generators[0][1]
            twins=mallard_law_twins(Us[i],Us[j],representative)
            if len(twins)!=2 or any(t.residual>5e-9 for t in twins):
                raise AssertionError("Analytical BJ Mallard-law rank-one twin failed")
            expected_shear=(source["compound_shear"]
                            if len(generators)==2 else
                            source["type_I_II_shear"])
            for twin in twins:
                if abs(twin.shear_magnitude-expected_shear)>.0006:
                    raise AssertionError(
                        "Predicted shear magnitude disagrees with published "
                        f"Hane–Shield Table 1: {twin.shear_magnitude} vs {expected_shear}"
                    )
            pair_records.append({
                "U_pair":(i+1,j+1),
                "parent_180_degree_generators":len(generators),
                "twin_solutions":tuple(
                    {"kind":t.kind,"shear":float(t.shear_magnitude),
                     "rank_one_relative_residual":float(t.residual)}
                    for t in twins
                ),
            })
    compound=[x for x in pair_records if x["parent_180_degree_generators"]==2]
    noncompound=[x for x in pair_records if x["parent_180_degree_generators"]==1]
    if len(compound)!=3 or len(noncompound)!=12:
        raise AssertionError("Cubic-orthorhombic U_i pair split differs from source")
    # Select a deterministic representative of the S2 family. The
    # published volume fractions refer to this symmetry-related class,
    # not an experimentally measured volume fraction in another specimen.
    i,j=0,2
    axis=next(e for e in axes if np.linalg.norm(
        Us[j]-(2*np.outer(e,e)-np.eye(3))@Us[i]@
        (2*np.outer(e,e)-np.eye(3)))<1e-10)
    twin_modes=mallard_law_twins(Us[i],Us[j],axis)
    fractions={}
    for t in twin_modes:
        roots=ptmc_volume_fractions(Us[j],t.a,t.n,tol=1.e-8)
        if len(roots)!=2:
            raise AssertionError("PTMC Hane–Shield type-I/II roots incomplete")
        smaller=min(roots)
        wanted=.290 if t.kind=="I" else .301
        if abs(smaller-wanted)>.002:
            raise AssertionError(
                f"Hane–Shield Table 1 PTMC volume fraction {t.kind} "
                f"{smaller} disagrees with {wanted}"
            )
        for fraction in roots:
            Fbar=Us[j]+fraction*np.outer(t.a,t.n)
            branches=analytical_rank_one_connections(
                Fbar,np.eye(3),tol=1e-8)
            if len(branches)!=2 or any(b.residual>1e-7 for b in branches):
                raise AssertionError(
                    "Hane–Shield PTMC habit equation did not yield two "
                    "independent Ball–James rank-one solutions"
                )
            if abs(middle_singular_value(Fbar)-1.)>1.e-8:
                raise AssertionError("PTMC candidate is not exactly A/M compatible")
        fractions[t.kind]=[float(x) for x in roots]

    # Independently certify EVERY S2 pair and laminate root using an
    # orthogonal direct-SVD bracketing procedure. This is separate from
    # the exact binary64 polynomial code used for the representative source
    # fraction above. Each admissible average has two rank-one habits.
    from scipy.optimize import brentq
    per_pair=[]
    all_habits=0
    for pair in noncompound:
        i,j=(k-1 for k in pair["U_pair"])
        twin_axis=next(
            e for e in axes
            if np.linalg.norm(
                Us[j]-(2*np.outer(e,e)-np.eye(3))@
                Us[i]@(2*np.outer(e,e)-np.eye(3))
            )<1e-10
        )
        for mode in mallard_law_twins(Us[i],Us[j],twin_axis):
            def g(frac):
                return middle_singular_value(
                    Us[j]+frac*np.outer(mode.a,mode.n))-1.
            grid=np.linspace(0.,1.,81)
            roots=[]
            for x,y in zip(grid[:-1],grid[1:]):
                gx,gy=g(float(x)),g(float(y))
                if gx*gy<0:
                    z=float(brentq(g,float(x),float(y),xtol=2.e-14))
                    if not any(abs(z-v)<1.e-8 for v in roots):
                        roots.append(z)
            roots.sort()
            if len(roots)!=2:
                raise AssertionError(
                    f"Missing PTMC roots for U pair {i+1}/{j+1} Type {mode.kind}"
                )
            for fraction in roots:
                F=Us[j]+fraction*np.outer(mode.a,mode.n)
                hab=analytical_rank_one_connections(F,np.eye(3),tol=1.e-8)
                if len(hab)!=2 or any(z.residual>1.e-7 for z in hab):
                    raise AssertionError(
                        "Rank-one habit solutions disappeared in full 2H census"
                    )
                all_habits+=len(hab)
            per_pair.append({
                "pair":(i+1,j+1),"twin_type":mode.kind,
                "PTMC_fractions":roots,"exact_habit_branches":4,
            })
    if all_habits!=source["twinned_habit_solution_count"]:
        raise AssertionError("Hane–Shield 1999 full 96-habit prediction not reproduced")
    # S1 compound pairs have no twinned A/M habit according to source.
    comp=mallard_law_twins(Us[0],Us[1],axes[1])
    compound_roots={
        t.kind:ptmc_volume_fractions(Us[1],t.a,t.n,tol=1.e-8)
        for t in comp
    }
    if any(compound_roots.values()):
        raise AssertionError("Source's compound S1 twins incorrectly generated A/M habit")
    return {
        "status":"NUMERIC_BALL_JAMES_TWIN_PTMC_SOURCE_VALUES_REPRODUCED",
        "ordered_stretch_pair_count":30,
        "unordered_stretch_pair_count":15,
        "compound_unordered_pairs":len(compound),
        "type_I_II_unordered_pairs":len(noncompound),
        "pair_results":pair_records,
        "computed_twinned_habit_solution_records":all_habits,
        "all_type_I_II_twin_pair_ptmc_results":per_pair,
        "source_predicted_habit_solution_records":source["twinned_habit_solution_count"],
        "representative_PTMC_type_I_II_fractions":fractions,
        "compound_family_PTMC_roots":compound_roots,
        "scope":(
            "BJ Mallard twin shears, PTMC volume fractions, and the "
            "twinned habit class census agree with Hane–Shield (1999). "
            "This does NOT validate one CT correspondence C or Cayron Eq.42 "
            "for an experimental single crystal."
        ),
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
        hane=_hane_shield_1999(p)
        out["numeric_checks"]["Hane_Shield_1999"]=hane
        out["numeric_checks"]["independent_BJ_PTMC"]=_hane_shield_full_twin_ptmc_benchmark(
            tuple(hane["Hane_Shield_independent_stretches"]),o,
        )
        out["numeric_checks"]["2H_cell"]=_metric_evidence(
            o["product_cell_A"],"orthorhombic")
        out["paper_numeric_input_reproduced"]=True
        out["outcome"]="HANE_SHIELD_1999_BALL_JAMES_MALLARD_PTMC_NUMERICALLY_REPRODUCED"
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
            "validation. Hane–Shield's independent six-U stretch, twin shear, "
            "PTMC volume fraction and habit-census numerical predictions are "
            "reproduced through BJ/PTMC; other papers constrain only their "
            "actual sourced lattice/phase observations. Complete specimen-paired "
            "3D data and source-verified C are still absent."
        ),
    }


if __name__=="__main__":
    print(json.dumps(run_campaign(),indent=2,default=str))
