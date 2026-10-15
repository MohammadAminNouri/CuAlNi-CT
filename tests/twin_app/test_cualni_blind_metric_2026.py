"""Blind numerical metamorphisms. Never initialize the solver with expected outputs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from twin_app.p1_cualni import CELLS,gram,metric_to_cell,reduced_gamma_to_unique_b,CrystalInputError
from twin_app.p1_numerics import stretch_from_metrics_independent,spectrum_diagnostics
from twin_app.p1_james_hane_6m import (
    six_m_compatibility_james_hane,m18r_to_6m_james_hane,LiteratureMetricError)

FIXTURE=json.loads((Path(__file__).resolve().parents[2] /
                     "data/benchmarks/cualni_james_hane_table4_input_only.json").read_text())
INPUTS=FIXTURE["cases"]


@pytest.mark.parametrize("case", INPUTS, ids=lambda x:x["id"])
def test_source_equation_from_lattice_inputs_not_published_answers(case):
    result=six_m_compatibility_james_hane(
        parent_do3_a0=case["a0"],a_6m=case["a"],c_6m=case["c"],
        theta_6m_deg=case["theta_measured"])
    assert result["source_correspondence_not_loaded"]
    assert not result["twinning_or_habit_solutions_computed"]
    assert not result["exactly_compatible_with_reported_rounded_input"]
    assert result["admissible_compatibility_angle"]
    assert 0 < result["nearest_branch_compatible_theta_deg"] < 180
    # HELD-OUT number from the publication, never read by any solver.
    # Rounding of the reported lattice data limits agreement.
    assert abs(result["nearest_branch_compatible_theta_deg"] -
               case["published_compatible_angle_approx"]) < 0.25


def test_6m_measured_compatibility_not_falsely_exact():
    c=next(x for x in INPUTS if x["id"]=="CuAlNi_DO3_6M")
    result=six_m_compatibility_james_hane(
        parent_do3_a0=c["a0"],a_6m=c["a"],c_6m=c["c"],
        theta_6m_deg=c["theta_measured"])
    assert abs(result["cos2_mismatch"]) > 1e-6
    assert abs(result["observed_minus_compatible_angle_deg"]) > 0.1
    assert result["experimental_verification"]=="NOT_ESTABLISHED"


@pytest.mark.parametrize("a,c,theta",[(4.4,37.5,89.7),(4.5,38.0,90.),
                                      (4.1,40.1,90.2)])
def test_m18r_6m_equation_24_layer_height_identity(a,c,theta):
    out=m18r_to_6m_james_hane(a=a,c_m18r=c,theta_m18r_deg=theta)
    assert out["theta_6m_deg"] > theta
    assert out["c_6m"] < c
    assert abs(out["invariant_layer_height_ratio"]-1) < 1e-12
    assert out["requires_transformed_correspondence"]
    assert out["experimental_phase_identity_verified"] is False


@pytest.mark.parametrize("phase",["CuAlNi_2H_diffraction_reported",
                                    "CuAlNi_monoclinic_C2m_EBSD_reported",
                                    "CuAlNi_6M_James_Hane_2000"])
def test_both_martensite_reference_metrics_positive_and_reconstructible(phase):
    cell=CELLS[phase].lattice
    G=gram(cell); recovered=metric_to_cell(G)
    assert np.allclose(recovered,cell,atol=3e-10,rtol=1e-12)
    assert np.linalg.eigvalsh(G)[0]>0


def test_physically_different_specimen_metrics_not_automatically_merged():
    # Source-specific lattice cases are not interchangeable, even if the alloy
    # name and index have similar crystallographic notation.
    values=[gram(v.lattice) for v in CELLS.values()]
    assert all(np.linalg.norm(values[i]-values[j])>0.01
               for i in range(len(values)) for j in range(i+1,len(values)))


@pytest.mark.parametrize("case",INPUTS,ids=lambda x:x["id"])
def test_DO3_composition_and_parent_definition_not_interchangeable_with_B2(case):
    original=six_m_compatibility_james_hane(
       parent_do3_a0=case["a0"],a_6m=case["a"],c_6m=case["c"],
       theta_6m_deg=case["theta_measured"])
    half_parent=six_m_compatibility_james_hane(
       parent_do3_a0=case["a0"]/2,a_6m=case["a"],c_6m=case["c"],
       theta_6m_deg=case["theta_measured"])
    assert abs(original["required_cos2_theta"] -
               half_parent["required_cos2_theta"]) > 0.1


def test_reduced_cell_rebasis_round_trip_monoclinic_gamma_and_volume():
    rng=np.random.default_rng(150115)
    for _ in range(40):
        cell=(rng.uniform(3.,8.),rng.uniform(3.,8.),rng.uniform(8.,43.),
              90.,90.,rng.uniform(90.1,110.))
        rebased=reduced_gamma_to_unique_b(cell)
        G=gram(cell)
        P=np.asarray(rebased["basis_new_to_old"],float)
        assert np.linalg.det(P)==1
        assert np.allclose(gram(rebased["new_cell"]),P.T@G@P,rtol=5e-14,atol=1e-10)
        assert abs(rebased["volume_ratio"]-1)<1e-10


def test_random_nontrivial_correspondence_rebasis_preserves_U_not_guessed_outputs():
    rng=np.random.default_rng(181026)
    for k in range(30):
        is_monoclinic=k%2==1
        parent=(rng.uniform(5.6,6.4),)*3+(90.,90.,90.)
        daughter=(rng.uniform(3.6,5.5),rng.uniform(3.6,5.5),
                  rng.uniform(4.,14.),90.,
                  rng.uniform(90.2,110.) if is_monoclinic else 90.,90.)
        Ga=gram(parent);Gm=gram(daughter)
        # A rational non-identity C supplied as a mathematical HYPOTHESIS.
        C=np.array([[1.,0.,.5],[0.,1.,0.],[0.,0.,1.]])
        # Unimodular basis reindexing, independent of phase/model.
        P=np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
        P[0,1]=int(rng.integers(-1,2))
        right=np.linalg.solve(P,C)
        U=stretch_from_metrics_independent(Ga,Gm,C)
        Urebase=stretch_from_metrics_independent(Ga,P.T@Gm@P,right)
        assert np.allclose(U,Urebase,rtol=2e-12,atol=2e-11)
        assert spectrum_diagnostics(U)["detU"]>0


@pytest.mark.parametrize("bad",[-1,0,float("nan"),float("inf")])
def test_negative_or_nonfinite_lattice_lengths_rejected(bad):
    with pytest.raises((LiteratureMetricError,CrystalInputError)):
        six_m_compatibility_james_hane(
            parent_do3_a0=5.836,a_6m=bad,c_6m=12.79,
            theta_6m_deg=95.68)


def test_close_to_orthorhombic_is_still_monoclinic_in_metric():
    G=gram((4.430,5.330,12.79,90.,90.00001,90.))
    assert np.linalg.norm(G-np.diag(np.diag(G)))>0
    assert np.linalg.eigvalsh(G)[0]>0


def test_james_hane_eq10_source_derived_cube_edge_tensor_matches_eq25():
    from twin_app.p1_james_hane_6m import cube_edge_stretch_james_hane
    c=next(x for x in INPUTS if x["id"]=="CuAlNi_DO3_6M")
    compat=six_m_compatibility_james_hane(
        parent_do3_a0=c["a0"],a_6m=c["a"],c_6m=c["c"],
        theta_6m_deg=c["theta_measured"])
    measured=cube_edge_stretch_james_hane(
        parent_do3_a0=c["a0"],a_6m=c["a"],b_6m=c["b"],
        c_6m=c["c"],theta_6m_deg=c["theta_measured"])
    theoretical=cube_edge_stretch_james_hane(
        parent_do3_a0=c["a0"],a_6m=c["a"],b_6m=c["b"],
        c_6m=c["c"],theta_6m_deg=compat["nearest_branch_compatible_theta_deg"])
    assert abs(np.linalg.eigvalsh(theoretical)[1]-1) < 3e-12
    # Do not silently replace measured inputs with exact-compatible synthetic ones.
    assert abs(np.linalg.eigvalsh(measured)[1]-1) > 1e-4
    assert np.linalg.eigvalsh(measured)[0]>0


@pytest.mark.parametrize("case",INPUTS,ids=lambda x:x["id"])
def test_cube_edge_volume_constrained_by_source_lattice_not_lookup(case):
    from twin_app.p1_james_hane_6m import cube_edge_stretch_james_hane
    import math
    U=cube_edge_stretch_james_hane(
        parent_do3_a0=case["a0"],a_6m=case["a"],b_6m=case["b"],
        c_6m=case["c"],theta_6m_deg=case["theta_measured"])
    alpha=math.sqrt(2)*case["a"]/case["a0"]
    beta=case["b"]/case["a0"]
    gamma=math.sqrt(2)*case["c"]/(3*case["a0"])
    expected_volume=alpha*beta*gamma*math.sin(math.radians(case["theta_measured"]))
    assert abs(np.linalg.det(U)-expected_volume)<2e-12


def test_cube_edge_twelve_variants_generated_by_parent_group_not_preloaded():
    from twin_app.p1_james_hane_6m import cube_edge_stretch_james_hane,proper_cubic_stretch_orbit
    c=next(x for x in INPUTS if x["id"]=="CuAlNi_DO3_6M")
    U=cube_edge_stretch_james_hane(
        parent_do3_a0=c["a0"],a_6m=c["a"],b_6m=c["b"],
        c_6m=c["c"],theta_6m_deg=c["theta_measured"])
    variants=proper_cubic_stretch_orbit(U)
    # 12 is HELD OUT from James-Hane Eq (10), NOT inserted into orbit generator.
    assert len(variants)==12
    source_spectrum=np.linalg.eigvalsh(U)
    assert all(np.allclose(np.linalg.eigvalsh(X),source_spectrum,atol=2e-12)
               for X in variants)
    assert all(abs(np.linalg.det(X)-np.linalg.det(U))<2e-12
               for X in variants)


def test_non_general_6m_source_law_is_not_a_substitute_for_2h():
    from twin_app.p1_james_hane_6m import cube_edge_stretch_james_hane
    # Distinct reported 2H data are not a source-compatible 6M representation.
    sample=CELLS["CuAlNi_2H_diffraction_reported"]
    assert sample.phase=="gamma1_prime_2H"
    assert sample.point_group=="mmm"
    assert sample.cell_setting.lower().startswith("orthorhombic")
    # No physical transformation result is computed from this reference.
