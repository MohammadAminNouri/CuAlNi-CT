"""Adversarial, independent hypotheses for Cu-Al-Ni 2H and M18R.

No published shear, lambda or habit-plane answers are used. The measured
lattice references have NO verified specimen-specific C or OR.
"""
import numpy as np
import pytest

from twin_app.p1_cualni import (
    CELLS,gram,check_cualni_specimen,CrystalInputError,
    reduced_gamma_to_unique_b,GAMMA_TO_UNIQUE_B)
from twin_app.p1_phase_discrimination import (
    discriminate_cualni_martensite,PhaseDiscriminationError)
from twin_app.p1_numerics import stretch_from_metrics_independent


def specimen(phase):
    key={"gamma1_prime_2H":"CuAlNi_2H_diffraction_reported",
         "beta_prime_monoclinic":"CuAlNi_monoclinic_C2m_EBSD_reported",
         "faulted_monoclinic_reduced_6M":"CuAlNi_6M_James_Hane_2000"}[phase]
    cell=CELLS[key]
    if phase=="faulted_monoclinic_reduced_6M":
        converted=reduced_gamma_to_unique_b(cell.lattice)
        return {"phase_label":phase,"daughter_cell":converted["new_cell"],
                "daughter_point_group":"2/m","cell_setting":"monoclinic unique-b rebased",
                "parent_cell":(3.0,3.0,3.0,90.,90.,90.),"parent_point_group":"m-3m"}
    return {"phase_label":phase,"daughter_cell":cell.lattice,
            "daughter_point_group":cell.point_group,"cell_setting":cell.cell_setting,
            "parent_cell":(3.0,3.0,3.0,90.,90.,90.),"parent_point_group":"m-3m"}


@pytest.mark.parametrize("phase",["gamma1_prime_2H","beta_prime_monoclinic",
                                  "faulted_monoclinic_reduced_6M"])
def test_no_stacking_evidence_means_not_identified(phase):
    out=check_cualni_specimen(specimen(phase))
    assert out["phase_identity_audit"]["status"]=="PHASE_UNRESOLVED_NO_INDEXED_STACKING"
    assert out["accepted_as_physical_prediction"] is False


@pytest.mark.parametrize("phase,wrong",[
    ("gamma1_prime_2H","18R"),
    ("beta_prime_monoclinic","2H"),
    ("faulted_monoclinic_reduced_6M","2H"),
])
def test_opposing_diffraction_rejects_declared_phase(phase,wrong):
    spec=specimen(phase)
    spec["stacking_evidence"]={"indexed_sequences":[wrong],
                               "indexing_method":"TEM_SAED",
                               "observation_reference":"adversarial_mock"}
    with pytest.raises(CrystalInputError,match="STACKING_PHASE_CONTRADICTION"):
        check_cualni_specimen(spec)


@pytest.mark.parametrize("phase",["gamma1_prime_2H","beta_prime_monoclinic"])
def test_two_phase_domains_cannot_be_one_homogeneous_solver_case(phase):
    spec=specimen(phase)
    spec["stacking_evidence"]={"indexed_sequences":["2H","18R"],
                               "indexing_method":"TEM_SAED",
                               "observation_reference":"adversarial_mock"}
    with pytest.raises(CrystalInputError,match="MIXED_MARTENSITE_DOMAINS"):
        check_cualni_specimen(spec)


def test_6r_stress_induced_phase_is_not_reduced_6m_representation():
    spec=specimen("faulted_monoclinic_reduced_6M")
    spec["stacking_evidence"]={"indexed_sequences":["6R"],
                               "indexing_method":"HRTEM_FFT",
                               "observation_reference":"adversarial_mock"}
    with pytest.raises(CrystalInputError,match="STRESS_INDUCED_6R"):
        check_cualni_specimen(spec)


@pytest.mark.parametrize("phase,sequence",[
    ("gamma1_prime_2H","2H"),("beta_prime_monoclinic","18R"),
    ("faulted_monoclinic_reduced_6M","18R"),
])
def test_matching_indexed_phase_is_still_not_independently_validated(phase,sequence):
    spec=specimen(phase)
    spec["stacking_evidence"]={"indexed_sequences":[sequence],
                               "indexing_method":"TEM_SAED",
                               "observation_reference":"unverified_user_note"}
    spec["correspondence_M_from_A"]=[["1","0","0"],["0","1","0"],["0","0","1"]]
    spec["correspondence_source"]="unverified_user_note"
    out=check_cualni_specimen(spec)
    assert out["phase_identity_audit"]["status"]=="REPORTED_STACKING_CONSISTENT_SOURCE_NOT_INDEPENDENTLY_VERIFIED"
    assert not out["phase_identity_audit"]["can_claim_phase_identified"]
    assert not out["accepted_as_physical_prediction"]


def test_ebsd_pattern_assignment_alone_not_sufficient_to_resolve_pseudosymmetry():
    out=discriminate_cualni_martensite(
        "beta_prime_monoclinic",
        {"indexed_sequences":["18R"],"indexing_method":"EBSD_PATTERN_INDEXING",
         "observation_reference":"mock_ebsd_index"})
    assert out["status"]=="CONSISTENT_EBSD_INDEXING_PSEUDOSYMMETRY_NOT_RESOLVED"
    assert out["independent_experimental_verification"] is False


def test_nearly_right_monoclinic_angle_does_not_transform_stacking_identity():
    ob=specimen("gamma1_prime_2H")
    mo=specimen("beta_prime_monoclinic")
    mo["daughter_cell"]=(4.42,5.27,37.92,90.,90.000001,90.)
    assert check_cualni_specimen(ob)["phase_identity_audit"]["status"].startswith("PHASE_UNRESOLVED")
    assert check_cualni_specimen(mo)["phase_identity_audit"]["status"].startswith("PHASE_UNRESOLVED")


def test_reduced_6m_gamma_cell_rebase_preserves_metric_and_volume():
    cell=CELLS["CuAlNi_6M_James_Hane_2000"].lattice
    res=reduced_gamma_to_unique_b(cell)
    G=gram(cell);Gb=gram(res["new_cell"]);P=np.asarray(GAMMA_TO_UNIQUE_B,float)
    assert np.linalg.det(P)==pytest.approx(1.)
    assert np.allclose(Gb,P.T@G@P,rtol=1e-12,atol=1e-10)
    assert res["volume_ratio"]==pytest.approx(1.,abs=1e-12)


def test_phase_metrics_produce_distinct_mathematical_strains_for_same_hypothetical_C():
    A=gram((3.,3.,3.,90.,90.,90.))
    C=np.eye(3)  # purely algebraic negative control, NOT a published correspondence
    U2=stretch_from_metrics_independent(A,gram(specimen("gamma1_prime_2H")["daughter_cell"]),C)
    U18=stretch_from_metrics_independent(A,gram(specimen("beta_prime_monoclinic")["daughter_cell"]),C)
    assert np.linalg.norm(U2-U18)>0.01
    assert np.isfinite(U2).all() and np.isfinite(U18).all()
    assert np.linalg.eigvalsh(U2)[0]>0 and np.linalg.eigvalsh(U18)[0]>0


def test_relabeling_without_structure_evidence_never_claims_real_phase():
    for phase in ("gamma1_prime_2H","beta_prime_monoclinic"):
        out=discriminate_cualni_martensite(phase,None)
        assert not out["can_claim_phase_identified"]
        assert not out["independent_experimental_verification"]

