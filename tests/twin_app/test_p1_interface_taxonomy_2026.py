"""Adversarial twin-versus-habit routing and experimental evidence provenance."""
from types import SimpleNamespace as N
import numpy as np
import pytest
from twin_app.p1_interface_validation import (
    compare_specimen_interfaces, InterfaceAuditError
)

G = np.diag([4.,9.,16.])
PAIR = "test-pair-A1-D1"

def report():
    # Independent fixture deliberately separates 2H twin and A/M habit normals.
    h=N(habit_normal_parent_cartesian=(0.,0.,1.))
    c=N(branch=1,twin_plane_product_crystal=(1.,2.,1.),
        habit_solutions=(h,))
    p=N(pair_id=PAIR,constructions=(c,))
    return N(families=(N(pair_records=(p,)),))

def specimen(obs):
    return {"specimen_id": "mock_specimen_2H",
            "observed_interface_planes": [obs]}

def twin(**edits):
    v={"specimen_id":"mock_specimen_2H",
       "interface_type":"martensite_martensite_twin",
       "observation_reference":"mock_TEM_SAED_A_C",
       "source_pair_id":PAIR,"source_branch":1,
       "indexed_variant":"base",
       "frame":"product_reciprocal_miller",
       "coordinates":[1.,2.,1.],
       "uncertainty_deg":1.}
    v.update(edits)
    return v

def habit(**edits):
    v={"specimen_id":"mock_specimen_2H",
       "interface_type":"austenite_martensite_habit",
       "observation_reference":"mock_TEM_A_M",
       "source_pair_id":PAIR,"source_branch":1,
       "frame":"parent_reciprocal_miller",
       "coordinates":[0.,0.,1.],
       "uncertainty_deg":1.}
    v.update(edits)
    return v

def audit(observation):
    return compare_specimen_interfaces(report(),specimen(observation),np.eye(3),G)

def test_twin_compared_to_twin_not_the_unrelated_habit():
    r=audit(twin())
    x=r["observations"][0]
    assert x["interface_type"]=="martensite_martensite_twin"
    assert x["geometric_comparison"]["angle_deg"]<1e-9
    assert not x["independently_validated"]

def test_austenite_martensite_habit_compared_only_to_habit():
    x=audit(habit())["observations"][0]
    assert x["geometric_comparison"]["angle_deg"]<1e-9
    assert x["interface_type"]=="austenite_martensite_habit"

def test_wrong_interface_target_does_not_pass_accidentally():
    # A twin normal (121)_M supplied as if it were an A/M habit is NOT equal.
    x=audit(habit(coordinates=[1.,2.,1.],
                  frame="parent_reciprocal_miller"))["observations"][0]
    assert x["geometric_comparison"]["status"]=="NO_GEOMETRIC_MATCH_AT_DECLARED_UNCERTAINTY"

@pytest.mark.parametrize("kind",["unknown", "twin", "habit", None, ""])
def test_untyped_or_imprecisely_typed_observation_refused(kind):
    with pytest.raises(InterfaceAuditError,match="AMBIGUOUS_INTERFACE"):
        audit(twin(interface_type=kind))

@pytest.mark.parametrize("bad",["other_specimen", "", None])
def test_specimen_identity_mismatch_rejected(bad):
    with pytest.raises(InterfaceAuditError,match="SPECIMEN_MISMATCH"):
        audit(twin(specimen_id=bad))

@pytest.mark.parametrize("bad",["",None])
def test_missing_reference_rejected(bad):
    with pytest.raises(InterfaceAuditError,match="observation_reference"):
        audit(twin(observation_reference=bad))

@pytest.mark.parametrize("case",[
    {"source_pair_id":"fabricated_source"},
    {"source_branch":99},
    {"source_branch":"1"},
    {"source_branch":True},
    {"indexed_variant":"other"},
    {"indexed_variant":None},
    {"frame":"parent_reciprocal_miller"},
])
def test_source_and_variant_gauge_mismatch_refused(case):
    with pytest.raises(InterfaceAuditError):
        audit(twin(**case))

@pytest.mark.parametrize("bad",[0,-1,46,None,float("nan")])
def test_missing_or_unphysical_uncertainties_refused(bad):
    with pytest.raises(InterfaceAuditError,match="uncertainty"):
        audit(twin(uncertainty_deg=bad))

def test_product_indexed_habit_requires_variant_specific_OR():
    with pytest.raises(InterfaceAuditError,match="variant-specific OR"):
        audit(habit(frame="product_reciprocal_miller",coordinates=[0.,0.,1.]))

def test_wrong_handedness_measured_OR_fails_closed():
    with pytest.raises(InterfaceAuditError,match="proper Cartesian rotation"):
        audit(habit(frame="product_reciprocal_miller",coordinates=[0.,0.,1.],
                    measured_OR_M_cart_to_A_cart=np.diag([-1.,1.,1.]).tolist(),
                    OR_observation_reference="mock_OR"))

def test_explicit_valid_or_geometrically_compares_but_not_certifies():
    x=audit(habit(frame="product_reciprocal_miller",
                  coordinates=[0.,0.,1.],
                  measured_OR_M_cart_to_A_cart=np.eye(3).tolist(),
                  OR_observation_reference="mock_OR"))["observations"][0]
    assert x["geometric_comparison"]["angle_deg"]<1e-9
    assert x["independently_validated"] is False

def test_two_independent_observations_never_cross_compare():
    r=compare_specimen_interfaces(report(),
        {"specimen_id":"mock_specimen_2H",
         "observed_interface_planes":[twin(),habit()]},
        np.eye(3),G)
    assert len(r["observations"])==2
    assert all(x["geometric_comparison"]["angle_deg"]<1e-9
               for x in r["observations"])

def test_no_observations_cannot_infer_experimental_confirmation():
    r=compare_specimen_interfaces(report(),
        {"specimen_id":"mock_specimen_2H"},np.eye(3),G)
    assert r["status"]=="NO_EXPERIMENTAL_INTERFACE_COMPARISON"
    assert not r["independently_validated"]

def test_no_habit_solutions_never_promotes_twin_as_habit():
    r=report()
    r.families[0].pair_records[0].constructions[0].habit_solutions=()
    x=compare_specimen_interfaces(
        r,specimen(habit()),np.eye(3),G)["observations"][0]
    assert x["geometric_comparison"]["status"]=="NO_CALCULATED_HABIT_FOR_SELECTED_BRANCH"


def test_null_specimen_ids_do_not_become_the_fake_string_none():
    s={"specimen_id":None,"observed_interface_planes":[twin(specimen_id=None)]}
    with pytest.raises(InterfaceAuditError,match="SPECIMEN_MISMATCH"):
        compare_specimen_interfaces(report(),s,np.eye(3),G)

@pytest.mark.parametrize("invalid_ref",[None,"",42,{}])
def test_missing_or_false_or_reference_prevents_product_habit_comparison(invalid_ref):
    with pytest.raises(InterfaceAuditError,match="variant-specific OR"):
        audit(habit(frame="product_reciprocal_miller",
                    coordinates=[0.,0.,1.],
                    measured_OR_M_cart_to_A_cart=np.eye(3).tolist(),
                    OR_observation_reference=invalid_ref))
