"""Synthetic adversarial tests. No physically validated Cu-Al-Ni data implied."""
import copy
import numpy as np
import pytest
from twin_app.p1_blind_validation import (
    BlindEvidenceError, compare_blind_interfaces, prediction_digest,
)


def fixture():
    p = {"specimen_id": "SPEC-01", "phase": "gamma1_prime_2H",
         "solver_commit": "0"*40, "crystal_inputs_sha256": "a"*64,
         "predictions": [
             {"candidate_id":"t1","interface_kind":"twin","variant_pair":"A-C","normal_parent_cartesian":[1,0,0]},
             {"candidate_id":"t2","interface_kind":"twin","variant_pair":"A-C","normal_parent_cartesian":[0,1,0]},
             {"candidate_id":"h1","interface_kind":"habit","variant_pair":"A-beta","normal_parent_cartesian":[0,0,1]},
         ]}
    e = {"specimen_id": "SPEC-01", "phase": "gamma1_prime_2H",
         "source_doi": "10.5555/TEST-NOT-REAL",
         "source_location": "synthetic fixture only", "measurement_method": "SYNTHETIC",
         "composition_wt_percent": {"Cu": 81, "Al": 15, "Ni": 4},
         "measurement_temperature_K": 295,
         "lattice_setting": "orthorhombic", "sample_identity": "SPEC-01",
         "measured_from_prediction": False, "source_independently_reviewed": False,
         "observations": [
             {"observation_id":"o1","interface_kind":"twin","frame":"parent_cartesian",
              "coordinates":[1,0,0],"uncertainty_deg":1},
             {"observation_id":"o2","interface_kind":"habit","frame":"parent_cartesian",
              "coordinates":[0,0,1],"uncertainty_deg":1},
         ]}
    return p, e


def run(p, e, **kw):
    return compare_blind_interfaces(p, e, parent_metric=np.eye(3), null_trials=1024, **kw)


def test_exact_geometry_is_never_experimental_certification():
    p, e = fixture()
    e["source_independently_reviewed"] = True
    result = run(p, e)
    assert result["status"] == "GEOMETRIC_ASSESSMENT_ONLY_NOT_PHYSICALLY_CERTIFIED"
    assert result["source_independently_authenticated_by_software"] is False
    assert result["externally_timestamped_prediction_freeze_verified"] is False
    best = result["observations"][0]
    assert best["within_declared_uncertainty"]
    assert best["unique_match_under_declared_uncertainty"]
    assert best["nearest_candidate"] == "t1"
    assert best["second_nearest_candidate"] == "t2"
    assert len(best["all_candidate_residuals_deg"]) == 2


def test_candidate_multiplicity_null_model_does_not_hide_look_elsewhere():
    p, e = fixture()
    e["observations"] = e["observations"][:1]
    e["observations"][0]["coordinates"] = [1,1,0]
    fewer = copy.deepcopy(p)
    fewer["predictions"] = fewer["predictions"][:1]
    a = run(fewer, e)["observations"][0]
    b = run(p, e)["observations"][0]
    assert a["compared_candidates"] == 1 and b["compared_candidates"] == 2
    assert (b["isotropic_null_chance_at_observed_best_angle"]
            >= a["isotropic_null_chance_at_observed_best_angle"])


def test_phase_and_specimen_swap_are_refused():
    p, e = fixture()
    e["phase"] = "beta1_prime_M18R"
    with pytest.raises(BlindEvidenceError, match="mismatch"): run(p, e)
    e["phase"] = p["phase"]
    e["specimen_id"] = "unrelated"
    with pytest.raises(BlindEvidenceError, match="mismatch"): run(p, e)
    e["specimen_id"] = p["specimen_id"]
    e["sample_identity"] = "different specimen"
    with pytest.raises(BlindEvidenceError, match="Source specimen"): run(p, e)
    e["sample_identity"] = p["specimen_id"]
    p["phase"] = e["phase"] = "beta_prime_monoclinic"
    with pytest.raises(BlindEvidenceError, match="indexed"): run(p, e)


def test_explicit_m18r_phase_works_without_fabricating_geometry():
    p, e = fixture()
    p["phase"] = e["phase"] = "beta1_prime_M18R"
    e["lattice_setting"] = "measured unique-b monoclinic M18R"
    assert run(p, e)["phase"] == "beta1_prime_M18R"


def test_twin_and_habit_labels_cannot_cross():
    p, e = fixture()
    e["observations"][0]["interface_kind"] = "habit"
    a = run(p, e)
    assert a["observations"][0]["nearest_candidate"] == "h1"
    assert a["unmatched_observation_ids"] == ["o1"]


def test_surface_trace_is_2d_and_rigid_rotation_covariant():
    p, e = fixture()
    e["observations"] = [{"observation_id":"trace","interface_kind":"twin",
         "frame":"sample_surface_trace_cartesian",
         "coordinates":[0,1,0], "surface_normal_sample_cartesian":[0,0,1],
         "uncertainty_deg":0.25}]
    orig = run(p, e, parent_to_sample_rotation=np.eye(3))["observations"][0]
    assert orig["geometry_dimension"] == "2D_TRACE_ONLY"
    assert orig["nearest_candidate"] == "t1" and orig["angle_deg"] < 1e-9
    R = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    e["observations"][0]["coordinates"] = (R@[0,1,0]).tolist()
    e["observations"][0]["surface_normal_sample_cartesian"] = (R@[0,0,1]).tolist()
    spun = run(p, e, parent_to_sample_rotation=R)["observations"][0]
    assert spun["nearest_candidate"] == orig["nearest_candidate"]
    assert abs(spun["angle_deg"]-orig["angle_deg"]) < 1e-9


def test_trace_geometry_requires_oriented_section():
    p, e = fixture()
    e["observations"] = [{"observation_id":"trace","interface_kind":"twin",
        "frame":"sample_surface_trace_cartesian", "coordinates":[1,0,0],
        "surface_normal_sample_cartesian":[0,0,1],"uncertainty_deg":1}]
    with pytest.raises(BlindEvidenceError, match="orientation"): run(p, e)
    e["observations"][0]["coordinates"] = [0,0,1]
    with pytest.raises(BlindEvidenceError, match="tangent"):
        run(p,e,parent_to_sample_rotation=np.eye(3))


def test_unobservable_surface_trace_must_fail():
    p, e = fixture()
    p["predictions"] = p["predictions"][2:]
    e["observations"] = [{"observation_id":"trace","interface_kind":"habit",
        "frame":"sample_surface_trace_cartesian", "coordinates":[1,0,0],
        "surface_normal_sample_cartesian":[0,0,1], "uncertainty_deg":1}]
    with pytest.raises(BlindEvidenceError, match="unobservable"):
        run(p,e,parent_to_sample_rotation=np.eye(3))


def test_product_miller_normal_requires_product_metric_and_proper_measured_or():
    p, e = fixture()
    p["predictions"] = p["predictions"][:1]
    e["observations"] = [{"observation_id":"o","interface_kind":"twin",
        "frame":"product_reciprocal_miller","coordinates":[2,0,0],"uncertainty_deg":1}]
    with pytest.raises(BlindEvidenceError, match="OR"): run(p,e)
    with pytest.raises(BlindEvidenceError, match="product metric"):
        run(p,e,product_to_parent_rotation=np.eye(3))
    result = compare_blind_interfaces(
        p,e,parent_metric=np.eye(3),product_metric=np.diag([4.,9.,16.]),
        product_to_parent_rotation=np.eye(3),null_trials=1024)
    assert result["observations"][0]["angle_deg"] < 1e-9
    with pytest.raises(BlindEvidenceError, match="proper"):
        run(p,e,product_to_parent_rotation=np.diag([-1,1,1]))


def test_wrong_or_is_falsified_by_angular_mismatch():
    p,e=fixture()
    e["observations"]=e["observations"][:1]
    e["observations"][0]["frame"]="product_cartesian"
    assert run(p,e,product_to_parent_rotation=np.eye(3))["observations"][0]["within_declared_uncertainty"]
    theta=np.pi/4
    wrong=np.array([[np.cos(theta),-np.sin(theta),0],
                    [np.sin(theta),np.cos(theta),0],[0,0,1]])
    assert not run(p,e,product_to_parent_rotation=wrong)["observations"][0]["within_declared_uncertainty"]


def test_bad_identifiers_covariance_and_fake_independence_rejected():
    p,e=fixture()
    p["predictions"].append(dict(p["predictions"][0]))
    with pytest.raises(BlindEvidenceError, match="Duplicate"): run(p,e)
    p,e=fixture()
    e["measured_from_prediction"]=True
    with pytest.raises(BlindEvidenceError, match="Independent"): run(p,e)
    p,e=fixture()
    e["composition_wt_percent"]["Cu"]=95
    with pytest.raises(BlindEvidenceError, match="composition"): run(p,e)
    p,e=fixture()
    e["observations"][0]["coordinates"]=[float("nan"),0,0]
    with pytest.raises(BlindEvidenceError, match="finite"): run(p,e)
    p,e=fixture()
    p["crystal_inputs_sha256"]="not-a-hash"
    with pytest.raises(BlindEvidenceError, match="fingerprint|hex"): run(p,e)


def test_fingerprints_are_content_sensitive_and_canonical():
    p,_=fixture()
    assert prediction_digest(p) == prediction_digest(copy.deepcopy(p))
    q=copy.deepcopy(p)
    q["predictions"][0]["normal_parent_cartesian"]=[0,1,0]
    assert prediction_digest(q) != prediction_digest(p)
    with pytest.raises(BlindEvidenceError): prediction_digest({"x":float("nan")})


def test_order_independent_nearest_and_ambiguity():
    p,e=fixture()
    first=run(p,e)["observations"][0]
    p["predictions"].reverse()
    second=run(p,e)["observations"][0]
    assert first["nearest_candidate"] == second["nearest_candidate"]
    assert first["second_nearest_candidate"] == second["second_nearest_candidate"]
    assert first["isotropic_null_chance_at_observed_best_angle"] == second["isotropic_null_chance_at_observed_best_angle"]
    p,e=fixture()
    p["predictions"][1]["normal_parent_cartesian"]=[1,0,0]
    result=run(p,e)["observations"][0]
    assert result["matching_candidate_count"] == 2
    assert not result["unique_match_under_declared_uncertainty"]
