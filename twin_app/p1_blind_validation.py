"""Fail-closed Cu-Al-Ni 2H/M18R experimental geometry assessment.

Geometry agreements do NOT establish physical correctness or authenticate
external experiments. A 2D surface trace is NOT a measured 3D plane normal.
The solver must never load these observations as its predictions.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import re
from typing import Any, Mapping
import numpy as np


class BlindEvidenceError(ValueError):
    """Refuse ambiguous, inconsistent, or improperly indexed evidence."""


PHASES = frozenset(("gamma1_prime_2H", "beta1_prime_M18R"))
KINDS = frozenset(("twin", "habit"))
FRAMES = frozenset((
    "parent_cartesian", "parent_reciprocal_miller",
    "product_cartesian", "product_reciprocal_miller",
    "sample_plane_normal_cartesian", "sample_surface_trace_cartesian",
))


def _nonblank(value, name):
    if not isinstance(value, str) or not value.strip():
        raise BlindEvidenceError(f"{name}: nonempty string required")
    return value.strip()


def _unit(value, name):
    try:
        v = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise BlindEvidenceError(f"{name}: finite nonzero 3-vector required") from exc
    if v.shape != (3,) or not np.isfinite(v).all():
        raise BlindEvidenceError(f"{name}: finite nonzero 3-vector required")
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm < 1e-12:
        raise BlindEvidenceError(f"{name}: finite nonzero 3-vector required")
    return v / norm


def _root(value, name):
    try:
        G = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise BlindEvidenceError(f"{name}: invalid metric") from exc
    if G.shape != (3, 3) or not np.isfinite(G).all():
        raise BlindEvidenceError(f"{name}: finite symmetric 3x3 metric required")
    if np.linalg.norm(G-G.T, ord="fro") > 1e-10*max(1., np.linalg.norm(G, ord="fro")):
        raise BlindEvidenceError(f"{name}: nonsymmetric metric")
    eig, V = np.linalg.eigh(G)
    if eig[0] <= 0 or eig[0]/eig[-1] < 1e-12:
        raise BlindEvidenceError(f"{name}: non-positive or ill-conditioned metric")
    return (V*np.sqrt(eig)) @ V.T


def _rotation(value, name):
    try:
        R = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise BlindEvidenceError(f"{name}: valid proper 3x3 rotation required") from exc
    if (R.shape != (3, 3) or not np.isfinite(R).all()
        or np.linalg.norm(R.T@R-np.eye(3), ord="fro") > 1e-7
        or abs(float(np.linalg.det(R))-1) > 1e-7):
        raise BlindEvidenceError(f"{name}: valid proper 3x3 rotation required")
    return R


def prediction_digest(prediction_set: Mapping[str, Any]) -> str:
    """Fingerprint content; external preregistration time is NOT verified."""
    try:
        raw = json.dumps(prediction_set, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise BlindEvidenceError("Prediction set must be finite canonical JSON") from exc
    return sha256(raw.encode("utf-8")).hexdigest()


def _reference_contract(predictions, observations):
    for key in ("specimen_id", "phase"):
        a = _nonblank(predictions.get(key), "predictions."+key)
        b = _nonblank(observations.get(key), "observations."+key)
        if a != b:
            raise BlindEvidenceError(f"Specimen/phase mismatch in {key}: {a} != {b}")
    for key, length in (("solver_commit", 40), ("crystal_inputs_sha256", 64)):
        if not re.fullmatch(r"[0-9a-f]{%d}" % length, str(predictions.get(key, ""))):
            raise BlindEvidenceError(f"Prediction freeze must record {key} as {length}-digit lowercase hex")
    if predictions["phase"] not in PHASES:
        raise BlindEvidenceError("Explicitly indexed 2H or M18R required; generic monoclinic/6M is not interchangeable")
    for field in ("source_doi", "source_location", "measurement_method",
                  "composition_wt_percent", "measurement_temperature_K",
                  "lattice_setting", "sample_identity"):
        if field not in observations or observations[field] is None:
            raise BlindEvidenceError(f"Missing experimental provenance field: {field}")
    for key in ("source_doi", "source_location", "measurement_method",
                "lattice_setting", "sample_identity"):
        _nonblank(observations[key], key)
    if observations["sample_identity"] != predictions["specimen_id"]:
        raise BlindEvidenceError("Source specimen identity must match predicted specimen_id")
    composition = observations["composition_wt_percent"]
    if not isinstance(composition, Mapping) or any(k not in composition for k in ("Cu", "Al", "Ni")):
        raise BlindEvidenceError("Experimental mass composition of Cu, Al, Ni required")
    try:
        weights = np.array([float(composition[k]) for k in ("Cu", "Al", "Ni")])
        temperature = float(observations["measurement_temperature_K"])
    except (TypeError, ValueError) as exc:
        raise BlindEvidenceError("Non-numeric composition or temperature") from exc
    if (not np.isfinite(weights).all() or min(weights) <= 0
        or abs(float(sum(weights))-100) > 0.5
        or not np.isfinite(temperature) or temperature <= 0):
        raise BlindEvidenceError("Unphysical or incompletely normalized composition/temperature")
    if observations.get("measured_from_prediction") is not False:
        raise BlindEvidenceError("Independent measurements must declare measured_from_prediction=false")


def _observed_parent_plane(record, root_a, root_m, or_m_to_a, a_to_sample):
    frame = record["frame"]
    v = _unit(record["coordinates"], "observation.coordinates")
    if frame == "parent_cartesian":
        return v
    if frame == "parent_reciprocal_miller":
        return _unit(np.linalg.solve(root_a.T, v), "parent reciprocal normal")
    if frame.startswith("product_"):
        if or_m_to_a is None:
            raise BlindEvidenceError("Product-frame observation requires independently measured product-to-parent OR")
        if frame == "product_reciprocal_miller":
            if root_m is None:
                raise BlindEvidenceError("Product Miller plane requires specimen product metric")
            v = _unit(np.linalg.solve(root_m.T, v), "product reciprocal normal")
        return _unit(or_m_to_a@v, "product-to-parent normal")
    if frame == "sample_plane_normal_cartesian":
        if a_to_sample is None:
            raise BlindEvidenceError("Sample observation requires measured parent-to-sample orientation")
        return _unit(a_to_sample.T@v, "sample-to-parent normal")
    raise BlindEvidenceError(f"Cannot infer a 3D plane normal from {frame}")


def compare_blind_interfaces(
    prediction_set: Mapping[str, Any],
    observation_set: Mapping[str, Any], *,
    parent_metric: Any, product_metric: Any = None,
    product_to_parent_rotation: Any = None,
    parent_to_sample_rotation: Any = None,
    null_trials: int = 4096, random_seed: int = 9147,
) -> dict[str, Any]:
    """Exhaustive candidate comparison, never experimental certification.

    The null diagnostic estimates the chance a *uniform isotropic random*
    undirected plane (3D) / tangent trace (2D) matches at least as closely
    to ANY eligible candidate. It is NOT a physical experimental p-value.
    """
    if type(null_trials) is not int or not 1000 <= null_trials <= 100000:
        raise BlindEvidenceError("null_trials must be integer in [1000,100000]")
    if type(random_seed) is not int or random_seed < 0:
        raise BlindEvidenceError("random_seed must be nonnegative integer")
    _reference_contract(prediction_set, observation_set)
    root_a = _root(parent_metric, "parent metric")
    root_m = _root(product_metric, "product metric") if product_metric is not None else None
    or_m_to_a = _rotation(product_to_parent_rotation, "product-to-parent OR") if product_to_parent_rotation is not None else None
    a_to_sample = _rotation(parent_to_sample_rotation, "parent-to-sample orientation") if parent_to_sample_rotation is not None else None
    candidates = prediction_set.get("predictions")
    observations = observation_set.get("observations")
    if not isinstance(candidates, list) or not candidates or not isinstance(observations, list) or not observations:
        raise BlindEvidenceError("Nonempty explicit prediction and observation lists required")

    normalized = []
    seen = set()
    for c in candidates:
        if not isinstance(c, Mapping):
            raise BlindEvidenceError("Each prediction must be a mapping")
        cid = _nonblank(c.get("candidate_id"), "candidate_id")
        if cid in seen:
            raise BlindEvidenceError("Duplicate prediction candidate_id")
        seen.add(cid)
        kind = c.get("interface_kind")
        if kind not in KINDS:
            raise BlindEvidenceError("Predicted interface_kind must be twin or habit")
        variant = _nonblank(c.get("variant_pair"), "candidate.variant_pair")
        normal = _unit(c.get("normal_parent_cartesian"), "candidate.normal_parent_cartesian")
        normalized.append((cid, kind, variant, normal))

    used = Counter()
    rng = np.random.default_rng(random_seed)
    seen = set()
    records = []
    for obs in observations:
        if not isinstance(obs, Mapping):
            raise BlindEvidenceError("Each observation must be a mapping")
        oid = _nonblank(obs.get("observation_id"), "observation_id")
        if oid in seen:
            raise BlindEvidenceError("Duplicate observation_id")
        seen.add(oid)
        kind, frame = obs.get("interface_kind"), obs.get("frame")
        if kind not in KINDS or frame not in FRAMES:
            raise BlindEvidenceError("Observation interface_kind and explicit frame required")
        try:
            uncertainty = float(obs["uncertainty_deg"])
        except (TypeError, ValueError, KeyError) as exc:
            raise BlindEvidenceError("Numeric angular uncertainty required") from exc
        if not np.isfinite(uncertainty) or not 0 < uncertainty <= 45:
            raise BlindEvidenceError("Measured uncertainty must be in (0,45] degrees")
        measured = _unit(obs.get("coordinates"), "observation.coordinates")
        pair = obs.get("variant_pair")
        if pair is not None:
            pair = _nonblank(pair, "observation.variant_pair")
        eligible = [c for c in normalized if c[1] == kind and (pair is None or c[2] == pair)]
        if not eligible:
            raise BlindEvidenceError(f"No {kind} candidate for {oid}/{pair}; do not match twin to habit")
        is_trace = frame == "sample_surface_trace_cartesian"
        rejected = []
        if is_trace:
            if a_to_sample is None:
                raise BlindEvidenceError("2D traces require measured parent-to-sample orientation")
            surface = _unit(obs.get("surface_normal_sample_cartesian"), "surface_normal_sample_cartesian")
            if abs(float(measured@surface)) > 1e-5:
                raise BlindEvidenceError("Observed trace is not tangent to measured surface")
            eligible_vectors = []
            for c in eligible:
                trace = np.cross(surface, a_to_sample@c[3])
                if np.linalg.norm(trace) < 1e-10:
                    rejected.append(c[0])
                else:
                    eligible_vectors.append((c, _unit(trace, "calculated trace")))
        else:
            measured = _observed_parent_plane(obs, root_a, root_m, or_m_to_a, a_to_sample)
            eligible_vectors = [(c, c[3]) for c in eligible]
        if not eligible_vectors:
            raise BlindEvidenceError(f"All predictions geometrically unobservable for {oid}")
        calc = np.stack([v for _, v in eligible_vectors])
        ids = [c[0] for c, _ in eligible_vectors]
        angles = np.rad2deg(np.arccos(np.clip(np.abs(calc@measured), 0., 1.)))
        order = sorted(range(len(ids)), key=lambda i: (float(angles[i]), ids[i]))
        best = float(angles[order[0]])
        nearest = ids[order[0]]
        used[nearest] += 1
        matching = [ids[i] for i in order if angles[i] <= uncertainty]
        if is_trace:
            e1 = _unit(np.cross(surface, measured), "trace tangent basis")
            e2 = _unit(np.cross(surface, e1), "trace second basis")
            phi = rng.uniform(0., np.pi, size=null_trials)
            null = np.cos(phi)[:, None]*e1+np.sin(phi)[:, None]*e2
        else:
            null = rng.standard_normal((null_trials, 3))
            null /= np.linalg.norm(null, axis=1)[:, None]
        random_best = np.rad2deg(np.arccos(np.clip(np.max(np.abs(null@calc.T), axis=1), 0., 1.)))
        chance = float((1+np.count_nonzero(random_best <= best+1e-12))/(null_trials+1))
        records.append({
            "observation_id": oid, "interface_kind": kind,
            "geometry_dimension": "2D_TRACE_ONLY" if is_trace else "3D_PLANE_NORMAL",
            "angular_uncertainty_deg": uncertainty,
            "nearest_candidate": nearest, "angle_deg": best,
            "normalized_angular_residual": best/uncertainty,
            "within_declared_uncertainty": bool(best <= uncertainty),
            "matching_candidate_ids": matching,
            "matching_candidate_count": len(matching),
            "unique_match_under_declared_uncertainty": len(matching) == 1,
            "independent_variant_pair_available": pair is not None,
            "second_nearest_candidate": ids[order[1]] if len(order) > 1 else None,
            "second_nearest_angle_deg": float(angles[order[1]]) if len(order) > 1 else None,
            "raw_eligible_candidates": len(eligible),
            "compared_candidates": len(eligible_vectors),
            "unobservable_candidate_ids": rejected,
            "all_candidate_residuals_deg": [
                {"candidate_id": ids[i], "angle_deg": float(angles[i])} for i in order
            ],
            "isotropic_null_chance_at_observed_best_angle": chance,
            "null_model": "UNIFORM_UNDIRECTED_2D_TRACE" if is_trace else "UNIFORM_UNDIRECTED_3D_NORMAL",
        })
    return {
        "status": "GEOMETRIC_ASSESSMENT_ONLY_NOT_PHYSICALLY_CERTIFIED",
        "specimen_id": prediction_set["specimen_id"],
        "phase": prediction_set["phase"],
        "prediction_sha256": prediction_digest(prediction_set),
        "solver_commit": prediction_set["solver_commit"],
        "crystal_inputs_sha256": prediction_set["crystal_inputs_sha256"],
        "externally_timestamped_prediction_freeze_verified": False,
        "source_independently_authenticated_by_software": False,
        "source_review_declaration": observation_set.get("source_independently_reviewed") is True,
        "prediction_count": len(normalized),
        "observation_count": len(records),
        "unmatched_observation_ids": [r["observation_id"] for r in records if not r["within_declared_uncertainty"]],
        "reused_nearest_candidates": {k: n for k, n in used.items() if n > 1},
        "observations": records,
        "null_trials": null_trials,
        "random_seed": random_seed,
        "limitations": [
            "Surface traces constrain azimuth, not 3D interface inclination.",
            "A multi-candidate isotropic-null geometric diagnostic is not an experimental p-value.",
            "Neither external source nor timestamp is authenticated by software.",
            "Nearest candidate does not prove an independently identified crystallographic variant.",
            "Phase stacking, setting, C, measured OR and sample identity need external review.",
            "Repeated nearest predictions are not automatically distinct interface variants.",
            "Solver and input hashes alone do not prove prior registration.",
        ],
    }
