"""Cu-Al-Ni experimental interface taxonomy and fail-closed geometric audits.

Two physically different observables must never be compared:
  martensite_martensite_twin -> a variant-specific M-side twin interface;
  austenite_martensite_habit -> a parent-frame A/M habit plane.
The solver's crystal-coordinate and Cartesian-frame conventions are explicit.
A geometric match cannot authenticate a specimen, an OR or its phase structure.
"""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np
from .p1_evidence import (
    EvidenceError, compare_observed_planes, projective_angle_deg,
    reciprocal_normal_to_cartesian, _unit
)

class InterfaceAuditError(ValueError):
    pass


def _required_source_string(value, label):
    # None -> 'None' is truthy under str(None); that is NOT provenance.
    if not isinstance(value, str) or not value.strip():
        raise InterfaceAuditError(f"Missing actual nonblank {label}")
    return value.strip()


INTERFACE_TYPES = frozenset({
    "martensite_martensite_twin",
    "austenite_martensite_habit",
})


def _source_construction(report, pair_id, branch):
    found = [
        c for family in report.families
        for pair in family.pair_records if pair.pair_id == pair_id
        for c in pair.constructions if int(c.branch) == branch
    ]
    if len(found) != 1:
        raise InterfaceAuditError(
            "Experimental observation must resolve exactly one existing "
            "source_pair_id + source_branch, never match arbitrary variants"
        )
    return found[0]


def _observation_provenance(spec, obs):
    a = spec.get("specimen_id")
    b = obs.get("specimen_id")
    if (not isinstance(a, str) or not a.strip() or
        not isinstance(b, str) or not b.strip() or a.strip() != b.strip()):
        raise InterfaceAuditError(
            "SPECIMEN_MISMATCH_OR_MISSING_ID: measured interface and solver "
            "lattices must be indexed to the SAME specimen"
        )
    ref = _required_source_string(
        obs.get("observation_reference"), "observation_reference"
    )
    if not isinstance(obs.get("coordinates"), (list, tuple)):
        raise InterfaceAuditError("Observation must supply three indexed coordinates")
    uncertainty = obs.get("uncertainty_deg")
    try: u = float(uncertainty)
    except (TypeError, ValueError) as exc:
        raise InterfaceAuditError("Positive angular uncertainty is mandatory") from exc
    if not np.isfinite(u) or not 0 < u <= 45:
        raise InterfaceAuditError("Angular uncertainty must be in (0,45] degrees")
    return ref, u


def _branch_selector(obs):
    pair_id = obs.get("source_pair_id")
    branch = obs.get("source_branch")
    if not isinstance(pair_id, str) or not pair_id.strip():
        raise InterfaceAuditError("A source_pair_id is mandatory for variant-specific comparisons")
    if isinstance(branch, bool) or not isinstance(branch, int):
        raise InterfaceAuditError("An integer source_branch is mandatory")
    return pair_id, branch


def compare_specimen_interfaces(report, spec: Mapping, parent_metric, product_metric):
    """Return each independently indexed comparison with strict interface typing.

    No comparison across a different specimen, different variant/gauge, or
    different type of interface. Only results conditional on user-provided C
    and indexed evidence are calculable, never independently certified.
    """
    observations = spec.get("observed_interface_planes", ())
    if not isinstance(observations, (list, tuple)):
        raise InterfaceAuditError("observed_interface_planes must be a list")
    if not observations:
        return {"status": "NO_EXPERIMENTAL_INTERFACE_COMPARISON",
                "observations": [], "independently_validated": False}
    out = []
    for index, obs in enumerate(observations):
        if not isinstance(obs, Mapping):
            raise InterfaceAuditError(f"Observation #{index} must be a mapping")
        kind = obs.get("interface_type")
        if kind not in INTERFACE_TYPES:
            raise InterfaceAuditError(
                "AMBIGUOUS_INTERFACE: specify martensite_martensite_twin or "
                "austenite_martensite_habit explicitly"
            )
        reference, uncertainty = _observation_provenance(spec, obs)
        pair_id, branch = _branch_selector(obs)
        construction = _source_construction(report, pair_id, branch)
        frame = obs.get("frame")
        if kind == "martensite_martensite_twin":
            if frame != "product_reciprocal_miller":
                raise InterfaceAuditError(
                    "Twin observations require product_reciprocal_miller coordinates "
                    "in the selected BASE martensite variant"
                )
            if obs.get("indexed_variant") != "base":
                raise InterfaceAuditError(
                    "Specify indexed_variant='base'; a different variant's "
                    "Miller indices must not be silently mixed"
                )
            try:
                observed = _unit(reciprocal_normal_to_cartesian(
                    obs["coordinates"], product_metric))
                predicted = _unit(reciprocal_normal_to_cartesian(
                    construction.twin_plane_product_crystal, product_metric))
                angular = projective_angle_deg(observed, predicted)
            except (EvidenceError, np.linalg.LinAlgError, ValueError) as exc:
                raise InterfaceAuditError(str(exc)) from exc
            geometric = {
                "angle_deg": angular, "compared_candidates": 1,
                "declared_uncertainty_deg": uncertainty,
                "status": ("GEOMETRIC_MATCH_ONLY_EXTERNAL_SOURCE_REVIEW_REQUIRED"
                           if angular <= uncertainty else
                           "NO_GEOMETRIC_MATCH_AT_DECLARED_UNCERTAINTY"),
            }
        else:
            if frame not in ("parent_cartesian", "parent_reciprocal_miller",
                             "product_reciprocal_miller", "product_cartesian"):
                raise InterfaceAuditError(
                    "Habit coordinates require an explicit parent/product "
                    "Cartesian or reciprocal Miller frame"
                )
            candidates = [
                h.habit_normal_parent_cartesian for h in construction.habit_solutions
            ]
            if not candidates:
                geometric = {
                    "status": "NO_CALCULATED_HABIT_FOR_SELECTED_BRANCH",
                    "angle_deg": None, "compared_candidates": 0,
                    "declared_uncertainty_deg": uncertainty,
                }
            else:
                orient = obs.get("measured_OR_M_cart_to_A_cart")
                if frame.startswith("product_"):
                    if (orient is None or
                        not isinstance(obs.get("OR_observation_reference"), str) or
                        not obs["OR_observation_reference"].strip()):
                        raise InterfaceAuditError(
                            "Product-indexed A/M habit requires a measured "
                            "variant-specific OR and its source in the SAME observation"
                        )
                try:
                    geometric = compare_observed_planes(
                        candidates, obs, parent_metric=parent_metric,
                        product_metric=product_metric,
                        OR_M_cart_to_A_cart=orient)
                except (EvidenceError, ValueError, np.linalg.LinAlgError) as exc:
                    raise InterfaceAuditError(str(exc)) from exc
        out.append({
            "index": index, "interface_type": kind,
            "specimen_id": str(spec["specimen_id"]).strip(),
            "observation_reference": reference,
            "source_pair_id": pair_id, "source_branch": branch,
            "observation_frame": frame,
            "geometric_comparison": geometric,
            "independently_validated": False,
            "warning": (
                "Mathematical agreement conditional on the specified C, phase, "
                "cell setting, variant indexing and OR where applicable. "
                "This does NOT establish independent experimental validation."
            ),
        })
    return {
        "status": "TYPED_GEOMETRIC_COMPARISONS_NOT_EXPERIMENTAL_CERTIFICATION",
        "observations": out, "independently_validated": False,
    }
