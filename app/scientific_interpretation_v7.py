from __future__ import annotations

"""Phase-2 professor-facing interpretation corrections."""

from dataclasses import replace
from typing import Any, Mapping

import app.scientific_interpretation_v6 as v6
from app.phase2_contracts import scientific_status, sanitize_finding


def transformation_findings(result: Mapping[str, Any]):
    return tuple(sanitize_finding(item) for item in v6.transformation_findings(result))


def topology_finding(result: Mapping[str, Any]):
    return sanitize_finding(v6.topology_finding(result))


def orientation_finding(analysis: Mapping[str, Any]):
    finding = v6.orientation_finding(analysis)
    origin = str(analysis.get("origin_note", ""))
    low = origin.lower()
    count = None
    try:
        count = int(analysis["report"]["orientation_variant_count"])
    except Exception:
        pass
    suffix = f" and generates {count} symmetry-distinct orientation variants" if count is not None else ""
    if "polar decomposition" in low or "polar" in low and "candidate" in low:
        conclusion = f"The calculated polar-rotation candidate is a proper rotation{suffix}."
    elif "euler" in low or "zxz" in low:
        conclusion = f"The Euler-derived OR is a proper rotation{suffix}."
    elif "parallelism" in low:
        conclusion = f"The OR derived from the supplied crystallographic parallelisms is a proper rotation{suffix}."
    elif "user-specified" in low or "user specified" in low:
        conclusion = f"The user-specified OR is a proper rotation{suffix}."
    else:
        conclusion = f"The active physical OR is a proper rotation{suffix}."
    return sanitize_finding(replace(finding, conclusion=conclusion))


def martensite_pair_finding(pair: Mapping[str, Any], *, unique_systems: int):
    finding = v6.martensite_pair_finding(pair, unique_systems=unique_systems)
    relations = list(pair.get("relations", []))
    raw = len(relations)
    if raw:
        if raw == unique_systems:
            conclusion = f"{raw} Mallard branch relation(s) correspond to {unique_systems} distinct physical twin system(s)."
        else:
            conclusion = f"{raw} Mallard generator/branch relation(s) reduce to {unique_systems} distinct physical twin system(s) after physical deduplication."
        finding = replace(finding, conclusion=conclusion)
    return sanitize_finding(finding)


def ebsd_audit_finding(audit: Mapping[str, Any]):
    return sanitize_finding(v6.ebsd_audit_finding(audit))


def am_existence_finding(checks: Mapping[str, Any], *, exact_ct_habits: int, bj_am: int, approx_ct_habits: int):
    finding = v6.am_existence_finding(
        checks,
        exact_ct_habits=exact_ct_habits,
        bj_am=bj_am,
        approx_ct_habits=approx_ct_habits,
    )
    ct_exact = bool(checks.get("ct_am_exact_compatible"))
    rationale = (
        f"CT exact A/M compatibility: {scientific_status(ct_exact)}; "
        f"exact CT habit branches: {exact_ct_habits}; "
        f"Ball–James exact A/M branches: {bj_am}."
    )
    return sanitize_finding(replace(finding, rationale=rationale))


def mm_audit_finding(audit):
    finding = v6.mm_audit_finding(audit)
    return None if finding is None else sanitize_finding(finding)


def ptmc_finding(unified: Any, *, enabled: bool):
    return sanitize_finding(v6.ptmc_finding(unified, enabled=enabled))


def or_comparison_finding(equivalence: Any, *, enabled: bool, tolerance_deg: float):
    return sanitize_finding(v6.or_comparison_finding(equivalence, enabled=enabled, tolerance_deg=tolerance_deg))


def cofactor_finding(unified: Any):
    return sanitize_finding(v6.cofactor_finding(unified))


def supercompatibility_finding(unified: Any, *, requested: bool, algebraic_tolerance: float):
    return sanitize_finding(v6.supercompatibility_finding(unified, requested=requested, algebraic_tolerance=algebraic_tolerance))


def experiment_finding(equivalence: Any):
    finding = v6.experiment_finding(equivalence)
    return None if finding is None else sanitize_finding(finding)
