from __future__ import annotations

"""Evidence-graded diagnosis of A/M and A/M/M compatibility failures.

The parent metric, correspondence and symmetry are frozen. Numerical
counterfactuals are HYPOTHETICAL product cells, never alloy recipes or
observed twin populations. CT/Cayron, Ball--James and PTMC are not replaced.
"""

from math import isfinite
from typing import Any, Mapping

from cualni_cryst.supercompatibility_atlas import (
    independent_product_lattice_parameters,
)
from twin_app.metric_design_targets import (
    metric_stretches, scan_lambda2_targets, state_at_parameter,
)

SOURCE_DOI = "10.1016/j.actamat.2026.122399"


def _cell(project: Any, transformation_id: str) -> dict[str, float]:
    t = project.transformation(transformation_id)
    lattice = project.phase(t.product_phase_id).lattice
    return {k: float(getattr(lattice, k)) for k in
            ("a", "b", "c", "alpha_deg", "beta_deg", "gamma_deg")}


def metric_failure_analysis(
    project: Any,
    transformation_id: str,
    *,
    length_relative_bound: float = 0.05,
    angle_bound_deg: float = 3.0,
    tolerance: float = 2e-8,
) -> dict:
    """Find verified, bounded one-axis CMC metric targets with exact magnitudes.

    A verified lambda2=1 target is NOT a solution to Eq.43 shear/shear,
    M/M twin rank-one compatibility, CC2 or CC3. Invalid states are not
    interpreted as compatible. A missing root is never a nonexistence proof.
    """
    before = _cell(project, transformation_id)
    l1, l2, l3 = metric_stretches(project, transformation_id)
    if not all(isfinite(x) and x > 0 for x in (l1, l2, l3)):
        raise ValueError("Invalid parent/product metric spectrum")
    axes = ("uniform_scale",) + independent_product_lattice_parameters(
        project, transformation_id
    )
    trials = []
    for parameter in axes:
        try:
            result = scan_lambda2_targets(
                project, transformation_id, parameter,
                length_relative_bound=length_relative_bound,
                angle_bound_deg=angle_bound_deg, tolerance=tolerance,
            )
            for candidate in result.candidates:
                proposed = state_at_parameter(
                    project, transformation_id, parameter,
                    float(candidate["parameter_value"]),
                )
                after = _cell(proposed, transformation_id)
                spectrum = metric_stretches(proposed, transformation_id)
                if abs(spectrum[1] - 1.0) > tolerance:
                    continue
                deltas = {k: after[k] - before[k] for k in before}
                if parameter == "uniform_scale":
                    changes = [f"{k}: {deltas[k]:+.9g} (entered length units)"
                               for k in ("a", "b", "c")]
                    magnitude = abs(float(candidate["change"])) * 100.0
                    magnitude_unit = "%"
                elif parameter.endswith("_offset_deg"):
                    k = parameter.removesuffix("_offset_deg") + "_deg"
                    changes = [f"{k}: {deltas[k]:+.9g}°"]
                    magnitude = abs(deltas[k])
                    magnitude_unit = "degree"
                else:
                    k = parameter.removesuffix("_scale")
                    changes = [f"{k}: {deltas[k]:+.9g} (entered length units)"]
                    magnitude = abs(float(candidate["change"])) * 100.0
                    magnitude_unit = "%"
                trials.append({
                    "parameter": parameter,
                    "target_parameter_value": float(candidate["parameter_value"]),
                    "parameter_delta": float(candidate["change"]),
                    "abs_change": magnitude,
                    "change_unit": magnitude_unit,
                    "physical_change": ", ".join(changes),
                    "original_cell": before,
                    "proposed_cell": after,
                    "lambda2": float(spectrum[1]),
                    "lambda2_residual": float(abs(spectrum[1]-1.0)),
                    "input_provenance": "HYPOTHETICAL metric design, not measurement",
                    "full_supercompatibility_verified": False,
                })
        except (AssertionError, ValueError, ArithmeticError) as exc:
            trials.append({
                "parameter": parameter,
                "status": "NOT_EVALUABLE",
                "reason": f"{type(exc).__name__}: {exc}",
            })
    solutions = [v for v in trials if "lambda2_residual" in v]
    solutions.sort(key=lambda r: (
        r["abs_change"] / (angle_bound_deg if r["change_unit"] == "degree"
                          else length_relative_bound*100),
        r["lambda2_residual"], r["parameter"]
    ))
    if abs(l2-1.0) <= tolerance and l1 <= 1+tolerance and l3 >= 1-tolerance:
        status = "AM_METRIC_CONDITION_SATISFIED"
        explanation = (
            "The middle stretch meets lambda2=1 within numerical tolerance. "
            "This does not certify an exact CT habit branch, an M/M twin, "
            "Cayron shear/shear closure, or the cofactor conditions."
        )
    else:
        status = "AM_METRIC_CONDITION_NOT_SATISFIED"
        explanation = (
            f"Principal stretches are ({l1:.9g}, {l2:.9g}, {l3:.9g}); "
            f"the middle-stretch defect is lambda2-1={l2-1:+.9g}. "
            "For the entered correspondence/metrics the single-variant "
            "A/M invariant-plane metric condition is not reached."
        )
    return {
        "source_doi": SOURCE_DOI,
        "status": status,
        "reason": explanation,
        "lambda123": (l1, l2, l3),
        "lambda2_minus_one": float(l2-1.0),
        "metric_tolerance": tolerance,
        "tested_parameter_axes": axes,
        "candidate_count": len(solutions),
        "candidate_changes": solutions,
        "unresolved_axes": [r for r in trials if "lambda2_residual" not in r],
        "bounded_exhaustive": False,
        "phase_chemical_feasibility": "NOT_ASSESSED",
        "twin_specific_supercompatibility": "NOT_ASSESSED",
        "warning": (
            "Each target was rechecked through the exact generalized metric "
            "eigenproblem. These changes reach only lambda2=1 and preserve "
            "the entered parent metric and C; full CT/BJ evaluation is necessary."
        ),
    }


def _condition(label: str, status: str, observed: Any, reason: str) -> dict:
    return {"condition": label, "status": status,
            "observed": observed, "interpretation": reason}


def supercompatibility_failure_analysis(
    report: Mapping[str, Any], *, metric_tolerance: float = 2e-8,
) -> dict:
    """Explain independently verified CT and BJ results without mixing twins.

    report is evaluate_supercompatibility_state(...).to_dict(), not a
    merged/invented theory score. No source-level CC2 numerical tolerance
    is guessed; rely on the backend's per-branch cofactor pass flag.
    """
    rows = []
    am = report.get("ct_am_exact_compatible")
    rows.append(_condition(
        "Exact CT A/M IPS habit (CMC/SMC)",
        "PASS" if am is True else "NOT_EVALUABLE" if am is None else "NOT_RESOLVED",
        am,
        "A/M seed must exist before the A/M/M shear/shear check can run."
        if am is not True else "Exact CT A/M seed available for follow-on twin analysis.",
    ))
    if am is not True:
        rows.append(_condition(
            "Cayron A/M/M shear/shear",
            "NOT_EVALUABLE", None,
            "No exact CT A/M habit seed. A missing Eq.43 result is not a failed Eq.43 root.",
        ))
    else:
        eps = report.get("ct_best_supercompatibility_residual")
        is_ok = report.get("ct_supercompatible")
        rows.append(_condition(
            "Cayron A/M/M shear/shear, CT Eq.43",
            "PASS" if is_ok is True else "NOT_EVALUABLE" if eps is None
            else "NOT_RESOLVED",
            eps,
            "The residual is dimensionless; exact closure must be assessed "
            "for one specific A/M habit and M/M twin, not by mixing branches.",
        ))
    pairs = list(report.get("pair_results") or ())
    qualified = [p for p in pairs if (
        p.get("match_completeness") == "full"
        and not p.get("missing_required_components")
        and p.get("match_geometry_within_tolerance") is True
    )]
    rows.append(_condition(
        "CT/Ball–James physical M/M twin agreement",
        "PASS" if qualified else "NOT_EVALUABLE",
        len(qualified),
        "Fully observed and geometrically agreeing comparisons are required. "
        "A partial assigned pair or data completeness alone is not a physical match.",
    ))
    for pair in pairs:
        note = {
            "CT twin row": pair.get("ct_row_id"),
            "BJ twin row": pair.get("ball_james_row_id"),
            "CT epsilon": pair.get("ct_supercompatibility_residual"),
            "CC1 |lambda2-1|": pair.get("cofactor_cc1_abs"),
            "CC2 absolute residual": pair.get("cofactor_cc2_abs"),
            "CC3 inequality margin": pair.get("cofactor_cc3_margin"),
            "CT pass": pair.get("ct_supercompatible"),
            "BJ cofactor pass": pair.get("cofactor_compatible"),
            "matching complete": pair.get("match_completeness"),
            "geometry agrees": pair.get("match_geometry_within_tolerance"),
            "all checks on this comparable pair": (
                pair.get("match_completeness") == "full"
                and not pair.get("missing_required_components")
                and pair.get("match_geometry_within_tolerance") is True
                and pair.get("ct_supercompatible") is True
                and pair.get("cofactor_compatible") is True
            ),
        }
        rows.append(_condition(
            f"Matched branch {pair.get('ct_row_id')} / {pair.get('ball_james_row_id')}",
            "PASS" if note["all checks on this comparable pair"] else
            "NOT_COMPARABLE" if not (
                note["matching complete"] == "full" and
                note["geometry agrees"] is True and
                not pair.get("missing_required_components")
            ) else "NOT_RESOLVED",
            note,
            "CC1, CC2, CC3 and Eq.43 must be assessed on this one "
            "specific pair; global minimum residuals are not interchangeable.",
        ))
    confirmed = any(
        r["status"] == "PASS" and r["condition"].startswith("Matched branch ")
        for r in rows
    )
    return {
        "status": "SAME_PAIR_NUMERICALLY_SATISFIED" if confirmed
        else "NO_SAME_PAIR_SUPERCOMPATIBILITY_CERTIFICATE",
        "conditions": rows,
        "matched_pairs_count": len(pairs),
        "physically_comparable_pairs": len(qualified),
        "cause_note": (
            "The output identifies which mathematical gate is unsatisfied, "
            "unavailable or not comparable. It cannot infer which chemical "
            "composition, annealing schedule or stress would generate a "
            "proposed lattice change. Numeric satisfaction is not experiment."
        ),
        "recommended_sequence": (
            "First target an exact CMC A/M condition (lambda2), "
            "then obtain a valid M/M twin with comparable geometry, "
            "then close Eq.43 shear/shear and verify CC1-CC3 for that pair. "
            "Use bounded continuation for candidate parameter magnitudes; "
            "do not claim global existence or follow one twin by array index."
        ),
    }


def summarize_sampled_changes(continuation: Mapping[str, Any],
                              original_cell: Mapping[str, float]) -> dict:
    """Provide quantified *sampled* counterfactuals and the actual failed gate."""
    samples = []
    for row in continuation.get("samples", ()):
        pair = row.get("paired_CT_BJ_branch")
        issues = []
        if row.get("lambda2_residual") is None:
            issues.append("No validated lambda2=1 candidate at this sampled point")
        if row.get("exact_CT_AM") is False:
            issues.append("No exact CT A/M habit; shear/shear unevaluable")
        if not pair:
            issues.append("No physically assigned CT–BJ twin information available")
        else:
            if not row.get("matching_is_complete", False):
                issues.append("CT/BJ M/M geometry incomplete or outside agreement tolerance")
            if pair.get("ct_satisfied_for_this_pair") is False:
                issues.append("Cayron shear/shear Eq.43 not closed")
            if pair.get("cofactor_satisfied_for_this_pair") is False:
                issues.append("At least one Ball–James cofactor condition fails")
        modified = {
            key: float(row[key])-float(original_cell[key])
            for key in ("a", "b", "c", "beta_deg")
            if row.get(key) is not None and original_cell.get(key) is not None
        }
        samples.append({
            "sample_id": row.get("sample_id"),
            "secondary_parameter": row.get("secondary_parameter"),
            "secondary_parameter_delta": row.get("secondary_delta"),
            "retuned_primary_value": row.get("primary_value"),
            "physical_cell_deltas": modified,
            "lambda2_residual": row.get("lambda2_residual"),
            "CT_epsilon": None if not pair else pair.get("ct_shear_shear_epsilon"),
            "CC1_abs": None if not pair else pair.get("CC1_abs_lambda2_minus_one"),
            "CC2_abs": None if not pair else pair.get("CC2_abs"),
            "CC3_margin": None if not pair else pair.get("CC3_margin"),
            "unmet_or_unevaluable_gates": issues,
            "same_pair_numerically_satisfied": (
                bool(pair) and bool(pair.get("both_satisfied_for_this_matched_pair"))
            ),
        })
    return {
        "samples": samples,
        "selected_sample_id": continuation.get("best_fully_matched_sample_id"),
        "proposed_direction": continuation.get("proposed_direction"),
        "cross_state_twin_identity_verified": False,
        "scope": (
            "These are actual sampled parameter changes relative to the frozen "
            "input cell, not a validated continuous solution or alloy recipe. "
            "Do not compare two different twin identities as a physical trend."
        ),
    }
