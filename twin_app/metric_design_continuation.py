from __future__ import annotations

"""Cayron-style conditional A/M/M lattice design on a lambda2=1 manifold.

For each allowed change in a second product-cell parameter, independently
retarget one free primary parameter to lambda2=1, then solve the EXACT
existing CT CMC, M/M and shear/shear equations plus Ball--James/cofactor
for the same candidate. Never use the minimum of unrelated twin branches
to claim joint supercompatibility. This is a bounded, sampled experiment,
not an exhaustive optimization or a physical alloy manufacturing route.
"""

from math import isfinite
from typing import Any, Callable

import numpy as np

from .metric_design_targets import (
    metric_stretches, scan_lambda2_targets, state_at_parameter,
)
from cualni_cryst.supercompatibility_atlas import (
    independent_product_lattice_parameters,
    evaluate_supercompatibility_state,
)


def _matched_branch_summary(state) -> dict | None:
    """One CT-BJ matched relation; never combine CC's from different pairs."""
    available = [
        x for x in state.pair_results
        if x.ct_supercompatibility_residual is not None
        and isfinite(float(x.ct_supercompatibility_residual))
    ]
    if not available:
        return None
    # A lower residual on a partial match must never mask a valid,
    # completely matched relation. Preserve partial evidence ONLY when
    # the current state has no full relation available.
    fully_matched = [
        x for x in available
        if x.match_completeness == "full"
        and not x.missing_required_components
    ]
    selected = min(
        fully_matched or available,
        key=lambda x: float(x.ct_supercompatibility_residual),
    )
    return {
        "operator_index": selected.ct_operator_index,
        "twin_index": selected.ct_twin_index,
        "habit_index": selected.ct_best_habit_index,
        "ct_shear_shear_epsilon":float(selected.ct_supercompatibility_residual),
        "CC1_abs_lambda2_minus_one": selected.cofactor_cc1_abs,
        "CC2_abs":selected.cofactor_cc2_abs,
        "CC3_margin":selected.cofactor_cc3_margin,
        "cofactor_satisfied_for_this_pair":bool(selected.cofactor_compatible),
        "ct_satisfied_for_this_pair":bool(selected.ct_supercompatible),
        "match_completeness": selected.match_completeness,
        "unmatched_required_components": list(selected.missing_required_components),
        "both_satisfied_for_this_matched_pair": bool(
            selected.ct_supercompatible and selected.cofactor_compatible
            and selected.match_completeness == "full"
            and not selected.missing_required_components
        ),
    }


def continue_supercompatibility_design(
    project: Any,
    transformation_id: str,
    *,
    primary_parameter: str,
    secondary_parameter: str,
    secondary_relative_bound: float = .02,
    angle_bound_deg: float = 2.,
    primary_relative_bound: float = .12,
    primary_angle_bound_deg: float = 10.,
    steps: int = 5,
    evaluator: Callable | None = None,
) -> dict:
    """Evaluate shear/shear and CC1/2/3 along a lambda2=1 family.

    Main constraint: exact generalized-metric middle stretch lambda2=1.
    Conditional optimization: lowest computed CT shear/shear epsilon for one
    *physically matched* CT-BJ twin relation, with its own CC1/2/3.
    Optimization is a finite scan only; it may have no admissible candidates.

    If an exact CMC/SMC habit is absent, CT supercompatibility is not
    evaluated. No 'failure' or score is fabricated in this case.
    """
    if primary_parameter == secondary_parameter:
        raise ValueError("Choose two independent, different parameter axes")
    axes = independent_product_lattice_parameters(project, transformation_id)
    possible_primary = ("uniform_scale",) + axes
    if primary_parameter not in possible_primary or secondary_parameter not in axes:
        raise ValueError("Select two allowed, independent product-cell parameters")
    if not 0 < secondary_relative_bound < .3:
        raise ValueError("Secondary length bound must be between 0 and 30%")
    if not 0 < primary_relative_bound < .5:
        raise ValueError("Primary length bound must be between 0 and 50%")
    if not 0 < angle_bound_deg < 25 or not 0 < primary_angle_bound_deg < 25:
        raise ValueError("Angle windows must lie between 0 and 25 degrees")
    if steps not in (3, 5, 7):
        raise ValueError("Scan must use 3, 5 or 7 explicit lattice states")
    if evaluator is None:
        evaluator = evaluate_supercompatibility_state

    is_angle = secondary_parameter.endswith("_offset_deg")
    base = 0. if is_angle else 1.
    bound = angle_bound_deg if is_angle else secondary_relative_bound
    grid = np.linspace(base-bound,base+bound,steps)
    if not np.any(np.isclose(grid,base,atol=1e-12)):
        raise AssertionError("Scan does not contain unchanged baseline")

    samples=[]
    for index,sec_value in enumerate(grid):
        row={
            "sample_id":index,
            "secondary_parameter":secondary_parameter,
            "secondary_value":float(sec_value),
            "secondary_delta":float(sec_value-base),
            "primary_parameter":primary_parameter,
            "status":"not_evaluated",
        }
        try:
            perturbed = state_at_parameter(
                project, transformation_id, secondary_parameter,
                float(sec_value),
            )
            target = scan_lambda2_targets(
                perturbed,transformation_id,primary_parameter,
                length_relative_bound=primary_relative_bound,
                angle_bound_deg=primary_angle_bound_deg,
                samples=61,tolerance=2e-8,
            )
            if not target.candidates:
                row.update({
                    "status":"no_resolved_lambda2_root",
                    "min_metric_scan_residual":target.minimum_abs_residual,
                })
                samples.append(row)
                continue
            root=target.candidates[0]
            achieved=state_at_parameter(
                perturbed,transformation_id,primary_parameter,
                float(root["parameter_value"]),
            )
            lam=metric_stretches(achieved,transformation_id)
            if abs(lam[1]-1.)>2e-8:
                raise ArithmeticError("Retargeted lambda2 outside root tolerance")
            trans=achieved.transformation(transformation_id)
            lattice=achieved.phase(trans.product_phase_id).lattice
            row.update({
                "primary_value":float(root["parameter_value"]),
                "lambda2":float(lam[1]),
                "lambda2_residual":abs(lam[1]-1.),
                "a":float(lattice.a),
                "b":float(lattice.b),
                "c":float(lattice.c),
                "alpha_deg":float(lattice.alpha_deg),
                "beta_deg":float(lattice.beta_deg),
                "gamma_deg":float(lattice.gamma_deg),
            })
            verified = evaluator(
                achieved, transformation_id,
                state_id=f"conditional-{index}",
                metadata={
                    "primary_parameter":primary_parameter,
                    "secondary_parameter":secondary_parameter,
                    "lambda2_target":1.,
                },
            )
            pair=_matched_branch_summary(verified)
            row.update({
                "status":(
                    "ct_am_evaluable" if verified.ct_am_exact_compatible
                    else "ct_exact_am_habit_unavailable"
                ),
                "exact_CT_AM":bool(verified.ct_am_exact_compatible),
                "CT_AMM_supercompatible":verified.ct_supercompatible,
                "cofactor_any_branch":verified.cofactor_compatible,
                "matched_relation_count":verified.matched_relation_count,
                "best_CT_epsilon_any_branch":
                    verified.ct_best_supercompatibility_residual,
                "paired_CT_BJ_branch":pair,
                "matching_is_complete":(
                    pair is not None
                    and pair["match_completeness"] == "full"
                    and not pair["unmatched_required_components"]
                ),
            })
        except (AssertionError,ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
            row.update({"status":"not_evaluable","error":f"{type(exc).__name__}: {exc}"})
        samples.append(row)

    qualified = [
        r for r in samples if r.get("paired_CT_BJ_branch")
        and r["paired_CT_BJ_branch"]["ct_shear_shear_epsilon"] is not None
        and r["matching_is_complete"]
    ]
    qualified.sort(key=lambda r: (
        r["paired_CT_BJ_branch"]["ct_shear_shear_epsilon"],
        abs(r["secondary_delta"]),
    ))
    best=qualified[0] if qualified else None
    center=next(r for r in samples if abs(r["secondary_delta"])<1e-10)
    if best is None:
        advice="NO_BRANCH_RESOLVED: do not infer which parameter improves full supercompatibility"
    elif abs(best["secondary_delta"])<1e-12:
        advice="UNCHANGED: no sampled change improved the matched shear/shear residual"
    elif best["secondary_delta"]>0:
        advice=f"INCREASE {secondary_parameter} toward {best['secondary_value']:.9g} (sampled target only)"
    else:
        advice=f"DECREASE {secondary_parameter} toward {best['secondary_value']:.9g} (sampled target only)"
    if best is not None and center in qualified:
        baseline_residual=center["paired_CT_BJ_branch"]["ct_shear_shear_epsilon"]
        improved=best["paired_CT_BJ_branch"]["ct_shear_shear_epsilon"]<baseline_residual
    else:
        improved=None
    return {
        "source_doi":"10.1016/j.actamat.2026.122399",
        "primary_lambda2_constraint":primary_parameter,
        "secondary_design_axis":secondary_parameter,
        "n_sampled_states":steps,
        "samples":samples,
        "best_fully_matched_sample_id":(
            None if best is None else best["sample_id"]
        ),
        "proposed_direction":advice,
        "reduction_against_baseline_if_comparable":improved,
        "scientific_limit":(
            "Each sampled point enforces generalized-metric lambda2=1, "
            "then independently solves CT CMC/SMC M/M shear/shear and "
            "Ball-James cofactor conditions for the same candidate. "
            "The best sampled complete CT-BJ matched relation is a "
            "hypothesis, not an exhaustive global optimum, a proven "
            "experimental twin, or an alloy composition recipe. "
            "Near-degenerate and unmatched states stay unevaluable. "
            "Branch identity across samples is not proven merely by "
            "similar operator/twin indices."
        ),
    }
