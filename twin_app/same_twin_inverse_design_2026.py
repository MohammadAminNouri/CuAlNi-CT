from __future__ import annotations

"""Bounded inverse Cayron Eq.42 design for ONE tracked physical CT twin.

Use exact lambda2=1 elimination (uniform product-cell scaling), then
minimize the *selected habit* / *selected twin* Eq.42 residual in the
remaining symmetry-allowed cell variables.  Do not optimize across changing
CT operator indices, unmatched twins, different habit solutions, or arbitrary
BJ cofactor branches.

The optimizer proposes numerical hypotheses and independently verifies its
final candidate. It neither proves global existence nor certifies Ball--James
cofactor conditions or an experimentally achievable alloy recipe.
"""
from math import isfinite
from typing import Any
import numpy as np
from scipy.optimize import minimize

from cualni_cryst.supercompatibility_atlas import (
    independent_product_lattice_parameters, uniform_product_scale_sweep,
)
from twin_app.joint_metric_design_2026 import state_at_joint_target
from twin_app.metric_design_targets import metric_stretches, _mark_product_design_hypothesis
from twin_app.same_twin_continuity_2026 import (
    TwinAnchor, enumerate_twin_anchors, reconstruct_same_twin,
    fixed_twin_am_habits, _projective_angle,
)


def _cell(state, transformation_id):
    t=state.transformation(transformation_id)
    p=state.phase(t.product_phase_id).lattice
    return {n:float(getattr(p,n)) for n in
            ("a","b","c","alpha_deg","beta_deg","gamma_deg")}


def _scaled_state(
    original: Any, transformation_id: str,
    parameters: dict[str,float], *,
    relative_length_bound: float, angle_bound_deg: float,
    max_uniform_scale_offset: float,
) -> tuple[Any,dict]:
    candidate=state_at_joint_target(original,transformation_id,parameters)
    pre=metric_stretches(candidate,transformation_id)
    if min(pre)<=0 or not np.all(np.isfinite(pre)):
        raise ValueError("Nonphysical generalized metric spectrum")
    uniform=1.0/pre[1]
    if abs(uniform-1.)>max_uniform_scale_offset+1e-10:
        raise ValueError("Exact lambda2=1 scaling exceeds allowed uniform factor")
    state=uniform_product_scale_sweep(
        candidate,transformation_id,(uniform,)
    )[0][1]
    state=_mark_product_design_hypothesis(state,transformation_id)
    state.validate().assert_passed()
    final=metric_stretches(state,transformation_id)
    if abs(final[1]-1.)>2e-8:
        raise ArithmeticError("Analytic scaling failed independent lambda2 recheck")
    initial=_cell(original,transformation_id)
    changed=_cell(state,transformation_id)
    lengths=("a","b","c")
    differences={k:100*(changed[k]/initial[k]-1.) for k in lengths}
    if any(abs(v)>100*relative_length_bound+1.e-8 for v in differences.values()):
        raise ValueError("Final cell violates individual length-change bounds")
    angle_deltas={
        k:changed[k]-initial[k] for k in ("alpha_deg","beta_deg","gamma_deg")}
    if any(abs(x)>angle_bound_deg+1e-8 for x in angle_deltas.values()):
        raise ValueError("Final cell violates individual angle-change bound")
    return state,{
        "uniform_lambda2_elimination":float(uniform),
        "metric_stretches":[float(v) for v in final],
        "lambda2_residual":float(final[1]-1.),
        "cell":changed,
        "length_change_percent":differences,
        "angle_change_deg":angle_deltas,
    }


def _minimize_with_budget(func, start, *, max_evaluations):
    if not 12<=max_evaluations<=1000:
        raise ValueError("Evaluation budget must lie between 12 and 1000")
    # scipy Powell is derivative-free. Each attempted candidate is tested
    # against exact physical constraints; infeasible points get a penalty.
    return minimize(
        func, np.asarray(start,float),
        method="Powell",bounds=[(-1.,1.) for _ in start],
        options={"maxfev":int(max_evaluations),"maxiter":35,
                 "xtol":1.e-3,"ftol":2.e-5},
    )


def optimize_one_tracked_twin(
    original: Any, transformation_id: str, *,
    branch_id: str, habit_index: int=0,
    relative_length_bound: float=.12,
    angle_bound_deg: float=4.,
    max_uniform_scale_offset: float=.16,
    max_evaluations: int=90,
    objective_regularization: float=1e-5,
    continuity_steps: int=7,
) -> dict:
    """Find a bounded, exact-lambda2 numerical CT Eq.42 candidate.

    Identity = frozen exact CT parent order-two generator and construction
    route. Habit = one A/M CMC solution continued from the lambda2-corrected
    initial cell.  Both branch geometries are checked at intermediate
    path states. A proposed zero epsilon is NOT a joint cofactor certificate.
    """
    if not 0<relative_length_bound<.4 or not isfinite(relative_length_bound):
        raise ValueError("Length bound must be positive and below 40%")
    if not 0<angle_bound_deg<20 or not isfinite(angle_bound_deg):
        raise ValueError("Angle bound must be positive and below 20 degrees")
    if not 0<max_uniform_scale_offset<.4 or not isfinite(max_uniform_scale_offset):
        raise ValueError("Uniform-scale bound must be positive and below 40%")
    if not 12<=max_evaluations<=1000:
        raise ValueError("Evaluation budget must lie between 12 and 1000")
    if continuity_steps not in (5,7,9,11):
        raise ValueError("Use 5, 7, 9 or 11 intermediate branch checks")
    if not 0<=objective_regularization<=.01 or not isfinite(objective_regularization):
        raise ValueError("Invalid minimum-change regularization")
    anchors=enumerate_twin_anchors(original,transformation_id)
    selected=next((a for a in anchors if a.branch_id==branch_id),None)
    if selected is None:
        raise ValueError("Selected twin generator does not exist for these exact inputs")
    axes=independent_product_lattice_parameters(original,transformation_id)
    base=np.array([0. if name.endswith("_offset_deg") else 1. for name in axes])
    width=np.array([
        angle_bound_deg if name.endswith("_offset_deg") else relative_length_bound
        for name in axes])
    def unpack(x):
        return {name:float(base[k]+width[k]*x[k])
                for k,name in enumerate(axes)}
    zeros=np.zeros(len(axes))
    try:
        anchor_state,anchor_metadata=_scaled_state(
            original,transformation_id,unpack(zeros),
            relative_length_bound=relative_length_bound,
            angle_bound_deg=angle_bound_deg,
            max_uniform_scale_offset=max_uniform_scale_offset,
        )
    except (ValueError,ArithmeticError,AssertionError) as exc:
        return {"status":"NO_EXACT_AM_ANCHOR_INSIDE_BOUNDS",
                "reason":str(exc),"branch_id":branch_id,
                "search_exhaustive":False,"candidate":None}
    baseline=fixed_twin_am_habits(
        anchor_state,transformation_id,selected)
    if baseline["status"]!="EXACT_CT_AM_HABITS_AND_FIXED_MM_TWIN":
        return {"status":"NO_UNAMBIGUOUS_STARTING_CMC_HABIT",
                "branch_id":branch_id,"reason":baseline["status"],
                "candidate":None,"search_exhaustive":False}
    if not 0<=habit_index<len(baseline["habits"]):
        raise ValueError(
            f"Requested habit branch {habit_index} is unavailable; "
            f"resolved count={len(baseline['habits'])}"
        )
    ref=np.asarray(baseline["habits"][habit_index]["habit_parent_reciprocal"],float)
    root_twin,_=reconstruct_same_twin(anchor_state,transformation_id,selected)
    initial_eps=float(baseline["habits"][habit_index]["epsilon"])
    original_cell=_cell(original,transformation_id)
    results={}
    evaluated=0

    def check(x):
        nonlocal evaluated
        key=tuple(round(float(z),9) for z in x)
        if key in results:
            return results[key]
        evaluated+=1
        record={"x":[float(v) for v in x],"status":"NOT_EVALUABLE"}
        try:
            if not all(isfinite(v) for v in x) or np.max(np.abs(x))>1.0000001:
                raise ValueError("Trial exceeds explicit optimization bounds")
            project,cell=_scaled_state(
                original,transformation_id,unpack(x),
                relative_length_bound=relative_length_bound,
                angle_bound_deg=angle_bound_deg,
                max_uniform_scale_offset=max_uniform_scale_offset)
            solved=fixed_twin_am_habits(
                project,transformation_id,selected,
                previous_twin=root_twin,reference_habit=ref,
                max_habit_angle_deg=28.)
            if solved["status"]!="EXACT_CT_AM_HABITS_AND_FIXED_MM_TWIN":
                raise ValueError("Fixed twin/habit not admissible: "+solved["status"])
            if not solved["habits"]:
                raise ValueError("Selected A/M habit could not be tracked")
            best=solved["habits"][0]
            if len(solved["habits"])>1:
                a=solved["habits"][0]["reference_habit_angle_deg"]
                b=solved["habits"][1]["reference_habit_angle_deg"]
                if abs(a-b)<1e-4:
                    raise ValueError("Two possible CMC habit branches have ambiguous identity")
            epsilon=float(best["epsilon"])
            if not isfinite(epsilon):
                raise ArithmeticError("Nonfinite Eq.42 epsilon")
            record={
                "x":[float(v) for v in x],"status":"VALID_ONE_BRANCH",
                "epsilon":epsilon,"cell":cell,
                "twin":solved["twin_trace"],
                "habit":best,
                "physical_cell":project,
            }
        except (ValueError,ArithmeticError,AssertionError,np.linalg.LinAlgError) as exc:
            record["reason"]=str(exc)
        results[key]=record
        return record

    def objective(x):
        record=check(x)
        if record["status"]!="VALID_ONE_BRANCH":
            return 100.+float(np.sum(np.asarray(x,float)**2))
        return record["epsilon"]**2+objective_regularization*float(np.dot(x,x))
    starts=[zeros]
    if len(axes):
        # Physically different deterministic perturbations, never a random
        # unrepeatable global search, and no source-target value hard-coded.
        for i in range(min(len(axes),3)):
            x=zeros.copy()
            x[i]=.45
            starts.append(x)
            x2=zeros.copy()
            x2[i]=-.45
            starts.append(x2)
    # Allocate independent budget across starts; do not report local status
    # as a proof of global optimality.
    per=max(12,int(max_evaluations//len(starts)))
    for start in starts:
        try:
            _minimize_with_budget(objective,start,max_evaluations=per)
        except (ValueError,ArithmeticError,np.linalg.LinAlgError):
            continue
    valid=[item for item in results.values()
           if item["status"]=="VALID_ONE_BRANCH"]
    if not valid:
        return {
            "status":"NO_VALID_TRACKED_CANDIDATE",
            "branch_id":selected.branch_id,"baseline_epsilon":initial_eps,
            "states_evaluated":evaluated,"candidate":None,
            "search_exhaustive":False,
            "reason":"No sampled admissible state preserved the selected twin and habit",
        }
    valid.sort(key=lambda r:(
        r["epsilon"]**2+objective_regularization*np.dot(r["x"],r["x"]),
        np.linalg.norm(r["x"])))
    # If a candidate's same-twin continuity fails on the path, try the
    # next valid numerical candidate rather than claiming a false solution.
    accepted=None
    rejected=[]
    for item in valid:
        previous_twin=root_twin
        previous_habit=ref
        path=[]
        for fraction in np.linspace(0.,1.,continuity_steps):
            x=np.asarray(item["x"])*float(fraction)
            try:
                state,cell=_scaled_state(
                    original,transformation_id,unpack(x),
                    relative_length_bound=relative_length_bound,
                    angle_bound_deg=angle_bound_deg,
                    max_uniform_scale_offset=max_uniform_scale_offset)
                twin,trace=reconstruct_same_twin(
                    state,transformation_id,selected,
                    previous_twin=previous_twin)
                rows=fixed_twin_am_habits(
                    state,transformation_id,selected,
                    previous_twin=previous_twin,reference_habit=previous_habit,
                    max_habit_angle_deg=20.)
                if rows["status"]!="EXACT_CT_AM_HABITS_AND_FIXED_MM_TWIN" or not rows["habits"]:
                    raise ValueError("A/M habit branch disappeared during continuation")
                h=rows["habits"][0]
                previous_twin=twin
                previous_habit=np.asarray(h["habit_parent_reciprocal"],float)
                path.append({
                    "fraction":float(fraction),
                    "epsilon":h["epsilon"],
                    "lambda2_minus_one":cell["lambda2_residual"],
                    "twin_plane_step_deg":trace["plane_step_deg"],
                    "twin_direction_step_deg":trace["direction_step_deg"],
                    "twin_shear":trace["shear"],
                    "habit_step_deg":h["reference_habit_angle_deg"],
                    "cell":cell["cell"],
                })
            except (ValueError,ArithmeticError,AssertionError,np.linalg.LinAlgError) as exc:
                rejected.append({"candidate_x":item["x"],"at_fraction":float(fraction),
                                 "reason":str(exc)})
                path=[]
                break
        if path:
            accepted=(item,path)
            break
    if accepted is None:
        return {
            "status":"NO_CONTINUOUS_PHYSICAL_TWIN_AND_HABIT_PATH",
            "branch_id":selected.branch_id,"baseline_epsilon":initial_eps,
            "search_exhaustive":False,"states_evaluated":evaluated,
            "candidate":None,"rejected_paths":rejected[:10],
        }
    winner,path=accepted
    product=winner["physical_cell"]
    _,independent=(
        winner["epsilon"],fixed_twin_am_habits(
            product,transformation_id,selected,
            previous_twin=root_twin,reference_habit=ref))
    final=independent["habits"][0]
    # Under no circumstances export ProjectState as JSON or claim it is
    # measured. Numerical candidates keep provenance HYPOTHETICAL_TEST.
    target={
        "baseline_cell":original_cell,
        "proposed_cell":winner["cell"]["cell"],
        "independent_parameters":unpack(winner["x"]),
        "parameter_normalized_changes":dict(zip(axes,winner["x"])),
        "length_change_percent":winner["cell"]["length_change_percent"],
        "angle_change_deg":winner["cell"]["angle_change_deg"],
        "uniform_lambda2_elimination":winner["cell"]["uniform_lambda2_elimination"],
        "principal_stretches":winner["cell"]["metric_stretches"],
        "lambda2_residual":winner["cell"]["lambda2_residual"],
        "baseline_epsilon":initial_eps,
        "candidate_epsilon":final["epsilon"],
        "epsilon_reduction_absolute":initial_eps-final["epsilon"],
        "numeric_eq42_closed":bool(final["epsilon"]<=2e-8),
        "selected_habit_parent_reciprocal":final["habit_parent_reciprocal"],
        "eq42_vector_decomposition":final["eq42"],
        "same_parent_symmetry_generator":selected.generator_key,
        "twin_trace":independent["twin_trace"],
        "continuation_path":path,
        "data_status":"HYPOTHETICAL_TEST",
        "ball_james_cofactor_on_same_physical_twin":"NOT_VERIFIED",
        "experimental_composition_or_processing":"NOT_MODELLED",
    }
    # One-factor removal tests quantify why the coupled proposal helped.
    # Nonadditive interactions mean these numbers are conditional diagnostics,
    # not additive causal or measured material effects.
    attribution=[]
    xw=np.asarray(winner["x"],float)
    for k,name in enumerate(axes):
        without=xw.copy()
        without[k]=0.
        other=check(without)
        attribution.append({
            "parameter":name,
            "proposed_value":float(unpack(xw)[name]),
            "baseline_value":float(base[k]),
            "normalized_displacement":float(xw[k]),
            "epsilon_with_all_changes":float(winner["epsilon"]),
            "epsilon_without_this_change":(
                float(other["epsilon"]) if other["status"]=="VALID_ONE_BRANCH"
                else None
            ),
            "conditional_epsilon_reduction_due_to_change":(
                float(other["epsilon"]-winner["epsilon"])
                if other["status"]=="VALID_ONE_BRANCH" else None
            ),
            "removal_status":other["status"],
            "interpretation":(
                "Finite counterfactual along the exact lambda2=1 manifold; "
                "all other free axes held constant and uniform scale retuned. "
                "Contributions can interact and must not be summed."
            ),
        })
    target["factor_by_factor_counterfactuals"]=attribution
    return {
        "status":(
            "CT_EQ42_NUMERICALLY_CLOSED_REQUIRES_BJ_AND_EXPERIMENT"
            if target["numeric_eq42_closed"]
            else "BEST_BOUNDED_CONTINUOUS_SAME_TWIN_CT_CANDIDATE"
        ),
        "branch_id":selected.branch_id,
        "generator_key":selected.generator_key,
        "habit_anchor_index":habit_index,
        "candidate":target,
        "states_evaluated":evaluated,
        "invalid_states":sum(v["status"]!="VALID_ONE_BRANCH" for v in results.values()),
        "rejected_continuation_paths":rejected[:10],
        "search_exhaustive":False,
        "cofactor_same_pair_certified":False,
        "scope":(
            "Exact lambda2 is eliminated analytically at each lattice cell. "
            "The fixed rational CT generator, Type-I/II route, habit branch, "
            "and actual crystallographic plane/shear continuity are validated "
            "along a multi-step path. This is a bounded local CT Eq42 "
            "inverse design, not a global proof or cofactor/EBSD certificate."
        ),
    }


def verify_candidate_same_twin_cofactor(
    original: Any, transformation_id: str,
    inverse_report: dict,
    *, length_bound: float, angle_bound_deg: float,
    uniform_scale_bound: float,
) -> dict:
    """Re-run Ball–James only at the *final* cell, preserving CT identity.

    A cofactor pass on another CT generator or another CMC habit is not
    accepted. Even if numerically BOTH, experimental crystal control and
    uncertainty are outside the scope of these purely lattice calculations.
    """
    candidate=inverse_report.get("candidate")
    if candidate is None:
        raise ValueError("No resolved tracked-twin candidate to verify")
    project,_=_scaled_state(
        original,transformation_id,
        candidate["independent_parameters"],
        relative_length_bound=length_bound,
        angle_bound_deg=angle_bound_deg,
        max_uniform_scale_offset=uniform_scale_bound,
    )
    from cualni_cryst.supercompatibility_atlas import evaluate_supercompatibility_state
    audit=evaluate_supercompatibility_state(
        project,transformation_id,state_id="same_twin_inverse_final",
        metadata={"hypothetical":True,"generator":inverse_report["generator_key"]},
    )
    generator=inverse_report["generator_key"]
    habit=np.asarray(candidate["selected_habit_parent_reciprocal"],float)
    ma=original.phase(original.transformation(transformation_id).parent_phase_id).lattice.metric()
    # The atlas stores a local CMC habit_index (not an experimental label).
    # Verify geometry directly rather than trusting a potentially permuted index.
    exact_rows=[]
    from cualni_cryst.ct import analyze_austenite_martensite
    t=project.transformation(transformation_id)
    am=analyze_austenite_martensite(
        project.phase(t.parent_phase_id).lattice.metric(),
        project.phase(t.product_phase_id).lattice.metric(),
        t.correspondence,tol=project.numerical_policy.exact_eigenvalue)
    for i,plane in enumerate(am.exact_habit_planes):
        try:
            angle=_projective_angle(habit,plane,ma,True)
        except ValueError:
            continue
        if angle<1.e-4:
            exact_rows.append(i)
    if len(exact_rows)!=1:
        return {
            "status":"NO_UNAMBIGUOUS_CMC_HABIT_FOR_BJ_VERIFICATION",
            "matched_habit_indices":exact_rows,
            "matched_generator":generator,
            "all_BJ_cofactor_pass_count":audit.cofactor_compatible_branch_count,
            "same_pair_full_supercompatibility_verified":False,
        }
    rows=[p for p in audit.pair_results if (
        p.ct_generator_key==generator
        and p.ct_best_habit_index==exact_rows[0]
    )]
    eligible=[p for p in rows if (
        p.match_completeness=="full"
        and not p.missing_required_components
        and p.match_geometry_within_tolerance is True
    )]
    same_pair=[p for p in eligible if p.ct_supercompatible and p.cofactor_compatible]
    return {
        "status":(
            "CT_BJ_SAME_PAIR_NUMERICALLY_SATISFIED"
            if same_pair else
            "CT_BRANCH_PRESENT_BUT_COFACTOR_OR_MATCH_UNRESOLVED"
            if rows else "NO_CT_BJ_ASSIGNMENT_FOR_THE_TRACKED_GENERATOR"
        ),
        "matched_generator":generator,
        "matched_CMC_habit_index":exact_rows[0],
        "CT_AM_exact":audit.ct_am_exact_compatible,
        "CT_any_branch_pass":audit.ct_supercompatible,
        "BJ_any_branch_pass":audit.cofactor_compatible,
        "all_BJ_cofactor_pass_count":audit.cofactor_compatible_branch_count,
        "candidate_matched_pair_count":len(rows),
        "physically_comparable_candidate_pairs":len(eligible),
        "qualified_same_pair_count":len(same_pair),
        "same_pair_full_supercompatibility_verified":bool(
            audit.ct_am_exact_compatible and same_pair
        ),
        "pairs":[p.to_dict() for p in rows],
        "warning":(
            "This certificate, if present, is numerical for the specified "
            "lattice metric and physical twin assignment. It is not an "
            "experimentally obtained phase or an alloy/composition recipe. "
            "No complete assignment means UNRESOLVED, not disproven."
        ),
    }
