from __future__ import annotations

import traceback

from .contracts import evaluate_case
from .ledger import write_failure
from .mutation_score import mutation_score
from .physical import first_order_case,incompatible_case,degeneracy_case
from .metamorphic import evaluate_metamorphic_family
from .shrink import shrink_failure
from .jsonutil import to_jsonable


PROFILE={
    "quick":{"exact":12,"incompatible":12,"meta":6,"max_examples":20},
    "full":{"exact":800,"incompatible":800,"meta":250,"max_examples":60},
    "torture":{"exact":10000,"incompatible":10000,"meta":3000,"max_examples":100},
}


def _evaluate(case,*,shrink:bool):
    try:
        ev=evaluate_case(case)
    except Exception as exc:
        return {
            "case_id":case.case_id,
            "contract":"evaluation_exception",
            "failure_class":"framework_or_production_exception",
            "exception":repr(exc),
            "traceback":traceback.format_exc(),
        }
    if ev.passed:
        return None
    first=ev.failed_contracts[0]
    result={
        "case_id":case.case_id,
        "contract":first.name,
        "failure_class":first.failure_class.value,
        "residual":first.residual,
        "allowed":first.allowed,
        "details":first.details,
        "diagnostics":ev.diagnostics,
    }
    if shrink:
        try:
            minimized,history=shrink_failure(case,first.name)
            path=write_failure(case,ev,minimized=minimized,shrink_history=history)
            result["ledger"]=str(path)
            result["shrink_history"]=history
        except Exception as exc:
            result["shrink_exception"]=repr(exc)
    return result


def _meta_bad(row):
    return (
        row["lambda_residual"]>3e-6
        or row.get("classification_same") is False
        or row.get("direct_classification_same") is False
        or row.get("generalized_classification_same") is False
        or row.get("cmc_covariance_residual",0.0)>3e-6
        or row.get("smc_covariance_residual",0.0)>3e-6
        or row.get("physical_habit_family_distance",0.0)>3e-5
        or row.get("cmc_dimensional_law",0.0)>3e-6
        or row.get("smc_dimensional_law",0.0)>3e-6
    )


def run(profile:str)->dict:
    cfg=PROFILE[profile]
    max_examples=cfg["max_examples"]
    physical_examples=[]
    counts={
        "exact_attempted":0,"exact_failed":0,
        "incompatible_attempted":0,"incompatible_failed":0,
        "degeneracy_attempted":0,"degeneracy_failed":0,
        "metamorphic_attempted":0,"metamorphic_failed":0,
    }
    shrink_slots=5

    for i in range(cfg["exact"]):
        counts["exact_attempted"]+=1
        failure=_evaluate(
            first_order_case("physical-exact",i),
            shrink=(len(physical_examples)<shrink_slots),
        )
        if failure:
            counts["exact_failed"]+=1
            if len(physical_examples)<max_examples: physical_examples.append(failure)

    for i in range(cfg["incompatible"]):
        counts["incompatible_attempted"]+=1
        failure=_evaluate(
            incompatible_case("physical-incompatible",i),
            shrink=(len(physical_examples)<shrink_slots),
        )
        if failure:
            counts["incompatible_failed"]+=1
            if len(physical_examples)<max_examples: physical_examples.append(failure)

    for kind in ("second_low","second_high","identity","same_sign_low","same_sign_high"):
        counts["degeneracy_attempted"]+=1
        failure=_evaluate(degeneracy_case(kind),shrink=(len(physical_examples)<shrink_slots))
        if failure:
            counts["degeneracy_failed"]+=1
            if len(physical_examples)<max_examples: physical_examples.append(failure)

    meta_examples=[]
    for i in range(cfg["meta"]):
        counts["metamorphic_attempted"]+=1
        case=first_order_case("metamorphic",i) if i%2==0 else incompatible_case("metamorphic",i)
        try:
            rows=evaluate_metamorphic_family(case,i)
        except Exception as exc:
            counts["metamorphic_failed"]+=1
            if len(meta_examples)<max_examples:
                meta_examples.append({
                    "case":case.case_id,"name":"metamorphic_exception",
                    "exception":repr(exc),"traceback":traceback.format_exc(),
                })
            continue
        failed=[row for row in rows if _meta_bad(row)]
        if failed:
            counts["metamorphic_failed"]+=1
            if len(meta_examples)<max_examples:
                meta_examples.append({"case":case.case_id,"failures":failed})

    mutants=mutation_score()
    survivors=sorted(name for name,killed in mutants.items() if not killed)
    passed=(
        counts["exact_failed"]==0
        and counts["incompatible_failed"]==0
        and counts["degeneracy_failed"]==0
        and counts["metamorphic_failed"]==0
        and not survivors
    )
    return to_jsonable({
        "profile":profile,
        "counts":counts,
        "physical_failure_examples":physical_examples,
        "metamorphic_failure_examples":meta_examples,
        "mutation_score":{
            "killed":sum(bool(v) for v in mutants.values()),
            "total":len(mutants),
            "survivors":survivors,
            "details":mutants,
        },
        "passed":passed,
    })
