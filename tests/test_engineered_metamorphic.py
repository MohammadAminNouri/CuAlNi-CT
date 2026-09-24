from __future__ import annotations

import json

from validation.engineered.harness import _meta_bad
from validation.engineered.physical import first_order_case, incompatible_case
from validation.engineered.metamorphic import evaluate_metamorphic_family


def test_metamorphic_relations_are_judged_against_each_encoded_problem_oracle():
    """Do not confuse solver failure with binary64 re-encoding drift.

    A hostile SL(3,Z) rebasing can change the *encoded* floating-point pencil
    enough to cross the fixed compatibility threshold.  The correct invariant
    is therefore: production must agree with an independent oracle for each
    encoded representation; cross-representation covariance is demanded only
    when the oracle itself says the encoded quantity remained invariant.
    """
    failures = []
    for i in range(40):
        case = (
            first_order_case("meta-test", i)
            if i % 2 == 0
            else incompatible_case("meta-test", i)
        )
        for row in evaluate_metamorphic_family(case, i):
            if _meta_bad(row):
                failures.append({"case": case.case_id, **row})
    assert not failures, (
        "Metamorphic solver/oracle failures detected. Encoded-input shifts are "
        "reported separately and are not hidden by tolerance changes:\n"
        + json.dumps(failures[:10], indent=2, sort_keys=True)
    )
