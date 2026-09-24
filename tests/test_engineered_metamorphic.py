from __future__ import annotations

import json

from validation.engineered.physical import first_order_case, incompatible_case
from validation.engineered.metamorphic import evaluate_metamorphic_family


def test_metamorphic_relations_hold_for_unseen_physical_cases():
    failures = []
    for i in range(40):
        case = (
            first_order_case("meta-test", i)
            if i % 2 == 0
            else incompatible_case("meta-test", i)
        )
        for row in evaluate_metamorphic_family(case, i):
            bad = (
                row["lambda_residual"] >= 3e-6
                or row.get("classification_same") is False
                or row.get("cmc_covariance_residual", 0.0) >= 3e-6
                or row.get("smc_covariance_residual", 0.0) >= 3e-6
                or row.get("physical_habit_family_distance", 0.0) >= 3e-5
                or row.get("cmc_dimensional_law", 0.0) >= 3e-6
                or row.get("smc_dimensional_law", 0.0) >= 3e-6
            )
            if bad:
                failures.append({"case": case.case_id, **row})
    assert not failures, (
        "Metamorphic failures detected.  Classification flaps are retained as "
        "real production diagnostics rather than hidden by tolerance changes:\n"
        + json.dumps(failures[:10], indent=2, sort_keys=True)
    )
