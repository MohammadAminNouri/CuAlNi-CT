from __future__ import annotations

import json
from pathlib import Path

from cualni_cryst.numerics import DEFAULT_NUMERICAL_POLICY
from validation.engineered.physical import first_order_case
from validation.engineered.metamorphic import evaluate_metamorphic_family
from validation.engineered.jsonutil import to_jsonable


def test_basis_change_failures_are_localized_by_independent_error_ladder():
    """Triangulate where exact compatibility is lost.

    Stages:
      A intended physical truth;
      B direct Cartesian F from encoded B_M C B_A^-1;
      C generalized metric pencil;
      D whitened CMC classification.

    A failure is only blamed on production when B preserves the transformation
    but C or D does not.
    """
    tol = DEFAULT_NUMERICAL_POLICY.exact_eigenvalue
    rows = []
    production_failures = []
    encoding_limits = []

    for i in range(300):
        case = first_order_case("metamorphic", i)
        row = evaluate_metamorphic_family(case,i)[0]
        if row["classification_same"]:
            continue
        rows.append({"case":case.case_id,**row})
        before=row["stage_before"]; after=row["stage_after"]

        if after["direct_lambda2_residual"] > tol:
            encoding_limits.append({"case":case.case_id,**row})
            continue

        if after["generalized_lambda2_residual"] > tol:
            production_failures.append({
                "case":case.case_id,
                "stage":"generalized_metric_pencil",
                **row,
            })
            continue

        if after["whitened_class"] != after["generalized_class"]:
            production_failures.append({
                "case":case.case_id,
                "stage":"whitened_cmc",
                **row,
            })
            continue

        production_failures.append({
            "case":case.case_id,
            "stage":"other_discrete_classification",
            **row,
        })

    report=Path("/tmp/engineered_representation_error_ladder.json")
    report.write_text(
        json.dumps(to_jsonable({
            "classification_flaps":rows,
            "encoding_precision_limits":encoding_limits,
            "production_failures":production_failures,
        }),indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )

    assert not production_failures, (
        "Equivalent representations preserve the direct Cartesian physical "
        "problem but production loses compatibility downstream. Full ladder: "
        f"{report}\nFirst failures:\n"
        +json.dumps(to_jsonable(production_failures[:8]),indent=2,sort_keys=True)
    )
