from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from validation.engineered.physical import first_order_case, encode_truth
from validation.engineered.high_precision import conditioning_diagnostic
from validation.engineered.certified_binary64 import (
    exact_binary64_nonsingular,
    exact_canonical_metric_spd,
)
from validation.engineered.production import evaluate
from validation.engineered.jsonutil import to_jsonable


def test_conditioning_sweep_uses_forward_accuracy_not_backward_residual_alone():
    """A tiny backward residual is not accepted as proof of forward accuracy."""
    exponents = np.linspace(0.0, 12.0, 25)
    rows = []
    unacceptable = []

    for i, exponent in enumerate(exponents):
        base = first_order_case("conditioning-base", i)
        Ba = np.diag([1.0, 10 ** (exponent / 2.0), 10**exponent])
        case = encode_truth(
            base.truth,
            "conditioning-encoding",
            i,
            B_parent=Ba,
            C=base.C_m_from_a,
            provenance=("conditioning sweep",),
        )
        cond_ma = float(np.linalg.cond(case.M_parent))
        try:
            prod = evaluate(case)
        except Exception as exc:
            parent_spd = exact_canonical_metric_spd(case.M_parent)
            product_spd = exact_canonical_metric_spd(case.M_product)
            corr_nonsingular = exact_binary64_nonsingular(case.C_m_from_a)
            encoded_problem_valid = bool(
                parent_spd and product_spd and corr_nonsingular
            )
            row = {
                "exponent": float(exponent),
                "condition_Ma": cond_ma,
                "production_exception": repr(exc),
                "canonical_parent_exact_spd": parent_spd,
                "canonical_product_exact_spd": product_spd,
                "correspondence_exact_nonsingular": corr_nonsingular,
                "classification": (
                    "unexpected_rejection_of_exact_valid_pencil"
                    if encoded_problem_valid
                    else "encoded_problem_left_spd_or_nonsingular_domain"
                ),
            }
            rows.append(row)
            # A production rejection is acceptable only when an independent
            # exact-rational check proves that the *encoded* binary64 problem
            # is no longer a valid SPD/nonsingular metric pencil.  Condition
            # number alone is not used as a validity criterion.
            if encoded_problem_valid:
                unacceptable.append(row)
            continue

        diag = conditioning_diagnostic(
            case.M_parent,
            case.M_product,
            case.C_m_from_a,
            prod.mu,
            digits=90,
        )
        row = {
            "exponent": float(exponent),
            **diag,
            "generalized_eigen_equation_residual": prod.generalized_eigen_equation_residual,
            "generalized_metric_orthonormality_residual": prod.generalized_metric_orthonormality_residual,
            "solver_source": prod.ct_analysis.solver_source,
            "precision_escalated": bool(prod.ct_analysis.precision_escalated),
        }

        if diag["max_imag"] > 1e-25:
            row["classification"] = "oracle_input_boundary"
        elif cond_ma <= 1e14:
            row["classification"] = "required_forward_accuracy_zone"
            if diag["max_relative_forward_error"] > 5e-10:
                unacceptable.append(row)
            if cond_ma >= 1e8 and not prod.ct_analysis.precision_escalated:
                unacceptable.append(row)
        else:
            row["classification"] = "extreme_binary64_conditioning"
            # Even here, a returned finite answer must not be grossly wrong.
            if diag["max_relative_forward_error"] > 5e-5:
                unacceptable.append(row)
        rows.append(row)

    out = Path("/tmp/engineered_conditioning_sweep.json")
    out.write_text(
        json.dumps(to_jsonable(rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert not unacceptable, (
        "High-precision conditioning failures. Full sweep: "
        f"{out}\nFirst unacceptable cases:\n"
        + json.dumps(to_jsonable(unacceptable[:5]), indent=2, sort_keys=True)
    )
