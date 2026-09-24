from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from validation.engineered.physical import first_order_case, encode_truth
from validation.engineered.high_precision import conditioning_diagnostic
from validation.engineered.production import evaluate
from validation.engineered.jsonutil import to_jsonable


def test_conditioning_sweep_is_differentiated_against_high_precision_oracle():
    """Separate solver error from unavoidable ill-conditioning.

    Stable/moderate pencils must agree tightly with the high-precision oracle.
    In extreme regimes explicit rejection is acceptable.  A finite but grossly
    wrong answer with a small backward residual is recorded as ill-conditioned,
    not mislabelled as an algebraic-formula error.
    """
    exponents = np.linspace(0.0, 12.0, 25)
    rows = []
    unacceptable = []

    for i, exponent in enumerate(exponents):
        base = first_order_case("conditioning-base", i)
        Ba = np.diag([1.0, 10**(exponent/2.0), 10**exponent])
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
            row = {
                "exponent": float(exponent),
                "condition_Ma": cond_ma,
                "production_exception": repr(exc),
                "classification": "explicit_rejection",
            }
            rows.append(row)
            # Rejection below roughly 1/eps^(1/2) in metric conditioning is too early.
            if cond_ma < 1e12:
                unacceptable.append(row)
            continue

        diag = conditioning_diagnostic(
            case.M_parent,
            case.M_product,
            case.C_m_from_a,
            prod.mu,
            digits=70,
        )
        row = {
            "exponent": float(exponent),
            **diag,
            "generalized_eigen_equation_residual": prod.generalized_eigen_equation_residual,
            "generalized_metric_orthonormality_residual": prod.generalized_metric_orthonormality_residual,
        }

        # Real SPD pencils should have real eigenvalues.  Significant imaginary
        # high-precision roots mean the decimalized input itself has crossed a
        # numerical validity boundary.
        if diag["max_imag"] > 1e-18:
            row["classification"] = "high_precision_input_boundary"
        elif cond_ma <= 1e8:
            row["classification"] = "stable_zone"
            if diag["max_relative_forward_error"] > 2e-7:
                unacceptable.append(row)
        elif cond_ma <= 1e14:
            row["classification"] = "conditioning_transition"
            if (
                diag["max_relative_forward_error"] > 2e-3
                and diag["max_pencil_backward_residual"] > 2e-8
            ):
                unacceptable.append(row)
        else:
            row["classification"] = "extreme_conditioning"
            # Still reject silent catastrophic nonsense.
            if (
                diag["max_relative_forward_error"] > 5e-2
                and diag["max_pencil_backward_residual"] > 1e-6
            ):
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
