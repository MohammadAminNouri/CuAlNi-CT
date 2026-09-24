from __future__ import annotations

import json
import numpy as np

from cualni_cryst.numerics import DEFAULT_NUMERICAL_POLICY
from validation.engineered.physical import first_order_case, crystal_rebasis
from validation.engineered.production import evaluate


def test_whitened_cmc_eigenvalues_track_generalized_metric_eigenvalues_under_rebasis():
    """Expose basis-sensitive whitening/classification separately from physics."""
    tol = DEFAULT_NUMERICAL_POLICY.exact_eigenvalue
    failures = []

    for i in range(120):
        case = first_order_case("whitening-vs-generalized", i)
        for label, candidate in (
            ("base", case),
            ("rebased", crystal_rebasis(case, "whitening-rebasis", i)),
        ):
            prod = evaluate(candidate, tol=tol)
            q_generalized = np.sort(prod.mu - 1.0)
            q_whitened = np.sort(np.asarray(prod.ct_analysis.eigenvalues,float))
            residual = float(np.max(np.abs(q_generalized-q_whitened)))

            generalized_exact = bool(
                np.sum(np.abs(q_generalized) <= tol) == 1
                and np.prod(q_generalized[np.abs(q_generalized) > tol]) < 0
            )
            if residual > 2e-7 or generalized_exact != bool(prod.ct_analysis.exact_compatible):
                failures.append({
                    "case": case.case_id,
                    "representation": label,
                    "condition_parent": candidate.condition_parent_basis,
                    "condition_product": candidate.condition_product_basis,
                    "q_generalized": q_generalized.tolist(),
                    "q_whitened": q_whitened.tolist(),
                    "residual": residual,
                    "generalized_exact": generalized_exact,
                    "ct_exact": bool(prod.ct_analysis.exact_compatible),
                    "tol": tol,
                })

    assert not failures, (
        "Normalized-whitening CMC disagrees with the generalized metric pencil "
        "under equivalent crystal representations:\n"
        + json.dumps(failures[:10], indent=2, sort_keys=True)
    )
