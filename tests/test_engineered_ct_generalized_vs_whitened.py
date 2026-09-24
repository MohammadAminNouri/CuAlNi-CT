from __future__ import annotations

import json
import numpy as np

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.ct import normalized_cmc
from cualni_cryst.numerics import DEFAULT_NUMERICAL_POLICY
from validation.engineered.certified_binary64 import certified_mu, classify_mu
from validation.engineered.physical import first_order_case, crystal_rebasis
from validation.engineered.production import evaluate


def test_generalized_metric_decision_matches_certified_pencil_and_whitening_where_safe():
    """Whitening is a representation diagnostic, not the authoritative decision route."""
    tol = DEFAULT_NUMERICAL_POLICY.exact_eigenvalue
    failures = []

    for i in range(120):
        case = first_order_case("whitening-vs-generalized", i)
        for label, candidate in (
            ("base", case),
            ("rebased", crystal_rebasis(case, "whitening-rebasis", i)),
        ):
            prod = evaluate(candidate, tol=tol)
            cert = certified_mu(
                candidate.M_parent, candidate.M_product, candidate.C_m_from_a
            )
            cert_class = classify_mu(cert, tol)
            got_class = (
                bool(prod.ct_analysis.exact_compatible),
                int(prod.ct_analysis.degeneracy_order),
            )
            cert_res = float(np.max(np.abs(np.asarray(prod.mu) - cert)))

            # Independently form the historical symmetric-whitened matrix only
            # as a diagnostic.  It should agree tightly in a safe conditioning
            # regime, but it is not allowed to overrule the certified pencil
            # when whitening itself is ill-conditioned.
            corr = prod.correspondence
            try:
                q_white = np.sort(
                    np.linalg.eigvalsh(
                        normalized_cmc(candidate.M_parent, candidate.M_product, corr)
                    )
                )
                white_res = float(
                    np.max(np.abs(q_white - np.sort(np.asarray(prod.mu) - 1.0)))
                )
            except Exception as exc:
                q_white = np.array([np.nan, np.nan, np.nan])
                white_res = float("inf")
                white_exc = repr(exc)
            else:
                white_exc = None

            cond = max(
                float(np.linalg.cond(candidate.M_parent)),
                float(np.linalg.cond(candidate.M_product)),
                float(np.linalg.cond(candidate.C_m_from_a)),
            )
            bad = got_class != cert_class or cert_res > 5e-11
            if cond <= 1e6 and white_res > 3e-8:
                bad = True
            if bad:
                failures.append(
                    {
                        "case": case.case_id,
                        "representation": label,
                        "condition": cond,
                        "certified_mu": cert.tolist(),
                        "production_mu": np.asarray(prod.mu).tolist(),
                        "certified_class": list(cert_class),
                        "production_class": list(got_class),
                        "certified_residual": cert_res,
                        "whitened_q": q_white.tolist(),
                        "whitened_residual": white_res,
                        "whitened_exception": white_exc,
                    }
                )

    assert not failures, (
        "Generalized metric decision disagrees with certified binary64 pencil "
        "or safe-regime whitening diagnostic:\n"
        + json.dumps(failures[:10], indent=2, sort_keys=True)
    )
