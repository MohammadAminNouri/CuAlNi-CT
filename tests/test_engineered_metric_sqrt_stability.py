from __future__ import annotations

import json
import numpy as np

from cualni_cryst.lattice import metric_sqrt, metric_inv_sqrt
from validation.engineered.physical import random_rotation
from validation.engineered.rng import rng


def test_metric_square_root_and_inverse_whitener_stability_across_conditioning():
    rows = []
    bad = []
    for i, exponent in enumerate(np.linspace(0, 14, 29)):
        g = rng("metric-sqrt-stability", i)
        Q = random_rotation(g)
        eig = np.array([1.0, 10**(exponent/2), 10**exponent])
        M = Q @ np.diag(eig) @ Q.T
        try:
            S = metric_sqrt(M)
            W = metric_inv_sqrt(M)
        except Exception as exc:
            row = {"exponent":float(exponent),"condition":float(np.linalg.cond(M)),"exception":repr(exc)}
            rows.append(row)
            if exponent < 10:
                bad.append(row)
            continue
        sqrt_res = float(np.linalg.norm(S@S-M)/max(np.linalg.norm(M),1.0))
        white_res = float(np.linalg.norm(W@M@W-np.eye(3)))
        row = {
            "exponent": float(exponent),
            "condition": float(np.linalg.cond(M)),
            "sqrt_residual": sqrt_res,
            "whitening_residual": white_res,
        }
        rows.append(row)
        if exponent <= 8 and (sqrt_res > 1e-8 or white_res > 1e-6):
            bad.append(row)
        if not np.all(np.isfinite(S)) or not np.all(np.isfinite(W)):
            bad.append(row)
    assert not bad, "metric sqrt/whitener instability:\n"+json.dumps(bad[:10],indent=2)
