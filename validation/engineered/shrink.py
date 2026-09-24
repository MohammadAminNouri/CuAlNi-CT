from __future__ import annotations

import numpy as np

from .physical import encode_truth, reencode
from .contracts import evaluate_case


def shrink_failure(case, contract_name: str):
    """Greedy semantics-preserving simplifier for a failing contract.

    It simplifies representation first (C, basis), then physical orientation.
    A candidate is retained only when the same named contract still fails.
    """
    def fails(candidate) -> bool:
        ev = evaluate_case(candidate)
        return any((x.name == contract_name and not x.passed) for x in ev.contracts)

    current = case
    history = []

    candidates = [
        reencode(current, C=np.eye(3), tag="shrink:C->I"),
        reencode(current, B_parent=np.eye(3), tag="shrink:B_A->I"),
        encode_truth(
            current.truth, "shrink-both", 0,
            C=np.eye(3), B_parent=np.eye(3),
            provenance=current.provenance + ("shrink:C,B_A->I",),
        ),
    ]
    for cand in candidates:
        try:
            if fails(cand):
                current = cand
                history.append(cand.provenance[-1])
        except Exception:
            pass

    # Remove physical rotations while retaining principal stretches.
    truth = current.truth
    lam = truth.lambdas
    Udiag = np.diag(lam)
    from .model import PhysicalTruth
    from .physical import classify_lambdas
    cls, compat, order = classify_lambdas(lam)
    simple_truth = PhysicalTruth(
        case_id=truth.case_id + "/shrunk",
        lambdas=lam.copy(),
        U_physical=Udiag,
        R_physical=np.eye(3),
        F_physical=Udiag,
        truth_class=cls,
        exact_compatible=compat,
        expected_degeneracy_order=order,
        notes="shrinker removed physical rotations",
    )
    simple = encode_truth(
        simple_truth, "shrink-orientation", 0,
        C=np.eye(3), B_parent=np.eye(3),
        provenance=current.provenance + ("shrink:R,Q,C,B_A->I",),
    )
    try:
        if fails(simple):
            current = simple
            history.append("shrink:R,Q,C,B_A->I")
    except Exception:
        pass

    return current, history
