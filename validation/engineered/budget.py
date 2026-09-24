from __future__ import annotations

import numpy as np
from .model import CrystalEncoding, ErrorBudget


def error_budget(
    case: CrystalEncoding,
    *,
    multiplier: float = 2.5e4,
    floor: float = 5e-11,
    ceiling: float = 2e-4,
) -> ErrorBudget:
    """Condition-aware budget based on first-order floating-point sensitivity.

    This is intentionally conservative.  A result exceeding the ceiling is not
    waved through: the case is classified as ill-conditioned and must be
    handled/reported explicitly.
    """
    eps = np.finfo(float).eps
    kappa = max(
        1.0,
        case.condition_parent_basis,
        case.condition_product_basis,
        case.condition_correspondence,
        np.sqrt(case.condition_metric_pair),
    )
    raw = multiplier * eps * (kappa ** 2)
    allowed = min(ceiling, max(floor, raw))
    return ErrorBudget(eps, float(kappa), multiplier, floor, ceiling, float(allowed))


def is_extremely_conditioned(case: CrystalEncoding, threshold: float = 1e9) -> bool:
    return max(
        case.condition_parent_basis,
        case.condition_product_basis,
        case.condition_correspondence,
    ) >= threshold
