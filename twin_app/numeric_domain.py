from __future__ import annotations

"""Fail-closed bridge from exact rational C to binary64 numerical crystallography.

Invertibility over Q does not guarantee a safe finite matrix in float64.
This guard does not authenticate any physically measured correspondence.
"""

from typing import Sequence

import numpy as np
import sympy as sp

from .input_logic import exact_matrix


def validate_solver_correspondence(
    rows: Sequence[Sequence[object]] | sp.Matrix,
    *,
    maximum_condition: float = 1.0e12,
) -> np.ndarray:
    """Reject overflow, underflow and unresolved near-singular input matrices."""
    if not np.isfinite(maximum_condition) or maximum_condition < 1:
        raise ValueError("maximum_condition must be finite and >= 1")
    exact = exact_matrix(rows.tolist() if isinstance(rows, sp.MatrixBase) else rows)
    try:
        candidate = np.asarray(exact, dtype=np.float64)
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError("Exact correspondence cannot be represented in binary64") from exc
    if not np.isfinite(candidate).all():
        raise ValueError("Exact correspondence overflows binary64")
    for i in range(3):
        for j in range(3):
            if exact[i, j] != 0 and candidate[i, j] == 0:
                raise ValueError(f"Correspondence[{i+1},{j+1}] underflows to zero in binary64")
    largest = float(np.max(np.abs(candidate)))
    if largest == 0.0 or largest > 1.0e100 or largest < 1.0e-100:
        raise ValueError("Correspondence magnitude is outside stable numerical range")
    try:
        singular_values = np.linalg.svd(candidate / largest, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Correspondence singular values could not be computed") from exc
    if not np.isfinite(singular_values).all() or singular_values[-1] <= 0:
        raise ValueError("Exact correspondence has become numerically singular")
    condition = float(singular_values[0] / singular_values[-1])
    if not np.isfinite(condition) or condition > maximum_condition:
        raise ValueError(
            f"Correspondence ill-conditioned (condition number {condition:.3g}); "
            "higher-precision analysis required"
        )
    return candidate
