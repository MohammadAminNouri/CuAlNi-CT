from __future__ import annotations

"""Fail-closed UI bridge reuses the exact same scientific-core float boundary."""

from typing import Sequence
import numpy as np
import sympy as sp

from cualni_cryst.correspondence import Correspondence
from .input_logic import exact_matrix


def validate_solver_correspondence(
    rows: Sequence[Sequence[object]] | sp.Matrix,
    *, maximum_condition: float = 1.0e12,
) -> np.ndarray:
    exact = exact_matrix(rows.tolist() if isinstance(rows, sp.MatrixBase) else rows)
    return Correspondence(exact).to_float64_checked(maximum_condition=maximum_condition)
