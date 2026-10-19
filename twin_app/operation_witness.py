from __future__ import annotations

"""An auditable coordinate-action witness for a point-group operation.

The witness is a *test direction*, not an atom, an EBSD direction, a twin,
 or an orientation relationship.  Metric-norm preservation is independently
 verified before the scene can display the action.
"""

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import sympy as sp


@dataclass(frozen=True)
class OperationWitness:
    before_fractional: tuple[float, float, float]
    after_fractional: tuple[float, float, float]
    before_cartesian: tuple[float, float, float]
    after_cartesian: tuple[float, float, float]
    relative_metric_residual: float
    determinant: int


def operation_witness(
    basis: np.ndarray,
    exact_matrix: Sequence[Sequence[str]],
    *, relative_tolerance: float = 1e-9,
) -> OperationWitness:
    """Apply G to a fixed generic direct-space test vector in a real metric.

    x' = G x and v' = B G x, where columns of B are cell vectors.  Never
    identify G with a physical rigid Cartesian rotation or an orientation OR.
    Refuse a matrix that does not preserve the supplied metric.
    """
    B = np.asarray(basis, dtype=float)
    if B.shape != (3, 3) or not np.isfinite(B).all() or np.linalg.det(B) <= 0:
        raise ValueError("A finite right-handed nonsingular cell basis is required")
    if not (0 < relative_tolerance <= 1e-4):
        raise ValueError("Invalid metric-preservation tolerance")
    G = sp.Matrix([[sp.Rational(v) for v in row] for row in exact_matrix])
    if G.shape != (3, 3):
        raise ValueError("A symmetry operator must be 3x3")
    det = sp.simplify(G.det())
    if det not in (sp.Integer(1), sp.Integer(-1)):
        raise ValueError("The selected operation must have determinant +1 or -1")
    # Distinct, non-axis-aligned coordinates make identities and swapping errors
    # evident; exact arithmetic is retained until the Cartesian embedding.
    x = sp.Matrix([sp.Rational(3, 7), sp.Rational(2, 11), sp.Rational(5, 13)])
    y = G * x
    before = np.array([float(v) for v in x], dtype=float)
    after = np.array([float(v) for v in y], dtype=float)
    v = B @ before
    w = B @ after
    # A single test vector is illustrative, not a proof of isometry. Check
    # the entire crystallographic metric independently: G.T M G = M.
    M = B.T @ B
    G_float = np.asarray(G.tolist(), dtype=float)
    transformed_metric = G_float.T @ M @ G_float
    metric_scale = max(float(np.linalg.norm(M, ord="fro")), np.finfo(float).tiny)
    residual = float(np.linalg.norm(transformed_metric - M, ord="fro")) / metric_scale
    if not np.isfinite(residual) or residual > relative_tolerance:
        raise ValueError(
            f"Displayed point-group action does not preserve the selected cell metric ({residual:.3e})"
        )
    return OperationWitness(
        tuple(float(t) for t in before), tuple(float(t) for t in after),
        tuple(float(t) for t in v), tuple(float(t) for t in w), residual,
        int(det),
    )
