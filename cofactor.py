from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .stretch import principal_stretches


def cofactor_matrix(A: np.ndarray) -> np.ndarray:
    """Cofactor matrix, valid even when A is singular."""
    A = np.asarray(A, dtype=float)
    c1, c2, c3 = A[:, 0], A[:, 1], A[:, 2]
    return np.column_stack((np.cross(c2, c3), np.cross(c3, c1), np.cross(c1, c2)))


@dataclass(frozen=True)
class CofactorResult:
    lambdas: np.ndarray
    cc1_residual: float
    cc2_residual: float
    cc3_margin: float
    satisfied: bool


def evaluate_cofactor_conditions(U: np.ndarray, a: np.ndarray, n: np.ndarray, tol: float = 1e-7) -> CofactorResult:
    """Chen et al. (2013) cofactor conditions for a specified twin system.

    CC1: lambda_2 = 1
    CC2: a · U cof(U^2 - I) n = 0
    CC3: tr(U^2) + det(U^2) - (|a|^2 |n|^2)/4 - 2 >= 0
    """
    U = np.asarray(U, dtype=float)
    a = np.asarray(a, dtype=float).reshape(3)
    n = np.asarray(n, dtype=float).reshape(3)
    lam, _ = principal_stretches(U)
    cc1 = float(lam[1] - 1.0)
    A = U @ U - np.eye(3)
    cc2 = float(a @ U @ cofactor_matrix(A) @ n)
    cc3 = float(np.trace(U @ U) + np.linalg.det(U @ U) - 0.25 * (a @ a) * (n @ n) - 2.0)
    ok = abs(cc1) <= tol and abs(cc2) <= tol and cc3 >= -tol
    return CofactorResult(lam, cc1, cc2, cc3, bool(ok))
