from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import sqrtm


def _cosd(x: float) -> float:
    return float(np.cos(np.deg2rad(x)))


@dataclass(frozen=True)
class Lattice:
    """Conventional crystallographic cell.

    Parameters are dimensional lengths (normally Angstrom) and degrees.
    The metric tensor therefore has length^2 units.
    """

    a: float
    b: float
    c: float
    alpha_deg: float = 90.0
    beta_deg: float = 90.0
    gamma_deg: float = 90.0
    label: str = ""

    def metric(self) -> np.ndarray:
        """Return direct-space metric M such that |u|^2 = u.T @ M @ u."""
        a, b, c = self.a, self.b, self.c
        ca, cb, cg = _cosd(self.alpha_deg), _cosd(self.beta_deg), _cosd(self.gamma_deg)
        M = np.array(
            [
                [a * a, a * b * cg, a * c * cb],
                [a * b * cg, b * b, b * c * ca],
                [a * c * cb, b * c * ca, c * c],
            ],
            dtype=float,
        )
        eig = np.linalg.eigvalsh(M)
        if np.any(eig <= 0):
            raise ValueError(f"Metric is not positive definite; eigenvalues={eig}")
        return M

    @classmethod
    def cubic(cls, a: float, label: str = "cubic") -> "Lattice":
        return cls(a=a, b=a, c=a, label=label)

    @classmethod
    def orthorhombic(cls, a: float, b: float, c: float, label: str = "orthorhombic") -> "Lattice":
        return cls(a=a, b=b, c=c, label=label)

    @classmethod
    def monoclinic_unique_b(
        cls, a: float, b: float, c: float, beta_deg: float, label: str = "monoclinic"
    ) -> "Lattice":
        return cls(a=a, b=b, c=c, alpha_deg=90.0, beta_deg=beta_deg, gamma_deg=90.0, label=label)


def reciprocal_metric(M: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(M, dtype=float))


def metric_norm(v: np.ndarray, M: np.ndarray) -> float:
    v = np.asarray(v, dtype=float).reshape(3)
    return float(np.sqrt(max(0.0, v @ M @ v)))


def normalize_direct(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = metric_norm(v, M)
    if n == 0:
        raise ValueError("Cannot normalize a zero direct-space vector.")
    return v / n


def plane_norm(p: np.ndarray, M: np.ndarray) -> float:
    """Norm of a reciprocal covector p, using M^{-1}."""
    p = np.asarray(p, dtype=float).reshape(3)
    Mi = np.linalg.inv(M)
    return float(np.sqrt(max(0.0, p @ Mi @ p)))


def normalize_plane(p: np.ndarray, M: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float).reshape(3)
    n = plane_norm(p, M)
    if n == 0:
        raise ValueError("Cannot normalize a zero plane covector.")
    return p / n


def plane_to_unit_normal(p: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Convert a plane covector p to its unit direct-space normal."""
    p_unit = normalize_plane(p, M)
    return np.linalg.inv(M) @ p_unit


def metric_sqrt(M: np.ndarray) -> np.ndarray:
    """Symmetric positive square root of a positive-definite metric."""
    S = np.real_if_close(sqrtm(np.asarray(M, dtype=float)))
    S = np.asarray(S, dtype=float)
    return 0.5 * (S + S.T)


def metric_inv_sqrt(M: np.ndarray) -> np.ndarray:
    return np.linalg.inv(metric_sqrt(M))


def to_orthonormal_vector(u_crystal: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Map direct crystallographic coordinates to an orthonormal metric-whitened basis."""
    return metric_sqrt(M) @ np.asarray(u_crystal, dtype=float).reshape(3)


def plane_normal_orthonormal(p_crystal: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Physical plane normal in the metric-whitened orthonormal basis."""
    n_cr = plane_to_unit_normal(p_crystal, M)
    n = to_orthonormal_vector(n_cr, M)
    return n / np.linalg.norm(n)


def transform_matrix_to_orthonormal(g_crystal: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Represent a crystallographic linear map in the metric-whitened orthonormal basis."""
    S = metric_sqrt(M)
    return S @ np.asarray(g_crystal, dtype=float) @ np.linalg.inv(S)
