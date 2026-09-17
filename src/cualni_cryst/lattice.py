from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import sqrtm


def _cosd(x: float) -> float:
    return float(np.cos(np.deg2rad(x)))


def _sind(x: float) -> float:
    return float(np.sin(np.deg2rad(x)))


@dataclass(frozen=True)
class Lattice:
    """Conventional crystallographic cell.

    Parameters are dimensional lengths (normally Angstrom) and angles in degrees.
    The direct-space metric therefore has units length^2.

    The class intentionally keeps the physical dimensional representation.  A
    dimensionless normalization is available separately; it is never applied silently.
    """

    a: float
    b: float
    c: float
    alpha_deg: float = 90.0
    beta_deg: float = 90.0
    gamma_deg: float = 90.0
    label: str = ""

    def __post_init__(self) -> None:
        if min(self.a, self.b, self.c) <= 0:
            raise ValueError("Lattice lengths must be positive.")
        if not all(0.0 < x < 180.0 for x in (self.alpha_deg, self.beta_deg, self.gamma_deg)):
            raise ValueError("Cell angles must lie strictly between 0 and 180 degrees.")
        # Validate positive definiteness immediately.
        _ = self.metric()

    def metric(self) -> np.ndarray:
        """Direct-space metric M, so |u|^2 = u^T M u for crystal coordinates u."""
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

    def reciprocal_metric(self) -> np.ndarray:
        return np.linalg.inv(self.metric())

    def structure_matrix(self) -> np.ndarray:
        """Return one conventional Cartesian realization B of the crystal basis.

        Columns of B are the Cartesian vectors a,b,c and B.T @ B = M.
        This is useful at the EBSD / sample-coordinate boundary.  Internal CT
        calculations do not require choosing B.
        """
        a, b, c = self.a, self.b, self.c
        ca, cb, cg = _cosd(self.alpha_deg), _cosd(self.beta_deg), _cosd(self.gamma_deg)
        sg = _sind(self.gamma_deg)
        if abs(sg) < 1e-14:
            raise ValueError("gamma too close to 0 or 180 degrees for conventional structure matrix")
        ax = a
        bx, by = b * cg, b * sg
        cx = c * cb
        cy = c * (ca - cb * cg) / sg
        cz2 = c * c - cx * cx - cy * cy
        if cz2 < -1e-10:
            raise ValueError("Invalid cell geometry produced negative c_z^2")
        cz = float(np.sqrt(max(0.0, cz2)))
        B = np.array([[ax, bx, cx], [0.0, by, cy], [0.0, 0.0, cz]], dtype=float)
        if not np.allclose(B.T @ B, self.metric(), atol=1e-10, rtol=1e-10):
            raise AssertionError("Structure matrix does not reproduce the metric")
        return B

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
    """Norm of a reciprocal covector p using M^{-1}."""
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
    """Convert a plane covector to a unit direct-space normal in crystal coordinates."""
    p_unit = normalize_plane(p, M)
    n = np.linalg.inv(M) @ p_unit
    # With p_unit normalized by M^-1, n is already unit in M.
    return n


def direct_to_plane_normal(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Reciprocal covector whose physical normal is parallel to direct vector v."""
    return np.asarray(M, dtype=float) @ np.asarray(v, dtype=float).reshape(3)


def metric_sqrt(M: np.ndarray) -> np.ndarray:
    """Symmetric positive square root of a positive-definite metric."""
    S = np.real_if_close(sqrtm(np.asarray(M, dtype=float)))
    S = np.asarray(S, dtype=float)
    return 0.5 * (S + S.T)


def metric_inv_sqrt(M: np.ndarray) -> np.ndarray:
    return np.linalg.inv(metric_sqrt(M))


def to_metric_orthonormal_vector(u_crystal: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Map direct crystal coordinates to a symmetric metric-whitened orthonormal basis."""
    return metric_sqrt(M) @ np.asarray(u_crystal, dtype=float).reshape(3)


def from_metric_orthonormal_vector(u_hat: np.ndarray, M: np.ndarray) -> np.ndarray:
    return metric_inv_sqrt(M) @ np.asarray(u_hat, dtype=float).reshape(3)


def plane_normal_cartesian(p_crystal: np.ndarray, lattice: Lattice) -> np.ndarray:
    """Physical Cartesian unit normal of plane p=(hkl)."""
    B = lattice.structure_matrix()
    n = np.linalg.inv(B).T @ np.asarray(p_crystal, dtype=float).reshape(3)
    n /= np.linalg.norm(n)
    return n


def direction_cartesian(u_crystal: np.ndarray, lattice: Lattice) -> np.ndarray:
    v = lattice.structure_matrix() @ np.asarray(u_crystal, dtype=float).reshape(3)
    v /= np.linalg.norm(v)
    return v


def transform_matrix_to_metric_orthonormal(g_crystal: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Represent a crystallographic map in the symmetric metric-whitened basis."""
    S = metric_sqrt(M)
    return S @ np.asarray(g_crystal, dtype=float) @ np.linalg.inv(S)


def metric_dot(u: np.ndarray, v: np.ndarray, M: np.ndarray) -> float:
    u = np.asarray(u, dtype=float).reshape(3)
    v = np.asarray(v, dtype=float).reshape(3)
    return float(u @ M @ v)


def reciprocal_dot(p: np.ndarray, q: np.ndarray, M: np.ndarray) -> float:
    p = np.asarray(p, dtype=float).reshape(3)
    q = np.asarray(q, dtype=float).reshape(3)
    return float(p @ np.linalg.inv(M) @ q)
