from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import mpmath as mp
import numpy as np

from .units import normalize_length_unit


_FLOAT_EPS = np.finfo(float).eps


def _cosd(x: float) -> float:
    return float(np.cos(np.deg2rad(x)))


def _sind(x: float) -> float:
    return float(np.sin(np.deg2rad(x)))


def _fraction(x: float) -> Fraction:
    num, den = float(x).as_integer_ratio()
    return Fraction(num, den)


def _exact_spd_binary64(A: np.ndarray) -> bool:
    A = np.asarray(A, dtype=float).reshape(3, 3)
    a00, a01, a02 = map(_fraction, A[0])
    a10, a11, a12 = map(_fraction, A[1])
    a20, a21, a22 = map(_fraction, A[2])
    d1 = a00
    d2 = a00 * a11 - a01 * a10
    d3 = (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )
    return bool(d1 > 0 and d2 > 0 and d3 > 0)


def _validated_spd(M: np.ndarray, name: str = "metric") -> np.ndarray:
    A = np.asarray(M, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{name} contains non-finite entries")
    scale = max(float(np.linalg.norm(A, ord="fro")), np.finfo(float).tiny)
    asym = float(np.linalg.norm(A - A.T, ord="fro") / scale)
    if asym > 1.0e-12:
        raise ValueError(f"{name} must be symmetric; relative asymmetry={asym:g}")
    A = 0.5 * (A + A.T)
    eig = np.linalg.eigvalsh(A)
    spectral_scale = max(float(np.linalg.norm(A, ord=2)), np.finfo(float).tiny)
    ambiguous = float(np.min(eig)) <= 256.0 * _FLOAT_EPS * spectral_scale
    if ambiguous:
        if not _exact_spd_binary64(A):
            raise ValueError(
                f"{name} is not positive definite as an exact canonical "
                f"binary64 symmetric matrix; binary64 eigenvalues={eig}"
            )
    elif np.min(eig) <= 0.0:
        raise ValueError(f"{name} is not positive definite; eigenvalues={eig}")
    return A


def _mp_exact_float(x: float) -> mp.mpf:
    num, den = float(x).as_integer_ratio()
    return mp.mpf(num) / mp.mpf(den)


def _spd_eigh(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = _validated_spd(M)
    w, V = np.linalg.eigh(A)
    scale = max(float(np.linalg.norm(A, ord=2)), np.finfo(float).tiny)
    ambiguous = float(np.min(w)) <= 256.0 * _FLOAT_EPS * scale
    if not ambiguous:
        return w, V

    # The exact Sylvester test above already certified SPD.  If binary64 has
    # lost the smallest positive eigenvalue, evaluate the same symmetric
    # eigendecomposition on the exact canonical binary64 entries at enough
    # precision to recover the principal matrix functions.
    cond = float(np.linalg.cond(A))
    if np.isfinite(cond) and cond > 1.0:
        digits = max(100, int(np.ceil(np.log10(cond))) + 60)
    else:
        digits = 180
    with mp.workdps(digits):
        Amp = mp.matrix(
            [[_mp_exact_float(x) for x in row] for row in np.asarray(A, float)]
        )
        values, vectors = mp.eigsy(Amp)
        w = np.asarray([float(values[i]) for i in range(3)], dtype=float)
        V = np.asarray(
            [[float(vectors[i, j]) for j in range(3)] for i in range(3)],
            dtype=float,
        )
    if np.min(w) <= 0.0:
        raise ArithmeticError(
            "exact-SPD metric could not be represented with a positive "
            f"floating eigenspectrum even after {digits} decimal digits; eigenvalues={w}"
        )
    return w, V


def _positive_quadratic_form(v: np.ndarray, M: np.ndarray, *, name: str) -> float:
    v = np.asarray(v, dtype=float).reshape(3)
    A = _validated_spd(M)
    q = float(v @ A @ v)
    if q < 0.0:
        # Do not silently clamp a negative metric norm to zero.  A negative
        # result means the floating-point representation is no longer reliable
        # for this requested calculation.
        scale = float(np.linalg.norm(A, ord=2) * np.dot(v, v))
        raise ArithmeticError(
            f"{name} produced a negative quadratic form {q:g}; "
            f"scale={scale:g}, ratio={q / max(scale, np.finfo(float).tiny):g}"
        )
    return q


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
    length_unit: str = ""

    def __post_init__(self) -> None:
        if min(self.a, self.b, self.c) <= 0:
            raise ValueError("Lattice lengths must be positive.")
        object.__setattr__(self, "length_unit", normalize_length_unit(self.length_unit))
        if not all(
            0.0 < x < 180.0 for x in (self.alpha_deg, self.beta_deg, self.gamma_deg)
        ):
            raise ValueError("Cell angles must lie strictly between 0 and 180 degrees.")
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
        return _validated_spd(M, "lattice metric")

    def reciprocal_metric(self) -> np.ndarray:
        return np.linalg.solve(self.metric(), np.eye(3))

    def structure_matrix(self) -> np.ndarray:
        """Return one conventional Cartesian realization B with ``B.T @ B = M``."""
        a, b, c = self.a, self.b, self.c
        ca, cb, cg = _cosd(self.alpha_deg), _cosd(self.beta_deg), _cosd(self.gamma_deg)
        sg = _sind(self.gamma_deg)
        if abs(sg) <= 64.0 * _FLOAT_EPS:
            raise ValueError(
                "gamma is numerically too close to 0 or 180 degrees for the "
                "conventional structure matrix"
            )
        ax = a
        bx, by = b * cg, b * sg
        cx = c * cb
        cy = c * (ca - cb * cg) / sg
        cz2 = c * c - cx * cx - cy * cy
        if cz2 <= 0.0:
            # The lattice metric has already been required to be SPD.  If this
            # independently evaluated conventional-coordinate expression loses
            # positivity, do not repair it by max(0, cz2); report the numerical
            # representation failure.
            scale = max(c * c, cx * cx + cy * cy, np.finfo(float).tiny)
            raise ArithmeticError(
                "valid SPD lattice could not be represented safely by the "
                f"conventional structure-matrix formula; cz^2={cz2:g}, "
                f"relative={cz2 / scale:g}"
            )
        cz = float(np.sqrt(cz2))
        B = np.array([[ax, bx, cx], [0.0, by, cy], [0.0, 0.0, cz]], dtype=float)
        M = self.metric()
        residual = float(
            np.linalg.norm(B.T @ B - M, ord="fro")
            / max(np.linalg.norm(M, ord="fro"), np.finfo(float).tiny)
        )
        if residual > 2.0e-12:
            raise AssertionError(
                f"Structure matrix does not reproduce the metric; residual={residual:g}"
            )
        return B

    @classmethod
    def cubic(
        cls,
        a: float,
        label: str = "cubic",
        *,
        length_unit: str = "",
    ) -> Lattice:
        return cls(a=a, b=a, c=a, label=label, length_unit=length_unit)

    @classmethod
    def orthorhombic(
        cls,
        a: float,
        b: float,
        c: float,
        label: str = "orthorhombic",
        *,
        length_unit: str = "",
    ) -> Lattice:
        return cls(a=a, b=b, c=c, label=label, length_unit=length_unit)

    @classmethod
    def monoclinic_unique_b(
        cls,
        a: float,
        b: float,
        c: float,
        beta_deg: float,
        label: str = "monoclinic",
        *,
        length_unit: str = "",
    ) -> Lattice:
        return cls(
            a=a,
            b=b,
            c=c,
            alpha_deg=90.0,
            beta_deg=beta_deg,
            gamma_deg=90.0,
            label=label,
            length_unit=length_unit,
        )


def reciprocal_metric(M: np.ndarray) -> np.ndarray:
    A = _validated_spd(M)
    return np.linalg.solve(A, np.eye(3))


def metric_norm(v: np.ndarray, M: np.ndarray) -> float:
    q = _positive_quadratic_form(v, M, name="metric_norm")
    return float(np.sqrt(q))


def normalize_direct(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = metric_norm(v, M)
    if n <= 0.0:
        raise ValueError("Cannot normalize a zero direct-space vector.")
    return v / n


def plane_norm(p: np.ndarray, M: np.ndarray) -> float:
    """Norm of a reciprocal covector p using M^{-1}."""
    p = np.asarray(p, dtype=float).reshape(3)
    A = _validated_spd(M)
    x = np.linalg.solve(A, p)
    q = float(p @ x)
    if q < 0.0:
        raise ArithmeticError(f"plane_norm produced a negative reciprocal norm^2: {q:g}")
    return float(np.sqrt(q))


def normalize_plane(p: np.ndarray, M: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float).reshape(3)
    n = plane_norm(p, M)
    if n <= 0.0:
        raise ValueError("Cannot normalize a zero plane covector.")
    return p / n


def plane_to_unit_normal(p: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Convert a plane covector to a unit direct-space normal in crystal coordinates."""
    p_unit = normalize_plane(p, M)
    n = np.linalg.solve(_validated_spd(M), p_unit)
    return n


def direct_to_plane_normal(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Reciprocal covector whose physical normal is parallel to direct vector v."""
    return _validated_spd(M) @ np.asarray(v, dtype=float).reshape(3)


def metric_sqrt(M: np.ndarray) -> np.ndarray:
    """Unique principal symmetric positive square root of an SPD metric."""
    w, V = _spd_eigh(M)
    S = (V * np.sqrt(w)) @ V.T
    return 0.5 * (S + S.T)


def metric_inv_sqrt(M: np.ndarray) -> np.ndarray:
    """Unique principal symmetric inverse square root of an SPD metric."""
    w, V = _spd_eigh(M)
    W = (V * (1.0 / np.sqrt(w))) @ V.T
    return 0.5 * (W + W.T)


def to_metric_orthonormal_vector(u_crystal: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Map direct crystal coordinates to a symmetric metric-whitened orthonormal basis."""
    return metric_sqrt(M) @ np.asarray(u_crystal, dtype=float).reshape(3)


def from_metric_orthonormal_vector(u_hat: np.ndarray, M: np.ndarray) -> np.ndarray:
    return metric_inv_sqrt(M) @ np.asarray(u_hat, dtype=float).reshape(3)


def plane_normal_cartesian(p_crystal: np.ndarray, lattice: Lattice) -> np.ndarray:
    """Physical Cartesian unit normal of plane p=(hkl)."""
    B = lattice.structure_matrix()
    n = np.linalg.solve(B.T, np.asarray(p_crystal, dtype=float).reshape(3))
    n /= np.linalg.norm(n)
    return n


def direction_cartesian(u_crystal: np.ndarray, lattice: Lattice) -> np.ndarray:
    v = lattice.structure_matrix() @ np.asarray(u_crystal, dtype=float).reshape(3)
    v /= np.linalg.norm(v)
    return v


def transform_matrix_to_metric_orthonormal(
    g_crystal: np.ndarray, M: np.ndarray
) -> np.ndarray:
    """Represent a crystallographic map in the symmetric metric-whitened basis."""
    S = metric_sqrt(M)
    left = S @ np.asarray(g_crystal, dtype=float)
    return np.linalg.solve(S.T, left.T).T


def metric_dot(u: np.ndarray, v: np.ndarray, M: np.ndarray) -> float:
    u = np.asarray(u, dtype=float).reshape(3)
    v = np.asarray(v, dtype=float).reshape(3)
    return float(u @ _validated_spd(M) @ v)


def reciprocal_dot(p: np.ndarray, q: np.ndarray, M: np.ndarray) -> float:
    p = np.asarray(p, dtype=float).reshape(3)
    q = np.asarray(q, dtype=float).reshape(3)
    return float(p @ np.linalg.solve(_validated_spd(M), q))
