from __future__ import annotations

"""Vendor-neutral EBSD data model and orientation-convention boundary.

Internal orientation convention
-------------------------------
Every indexed point stores a proper rotation ``g`` with

    v_sample = g @ v_crystal_cartesian.

This is the same physical convention already used by :mod:`cualni_cryst.ebsd`.
Vendor Euler/matrix/quaternion conventions are converted exactly once at the
I/O boundary.  No downstream algorithm is allowed to guess a vendor convention.
"""

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .ebsd import (
    bunge_euler_to_g,
    crystal_symmetry_to_cartesian,
)
from .lattice import Lattice
from .orientation_kernel import require_so3, rotation_angle_deg
from .point_groups import point_group_operations


class MatrixDirection(str, Enum):
    CRYSTAL_TO_SAMPLE = "crystal_to_sample"
    SAMPLE_TO_CRYSTAL = "sample_to_crystal"


class AngleUnit(str, Enum):
    DEGREE = "degree"
    RADIAN = "radian"


@dataclass(frozen=True)
class OrientationConvention:
    """Explicit conversion contract for raw orientation data.

    ``scipy_sequence`` follows SciPy's convention: upper-case axes are
    intrinsic rotations, lower-case axes are extrinsic.  EBSD Bunge data are
    normally represented as intrinsic ``ZXZ``.

    ``raw_matrix_direction`` states what the rotation created from the raw
    Euler/quaternion/matrix means.  Corrections are then applied as

        g_internal = Q_sample @ g_raw_crystal_to_sample @ Q_crystal.T

    where both Q matrices are proper active Cartesian rotations.
    """

    scipy_sequence: str = "ZXZ"
    angle_unit: AngleUnit = AngleUnit.DEGREE
    raw_matrix_direction: MatrixDirection = MatrixDirection.CRYSTAL_TO_SAMPLE
    sample_correction: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    crystal_correction: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    label: str = "explicit"

    def __post_init__(self) -> None:
        if len(self.scipy_sequence) != 3:
            raise ValueError("Euler sequence must contain exactly three axes")
        sample = require_so3(
            np.asarray(self.sample_correction, dtype=float),
            tolerance=1.0e-9,
            name="sample_correction",
        )
        crystal = require_so3(
            np.asarray(self.crystal_correction, dtype=float),
            tolerance=1.0e-9,
            name="crystal_correction",
        )
        object.__setattr__(
            self,
            "sample_correction",
            tuple(tuple(float(x) for x in row) for row in sample),
        )
        object.__setattr__(
            self,
            "crystal_correction",
            tuple(tuple(float(x) for x in row) for row in crystal),
        )

    @classmethod
    def bunge_crystal_to_sample_degrees(
        cls, *, label: str = "Bunge ZXZ, crystal->sample, degrees"
    ) -> "OrientationConvention":
        return cls(
            scipy_sequence="ZXZ",
            angle_unit=AngleUnit.DEGREE,
            raw_matrix_direction=MatrixDirection.CRYSTAL_TO_SAMPLE,
            label=label,
        )

    @classmethod
    def bunge_crystal_to_sample_radians(
        cls, *, label: str = "Bunge ZXZ, crystal->sample, radians"
    ) -> "OrientationConvention":
        return cls(
            scipy_sequence="ZXZ",
            angle_unit=AngleUnit.RADIAN,
            raw_matrix_direction=MatrixDirection.CRYSTAL_TO_SAMPLE,
            label=label,
        )


def _as_rotation_stack(
    matrices: np.ndarray,
    *,
    name: str,
    repair_tolerance: float | None = 1.0e-10,
) -> np.ndarray:
    """Validate a stack of proper rotations; only roundoff-scale repair is allowed."""

    arr = np.asarray(matrices, dtype=float)
    if arr.ndim == 2:
        arr = arr.reshape(1, 3, 3)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        raise ValueError(f"{name} must have shape (N,3,3); got {arr.shape}")

    out = np.empty_like(arr)
    for index, matrix in enumerate(arr):
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name}[{index}] contains non-finite values")
        residual = max(
            float(np.linalg.norm(matrix.T @ matrix - np.eye(3), ord="fro")),
            abs(float(np.linalg.det(matrix)) - 1.0),
        )
        if residual <= 1.0e-12:
            out[index] = matrix
            continue
        if repair_tolerance is not None and residual <= repair_tolerance:
            U, _, Vt = np.linalg.svd(matrix)
            correction = np.eye(3)
            correction[2, 2] = np.sign(np.linalg.det(U @ Vt))
            repaired = U @ correction @ Vt
            require_so3(repaired, tolerance=1.0e-10, name=f"{name}[{index}]")
            out[index] = repaired
            continue
        raise ValueError(
            f"{name}[{index}] is not in SO(3); residual={residual:.3e}. "
            "A non-roundoff orientation error is never repaired silently."
        )
    return out


def convert_raw_matrices(
    matrices: np.ndarray,
    convention: OrientationConvention,
) -> np.ndarray:
    raw = _as_rotation_stack(matrices, name="raw orientation matrices")
    if convention.raw_matrix_direction == MatrixDirection.SAMPLE_TO_CRYSTAL:
        raw = np.transpose(raw, (0, 2, 1))

    Qs = np.asarray(convention.sample_correction, dtype=float)
    Qc = np.asarray(convention.crystal_correction, dtype=float)
    converted = np.einsum("ab,nbc,cd->nad", Qs, raw, Qc.T)
    return _as_rotation_stack(converted, name="converted orientation matrices")


def eulers_to_matrices(
    eulers: np.ndarray,
    convention: OrientationConvention,
) -> np.ndarray:
    values = np.asarray(eulers, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, 3)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Euler array must have shape (N,3); got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Euler array contains non-finite values")

    degrees = convention.angle_unit == AngleUnit.DEGREE
    raw = Rotation.from_euler(
        convention.scipy_sequence,
        values,
        degrees=degrees,
    ).as_matrix()
    return convert_raw_matrices(raw, convention)


def quaternions_to_matrices(
    quaternions: np.ndarray,
    convention: OrientationConvention,
    *,
    order: str = "wxyz",
) -> np.ndarray:
    values = np.asarray(quaternions, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, 4)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(
            f"Quaternion array must have shape (N,4); got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Quaternion array contains non-finite values")

    order_key = order.lower()
    if order_key == "wxyz":
        xyzw = values[:, [1, 2, 3, 0]]
    elif order_key == "xyzw":
        xyzw = values
    else:
        raise ValueError("quaternion order must be 'wxyz' or 'xyzw'")

    norms = np.linalg.norm(xyzw, axis=1)
    if np.any(norms <= 1.0e-15):
        raise ValueError("zero quaternion is invalid")
    xyzw = xyzw / norms[:, None]
    raw = Rotation.from_quat(xyzw).as_matrix()
    return convert_raw_matrices(raw, convention)


@dataclass(frozen=True)
class EBSDPhase:
    phase_id: int
    name: str
    lattice: Lattice
    point_group: str
    proper_symmetry_cartesian: tuple[np.ndarray, ...]

    @classmethod
    def from_point_group(
        cls,
        phase_id: int,
        name: str,
        lattice: Lattice,
        point_group: str,
    ) -> "EBSDPhase":
        if phase_id <= 0:
            raise ValueError("indexed phase IDs must be positive integers")
        exact = point_group_operations(point_group)
        cart = crystal_symmetry_to_cartesian(
            [np.asarray(item, dtype=float) for item in exact],
            lattice,
        )
        if not cart:
            raise ValueError("phase has no proper rotational symmetry")
        audited = tuple(
            require_so3(item, tolerance=2.0e-8, name="phase symmetry")
            for item in cart
        )
        return cls(
            phase_id=int(phase_id),
            name=str(name),
            lattice=lattice,
            point_group=point_group,
            proper_symmetry_cartesian=audited,
        )


@dataclass(frozen=True)
class EBSDMap:
    """Vendor-neutral point cloud / 2-D / 3-D EBSD map."""

    orientations: np.ndarray
    phase_id: np.ndarray
    indexed: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    quality: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        phase = np.asarray(self.phase_id)
        indexed = np.asarray(self.indexed, dtype=bool)
        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        z = np.asarray(self.z, dtype=float)
        n = len(phase)

        for name, array in (
            ("indexed", indexed),
            ("x", x),
            ("y", y),
            ("z", z),
        ):
            if len(array) != n:
                raise ValueError(f"{name} length {len(array)} != phase length {n}")

        if phase.ndim != 1:
            raise ValueError("phase_id must be one-dimensional")
        if not np.issubdtype(phase.dtype, np.integer):
            if np.any(np.asarray(phase, dtype=float) != np.asarray(phase, dtype=int)):
                raise ValueError("phase_id must contain integers")
            phase = np.asarray(phase, dtype=int)
        else:
            phase = phase.astype(int, copy=False)

        if np.any(indexed & (phase <= 0)):
            raise ValueError("indexed EBSD points must have positive phase IDs")
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(~np.isfinite(z)):
            raise ValueError("EBSD coordinates must be finite")

        rotations = np.asarray(self.orientations, dtype=float)
        if rotations.shape != (n, 3, 3):
            raise ValueError(
                f"orientations must have shape ({n},3,3); got {rotations.shape}"
            )

        # Unindexed points may intentionally store NaNs. Indexed points may not.
        cleaned = rotations.copy()
        if np.any(indexed):
            cleaned[indexed] = _as_rotation_stack(
                cleaned[indexed],
                name="indexed EBSD orientations",
            )
        if np.any(~indexed):
            invalid_rows = cleaned[~indexed]
            finite_invalid = np.all(np.isfinite(invalid_rows), axis=(1, 2))
            if np.any(finite_invalid):
                # A finite orientation at an explicitly unindexed point is
                # preserved, but it is never used unless the user changes mask.
                _ = _as_rotation_stack(
                    invalid_rows[finite_invalid],
                    name="finite unindexed orientations",
                )

        qdict: dict[str, np.ndarray] = {}
        for key, values in dict(self.quality).items():
            arr = np.asarray(values)
            if len(arr) != n:
                raise ValueError(
                    f"quality field {key!r} length {len(arr)} != {n}"
                )
            qdict[str(key)] = arr.copy()

        object.__setattr__(self, "orientations", cleaned)
        object.__setattr__(self, "phase_id", phase)
        object.__setattr__(self, "indexed", indexed)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "quality", MappingProxyType(qdict))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    @property
    def n_points(self) -> int:
        return int(len(self.phase_id))

    @property
    def coordinates(self) -> np.ndarray:
        return np.column_stack((self.x, self.y, self.z))

    @classmethod
    def from_eulers(
        cls,
        eulers: np.ndarray,
        phase_id: Sequence[int],
        x: Sequence[float],
        y: Sequence[float],
        *,
        convention: OrientationConvention,
        z: Sequence[float] | None = None,
        indexed: Sequence[bool] | None = None,
        quality: Mapping[str, np.ndarray] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> "EBSDMap":
        phase = np.asarray(phase_id, dtype=int)
        n = len(phase)
        if z is None:
            z = np.zeros(n, dtype=float)
        if indexed is None:
            indexed = phase > 0

        matrices = np.full((n, 3, 3), np.nan, dtype=float)
        mask = np.asarray(indexed, dtype=bool)
        euler_array = np.asarray(eulers, dtype=float)
        if euler_array.shape != (n, 3):
            raise ValueError(f"eulers must have shape ({n},3)")
        if np.any(mask):
            matrices[mask] = eulers_to_matrices(euler_array[mask], convention)

        return cls(
            orientations=matrices,
            phase_id=phase,
            indexed=mask,
            x=np.asarray(x, dtype=float),
            y=np.asarray(y, dtype=float),
            z=np.asarray(z, dtype=float),
            quality={} if quality is None else quality,
            metadata={} if metadata is None else metadata,
        )


@dataclass(frozen=True)
class MapAudit:
    n_points: int
    n_indexed: int
    indexed_fraction: float
    phase_counts: Mapping[int, int]
    duplicate_coordinate_pairs: int
    maximum_so3_residual: float
    coordinate_dimension: int
    nearest_neighbor_median: float | None
    nearest_neighbor_minimum: float | None


def audit_map(data: EBSDMap) -> MapAudit:
    from scipy.spatial import cKDTree

    indexed_count = int(np.count_nonzero(data.indexed))
    counts = {
        int(phase): int(np.count_nonzero(data.phase_id == phase))
        for phase in np.unique(data.phase_id)
    }

    coords = data.coordinates
    dimension = 3 if np.ptp(data.z) > 0 else 2
    active_coords = coords[:, :dimension]

    duplicate_pairs = 0
    nn_median: float | None = None
    nn_min: float | None = None
    if data.n_points > 1:
        tree = cKDTree(active_coords)
        distances, _ = tree.query(active_coords, k=2)
        nearest = np.asarray(distances[:, 1], dtype=float)
        duplicate_pairs = int(np.count_nonzero(nearest <= 1.0e-12) // 2)
        positive = nearest[nearest > 1.0e-12]
        if len(positive):
            nn_median = float(np.median(positive))
            nn_min = float(np.min(positive))

    worst = 0.0
    for matrix in data.orientations[data.indexed]:
        worst = max(
            worst,
            float(np.linalg.norm(matrix.T @ matrix - np.eye(3), ord="fro")),
            abs(float(np.linalg.det(matrix)) - 1.0),
        )

    return MapAudit(
        n_points=data.n_points,
        n_indexed=indexed_count,
        indexed_fraction=(
            float(indexed_count / data.n_points) if data.n_points else 0.0
        ),
        phase_counts=MappingProxyType(counts),
        duplicate_coordinate_pairs=duplicate_pairs,
        maximum_so3_residual=worst,
        coordinate_dimension=dimension,
        nearest_neighbor_median=nn_median,
        nearest_neighbor_minimum=nn_min,
    )


def assert_bunge_adapter_parity() -> None:
    """Cross-lock the new vectorized converter to the existing package adapter."""

    test = np.array(
        [
            [0.0, 0.0, 0.0],
            [13.0, 27.0, 41.0],
            [250.0, 89.5, 17.0],
        ],
        dtype=float,
    )
    convention = OrientationConvention.bunge_crystal_to_sample_degrees()
    vectorized = eulers_to_matrices(test, convention)
    for index, angles in enumerate(test):
        legacy = bunge_euler_to_g(*angles)
        if rotation_angle_deg(vectorized[index] @ legacy.T) > 1.0e-9:
            raise AssertionError("Bunge adapter parity check failed")
