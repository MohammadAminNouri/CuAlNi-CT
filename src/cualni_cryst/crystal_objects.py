from __future__ import annotations

"""Typed crystallographic objects for direct and reciprocal space.

A crystallographic direction [uvw] is a direct-space vector.
A crystallographic plane (hkl) is a reciprocal-space covector.

They are dual objects.  They are intentionally represented by different Python
types so that the API cannot silently feed a plane into a direction operation
or vice versa.

Stored indices are never silently normalized or converted.  Metric-normalized
and Cartesian forms are derived views of the same immutable object.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from .lattice import (
    Lattice,
    metric_dot,
    metric_norm,
    normalize_direct,
    normalize_plane,
    plane_norm,
    plane_to_unit_normal,
    reciprocal_dot,
)
from .numerics import DEFAULT_NUMERICAL_POLICY
from .provenance import DataStatus
from .representation import CartesianConvention, CartesianFrame

Index3 = tuple[float, float, float]


def _validated_indices(values: Iterable[float]) -> Index3:
    arr = np.asarray(tuple(values), dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"Crystallographic indices must contain exactly 3 values; got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Crystallographic indices must be finite")
    if float(np.linalg.norm(arr)) <= 0.0:
        raise ValueError("The zero index triplet is not a valid direction or plane")
    return tuple(float(x) for x in arr)


def _matrix3(g: np.ndarray) -> np.ndarray:
    arr = np.asarray(g, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"Crystallographic operator must be 3x3; got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Crystallographic operator contains non-finite values")
    if abs(float(np.linalg.det(arr))) <= 1.0e-14:
        raise ValueError("Crystallographic operator must be invertible")
    return arr


def _unit_interval_clip(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def _projective_key(v: np.ndarray, *, projective: bool, decimals: int = 12) -> tuple[float, ...]:
    """Scale-invariant key; optionally identify v and -v."""

    x = np.asarray(v, dtype=float).reshape(3)
    scale = float(np.max(np.abs(x)))
    if scale <= 0.0:
        raise ValueError("Cannot canonicalize the zero vector")
    x = x / scale

    if projective:
        nz = np.flatnonzero(np.abs(x) > 10.0 ** (-decimals))
        if len(nz) and x[int(nz[0])] < 0.0:
            x = -x

    return tuple(float(y) for y in np.round(x, decimals))


@dataclass(frozen=True)
class CrystalBasisRef:
    """Identity of the crystallographic coordinate basis used by an object.

    This stores *identity*, not lattice parameters.  The future ProjectState
    resolves this reference to the corresponding Lattice and symmetry model.
    """

    phase_id: str
    basis_id: str = "conventional"
    cell_representation: str = ""

    def __post_init__(self) -> None:
        if not self.phase_id.strip():
            raise ValueError("phase_id must be non-empty")
        if not self.basis_id.strip():
            raise ValueError("basis_id must be non-empty")


@dataclass(frozen=True)
class ObjectProvenance:
    """Minimal provenance attached to a crystallographic object."""

    status: DataStatus = DataStatus.UNVERIFIED
    source_key: str = ""
    uncertainty: str = ""
    notes: str = ""


@dataclass(frozen=True)
class Direction:
    """Immutable direct-space crystallographic direction [uvw]."""

    indices: Index3
    basis: CrystalBasisRef
    label: str = ""
    provenance: ObjectProvenance = field(default_factory=ObjectProvenance)

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", _validated_indices(self.indices))

    @property
    def array(self) -> np.ndarray:
        return np.asarray(self.indices, dtype=float)

    def length(self, lattice: Lattice) -> float:
        """Physical direct-vector length sqrt(u^T M u)."""

        return metric_norm(self.array, lattice.metric())

    def unit_coordinates(self, lattice: Lattice) -> np.ndarray:
        """Metric-unit direct coordinates; stored indices remain unchanged."""

        return normalize_direct(self.array, lattice.metric())

    def cartesian(
        self,
        lattice: Lattice,
        convention: CartesianConvention,
        *,
        normalize: bool = False,
    ) -> np.ndarray:
        return CartesianFrame(lattice, convention).direct_to_cartesian(
            self.array,
            normalize=normalize,
        )

    def transformed(self, direct_operator: np.ndarray) -> Direction:
        """Apply a direct-space crystallographic operator u' = g u."""

        g = _matrix3(direct_operator)
        return Direction(
            tuple(g @ self.array),
            self.basis,
            label=self.label,
            provenance=self.provenance,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "direction",
            "notation": f"[{self.indices[0]:g} {self.indices[1]:g} {self.indices[2]:g}]",
            "indices": list(self.indices),
            "phase_id": self.basis.phase_id,
            "basis_id": self.basis.basis_id,
            "cell_representation": self.basis.cell_representation,
            "label": self.label,
            "status": self.provenance.status.value,
            "source_key": self.provenance.source_key,
            "uncertainty": self.provenance.uncertainty,
            "notes": self.provenance.notes,
        }


@dataclass(frozen=True)
class Plane:
    """Immutable reciprocal-space crystallographic plane covector (hkl)."""

    indices: Index3
    basis: CrystalBasisRef
    label: str = ""
    provenance: ObjectProvenance = field(default_factory=ObjectProvenance)

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", _validated_indices(self.indices))

    @property
    def array(self) -> np.ndarray:
        return np.asarray(self.indices, dtype=float)

    def reciprocal_length(self, lattice: Lattice) -> float:
        """Physical reciprocal-vector magnitude sqrt(p^T M^-1 p)."""

        return plane_norm(self.array, lattice.metric())

    def spacing(self, lattice: Lattice) -> float:
        """Interplanar spacing d=1/|g| in the package's no-2π convention."""

        g = self.reciprocal_length(lattice)
        if g <= 0.0:
            raise ValueError("Plane spacing is undefined for zero reciprocal length")
        return 1.0 / g

    def unit_covector(self, lattice: Lattice) -> np.ndarray:
        """Metric-unit reciprocal coordinates; stored indices remain unchanged."""

        return normalize_plane(self.array, lattice.metric())

    def unit_normal_direct_coordinates(self, lattice: Lattice) -> np.ndarray:
        """Unit physical plane normal expressed in direct crystal coordinates."""

        return plane_to_unit_normal(self.array, lattice.metric())

    def cartesian_normal(
        self,
        lattice: Lattice,
        convention: CartesianConvention,
        *,
        normalize: bool = False,
    ) -> np.ndarray:
        return CartesianFrame(lattice, convention).plane_to_cartesian(
            self.array,
            normalize=normalize,
        )

    def transformed(self, direct_operator: np.ndarray) -> Plane:
        """Apply dual action p' = g^-T p for a direct-space operator g."""

        g = _matrix3(direct_operator)
        p_new = np.linalg.inv(g).T @ self.array
        return Plane(
            tuple(p_new),
            self.basis,
            label=self.label,
            provenance=self.provenance,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "plane",
            "notation": f"({self.indices[0]:g} {self.indices[1]:g} {self.indices[2]:g})",
            "indices": list(self.indices),
            "phase_id": self.basis.phase_id,
            "basis_id": self.basis.basis_id,
            "cell_representation": self.basis.cell_representation,
            "label": self.label,
            "status": self.provenance.status.value,
            "source_key": self.provenance.source_key,
            "uncertainty": self.provenance.uncertainty,
            "notes": self.provenance.notes,
        }


def _require_same_basis(a: Direction | Plane, b: Direction | Plane) -> None:
    if a.basis != b.basis:
        raise ValueError(
            "Objects are expressed in different crystallographic bases. "
            "An explicit orientation/correspondence transformation is required "
            "before comparing them."
        )


def direction_angle_deg(a: Direction, b: Direction, lattice: Lattice) -> float:
    """Oriented angle in [0,180] between two direct directions.

    Evaluate the metric angle with ``atan2(||x×y||, x·y)`` rather than
    ``acos(cos θ)``. For nearly parallel or antiparallel vectors this avoids
    the endpoint loss of significance of ``acos`` while preserving the exact
    metric definition. If ``M=L L^T`` then ``x=L^T u`` satisfies
    ``x·y = u^T M v``.
    """

    _require_same_basis(a, b)
    M = lattice.metric()
    L = np.linalg.cholesky(M)
    x = L.T @ a.array
    y = L.T @ b.array
    sine_numerator = float(np.linalg.norm(np.cross(x, y)))
    cosine_numerator = float(x @ y)
    return float(np.rad2deg(np.arctan2(sine_numerator, cosine_numerator)))


def axis_angle_deg(a: Direction, b: Direction, lattice: Lattice) -> float:
    """Projective acute angle in [0,90], identifying u and -u."""

    angle = direction_angle_deg(a, b, lattice)
    return float(min(angle, 180.0 - angle))


def plane_normal_angle_deg(a: Plane, b: Plane, lattice: Lattice) -> float:
    """Oriented angle in [0,180] between reciprocal plane normals.

    With ``M=L L^T``, reciprocal Cartesian representatives may be taken as
    ``x=L^-1 p`` because ``x·y = p^T M^-1 q``. The ``atan2`` form is stable
    at the projective endpoints and returns exactly 0/180 for exactly
    proportional binary64 index vectors such as ``(hkl)`` and ``(-h-k-l)``.
    """

    _require_same_basis(a, b)
    M = lattice.metric()
    L = np.linalg.cholesky(M)
    x = np.linalg.solve(L, a.array)
    y = np.linalg.solve(L, b.array)
    sine_numerator = float(np.linalg.norm(np.cross(x, y)))
    cosine_numerator = float(x @ y)
    return float(np.rad2deg(np.arctan2(sine_numerator, cosine_numerator)))


def interplanar_angle_deg(a: Plane, b: Plane, lattice: Lattice) -> float:
    """Acute crystallographic plane angle in [0,90], identifying ± normals."""

    angle = plane_normal_angle_deg(a, b, lattice)
    return float(min(angle, 180.0 - angle))


def incidence_residual(direction: Direction, plane: Plane, lattice: Lattice) -> float:
    """Dimensionless |p^T u|/(|u|_M |p|_M^-1).

    Zero means the direction lies in the plane.  The quantity is exactly the
    absolute cosine between the physical direction and plane normal.
    """

    _require_same_basis(direction, plane)
    M = lattice.metric()
    denominator = metric_norm(direction.array, M) * plane_norm(plane.array, M)
    return float(abs(plane.array @ direction.array) / denominator)


def direction_plane_angle_deg(
    direction: Direction,
    plane: Plane,
    lattice: Lattice,
) -> float:
    """Acute angle in [0,90] between a direction and a plane.

    0° means the direction lies in the plane.
    90° means it is parallel to the plane normal.
    """

    residual = incidence_residual(direction, plane, lattice)
    return float(np.rad2deg(np.arcsin(np.clip(residual, 0.0, 1.0))))


def lies_in_plane(
    direction: Direction,
    plane: Plane,
    lattice: Lattice,
    *,
    tolerance: float = DEFAULT_NUMERICAL_POLICY.algebraic,
) -> bool:
    """Numerical incidence test with an explicit dimensionless tolerance."""

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    return bool(incidence_residual(direction, plane, lattice) <= tolerance)


def equivalent_directions(
    direction: Direction,
    direct_symmetry_operators: Iterable[np.ndarray],
    *,
    projective: bool = False,
) -> tuple[Direction, ...]:
    """Generate unique symmetry-equivalent directions.

    ``projective=False`` preserves opposite senses [uvw] and [-u-v-w] as
    distinct directions.  ``projective=True`` treats them as one unoriented
    axis.  This choice is explicit because the correct interpretation depends
    on the scientific task.
    """

    unique: dict[tuple[float, ...], Direction] = {}
    for g in direct_symmetry_operators:
        item = direction.transformed(np.asarray(g, dtype=float))
        key = _projective_key(item.array, projective=projective)
        unique.setdefault(key, item)
    return tuple(unique.values())


def equivalent_planes(
    plane: Plane,
    direct_symmetry_operators: Iterable[np.ndarray],
    *,
    projective: bool = True,
) -> tuple[Plane, ...]:
    """Generate unique symmetry-equivalent reciprocal planes.

    Planes default to projective equivalence because (hkl) and (-h-k-l)
    describe the same unoriented lattice-plane family.  Set
    ``projective=False`` when plane-normal sense matters.
    """

    unique: dict[tuple[float, ...], Plane] = {}
    for g in direct_symmetry_operators:
        item = plane.transformed(np.asarray(g, dtype=float))
        key = _projective_key(item.array, projective=projective)
        unique.setdefault(key, item)
    return tuple(unique.values())
