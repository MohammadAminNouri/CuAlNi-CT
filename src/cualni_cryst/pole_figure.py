
from __future__ import annotations

"""Metric-correct stereographic pole data for theory/experiment overlays.

The backend returns geometry only.  Plotting is deliberately left to the UI so
scientific calculations do not depend on a graphics library.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .lattice import metric_sqrt
from .orientation_kernel import require_so3

PoleKind = Literal["direction", "plane"]


@dataclass(frozen=True)
class PolePoint:
    x: float
    y: float
    z: float
    cartesian_reference: tuple[float, float, float]
    crystal_coefficients: tuple[float, float, float]
    label: str
    phase_id: str
    kind: PoleKind
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "cartesian_reference": list(self.cartesian_reference),
            "crystal_coefficients": list(self.crystal_coefficients),
            "label": self.label,
            "phase_id": self.phase_id,
            "kind": self.kind,
            "source": self.source,
        }


@dataclass(frozen=True)
class PoleFamily:
    label: str
    phase_id: str
    kind: PoleKind
    projective: bool
    points: tuple[PolePoint, ...]
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "phase_id": self.phase_id,
            "kind": self.kind,
            "projective": self.projective,
            "source": self.source,
            "points": [point.to_dict() for point in self.points],
        }


@dataclass(frozen=True)
class PoleFigureOverlay:
    reference_phase_id: str
    families: tuple[PoleFamily, ...]

    def table_rows(self) -> list[dict[str, Any]]:
        rows = []
        for family in self.families:
            for point in family.points:
                rows.append(point.to_dict())
        return rows


def _unit(value: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-15:
        raise ValueError(f"{name} must be nonzero")
    return vector / norm


def _canonical_projective(vector: np.ndarray) -> np.ndarray:
    value = _unit(vector, name="pole")
    # Pole figures use one representative of an unoriented line/normal.
    if value[2] < -1.0e-14:
        value = -value
    elif abs(value[2]) <= 1.0e-14:
        for component in value[:2]:
            if abs(component) > 1.0e-14:
                if component < 0.0:
                    value = -value
                break
    value[np.abs(value) < 1.0e-14] = 0.0
    return value


def stereographic_project(
    vector: np.ndarray,
    *,
    projective: bool = True,
) -> tuple[float, float, float]:
    """Project a unit vector from the upper hemisphere to the equatorial plane.

    For projective crystallographic poles, ``v`` and ``-v`` are identified and
    the upper-hemisphere representative is selected first.
    """

    value = (
        _canonical_projective(vector)
        if projective
        else _unit(vector, name="pole")
    )
    denominator = 1.0 + float(value[2])
    if denominator <= 1.0e-14:
        raise ValueError(
            "Oriented south-pole vector is singular in this stereographic chart"
        )
    return (
        float(value[0] / denominator),
        float(value[1] / denominator),
        float(value[2]),
    )


def _physical_vector(
    lattice: Any,
    coefficients: np.ndarray,
    *,
    kind: PoleKind,
) -> np.ndarray:
    B = np.asarray(metric_sqrt(lattice.metric()), dtype=float)
    if kind == "direction":
        return _unit(B @ coefficients, name="physical direction")
    if kind == "plane":
        return _unit(
            np.linalg.solve(B.T, coefficients),
            name="physical plane normal",
        )
    raise ValueError(f"Unknown pole kind {kind!r}")


def _equivalent_coefficients(
    coefficients: np.ndarray,
    symmetry: tuple[np.ndarray, ...],
    *,
    kind: PoleKind,
    projective: bool,
) -> list[np.ndarray]:
    raw = []
    for operation in symmetry:
        g = np.asarray(operation, dtype=float).reshape(3, 3)
        if kind == "direction":
            candidate = g @ coefficients
        else:
            candidate = np.linalg.solve(g.T, coefficients)
        raw.append(candidate)

    unique: dict[tuple[float, float, float], np.ndarray] = {}
    for item in raw:
        value = np.asarray(item, dtype=float).reshape(3)
        scale = float(np.max(np.abs(value)))
        if scale <= 1.0e-15:
            continue
        value = value / scale
        if projective:
            nz = np.flatnonzero(np.abs(value) > 1.0e-12)
            if len(nz) and value[int(nz[0])] < 0.0:
                value = -value
        key = tuple(float(x) for x in np.round(value, 10))
        unique.setdefault(key, np.asarray(item, dtype=float).reshape(3))
    return list(unique.values())


def pole_family(
    project: Any,
    phase_id: str,
    *,
    kind: PoleKind,
    coefficients: tuple[float, float, float] | list[float] | np.ndarray,
    reference_phase_id: str | None = None,
    R_reference_from_phase: np.ndarray | None = None,
    projective: bool = True,
    label: str = "",
    source: str = "",
) -> PoleFamily:
    """Generate a full symmetry-equivalent pole family in one reference frame.

    If ``phase_id`` differs from ``reference_phase_id``, an explicit physical
    rotation ``R_reference_from_phase`` is required.  A correspondence matrix is
    never substituted for this rotation.
    """

    phase = project.phase(phase_id)
    reference = phase_id if reference_phase_id is None else str(reference_phase_id)
    if reference not in {item.phase_id for item in project.phases}:
        raise KeyError(f"Unknown reference phase {reference!r}")

    if phase_id == reference:
        R = np.eye(3)
    else:
        if R_reference_from_phase is None:
            raise ValueError(
                "An explicit physical orientation is required to overlay poles "
                "from different phases"
            )
        R = require_so3(
            np.asarray(R_reference_from_phase, dtype=float),
            tolerance=2.0e-8,
            name="pole overlay orientation",
        )

    initial = np.asarray(coefficients, dtype=float).reshape(3)
    if not np.all(np.isfinite(initial)) or float(np.linalg.norm(initial)) <= 1.0e-15:
        raise ValueError("Pole coefficients must be finite and nonzero")

    equivalent = _equivalent_coefficients(
        initial,
        phase.symmetry_matrices(),
        kind=kind,
        projective=projective,
    )
    points: list[PolePoint] = []
    seen_cart: set[tuple[float, float, float]] = set()
    for item in equivalent:
        physical = R @ _physical_vector(phase.lattice, item, kind=kind)
        display = _canonical_projective(physical) if projective else _unit(
            physical, name="oriented pole"
        )
        key = tuple(float(x) for x in np.round(display, 10))
        if key in seen_cart:
            continue
        seen_cart.add(key)
        x, y, z = stereographic_project(display, projective=projective)
        scale = float(np.max(np.abs(item)))
        crystal = item / scale
        points.append(
            PolePoint(
                x=x,
                y=y,
                z=z,
                cartesian_reference=tuple(float(v) for v in display),
                crystal_coefficients=tuple(float(v) for v in crystal),
                label=label or f"{kind} {tuple(float(v) for v in initial)}",
                phase_id=phase_id,
                kind=kind,
                source=source,
            )
        )

    return PoleFamily(
        label=label or f"{phase_id} {kind}",
        phase_id=phase_id,
        kind=kind,
        projective=projective,
        points=tuple(points),
        source=source,
    )


def overlay(*families: PoleFamily, reference_phase_id: str) -> PoleFigureOverlay:
    return PoleFigureOverlay(
        reference_phase_id=reference_phase_id,
        families=tuple(families),
    )
