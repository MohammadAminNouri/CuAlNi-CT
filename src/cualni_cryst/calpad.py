from __future__ import annotations

"""PTCLab-style crystallography workbench built on CuAlNi-CT's typed core.

This module adds the first Calpad-parity layer without duplicating the
crystallographic mathematics already verified elsewhere in the package.

Scientific rules
----------------
- direct directions [uvw] and reciprocal planes (hkl) remain distinct types;
- non-cubic plane-normal conversion uses the metric, never the cubic shortcut;
- low-index searches are ranked with the actual direct/reciprocal metric;
- approximate low-index representatives are labelled as approximations and
  carry their angular mismatch;
- all dimensional output respects the lattice's explicit unit metadata.
"""

import argparse
import json
import re
from dataclasses import dataclass
from enum import Enum
from itertools import product
from math import gcd
from typing import ClassVar

import numpy as np

from .crystal_objects import (
    Direction,
    Plane,
    axis_angle_deg,
    direction_angle_deg,
    direction_plane_angle_deg,
    incidence_residual,
    interplanar_angle_deg,
    plane_normal_angle_deg,
)
from .crystallography_console import (
    ConsoleInputKind,
    ConsoleRenderer,
    CrystallographyConsole,
    DerivationStep,
    ParsedCrystalInput,
    parse_crystal_input,
)
from .project_io import load_project
from .project_state import PhaseState, ProjectState
from .representation import CartesianConvention, CartesianFrame
from .units import length_unit_symbol, reciprocal_length_unit_symbol

CrystalObject = Direction | Plane
Index3i = tuple[int, int, int]
Index3f = tuple[float, float, float]


class CandidateKind(str, Enum):
    DIRECTION = "direction"
    PLANE = "plane"


class AngleSense(str, Enum):
    PROJECTIVE = "projective"
    ORIENTED = "oriented"


@dataclass(frozen=True)
class ReciprocalCell:
    a_star: float
    b_star: float
    c_star: float
    alpha_star_deg: float
    beta_star_deg: float
    gamma_star_deg: float
    volume_star: float

    def to_dict(self) -> dict[str, float]:
        return {
            "a_star": self.a_star,
            "b_star": self.b_star,
            "c_star": self.c_star,
            "alpha_star_deg": self.alpha_star_deg,
            "beta_star_deg": self.beta_star_deg,
            "gamma_star_deg": self.gamma_star_deg,
            "volume_star": self.volume_star,
        }


@dataclass(frozen=True)
class PhaseCellReport:
    phase_id: str
    label: str
    physical_phase: str
    cell_representation: str
    basis_id: str
    point_group_symbol: str
    symmetry_order: int
    length_unit: str
    a: float
    b: float
    c: float
    alpha_deg: float
    beta_deg: float
    gamma_deg: float
    volume: float
    direct_metric: np.ndarray
    reciprocal_metric: np.ndarray
    reciprocal_cell: ReciprocalCell
    provenance_status: str
    provenance_source_key: str
    provenance_uncertainty: str
    provenance_notes: str
    metric_inverse_residual: float

    def to_dict(self) -> dict[str, object]:
        return {
            "phase_id": self.phase_id,
            "label": self.label,
            "physical_phase": self.physical_phase,
            "cell_representation": self.cell_representation,
            "basis_id": self.basis_id,
            "point_group_symbol": self.point_group_symbol,
            "symmetry_order": self.symmetry_order,
            "length_unit": self.length_unit,
            "direct_cell": {
                "a": self.a,
                "b": self.b,
                "c": self.c,
                "alpha_deg": self.alpha_deg,
                "beta_deg": self.beta_deg,
                "gamma_deg": self.gamma_deg,
                "volume": self.volume,
            },
            "direct_metric": self.direct_metric.tolist(),
            "reciprocal_metric": self.reciprocal_metric.tolist(),
            "reciprocal_cell": self.reciprocal_cell.to_dict(),
            "provenance": {
                "status": self.provenance_status,
                "source_key": self.provenance_source_key,
                "uncertainty": self.provenance_uncertainty,
                "notes": self.provenance_notes,
            },
            "metric_inverse_residual": self.metric_inverse_residual,
        }


@dataclass(frozen=True)
class CartesianUnitView:
    convention: str
    coordinates: Index3f

    def to_dict(self) -> dict[str, object]:
        return {
            "convention": self.convention,
            "coordinates": list(self.coordinates),
        }


@dataclass(frozen=True)
class NearestLowIndex:
    notation: str
    indices: Index3i
    angular_mismatch_deg: float
    search_max_index: int
    exact_within_numerical_tolerance: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "notation": self.notation,
            "indices": list(self.indices),
            "angular_mismatch_deg": self.angular_mismatch_deg,
            "search_max_index": self.search_max_index,
            "exact_within_numerical_tolerance": (self.exact_within_numerical_tolerance),
        }


@dataclass(frozen=True)
class NormalConversionReport:
    phase_id: str
    source: ParsedCrystalInput
    target_kind: CandidateKind
    relation: str
    raw_target_coefficients: Index3f
    projective_target_coefficients: Index3f
    metric_unit_target_coefficients: Index3f
    cartesian_unit_views: tuple[CartesianUnitView, ...]
    nearest_low_index: NearestLowIndex
    roundtrip_projective_residual: float
    derivation: tuple[DerivationStep, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "phase_id": self.phase_id,
            "source": self.source.to_dict(),
            "target_kind": self.target_kind.value,
            "relation": self.relation,
            "raw_target_coefficients": list(self.raw_target_coefficients),
            "projective_target_coefficients": list(self.projective_target_coefficients),
            "metric_unit_target_coefficients": list(
                self.metric_unit_target_coefficients
            ),
            "cartesian_unit_views": [
                view.to_dict() for view in self.cartesian_unit_views
            ],
            "nearest_low_index": self.nearest_low_index.to_dict(),
            "roundtrip_projective_residual": self.roundtrip_projective_residual,
            "derivation": [step.to_dict() for step in self.derivation],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class LowIndexRow:
    rank: int
    notation: str
    indices: Index3i
    angle_deg: float
    relation: str
    geometric_quantity_name: str
    geometric_quantity: float
    geometric_quantity_unit: str
    incidence_residual_value: float | None
    incidence_exact: bool | None

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "notation": self.notation,
            "indices": list(self.indices),
            "angle_deg": self.angle_deg,
            "relation": self.relation,
            "geometric_quantity_name": self.geometric_quantity_name,
            "geometric_quantity": self.geometric_quantity,
            "geometric_quantity_unit": self.geometric_quantity_unit,
            "incidence_residual": self.incidence_residual_value,
            "incidence_exact": self.incidence_exact,
        }


@dataclass(frozen=True)
class LowIndexTableReport:
    phase_id: str
    target: ParsedCrystalInput
    candidate_kind: CandidateKind
    angle_sense: AngleSense
    max_index: int
    limit: int
    total_candidates: int
    rows: tuple[LowIndexRow, ...]
    derivation: tuple[DerivationStep, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "phase_id": self.phase_id,
            "target": self.target.to_dict(),
            "candidate_kind": self.candidate_kind.value,
            "angle_sense": self.angle_sense.value,
            "max_index": self.max_index,
            "limit": self.limit,
            "total_candidates": self.total_candidates,
            "rows": [row.to_dict() for row in self.rows],
            "derivation": [step.to_dict() for step in self.derivation],
        }


def _clip_unit(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def _angle_deg_from_cosine(value: float) -> float:
    return float(np.rad2deg(np.arccos(_clip_unit(value))))


def _tuple3(values: np.ndarray) -> Index3f:
    arr = np.asarray(values, dtype=float).reshape(3)
    return tuple(float(value) for value in arr)  # type: ignore[return-value]


def _projective_normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(3)
    scale = float(np.max(np.abs(arr)))
    if scale <= 0.0:
        raise ValueError("Cannot projectively normalize the zero vector.")

    normalized = arr / scale
    threshold = 1.0e-14
    normalized[np.abs(normalized) < threshold] = 0.0

    nonzero = np.flatnonzero(np.abs(normalized) > threshold)
    if nonzero.size and normalized[int(nonzero[0])] < 0.0:
        normalized = -normalized
    return normalized


def _projective_residual(a: np.ndarray, b: np.ndarray) -> float:
    aa = _projective_normalize(a)
    bb = _projective_normalize(b)
    return float(min(np.linalg.norm(aa - bb), np.linalg.norm(aa + bb)))


def _primitive_triplet(values: Index3i) -> Index3i:
    nonzero = [abs(value) for value in values if value]
    divisor = 0
    for value in nonzero:
        divisor = gcd(divisor, value)
    if divisor == 0:
        raise ValueError("The zero triplet is not valid.")

    reduced = tuple(value // divisor for value in values)
    return reduced  # type: ignore[return-value]


def _canonical_projective_triplet(values: Index3i) -> Index3i:
    reduced = _primitive_triplet(values)
    for value in reduced:
        if value > 0:
            return reduced
        if value < 0:
            return tuple(-item for item in reduced)  # type: ignore[return-value]
    raise ValueError("The zero triplet is not valid.")


def primitive_low_index_triplets(
    max_index: int,
    *,
    projective: bool = True,
) -> tuple[Index3i, ...]:
    """Enumerate primitive integer triplets within |index| <= max_index.

    Multiples such as [2 0 2] and [1 0 1] are represented once.  With
    ``projective=True``, opposite senses are also identified.
    """

    if max_index < 1:
        raise ValueError("max_index must be >= 1.")

    unique: set[Index3i] = set()
    values = range(-max_index, max_index + 1)
    for triplet in product(values, repeat=3):
        if triplet == (0, 0, 0):
            continue
        primitive = _primitive_triplet(triplet)
        if projective:
            primitive = _canonical_projective_triplet(primitive)
        unique.add(primitive)

    return tuple(
        sorted(
            unique,
            key=lambda item: (
                max(abs(value) for value in item),
                sum(abs(value) for value in item),
                item,
            ),
        )
    )


def _notation(
    indices: tuple[int | float, int | float, int | float],
    kind: CandidateKind,
    *,
    family: bool,
) -> str:
    body = " ".join(f"{float(value):g}" for value in indices)
    if kind is CandidateKind.DIRECTION:
        return f"<{body}>" if family else f"[{body}]"
    return f"{{{body}}}" if family else f"({body})"


def _reciprocal_cell(metric_inverse: np.ndarray) -> ReciprocalCell:
    reciprocal = np.asarray(metric_inverse, dtype=float)
    lengths = np.sqrt(np.diag(reciprocal))
    a_star, b_star, c_star = (float(value) for value in lengths)

    alpha_star = _angle_deg_from_cosine(reciprocal[1, 2] / (b_star * c_star))
    beta_star = _angle_deg_from_cosine(reciprocal[0, 2] / (a_star * c_star))
    gamma_star = _angle_deg_from_cosine(reciprocal[0, 1] / (a_star * b_star))
    volume_star = float(np.sqrt(np.linalg.det(reciprocal)))

    return ReciprocalCell(
        a_star,
        b_star,
        c_star,
        alpha_star,
        beta_star,
        gamma_star,
        volume_star,
    )


def _cartesian_unit_views(
    obj: CrystalObject,
    phase: PhaseState,
) -> tuple[CartesianUnitView, ...]:
    views: list[CartesianUnitView] = []
    for convention in CartesianConvention:
        frame = CartesianFrame(phase.lattice, convention)
        if isinstance(obj, Direction):
            values = frame.direct_to_cartesian(obj.array, normalize=True)
        else:
            values = frame.plane_to_cartesian(obj.array, normalize=True)
        views.append(CartesianUnitView(convention.value, _tuple3(values)))
    return tuple(views)


def _candidate_kind(value: str | CandidateKind | None) -> CandidateKind | None:
    if value is None or isinstance(value, CandidateKind):
        return value

    key = re.sub(r"[^a-z]", "", value.lower())
    aliases = {
        "d": CandidateKind.DIRECTION,
        "dir": CandidateKind.DIRECTION,
        "direction": CandidateKind.DIRECTION,
        "directions": CandidateKind.DIRECTION,
        "uvw": CandidateKind.DIRECTION,
        "p": CandidateKind.PLANE,
        "plane": CandidateKind.PLANE,
        "planes": CandidateKind.PLANE,
        "hkl": CandidateKind.PLANE,
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown candidate kind {value!r}; use 'direction' or 'plane'."
        )
    return aliases[key]


def _angle_sense(value: str | AngleSense) -> AngleSense:
    if isinstance(value, AngleSense):
        return value
    key = value.strip().lower()
    aliases = {
        "projective": AngleSense.PROJECTIVE,
        "family": AngleSense.PROJECTIVE,
        "axis": AngleSense.PROJECTIVE,
        "oriented": AngleSense.ORIENTED,
        "vector": AngleSense.ORIENTED,
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown angle sense {value!r}; use 'projective' or 'oriented'."
        )
    return aliases[key]


class CalPadService:
    """PTCLab-style calculator over a validated CuAlNi-CT ProjectState."""

    def __init__(
        self,
        project: ProjectState,
        console: CrystallographyConsole | None = None,
    ):
        self.project = project
        self.console = console or CrystallographyConsole(project)

    def phase_cell(self, phase_query: str) -> PhaseCellReport:
        phase = self.console.resolve_phase(phase_query)
        lattice = phase.lattice
        metric = lattice.metric()
        reciprocal = lattice.reciprocal_metric()
        identity = metric @ reciprocal
        residual = float(np.linalg.norm(identity - np.eye(3)))
        volume = float(np.sqrt(np.linalg.det(metric)))

        return PhaseCellReport(
            phase_id=phase.phase_id,
            label=phase.label,
            physical_phase=phase.physical_phase,
            cell_representation=phase.cell_representation,
            basis_id=phase.basis.basis_id,
            point_group_symbol=phase.point_group_symbol,
            symmetry_order=len(phase.symmetry_operators),
            length_unit=lattice.length_unit,
            a=lattice.a,
            b=lattice.b,
            c=lattice.c,
            alpha_deg=lattice.alpha_deg,
            beta_deg=lattice.beta_deg,
            gamma_deg=lattice.gamma_deg,
            volume=volume,
            direct_metric=metric,
            reciprocal_metric=reciprocal,
            reciprocal_cell=_reciprocal_cell(reciprocal),
            provenance_status=phase.provenance.status.value,
            provenance_source_key=phase.provenance.source_key,
            provenance_uncertainty=phase.provenance.uncertainty,
            provenance_notes=phase.provenance.notes,
            metric_inverse_residual=residual,
        )

    def _nearest_low_index(
        self,
        phase: PhaseState,
        target: CrystalObject,
        kind: CandidateKind,
        *,
        max_index: int,
    ) -> NearestLowIndex:
        triplets = primitive_low_index_triplets(max_index, projective=True)
        ranked: list[tuple[float, Index3i]] = []

        for triplet in triplets:
            if kind is CandidateKind.DIRECTION:
                candidate = Direction(triplet, phase.basis)
                if not isinstance(target, Direction):
                    raise TypeError(
                        "Direction low-index approximation requires a "
                        "direct-space target."
                    )
                angle = axis_angle_deg(target, candidate, phase.lattice)
            else:
                candidate = Plane(triplet, phase.basis)
                if not isinstance(target, Plane):
                    raise TypeError(
                        "Plane low-index approximation requires a "
                        "reciprocal-space target."
                    )
                angle = interplanar_angle_deg(target, candidate, phase.lattice)

            ranked.append((angle, triplet))

        angle, indices = min(
            ranked,
            key=lambda item: (
                item[0],
                max(abs(value) for value in item[1]),
                sum(abs(value) for value in item[1]),
                item[1],
            ),
        )
        return NearestLowIndex(
            notation=_notation(indices, kind, family=True),
            indices=indices,
            angular_mismatch_deg=float(angle),
            search_max_index=max_index,
            exact_within_numerical_tolerance=bool(
                angle <= self.project.numerical_policy.projective_angle_deg
            ),
        )

    def normal_conversion(
        self,
        phase_query: str,
        text: str,
        *,
        max_index: int = 12,
    ) -> NormalConversionReport:
        """Convert plane <-> physical-normal direction using the metric.

        Plane to direction:
            n_crystal ∝ M^-1 p

        Direction to plane:
            p ∝ M u

        The raw result is generally non-integer for non-cubic crystals.  A
        nearest low-index lattice object is therefore reported separately,
        together with its angular mismatch.
        """

        phase = self.console.resolve_phase(phase_query)
        parsed = parse_crystal_input(text)
        metric = phase.lattice.metric()
        warnings: list[str] = []

        if parsed.kind is ConsoleInputKind.PLANE:
            source = Plane(parsed.indices, phase.basis)
            raw = np.linalg.solve(metric, source.array)
            target = Direction(_tuple3(raw), phase.basis)
            target_kind = CandidateKind.DIRECTION
            relation = "physical plane normal expressed in direct-basis coefficients"
            roundtrip = metric @ raw
            projective_residual = _projective_residual(
                roundtrip,
                source.array,
            )
            derivation = (
                DerivationStep(
                    "Plane covector to physical normal",
                    "M n_crystal ∝ p  =>  n_crystal ∝ M^-1 p",
                    f"n_crystal ∝ {np.array(_projective_normalize(raw))}",
                ),
                DerivationStep(
                    "Round trip",
                    "p_reconstructed ∝ M n_crystal",
                    f"projective residual = {projective_residual:.3e}",
                ),
            )
        else:
            source = Direction(parsed.indices, phase.basis)
            raw = metric @ source.array
            target = Plane(_tuple3(raw), phase.basis)
            target_kind = CandidateKind.PLANE
            relation = "direct direction -> reciprocal plane with parallel normal"
            roundtrip = np.linalg.solve(metric, raw)
            projective_residual = _projective_residual(
                roundtrip,
                source.array,
            )
            derivation = (
                DerivationStep(
                    "Direction to normal plane",
                    "B^-T p ∥ B u  =>  p ∝ B^T B u = M u",
                    f"p ∝ {np.array(_projective_normalize(raw))}",
                ),
                DerivationStep(
                    "Round trip",
                    "u_reconstructed ∝ M^-1 p",
                    f"projective residual = {projective_residual:.3e}",
                ),
            )

        projective = _projective_normalize(raw)
        if isinstance(target, Direction):
            metric_unit = target.unit_coordinates(phase.lattice)
        else:
            metric_unit = target.unit_covector(phase.lattice)

        nearest = self._nearest_low_index(
            phase,
            target,
            target_kind,
            max_index=max_index,
        )
        if not nearest.exact_within_numerical_tolerance:
            warnings.append(
                "The nearest low-index result is an approximation, not an "
                "exact Miller index identity. Use the angular mismatch shown."
            )

        return NormalConversionReport(
            phase_id=phase.phase_id,
            source=parsed,
            target_kind=target_kind,
            relation=relation,
            raw_target_coefficients=_tuple3(raw),
            projective_target_coefficients=_tuple3(projective),
            metric_unit_target_coefficients=_tuple3(metric_unit),
            cartesian_unit_views=_cartesian_unit_views(target, phase),
            nearest_low_index=nearest,
            roundtrip_projective_residual=projective_residual,
            derivation=derivation,
            warnings=tuple(warnings),
        )

    def low_index_table(
        self,
        phase_query: str,
        target_text: str,
        *,
        candidate_kind: str | CandidateKind | None = None,
        max_index: int = 3,
        limit: int = 20,
        angle_sense: str | AngleSense = AngleSense.PROJECTIVE,
    ) -> LowIndexTableReport:
        """Rank low-index directions/planes by the actual lattice metric."""

        if limit < 1:
            raise ValueError("limit must be >= 1.")

        phase = self.console.resolve_phase(phase_query)
        parsed = parse_crystal_input(target_text)
        target = (
            Direction(parsed.indices, phase.basis)
            if parsed.kind is ConsoleInputKind.DIRECTION
            else Plane(parsed.indices, phase.basis)
        )
        requested_kind = _candidate_kind(candidate_kind)
        if requested_kind is None:
            requested_kind = (
                CandidateKind.DIRECTION
                if isinstance(target, Direction)
                else CandidateKind.PLANE
            )
        sense = _angle_sense(angle_sense)

        projective_candidates = not (
            requested_kind is CandidateKind.DIRECTION
            and isinstance(target, Direction)
            and sense is AngleSense.ORIENTED
        )
        triplets = primitive_low_index_triplets(
            max_index,
            projective=projective_candidates,
        )

        length_symbol = length_unit_symbol(phase.lattice.length_unit)
        rows_unranked: list[
            tuple[
                float,
                Index3i,
                str,
                float,
                str,
                float | None,
            ]
        ] = []

        for triplet in triplets:
            incidence = None
            if requested_kind is CandidateKind.DIRECTION:
                candidate: CrystalObject = Direction(triplet, phase.basis)
                geometry_name = "direct_length"
                geometry = candidate.length(phase.lattice)
                geometry_unit = length_symbol

                if isinstance(target, Direction):
                    if sense is AngleSense.ORIENTED:
                        angle = direction_angle_deg(
                            target,
                            candidate,
                            phase.lattice,
                        )
                        relation = "oriented direction-direction"
                    else:
                        angle = axis_angle_deg(
                            target,
                            candidate,
                            phase.lattice,
                        )
                        relation = "projective direction-axis"
                else:
                    angle = direction_plane_angle_deg(
                        candidate,
                        target,
                        phase.lattice,
                    )
                    incidence = incidence_residual(
                        candidate,
                        target,
                        phase.lattice,
                    )
                    relation = "direction-plane"
            else:
                candidate = Plane(triplet, phase.basis)
                geometry_name = "interplanar_spacing"
                geometry = candidate.spacing(phase.lattice)
                geometry_unit = length_symbol

                if isinstance(target, Plane):
                    if sense is AngleSense.ORIENTED:
                        angle = plane_normal_angle_deg(
                            target,
                            candidate,
                            phase.lattice,
                        )
                        relation = "oriented plane-normal"
                    else:
                        angle = interplanar_angle_deg(
                            target,
                            candidate,
                            phase.lattice,
                        )
                        relation = "projective plane-plane"
                else:
                    angle = direction_plane_angle_deg(
                        target,
                        candidate,
                        phase.lattice,
                    )
                    incidence = incidence_residual(
                        target,
                        candidate,
                        phase.lattice,
                    )
                    relation = "direction-plane"

            rows_unranked.append(
                (
                    float(angle),
                    triplet,
                    relation,
                    float(geometry),
                    geometry_unit,
                    incidence,
                )
            )

        rows_unranked.sort(
            key=lambda item: (
                item[0],
                max(abs(value) for value in item[1]),
                sum(abs(value) for value in item[1]),
                item[1],
            )
        )

        rows: list[LowIndexRow] = []
        for rank, row in enumerate(rows_unranked[:limit], start=1):
            angle, indices, relation, geometry, geometry_unit, incidence = row
            rows.append(
                LowIndexRow(
                    rank=rank,
                    notation=_notation(
                        indices,
                        requested_kind,
                        family=projective_candidates,
                    ),
                    indices=indices,
                    angle_deg=angle,
                    relation=relation,
                    geometric_quantity_name=geometry_name,
                    geometric_quantity=geometry,
                    geometric_quantity_unit=geometry_unit,
                    incidence_residual_value=incidence,
                    incidence_exact=(
                        None
                        if incidence is None
                        else incidence <= self.project.numerical_policy.algebraic
                    ),
                )
            )

        if requested_kind is CandidateKind.DIRECTION:
            candidate_symbol = "[uvw]"
            candidate_metric = "u^T M v"
        else:
            candidate_symbol = "(hkl)"
            candidate_metric = "p^T M^-1 q"

        derivation = (
            DerivationStep(
                "Candidate generation",
                (f"primitive integer {candidate_symbol}, |index| <= {max_index}"),
                f"{len(triplets)} candidates before ranking",
            ),
            DerivationStep(
                "Metric ranking",
                candidate_metric,
                (
                    f"sorted by physical angle; "
                    f"sense={sense.value}, returned={len(rows)}"
                ),
            ),
        )

        return LowIndexTableReport(
            phase_id=phase.phase_id,
            target=parsed,
            candidate_kind=requested_kind,
            angle_sense=sense,
            max_index=max_index,
            limit=limit,
            total_candidates=len(triplets),
            rows=tuple(rows),
            derivation=derivation,
        )


class CalPadRenderer:
    """Readable text renderer for the structured CalPad result objects."""

    CARTESIAN_LABELS: ClassVar[dict[str, str]] = {
        "legacy_a_x_b_xy": "Cartesian legacy (x||a, b in xy)",
        "ptclab_a_x_c_xz": "PTCLab Cartesian (x||a, c in xz)",
        "symmetric_metric": "Metric-symmetric Cartesian (B=M^1/2)",
    }

    @staticmethod
    def _vector(values: Index3f) -> str:
        return "[" + ", ".join(f"{value:.9g}" for value in values) + "]"

    @staticmethod
    def _matrix(values: np.ndarray) -> list[str]:
        array = np.asarray(values, dtype=float).copy()
        # Presentation only: suppress machine-zero noise without altering data.
        array[np.abs(array) < 1.0e-14] = 0.0
        return [
            "    [" + "  ".join(f"{value: .9g}" for value in row) + "]" for row in array
        ]

    def phase_cell(self, report: PhaseCellReport) -> str:
        unit = length_unit_symbol(report.length_unit)
        reciprocal_unit = reciprocal_length_unit_symbol(report.length_unit)
        volume_unit = f"{unit}³" if unit else ""
        reciprocal_volume_unit = f"{reciprocal_unit}³" if reciprocal_unit else ""

        lines = [
            "=" * 82,
            f"PHASE / CELL  |  {report.label}",
            "=" * 82,
            f"phase id          : {report.phase_id}",
            f"physical phase    : {report.physical_phase}",
            f"cell representation: {report.cell_representation}",
            f"basis             : {report.basis_id}",
            f"point group       : {report.point_group_symbol}",
            f"symmetry order    : {report.symmetry_order}",
            "",
            "DIRECT CELL",
            (
                f"  a, b, c         : {report.a:.12g}, {report.b:.12g}, "
                f"{report.c:.12g} {unit}"
            ),
            (
                f"  alpha,beta,gamma: {report.alpha_deg:.9g}, "
                f"{report.beta_deg:.9g}, {report.gamma_deg:.9g} deg"
            ),
            f"  volume          : {report.volume:.12g} {volume_unit}",
            "",
            "RECIPROCAL CELL  (crystallographic convention, no 2π)",
            (
                f"  a*, b*, c*      : {report.reciprocal_cell.a_star:.12g}, "
                f"{report.reciprocal_cell.b_star:.12g}, "
                f"{report.reciprocal_cell.c_star:.12g} {reciprocal_unit}"
            ),
            (
                f"  alpha*,beta*,gamma*: "
                f"{report.reciprocal_cell.alpha_star_deg:.9g}, "
                f"{report.reciprocal_cell.beta_star_deg:.9g}, "
                f"{report.reciprocal_cell.gamma_star_deg:.9g} deg"
            ),
            (
                f"  reciprocal volume: "
                f"{report.reciprocal_cell.volume_star:.12g} "
                f"{reciprocal_volume_unit}"
            ),
            "",
            "DIRECT METRIC M",
            *self._matrix(report.direct_metric),
            "",
            "RECIPROCAL METRIC M^-1",
            *self._matrix(report.reciprocal_metric),
            "",
            "RELIABILITY / PROVENANCE",
            f"  ||M M^-1 - I|| : {report.metric_inverse_residual:.3e}",
            f"  status          : {report.provenance_status}",
            f"  source          : {report.provenance_source_key or '-'}",
            f"  uncertainty     : {report.provenance_uncertainty or '-'}",
        ]
        if report.provenance_notes:
            lines.append(f"  notes           : {report.provenance_notes}")
        return "\n".join(lines)

    def normal_conversion(
        self,
        report: NormalConversionReport,
        *,
        show_derivation: bool = False,
    ) -> str:
        target_word = (
            "direct-space normal coefficients"
            if report.target_kind is CandidateKind.DIRECTION
            else "reciprocal plane coefficients"
        )
        lines = [
            "=" * 82,
            f"NORMAL / INDEX CONVERSION  |  {report.source.canonical_text}",
            "=" * 82,
            f"phase             : {report.phase_id}",
            f"relation          : {report.relation}",
            f"target type       : {report.target_kind.value}",
            "",
            "EXACT METRIC RESULT",
            (
                f"  raw {target_word:34s}: "
                f"{self._vector(report.raw_target_coefficients)}"
            ),
            (
                f"  projective coefficients             : "
                f"{self._vector(report.projective_target_coefficients)}"
            ),
            (
                f"  metric-unit coefficients            : "
                f"{self._vector(report.metric_unit_target_coefficients)}"
            ),
            "",
            "UNIT CARTESIAN PHYSICAL DIRECTION / NORMAL",
        ]
        for view in report.cartesian_unit_views:
            label = self.CARTESIAN_LABELS.get(
                view.convention,
                view.convention,
            )
            lines.append(f"  {label:42s} {self._vector(view.coordinates)}")

        nearest = report.nearest_low_index
        exact_label = (
            "YES"
            if nearest.exact_within_numerical_tolerance
            else "NO — approximation only"
        )
        lines.extend(
            [
                "",
                "NEAREST LOW-INDEX LATTICE OBJECT",
                f"  search bound     : |index| <= {nearest.search_max_index}",
                f"  candidate        : {nearest.notation}",
                f"  angular mismatch : {nearest.angular_mismatch_deg:.12g} deg",
                f"  numerically exact: {exact_label}",
                "",
                "ROUND-TRIP CHECK",
                (f"  projective residual: {report.roundtrip_projective_residual:.3e}"),
            ]
        )

        if show_derivation:
            lines.extend(["", "DERIVATION"])
            for step in report.derivation:
                lines.extend(
                    [
                        f"  {step.title}",
                        f"    {step.formula}",
                        f"    {step.value}",
                    ]
                )

        if report.warnings:
            lines.extend(["", "NOTES"])
            lines.extend(f"  - {warning}" for warning in report.warnings)

        return "\n".join(lines)

    @staticmethod
    def low_index_table(
        report: LowIndexTableReport,
        *,
        show_derivation: bool = False,
    ) -> str:
        lines = [
            "=" * 82,
            f"LOW-INDEX TABLE  |  target {report.target.canonical_text}",
            "=" * 82,
            f"phase          : {report.phase_id}",
            f"candidate kind : {report.candidate_kind.value}",
            f"angle sense    : {report.angle_sense.value}",
            f"search bound   : |index| <= {report.max_index}",
            f"candidate count: {report.total_candidates}",
            "",
            " rank  candidate          angle (deg)     geometry            value",
            " ----  -----------------  --------------  ------------------  ----------------",
        ]
        for row in report.rows:
            geometry = (
                f"{row.geometric_quantity:.9g} {row.geometric_quantity_unit}"
            ).rstrip()
            lines.append(
                f" {row.rank:>4d}  {row.notation:<17s}  "
                f"{row.angle_deg:>14.8g}  "
                f"{row.geometric_quantity_name:<18s}  {geometry}"
            )
            if row.incidence_residual_value is not None:
                incidence_label = "  EXACT INCIDENCE" if row.incidence_exact else ""
                lines.append(
                    f"       incidence residual = "
                    f"{row.incidence_residual_value:.3e}"
                    f"{incidence_label}"
                )

        if show_derivation:
            lines.extend(["", "DERIVATION"])
            for step in report.derivation:
                lines.extend(
                    [
                        f"  {step.title}",
                        f"    {step.formula}",
                        f"    {step.value}",
                    ]
                )
        return "\n".join(lines)


def reference_calpad() -> CalPadService:
    from .project_state import james_hane_6m_reference_project

    return CalPadService(james_hane_6m_reference_project())


def _extract_object(text: str) -> tuple[str, str]:
    raw = text.strip()
    if not raw:
        raise ValueError("Missing crystallographic object.")

    closing = {
        "[": "]",
        "(": ")",
        "<": ">",
        "⟨": "⟩",
        "{": "}",
    }
    opener = raw[0]
    if opener not in closing:
        raise ValueError(
            "Start the object with [, (, <, ⟨ or {. Example: [1 0 1] or (0 1 1)."
        )
    close = closing[opener]
    position = raw.find(close, 1)
    if position < 0:
        raise ValueError(f"Missing closing delimiter {close!r}.")
    return raw[: position + 1], raw[position + 1 :].strip()


def _parse_inline_options(text: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for token in text.split():
        if "=" in token:
            key, value = token.split("=", 1)
            options[key.strip().lower()] = value.strip()
        elif token:
            options[token.strip().lower()] = "true"
    return options


_HELP = """CuAlNi-CT CalPad commands

  phases
  use 6m
  cell
  inspect [1 0 1]
  compare [1 0 1] ; (0 1 1)

  normal (1 0 1)
      plane -> physical-normal direct coefficients

  normal [1 0 1]
      direction -> reciprocal plane whose physical normal is parallel

  normal (1 0 1) max=12
      also searches the nearest low-index lattice direction/axis

  low [1 0 1] direction max=3 limit=15
  low [1 0 1] plane     max=3 limit=15
  low (1 0 1) plane     max=3 limit=15
  low (1 0 1) direction max=3 limit=15

  Add 'oriented' for oriented direction/normal angles.
  Default low-index ranking is projective (family/axis) geometry.

  derive on|off
  help
  quit

Notation
  [uvw]  direct direction
  (hkl)  reciprocal plane
  <uvw>  direction/axis family
  {hkl}  plane family
"""


def _interactive_repl(service: CalPadService) -> int:
    renderer = CalPadRenderer()
    base_renderer = ConsoleRenderer()
    current_phase = service.project.phases[0].phase_id
    show_derivation = False

    print("CuAlNi-CT CalPad")
    print("Metric-correct crystallography with PTCLab-compatible Cartesian view.")
    print(f"Current phase: {current_phase}")
    print("Type 'help' for commands.")

    while True:
        try:
            line = input("calpad> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue

        command, _, remainder = line.partition(" ")
        command = command.lower()

        try:
            if command in {"quit", "exit", "q"}:
                return 0
            if command == "help":
                print(_HELP)
                continue
            if command == "phases":
                print(json.dumps(service.console.phase_choices(), indent=2))
                continue
            if command == "use":
                phase = service.console.resolve_phase(remainder)
                current_phase = phase.phase_id
                print(f"Current phase: {current_phase}")
                continue
            if command == "derive":
                value = remainder.strip().lower()
                if value not in {"on", "off"}:
                    raise ValueError("Use 'derive on' or 'derive off'.")
                show_derivation = value == "on"
                print(f"Derivations: {'on' if show_derivation else 'off'}")
                continue
            if command in {"cell", "phase"}:
                phase_query = remainder.strip() or current_phase
                report = service.phase_cell(phase_query)
                print(renderer.phase_cell(report))
                continue
            if command == "inspect":
                report = service.console.inspect(current_phase, remainder)
                print(
                    base_renderer.object_report(
                        report,
                        show_derivation=show_derivation,
                    )
                )
                continue
            if command == "compare":
                left, separator, right = remainder.partition(";")
                if not separator:
                    raise ValueError(
                        "Separate objects with ';'. Example: compare [1 0 1] ; (0 1 1)"
                    )
                report = service.console.compare(
                    current_phase,
                    left,
                    right,
                )
                print(
                    base_renderer.comparison_report(
                        report,
                        show_derivation=show_derivation,
                    )
                )
                continue
            if command == "normal":
                object_text, option_text = _extract_object(remainder)
                options = _parse_inline_options(option_text)
                max_index = int(options.get("max", options.get("maxindex", "12")))
                report = service.normal_conversion(
                    current_phase,
                    object_text,
                    max_index=max_index,
                )
                print(
                    renderer.normal_conversion(
                        report,
                        show_derivation=show_derivation,
                    )
                )
                continue
            if command in {"low", "lowindex"}:
                object_text, option_text = _extract_object(remainder)
                options = _parse_inline_options(option_text)

                candidate = None
                for key in ("direction", "directions", "plane", "planes"):
                    if key in options:
                        candidate = key
                        break

                sense = (
                    AngleSense.ORIENTED
                    if "oriented" in options
                    else AngleSense.PROJECTIVE
                )
                report = service.low_index_table(
                    current_phase,
                    object_text,
                    candidate_kind=candidate,
                    max_index=int(options.get("max", "3")),
                    limit=int(options.get("limit", "20")),
                    angle_sense=sense,
                )
                print(
                    renderer.low_index_table(
                        report,
                        show_derivation=show_derivation,
                    )
                )
                continue

            print("Unknown command. Type 'help'.")
        except (KeyError, TypeError, ValueError) as exc:
            print(f"INPUT ERROR: {exc}")


def _add_subcommand_project_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project",
        default=argparse.SUPPRESS,
        help="Project source: preset:NAME or path to generic project JSON.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CuAlNi-CT CalPad: metric-correct single-phase crystallography calculator."
        )
    )
    parser.add_argument(
        "--project",
        default="preset:james_hane_2000",
        help=(
            "Project source: preset:NAME or path to generic project JSON. "
            "James-Hane remains only the backward-compatible benchmark default."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    phases_parser = subparsers.add_parser("phases")
    _add_subcommand_project_arg(phases_parser)

    cell_parser = subparsers.add_parser("cell")
    _add_subcommand_project_arg(cell_parser)
    cell_parser.add_argument("--phase", required=True)

    normal_parser = subparsers.add_parser("normal")
    _add_subcommand_project_arg(normal_parser)
    normal_parser.add_argument("--phase", required=True)
    normal_parser.add_argument("--max-index", type=int, default=12)
    normal_parser.add_argument("--derive", action="store_true")
    normal_parser.add_argument("object_text")

    low_parser = subparsers.add_parser("lowindex")
    _add_subcommand_project_arg(low_parser)
    low_parser.add_argument("--phase", required=True)
    low_parser.add_argument(
        "--candidate",
        choices=("direction", "plane"),
        default=None,
    )
    low_parser.add_argument("--max-index", type=int, default=3)
    low_parser.add_argument("--limit", type=int, default=20)
    low_parser.add_argument(
        "--sense",
        choices=("projective", "oriented"),
        default="projective",
    )
    low_parser.add_argument("--derive", action="store_true")
    low_parser.add_argument("target")

    inspect_parser = subparsers.add_parser("inspect")
    _add_subcommand_project_arg(inspect_parser)
    inspect_parser.add_argument("--phase", required=True)
    inspect_parser.add_argument("--derive", action="store_true")
    inspect_parser.add_argument("object_text")

    compare_parser = subparsers.add_parser("compare")
    _add_subcommand_project_arg(compare_parser)
    compare_parser.add_argument("--phase", required=True)
    compare_parser.add_argument("--derive", action="store_true")
    compare_parser.add_argument("left")
    compare_parser.add_argument("right")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        service = CalPadService(load_project(args.project).project)
    except (AssertionError, KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        parser.error(f"project: {exc}")
    renderer = CalPadRenderer()
    base_renderer = ConsoleRenderer()

    if args.command is None:
        return _interactive_repl(service)
    if args.command == "phases":
        print(json.dumps(service.console.phase_choices(), indent=2))
        return 0
    if args.command == "cell":
        print(renderer.phase_cell(service.phase_cell(args.phase)))
        return 0
    if args.command == "normal":
        report = service.normal_conversion(
            args.phase,
            args.object_text,
            max_index=args.max_index,
        )
        print(
            renderer.normal_conversion(
                report,
                show_derivation=args.derive,
            )
        )
        return 0
    if args.command == "lowindex":
        report = service.low_index_table(
            args.phase,
            args.target,
            candidate_kind=args.candidate,
            max_index=args.max_index,
            limit=args.limit,
            angle_sense=args.sense,
        )
        print(
            renderer.low_index_table(
                report,
                show_derivation=args.derive,
            )
        )
        return 0
    if args.command == "inspect":
        report = service.console.inspect(args.phase, args.object_text)
        print(
            base_renderer.object_report(
                report,
                show_derivation=args.derive,
            )
        )
        return 0
    if args.command == "compare":
        report = service.console.compare(
            args.phase,
            args.left,
            args.right,
        )
        print(
            base_renderer.comparison_report(
                report,
                show_derivation=args.derive,
            )
        )
        return 0

    parser.error(f"Unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
