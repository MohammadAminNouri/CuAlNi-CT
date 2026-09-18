from __future__ import annotations

"""Human-facing crystallography console built on the verified scientific core.

This module is intentionally split into two layers:

- ``CrystallographyConsole`` returns structured, JSON-ready scientific results.
- ``ConsoleRenderer`` and ``main`` provide a simple terminal interface.

The calculation formulas are not reimplemented in the UI layer. Typed
Direction/Plane objects, ProjectState, CartesianFrame, and CalculationService
remain the scientific authorities.
"""

import argparse
import json
import re
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

import numpy as np

from .calculation_service import CalculationKind, CalculationService
from .crystal_objects import (
    Direction,
    Plane,
    axis_angle_deg,
    direction_angle_deg,
    direction_plane_angle_deg,
    equivalent_directions,
    equivalent_planes,
    incidence_residual,
    interplanar_angle_deg,
    lies_in_plane,
    plane_normal_angle_deg,
)
from .project_state import PhaseState, ProjectState, TransformationState
from .representation import CartesianConvention
from .units import length_unit_symbol, reciprocal_length_unit_symbol

CrystalObject = Direction | Plane


class ConsoleInputKind(str, Enum):
    DIRECTION = "direction"
    PLANE = "plane"


class ConsoleNotation(str, Enum):
    DIRECTION = "direction"
    PLANE = "plane"
    DIRECTION_FAMILY = "direction_family"
    PLANE_FAMILY = "plane_family"


class ComparisonKind(str, Enum):
    DIRECTION_DIRECTION = "direction_direction"
    PLANE_PLANE = "plane_plane"
    DIRECTION_PLANE = "direction_plane"


@dataclass(frozen=True)
class ParsedCrystalInput:
    indices: tuple[float, float, float]
    kind: ConsoleInputKind
    family: bool
    notation: ConsoleNotation
    canonical_text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "indices": list(self.indices),
            "kind": self.kind.value,
            "family": self.family,
            "notation": self.notation.value,
            "canonical_text": self.canonical_text,
        }


@dataclass(frozen=True)
class DerivationStep:
    title: str
    formula: str
    value: str

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "formula": self.formula,
            "value": self.value,
        }


@dataclass(frozen=True)
class CartesianView:
    convention: str
    coordinates: tuple[float, float, float]
    unit_coordinates: tuple[float, float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "convention": self.convention,
            "coordinates": list(self.coordinates),
            "unit_coordinates": list(self.unit_coordinates),
        }


@dataclass(frozen=True)
class ConsoleObjectReport:
    phase_id: str
    phase_label: str
    physical_phase: str
    cell_representation: str
    basis_id: str
    parsed: ParsedCrystalInput
    object_payload: dict[str, object]
    dimensional_quantity_name: str
    dimensional_quantity: float
    dimensional_unit_label: str
    secondary_quantity_name: str
    secondary_quantity: float | None
    secondary_unit_label: str
    metric_normalized_coordinates: tuple[float, float, float]
    cartesian_views: tuple[CartesianView, ...]
    maximum_representation_residual: float
    symmetry_equivalents: tuple[str, ...]
    symmetry_equivalent_count: int
    derivation: tuple[DerivationStep, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "phase_id": self.phase_id,
            "phase_label": self.phase_label,
            "physical_phase": self.physical_phase,
            "cell_representation": self.cell_representation,
            "basis_id": self.basis_id,
            "parsed": self.parsed.to_dict(),
            "object": self.object_payload,
            "dimensional_quantity_name": self.dimensional_quantity_name,
            "dimensional_quantity": self.dimensional_quantity,
            "dimensional_unit_label": self.dimensional_unit_label,
            "secondary_quantity_name": self.secondary_quantity_name,
            "secondary_quantity": self.secondary_quantity,
            "secondary_unit_label": self.secondary_unit_label,
            "metric_normalized_coordinates": list(self.metric_normalized_coordinates),
            "cartesian_views": [view.to_dict() for view in self.cartesian_views],
            "maximum_representation_residual": self.maximum_representation_residual,
            "symmetry_equivalents": list(self.symmetry_equivalents),
            "symmetry_equivalent_count": self.symmetry_equivalent_count,
            "derivation": [step.to_dict() for step in self.derivation],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ConsoleComparisonReport:
    phase_id: str
    left: ParsedCrystalInput
    right: ParsedCrystalInput
    comparison_kind: ComparisonKind
    primary_name: str
    primary_value: float
    primary_unit: str
    secondary_name: str
    secondary_value: float | None
    secondary_unit: str
    incidence_residual_value: float | None
    lies_in_plane_value: bool | None
    maximum_representation_residual: float
    derivation: tuple[DerivationStep, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "phase_id": self.phase_id,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "comparison_kind": self.comparison_kind.value,
            "primary_name": self.primary_name,
            "primary_value": self.primary_value,
            "primary_unit": self.primary_unit,
            "secondary_name": self.secondary_name,
            "secondary_value": self.secondary_value,
            "secondary_unit": self.secondary_unit,
            "incidence_residual": self.incidence_residual_value,
            "lies_in_plane": self.lies_in_plane_value,
            "maximum_representation_residual": self.maximum_representation_residual,
            "derivation": [step.to_dict() for step in self.derivation],
        }


@dataclass(frozen=True)
class ConsoleMappingReport:
    transformation_id: str
    source_phase_id: str
    target_phase_id: str
    source: ParsedCrystalInput
    mapped_object_payload: dict[str, object]
    mapped_canonical_text: str
    derivation: tuple[DerivationStep, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "transformation_id": self.transformation_id,
            "source_phase_id": self.source_phase_id,
            "target_phase_id": self.target_phase_id,
            "source": self.source.to_dict(),
            "mapped_object": self.mapped_object_payload,
            "mapped_canonical_text": self.mapped_canonical_text,
            "derivation": [step.to_dict() for step in self.derivation],
        }


_OPEN_CLOSE: dict[str, tuple[str, ConsoleInputKind, bool, ConsoleNotation]] = {
    "[": ("]", ConsoleInputKind.DIRECTION, False, ConsoleNotation.DIRECTION),
    "(": (")", ConsoleInputKind.PLANE, False, ConsoleNotation.PLANE),
    "<": (">", ConsoleInputKind.DIRECTION, True, ConsoleNotation.DIRECTION_FAMILY),
    "⟨": ("⟩", ConsoleInputKind.DIRECTION, True, ConsoleNotation.DIRECTION_FAMILY),
    "{": ("}", ConsoleInputKind.PLANE, True, ConsoleNotation.PLANE_FAMILY),
}

_UNICODE_MINUS = str.maketrans(
    {
        "−": "-",
        "–": "-",
        "—": "-",
        "﹣": "-",
    }
)


def _format_index(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) <= 1.0e-12:
        return str(int(rounded))

    fraction = Fraction(value).limit_denominator(1000)
    if abs(float(fraction) - value) <= 1.0e-12:
        return str(fraction)
    return f"{value:.12g}"


def _canonical_notation(
    indices: tuple[float, float, float],
    kind: ConsoleInputKind,
    *,
    family: bool,
) -> str:
    body = " ".join(_format_index(value) for value in indices)
    if kind is ConsoleInputKind.DIRECTION:
        return f"<{body}>" if family else f"[{body}]"
    return f"{{{body}}}" if family else f"({body})"


def _parse_kind_hint(
    kind_hint: str | ConsoleInputKind | None,
) -> ConsoleInputKind | None:
    if kind_hint is None:
        return None
    if isinstance(kind_hint, ConsoleInputKind):
        return kind_hint

    normalized = re.sub(r"[^a-z]", "", kind_hint.lower())
    aliases = {
        "d": ConsoleInputKind.DIRECTION,
        "dir": ConsoleInputKind.DIRECTION,
        "direction": ConsoleInputKind.DIRECTION,
        "uvw": ConsoleInputKind.DIRECTION,
        "p": ConsoleInputKind.PLANE,
        "plane": ConsoleInputKind.PLANE,
        "hkl": ConsoleInputKind.PLANE,
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown object kind {kind_hint!r}. Use 'direction' or 'plane'."
        )
    return aliases[normalized]


def _parse_number(token: str) -> float:
    try:
        value = float(Fraction(token))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(
            f"Invalid crystallographic index {token!r}. "
            "Use integers, decimals, or fractions such as -1/2."
        ) from exc
    if not np.isfinite(value):
        raise ValueError(f"Crystallographic index {token!r} is not finite.")
    return value


def parse_crystal_input(
    text: str,
    *,
    kind_hint: str | ConsoleInputKind | None = None,
) -> ParsedCrystalInput:
    """Parse a safe human crystallographic input.

    Accepted examples:

    - ``[1 0 -1]`` or ``[1, 0, -1]``: direction
    - ``(1 1 0)``: plane
    - ``<1 0 0>`` or ``⟨1 0 0⟩``: direction family/axis family
    - ``{1 1 0}``: plane family
    - bare ``1 0 -1`` only when ``kind_hint`` is supplied

    Compact strings such as ``[10-1]`` are deliberately rejected because they
    are ambiguous once multi-digit or fractional indices are allowed.
    """

    raw = text.strip().translate(_UNICODE_MINUS)
    if not raw:
        raise ValueError("Crystallographic input is empty.")

    hint = _parse_kind_hint(kind_hint)
    wrapper = _OPEN_CLOSE.get(raw[0])

    family = False
    if wrapper is not None:
        close, inferred_kind, family, notation = wrapper
        if not raw.endswith(close):
            raise ValueError(
                f"Input starts with {raw[0]!r} but does not end with {close!r}."
            )
        if hint is not None and hint is not inferred_kind:
            raise ValueError(
                f"Notation {raw!r} means {inferred_kind.value}, but the explicit "
                f"kind hint says {hint.value}. Refusing to guess."
            )
        kind = inferred_kind
        body = raw[1:-1].strip()
    else:
        if hint is None:
            raise ValueError(
                "Bare indices are ambiguous. Use [u v w] for a direction, "
                "(h k l) for a plane, or provide an explicit kind."
            )
        kind = hint
        notation = (
            ConsoleNotation.DIRECTION
            if kind is ConsoleInputKind.DIRECTION
            else ConsoleNotation.PLANE
        )
        body = raw

    body = body.replace(",", " ").replace(";", " ")
    tokens = body.split()
    if len(tokens) != 3:
        raise ValueError(
            "Exactly three crystallographic indices are required. "
            "Separate them with spaces or commas; for example [1 0 -1]."
        )

    indices = tuple(_parse_number(token) for token in tokens)
    if np.linalg.norm(np.asarray(indices, dtype=float)) <= 0.0:
        raise ValueError("The zero triplet is not a valid direction or plane.")

    canonical = _canonical_notation(indices, kind, family=family)
    return ParsedCrystalInput(
        indices=indices,  # type: ignore[arg-type]
        kind=kind,
        family=family,
        notation=notation,
        canonical_text=canonical,
    )


def _normalized_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _tuple3(values: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float).reshape(3)
    return tuple(float(value) for value in arr)  # type: ignore[return-value]


def _relative_scalar_residual(reference: float, value: float) -> float:
    return abs(value - reference) / max(abs(reference), abs(value), 1.0)


def _projective_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).reshape(3)
    bb = np.asarray(b, dtype=float).reshape(3)
    aa = aa / np.linalg.norm(aa)
    bb = bb / np.linalg.norm(bb)
    cosine = float(np.clip(abs(aa @ bb), 0.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))


class CrystallographyConsole:
    """Structured console API over one validated ProjectState."""

    def __init__(
        self,
        project: ProjectState,
        calculation_service: CalculationService | None = None,
    ):
        self.project = project
        self.calculations = calculation_service or CalculationService(project)
        if self.calculations.project is not project:
            self.calculations.set_project(project)

        validation = self.calculations.compute(CalculationKind.PROJECT_VALIDATION)
        validation.assert_passed()  # type: ignore[attr-defined]

    def phase_choices(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {
                "phase_id": phase.phase_id,
                "label": phase.label,
                "cell_representation": phase.cell_representation,
                "physical_phase": phase.physical_phase,
            }
            for phase in self.project.phases
        )

    def transformation_choices(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {
                "transformation_id": transformation.transformation_id,
                "label": transformation.label,
                "parent_phase_id": transformation.parent_phase_id,
                "product_phase_id": transformation.product_phase_id,
            }
            for transformation in self.project.transformations
        )

    def resolve_phase(self, query: str) -> PhaseState:
        raw = query.strip()
        if not raw:
            raise ValueError("Phase query is empty.")

        for phase in self.project.phases:
            if raw == phase.phase_id:
                return phase

        key = _normalized_key(raw)
        exact_aliases: list[PhaseState] = []
        fuzzy: list[PhaseState] = []
        for phase in self.project.phases:
            aliases = {
                _normalized_key(phase.phase_id),
                _normalized_key(phase.label),
                _normalized_key(phase.cell_representation),
                _normalized_key(phase.physical_phase),
            }
            if key in aliases:
                exact_aliases.append(phase)
            elif any(key and key in alias for alias in aliases):
                fuzzy.append(phase)

        candidates = exact_aliases or fuzzy
        unique = {phase.phase_id: phase for phase in candidates}
        if len(unique) == 1:
            return next(iter(unique.values()))
        if not unique:
            choices = ", ".join(phase.phase_id for phase in self.project.phases)
            raise ValueError(f"Unknown phase {query!r}. Available phases: {choices}")
        choices = ", ".join(unique)
        raise ValueError(
            f"Phase query {query!r} is ambiguous. Matches: {choices}. "
            "Use an exact phase_id."
        )

    def resolve_transformation(self, query: str) -> TransformationState:
        raw = query.strip()
        for transformation in self.project.transformations:
            if raw == transformation.transformation_id:
                return transformation

        key = _normalized_key(raw)
        matches = [
            transformation
            for transformation in self.project.transformations
            if key
            and (
                key in _normalized_key(transformation.transformation_id)
                or key in _normalized_key(transformation.label)
            )
        ]
        unique = {
            transformation.transformation_id: transformation
            for transformation in matches
        }
        if len(unique) == 1:
            return next(iter(unique.values()))
        if not unique:
            choices = ", ".join(
                transformation.transformation_id
                for transformation in self.project.transformations
            )
            raise ValueError(
                f"Unknown transformation {query!r}. Available: {choices or 'none'}"
            )
        choices = ", ".join(unique)
        raise ValueError(
            f"Transformation query {query!r} is ambiguous. Matches: {choices}."
        )

    @staticmethod
    def _object(parsed: ParsedCrystalInput, phase: PhaseState) -> CrystalObject:
        if parsed.kind is ConsoleInputKind.DIRECTION:
            return Direction(parsed.indices, phase.basis)
        return Plane(parsed.indices, phase.basis)

    @staticmethod
    def _warnings(phase: PhaseState) -> tuple[str, ...]:
        warnings: list[str] = []
        if not phase.symmetry_operators:
            warnings.append(
                "No symmetry operators are registered for this phase; "
                "symmetry-equivalent sets are unavailable."
            )
        if not phase.lattice.length_unit:
            warnings.append(
                "Lattice length unit is not declared; dimensional outputs are "
                "reported without a physical unit symbol."
            )
        return tuple(warnings)

    @staticmethod
    def _equivalent_notations(
        obj: CrystalObject,
        phase: PhaseState,
        *,
        family: bool,
    ) -> tuple[str, ...]:
        operators = phase.symmetry_matrices()
        if isinstance(obj, Direction):
            equivalent = equivalent_directions(
                obj,
                operators,
                projective=family,
            )
            return tuple(
                _canonical_notation(
                    item.indices,
                    ConsoleInputKind.DIRECTION,
                    family=family,
                )
                for item in equivalent
            )

        equivalent = equivalent_planes(
            obj,
            operators,
            projective=family or True,
        )
        return tuple(
            _canonical_notation(
                item.indices,
                ConsoleInputKind.PLANE,
                family=family,
            )
            for item in equivalent
        )

    @staticmethod
    def _cartesian_views(
        obj: CrystalObject,
        phase: PhaseState,
    ) -> tuple[CartesianView, ...]:
        views: list[CartesianView] = []
        for convention in CartesianConvention:
            if isinstance(obj, Direction):
                coordinates = obj.cartesian(phase.lattice, convention)
                unit = obj.cartesian(phase.lattice, convention, normalize=True)
            else:
                coordinates = obj.cartesian_normal(phase.lattice, convention)
                unit = obj.cartesian_normal(
                    phase.lattice,
                    convention,
                    normalize=True,
                )
            views.append(
                CartesianView(
                    convention.value,
                    _tuple3(coordinates),
                    _tuple3(unit),
                )
            )
        return tuple(views)

    @staticmethod
    def _representation_residual(
        obj: CrystalObject,
        phase: PhaseState,
        views: tuple[CartesianView, ...],
    ) -> float:
        if isinstance(obj, Direction):
            reference = obj.length(phase.lattice)
        else:
            reference = obj.reciprocal_length(phase.lattice)

        residuals = [
            _relative_scalar_residual(
                reference,
                float(np.linalg.norm(np.asarray(view.coordinates, dtype=float))),
            )
            for view in views
        ]
        return max(residuals, default=0.0)

    def inspect(
        self,
        phase_query: str,
        text: str,
        *,
        kind_hint: str | ConsoleInputKind | None = None,
    ) -> ConsoleObjectReport:
        phase = self.resolve_phase(phase_query)
        parsed = parse_crystal_input(text, kind_hint=kind_hint)
        obj = self._object(parsed, phase)
        views = self._cartesian_views(obj, phase)
        length_symbol = length_unit_symbol(phase.lattice.length_unit)
        reciprocal_symbol = reciprocal_length_unit_symbol(phase.lattice.length_unit)

        if isinstance(obj, Direction):
            dimensional_name = "direct_length"
            dimensional_value = obj.length(phase.lattice)
            secondary_name = ""
            secondary_value = None
            secondary_unit = ""
            dimensional_unit = length_symbol
            normalized = _tuple3(obj.unit_coordinates(phase.lattice))
            derivation = (
                DerivationStep(
                    "Direct-vector length",
                    "|u| = sqrt(u^T M u)",
                    f"|u| = {dimensional_value:.12g}",
                ),
                DerivationStep(
                    "Metric normalization",
                    "u_hat = u / sqrt(u^T M u)",
                    f"u_hat = {np.array(normalized)}",
                ),
            )
        else:
            reciprocal_length = obj.reciprocal_length(phase.lattice)
            spacing = obj.spacing(phase.lattice)
            dimensional_name = "reciprocal_length"
            dimensional_value = reciprocal_length
            secondary_name = "interplanar_spacing"
            secondary_value = spacing
            secondary_unit = length_symbol
            dimensional_unit = reciprocal_symbol
            normalized = _tuple3(obj.unit_covector(phase.lattice))
            derivation = (
                DerivationStep(
                    "Reciprocal-plane norm",
                    "|p|_* = sqrt(p^T M^-1 p)",
                    f"|p|_* = {reciprocal_length:.12g}",
                ),
                DerivationStep(
                    "Interplanar spacing",
                    "d_hkl = 1 / |p|_*",
                    f"d_hkl = {spacing:.12g}",
                ),
                DerivationStep(
                    "Reciprocal normalization",
                    "p_hat = p / sqrt(p^T M^-1 p)",
                    f"p_hat = {np.array(normalized)}",
                ),
            )

        equivalents = self._equivalent_notations(
            obj,
            phase,
            family=parsed.family,
        )
        return ConsoleObjectReport(
            phase_id=phase.phase_id,
            phase_label=phase.label,
            physical_phase=phase.physical_phase,
            cell_representation=phase.cell_representation,
            basis_id=phase.basis.basis_id,
            parsed=parsed,
            object_payload=obj.to_dict(),
            dimensional_quantity_name=dimensional_name,
            dimensional_quantity=dimensional_value,
            dimensional_unit_label=dimensional_unit,
            secondary_quantity_name=secondary_name,
            secondary_quantity=secondary_value,
            secondary_unit_label=secondary_unit,
            metric_normalized_coordinates=normalized,
            cartesian_views=views,
            maximum_representation_residual=self._representation_residual(
                obj,
                phase,
                views,
            ),
            symmetry_equivalents=equivalents,
            symmetry_equivalent_count=len(equivalents),
            derivation=derivation,
            warnings=self._warnings(phase),
        )

    @staticmethod
    def _cartesian_comparison_values(
        left: CrystalObject,
        right: CrystalObject,
        phase: PhaseState,
    ) -> tuple[float, ...]:
        values: list[float] = []
        for convention in CartesianConvention:
            if isinstance(left, Direction) and isinstance(right, Direction):
                a = left.cartesian(phase.lattice, convention, normalize=True)
                b = right.cartesian(phase.lattice, convention, normalize=True)
                cosine = float(np.clip(a @ b, -1.0, 1.0))
                values.append(float(np.rad2deg(np.arccos(cosine))))
                continue

            if isinstance(left, Plane) and isinstance(right, Plane):
                a = left.cartesian_normal(
                    phase.lattice,
                    convention,
                    normalize=True,
                )
                b = right.cartesian_normal(
                    phase.lattice,
                    convention,
                    normalize=True,
                )
                values.append(_projective_angle_deg(a, b))
                continue

            direction = left if isinstance(left, Direction) else right
            plane = left if isinstance(left, Plane) else right
            u = direction.cartesian(phase.lattice, convention, normalize=True)
            normal = plane.cartesian_normal(
                phase.lattice,
                convention,
                normalize=True,
            )
            sine = float(np.clip(abs(u @ normal), 0.0, 1.0))
            values.append(float(np.rad2deg(np.arcsin(sine))))
        return tuple(values)

    def compare(
        self,
        phase_query: str,
        left_text: str,
        right_text: str,
        *,
        left_kind_hint: str | ConsoleInputKind | None = None,
        right_kind_hint: str | ConsoleInputKind | None = None,
    ) -> ConsoleComparisonReport:
        phase = self.resolve_phase(phase_query)
        left_parsed = parse_crystal_input(left_text, kind_hint=left_kind_hint)
        right_parsed = parse_crystal_input(right_text, kind_hint=right_kind_hint)
        left = self._object(left_parsed, phase)
        right = self._object(right_parsed, phase)

        incidence = None
        is_in_plane = None
        if isinstance(left, Direction) and isinstance(right, Direction):
            primary = direction_angle_deg(left, right, phase.lattice)
            secondary = axis_angle_deg(left, right, phase.lattice)
            comparison_kind = ComparisonKind.DIRECTION_DIRECTION
            primary_name = "oriented_direction_angle"
            secondary_name = "projective_axis_angle"
            derivation = (
                DerivationStep(
                    "Direction angle",
                    "cos(theta) = (u^T M v) / (|u|_M |v|_M)",
                    f"theta = {primary:.12g} deg",
                ),
                DerivationStep(
                    "Axis angle",
                    "theta_axis = min(theta, 180 deg - theta)",
                    f"theta_axis = {secondary:.12g} deg",
                ),
            )
        elif isinstance(left, Plane) and isinstance(right, Plane):
            primary = interplanar_angle_deg(left, right, phase.lattice)
            secondary = plane_normal_angle_deg(left, right, phase.lattice)
            comparison_kind = ComparisonKind.PLANE_PLANE
            primary_name = "interplanar_angle"
            secondary_name = "oriented_normal_angle"
            derivation = (
                DerivationStep(
                    "Plane angle",
                    "cos(theta) = |p^T M^-1 q| / (|p|_* |q|_*)",
                    f"theta = {primary:.12g} deg",
                ),
                DerivationStep(
                    "Oriented normal angle",
                    "cos(theta_n) = (p^T M^-1 q) / (|p|_* |q|_*)",
                    f"theta_n = {secondary:.12g} deg",
                ),
            )
        else:
            direction = left if isinstance(left, Direction) else right
            plane = left if isinstance(left, Plane) else right
            primary = direction_plane_angle_deg(direction, plane, phase.lattice)
            secondary = None
            incidence = incidence_residual(direction, plane, phase.lattice)
            is_in_plane = lies_in_plane(
                direction,
                plane,
                phase.lattice,
                tolerance=self.project.numerical_policy.algebraic,
            )
            comparison_kind = ComparisonKind.DIRECTION_PLANE
            primary_name = "direction_plane_angle"
            secondary_name = ""
            derivation = (
                DerivationStep(
                    "Incidence residual",
                    "r = |p^T u| / (|u|_M |p|_*)",
                    f"r = {incidence:.12g}",
                ),
                DerivationStep(
                    "Direction-plane angle",
                    "alpha = asin(r)",
                    f"alpha = {primary:.12g} deg",
                ),
            )

        cartesian_values = self._cartesian_comparison_values(left, right, phase)
        if comparison_kind is ComparisonKind.DIRECTION_DIRECTION:
            reference_for_parity = primary
        else:
            reference_for_parity = primary

        residuals = [
            _relative_scalar_residual(reference_for_parity, value)
            for value in cartesian_values
        ]

        return ConsoleComparisonReport(
            phase_id=phase.phase_id,
            left=left_parsed,
            right=right_parsed,
            comparison_kind=comparison_kind,
            primary_name=primary_name,
            primary_value=primary,
            primary_unit="deg",
            secondary_name=secondary_name,
            secondary_value=secondary,
            secondary_unit="deg" if secondary is not None else "",
            incidence_residual_value=incidence,
            lies_in_plane_value=is_in_plane,
            maximum_representation_residual=max(residuals, default=0.0),
            derivation=derivation,
        )

    def map(
        self,
        transformation_query: str,
        source_phase_query: str,
        text: str,
        *,
        kind_hint: str | ConsoleInputKind | None = None,
    ) -> ConsoleMappingReport:
        transformation = self.resolve_transformation(transformation_query)
        source_phase = self.resolve_phase(source_phase_query)
        parsed = parse_crystal_input(text, kind_hint=kind_hint)
        obj = self._object(parsed, source_phase)

        if source_phase.phase_id not in {
            transformation.parent_phase_id,
            transformation.product_phase_id,
        }:
            raise ValueError(
                f"Phase {source_phase.phase_id!r} is not an endpoint of "
                f"transformation {transformation.transformation_id!r}."
            )

        if isinstance(obj, Direction):
            mapped = self.project.map_direction(
                transformation.transformation_id,
                obj,
            )
            law = "u_target = C u_source (or C^-1 for reverse mapping)"
            mapped_kind = ConsoleInputKind.DIRECTION
        else:
            mapped = self.project.map_plane(
                transformation.transformation_id,
                obj,
            )
            law = "p_target = C^-T p_source (or C^T for reverse mapping)"
            mapped_kind = ConsoleInputKind.PLANE

        target_phase_id = mapped.basis.phase_id
        mapped_text = _canonical_notation(
            mapped.indices,
            mapped_kind,
            family=parsed.family,
        )
        return ConsoleMappingReport(
            transformation_id=transformation.transformation_id,
            source_phase_id=source_phase.phase_id,
            target_phase_id=target_phase_id,
            source=parsed,
            mapped_object_payload=mapped.to_dict(),
            mapped_canonical_text=mapped_text,
            derivation=(
                DerivationStep(
                    "Cross-phase mapping",
                    law,
                    f"{parsed.canonical_text} -> {mapped_text}",
                ),
            ),
        )


class ConsoleRenderer:
    """Plain-text renderer; future GUI consumes the same structured reports."""

    @staticmethod
    def _vector(values: tuple[float, float, float]) -> str:
        return "[" + ", ".join(f"{value:.8g}" for value in values) + "]"

    def object_report(
        self,
        report: ConsoleObjectReport,
        *,
        show_derivation: bool = False,
    ) -> str:
        lines = [
            "=" * 72,
            f"{report.phase_label}  |  {report.parsed.canonical_text}",
            "=" * 72,
            f"Physical phase : {report.physical_phase}",
            f"Cell / basis   : {report.cell_representation} / {report.basis_id}",
            f"Object type    : {report.parsed.kind.value}",
            f"Family input   : {'yes' if report.parsed.family else 'no'}",
            "",
            "METRIC",
            (
                f"  {report.dimensional_quantity_name:24s}"
                f" {report.dimensional_quantity:.12g} "
                f"{report.dimensional_unit_label}"
            ),
        ]
        if report.secondary_quantity is not None:
            lines.append(
                f"  {report.secondary_quantity_name:24s}"
                f" {report.secondary_quantity:.12g} "
                f"{report.secondary_unit_label}"
            )

        lines.extend(
            [
                (
                    "  "
                    + (
                        "metric-unit direct coeffs      "
                        if report.parsed.kind is ConsoleInputKind.DIRECTION
                        else "metric-unit reciprocal coeffs  "
                    )
                    + self._vector(report.metric_normalized_coordinates)
                ),
                "",
                "CARTESIAN VIEWS",
            ]
        )
        cartesian_labels = {
            "legacy_a_x_b_xy": "Cartesian legacy (x||a, b in xy)",
            "ptclab_a_x_c_xz": "PTCLab Cartesian (x||a, c in xz)",
            "symmetric_metric": "Metric-symmetric Cartesian (B=M^1/2)",
        }
        for view in report.cartesian_views:
            label = cartesian_labels.get(view.convention, view.convention)
            lines.append(
                f"  {label:40s} {self._vector(view.coordinates)} "
                f"{report.dimensional_unit_label}"
            )
        lines.extend(
            [
                "",
                "RELIABILITY",
                (
                    f"  representation residual     "
                    f"{report.maximum_representation_residual:.3e}"
                ),
                "",
                "SYMMETRY",
                f"  equivalents                  {report.symmetry_equivalent_count}",
                "  " + "  ".join(report.symmetry_equivalents),
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

    def comparison_report(
        self,
        report: ConsoleComparisonReport,
        *,
        show_derivation: bool = False,
    ) -> str:
        lines = [
            "=" * 72,
            (
                f"COMPARE  {report.left.canonical_text}  ↔  "
                f"{report.right.canonical_text}"
            ),
            "=" * 72,
            f"Phase        : {report.phase_id}",
            f"Relation     : {report.comparison_kind.value}",
            (
                f"{report.primary_name:14s}: "
                f"{report.primary_value:.12g} {report.primary_unit}"
            ),
        ]
        if report.secondary_value is not None:
            lines.append(
                f"{report.secondary_name:14s}: "
                f"{report.secondary_value:.12g} {report.secondary_unit}"
            )
        if report.incidence_residual_value is not None:
            lines.append(f"incidence r  : {report.incidence_residual_value:.12g}")
            lines.append(
                f"lies in plane: {'yes' if report.lies_in_plane_value else 'no'}"
            )

        lines.extend(
            [
                (
                    f"representation residual: "
                    f"{report.maximum_representation_residual:.3e}"
                ),
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
        return "\n".join(lines)

    @staticmethod
    def mapping_report(
        report: ConsoleMappingReport,
        *,
        show_derivation: bool = False,
    ) -> str:
        lines = [
            "=" * 72,
            f"MAP  {report.transformation_id}",
            "=" * 72,
            f"source phase : {report.source_phase_id}",
            f"target phase : {report.target_phase_id}",
            f"source       : {report.source.canonical_text}",
            f"mapped       : {report.mapped_canonical_text}",
        ]
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


def reference_console() -> CrystallographyConsole:
    """Reference CLI project; never presented as an unknown specimen default."""

    from .project_state import james_hane_6m_reference_project

    return CrystallographyConsole(james_hane_6m_reference_project())


_HELP = """Commands
  phases
      list available phases

  use <phase>
      select the current phase; examples: use 6m, use do3

  inspect <object>
      examples: inspect [1 0 -1]
                inspect (0 1 1)
                inspect <1 0 0>
                inspect {1 1 0}

  compare <object A> ; <object B>
      example: compare [1 0 1] ; (0 1 1)

  transformations
      list available transformations

  map <transformation> <object>
      maps from the currently selected phase
      example: map do3_to_6m_reference [1 0 0]

  derive on|off
      show or hide derivation steps

  help
  quit

Input rules
  [u v w]  direct direction
  (h k l)  reciprocal plane
  <u v w>  direction/axis family
  {h k l}  plane family

  Spaces or commas are accepted.
  Fractions such as -1/2 are accepted.
  Bare triplets require an explicit object type in the programmatic API.
"""


def _interactive_repl(console: CrystallographyConsole) -> int:
    renderer = ConsoleRenderer()
    current_phase = console.project.phases[0].phase_id
    show_derivation = False

    print("CuAlNi-CT Crystallography Console")
    print("Reference project:", console.project.title)
    print("Type 'help' for commands.")
    print(f"Current phase: {current_phase}")

    while True:
        try:
            line = input("cualni> ").strip()
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
                print(json.dumps(console.phase_choices(), indent=2))
                continue
            if command == "transformations":
                print(json.dumps(console.transformation_choices(), indent=2))
                continue
            if command == "use":
                phase = console.resolve_phase(remainder)
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
            if command == "inspect":
                report = console.inspect(current_phase, remainder)
                print(
                    renderer.object_report(
                        report,
                        show_derivation=show_derivation,
                    )
                )
                continue
            if command == "compare":
                left, separator, right = remainder.partition(";")
                if not separator:
                    raise ValueError(
                        "Separate the two objects with ';'. "
                        "Example: compare [1 0 1] ; (0 1 1)"
                    )
                report = console.compare(current_phase, left, right)
                print(
                    renderer.comparison_report(
                        report,
                        show_derivation=show_derivation,
                    )
                )
                continue
            if command == "map":
                transformation_query, separator, object_text = remainder.partition(" ")
                if not separator or not object_text.strip():
                    raise ValueError(
                        "Use: map <transformation> <object>. "
                        "Example: map do3_to_6m_reference [1 0 0]"
                    )
                report = console.map(
                    transformation_query,
                    current_phase,
                    object_text,
                )
                print(
                    renderer.mapping_report(
                        report,
                        show_derivation=show_derivation,
                    )
                )
                continue

            print("Unknown command. Type 'help'.")
        except (KeyError, ValueError) as exc:
            print(f"INPUT ERROR: {exc}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CuAlNi-CT crystallography console using the verified "
            "ProjectState/typed-object backend."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("phases")
    subparsers.add_parser("transformations")

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--phase", required=True)
    inspect_parser.add_argument("--derive", action="store_true")
    inspect_parser.add_argument("object_text")

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--phase", required=True)
    compare_parser.add_argument("--derive", action="store_true")
    compare_parser.add_argument("left")
    compare_parser.add_argument("right")

    map_parser = subparsers.add_parser("map")
    map_parser.add_argument("--phase", required=True)
    map_parser.add_argument("--transformation", required=True)
    map_parser.add_argument("--derive", action="store_true")
    map_parser.add_argument("object_text")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    console = reference_console()
    renderer = ConsoleRenderer()

    if args.command is None:
        return _interactive_repl(console)
    if args.command == "phases":
        print(json.dumps(console.phase_choices(), indent=2))
        return 0
    if args.command == "transformations":
        print(json.dumps(console.transformation_choices(), indent=2))
        return 0
    if args.command == "inspect":
        report = console.inspect(args.phase, args.object_text)
        print(renderer.object_report(report, show_derivation=args.derive))
        return 0
    if args.command == "compare":
        report = console.compare(args.phase, args.left, args.right)
        print(renderer.comparison_report(report, show_derivation=args.derive))
        return 0
    if args.command == "map":
        report = console.map(
            args.transformation,
            args.phase,
            args.object_text,
        )
        print(renderer.mapping_report(report, show_derivation=args.derive))
        return 0

    parser.error(f"Unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
