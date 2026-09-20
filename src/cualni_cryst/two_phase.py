from __future__ import annotations

"""General PTCLab-style two-phase crystallography workbench.

This module is deliberately theory-neutral.  It consumes a physical
OrientationState and the already verified typed/metric crystallography core.

Scientific boundaries
---------------------
- physical orientation R is not crystallographic correspondence C;
- direct directions [uvw] and reciprocal planes (hkl) remain distinct;
- all non-cubic geometry is metric/Cartesian correct;
- coordinate transforms are derived from the physical OR, not from C;
- low-index results are approximations unless the angular residual is within
  the explicit project tolerance;
- variant/equivalent tables are geometric tools, not Cayron-only theory tools.

The Cayron/CT, Ball-James, PTMC, and experimental layers can all provide an
OrientationState to this general workbench without owning the workbench.
"""

import argparse
import json
import re
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from .calpad import primitive_low_index_triplets
from .crystal_objects import (
    Direction,
    Plane,
    equivalent_directions,
    equivalent_planes,
)
from .crystallography_console import (
    ConsoleInputKind,
    ParsedCrystalInput,
    parse_crystal_input,
)
from .orientation import (
    EulerConvention,
    OrientationService,
    OrientationState,
)
from .project_io import load_project
from .project_state import ProjectState
from .representation import CartesianConvention, CartesianFrame
from .units import length_unit_symbol

Index3f = tuple[float, float, float]
Index3i = tuple[int, int, int]
CrystalObject = Direction | Plane


class WorkbenchSide(str, Enum):
    REFERENCE = "reference"
    MOVING = "moving"


@dataclass(frozen=True)
class LowIndexMatch:
    notation: str
    indices: Index3i
    angular_mismatch_deg: float
    search_max_index: int
    exact_within_tolerance: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "notation": self.notation,
            "indices": list(self.indices),
            "angular_mismatch_deg": self.angular_mismatch_deg,
            "search_max_index": self.search_max_index,
            "exact_within_tolerance": self.exact_within_tolerance,
        }


@dataclass(frozen=True)
class MappingReport:
    orientation_id: str
    source_phase_id: str
    target_phase_id: str
    source_object: str
    object_kind: str
    exact_target_coefficients: Index3f
    projective_target_coefficients: Index3f
    nearest_low_index: LowIndexMatch
    roundtrip_projective_residual: float
    physical_quantity_name: str
    physical_quantity: float
    physical_quantity_unit: str
    derivation: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "orientation_id": self.orientation_id,
            "source_phase_id": self.source_phase_id,
            "target_phase_id": self.target_phase_id,
            "source_object": self.source_object,
            "object_kind": self.object_kind,
            "exact_target_coefficients": list(self.exact_target_coefficients),
            "projective_target_coefficients": list(self.projective_target_coefficients),
            "nearest_low_index": self.nearest_low_index.to_dict(),
            "roundtrip_projective_residual": self.roundtrip_projective_residual,
            "physical_quantity_name": self.physical_quantity_name,
            "physical_quantity": self.physical_quantity,
            "physical_quantity_unit": self.physical_quantity_unit,
            "derivation": list(self.derivation),
        }


@dataclass(frozen=True)
class PairReport:
    orientation_id: str
    reference_phase_id: str
    moving_phase_id: str
    reference_object: str
    moving_object: str
    relation: str
    angle_deg: float
    projective: bool
    reference_quantity_name: str
    reference_quantity: float
    moving_quantity_name: str
    moving_quantity: float
    reference_quantity_unit: str
    moving_quantity_unit: str
    signed_relative_misfit_percent: float | None
    absolute_relative_misfit_percent: float | None
    moving_to_reference: MappingReport
    reference_to_moving: MappingReport
    derivation: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "orientation_id": self.orientation_id,
            "reference_phase_id": self.reference_phase_id,
            "moving_phase_id": self.moving_phase_id,
            "reference_object": self.reference_object,
            "moving_object": self.moving_object,
            "relation": self.relation,
            "angle_deg": self.angle_deg,
            "projective": self.projective,
            "reference_quantity_name": self.reference_quantity_name,
            "reference_quantity": self.reference_quantity,
            "moving_quantity_name": self.moving_quantity_name,
            "moving_quantity": self.moving_quantity,
            "reference_quantity_unit": self.reference_quantity_unit,
            "moving_quantity_unit": self.moving_quantity_unit,
            "signed_relative_misfit_percent": (self.signed_relative_misfit_percent),
            "absolute_relative_misfit_percent": (self.absolute_relative_misfit_percent),
            "moving_to_reference": self.moving_to_reference.to_dict(),
            "reference_to_moving": self.reference_to_moving.to_dict(),
            "derivation": list(self.derivation),
        }


@dataclass(frozen=True)
class CoordinateMatricesReport:
    orientation_id: str
    reference_phase_id: str
    moving_phase_id: str
    direct_reference_from_moving: tuple[Index3f, Index3f, Index3f]
    direct_moving_from_reference: tuple[Index3f, Index3f, Index3f]
    plane_reference_from_moving: tuple[Index3f, Index3f, Index3f]
    plane_moving_from_reference: tuple[Index3f, Index3f, Index3f]
    direct_inverse_residual: float
    plane_inverse_residual: float
    note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "orientation_id": self.orientation_id,
            "reference_phase_id": self.reference_phase_id,
            "moving_phase_id": self.moving_phase_id,
            "direct_reference_from_moving": [
                list(row) for row in self.direct_reference_from_moving
            ],
            "direct_moving_from_reference": [
                list(row) for row in self.direct_moving_from_reference
            ],
            "plane_reference_from_moving": [
                list(row) for row in self.plane_reference_from_moving
            ],
            "plane_moving_from_reference": [
                list(row) for row in self.plane_moving_from_reference
            ],
            "direct_inverse_residual": self.direct_inverse_residual,
            "plane_inverse_residual": self.plane_inverse_residual,
            "note": self.note,
        }


@dataclass(frozen=True)
class SummaryReport:
    orientation_id: str
    reference_phase_id: str
    moving_phase_id: str
    definition_method: str
    theory_origin: str
    transformation_id: str
    reference_frame: str
    moving_frame: str
    ptclab_matrix_reference_from_moving: tuple[Index3f, Index3f, Index3f]
    variant_count: int
    coordinate_matrices: CoordinateMatricesReport
    note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "orientation_id": self.orientation_id,
            "reference_phase_id": self.reference_phase_id,
            "moving_phase_id": self.moving_phase_id,
            "definition_method": self.definition_method,
            "theory_origin": self.theory_origin,
            "transformation_id": self.transformation_id,
            "reference_frame": self.reference_frame,
            "moving_frame": self.moving_frame,
            "ptclab_matrix_reference_from_moving": [
                list(row) for row in self.ptclab_matrix_reference_from_moving
            ],
            "variant_count": self.variant_count,
            "coordinate_matrices": self.coordinate_matrices.to_dict(),
            "note": self.note,
        }


@dataclass(frozen=True)
class VariantAngleRow:
    variant_index: int
    reference_coset_symmetry_indices: tuple[int, ...]
    angle_deg: float
    mapped_moving_projective_coefficients: Index3f
    nearest_reference_low_index: str
    nearest_mismatch_deg: float

    def to_dict(self) -> dict[str, object]:
        return {
            "variant_index": self.variant_index,
            "reference_coset_symmetry_indices": list(
                self.reference_coset_symmetry_indices
            ),
            "angle_deg": self.angle_deg,
            "mapped_moving_projective_coefficients": list(
                self.mapped_moving_projective_coefficients
            ),
            "nearest_reference_low_index": self.nearest_reference_low_index,
            "nearest_mismatch_deg": self.nearest_mismatch_deg,
        }


@dataclass(frozen=True)
class VariantAngleTable:
    orientation_id: str
    reference_object: str
    moving_object: str
    projective: bool
    rows: tuple[VariantAngleRow, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "orientation_id": self.orientation_id,
            "reference_object": self.reference_object,
            "moving_object": self.moving_object,
            "projective": self.projective,
            "row_count": len(self.rows),
            "rows": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True)
class EquivalentAngleRow:
    rank: int
    reference_object: str
    moving_object: str
    angle_deg: float

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "reference_object": self.reference_object,
            "moving_object": self.moving_object,
            "angle_deg": self.angle_deg,
        }


@dataclass(frozen=True)
class EquivalentAngleTable:
    orientation_id: str
    reference_seed: str
    moving_seed: str
    projective: bool
    proper_only: bool
    total_pairs: int
    limit: int
    rows: tuple[EquivalentAngleRow, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "orientation_id": self.orientation_id,
            "reference_seed": self.reference_seed,
            "moving_seed": self.moving_seed,
            "projective": self.projective,
            "proper_only": self.proper_only,
            "total_pairs": self.total_pairs,
            "limit": self.limit,
            "rows": [row.to_dict() for row in self.rows],
        }


def _matrix_tuple(matrix: np.ndarray) -> tuple[Index3f, Index3f, Index3f]:
    arr = np.asarray(matrix, dtype=float).reshape(3, 3)
    return tuple(tuple(float(value) for value in row) for row in arr)  # type: ignore[return-value]


def _tuple3(values: np.ndarray) -> Index3f:
    arr = np.asarray(values, dtype=float).reshape(3)
    return tuple(float(value) for value in arr)  # type: ignore[return-value]


def _projective_normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(3).copy()
    scale = float(np.max(np.abs(arr)))
    if scale <= 0.0:
        raise ValueError("Cannot normalize the zero crystallographic object.")
    arr /= scale
    arr[np.abs(arr) < 1.0e-14] = 0.0
    nonzero = np.flatnonzero(np.abs(arr) > 1.0e-14)
    if nonzero.size and arr[int(nonzero[0])] < 0.0:
        arr = -arr
    return arr


def _projective_residual(first: np.ndarray, second: np.ndarray) -> float:
    a = _projective_normalize(first)
    b = _projective_normalize(second)
    return float(min(np.linalg.norm(a - b), np.linalg.norm(a + b)))


def _notation(obj: CrystalObject) -> str:
    values = obj.indices
    body = " ".join(f"{value:g}" for value in values)
    return f"[{body}]" if isinstance(obj, Direction) else f"({body})"


def _typed_object(parsed: ParsedCrystalInput, phase) -> CrystalObject:
    if parsed.kind is ConsoleInputKind.DIRECTION:
        return Direction(parsed.indices, phase.basis)
    return Plane(parsed.indices, phase.basis)


def _object_quantity(obj: CrystalObject, phase) -> tuple[str, float, str]:
    unit = length_unit_symbol(phase.lattice.length_unit)
    if isinstance(obj, Direction):
        return "direct_length", obj.length(phase.lattice), unit
    return "interplanar_spacing", obj.spacing(phase.lattice), unit


def _physical_unit_vector(
    obj: CrystalObject,
    phase,
    convention: CartesianConvention,
) -> np.ndarray:
    frame = CartesianFrame(phase.lattice, convention)
    if isinstance(obj, Direction):
        return frame.direct_to_cartesian(obj.array, normalize=True)
    return frame.plane_to_cartesian(obj.array, normalize=True)


def _cross_phase_angle(
    state: OrientationState,
    reference_obj: CrystalObject,
    moving_obj: CrystalObject,
    reference_phase,
    moving_phase,
    *,
    projective: bool,
) -> tuple[str, float]:
    reference_vector = _physical_unit_vector(
        reference_obj,
        reference_phase,
        state.reference_cartesian_convention,
    )
    moving_vector = _physical_unit_vector(
        moving_obj,
        moving_phase,
        state.moving_cartesian_convention,
    )
    moving_in_reference = (
        np.asarray(state.R_reference_from_moving, dtype=float) @ moving_vector
    )

    cosine = float(np.clip(reference_vector @ moving_in_reference, -1.0, 1.0))
    if isinstance(reference_obj, Direction) and isinstance(moving_obj, Direction):
        if projective:
            cosine = abs(cosine)
        return "direction-direction", float(
            np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
        )

    if isinstance(reference_obj, Plane) and isinstance(moving_obj, Plane):
        if projective:
            cosine = abs(cosine)
        return "plane-plane", float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))

    # For a direction and a plane, 0° means incidence/in-plane and 90°
    # means direction parallel to the plane normal. Plane-normal sign is
    # intrinsically irrelevant here.
    return "direction-plane", float(
        np.rad2deg(np.arcsin(np.clip(abs(cosine), 0.0, 1.0)))
    )


def _nearest_low_index(
    obj: CrystalObject,
    phase,
    *,
    max_index: int,
    projective: bool,
    tolerance_deg: float,
) -> LowIndexMatch:
    if max_index < 1:
        raise ValueError("max_index must be >= 1.")

    candidates = primitive_low_index_triplets(
        max_index,
        projective=projective,
    )
    C = np.asarray(candidates, dtype=float)
    target = obj.array

    if isinstance(obj, Direction):
        metric = phase.lattice.metric()
        candidate_metric = np.einsum("ni,ij,nj->n", C, metric, C)
        target_metric = float(target @ metric @ target)
        dots = C @ (metric @ target)
        notation_open, notation_close = "[", "]"
    else:
        metric = phase.lattice.reciprocal_metric()
        candidate_metric = np.einsum("ni,ij,nj->n", C, metric, C)
        target_metric = float(target @ metric @ target)
        dots = C @ (metric @ target)
        notation_open, notation_close = "(", ")"

    denominator = np.sqrt(candidate_metric * target_metric)
    cosines = np.clip(dots / denominator, -1.0, 1.0)
    if projective:
        cosines = np.abs(cosines)
    angles = np.rad2deg(np.arccos(cosines))
    index = int(np.argmin(angles))
    triplet = candidates[index]
    angle = float(angles[index])
    body = " ".join(str(value) for value in triplet)

    return LowIndexMatch(
        notation=f"{notation_open}{body}{notation_close}",
        indices=triplet,
        angular_mismatch_deg=angle,
        search_max_index=max_index,
        exact_within_tolerance=angle <= tolerance_deg,
    )


def _parse_float_vector(text: str, size: int) -> np.ndarray:
    cleaned = text.strip().replace(",", " ")
    tokens = cleaned.split()
    if len(tokens) != size:
        raise ValueError(f"Expected {size} numeric values; got {len(tokens)}.")
    values = np.asarray([float(token) for token in tokens], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Numeric orientation input contains non-finite values.")
    return values


def _parse_matrix(text: str) -> np.ndarray:
    rows = [row.strip() for row in text.split(";") if row.strip()]
    if len(rows) == 3:
        return np.vstack([_parse_float_vector(row, 3) for row in rows])
    return _parse_float_vector(text.replace(";", " "), 9).reshape(3, 3)


class TwoPhaseWorkbench:
    """Theory-neutral PTCLab-style two-crystal workbench."""

    def __init__(self, project: ProjectState):
        self.project = project
        self.project.validate().assert_passed()
        self.orientations = OrientationService(project)

    def _resolve_phase_pair(
        self,
        reference: str,
        moving: str,
    ) -> tuple[str, str]:
        reference = reference.strip()
        moving = moving.strip()
        phases = self.project.phases

        if reference and moving:
            ref = self.orientations.resolve_phase(reference)
            mov = self.orientations.resolve_phase(moving)
            if ref.phase_id == mov.phase_id:
                raise ValueError("Reference and moving phases must be distinct.")
            return ref.phase_id, mov.phase_id

        if len(phases) == 2:
            if reference:
                ref = self.orientations.resolve_phase(reference)
                other = next(
                    phase for phase in phases if phase.phase_id != ref.phase_id
                )
                return ref.phase_id, other.phase_id
            if moving:
                mov = self.orientations.resolve_phase(moving)
                other = next(
                    phase for phase in phases if phase.phase_id != mov.phase_id
                )
                return other.phase_id, mov.phase_id
            return phases[0].phase_id, phases[1].phase_id

        choices = ", ".join(phase.phase_id for phase in phases)
        raise ValueError(
            "This project does not have exactly two phases. Supply both "
            f"--reference and --moving explicitly. Available phases: {choices}"
        )

    def resolve_orientation(
        self,
        spec: str,
        *,
        reference: str = "",
        moving: str = "",
        bind_transformation: str = "",
        parallel_candidate: int = 1,
    ) -> OrientationState:
        """Resolve one compact, explicit OR specification.

        Supported forms:
            polar:TRANSFORMATION_ID
            stored:ORIENTATION_ID
            matrix:9 values or 3 semicolon-separated rows
            euler:phi1 Phi phi2
            quat:w x y z
            axis:x y z ; angle_deg
            axis-crystal:[u v w] ; angle_deg
            parallel:REF1|MOV1|REF2|MOV2
            identity

        No correspondence is inferred from phase names. `bind_transformation`
        is optional metadata for non-polar ORs.
        """

        raw = spec.strip()
        if not raw:
            raise ValueError("Orientation specification must be non-empty.")

        # Stored ORs and polar rotations already carry phase endpoints. Every
        # user-defined OR form requires an explicit or unambiguous phase pair.
        preview_kind = ""
        if ":" in raw:
            preview_kind = re.sub(r"[^a-z-]+", "", raw.split(":", 1)[0].lower())
        if raw.lower() == "identity" or preview_kind not in {"polar", "stored"}:
            reference, moving = self._resolve_phase_pair(reference, moving)

        if raw.lower() == "identity":
            state = self.orientations.state_from_matrix(
                reference,
                moving,
                np.eye(3),
            )
        else:
            if ":" not in raw:
                raise ValueError(
                    "Orientation specification must be one of polar:, stored:, "
                    "matrix:, euler:, quat:, axis:, axis-crystal:, parallel:, "
                    "or identity."
                )
            kind, payload = raw.split(":", 1)
            kind = re.sub(r"[^a-z-]+", "", kind.lower())

            if kind == "polar":
                state = self.orientations.polar_orientation(payload.strip())
            elif kind == "stored":
                state = self.project.orientation(payload.strip())
            elif kind == "matrix":
                state = self.orientations.state_from_matrix(
                    reference,
                    moving,
                    _parse_matrix(payload),
                )
            elif kind == "euler":
                values = _parse_float_vector(payload, 3)
                state = self.orientations.state_from_euler(
                    reference,
                    moving,
                    float(values[0]),
                    float(values[1]),
                    float(values[2]),
                    convention=EulerConvention.ZXZ_ACTIVE,
                )
            elif kind in {"quat", "quaternion"}:
                state = self.orientations.state_from_quaternion(
                    reference,
                    moving,
                    _parse_float_vector(payload, 4),
                )
            elif kind == "axis":
                parts = [item.strip() for item in payload.split(";")]
                if len(parts) != 2:
                    raise ValueError("axis OR syntax is 'axis:x y z ; angle_deg'.")
                state = self.orientations.state_from_axis_angle(
                    reference,
                    moving,
                    _parse_float_vector(parts[0], 3),
                    float(parts[1]),
                )
            elif kind in {"axis-crystal", "crystal-axis"}:
                parts = [item.strip() for item in payload.split(";")]
                if len(parts) != 2:
                    raise ValueError(
                        "axis-crystal OR syntax is 'axis-crystal:[u v w] ; angle_deg'."
                    )
                state = self.orientations.state_from_reference_crystal_axis(
                    reference,
                    moving,
                    parts[0],
                    float(parts[1]),
                )
            elif kind == "parallel":
                parts = [item.strip() for item in payload.split("|")]
                if len(parts) != 4:
                    raise ValueError(
                        "parallel OR syntax is 'parallel:REF1|MOV1|REF2|MOV2'."
                    )
                solved = self.orientations.state_from_parallelisms(
                    reference,
                    moving,
                    parts[0],
                    parts[1],
                    parts[2],
                    parts[3],
                )
                if not 1 <= parallel_candidate <= len(solved.candidates):
                    raise ValueError(
                        f"parallel_candidate must be between 1 and "
                        f"{len(solved.candidates)}."
                    )
                state = solved.candidates[parallel_candidate - 1].state
            else:
                raise ValueError(f"Unknown orientation specification {kind!r}.")

        if reference:
            expected_reference = self.orientations.resolve_phase(reference).phase_id
            if expected_reference != state.reference_phase_id:
                raise ValueError(
                    f"Explicit reference phase {expected_reference!r} conflicts "
                    f"with OR reference phase {state.reference_phase_id!r}."
                )
        if moving:
            expected_moving = self.orientations.resolve_phase(moving).phase_id
            if expected_moving != state.moving_phase_id:
                raise ValueError(
                    f"Explicit moving phase {expected_moving!r} conflicts "
                    f"with OR moving phase {state.moving_phase_id!r}."
                )

        if bind_transformation:
            transformation = self.project.transformation(bind_transformation)
            if (
                transformation.parent_phase_id != state.reference_phase_id
                or transformation.product_phase_id != state.moving_phase_id
            ):
                raise ValueError(
                    f"Transformation {bind_transformation!r} connects "
                    f"{transformation.parent_phase_id} -> "
                    f"{transformation.product_phase_id}, but the OR connects "
                    f"{state.reference_phase_id} <- {state.moving_phase_id}."
                )
            state = replace(state, transformation_id=bind_transformation)
        return state

    def coordinate_matrices(
        self,
        state: OrientationState,
    ) -> CoordinateMatricesReport:
        reference = self.project.phase(state.reference_phase_id)
        moving = self.project.phase(state.moving_phase_id)
        basis = np.eye(3)

        d_ref_from_mov = np.column_stack(
            [
                self.orientations.map_direction(
                    state,
                    Direction(tuple(basis[:, i]), moving.basis),
                ).array
                for i in range(3)
            ]
        )
        d_mov_from_ref = np.column_stack(
            [
                self.orientations.map_direction(
                    state,
                    Direction(tuple(basis[:, i]), reference.basis),
                ).array
                for i in range(3)
            ]
        )
        p_ref_from_mov = np.column_stack(
            [
                self.orientations.map_plane(
                    state,
                    Plane(tuple(basis[:, i]), moving.basis),
                ).array
                for i in range(3)
            ]
        )
        p_mov_from_ref = np.column_stack(
            [
                self.orientations.map_plane(
                    state,
                    Plane(tuple(basis[:, i]), reference.basis),
                ).array
                for i in range(3)
            ]
        )

        direct_residual = float(
            np.linalg.norm(d_mov_from_ref @ d_ref_from_mov - np.eye(3))
        )
        plane_residual = float(
            np.linalg.norm(p_mov_from_ref @ p_ref_from_mov - np.eye(3))
        )

        return CoordinateMatricesReport(
            orientation_id=state.orientation_id,
            reference_phase_id=state.reference_phase_id,
            moving_phase_id=state.moving_phase_id,
            direct_reference_from_moving=_matrix_tuple(d_ref_from_mov),
            direct_moving_from_reference=_matrix_tuple(d_mov_from_ref),
            plane_reference_from_moving=_matrix_tuple(p_ref_from_mov),
            plane_moving_from_reference=_matrix_tuple(p_mov_from_ref),
            direct_inverse_residual=direct_residual,
            plane_inverse_residual=plane_residual,
            note=(
                "Direct and reciprocal coordinate transforms are physical OR "
                "mappings derived from R and the two lattices. They are not "
                "crystallographic correspondence C."
            ),
        )

    def summary(self, state: OrientationState) -> SummaryReport:
        ptclab = self.orientations.reexpress(
            state,
            CartesianConvention.PTCLAB_A_X_C_XZ,
            CartesianConvention.PTCLAB_A_X_C_XZ,
        )
        variants = self.orientations.variants(state)
        return SummaryReport(
            orientation_id=state.orientation_id,
            reference_phase_id=state.reference_phase_id,
            moving_phase_id=state.moving_phase_id,
            definition_method=state.definition_method.value,
            theory_origin=state.theory_origin.value,
            transformation_id=state.transformation_id,
            reference_frame=state.reference_cartesian_convention.value,
            moving_frame=state.moving_cartesian_convention.value,
            ptclab_matrix_reference_from_moving=_matrix_tuple(
                np.asarray(ptclab.R_reference_from_moving, dtype=float)
            ),
            variant_count=len(variants),
            coordinate_matrices=self.coordinate_matrices(state),
            note=(
                "General two-phase crystallography workbench. Cayron CT, "
                "Ball-James, PTMC, polar(F), literature, or experimental ORs "
                "can all be supplied as OrientationState inputs."
            ),
        )

    def map_object(
        self,
        state: OrientationState,
        source_side: WorkbenchSide | str,
        object_text: str,
        *,
        max_index: int = 12,
        projective: bool = True,
    ) -> MappingReport:
        side = WorkbenchSide(source_side)
        reference = self.project.phase(state.reference_phase_id)
        moving = self.project.phase(state.moving_phase_id)
        parsed = parse_crystal_input(object_text)

        if side is WorkbenchSide.MOVING:
            source_phase = moving
            target_phase = reference
        else:
            source_phase = reference
            target_phase = moving

        source_obj = _typed_object(parsed, source_phase)
        if isinstance(source_obj, Direction):
            mapped = self.orientations.map_direction(state, source_obj)
            roundtrip = self.orientations.map_direction(state, mapped)
            object_kind = "direction"
            direct_formula = (
                "u_ref = B_ref^-1 R_ref<-mov B_mov u_mov"
                if side is WorkbenchSide.MOVING
                else "u_mov = B_mov^-1 R_ref<-mov^T B_ref u_ref"
            )
        else:
            mapped = self.orientations.map_plane(state, source_obj)
            roundtrip = self.orientations.map_plane(state, mapped)
            object_kind = "plane"
            direct_formula = (
                "p_ref = B_ref^T R_ref<-mov B_mov^-T p_mov"
                if side is WorkbenchSide.MOVING
                else "p_mov = B_mov^T R_ref<-mov^T B_ref^-T p_ref"
            )

        nearest = _nearest_low_index(
            mapped,
            target_phase,
            max_index=max_index,
            projective=projective,
            tolerance_deg=self.project.numerical_policy.projective_angle_deg,
        )
        quantity_name, quantity, quantity_unit = _object_quantity(
            source_obj,
            source_phase,
        )

        return MappingReport(
            orientation_id=state.orientation_id,
            source_phase_id=source_phase.phase_id,
            target_phase_id=target_phase.phase_id,
            source_object=parsed.canonical_text,
            object_kind=object_kind,
            exact_target_coefficients=_tuple3(mapped.array),
            projective_target_coefficients=_tuple3(_projective_normalize(mapped.array)),
            nearest_low_index=nearest,
            roundtrip_projective_residual=_projective_residual(
                roundtrip.array,
                source_obj.array,
            ),
            physical_quantity_name=quantity_name,
            physical_quantity=quantity,
            physical_quantity_unit=quantity_unit,
            derivation=(
                direct_formula,
                (
                    "nearest low-index object is a bounded metric search and "
                    "is not an exact identity unless the angular residual "
                    "passes the project tolerance"
                ),
            ),
        )

    def pair(
        self,
        state: OrientationState,
        reference_text: str,
        moving_text: str,
        *,
        max_index: int = 12,
        projective: bool = True,
    ) -> PairReport:
        reference = self.project.phase(state.reference_phase_id)
        moving = self.project.phase(state.moving_phase_id)
        ref_parsed = parse_crystal_input(reference_text)
        mov_parsed = parse_crystal_input(moving_text)
        ref_obj = _typed_object(ref_parsed, reference)
        mov_obj = _typed_object(mov_parsed, moving)

        relation, angle = _cross_phase_angle(
            state,
            ref_obj,
            mov_obj,
            reference,
            moving,
            projective=projective,
        )

        ref_q_name, ref_q, ref_unit = _object_quantity(ref_obj, reference)
        mov_q_name, mov_q, mov_unit = _object_quantity(mov_obj, moving)
        signed_misfit: float | None = None
        absolute_misfit: float | None = None
        if type(ref_obj) is type(mov_obj) and ref_unit == mov_unit:
            signed_misfit = 100.0 * (mov_q - ref_q) / ref_q
            absolute_misfit = abs(signed_misfit)

        return PairReport(
            orientation_id=state.orientation_id,
            reference_phase_id=reference.phase_id,
            moving_phase_id=moving.phase_id,
            reference_object=ref_parsed.canonical_text,
            moving_object=mov_parsed.canonical_text,
            relation=relation,
            angle_deg=angle,
            projective=projective,
            reference_quantity_name=ref_q_name,
            reference_quantity=ref_q,
            moving_quantity_name=mov_q_name,
            moving_quantity=mov_q,
            reference_quantity_unit=ref_unit,
            moving_quantity_unit=mov_unit,
            signed_relative_misfit_percent=signed_misfit,
            absolute_relative_misfit_percent=absolute_misfit,
            moving_to_reference=self.map_object(
                state,
                WorkbenchSide.MOVING,
                moving_text,
                max_index=max_index,
                projective=projective,
            ),
            reference_to_moving=self.map_object(
                state,
                WorkbenchSide.REFERENCE,
                reference_text,
                max_index=max_index,
                projective=projective,
            ),
            derivation=(
                (
                    "cross-phase comparison maps the moving physical vector/"
                    "plane normal through R_ref<-mov before taking the angle"
                ),
                (
                    "direction-plane angle is measured from the plane itself: "
                    "0 deg = incidence, 90 deg = parallel to plane normal"
                ),
                (
                    "same-kind misfit uses reference quantity in the denominator: "
                    "100*(moving-reference)/reference"
                ),
            ),
        )

    def variant_angles(
        self,
        state: OrientationState,
        reference_text: str,
        moving_text: str,
        *,
        max_index: int = 12,
        projective: bool = True,
    ) -> VariantAngleTable:
        reference = self.project.phase(state.reference_phase_id)
        moving = self.project.phase(state.moving_phase_id)
        ref_parsed = parse_crystal_input(reference_text)
        mov_parsed = parse_crystal_input(moving_text)
        ref_obj = _typed_object(ref_parsed, reference)
        mov_obj = _typed_object(mov_parsed, moving)

        rows: list[VariantAngleRow] = []
        for variant in self.orientations.variants(state):
            current = replace(
                state,
                orientation_id=f"{state.orientation_id}:variant:{variant.index}",
                R_reference_from_moving=variant.matrix_reference_from_moving,
            )
            _, angle = _cross_phase_angle(
                current,
                ref_obj,
                mov_obj,
                reference,
                moving,
                projective=projective,
            )
            mapped = self.map_object(
                current,
                WorkbenchSide.MOVING,
                moving_text,
                max_index=max_index,
                projective=projective,
            )
            rows.append(
                VariantAngleRow(
                    variant_index=variant.index,
                    reference_coset_symmetry_indices=(
                        variant.reference_coset_symmetry_indices
                    ),
                    angle_deg=angle,
                    mapped_moving_projective_coefficients=(
                        mapped.projective_target_coefficients
                    ),
                    nearest_reference_low_index=(mapped.nearest_low_index.notation),
                    nearest_mismatch_deg=(
                        mapped.nearest_low_index.angular_mismatch_deg
                    ),
                )
            )

        rows.sort(key=lambda row: (row.angle_deg, row.variant_index))
        return VariantAngleTable(
            orientation_id=state.orientation_id,
            reference_object=ref_parsed.canonical_text,
            moving_object=mov_parsed.canonical_text,
            projective=projective,
            rows=tuple(rows),
        )

    def equivalent_angles(
        self,
        state: OrientationState,
        reference_text: str,
        moving_text: str,
        *,
        limit: int = 30,
        projective: bool = True,
        proper_only: bool = False,
    ) -> EquivalentAngleTable:
        if limit < 1:
            raise ValueError("limit must be >= 1.")

        reference = self.project.phase(state.reference_phase_id)
        moving = self.project.phase(state.moving_phase_id)
        ref_parsed = parse_crystal_input(reference_text)
        mov_parsed = parse_crystal_input(moving_text)
        ref_seed = _typed_object(ref_parsed, reference)
        mov_seed = _typed_object(mov_parsed, moving)

        def operators(phase) -> tuple[np.ndarray, ...]:
            all_ops = tuple(
                np.asarray(item, dtype=float) for item in phase.symmetry_matrices()
            )
            if not proper_only:
                return all_ops
            return tuple(item for item in all_ops if float(np.linalg.det(item)) > 0.0)

        ref_ops = operators(reference)
        mov_ops = operators(moving)

        if isinstance(ref_seed, Direction):
            ref_equiv: tuple[CrystalObject, ...] = equivalent_directions(
                ref_seed,
                ref_ops,
                projective=projective,
            )
        else:
            ref_equiv = equivalent_planes(
                ref_seed,
                ref_ops,
                projective=projective,
            )

        if isinstance(mov_seed, Direction):
            mov_equiv: tuple[CrystalObject, ...] = equivalent_directions(
                mov_seed,
                mov_ops,
                projective=projective,
            )
        else:
            mov_equiv = equivalent_planes(
                mov_seed,
                mov_ops,
                projective=projective,
            )

        raw_rows: list[tuple[str, str, float]] = []
        for ref_obj in ref_equiv:
            for mov_obj in mov_equiv:
                _, angle = _cross_phase_angle(
                    state,
                    ref_obj,
                    mov_obj,
                    reference,
                    moving,
                    projective=projective,
                )
                raw_rows.append((_notation(ref_obj), _notation(mov_obj), angle))

        raw_rows.sort(key=lambda item: (item[2], item[0], item[1]))
        shown = raw_rows[:limit]
        rows = tuple(
            EquivalentAngleRow(
                rank=index,
                reference_object=ref_text,
                moving_object=mov_text,
                angle_deg=angle,
            )
            for index, (ref_text, mov_text, angle) in enumerate(shown, start=1)
        )
        return EquivalentAngleTable(
            orientation_id=state.orientation_id,
            reference_seed=ref_parsed.canonical_text,
            moving_seed=mov_parsed.canonical_text,
            projective=projective,
            proper_only=proper_only,
            total_pairs=len(raw_rows),
            limit=limit,
            rows=rows,
        )


class TwoPhaseRenderer:
    """Compact PTCLab-like text output; advanced detail is opt-in."""

    @staticmethod
    def _matrix(matrix) -> list[str]:
        values = np.asarray(matrix, dtype=float).copy()
        values[np.abs(values) < 1.0e-14] = 0.0
        return [
            "    [" + "  ".join(f"{value: .10g}" for value in row) + "]"
            for row in values
        ]

    @staticmethod
    def _coefficients(values: Index3f) -> str:
        return "[" + " ".join(f"{value:.9g}" for value in values) + "]"

    def summary(self, report: SummaryReport) -> str:
        lines = [
            "=" * 88,
            "TWO-PHASE CRYSTALLOGRAPHY WORKBENCH",
            "=" * 88,
            f"reference crystal : {report.reference_phase_id}",
            f"moving crystal    : {report.moving_phase_id}",
            f"orientation       : {report.orientation_id}",
            f"definition        : {report.definition_method}",
            f"origin            : {report.theory_origin}",
            f"transformation    : {report.transformation_id or '-'}",
            f"OR variants       : {report.variant_count}",
            "",
            "PTCLab Cartesian OR matrix  (x||a, c in xz)",
            *self._matrix(report.ptclab_matrix_reference_from_moving),
            "",
            "Use: pair | map | variants | equivalents | matrices",
            "Cayron/CT is optional theory analysis, not required by this workbench.",
        ]
        return "\n".join(lines)

    def mapping(self, report: MappingReport, *, derive: bool = False) -> str:
        nearest = report.nearest_low_index
        lines = [
            "=" * 88,
            "TWO-PHASE COORDINATE MAP",
            "=" * 88,
            f"orientation       : {report.orientation_id}",
            f"source            : {report.source_phase_id}  {report.source_object}",
            f"target            : {report.target_phase_id}",
            f"kind              : {report.object_kind}",
            "",
            "EXACT OR-MAPPED COEFFICIENTS",
            f"  raw             : {self._coefficients(report.exact_target_coefficients)}",
            (
                "  projective      : "
                f"{self._coefficients(report.projective_target_coefficients)}"
            ),
            "",
            "NEAREST LOW-INDEX OBJECT",
            f"  candidate       : {nearest.notation}",
            f"  mismatch        : {nearest.angular_mismatch_deg:.12g} deg",
            f"  search bound    : |index| <= {nearest.search_max_index}",
            (
                "  exact?          : "
                + ("YES" if nearest.exact_within_tolerance else "NO — approximation")
            ),
            "",
            (
                f"{report.physical_quantity_name:18}: "
                f"{report.physical_quantity:.12g} {report.physical_quantity_unit}"
            ),
            (f"round-trip residual: {report.roundtrip_projective_residual:.3e}"),
        ]
        if derive:
            lines.extend(["", "DERIVATION"])
            lines.extend(f"  - {item}" for item in report.derivation)
        return "\n".join(lines)

    def pair(self, report: PairReport, *, derive: bool = False) -> str:
        lines = [
            "=" * 88,
            "TWO-PHASE CALPAD",
            "=" * 88,
            f"orientation       : {report.orientation_id}",
            (
                f"pair              : {report.reference_phase_id} "
                f"{report.reference_object}  vs  "
                f"{report.moving_phase_id} {report.moving_object}"
            ),
            f"relation          : {report.relation}",
            f"angle             : {report.angle_deg:.12g} deg",
            (
                "angle sense       : "
                + ("projective / unoriented" if report.projective else "oriented")
            ),
            "",
            "GEOMETRY",
            (
                f"  reference {report.reference_quantity_name}: "
                f"{report.reference_quantity:.12g} {report.reference_quantity_unit}"
            ),
            (
                f"  moving    {report.moving_quantity_name}: "
                f"{report.moving_quantity:.12g} {report.moving_quantity_unit}"
            ),
        ]
        if report.signed_relative_misfit_percent is not None:
            lines.extend(
                [
                    (
                        "  signed misfit   : "
                        f"{report.signed_relative_misfit_percent:.9g}%"
                    ),
                    (
                        "  absolute misfit : "
                        f"{report.absolute_relative_misfit_percent:.9g}%"
                    ),
                ]
            )

        lines.extend(
            [
                "",
                "MOVING -> REFERENCE",
                (
                    "  exact coeffs     : "
                    f"{self._coefficients(report.moving_to_reference.exact_target_coefficients)}"
                ),
                (
                    "  projective       : "
                    f"{self._coefficients(report.moving_to_reference.projective_target_coefficients)}"
                ),
                (
                    "  nearest low-index: "
                    f"{report.moving_to_reference.nearest_low_index.notation}  "
                    f"({report.moving_to_reference.nearest_low_index.angular_mismatch_deg:.6g} deg)"
                ),
                "",
                "REFERENCE -> MOVING",
                (
                    "  exact coeffs     : "
                    f"{self._coefficients(report.reference_to_moving.exact_target_coefficients)}"
                ),
                (
                    "  projective       : "
                    f"{self._coefficients(report.reference_to_moving.projective_target_coefficients)}"
                ),
                (
                    "  nearest low-index: "
                    f"{report.reference_to_moving.nearest_low_index.notation}  "
                    f"({report.reference_to_moving.nearest_low_index.angular_mismatch_deg:.6g} deg)"
                ),
            ]
        )
        if derive:
            lines.extend(["", "DERIVATION"])
            lines.extend(f"  - {item}" for item in report.derivation)
        return "\n".join(lines)

    def matrices(self, report: CoordinateMatricesReport) -> str:
        return "\n".join(
            [
                "=" * 88,
                "TWO-PHASE COORDINATE TRANSFORMATION MATRICES",
                "=" * 88,
                (
                    f"{report.reference_phase_id} <- "
                    f"{report.moving_phase_id}   ({report.orientation_id})"
                ),
                "",
                "DIRECT: u_ref = A u_mov",
                *self._matrix(report.direct_reference_from_moving),
                "",
                "DIRECT INVERSE: u_mov = A^-1 u_ref",
                *self._matrix(report.direct_moving_from_reference),
                f"  inverse residual: {report.direct_inverse_residual:.3e}",
                "",
                "RECIPROCAL: p_ref = P p_mov",
                *self._matrix(report.plane_reference_from_moving),
                "",
                "RECIPROCAL INVERSE: p_mov = P^-1 p_ref",
                *self._matrix(report.plane_moving_from_reference),
                f"  inverse residual: {report.plane_inverse_residual:.3e}",
                "",
                report.note,
            ]
        )

    def variants(self, report: VariantAngleTable) -> str:
        lines = [
            "=" * 88,
            "OR-VARIANT ANGLE TABLE",
            "=" * 88,
            f"orientation : {report.orientation_id}",
            f"pair        : {report.reference_object} vs {report.moving_object}",
            "",
            " rank  variant   angle(deg)   nearest mapped moving object   mismatch(deg)",
            " ----  -------   ----------   ----------------------------   -------------",
        ]
        for rank, row in enumerate(report.rows, start=1):
            lines.append(
                f" {rank:>4d}  {row.variant_index:>7d}   "
                f"{row.angle_deg:>10.6g}   "
                f"{row.nearest_reference_low_index:>28s}   "
                f"{row.nearest_mismatch_deg:>13.6g}"
            )
        return "\n".join(lines)

    def equivalents(self, report: EquivalentAngleTable) -> str:
        lines = [
            "=" * 88,
            "SYMMETRY-EQUIVALENT CROSS-PHASE ANGLES",
            "=" * 88,
            f"orientation      : {report.orientation_id}",
            f"reference seed   : {report.reference_seed}",
            f"moving seed      : {report.moving_seed}",
            f"symmetry         : {'proper only' if report.proper_only else 'full point group'}",
            f"candidate pairs  : {report.total_pairs}",
            "",
            " rank   reference        moving           angle(deg)",
            " ----   ---------------  ---------------  ----------",
        ]
        for row in report.rows:
            lines.append(
                f" {row.rank:>4d}   {row.reference_object:<15s}  "
                f"{row.moving_object:<15s}  {row.angle_deg:>10.6g}"
            )
        if report.total_pairs > len(report.rows):
            lines.append(
                f"\nShowing {len(report.rows)} of {report.total_pairs}; "
                "increase --limit to show more."
            )
        return "\n".join(lines)


def _add_or_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project",
        default=argparse.SUPPRESS,
        help="Project source: preset:NAME or path to generic project JSON.",
    )
    parser.add_argument(
        "--or",
        dest="or_spec",
        required=True,
        help=(
            "Orientation: polar:ID | stored:ID | matrix:... | euler:... | "
            "quat:... | axis:... | axis-crystal:... | parallel:... | identity"
        ),
    )
    parser.add_argument("--reference", default="")
    parser.add_argument("--moving", default="")
    parser.add_argument(
        "--bind",
        default="",
        help="Explicitly bind a non-polar OR to a transformation hypothesis.",
    )
    parser.add_argument(
        "--parallel-candidate",
        type=int,
        default=1,
        help="1-based candidate when a parallelism definition has sign ambiguity.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "General PTCLab-style two-phase crystallography workbench. "
            "This layer is theory-neutral."
        )
    )
    parser.add_argument(
        "--project",
        default="preset:james_hane_2000",
        help=(
            "Project source: preset:NAME or path to generic project JSON. "
            "James-Hane is only the backward-compatible benchmark default."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    summary = sub.add_parser("summary")
    _add_or_args(summary)
    summary.add_argument("--json", action="store_true")

    pair = sub.add_parser("pair")
    _add_or_args(pair)
    pair.add_argument("reference_object")
    pair.add_argument("moving_object")
    pair.add_argument("--max-index", type=int, default=12)
    pair.add_argument("--oriented", action="store_true")
    pair.add_argument("--derive", action="store_true")
    pair.add_argument("--json", action="store_true")

    mapping = sub.add_parser("map")
    _add_or_args(mapping)
    mapping.add_argument(
        "--from", dest="source_side", choices=("reference", "moving"), required=True
    )
    mapping.add_argument("object")
    mapping.add_argument("--max-index", type=int, default=12)
    mapping.add_argument("--oriented", action="store_true")
    mapping.add_argument("--derive", action="store_true")
    mapping.add_argument("--json", action="store_true")

    matrices = sub.add_parser("matrices")
    _add_or_args(matrices)
    matrices.add_argument("--json", action="store_true")

    variants = sub.add_parser("variants")
    _add_or_args(variants)
    variants.add_argument("reference_object")
    variants.add_argument("moving_object")
    variants.add_argument("--max-index", type=int, default=12)
    variants.add_argument("--oriented", action="store_true")
    variants.add_argument("--json", action="store_true")

    equivalents = sub.add_parser("equivalents")
    _add_or_args(equivalents)
    equivalents.add_argument("reference_object")
    equivalents.add_argument("moving_object")
    equivalents.add_argument("--limit", type=int, default=30)
    equivalents.add_argument("--proper-only", action="store_true")
    equivalents.add_argument("--oriented", action="store_true")
    equivalents.add_argument("--json", action="store_true")

    shell = sub.add_parser("shell")
    shell.add_argument(
        "--project",
        default=argparse.SUPPRESS,
        help="Project source: preset:NAME or path to project JSON.",
    )
    shell.add_argument(
        "--or",
        dest="or_spec",
        default="",
        help=(
            "Explicit initial OR. If omitted, a single stored OR is used. "
            "The James-Hane benchmark alone retains its historical polar default."
        ),
    )
    shell.add_argument("--reference", default="")
    shell.add_argument("--moving", default="")
    return parser


def _resolve_from_args(
    workbench: TwoPhaseWorkbench,
    args: argparse.Namespace,
) -> OrientationState:
    return workbench.resolve_orientation(
        args.or_spec,
        reference=args.reference,
        moving=args.moving,
        bind_transformation=getattr(args, "bind", ""),
        parallel_candidate=getattr(args, "parallel_candidate", 1),
    )


def _split_pair(text: str) -> tuple[str, str]:
    parts = [item.strip() for item in text.split(";")]
    if len(parts) != 2 or not all(parts):
        raise ValueError("Enter two objects separated by ';'.")
    return parts[0], parts[1]


def _shell_help() -> str:
    return """Commands:
  phases
  use REF MOV
  or SPEC
  summary
  pair REF_OBJECT ; MOV_OBJECT
  map reference OBJECT
  map moving OBJECT
  variants REF_OBJECT ; MOV_OBJECT
  equivalents REF_OBJECT ; MOV_OBJECT
  matrices
  max N
  derive on|off
  help
  quit

OR SPEC examples:
  polar:do3_to_6m_reference
  identity
  matrix:1 0 0; 0 1 0; 0 0 1
  euler:10 20 30
  quat:1 0 0 0
  axis:0 0 1; 5
  axis-crystal:[1 1 0]; 5
  parallel:(0 1 0)|(0 1 0)|[1 0 0]|[1 0 0]
"""


def interactive_shell(
    workbench: TwoPhaseWorkbench,
    *,
    initial_spec: str,
    reference: str,
    moving: str,
) -> int:
    renderer = TwoPhaseRenderer()
    state = workbench.resolve_orientation(
        initial_spec,
        reference=reference,
        moving=moving,
    )
    max_index = 12
    derive = False

    print("=" * 88)
    print("TWO-PHASE WORKBENCH — interactive")
    print("=" * 88)
    print(f"reference={reference}  moving={moving}  OR={state.orientation_id}")
    print("Type 'help' for commands.")

    while True:
        try:
            raw = input("two-phase> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not raw:
            continue

        try:
            command, _, rest = raw.partition(" ")
            key = command.lower()

            if key in {"quit", "exit", "q"}:
                return 0
            if key == "help":
                print(_shell_help())
                continue
            if key == "phases":
                for phase in workbench.project.phases:
                    print(
                        f"{phase.phase_id:26s}  {phase.label}  "
                        f"[{phase.cell_representation}]"
                    )
                continue
            if key == "use":
                tokens = rest.split()
                if len(tokens) != 2:
                    raise ValueError("use syntax: use REF MOV")
                reference, moving = tokens
                print(f"phase pair set to {reference} <- {moving}; choose/set OR next.")
                continue
            if key == "or":
                state = workbench.resolve_orientation(
                    rest,
                    reference=reference,
                    moving=moving,
                )
                print(f"OR set: {state.orientation_id}")
                continue
            if key == "summary":
                print(renderer.summary(workbench.summary(state)))
                continue
            if key == "matrices":
                print(renderer.matrices(workbench.coordinate_matrices(state)))
                continue
            if key == "max":
                max_index = int(rest)
                if max_index < 1:
                    raise ValueError("max index must be >= 1")
                print(f"low-index search bound = {max_index}")
                continue
            if key == "derive":
                value = rest.lower()
                if value not in {"on", "off"}:
                    raise ValueError("derive syntax: derive on|off")
                derive = value == "on"
                print(f"derivation display {'ON' if derive else 'OFF'}")
                continue
            if key == "pair":
                ref_text, mov_text = _split_pair(rest)
                print(
                    renderer.pair(
                        workbench.pair(
                            state,
                            ref_text,
                            mov_text,
                            max_index=max_index,
                        ),
                        derive=derive,
                    )
                )
                continue
            if key == "map":
                side_text, _, object_text = rest.partition(" ")
                if not object_text:
                    raise ValueError("map syntax: map reference|moving OBJECT")
                print(
                    renderer.mapping(
                        workbench.map_object(
                            state,
                            WorkbenchSide(side_text.lower()),
                            object_text,
                            max_index=max_index,
                        ),
                        derive=derive,
                    )
                )
                continue
            if key == "variants":
                ref_text, mov_text = _split_pair(rest)
                print(
                    renderer.variants(
                        workbench.variant_angles(
                            state,
                            ref_text,
                            mov_text,
                            max_index=max_index,
                        )
                    )
                )
                continue
            if key in {"equiv", "equivalents"}:
                ref_text, mov_text = _split_pair(rest)
                print(
                    renderer.equivalents(
                        workbench.equivalent_angles(
                            state,
                            ref_text,
                            mov_text,
                        )
                    )
                )
                continue

            raise ValueError(f"Unknown command {command!r}; type 'help'.")
        except (ValueError, KeyError) as exc:
            print(f"ERROR: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        loaded = load_project(args.project)
        workbench = TwoPhaseWorkbench(loaded.project)
    except (AssertionError, KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        parser.error(f"project: {exc}")
    renderer = TwoPhaseRenderer()

    if args.command == "shell":
        initial_spec = args.or_spec.strip()
        if not initial_spec:
            if len(workbench.project.orientations) == 1:
                initial_spec = (
                    f"stored:{workbench.project.orientations[0].orientation_id}"
                )
            elif workbench.project.project_id == "james_hane_cualni_6m_reference":
                initial_spec = "polar:do3_to_6m_reference"
            else:
                parser.error(
                    "shell requires --or for this project; no physical OR is assumed"
                )
        return interactive_shell(
            workbench,
            initial_spec=initial_spec,
            reference=args.reference,
            moving=args.moving,
        )

    state = _resolve_from_args(workbench, args)

    if args.command == "summary":
        report = workbench.summary(state)
        print(
            json.dumps(report.to_dict(), indent=2)
            if args.json
            else renderer.summary(report)
        )
        return 0

    if args.command == "pair":
        report = workbench.pair(
            state,
            args.reference_object,
            args.moving_object,
            max_index=args.max_index,
            projective=not args.oriented,
        )
        print(
            json.dumps(report.to_dict(), indent=2)
            if args.json
            else renderer.pair(report, derive=args.derive)
        )
        return 0

    if args.command == "map":
        report = workbench.map_object(
            state,
            args.source_side,
            args.object,
            max_index=args.max_index,
            projective=not args.oriented,
        )
        print(
            json.dumps(report.to_dict(), indent=2)
            if args.json
            else renderer.mapping(report, derive=args.derive)
        )
        return 0

    if args.command == "matrices":
        report = workbench.coordinate_matrices(state)
        print(
            json.dumps(report.to_dict(), indent=2)
            if args.json
            else renderer.matrices(report)
        )
        return 0

    if args.command == "variants":
        report = workbench.variant_angles(
            state,
            args.reference_object,
            args.moving_object,
            max_index=args.max_index,
            projective=not args.oriented,
        )
        print(
            json.dumps(report.to_dict(), indent=2)
            if args.json
            else renderer.variants(report)
        )
        return 0

    if args.command == "equivalents":
        report = workbench.equivalent_angles(
            state,
            args.reference_object,
            args.moving_object,
            limit=args.limit,
            projective=not args.oriented,
            proper_only=args.proper_only,
        )
        print(
            json.dumps(report.to_dict(), indent=2)
            if args.json
            else renderer.equivalents(report)
        )
        return 0

    raise AssertionError(f"Unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
