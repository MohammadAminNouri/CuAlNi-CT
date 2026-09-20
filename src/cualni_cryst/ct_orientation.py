from __future__ import annotations

"""Cayron orientation adapter: natural OR roles and closing-gap OR construction.

This module is intentionally additive.  It does not replace the generic
orientation engine and it does not identify correspondence, deformation,
stretch, polar rotation, or experimental ORs with one another.

Central physical orientation convention
---------------------------------------
For parent/reference A and martensite/moving M,

    x_A = R_A_from_M @ x_M

where both x_A and x_M are Cartesian components in explicitly named
right-handed Cartesian embeddings of their respective crystallographic
metrics.

Cayron Correspondence Theory assumptions used here
---------------------------------------------------
Cayron's CT assumes a natural OR and allows small additional closing-gap
rotations to make intervariant relations compatible.  For exact Type-I twins
the closing-gap OR preserves the parallelisms

    K1_A || K1_M
    eta1_A || eta1_M

and for exact Type-II twins

    eta2_A || eta2_M
    K2_A || K2_M.

The CT twin elements are supplied by :mod:`cualni_cryst.twinning_ct`, which is
metric/correspondence based and independent of the Ball-James/PTMC adapters.

A natural OR is never generated here from metrics + correspondence alone.
It must be supplied explicitly from literature, experiment, or another
physical model.  When supplied, candidate closing-gap ORs are ranked by the
same kind of quantity Cayron reports: the symmetry-reduced disorientation to
the natural OR.  All sign/projective branches are retained and audited.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
import sympy as sp

from .orientation import OrientationService, rotation_audit
from .project_state import (
    OrientationDefinition,
    OrientationState,
    OrientationTheoryOrigin,
    StateProvenance,
)
from .provenance import DataStatus
from .representation import CartesianConvention, CartesianFrame
from .twinning_ct import (
    CTTwin,
    classify_parent_order_two_isometry,
    twins_from_operator,
    type_i_from_parent_reflection,
    type_ii_from_parent_twofold,
)


class CTOrientationKind(str, Enum):
    """Scientific role of an orientation in the CT workflow."""

    NATURAL = "natural_or"
    CLOSING_GAP_TYPE_I = "closing_gap_type_i"
    CLOSING_GAP_TYPE_II = "closing_gap_type_ii"
    POLAR_COMPARATOR = "polar_comparator"


@dataclass(frozen=True)
class NaturalOR:
    """Explicit natural-OR hypothesis.

    CT does not derive a unique natural OR from ``(M_A, M_M, C)`` alone.
    Therefore a natural OR must be supplied as an :class:`OrientationState`.
    """

    state: OrientationState
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": CTOrientationKind.NATURAL.value,
            "state": self.state.to_dict(),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ClosingGapCandidate:
    """One proper-rotation branch satisfying the CT parallelism constraints."""

    index: int
    orientation: OrientationState
    direction_sign: int
    plane_normal_sign: int
    direction_parallelism_residual_deg: float
    plane_parallelism_residual_deg: float
    parent_incidence_residual_deg: float
    martensite_incidence_residual_deg: float
    parent_frame_projection_correction: float
    martensite_frame_projection_correction: float
    rotation_residual: float
    raw_deviation_from_natural_deg: float | None
    symmetry_reduced_deviation_from_natural_deg: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "orientation": self.orientation.to_dict(),
            "direction_sign": self.direction_sign,
            "plane_normal_sign": self.plane_normal_sign,
            "direction_parallelism_residual_deg": (
                self.direction_parallelism_residual_deg
            ),
            "plane_parallelism_residual_deg": self.plane_parallelism_residual_deg,
            "parent_incidence_residual_deg": self.parent_incidence_residual_deg,
            "martensite_incidence_residual_deg": (
                self.martensite_incidence_residual_deg
            ),
            "parent_frame_projection_correction": (
                self.parent_frame_projection_correction
            ),
            "martensite_frame_projection_correction": (
                self.martensite_frame_projection_correction
            ),
            "rotation_residual": self.rotation_residual,
            "raw_deviation_from_natural_deg": self.raw_deviation_from_natural_deg,
            "symmetry_reduced_deviation_from_natural_deg": (
                self.symmetry_reduced_deviation_from_natural_deg
            ),
        }


@dataclass(frozen=True)
class ClosingGapReport:
    """Auditable CT closing-gap OR result for one exact CT twin."""

    transformation_id: str
    twin_kind: str
    orientation_kind: CTOrientationKind
    rational_element: str
    correspondence_plane_residual_deg: float
    correspondence_direction_residual_deg: float
    intercorrespondence_residual: float
    candidates: tuple[ClosingGapCandidate, ...]
    natural_orientation_id: str
    selected_candidate_index: int | None
    selection_tie_indices: tuple[int, ...]
    selection_rule: str
    source_note: str
    warnings: tuple[str, ...]

    @property
    def selected(self) -> ClosingGapCandidate | None:
        if self.selected_candidate_index is None:
            return None
        for candidate in self.candidates:
            if candidate.index == self.selected_candidate_index:
                return candidate
        raise AssertionError("selected_candidate_index does not exist")

    @property
    def minimum_natural_deviation_deg(self) -> float | None:
        selected = self.selected
        if selected is None:
            return None
        return selected.symmetry_reduced_deviation_from_natural_deg

    def to_dict(self) -> dict[str, object]:
        return {
            "transformation_id": self.transformation_id,
            "twin_kind": self.twin_kind,
            "orientation_kind": self.orientation_kind.value,
            "rational_element": self.rational_element,
            "correspondence_plane_residual_deg": self.correspondence_plane_residual_deg,
            "correspondence_direction_residual_deg": (
                self.correspondence_direction_residual_deg
            ),
            "intercorrespondence_residual": self.intercorrespondence_residual,
            "natural_orientation_id": self.natural_orientation_id,
            "selected_candidate_index": self.selected_candidate_index,
            "selection_tie_indices": list(self.selection_tie_indices),
            "selection_rule": self.selection_rule,
            "source_note": self.source_note,
            "warnings": list(self.warnings),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class OperatorClosingGapReport:
    """Closing-gap solutions generated from one exact correspondence operator."""

    transformation_id: str
    twin_reports: tuple[ClosingGapReport, ...]
    best_report_index: int | None
    best_candidate_index: int | None
    minimum_natural_deviation_deg: float | None
    tie_report_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "transformation_id": self.transformation_id,
            "best_report_index": self.best_report_index,
            "best_candidate_index": self.best_candidate_index,
            "minimum_natural_deviation_deg": self.minimum_natural_deviation_deg,
            "tie_report_indices": list(self.tie_report_indices),
            "twin_reports": [report.to_dict() for report in self.twin_reports],
        }


def _normalize(vector: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(vector, dtype=float).reshape(3)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-15:
        raise ValueError(f"{name} must be nonzero")
    return array / norm


def _projective_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    left = _normalize(a, name="first vector")
    right = _normalize(b, name="second vector")
    sine = float(np.linalg.norm(np.cross(left, right)))
    cosine = abs(float(left @ right))
    return float(np.degrees(np.arctan2(sine, cosine)))


def _relative_residual(lhs: np.ndarray, rhs: np.ndarray) -> float:
    left = np.asarray(lhs, dtype=float)
    right = np.asarray(rhs, dtype=float)
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1.0)
    return float(np.linalg.norm(left - right) / scale)


def _incidence_residual_deg(direction: np.ndarray, plane_normal: np.ndarray) -> float:
    """Return angular departure from exact direction-in-plane incidence."""

    d = _normalize(direction, name="direction")
    n = _normalize(plane_normal, name="plane normal")
    sine = abs(float(d @ n))
    return float(np.degrees(np.arcsin(np.clip(sine, 0.0, 1.0))))


def _pair_frame(
    direction: np.ndarray,
    plane_normal: np.ndarray,
    *,
    incidence_tolerance_deg: float,
) -> tuple[np.ndarray, float, float]:
    """Build a right-handed orthonormal frame from a line in a plane.

    The input pair should be physically orthogonal.  A tiny floating-point
    non-orthogonality is removed by one explicit projection; the correction
    magnitude and original incidence residual are returned and exposed in the
    report.  Inputs outside the requested tolerance are rejected rather than
    silently repaired.
    """

    d_raw = _normalize(direction, name="direction")
    n = _normalize(plane_normal, name="plane normal")
    incidence = _incidence_residual_deg(d_raw, n)
    if incidence > incidence_tolerance_deg:
        raise ValueError(
            "Direction is not incident in the supplied plane within tolerance: "
            f"residual={incidence:.6g} deg > {incidence_tolerance_deg:.6g} deg"
        )

    projected = d_raw - float(d_raw @ n) * n
    d = _normalize(projected, name="projected in-plane direction")
    correction = float(np.linalg.norm(d - d_raw))

    second = _normalize(np.cross(n, d), name="frame second axis")
    frame = np.column_stack((d, second, n))
    audit = rotation_audit(frame)
    if audit.maximum_residual > 1.0e-12:
        raise AssertionError(
            "Plane-direction frame construction failed SO(3) audit: "
            f"{audit.maximum_residual:.3e}"
        )
    return frame, incidence, correction


def _candidate_rotation(
    parent_direction: np.ndarray,
    parent_plane_normal: np.ndarray,
    martensite_direction: np.ndarray,
    martensite_plane_normal: np.ndarray,
    *,
    direction_sign: int,
    plane_normal_sign: int,
    incidence_tolerance_deg: float,
) -> tuple[np.ndarray, float, float, float, float]:
    if direction_sign not in {-1, 1} or plane_normal_sign not in {-1, 1}:
        raise ValueError("Parallelism signs must be ±1")

    parent_frame, incidence_a, correction_a = _pair_frame(
        direction_sign * np.asarray(parent_direction, dtype=float),
        plane_normal_sign * np.asarray(parent_plane_normal, dtype=float),
        incidence_tolerance_deg=incidence_tolerance_deg,
    )
    martensite_frame, incidence_m, correction_m = _pair_frame(
        martensite_direction,
        martensite_plane_normal,
        incidence_tolerance_deg=incidence_tolerance_deg,
    )

    rotation = parent_frame @ martensite_frame.T
    audit = rotation_audit(rotation)
    if audit.maximum_residual > 1.0e-12:
        raise AssertionError(
            f"Closing-gap construction failed SO(3) audit: {audit.maximum_residual:.3e}"
        )
    return rotation, incidence_a, incidence_m, correction_a, correction_m


class CayronOrientationAdapter:
    """CT orientation adapter bound to one explicit transformation state.

    The adapter consumes the existing generic :class:`OrientationService` and
    the existing CT twin engine.  It does not own project state and does not
    alter CalPad/PTCLab-oriented layers.
    """

    def __init__(
        self,
        service: OrientationService,
        transformation_id: str,
        *,
        reference_convention: CartesianConvention = (
            CartesianConvention.PTCLAB_A_X_C_XZ
        ),
        moving_convention: CartesianConvention = (CartesianConvention.PTCLAB_A_X_C_XZ),
        incidence_tolerance_deg: float = 1.0e-7,
        algebraic_tolerance: float = 1.0e-10,
        selection_tolerance_deg: float = 1.0e-7,
    ) -> None:
        if incidence_tolerance_deg <= 0.0:
            raise ValueError("incidence_tolerance_deg must be positive")
        if algebraic_tolerance <= 0.0:
            raise ValueError("algebraic_tolerance must be positive")
        if selection_tolerance_deg <= 0.0:
            raise ValueError("selection_tolerance_deg must be positive")

        self.service = service
        self.transformation_id = transformation_id
        self.transformation = service.project.transformation(transformation_id)
        self.parent = service.project.phase(self.transformation.parent_phase_id)
        self.martensite = service.project.phase(self.transformation.product_phase_id)
        self.reference_convention = CartesianConvention(reference_convention)
        self.moving_convention = CartesianConvention(moving_convention)
        self.incidence_tolerance_deg = float(incidence_tolerance_deg)
        self.algebraic_tolerance = float(algebraic_tolerance)
        self.selection_tolerance_deg = float(selection_tolerance_deg)

    @property
    def correspondence_matrix(self) -> np.ndarray:
        return np.asarray(
            self.transformation.correspondence.C_M_from_A,
            dtype=float,
        )

    def natural_orientation(self, state: OrientationState) -> NaturalOR:
        """Validate an explicitly supplied natural-OR hypothesis.

        No attempt is made to infer or synthesize a natural OR.
        """

        if state.reference_phase_id != self.parent.phase_id:
            raise ValueError(
                "Natural OR reference phase must be the transformation parent: "
                f"{self.parent.phase_id!r}"
            )
        if state.moving_phase_id != self.martensite.phase_id:
            raise ValueError(
                "Natural OR moving phase must be the transformation product: "
                f"{self.martensite.phase_id!r}"
            )
        if (
            state.transformation_id
            and state.transformation_id != self.transformation_id
        ):
            raise ValueError(
                "Natural OR is bound to a different transformation_id: "
                f"{state.transformation_id!r}"
            )

        warnings: list[str] = []
        if state.theory_origin is OrientationTheoryOrigin.POLAR_CORRESPONDENCE:
            warnings.append(
                "The supplied natural-OR hypothesis is a polar-decomposition "
                "candidate. CT does not establish that R_polar is the natural OR; "
                "this use is explicit and hypothetical."
            )
        elif state.theory_origin is OrientationTheoryOrigin.BALL_JAMES:
            warnings.append(
                "The supplied natural-OR hypothesis originates from Ball-James. "
                "It is being used explicitly as a comparison hypothesis, not as "
                "a CT identity."
            )
        elif state.theory_origin is OrientationTheoryOrigin.PTMC:
            warnings.append(
                "The supplied natural-OR hypothesis originates from PTMC. It is "
                "being used explicitly as a comparison hypothesis, not as a CT identity."
            )

        return NaturalOR(state=state, warnings=tuple(warnings))

    def polar_comparator(self) -> OrientationState:
        """Return the existing polar OR comparator without relabelling it as CT."""

        state = self.service.polar_orientation(self.transformation_id)
        if state.theory_origin is not OrientationTheoryOrigin.POLAR_CORRESPONDENCE:
            raise AssertionError("OrientationService returned a mislabelled polar OR")
        return state

    @staticmethod
    def _rotation_angle_deg(matrix: np.ndarray) -> float:
        """Return a numerically stable principal SO(3) angle in degrees.

        The trace/arccos formula loses resolution near the identity because
        ``cos(theta)`` rounds to 1 before ``acos`` is evaluated.  Natural-OR
        ranking must distinguish sub-microdegree closing-gap differences, so
        use both sine and cosine information:

            sin(theta) = ||R - R.T||_F / (2 sqrt(2))
            cos(theta) = (tr(R) - 1) / 2

        and recover theta with atan2(sin, cos).

        This remains stable near 0 degrees and 180 degrees and preserves the
        exact identity case.
        """

        rotation = np.asarray(matrix, dtype=float).reshape(3, 3)

        audit = rotation_audit(rotation)
        if audit.maximum_residual > 1.0e-8:
            raise ValueError(
                "Natural-OR comparison requires proper rotations; "
                f"residual={audit.maximum_residual:.3e}."
            )

        sine = float(
            np.linalg.norm(
                rotation - rotation.T,
                ord="fro",
            )
            / (2.0 * np.sqrt(2.0))
        )

        cosine = float((float(np.trace(rotation)) - 1.0) / 2.0)

        sine = float(
            np.clip(
                sine,
                0.0,
                1.0,
            )
        )

        cosine = float(
            np.clip(
                cosine,
                -1.0,
                1.0,
            )
        )

        return float(
            np.degrees(
                np.arctan2(
                    sine,
                    cosine,
                )
            )
        )

    def _orientation_distance_to_natural(
        self,
        natural: OrientationState,
        candidate: OrientationState,
    ) -> tuple[float, float]:
        """Return raw and symmetry-reduced OR distances.

        This reproduces the scalar part of the generic OR comparison while avoiding
        construction of a disorientation axis, which is undefined for an exact
        identity rotation.  Both states are first re-expressed in the candidate's
        explicit Cartesian frame pair.
        """

        if (
            natural.reference_phase_id != candidate.reference_phase_id
            or natural.moving_phase_id != candidate.moving_phase_id
        ):
            raise ValueError(
                "Natural/candidate ORs must use the same ordered phase pair"
            )

        natural_common = self.service.reexpress(
            natural,
            candidate.reference_cartesian_convention,
            candidate.moving_cartesian_convention,
        )

        natural_matrix = np.asarray(
            natural_common.R_reference_from_moving,
            dtype=float,
        )
        candidate_matrix = np.asarray(
            candidate.R_reference_from_moving,
            dtype=float,
        )
        raw = self._rotation_angle_deg(natural_matrix @ candidate_matrix.T)

        natural_variants = self.service.variants(natural_common)
        candidate_variants = self.service.variants(candidate)
        if not natural_variants or not candidate_variants:
            raise AssertionError("OR topology returned no proper orientation variants")

        reduced = min(
            self._rotation_angle_deg(
                np.asarray(left.matrix_reference_from_moving, dtype=float)
                @ np.asarray(right.matrix_reference_from_moving, dtype=float).T
            )
            for left in natural_variants
            for right in candidate_variants
        )
        return raw, reduced

    def _validate_twin(self, twin: CTTwin) -> tuple[float, float, float]:
        if twin.kind not in {"I", "II"}:
            raise ValueError(f"Unsupported exact CT twin kind {twin.kind!r}")

        order_two_kind = classify_parent_order_two_isometry(
            twin.parent_symmetry,
            self.parent.lattice.metric(),
            tol=self.algebraic_tolerance,
        )
        if twin.kind == "I" and order_two_kind != "reflection":
            raise ValueError("Type-I CTTwin does not carry a parent reflection")
        if twin.kind == "II" and order_two_kind != "twofold":
            raise ValueError(
                "Type-II CTTwin does not carry a proper parent 180-degree rotation"
            )

        C = self.correspondence_matrix
        C_inv = np.linalg.inv(C)
        g = np.asarray(twin.parent_symmetry, dtype=float)
        expected_intercorrespondence = C @ g @ C_inv
        intercorrespondence_residual = _relative_residual(
            np.asarray(twin.intercorrespondence, dtype=float),
            expected_intercorrespondence,
        )
        if intercorrespondence_residual > self.algebraic_tolerance:
            raise ValueError(
                "CTTwin intercorrespondence is inconsistent with this "
                f"transformation; residual={intercorrespondence_residual:.3e}"
            )

        expected_plane_m = np.linalg.inv(C).T @ np.asarray(twin.plane_a, dtype=float)
        plane_residual = _projective_angle_deg(expected_plane_m, twin.plane_m)
        if plane_residual > self.incidence_tolerance_deg:
            raise ValueError(
                "CTTwin plane correspondence is inconsistent with this "
                f"transformation; angular residual={plane_residual:.6g} deg"
            )

        expected_direction_m = C @ np.asarray(twin.direction_a, dtype=float)
        direction_residual = _projective_angle_deg(
            expected_direction_m,
            twin.direction_m,
        )
        if direction_residual > self.incidence_tolerance_deg:
            raise ValueError(
                "CTTwin direction correspondence is inconsistent with this "
                f"transformation; angular residual={direction_residual:.6g} deg"
            )

        return plane_residual, direction_residual, intercorrespondence_residual

    def closing_gap_from_twin(
        self,
        twin: CTTwin,
        *,
        natural_orientation: OrientationState | None = None,
        orientation_id_prefix: str = "",
    ) -> ClosingGapReport:
        """Construct all proper closing-gap OR branches for one exact CT twin.

        Planes and crystallographic directions are treated as projective
        parallelisms.  Therefore all four sign branches are constructed.  If an
        explicit natural OR is supplied, the candidate with the smallest
        symmetry-reduced disorientation is selected, with all ties retained.
        """

        (
            correspondence_plane_residual,
            correspondence_direction_residual,
            intercorrespondence_residual,
        ) = self._validate_twin(twin)

        natural = (
            self.natural_orientation(natural_orientation)
            if natural_orientation is not None
            else None
        )

        parent_frame = CartesianFrame(self.parent.lattice, self.reference_convention)
        martensite_frame = CartesianFrame(
            self.martensite.lattice,
            self.moving_convention,
        )

        d_a = parent_frame.direct_to_cartesian(twin.direction_a, normalize=True)
        n_a = parent_frame.plane_to_cartesian(twin.plane_a, normalize=True)
        d_m = martensite_frame.direct_to_cartesian(
            twin.direction_m,
            normalize=True,
        )
        n_m = martensite_frame.plane_to_cartesian(twin.plane_m, normalize=True)

        orientation_kind = (
            CTOrientationKind.CLOSING_GAP_TYPE_I
            if twin.kind == "I"
            else CTOrientationKind.CLOSING_GAP_TYPE_II
        )
        prefix = orientation_id_prefix.strip() or (
            f"ct_{self.transformation_id}_{orientation_kind.value}"
        )

        candidates: list[ClosingGapCandidate] = []
        seen: list[np.ndarray] = []
        for direction_sign in (-1, 1):
            for plane_sign in (-1, 1):
                (
                    rotation,
                    incidence_a,
                    incidence_m,
                    correction_a,
                    correction_m,
                ) = _candidate_rotation(
                    d_a,
                    n_a,
                    d_m,
                    n_m,
                    direction_sign=direction_sign,
                    plane_normal_sign=plane_sign,
                    incidence_tolerance_deg=self.incidence_tolerance_deg,
                )

                if any(
                    _relative_residual(rotation, previous) <= self.algebraic_tolerance
                    for previous in seen
                ):
                    continue
                seen.append(rotation)

                candidate_index = len(candidates)
                notes = (
                    "Cayron CT closing-gap OR constructed from exact twin "
                    f"parallelisms ({orientation_kind.value}); "
                    f"direction_sign={direction_sign:+d}, "
                    f"plane_normal_sign={plane_sign:+d}. "
                    "This is distinct from correspondence C, deformation F, "
                    "stretch U, and polar rotation R_polar."
                )
                state = self.service.state_from_matrix(
                    self.parent.phase_id,
                    self.martensite.phase_id,
                    rotation,
                    orientation_id=f"{prefix}_{candidate_index}",
                    label=f"CT {orientation_kind.value} candidate {candidate_index}",
                    reference_convention=self.reference_convention,
                    moving_convention=self.moving_convention,
                    definition_method=OrientationDefinition.PARALLELISMS,
                    theory_origin=OrientationTheoryOrigin.CAYRON_CT,
                    provenance=StateProvenance(
                        status=DataStatus.COMPUTATION_DERIVED,
                        notes=(
                            "Source equations: Cayron Correspondence Theory "
                            "closing-gap parallelisms; exact twin geometry comes "
                            "from the CT twin adapter."
                        ),
                    ),
                    transformation_id=self.transformation_id,
                    notes=notes,
                )

                mapped_d = rotation @ d_m
                mapped_n = rotation @ n_m
                direction_parallelism_residual = _projective_angle_deg(
                    mapped_d,
                    d_a,
                )
                plane_parallelism_residual = _projective_angle_deg(
                    mapped_n,
                    n_a,
                )

                raw_deviation: float | None = None
                reduced_deviation: float | None = None
                if natural is not None:
                    raw_deviation, reduced_deviation = (
                        self._orientation_distance_to_natural(
                            natural.state,
                            state,
                        )
                    )

                candidates.append(
                    ClosingGapCandidate(
                        index=candidate_index,
                        orientation=state,
                        direction_sign=direction_sign,
                        plane_normal_sign=plane_sign,
                        direction_parallelism_residual_deg=(
                            direction_parallelism_residual
                        ),
                        plane_parallelism_residual_deg=(plane_parallelism_residual),
                        parent_incidence_residual_deg=incidence_a,
                        martensite_incidence_residual_deg=incidence_m,
                        parent_frame_projection_correction=correction_a,
                        martensite_frame_projection_correction=correction_m,
                        rotation_residual=rotation_audit(rotation).maximum_residual,
                        raw_deviation_from_natural_deg=raw_deviation,
                        symmetry_reduced_deviation_from_natural_deg=(reduced_deviation),
                    )
                )

        if not candidates:
            raise AssertionError("No proper closing-gap OR branch was generated")

        selected_index: int | None = None
        tie_indices: tuple[int, ...] = ()
        natural_id = ""
        if natural is not None:
            natural_id = natural.state.orientation_id
            finite = [
                candidate
                for candidate in candidates
                if candidate.symmetry_reduced_deviation_from_natural_deg is not None
            ]
            if len(finite) != len(candidates):
                raise AssertionError("Natural-OR comparison was not evaluated")
            best_value = min(
                float(candidate.symmetry_reduced_deviation_from_natural_deg)
                for candidate in finite
            )
            ties = tuple(
                candidate.index
                for candidate in finite
                if abs(
                    float(candidate.symmetry_reduced_deviation_from_natural_deg)
                    - best_value
                )
                <= self.selection_tolerance_deg
            )
            tie_indices = ties
            selected_index = min(
                ties,
                key=lambda index: (
                    float(candidates[index].raw_deviation_from_natural_deg),
                    index,
                ),
            )

        selection_rule = (
            "No natural OR supplied: all projective sign branches are retained "
            "without selecting a preferred closing-gap OR."
            if natural is None
            else (
                "Select the branch with minimum symmetry-reduced disorientation "
                "to the explicitly supplied natural OR; ties within "
                f"{self.selection_tolerance_deg:.3g} deg are retained. This "
                "implements Cayron's natural-OR/minimum-deviation comparison "
                "logic and is not a proof that the selected branch is realized "
                "experimentally."
            )
        )

        source_note = (
            "Cayron 2022 Correspondence Theory: CT assumes a natural OR and "
            "additional closing-gap rotations. Type I uses K1_A || K1_M and "
            "eta1_A || eta1_M; Type II uses eta2_A || eta2_M and K2_A || K2_M. "
            "Cayron 2026 restates the same Type-I/II closing-gap parallelisms."
        )
        warnings = list(natural.warnings if natural is not None else ())
        warnings.append(
            "A closing-gap OR is an operator/twin-specific local orientation. "
            "It is not a universal CT orientation and does not by itself establish "
            "experimental variant selection."
        )

        return ClosingGapReport(
            transformation_id=self.transformation_id,
            twin_kind=twin.kind,
            orientation_kind=orientation_kind,
            rational_element=twin.rational_element,
            correspondence_plane_residual_deg=correspondence_plane_residual,
            correspondence_direction_residual_deg=(correspondence_direction_residual),
            intercorrespondence_residual=intercorrespondence_residual,
            candidates=tuple(candidates),
            natural_orientation_id=natural_id,
            selected_candidate_index=selected_index,
            selection_tie_indices=tie_indices,
            selection_rule=selection_rule,
            source_note=source_note,
            warnings=tuple(warnings),
        )

    def type_i_from_parent_reflection(
        self,
        reflection_a: sp.Matrix,
        *,
        natural_orientation: OrientationState | None = None,
        orientation_id_prefix: str = "",
    ) -> ClosingGapReport:
        """CT Type-I twin + closing-gap OR from one parent reflection."""

        twin = type_i_from_parent_reflection(
            reflection_a,
            self.parent.lattice.metric(),
            self.martensite.lattice.metric(),
            self.transformation.correspondence,
        )
        return self.closing_gap_from_twin(
            twin,
            natural_orientation=natural_orientation,
            orientation_id_prefix=orientation_id_prefix,
        )

    def type_ii_from_parent_twofold(
        self,
        rotation_a: sp.Matrix,
        *,
        natural_orientation: OrientationState | None = None,
        orientation_id_prefix: str = "",
    ) -> ClosingGapReport:
        """CT Type-II twin + closing-gap OR from one parent proper twofold."""

        twin = type_ii_from_parent_twofold(
            rotation_a,
            self.parent.lattice.metric(),
            self.martensite.lattice.metric(),
            self.transformation.correspondence,
        )
        return self.closing_gap_from_twin(
            twin,
            natural_orientation=natural_orientation,
            orientation_id_prefix=orientation_id_prefix,
        )

    def from_operator(
        self,
        operator: Sequence[sp.Matrix],
        *,
        natural_orientation: OrientationState | None = None,
        orientation_id_prefix: str = "",
    ) -> OperatorClosingGapReport:
        """Generate CT closing-gap ORs from one exact parent double coset.

        The caller must pass the complete exact double-coset member list from
        the correspondence/groupoid layer.  The full crystallographic group is
        appropriate here; reflections are required for Type-I routes and proper
        twofold rotations for Type-II routes.
        """

        operator_list = [sp.Matrix(item) for item in operator]
        if not operator_list:
            raise ValueError("operator must contain at least one symmetry matrix")

        twins = twins_from_operator(
            operator_list,
            self.parent.lattice.metric(),
            self.martensite.lattice.metric(),
            self.transformation.correspondence,
        )

        reports: list[ClosingGapReport] = []
        for index, twin in enumerate(twins):
            prefix = orientation_id_prefix.strip() or (
                f"ct_{self.transformation_id}_operator_twin_{index}"
            )
            reports.append(
                self.closing_gap_from_twin(
                    twin,
                    natural_orientation=natural_orientation,
                    orientation_id_prefix=prefix,
                )
            )

        if natural_orientation is None or not reports:
            return OperatorClosingGapReport(
                transformation_id=self.transformation_id,
                twin_reports=tuple(reports),
                best_report_index=None,
                best_candidate_index=None,
                minimum_natural_deviation_deg=None,
                tie_report_indices=(),
            )

        values: list[tuple[int, float, int]] = []
        for report_index, report in enumerate(reports):
            selected = report.selected
            if (
                selected is None
                or selected.symmetry_reduced_deviation_from_natural_deg is None
            ):
                raise AssertionError("Natural-OR operator report lacks a selection")
            values.append(
                (
                    report_index,
                    float(selected.symmetry_reduced_deviation_from_natural_deg),
                    selected.index,
                )
            )

        best_value = min(value for _, value, _ in values)
        tie_reports = tuple(
            report_index
            for report_index, value, _ in values
            if abs(value - best_value) <= self.selection_tolerance_deg
        )
        best_report_index = min(
            tie_reports,
            key=lambda report_index: (
                float(reports[report_index].selected.raw_deviation_from_natural_deg),
                report_index,
            ),
        )
        best_selected = reports[best_report_index].selected
        if best_selected is None:
            raise AssertionError("Best operator report unexpectedly has no selection")

        return OperatorClosingGapReport(
            transformation_id=self.transformation_id,
            twin_reports=tuple(reports),
            best_report_index=best_report_index,
            best_candidate_index=best_selected.index,
            minimum_natural_deviation_deg=best_value,
            tie_report_indices=tie_reports,
        )

    def weak_orientation_prediction(self, *args: object, **kwargs: object) -> None:
        """Refuse to invent a generic weak-twin OR.

        Cayron's weak-plane construction is tolerance/search based and is not
        equivalent to enforcing two exact parallelisms.  The existing
        ``weak_planes`` layer can later feed a dedicated weak-orientation
        adapter with its own residual model.
        """

        del args, kwargs
        raise NotImplementedError(
            "Weak-twin OR prediction is intentionally not implemented by the "
            "exact closing-gap adapter. A weak plane has intrinsic distortion "
            "and requires Cayron's weak-plane search/tolerance model; treating "
            "it as an exact Type-I/II OR would be scientifically incorrect."
        )
