from __future__ import annotations

"""Harsh experimental validation for EBSD transformation crystallography.

Performance contract
--------------------
The numerical backend is optimized by algebraic reduction and precomputation,
not by dropping symmetry operations, variants, operator branches, robust starts,
or validation cases.

Two expensive quotient calculations are prepared once:

1. product/product boundary operators;
2. candidate-parent compatibility.

The optimized operator residual is *exactly the same quotient problem* as the
reference implementation in :mod:`ebsd_reconstruction`.

For a measured child misorientation ``Delta``, theoretical operator ``Q`` and
proper product group ``G``, the frozen reference implementation minimizes

    angle(S1.T @ Delta @ S2 @ T.T)

over ``S1,S2 in G`` and over every

    T = A.T @ Q @ B

(and its transpose) in the theoretical operator orbit.

By group closure and cyclic invariance of the trace of an SO(3) matrix,

    min angle(S1.T Delta S2 B.T Q.T A)
      = min angle(L Delta K Q.T),

with ``L,K in G``.

The transpose branch similarly gives ``min angle(L Delta K Q)``.

Therefore the complete four-group-loop reference quotient reduces to two group
loops and the two targets ``Q`` / ``Q.T``.  Nothing is omitted.  Those measured
quotient rotations are precomputed once and compared to trial theoretical
operators by vectorized unit-quaternion geodesic distance.

The OR-refinement objective additionally avoids rebuilding operator *classes*:
the objective only needs the minimum over all theoretical variant-pair
operators, so evaluating every physical variant pair is mathematically
identical to taking the minimum over the classes that partition those pairs.
Class construction is still performed for final scientific reporting.

No timing heuristic is part of the science; equivalence to the slower frozen
reference formulation is regression-tested directly.
"""

from dataclasses import dataclass
from math import ceil, comb
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .ebsd_analysis import (
    Grain,
    GrainSegmentation,
    NeighborGraph,
    grain_statistics,
    minimum_disorientation,
    segment_grains,
)
from .ebsd_map import EBSDMap, EBSDPhase, MapAudit, audit_map
from .ebsd_reconstruction import (
    OperatorClass,
    ParentReconstruction,
    VariantOR,
    classify_boundary_operator,
    reconstruct_parent,
)
from .ebsd_theory_bridge import (
    ORVariantSet,
    TheoryLibrary,
    build_or_variant_set,
    build_theory_library,
    orientation_relationship_distance_deg,
)
from .orientation_kernel import require_so3, rotation_angle_deg


def _rotation_quaternions(matrices: np.ndarray) -> np.ndarray:
    """Return normalized SciPy-order (x,y,z,w) unit quaternions."""

    values = np.asarray(matrices, dtype=float)
    if values.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix array must end in (3,3)")
    flat = values.reshape(-1, 3, 3)
    quaternions = Rotation.from_matrix(flat).as_quat()
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms <= 1.0e-15):
        raise ValueError("invalid zero-norm rotation quaternion")
    quaternions = quaternions / norms[:, None]
    return quaternions.reshape(values.shape[:-2] + (4,))


def _angles_from_projective_quaternion_dots(dots: np.ndarray) -> np.ndarray:
    """Stable SO(3) angles from |q1.q2|, in degrees."""

    value = np.clip(np.asarray(dots, dtype=float), 0.0, 1.0)
    sine_half = np.sqrt(np.maximum(0.0, 1.0 - value * value))
    return np.degrees(2.0 * np.arctan2(sine_half, value))


def adjusted_rand_index(
    labels_first: np.ndarray,
    labels_second: np.ndarray,
    *,
    ignore_negative: bool = True,
) -> float:
    """Adjusted Rand index implemented without an external ML dependency."""

    a = np.asarray(labels_first, dtype=int).reshape(-1)
    b = np.asarray(labels_second, dtype=int).reshape(-1)
    if len(a) != len(b):
        raise ValueError("label arrays must have the same length")

    if ignore_negative:
        mask = (a >= 0) & (b >= 0)
        a = a[mask]
        b = b[mask]
    n = len(a)
    if n < 2:
        return 1.0

    unique_a, inverse_a = np.unique(a, return_inverse=True)
    unique_b, inverse_b = np.unique(b, return_inverse=True)
    contingency = np.zeros((len(unique_a), len(unique_b)), dtype=int)
    np.add.at(contingency, (inverse_a, inverse_b), 1)

    def choose2(values: np.ndarray) -> float:
        return float(
            sum(comb(int(value), 2) for value in values if int(value) >= 2)
        )

    sum_cells = choose2(contingency.ravel())
    rows = np.sum(contingency, axis=1)
    cols = np.sum(contingency, axis=0)
    sum_rows = choose2(rows)
    sum_cols = choose2(cols)
    total_pairs = comb(n, 2)
    expected = (sum_rows * sum_cols) / total_pairs
    maximum = 0.5 * (sum_rows + sum_cols)
    denominator = maximum - expected
    if abs(denominator) <= 1.0e-15:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float((sum_cells - expected) / denominator)


@dataclass(frozen=True)
class SegmentationSweepEntry:
    threshold_deg: float
    n_grains: int
    retained_fraction: float
    median_gos_deg: float
    p95_gos_deg: float
    labels: np.ndarray


@dataclass(frozen=True)
class SegmentationSweep:
    entries: tuple[SegmentationSweepEntry, ...]
    consecutive_adjusted_rand: tuple[float, ...]


def sweep_segmentation_thresholds(
    data: EBSDMap,
    phases: Mapping[int, EBSDPhase],
    thresholds_deg: Sequence[float],
    *,
    neighbor_graph: NeighborGraph | None = None,
    minimum_points: int = 1,
) -> SegmentationSweep:
    """Report sensitivity; deliberately do not invent one 'correct' threshold."""

    thresholds = [float(value) for value in thresholds_deg]
    if not thresholds or any(value <= 0 for value in thresholds):
        raise ValueError("thresholds must be nonempty and positive")
    if any(b <= a for a, b in zip(thresholds, thresholds[1:])):
        raise ValueError("thresholds must be strictly increasing")

    entries: list[SegmentationSweepEntry] = []
    for threshold in thresholds:
        segmentation = segment_grains(
            data,
            phases,
            threshold_deg=threshold,
            neighbor_graph=neighbor_graph,
            minimum_points=minimum_points,
        )
        grains = grain_statistics(data, segmentation, phases)
        gos = np.asarray([grain.gos_deg for grain in grains], dtype=float)
        retained = float(np.mean(segmentation.grain_id >= 0))
        entries.append(
            SegmentationSweepEntry(
                threshold_deg=threshold,
                n_grains=segmentation.n_grains,
                retained_fraction=retained,
                median_gos_deg=(
                    float(np.median(gos)) if len(gos) else float("nan")
                ),
                p95_gos_deg=(
                    float(np.quantile(gos, 0.95)) if len(gos) else float("nan")
                ),
                labels=segmentation.grain_id.copy(),
            )
        )

    ari = tuple(
        adjusted_rand_index(entries[i].labels, entries[i + 1].labels)
        for i in range(len(entries) - 1)
    )
    return SegmentationSweep(
        entries=tuple(entries),
        consecutive_adjusted_rand=ari,
    )


def unique_grain_adjacency(
    segmentation: GrainSegmentation,
    *,
    allowed_grains: set[int] | None = None,
) -> tuple[tuple[int, int], ...]:
    pairs: set[tuple[int, int]] = set()
    labels = segmentation.grain_id
    for point_a, point_b in segmentation.neighbor_graph.edges:
        ga = int(labels[int(point_a)])
        gb = int(labels[int(point_b)])
        if ga < 0 or gb < 0 or ga == gb:
            continue
        a, b = sorted((ga, gb))
        if allowed_grains is not None and (
            a not in allowed_grains or b not in allowed_grains
        ):
            continue
        pairs.add((a, b))
    return tuple(sorted(pairs))


def _candidate_parents(
    child_orientation: np.ndarray,
    variants: Sequence[VariantOR],
) -> tuple[np.ndarray, ...]:
    g = np.asarray(child_orientation, dtype=float)
    return tuple(
        g @ variant.R_parent_from_product.T
        for variant in variants
    )


@dataclass(frozen=True)
class PreparedBoundaryOperatorKernel:
    """Precomputed complete product-symmetry quotient for measured boundaries."""

    adjacency: tuple[tuple[int, int], ...]
    transformed_measured_quaternions: np.ndarray
    product_group_order: int

    @classmethod
    def prepare(
        cls,
        grain_orientations: Sequence[np.ndarray],
        adjacency: Iterable[tuple[int, int]],
        product_symmetry: Sequence[np.ndarray],
    ) -> "PreparedBoundaryOperatorKernel":
        orientations = np.asarray(grain_orientations, dtype=float)
        if (
            orientations.ndim != 3
            or orientations.shape[1:] != (3, 3)
            or len(orientations) == 0
        ):
            raise ValueError(
                "grain_orientations must have shape (N,3,3), N>0"
            )
        for index, matrix in enumerate(orientations):
            require_so3(
                matrix,
                tolerance=2.0e-8,
                name=f"grain orientation {index}",
            )

        pairs = tuple((int(a), int(b)) for a, b in adjacency)
        if not pairs:
            raise ValueError("at least one child-grain boundary is required")
        for a, b in pairs:
            if not (0 <= a < len(orientations) and 0 <= b < len(orientations)):
                raise IndexError("grain adjacency index out of range")
            if a == b:
                raise ValueError("grain adjacency cannot contain self-edges")

        symmetry = np.asarray(product_symmetry, dtype=float)
        if (
            symmetry.ndim != 3
            or symmetry.shape[1:] != (3, 3)
            or len(symmetry) == 0
        ):
            raise ValueError("product_symmetry must have shape (S,3,3)")
        for index, matrix in enumerate(symmetry):
            require_so3(
                matrix,
                tolerance=2.0e-8,
                name=f"product symmetry {index}",
            )

        raw = np.asarray(
            [
                orientations[a].T @ orientations[b]
                for a, b in pairs
            ],
            dtype=float,
        )

        # Exact quotient reduction:
        # reference: S1.T Delta S2 versus the full A.T Q B orbit;
        # reduced:   L Delta K versus Q and Q.T.
        left = np.einsum(
            "sij,bjk->bsik",
            symmetry,
            raw,
            optimize=True,
        )
        transformed = np.einsum(
            "bsij,tjk->bstik",
            left,
            symmetry,
            optimize=True,
        )
        quaternions = _rotation_quaternions(
            transformed.reshape(len(pairs), -1, 3, 3)
        )
        return cls(
            adjacency=pairs,
            transformed_measured_quaternions=quaternions,
            product_group_order=len(symmetry),
        )

    @property
    def n_boundaries(self) -> int:
        return len(self.adjacency)

    @property
    def quotient_representatives_per_boundary(self) -> int:
        return int(self.transformed_measured_quaternions.shape[1])

    def _target_residual_matrix(
        self,
        representatives: np.ndarray,
    ) -> np.ndarray:
        reps = np.asarray(representatives, dtype=float)
        if reps.ndim == 2:
            reps = reps.reshape(1, 3, 3)
        if reps.ndim != 3 or reps.shape[1:] != (3, 3) or len(reps) == 0:
            raise ValueError(
                "operator representatives must have shape (K,3,3), K>0"
            )
        for index, matrix in enumerate(reps):
            require_so3(
                matrix,
                tolerance=2.0e-8,
                name=f"operator representative {index}",
            )

        q_forward = _rotation_quaternions(reps)
        q_inverse = _rotation_quaternions(
            np.transpose(reps, (0, 2, 1))
        )
        targets = np.stack((q_forward, q_inverse), axis=1)  # (K,2,4)

        measured = self.transformed_measured_quaternions  # (B,S^2,4)
        dots = np.abs(
            np.einsum(
                "bsi,kui->bsku",
                measured,
                targets,
                optimize=True,
            )
        )
        best_dots = np.max(dots, axis=(1, 3))  # (B,K)
        return _angles_from_projective_quaternion_dots(best_dots)

    def class_residuals_deg(
        self,
        operators: Sequence[OperatorClass],
    ) -> np.ndarray:
        if not operators:
            raise ValueError("operator library is empty")
        representatives = np.asarray(
            [operator.representative for operator in operators],
            dtype=float,
        )
        return self._target_residual_matrix(representatives)

    def best_class_residuals_deg(
        self,
        operators: Sequence[OperatorClass],
    ) -> np.ndarray:
        residuals = self.class_residuals_deg(operators)
        return np.min(residuals, axis=1)

    def pair_operator_residuals_deg(
        self,
        pair_operators: np.ndarray,
    ) -> np.ndarray:
        """Minimum residual over every supplied physical variant pair."""

        residuals = self._target_residual_matrix(pair_operators)
        return np.min(residuals, axis=1)


@dataclass(frozen=True)
class PreparedParentCompatibilityKernel:
    """Precompute every candidate-parent symmetry representative per grain."""

    candidate_parent_quaternions: np.ndarray
    n_variants: int
    parent_group_order: int

    @classmethod
    def prepare(
        cls,
        grain_orientations: Sequence[np.ndarray],
        variants: Sequence[VariantOR],
        parent_symmetry: Sequence[np.ndarray],
    ) -> "PreparedParentCompatibilityKernel":
        orientations = np.asarray(grain_orientations, dtype=float)
        if (
            orientations.ndim != 3
            or orientations.shape[1:] != (3, 3)
            or len(orientations) == 0
        ):
            raise ValueError(
                "grain_orientations must have shape (N,3,3), N>0"
            )
        if not variants:
            raise ValueError("variants must not be empty")
        symmetry = np.asarray(parent_symmetry, dtype=float)
        if (
            symmetry.ndim != 3
            or symmetry.shape[1:] != (3, 3)
            or len(symmetry) == 0
        ):
            raise ValueError("parent_symmetry must have shape (S,3,3)")

        variant_matrices = np.asarray(
            [variant.R_parent_from_product for variant in variants],
            dtype=float,
        )
        candidates = np.einsum(
            "nij,vkj->nvik",
            orientations,
            variant_matrices,
            optimize=True,
        )
        # Exact candidate parent orientation g_A = g_M @ R_A<-M.T:
        # candidates[n,v,i,k] = sum_j g[n,i,j] R[v,k,j]
        equivalent = np.einsum(
            "nvij,sjk->nvsik",
            candidates,
            symmetry,
            optimize=True,
        )
        quaternions = _rotation_quaternions(
            equivalent.reshape(len(orientations), -1, 3, 3)
        )
        return cls(
            candidate_parent_quaternions=quaternions,
            n_variants=len(variants),
            parent_group_order=len(symmetry),
        )

    def residuals_deg(
        self,
        adjacency: Iterable[tuple[int, int]],
    ) -> np.ndarray:
        pairs = tuple((int(a), int(b)) for a, b in adjacency)
        if not pairs:
            raise ValueError("at least one child-grain boundary is required")
        n = len(self.candidate_parent_quaternions)
        output = np.empty(len(pairs), dtype=float)
        for index, (a, b) in enumerate(pairs):
            if not (0 <= a < n and 0 <= b < n):
                raise IndexError("grain adjacency index out of range")
            first = self.candidate_parent_quaternions[a]
            second = self.candidate_parent_quaternions[b]
            best_dot = float(
                np.max(np.abs(first @ second.T))
            )
            output[index] = float(
                _angles_from_projective_quaternion_dots(best_dot)
            )
        return output


def parent_pair_compatibility_deg(
    first_child: np.ndarray,
    second_child: np.ndarray,
    variants: Sequence[VariantOR],
    parent_symmetry: Sequence[np.ndarray],
) -> float:
    prepared = PreparedParentCompatibilityKernel.prepare(
        [first_child, second_child],
        variants,
        parent_symmetry,
    )
    return float(prepared.residuals_deg([(0, 1)])[0])


def pair_operators_from_variants(
    variants: Sequence[VariantOR],
) -> np.ndarray:
    if len(variants) < 2:
        raise ValueError("at least two physical variants are required")
    matrices = np.asarray(
        [variant.R_parent_from_product for variant in variants],
        dtype=float,
    )
    operators = [
        matrices[i].T @ matrices[j]
        for i in range(len(matrices))
        for j in range(i + 1, len(matrices))
    ]
    return np.asarray(operators, dtype=float)


@dataclass(frozen=True)
class PreparedORVariantEnumerator:
    """Fast exact-equivalent right-product quotient for trial OR matrices."""

    parent_symmetry: np.ndarray
    product_symmetry: np.ndarray
    quotient_tolerance_deg: float

    @classmethod
    def prepare(
        cls,
        parent_phase: EBSDPhase,
        product_phase: EBSDPhase,
        *,
        quotient_tolerance_deg: float = 2.0e-7,
    ) -> "PreparedORVariantEnumerator":
        if quotient_tolerance_deg <= 0.0:
            raise ValueError("quotient_tolerance_deg must be positive")
        parent = np.asarray(
            parent_phase.proper_symmetry_cartesian,
            dtype=float,
        )
        product = np.asarray(
            product_phase.proper_symmetry_cartesian,
            dtype=float,
        )
        return cls(
            parent_symmetry=parent,
            product_symmetry=product,
            quotient_tolerance_deg=float(quotient_tolerance_deg),
        )

    def variants(
        self,
        R_parent_from_product: np.ndarray,
    ) -> tuple[VariantOR, ...]:
        R = require_so3(
            R_parent_from_product,
            tolerance=2.0e-8,
            name="trial OR",
        )
        raw = np.einsum(
            "sij,jk->sik",
            self.parent_symmetry,
            R,
            optimize=True,
        )

        representatives: list[np.ndarray] = []
        labels: list[str] = []
        representative_quaternions: list[np.ndarray] = []

        for parent_index, candidate in enumerate(raw):
            # Right product-symmetry orbit candidate @ S_M.
            orbit = np.einsum(
                "ij,sjk->sik",
                candidate,
                self.product_symmetry,
                optimize=True,
            )
            q_orbit = _rotation_quaternions(orbit)

            duplicate = False
            for q_existing in representative_quaternions:
                best_dot = float(
                    np.max(np.abs(q_orbit @ q_existing))
                )
                angle = float(
                    _angles_from_projective_quaternion_dots(best_dot)
                )
                if angle <= self.quotient_tolerance_deg:
                    duplicate = True
                    break
            if duplicate:
                continue

            representatives.append(candidate)
            labels.append(f"parent_symmetry_{parent_index}")
            representative_quaternions.append(
                _rotation_quaternions(candidate.reshape(1, 3, 3))[0]
            )

        return tuple(
            VariantOR(index, matrix, labels[index])
            for index, matrix in enumerate(representatives)
        )


@dataclass(frozen=True)
class TheoryConsistencyScore:
    n_boundaries: int
    operator_residuals_deg: np.ndarray
    parent_compatibility_residuals_deg: np.ndarray
    operator_median_deg: float
    operator_p90_deg: float
    parent_median_deg: float
    parent_p90_deg: float
    combined_robust_score_deg: float


def score_theory_consistency(
    grain_orientations: Sequence[np.ndarray],
    adjacency: Iterable[tuple[int, int]],
    theory: TheoryLibrary,
    parent_symmetry: Sequence[np.ndarray],
    product_symmetry: Sequence[np.ndarray],
) -> TheoryConsistencyScore:
    """Vectorized but mathematically reference-equivalent theory score."""

    orientations = np.asarray(grain_orientations, dtype=float)
    pairs = tuple((int(a), int(b)) for a, b in adjacency)
    if not pairs:
        raise ValueError("at least one child-grain boundary is required")

    operator_kernel = PreparedBoundaryOperatorKernel.prepare(
        orientations,
        pairs,
        product_symmetry,
    )
    op = operator_kernel.best_class_residuals_deg(
        theory.boundary_operators
    )

    parent_kernel = PreparedParentCompatibilityKernel.prepare(
        orientations,
        theory.variant_set.variants,
        parent_symmetry,
    )
    pa = parent_kernel.residuals_deg(pairs)

    op50 = float(np.median(op))
    op90 = float(np.quantile(op, 0.90))
    pa50 = float(np.median(pa))
    pa90 = float(np.quantile(pa, 0.90))

    score = 0.5 * (op50 + pa50) + 0.125 * (op90 + pa90)
    return TheoryConsistencyScore(
        n_boundaries=len(pairs),
        operator_residuals_deg=op,
        parent_compatibility_residuals_deg=pa,
        operator_median_deg=op50,
        operator_p90_deg=op90,
        parent_median_deg=pa50,
        parent_p90_deg=pa90,
        combined_robust_score_deg=float(score),
    )


@dataclass(frozen=True)
class ConventionHypothesisResult:
    label: str
    score: TheoryConsistencyScore


@dataclass(frozen=True)
class ConventionHypothesisRanking:
    results: tuple[ConventionHypothesisResult, ...]
    best_labels: tuple[str, ...]
    score_gap_to_next_deg: float
    internally_identifiable: bool
    identifiability_note: str


def rank_orientation_hypotheses(
    hypotheses: Mapping[str, np.ndarray],
    adjacency: Iterable[tuple[int, int]],
    theory: TheoryLibrary,
    parent_symmetry: Sequence[np.ndarray],
    product_symmetry: Sequence[np.ndarray],
    *,
    tie_tolerance_deg: float = 1.0e-8,
    minimum_identifiable_gap_deg: float = 0.25,
) -> ConventionHypothesisRanking:
    """Rank explicit convention hypotheses by transformation consistency."""

    if len(hypotheses) < 2:
        raise ValueError("at least two explicit hypotheses are required")
    pairs = tuple(adjacency)
    results = []
    for label, orientations in hypotheses.items():
        score = score_theory_consistency(
            orientations,
            pairs,
            theory,
            parent_symmetry,
            product_symmetry,
        )
        results.append(ConventionHypothesisResult(str(label), score))

    results.sort(
        key=lambda item: (
            item.score.combined_robust_score_deg,
            item.label,
        )
    )
    best_value = results[0].score.combined_robust_score_deg
    best_labels = tuple(
        item.label
        for item in results
        if abs(item.score.combined_robust_score_deg - best_value)
        <= tie_tolerance_deg
    )
    if len(best_labels) > 1:
        gap = 0.0
    else:
        gap = (
            results[1].score.combined_robust_score_deg - best_value
            if len(results) > 1
            else float("inf")
        )
    identifiable = (
        len(best_labels) == 1
        and gap >= minimum_identifiable_gap_deg
    )
    note = (
        "Internal child/child crystallography distinguishes one supplied "
        "hypothesis."
        if identifiable
        else
        "The supplied hypotheses are not uniquely identifiable from internal "
        "child/child crystallography at the requested gap. A common global "
        "sample rotation is an important exact non-identifiability."
    )
    return ConventionHypothesisRanking(
        results=tuple(results),
        best_labels=best_labels,
        score_gap_to_next_deg=float(gap),
        internally_identifiable=identifiable,
        identifiability_note=note,
    )


def _huber(values: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(values)
    quadratic = np.minimum(absolute, delta)
    linear = absolute - quadratic
    return 0.5 * quadratic**2 + delta * linear


@dataclass(frozen=True)
class ORRefinementResult:
    initial_R_parent_from_product: np.ndarray
    fitted_R_parent_from_product: np.ndarray
    raw_correction_angle_deg: float
    symmetry_reduced_change_deg: float
    initial_score: TheoryConsistencyScore
    fitted_score: TheoryConsistencyScore
    objective_initial: float
    objective_fitted: float
    optimizer_success: bool
    optimizer_message: str
    accepted_improvement: bool
    n_starts: int
    objective_evaluations: int


def refine_orientation_relationship_from_child_boundaries(
    grain_orientations: Sequence[np.ndarray],
    adjacency: Iterable[tuple[int, int]],
    parent_phase: EBSDPhase,
    product_phase: EBSDPhase,
    initial_R_parent_from_product: np.ndarray,
    *,
    maximum_correction_deg: float = 5.0,
    trim_fraction: float = 0.65,
    huber_delta_deg: float = 2.0,
    minimum_improvement_deg2: float = 0.02,
    multi_start_step_deg: float = 0.75,
    maximum_iterations: int = 120,
) -> ORRefinementResult:
    """Fit an OR from child/child boundaries with a prepared exact quotient.

    Scientific search space, symmetry quotient, robust objective, multistarts,
    bounds and acceptance criteria are unchanged from the original formulation.
    Only invariant algebra is precomputed.
    """

    orientations = np.asarray(grain_orientations, dtype=float)
    pairs = tuple((int(a), int(b)) for a, b in adjacency)
    if len(orientations) < 2 or not pairs:
        raise ValueError("OR refinement requires child grains and boundaries")
    if maximum_correction_deg <= 0.0:
        raise ValueError("maximum_correction_deg must be positive")
    if not (0.0 < trim_fraction <= 1.0):
        raise ValueError("trim_fraction must lie in (0,1]")
    if huber_delta_deg <= 0.0:
        raise ValueError("huber_delta_deg must be positive")

    initial_R = require_so3(
        initial_R_parent_from_product,
        tolerance=2.0e-8,
        name="initial OR",
    )

    initial_theory = build_theory_library(
        parent_phase,
        product_phase,
        initial_R,
        crosscheck_topology=True,
    )
    initial_score = score_theory_consistency(
        orientations,
        pairs,
        initial_theory,
        parent_phase.proper_symmetry_cartesian,
        product_phase.proper_symmetry_cartesian,
    )

    # Invariant experimental quotient is prepared exactly once.
    boundary_kernel = PreparedBoundaryOperatorKernel.prepare(
        orientations,
        pairs,
        product_phase.proper_symmetry_cartesian,
    )
    # Trial-OR variant quotient is regenerated exactly at each candidate OR, but
    # without rebuilding operator classes or symmetry orbits.
    variant_enumerator = PreparedORVariantEnumerator.prepare(
        parent_phase,
        product_phase,
        quotient_tolerance_deg=(
            initial_theory.variant_set.quotient_deduplication_tolerance_deg
        ),
    )

    max_rad = np.deg2rad(maximum_correction_deg)
    cache: dict[tuple[float, float, float], float] = {}
    evaluation_count = 0

    def objective(rotvec: np.ndarray) -> float:
        nonlocal evaluation_count
        vector = np.asarray(rotvec, dtype=float).reshape(3)
        norm = float(np.linalg.norm(vector))
        if norm > max_rad:
            overflow = (norm - max_rad) / max(max_rad, 1.0e-15)
            return 1.0e4 + 1.0e4 * overflow * overflow

        key = tuple(float(x) for x in np.round(vector, 12))
        if key in cache:
            return cache[key]

        evaluation_count += 1
        correction = Rotation.from_rotvec(vector).as_matrix()
        candidate_R = correction @ initial_R
        variants = variant_enumerator.variants(candidate_R)
        pair_operators = pair_operators_from_variants(variants)
        residuals = np.sort(
            boundary_kernel.pair_operator_residuals_deg(pair_operators)
        )
        keep = max(1, int(ceil(trim_fraction * len(residuals))))
        selected = residuals[:keep]
        value = float(np.mean(_huber(selected, huber_delta_deg)))
        cache[key] = value
        return value

    zero = np.zeros(3, dtype=float)
    step = np.deg2rad(
        min(multi_start_step_deg, maximum_correction_deg / 2.0)
    )
    starts = [zero]
    if step > 0.0:
        for axis in np.eye(3):
            starts.append(step * axis)
            starts.append(-step * axis)

    best = None
    for start_index, start in enumerate(starts):
        result = minimize(
            objective,
            start,
            method="Powell",
            options={
                "maxiter": int(maximum_iterations),
                "xtol": 2.0e-6,
                "ftol": 1.0e-7,
                "disp": False,
            },
        )
        value = float(objective(result.x))
        key = (value, float(np.linalg.norm(result.x)), start_index)
        if best is None or key < best[0]:
            best = (key, result)

    assert best is not None
    result = best[1]
    fitted_rotvec = np.asarray(result.x, dtype=float)
    if np.linalg.norm(fitted_rotvec) > max_rad:
        fitted_rotvec = fitted_rotvec * (
            max_rad / float(np.linalg.norm(fitted_rotvec))
        )
    fitted_R = Rotation.from_rotvec(fitted_rotvec).as_matrix() @ initial_R

    fitted_theory = build_theory_library(
        parent_phase,
        product_phase,
        fitted_R,
        crosscheck_topology=True,
    )
    fitted_score = score_theory_consistency(
        orientations,
        pairs,
        fitted_theory,
        parent_phase.proper_symmetry_cartesian,
        product_phase.proper_symmetry_cartesian,
    )

    objective_initial = float(objective(zero))
    objective_fitted = float(objective(fitted_rotvec))
    raw_change = rotation_angle_deg(fitted_R @ initial_R.T)
    quotient_change = orientation_relationship_distance_deg(
        fitted_R,
        initial_R,
        parent_phase.proper_symmetry_cartesian,
        product_phase.proper_symmetry_cartesian,
    )
    accepted = (
        objective_initial - objective_fitted >= minimum_improvement_deg2
        and raw_change <= maximum_correction_deg + 1.0e-8
    )

    return ORRefinementResult(
        initial_R_parent_from_product=initial_R,
        fitted_R_parent_from_product=fitted_R,
        raw_correction_angle_deg=float(raw_change),
        symmetry_reduced_change_deg=float(quotient_change),
        initial_score=initial_score,
        fitted_score=fitted_score,
        objective_initial=objective_initial,
        objective_fitted=objective_fitted,
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        accepted_improvement=bool(accepted),
        n_starts=len(starts),
        objective_evaluations=int(evaluation_count),
    )


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=int)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = int(self.parent[value])
        return value

    def union(self, first: int, second: int) -> None:
        a = self.find(first)
        b = self.find(second)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


@dataclass(frozen=True)
class VariantGraphDomainCandidate:
    grain_indices: tuple[int, ...]
    reconstruction: ParentReconstruction
    support_weight: float


@dataclass(frozen=True)
class VariantGraphReport:
    domain_candidates: tuple[VariantGraphDomainCandidate, ...]
    grain_membership_count: np.ndarray
    unassigned_grains: tuple[int, ...]
    ambiguous_grains: tuple[int, ...]
    link_tolerance_deg: float


def reconstruct_variant_graph_candidates(
    grain_orientations: Sequence[np.ndarray],
    grain_adjacency: Iterable[tuple[int, int]],
    variants: Sequence[VariantOR],
    product_symmetry: Sequence[np.ndarray],
    parent_symmetry: Sequence[np.ndarray],
    *,
    grain_weights: np.ndarray | None = None,
    link_tolerance_deg: float = 3.0,
    reconstruction_tolerance_deg: float = 3.0,
    minimum_grains: int = 2,
) -> VariantGraphReport:
    """Candidate-level graph reconstruction without forced ambiguous merges."""

    orientations = np.asarray(grain_orientations, dtype=float)
    n_grains = len(orientations)
    n_variants = len(variants)
    if n_grains == 0 or n_variants == 0:
        raise ValueError("grain orientations and variants must be nonempty")
    if link_tolerance_deg <= 0 or reconstruction_tolerance_deg <= 0:
        raise ValueError("graph tolerances must be positive")
    if minimum_grains < 2:
        raise ValueError("minimum_grains must be >= 2")

    if grain_weights is None:
        weights = np.ones(n_grains, dtype=float)
    else:
        weights = np.asarray(grain_weights, dtype=float).reshape(-1)
        if len(weights) != n_grains:
            raise ValueError("grain_weights length mismatch")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("grain_weights must be finite and nonnegative")
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("grain_weights must not all be zero")

    candidates = np.asarray(
        [
            _candidate_parents(orientations[grain], variants)
            for grain in range(n_grains)
        ],
        dtype=float,
    )

    node_count = n_grains * n_variants
    uf = _UnionFind(node_count)

    def node(grain: int, variant: int) -> int:
        return grain * n_variants + variant

    adjacency = tuple((int(a), int(b)) for a, b in grain_adjacency)
    for a, b in adjacency:
        if not (0 <= a < n_grains and 0 <= b < n_grains):
            raise IndexError("grain adjacency index out of range")
        for va in range(n_variants):
            for vb in range(n_variants):
                residual = minimum_disorientation(
                    candidates[a, va],
                    candidates[b, vb],
                    parent_symmetry,
                ).angle_deg
                if residual <= link_tolerance_deg:
                    uf.union(node(a, va), node(b, vb))

    components: dict[int, set[int]] = {}
    for grain in range(n_grains):
        for variant in range(n_variants):
            root = uf.find(node(grain, variant))
            components.setdefault(root, set()).add(grain)

    deduplicated: dict[frozenset[int], VariantGraphDomainCandidate] = {}
    for grains in components.values():
        if len(grains) < minimum_grains:
            continue
        ordered = np.asarray(sorted(grains), dtype=int)
        reconstruction = reconstruct_parent(
            orientations[ordered],
            variants,
            product_symmetry,
            parent_symmetry,
            weights=weights[ordered],
            inlier_tolerance_deg=reconstruction_tolerance_deg,
            variant_acceptance_deg=max(5.0, reconstruction_tolerance_deg),
        )
        inlier_global = tuple(
            int(ordered[index])
            for index in np.flatnonzero(reconstruction.inlier_mask)
        )
        if len(inlier_global) < minimum_grains:
            continue
        support = float(np.sum(weights[list(inlier_global)]))
        key = frozenset(inlier_global)
        candidate = VariantGraphDomainCandidate(
            grain_indices=tuple(sorted(inlier_global)),
            reconstruction=reconstruction,
            support_weight=support,
        )
        previous = deduplicated.get(key)
        if previous is None or (
            candidate.reconstruction.mean_inlier_residual_deg
            < previous.reconstruction.mean_inlier_residual_deg
        ):
            deduplicated[key] = candidate

    domains = list(deduplicated.values())
    domains.sort(
        key=lambda item: (
            -item.support_weight,
            -len(item.grain_indices),
            item.reconstruction.mean_inlier_residual_deg,
            item.grain_indices,
        )
    )

    membership = np.zeros(n_grains, dtype=int)
    for domain in domains:
        membership[list(domain.grain_indices)] += 1

    return VariantGraphReport(
        domain_candidates=tuple(domains),
        grain_membership_count=membership,
        unassigned_grains=tuple(
            int(index) for index in np.flatnonzero(membership == 0)
        ),
        ambiguous_grains=tuple(
            int(index) for index in np.flatnonzero(membership > 1)
        ),
        link_tolerance_deg=float(link_tolerance_deg),
    )


@dataclass(frozen=True)
class ExperimentalValidationReport:
    map_audit: MapAudit
    segmentation: GrainSegmentation
    grains: tuple[Grain, ...]
    segmentation_sweep: SegmentationSweep
    product_grain_global_ids: tuple[int, ...]
    product_grain_adjacency_local: tuple[tuple[int, int], ...]
    theory: TheoryLibrary
    theory_consistency: TheoryConsistencyScore | None
    variant_graph: VariantGraphReport | None


def validate_experimental_map(
    data: EBSDMap,
    phases: Mapping[int, EBSDPhase],
    *,
    parent_phase: EBSDPhase,
    product_phase: EBSDPhase,
    base_R_parent_from_product: np.ndarray,
    segmentation_threshold_deg: float,
    sweep_thresholds_deg: Sequence[float],
    minimum_grain_points: int = 2,
) -> ExperimentalValidationReport:
    """One auditable map-level workflow without hidden parameter fitting."""

    map_audit = audit_map(data)
    segmentation = segment_grains(
        data,
        phases,
        threshold_deg=segmentation_threshold_deg,
        minimum_points=minimum_grain_points,
    )
    grains = grain_statistics(data, segmentation, phases)
    sweep = sweep_segmentation_thresholds(
        data,
        phases,
        sweep_thresholds_deg,
        neighbor_graph=segmentation.neighbor_graph,
        minimum_points=minimum_grain_points,
    )
    theory = build_theory_library(
        parent_phase,
        product_phase,
        base_R_parent_from_product,
    )

    product_grains = tuple(
        grain for grain in grains if grain.phase_id == product_phase.phase_id
    )
    global_ids = tuple(grain.grain_id for grain in product_grains)
    global_to_local = {
        grain_id: index for index, grain_id in enumerate(global_ids)
    }
    global_pairs = unique_grain_adjacency(
        segmentation,
        allowed_grains=set(global_ids),
    )
    local_pairs = tuple(
        (global_to_local[a], global_to_local[b]) for a, b in global_pairs
    )
    orientations = np.asarray(
        [grain.mean_orientation for grain in product_grains],
        dtype=float,
    )

    score = None
    graph = None
    if len(orientations) >= 2 and local_pairs:
        score = score_theory_consistency(
            orientations,
            local_pairs,
            theory,
            parent_phase.proper_symmetry_cartesian,
            product_phase.proper_symmetry_cartesian,
        )
        graph = reconstruct_variant_graph_candidates(
            orientations,
            local_pairs,
            theory.variant_set.variants,
            product_phase.proper_symmetry_cartesian,
            parent_phase.proper_symmetry_cartesian,
            grain_weights=np.asarray(
                [grain.size for grain in product_grains],
                dtype=float,
            ),
            minimum_grains=2,
        )

    return ExperimentalValidationReport(
        map_audit=map_audit,
        segmentation=segmentation,
        grains=grains,
        segmentation_sweep=sweep,
        product_grain_global_ids=global_ids,
        product_grain_adjacency_local=local_pairs,
        theory=theory,
        theory_consistency=score,
        variant_graph=graph,
    )
