from __future__ import annotations

"""Parent reconstruction, variant assignment and operator classification for EBSD."""

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .ebsd_analysis import minimum_disorientation, symmetry_aligned_mean
from .orientation_kernel import require_so3, rotation_angle_deg


@dataclass(frozen=True)
class VariantOR:
    """One parent<-product orientation-relationship variant."""

    variant_index: int
    R_parent_from_product: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        R = require_so3(
            self.R_parent_from_product,
            tolerance=2.0e-8,
            name="variant OR",
        )
        object.__setattr__(self, "R_parent_from_product", R)


@dataclass(frozen=True)
class VariantAssignment:
    variant_index: int | None
    best_residual_deg: float
    second_best_residual_deg: float
    ambiguity_gap_deg: float
    accepted: bool
    ambiguous: bool
    all_residuals_deg: tuple[float, ...]


def predict_product_orientation(
    parent_g: np.ndarray,
    variant: VariantOR,
) -> np.ndarray:
    parent = require_so3(parent_g, tolerance=2.0e-8, name="parent orientation")
    # x_sample = g_A x_A = g_A R_A<-M x_M
    return parent @ variant.R_parent_from_product


def assign_variant(
    measured_product_g: np.ndarray,
    parent_g: np.ndarray,
    variants: Sequence[VariantOR],
    product_symmetry: Sequence[np.ndarray],
    *,
    maximum_residual_deg: float = 5.0,
    minimum_margin_deg: float = 0.5,
) -> VariantAssignment:
    if not variants:
        raise ValueError("variant list must not be empty")
    residuals = []
    for variant in variants:
        predicted = predict_product_orientation(parent_g, variant)
        residuals.append(
            minimum_disorientation(
                predicted,
                measured_product_g,
                product_symmetry,
            ).angle_deg
        )
    order = np.argsort(residuals)
    best_index = int(order[0])
    best = float(residuals[best_index])
    second = (
        float(residuals[int(order[1])])
        if len(order) > 1
        else 180.0
    )
    gap = second - best
    accepted = best <= maximum_residual_deg
    return VariantAssignment(
        variant_index=(
            int(variants[best_index].variant_index) if accepted else None
        ),
        best_residual_deg=best,
        second_best_residual_deg=second,
        ambiguity_gap_deg=gap,
        accepted=accepted,
        ambiguous=bool(accepted and gap < minimum_margin_deg),
        all_residuals_deg=tuple(float(x) for x in residuals),
    )


def _parent_candidate(
    product_g: np.ndarray,
    variant: VariantOR,
) -> np.ndarray:
    product = require_so3(product_g, tolerance=2.0e-8, name="product orientation")
    # g_M = g_A R_A<-M  ->  g_A = g_M R_A<-M^T
    return product @ variant.R_parent_from_product.T


@dataclass(frozen=True)
class ParentReconstruction:
    parent_orientation: np.ndarray
    variant_assignments: tuple[VariantAssignment, ...]
    inlier_mask: np.ndarray
    weighted_inlier_fraction: float
    mean_inlier_residual_deg: float
    maximum_inlier_residual_deg: float
    iterations: int


def reconstruct_parent(
    product_orientations: np.ndarray,
    variants: Sequence[VariantOR],
    product_symmetry: Sequence[np.ndarray],
    parent_symmetry: Sequence[np.ndarray],
    *,
    weights: np.ndarray | None = None,
    inlier_tolerance_deg: float = 3.0,
    maximum_seed_candidates: int = 512,
    maximum_iterations: int = 30,
    variant_acceptance_deg: float = 5.0,
    ambiguity_gap_deg: float = 0.5,
) -> ParentReconstruction:
    """Deterministic robust parent orientation reconstruction.

    Each measured product orientation yields one parent candidate per OR
    variant.  A bounded deterministic consensus search finds the parent
    orientation supported by the largest total weight.  The winning solution is
    then refined by alternating variant assignment and symmetry-aware SO(3)
    averaging.
    """

    measured = np.asarray(product_orientations, dtype=float)
    if measured.ndim != 3 or measured.shape[1:] != (3, 3) or len(measured) == 0:
        raise ValueError("product_orientations must have shape (N,3,3), N>0")
    for index, matrix in enumerate(measured):
        require_so3(matrix, tolerance=2.0e-8, name=f"product[{index}]")
    if not variants:
        raise ValueError("variants must not be empty")
    if inlier_tolerance_deg <= 0 or maximum_iterations < 1:
        raise ValueError("invalid reconstruction tolerances/iterations")

    n = len(measured)
    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if len(w) != n or np.any(~np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite, nonnegative and length N")
        if float(np.sum(w)) <= 0:
            raise ValueError("weights must not all be zero")
    w = w / float(np.sum(w))

    candidates = np.asarray(
        [
            [_parent_candidate(g, variant) for variant in variants]
            for g in measured
        ],
        dtype=float,
    )

    flat = candidates.reshape(-1, 3, 3)
    if len(flat) > maximum_seed_candidates:
        indices = np.linspace(
            0,
            len(flat) - 1,
            maximum_seed_candidates,
            dtype=int,
        )
        seeds = flat[indices]
    else:
        seeds = flat

    best_key = None
    best_seed = None
    for seed_index, seed in enumerate(seeds):
        point_best = []
        for point in range(n):
            residual = min(
                minimum_disorientation(
                    seed,
                    candidates[point, variant_index],
                    parent_symmetry,
                ).angle_deg
                for variant_index in range(len(variants))
            )
            point_best.append(residual)
        point_best = np.asarray(point_best, dtype=float)
        inliers = point_best <= inlier_tolerance_deg
        support = float(np.sum(w[inliers]))
        weighted_error = float(np.sum(w[inliers] * point_best[inliers]))
        key = (-support, weighted_error, seed_index)
        if best_key is None or key < best_key:
            best_key = key
            best_seed = seed

    assert best_seed is not None
    parent = best_seed.copy()
    iterations = 0

    for iteration in range(maximum_iterations):
        iterations = iteration + 1
        chosen = []
        chosen_weights = []
        for point in range(n):
            residuals = [
                minimum_disorientation(
                    parent,
                    candidates[point, variant_index],
                    parent_symmetry,
                ).angle_deg
                for variant_index in range(len(variants))
            ]
            best_variant = int(np.argmin(residuals))
            if residuals[best_variant] <= inlier_tolerance_deg:
                chosen.append(candidates[point, best_variant])
                chosen_weights.append(w[point])

        if not chosen:
            raise ValueError(
                "parent reconstruction lost all inliers during refinement"
            )
        updated = symmetry_aligned_mean(
            np.asarray(chosen),
            parent_symmetry,
            weights=np.asarray(chosen_weights),
        )
        change = minimum_disorientation(
            parent,
            updated,
            parent_symmetry,
        ).angle_deg
        parent = updated
        if change <= 1.0e-8:
            break

    assignments = tuple(
        assign_variant(
            measured[index],
            parent,
            variants,
            product_symmetry,
            maximum_residual_deg=variant_acceptance_deg,
            minimum_margin_deg=ambiguity_gap_deg,
        )
        for index in range(n)
    )
    residuals = np.array(
        [item.best_residual_deg for item in assignments],
        dtype=float,
    )
    inlier = residuals <= inlier_tolerance_deg
    support = float(np.sum(w[inlier]))
    if not np.any(inlier):
        raise AssertionError("refined parent has no inliers")

    return ParentReconstruction(
        parent_orientation=parent,
        variant_assignments=assignments,
        inlier_mask=inlier,
        weighted_inlier_fraction=support,
        mean_inlier_residual_deg=float(
            np.sum(w[inlier] * residuals[inlier]) / np.sum(w[inlier])
        ),
        maximum_inlier_residual_deg=float(np.max(residuals[inlier])),
        iterations=iterations,
    )


@dataclass(frozen=True)
class OperatorClass:
    operator_index: int
    representative: np.ndarray
    equivalent_rotations: tuple[np.ndarray, ...]
    variant_pairs: tuple[tuple[int, int], ...]


def _matrix_unique(
    matrices: Iterable[np.ndarray],
    *,
    tolerance_deg: float = 1.0e-7,
) -> tuple[np.ndarray, ...]:
    unique: list[np.ndarray] = []
    for matrix in matrices:
        candidate = require_so3(matrix, tolerance=2.0e-8, name="operator rotation")
        if not any(
            rotation_angle_deg(candidate @ existing.T, tolerance=2.0e-8)
            <= tolerance_deg
            for existing in unique
        ):
            unique.append(candidate)
    return tuple(unique)


def _symmetry_orbit(
    Q: np.ndarray,
    symmetry: Sequence[np.ndarray],
) -> tuple[np.ndarray, ...]:
    return _matrix_unique(
        S1.T @ Q @ S2
        for S1 in symmetry
        for S2 in symmetry
    )


def build_operator_library(
    variants: Sequence[VariantOR],
    product_symmetry: Sequence[np.ndarray],
    *,
    equivalence_tolerance_deg: float = 1.0e-6,
) -> tuple[OperatorClass, ...]:
    """Build crystallographic product/product operator classes from OR variants."""

    classes: list[dict[str, object]] = []
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            # Product crystal-j -> product crystal-i for common parent.
            Q = variants[i].R_parent_from_product.T @ variants[j].R_parent_from_product
            orbit = _symmetry_orbit(Q, product_symmetry)

            assigned = None
            for class_index, item in enumerate(classes):
                existing = item["equivalent"]
                assert isinstance(existing, tuple)
                residual = min(
                    rotation_angle_deg(Q @ candidate.T, tolerance=2.0e-8)
                    for candidate in existing
                )
                if residual <= equivalence_tolerance_deg:
                    assigned = class_index
                    break

            if assigned is None:
                classes.append(
                    {
                        "representative": Q,
                        "equivalent": _matrix_unique(
                            list(orbit) + [item.T for item in orbit]
                        ),
                        "pairs": [(variants[i].variant_index, variants[j].variant_index)],
                    }
                )
            else:
                pairs = classes[assigned]["pairs"]
                assert isinstance(pairs, list)
                pairs.append((variants[i].variant_index, variants[j].variant_index))

    return tuple(
        OperatorClass(
            operator_index=index,
            representative=np.asarray(item["representative"], dtype=float),
            equivalent_rotations=tuple(item["equivalent"]),  # type: ignore[arg-type]
            variant_pairs=tuple(item["pairs"]),  # type: ignore[arg-type]
        )
        for index, item in enumerate(classes)
    )


@dataclass(frozen=True)
class OperatorAssignment:
    operator_index: int | None
    best_residual_deg: float
    second_best_residual_deg: float
    ambiguity_gap_deg: float
    accepted: bool
    ambiguous: bool


def classify_boundary_operator(
    g1: np.ndarray,
    g2: np.ndarray,
    operators: Sequence[OperatorClass],
    product_symmetry: Sequence[np.ndarray],
    *,
    maximum_residual_deg: float = 3.0,
    minimum_margin_deg: float = 0.5,
) -> OperatorAssignment:
    if not operators:
        raise ValueError("operator library is empty")
    A = require_so3(g1, tolerance=2.0e-8, name="g1")
    B = require_so3(g2, tolerance=2.0e-8, name="g2")
    raw = A.T @ B

    residuals = []
    for operator in operators:
        best = 180.0
        for S1 in product_symmetry:
            for S2 in product_symmetry:
                measured = S1.T @ raw @ S2
                for theoretical in operator.equivalent_rotations:
                    residual = rotation_angle_deg(
                        measured @ theoretical.T,
                        tolerance=2.0e-8,
                    )
                    best = min(best, residual)
        residuals.append(best)

    order = np.argsort(residuals)
    best_position = int(order[0])
    best = float(residuals[best_position])
    second = (
        float(residuals[int(order[1])])
        if len(order) > 1
        else 180.0
    )
    gap = second - best
    accepted = best <= maximum_residual_deg
    return OperatorAssignment(
        operator_index=(
            int(operators[best_position].operator_index)
            if accepted
            else None
        ),
        best_residual_deg=best,
        second_best_residual_deg=second,
        ambiguity_gap_deg=gap,
        accepted=accepted,
        ambiguous=bool(accepted and gap < minimum_margin_deg),
    )


@dataclass(frozen=True)
class ParentDomain:
    grain_indices: tuple[int, ...]
    reconstruction: ParentReconstruction


def reconstruct_parent_domains(
    grain_orientations: Sequence[np.ndarray],
    grain_adjacency: Iterable[tuple[int, int]],
    variants: Sequence[VariantOR],
    product_symmetry: Sequence[np.ndarray],
    parent_symmetry: Sequence[np.ndarray],
    *,
    compatibility_tolerance_deg: float = 3.0,
    grain_weights: np.ndarray | None = None,
) -> tuple[ParentDomain, ...]:
    """Group adjacent product grains that admit a common reconstructed parent."""

    orientations = np.asarray(grain_orientations, dtype=float)
    n = len(orientations)
    if grain_weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(grain_weights, dtype=float).reshape(-1)
        if len(weights) != n:
            raise ValueError("grain_weights length mismatch")

    parent_candidates = [
        [_parent_candidate(orientations[i], variant) for variant in variants]
        for i in range(n)
    ]

    parent = list(range(n))
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for a, b in grain_adjacency:
        a, b = int(a), int(b)
        compatible = False
        for ca in parent_candidates[a]:
            for cb in parent_candidates[b]:
                if minimum_disorientation(ca, cb, parent_symmetry).angle_deg <= compatibility_tolerance_deg:
                    compatible = True
                    break
            if compatible:
                break
        if compatible:
            union(a, b)

    components: dict[int, list[int]] = {}
    for index in range(n):
        components.setdefault(find(index), []).append(index)

    domains = []
    for points in components.values():
        local = reconstruct_parent(
            orientations[points],
            variants,
            product_symmetry,
            parent_symmetry,
            weights=weights[points],
            inlier_tolerance_deg=compatibility_tolerance_deg,
        )
        domains.append(
            ParentDomain(
                grain_indices=tuple(points),
                reconstruction=local,
            )
        )
    domains.sort(key=lambda item: item.grain_indices[0])
    return tuple(domains)
