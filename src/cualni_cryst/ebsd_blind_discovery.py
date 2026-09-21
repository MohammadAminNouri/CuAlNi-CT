from __future__ import annotations

"""Blind discovery of a parent/product orientation relationship from child EBSD.

This module deliberately accepts *no supplied OR*.  The only crystallographic
inputs are:

- parent phase metric + proper point group;
- product phase metric + proper point group;
- measured product-grain orientations;
- product-grain adjacency.

The hidden quantity is the physical orientation relationship

    x_A = R_A_from_M x_M.

For a common parent, product variants satisfy

    R_i = S_A^(i) R_A_from_M,

so an inter-variant product/product operator is

    Q_ij = R_i^T R_j
         = R^T (S_A^(i))^T S_A^(j) R.

The solver therefore searches SO(3) for an R whose complete variant-pair
operator family explains the measured child/child boundary network under the
full proper product symmetry quotient.

Architecture
------------
The search is intentionally two-stage.

1. A relaxed *proposal* scan uses conjugated non-identity parent operations
   ``R.T @ P @ R``.  This cannot certify a solution; it only finds promising
   regions of SO(3).

2. Every retained seed is reranked and locally optimized with the exact frozen
   machinery:
   - complete parent/product variant quotient;
   - every physical variant-pair operator;
   - complete product symmetry quotient;
   - robust trimmed Huber objective.

Thus the coarse stage may only affect speed.  It cannot define the final score,
acceptance, operator family, or reported OR.

Identifiability is explicit.  A low residual is insufficient by itself.

Most importantly, child/child data observe the conjugated parent subgroup
``R.T @ G_A @ R`` rather than a unique physical OR matrix.  Therefore any left
rotation in the SO(3) normalizer of ``G_A`` is an exact inverse-problem gauge.
The code compares and deduplicates candidates in this observable subgroup-
embedding quotient instead of incorrectly demanding equality in the narrower
physical OR quotient.

Multiple inequivalent observable minima, too few transformation-related
boundaries, or too few distinct accepted operator classes are reported as
ambiguous/insufficient, not silently collapsed to one OR.
"""

from dataclasses import dataclass
from math import ceil, log2, pi
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from .ebsd_map import EBSDPhase
from .ebsd_theory_bridge import (
    TheoryLibrary,
    build_theory_library,
    orientation_relationship_distance_deg,
)
from .ebsd_validation import (
    PreparedBoundaryOperatorKernel,
    PreparedORVariantEnumerator,
    pair_operators_from_variants,
)
from .orientation_kernel import require_so3, rotation_angle_deg


@dataclass(frozen=True)
class BlindORSettings:
    global_samples: int = 1024
    coarse_boundary_limit: int = 24
    coarse_keep: int = 32
    exact_keep: int = 12
    local_radius_deg: float = 25.0
    local_max_iterations: int = 80
    trim_fraction: float = 0.70
    huber_delta_deg: float = 2.0
    support_threshold_deg: float = 3.0
    minimum_support_fraction: float = 0.60
    minimum_boundaries: int = 4
    minimum_distinct_operator_classes: int = 2
    candidate_dedup_deg: float = 0.50
    ambiguity_median_gap_deg: float = 0.25
    acceptance_median_deg: float = 2.0
    sobol_seed: int = 20260921
    quotient_tolerance_deg: float = 2.0e-7
    operator_equivalence_tolerance_deg: float = 2.0e-6

    def __post_init__(self) -> None:
        if self.global_samples < 64:
            raise ValueError("global_samples must be >= 64")
        if self.coarse_boundary_limit < 2:
            raise ValueError("coarse_boundary_limit must be >= 2")
        if self.coarse_keep < 2:
            raise ValueError("coarse_keep must be >= 2")
        if self.exact_keep < 1 or self.exact_keep > self.coarse_keep:
            raise ValueError("exact_keep must lie in [1, coarse_keep]")
        if self.local_max_iterations < 1:
            raise ValueError("local_max_iterations must be >= 1")
        if not (0.0 < self.trim_fraction <= 1.0):
            raise ValueError("trim_fraction must lie in (0,1]")
        if self.huber_delta_deg <= 0.0:
            raise ValueError("huber_delta_deg must be positive")
        if self.support_threshold_deg <= 0.0:
            raise ValueError("support_threshold_deg must be positive")
        if not (0.0 < self.minimum_support_fraction <= 1.0):
            raise ValueError("minimum_support_fraction must lie in (0,1]")
        if self.minimum_boundaries < 2:
            raise ValueError("minimum_boundaries must be >= 2")
        if self.minimum_distinct_operator_classes < 1:
            raise ValueError(
                "minimum_distinct_operator_classes must be >= 1"
            )
        if self.candidate_dedup_deg <= 0.0:
            raise ValueError("candidate_dedup_deg must be positive")
        if self.ambiguity_median_gap_deg < 0.0:
            raise ValueError(
                "ambiguity_median_gap_deg must be nonnegative"
            )
        if self.acceptance_median_deg <= 0.0:
            raise ValueError("acceptance_median_deg must be positive")


@dataclass(frozen=True)
class BlindORCandidate:
    R_parent_from_product: np.ndarray
    objective: float
    median_residual_deg: float
    p90_residual_deg: float
    maximum_residual_deg: float
    support_fraction: float
    n_variants: int
    n_operator_classes: int
    n_distinct_accepted_operator_classes: int
    accepted_boundary_fraction: float
    local_optimizer_success: bool
    local_optimizer_message: str


@dataclass(frozen=True)
class BlindORResult:
    status: str
    best: BlindORCandidate | None
    alternatives: tuple[BlindORCandidate, ...]
    n_boundaries: int
    n_global_samples: int
    n_exact_seeds: int
    n_local_optimizations: int
    evidence_note: str
    identification_target: str = "child_child_observable_subgroup_embedding"

    @property
    def identified(self) -> bool:
        return self.status == "identified_observable_class"


def _huber(values: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(np.asarray(values, dtype=float))
    quadratic = np.minimum(absolute, delta)
    linear = absolute - quadratic
    return 0.5 * quadratic**2 + delta * linear


def _robust_objective(
    residuals_deg: np.ndarray,
    *,
    trim_fraction: float,
    huber_delta_deg: float,
) -> float:
    residuals = np.sort(np.asarray(residuals_deg, dtype=float))
    if residuals.ndim != 1 or len(residuals) == 0:
        return float("inf")
    keep = max(1, int(ceil(trim_fraction * len(residuals))))
    return float(
        np.mean(_huber(residuals[:keep], huber_delta_deg))
    )


def _uniform_so3_sobol(count: int, *, seed: int) -> np.ndarray:
    """Deterministic low-discrepancy SO(3) samples via Shoemake quaternions."""

    if count < 1:
        raise ValueError("count must be positive")
    exponent = int(ceil(log2(count)))
    sampler = qmc.Sobol(d=3, scramble=True, seed=int(seed))
    u = sampler.random_base2(exponent)[:count]
    u1 = u[:, 0]
    u2 = u[:, 1]
    u3 = u[:, 2]

    # SciPy quaternion order is (x, y, z, w).
    q = np.column_stack(
        (
            np.sqrt(1.0 - u1) * np.sin(2.0 * pi * u2),
            np.sqrt(1.0 - u1) * np.cos(2.0 * pi * u2),
            np.sqrt(u1) * np.sin(2.0 * pi * u3),
            np.sqrt(u1) * np.cos(2.0 * pi * u3),
        )
    )
    q /= np.linalg.norm(q, axis=1)[:, None]
    return Rotation.from_quat(q).as_matrix()


def _rotation_quaternions(matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrices, dtype=float)
    if values.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix array must end in (3,3)")
    flat = values.reshape(-1, 3, 3)
    quaternions = Rotation.from_matrix(flat).as_quat()
    quaternions /= np.linalg.norm(quaternions, axis=1)[:, None]
    return quaternions.reshape(values.shape[:-2] + (4,))


def _angles_from_projective_dots(dots: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(dots, dtype=float), 0.0, 1.0)
    sine_half = np.sqrt(np.maximum(0.0, 1.0 - value * value))
    return np.degrees(2.0 * np.arctan2(sine_half, value))


def blind_observable_embedding_distance_deg(
    first_R_parent_from_product: np.ndarray,
    second_R_parent_from_product: np.ndarray,
    parent_symmetry: Sequence[np.ndarray],
    product_symmetry: Sequence[np.ndarray],
) -> float:
    """Distance between ORs in the *child/child observable* quotient.

    Child/child boundary operators depend on the conjugated parent subgroup

        H(R) = R.T @ G_A @ R.

    Consequently the inverse problem cannot distinguish ``R`` from ``N @ R``
    for any rotation ``N`` in the normalizer of the parent proper group:
    ``N.T @ G_A @ N = G_A``.  This gauge can be strictly larger than the
    crystallographic parent group itself (for example D4 -> D8, D6 -> D12,
    and D2 -> O in SO(3)).

    Product-frame symmetry further identifies embeddings by conjugation:
    ``H ~ S_M.T @ H @ S_M``.

    Rather than hard-code normalizers point-group by point-group, this function
    compares the complete finite conjugated parent subgroups directly.  The
    result is a symmetric Hausdorff SO(3) distance minimized over all proper
    product symmetries.  It is zero exactly for the finite subgroup embedding
    equivalence that child/child operators can observe.

    This metric is intentionally *not* the standard physical OR quotient
    distance.  The latter is appropriate when an OR itself is externally
    defined; this one is appropriate for the blind inverse problem where only
    child/child relations are observed.
    """

    R1 = require_so3(
        first_R_parent_from_product,
        tolerance=2.0e-8,
        name="first blind OR",
    )
    R2 = require_so3(
        second_R_parent_from_product,
        tolerance=2.0e-8,
        name="second blind OR",
    )
    parent = np.asarray(parent_symmetry, dtype=float)
    product = np.asarray(product_symmetry, dtype=float)
    if (
        parent.ndim != 3
        or parent.shape[1:] != (3, 3)
        or len(parent) == 0
    ):
        raise ValueError("parent_symmetry must have shape (P,3,3), P>0")
    if (
        product.ndim != 3
        or product.shape[1:] != (3, 3)
        or len(product) == 0
    ):
        raise ValueError("product_symmetry must have shape (M,3,3), M>0")

    H1 = np.einsum(
        "ji,pjk,kl->pil",
        R1,
        parent,
        R1,
        optimize=True,
    )
    H2 = np.einsum(
        "ji,pjk,kl->pil",
        R2,
        parent,
        R2,
        optimize=True,
    )
    q1 = _rotation_quaternions(H1)

    best = 180.0
    for symmetry in product:
        S = require_so3(
            symmetry,
            tolerance=2.0e-8,
            name="product proper symmetry",
        )
        transformed = np.einsum(
            "ji,pjk,kl->pil",
            S,
            H2,
            S,
            optimize=True,
        )
        q2 = _rotation_quaternions(transformed)
        dots = np.abs(q1 @ q2.T)
        angles = _angles_from_projective_dots(dots)
        directed_12 = float(np.max(np.min(angles, axis=1)))
        directed_21 = float(np.max(np.min(angles, axis=0)))
        distance = max(directed_12, directed_21)
        if distance < best:
            best = distance
    return float(best)


def _deterministic_boundary_subset(
    adjacency: tuple[tuple[int, int], ...],
    limit: int,
) -> tuple[tuple[int, int], ...]:
    if len(adjacency) <= limit:
        return adjacency
    # Evenly sample the sorted topology.  This is deliberately independent of
    # any theoretical OR or expected answer.
    positions = np.linspace(0, len(adjacency) - 1, limit, dtype=int)
    return tuple(adjacency[int(index)] for index in positions)


def _nonidentity_parent_operations(
    parent_phase: EBSDPhase,
) -> np.ndarray:
    operations = []
    for operation in parent_phase.proper_symmetry_cartesian:
        R = require_so3(
            operation,
            tolerance=2.0e-8,
            name="parent proper symmetry",
        )
        if rotation_angle_deg(R, tolerance=2.0e-8) > 1.0e-8:
            operations.append(R)
    if not operations:
        return np.empty((0, 3, 3), dtype=float)
    return np.asarray(operations, dtype=float)


def _relaxed_batch_scores(
    rotations: np.ndarray,
    kernel: PreparedBoundaryOperatorKernel,
    parent_operations: np.ndarray,
    *,
    trim_fraction: float,
    huber_delta_deg: float,
    batch_size: int = 16,
) -> np.ndarray:
    """Proposal-only score over conjugated parent symmetry operators."""

    rotations = np.asarray(rotations, dtype=float)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("rotations must have shape (N,3,3)")
    if len(parent_operations) == 0:
        return np.full(len(rotations), np.inf, dtype=float)

    measured = kernel.transformed_measured_quaternions
    scores = np.empty(len(rotations), dtype=float)

    for start in range(0, len(rotations), batch_size):
        stop = min(start + batch_size, len(rotations))
        R = rotations[start:stop]

        # Q = R.T @ P @ R
        Q = np.einsum(
            "kji,pjl,klm->kpim",
            R,
            parent_operations,
            R,
            optimize=True,
        )
        q_forward = _rotation_quaternions(Q)
        q_inverse = _rotation_quaternions(
            np.transpose(Q, (0, 1, 3, 2))
        )

        dots_forward = np.abs(
            np.einsum(
                "bsi,kpi->kbsp",
                measured,
                q_forward,
                optimize=True,
            )
        )
        dots_inverse = np.abs(
            np.einsum(
                "bsi,kpi->kbsp",
                measured,
                q_inverse,
                optimize=True,
            )
        )
        best_dots = np.maximum(
            np.max(dots_forward, axis=(2, 3)),
            np.max(dots_inverse, axis=(2, 3)),
        )
        residuals = _angles_from_projective_dots(best_dots)

        for local_index, row in enumerate(residuals):
            scores[start + local_index] = _robust_objective(
                row,
                trim_fraction=trim_fraction,
                huber_delta_deg=huber_delta_deg,
            )

    return scores


class _ExactObjective:
    def __init__(
        self,
        *,
        kernel: PreparedBoundaryOperatorKernel,
        variant_enumerator: PreparedORVariantEnumerator,
        settings: BlindORSettings,
    ) -> None:
        self.kernel = kernel
        self.variant_enumerator = variant_enumerator
        self.settings = settings
        self.cache: dict[tuple[float, ...], tuple[float, np.ndarray, int]] = {}

    def evaluate(
        self,
        R_parent_from_product: np.ndarray,
    ) -> tuple[float, np.ndarray, int]:
        R = require_so3(
            R_parent_from_product,
            tolerance=2.0e-8,
            name="blind OR candidate",
        )
        key = tuple(float(x) for x in np.round(R.ravel(), 11))
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        variants = self.variant_enumerator.variants(R)
        if len(variants) < 2:
            result = (
                float("inf"),
                np.full(self.kernel.n_boundaries, 180.0, dtype=float),
                len(variants),
            )
            self.cache[key] = result
            return result

        pair_operators = pair_operators_from_variants(variants)
        residuals = self.kernel.pair_operator_residuals_deg(
            pair_operators
        )
        objective = _robust_objective(
            residuals,
            trim_fraction=self.settings.trim_fraction,
            huber_delta_deg=self.settings.huber_delta_deg,
        )
        result = (objective, residuals, len(variants))
        self.cache[key] = result
        return result


def _candidate_summary(
    R: np.ndarray,
    *,
    objective_engine: _ExactObjective,
    parent_phase: EBSDPhase,
    product_phase: EBSDPhase,
    settings: BlindORSettings,
    optimizer_success: bool,
    optimizer_message: str,
) -> BlindORCandidate:
    objective, residuals, n_variants = objective_engine.evaluate(R)

    theory = build_theory_library(
        parent_phase,
        product_phase,
        R,
        quotient_tolerance_deg=settings.quotient_tolerance_deg,
        operator_equivalence_tolerance_deg=(
            settings.operator_equivalence_tolerance_deg
        ),
        crosscheck_topology=True,
    )
    class_residuals = objective_engine.kernel.class_residuals_deg(
        theory.boundary_operators
    )
    best_class = np.min(class_residuals, axis=1)
    best_class_index = np.argmin(class_residuals, axis=1)
    accepted = best_class <= settings.support_threshold_deg
    distinct = len(set(int(x) for x in best_class_index[accepted]))

    return BlindORCandidate(
        R_parent_from_product=np.asarray(R, dtype=float),
        objective=float(objective),
        median_residual_deg=float(np.median(residuals)),
        p90_residual_deg=float(np.quantile(residuals, 0.90)),
        maximum_residual_deg=float(np.max(residuals)),
        support_fraction=float(
            np.mean(residuals <= settings.support_threshold_deg)
        ),
        n_variants=int(n_variants),
        n_operator_classes=theory.n_boundary_operators,
        n_distinct_accepted_operator_classes=int(distinct),
        accepted_boundary_fraction=float(np.mean(accepted)),
        local_optimizer_success=bool(optimizer_success),
        local_optimizer_message=str(optimizer_message),
    )


def discover_orientation_relationship_blind(
    grain_orientations: Sequence[np.ndarray],
    adjacency: Iterable[tuple[int, int]],
    parent_phase: EBSDPhase,
    product_phase: EBSDPhase,
    *,
    settings: BlindORSettings | None = None,
) -> BlindORResult:
    """Discover an OR without an initial, literature, experimental, or CT OR.

    The solver receives no expected OR and no expected variant/operator answer.
    """

    settings = BlindORSettings() if settings is None else settings

    orientations = np.asarray(grain_orientations, dtype=float)
    if (
        orientations.ndim != 3
        or orientations.shape[1:] != (3, 3)
        or len(orientations) < 2
    ):
        raise ValueError(
            "grain_orientations must have shape (N,3,3) with N>=2"
        )
    for index, orientation in enumerate(orientations):
        require_so3(
            orientation,
            tolerance=2.0e-8,
            name=f"grain orientation {index}",
        )

    pairs = tuple(sorted(set((int(a), int(b)) for a, b in adjacency)))
    for a, b in pairs:
        if a == b:
            raise ValueError("adjacency cannot contain self-edges")
        if not (0 <= a < len(orientations) and 0 <= b < len(orientations)):
            raise IndexError("adjacency index out of range")

    if len(pairs) < settings.minimum_boundaries:
        return BlindORResult(
            status="insufficient_evidence",
            best=None,
            alternatives=tuple(),
            n_boundaries=len(pairs),
            n_global_samples=0,
            n_exact_seeds=0,
            n_local_optimizations=0,
            evidence_note=(
                f"Only {len(pairs)} child/child boundaries are available; "
                f"at least {settings.minimum_boundaries} are required."
            ),
        )

    parent_operations = _nonidentity_parent_operations(parent_phase)
    if len(parent_operations) == 0:
        return BlindORResult(
            status="insufficient_evidence",
            best=None,
            alternatives=tuple(),
            n_boundaries=len(pairs),
            n_global_samples=0,
            n_exact_seeds=0,
            n_local_optimizations=0,
            evidence_note=(
                "The parent proper point group has no non-identity rotation; "
                "child/child operator conjugation cannot determine an OR."
            ),
        )

    full_kernel = PreparedBoundaryOperatorKernel.prepare(
        orientations,
        pairs,
        product_phase.proper_symmetry_cartesian,
    )
    coarse_pairs = _deterministic_boundary_subset(
        pairs, settings.coarse_boundary_limit
    )
    coarse_kernel = PreparedBoundaryOperatorKernel.prepare(
        orientations,
        coarse_pairs,
        product_phase.proper_symmetry_cartesian,
    )
    variant_enumerator = PreparedORVariantEnumerator.prepare(
        parent_phase,
        product_phase,
        quotient_tolerance_deg=settings.quotient_tolerance_deg,
    )
    exact = _ExactObjective(
        kernel=full_kernel,
        variant_enumerator=variant_enumerator,
        settings=settings,
    )

    rotations = _uniform_so3_sobol(
        settings.global_samples,
        seed=settings.sobol_seed,
    )
    # Deterministically include identity.  It is not privileged scientifically;
    # it prevents a low-discrepancy scramble from accidentally omitting the
    # neighbourhood of one coordinate-origin representative.
    rotations = np.concatenate((np.eye(3)[None, :, :], rotations), axis=0)

    relaxed_scores = _relaxed_batch_scores(
        rotations,
        coarse_kernel,
        parent_operations,
        trim_fraction=settings.trim_fraction,
        huber_delta_deg=settings.huber_delta_deg,
    )
    coarse_order = np.argsort(relaxed_scores)[: settings.coarse_keep]

    exact_seed_records: list[tuple[float, int, np.ndarray]] = []
    for rank, index in enumerate(coarse_order):
        R = rotations[int(index)]
        objective, _, _ = exact.evaluate(R)
        exact_seed_records.append((float(objective), rank, R))
    exact_seed_records.sort(key=lambda item: (item[0], item[1]))
    seeds = [
        record[2]
        for record in exact_seed_records[: settings.exact_keep]
        if np.isfinite(record[0])
    ]

    if not seeds:
        return BlindORResult(
            status="inconsistent",
            best=None,
            alternatives=tuple(),
            n_boundaries=len(pairs),
            n_global_samples=len(rotations),
            n_exact_seeds=0,
            n_local_optimizations=0,
            evidence_note=(
                "No globally sampled orientation produced at least two "
                "crystallographically distinct variants."
            ),
        )

    radius = np.deg2rad(settings.local_radius_deg)
    optimized: list[tuple[float, np.ndarray, bool, str]] = []

    for seed_index, seed_R in enumerate(seeds):
        local_cache: dict[tuple[float, float, float], float] = {}

        def local_objective(vector: np.ndarray) -> float:
            v = np.asarray(vector, dtype=float).reshape(3)
            key = tuple(float(x) for x in np.round(v, 11))
            if key in local_cache:
                return local_cache[key]
            candidate = Rotation.from_rotvec(v).as_matrix() @ seed_R
            value = exact.evaluate(candidate)[0]
            local_cache[key] = float(value)
            return float(value)

        result = minimize(
            local_objective,
            np.zeros(3, dtype=float),
            method="Powell",
            bounds=[(-radius, radius)] * 3,
            options={
                "maxiter": int(settings.local_max_iterations),
                "xtol": 1.0e-6,
                "ftol": 1.0e-8,
                "disp": False,
            },
        )
        R_final = Rotation.from_rotvec(
            np.asarray(result.x, dtype=float)
        ).as_matrix() @ seed_R
        objective = exact.evaluate(R_final)[0]
        optimized.append(
            (
                float(objective),
                R_final,
                bool(result.success),
                str(result.message),
            )
        )

    optimized.sort(key=lambda item: item[0])

    # Deduplicate only after exact local optimization.  Crucially, the inverse
    # child/child problem is identifiable only up to the embedding
    #
    #     H(R) = R.T G_A R
    #
    # modulo product conjugation.  Using the ordinary physical OR quotient here
    # is mathematically wrong whenever the parent-group normalizer is larger
    # than G_A itself (D4, D6, D2 are important examples).
    unique: list[tuple[float, np.ndarray, bool, str]] = []
    for record in optimized:
        _, R, _, _ = record
        if any(
            blind_observable_embedding_distance_deg(
                R,
                existing[1],
                parent_phase.proper_symmetry_cartesian,
                product_phase.proper_symmetry_cartesian,
            )
            <= settings.candidate_dedup_deg
            for existing in unique
        ):
            continue
        unique.append(record)

    summaries = tuple(
        _candidate_summary(
            record[1],
            objective_engine=exact,
            parent_phase=parent_phase,
            product_phase=product_phase,
            settings=settings,
            optimizer_success=record[2],
            optimizer_message=record[3],
        )
        for record in unique[: max(2, settings.exact_keep)]
    )

    if not summaries:
        return BlindORResult(
            status="inconsistent",
            best=None,
            alternatives=tuple(),
            n_boundaries=len(pairs),
            n_global_samples=len(rotations),
            n_exact_seeds=len(seeds),
            n_local_optimizations=len(optimized),
            evidence_note="No finite exact local solution survived quotient deduplication.",
        )

    best = summaries[0]
    accepted_fit = (
        best.median_residual_deg <= settings.acceptance_median_deg
        and best.support_fraction >= settings.minimum_support_fraction
    )
    operator_evidence = (
        best.n_distinct_accepted_operator_classes
        >= settings.minimum_distinct_operator_classes
    )

    if not accepted_fit:
        status = "inconsistent"
        note = (
            "The best blind OR does not explain enough measured boundaries "
            "within the configured residual/support criteria."
        )
    elif not operator_evidence:
        status = "insufficient_evidence"
        note = (
            "A low-residual OR exists, but too few distinct accepted "
            "product/product operator classes are observed to claim a unique "
            "orientation relationship."
        )
    else:
        competing = [
            candidate
            for candidate in summaries[1:]
            if (
                candidate.median_residual_deg
                - best.median_residual_deg
                <= settings.ambiguity_median_gap_deg
            )
        ]
        if competing:
            status = "ambiguous_observable_class"
            note = (
                "Multiple child/child-observable subgroup embeddings explain the "
                "boundary network within the configured ambiguity gap."
            )
        else:
            status = "identified_observable_class"
            note = (
                "One child/child-observable subgroup embedding is supported by "
                "multiple distinct operator classes and separated from "
                "the retained observable alternatives. The returned OR "
                "matrix is one representative; child/child data alone "
                "cannot resolve parent-normalizer gauge freedom."
            )

    return BlindORResult(
        status=status,
        best=best,
        alternatives=summaries[1:],
        n_boundaries=len(pairs),
        n_global_samples=len(rotations),
        n_exact_seeds=len(seeds),
        n_local_optimizations=len(optimized),
        evidence_note=note,
    )
