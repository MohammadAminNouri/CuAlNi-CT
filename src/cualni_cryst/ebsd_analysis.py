from __future__ import annotations

"""Phase-aware EBSD neighborhood, grain, KAM and boundary analysis."""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .ebsd_map import EBSDMap, EBSDPhase
from .orientation_kernel import require_so3, rotation_angle_deg


def _project_so3(matrix: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(matrix, dtype=float))
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ correction @ Vt
    return require_so3(R, tolerance=2.0e-9, name="mean orientation")


@dataclass(frozen=True)
class Disorientation:
    angle_deg: float
    rotation_crystal1_from_crystal2: np.ndarray
    symmetry_index_1: int
    symmetry_index_2: int


def minimum_disorientation(
    g1: np.ndarray,
    g2: np.ndarray,
    symmetry: Sequence[np.ndarray],
) -> Disorientation:
    """Full same-phase crystal-symmetry quotient distance.

    ``g`` maps crystal -> sample.  For equivalent representatives
    ``g1 S1`` and ``g2 S2`` the crystal-frame relative rotation is

        Delta = S1.T @ g1.T @ g2 @ S2.

    The minimum principal SO(3) angle is returned.
    """

    A = require_so3(g1, tolerance=2.0e-8, name="g1")
    B = require_so3(g2, tolerance=2.0e-8, name="g2")
    if not symmetry:
        raise ValueError("symmetry list must not be empty")

    best = None
    raw = A.T @ B
    for i, S1 in enumerate(symmetry):
        Q1 = require_so3(S1, tolerance=2.0e-8, name="symmetry")
        for j, S2 in enumerate(symmetry):
            Q2 = require_so3(S2, tolerance=2.0e-8, name="symmetry")
            delta = Q1.T @ raw @ Q2
            angle = rotation_angle_deg(delta, tolerance=2.0e-8)
            key = (angle, i, j)
            if best is None or key < best[0]:
                best = (key, delta)
    assert best is not None
    (angle, i, j), delta = best
    return Disorientation(
        angle_deg=float(angle),
        rotation_crystal1_from_crystal2=delta,
        symmetry_index_1=int(i),
        symmetry_index_2=int(j),
    )


def symmetry_aligned_mean(
    orientations: np.ndarray,
    symmetry: Sequence[np.ndarray],
    *,
    weights: np.ndarray | None = None,
    max_iterations: int = 50,
    convergence_deg: float = 1.0e-9,
) -> np.ndarray:
    """Iterative symmetry-aware chordal mean on SO(3).

    Each measured orientation is first moved to the symmetry-equivalent
    representative closest to the current mean.  Their weighted matrix average
    is then projected to SO(3).  This avoids averaging different symmetry copies.
    """

    values = np.asarray(orientations, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or len(values) == 0:
        raise ValueError("orientations must have shape (N,3,3), N>0")
    for index, matrix in enumerate(values):
        require_so3(matrix, tolerance=2.0e-8, name=f"orientation[{index}]")

    if weights is None:
        w = np.ones(len(values), dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if len(w) != len(values):
            raise ValueError("weights length mismatch")
        if np.any(~np.isfinite(w)) or np.any(w < 0) or float(np.sum(w)) <= 0:
            raise ValueError("weights must be finite, nonnegative and not all zero")
    w = w / float(np.sum(w))

    mean = values[0].copy()
    for _ in range(max_iterations):
        aligned = []
        for matrix in values:
            candidates = [matrix @ S for S in symmetry]
            angles = [
                rotation_angle_deg(mean.T @ candidate, tolerance=2.0e-8)
                for candidate in candidates
            ]
            aligned.append(candidates[int(np.argmin(angles))])
        raw_mean = np.tensordot(w, np.asarray(aligned), axes=(0, 0))
        updated = _project_so3(raw_mean)
        change = rotation_angle_deg(mean.T @ updated, tolerance=2.0e-8)
        mean = updated
        if change <= convergence_deg:
            break
    return mean


@dataclass(frozen=True)
class NeighborGraph:
    edges: np.ndarray
    radius: float
    coordinate_dimension: int


def build_neighbor_graph(
    data: EBSDMap,
    *,
    radius: float | None = None,
    radius_factor: float = 1.15,
) -> NeighborGraph:
    """Build immediate spatial neighbors from physical coordinates.

    If no radius is supplied, map header X/Y/Z step sizes are preferred.
    Otherwise the median positive nearest-neighbor distance is used.
    """

    if data.n_points < 2:
        return NeighborGraph(
            edges=np.empty((0, 2), dtype=int),
            radius=0.0,
            coordinate_dimension=2,
        )
    if radius_factor <= 1.0:
        raise ValueError("radius_factor must be > 1")

    dimension = 3 if np.ptp(data.z) > 0 else 2
    coords = data.coordinates[:, :dimension]

    if radius is None:
        step_candidates = []
        for key in ("XSTEP", "YSTEP", "ZSTEP"):
            value = data.metadata.get(key)
            if value is not None:
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if numeric > 0:
                    step_candidates.append(numeric)
        if step_candidates:
            radius = max(step_candidates) * radius_factor
        else:
            tree = cKDTree(coords)
            distances, _ = tree.query(coords, k=2)
            nearest = np.asarray(distances[:, 1], dtype=float)
            positive = nearest[nearest > 1.0e-12]
            if len(positive) == 0:
                raise ValueError(
                    "cannot infer neighbor spacing: all coordinates are duplicated"
                )
            radius = float(np.median(positive) * radius_factor)

    if radius <= 0:
        raise ValueError("neighbor radius must be positive")

    tree = cKDTree(coords)
    pairs = sorted(tree.query_pairs(radius))
    edges = (
        np.asarray(pairs, dtype=int).reshape(-1, 2)
        if pairs
        else np.empty((0, 2), dtype=int)
    )
    return NeighborGraph(edges=edges, radius=float(radius), coordinate_dimension=dimension)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=int)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, value: int) -> int:
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass(frozen=True)
class GrainSegmentation:
    grain_id: np.ndarray
    n_grains: int
    threshold_deg: float
    neighbor_graph: NeighborGraph


def segment_grains(
    data: EBSDMap,
    phases: Mapping[int, EBSDPhase],
    *,
    threshold_deg: float = 5.0,
    neighbor_graph: NeighborGraph | None = None,
    valid_mask: np.ndarray | None = None,
    minimum_points: int = 1,
) -> GrainSegmentation:
    if threshold_deg <= 0:
        raise ValueError("threshold_deg must be positive")
    if minimum_points < 1:
        raise ValueError("minimum_points must be >= 1")
    if neighbor_graph is None:
        neighbor_graph = build_neighbor_graph(data)
    if valid_mask is None:
        valid = data.indexed.copy()
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if len(valid) != data.n_points:
            raise ValueError("valid_mask length mismatch")
        valid &= data.indexed

    unknown = sorted(set(data.phase_id[valid]).difference(phases))
    if unknown:
        raise ValueError(f"indexed map contains undefined phase IDs: {unknown}")

    uf = _UnionFind(data.n_points)
    for a, b in neighbor_graph.edges:
        a = int(a)
        b = int(b)
        if not valid[a] or not valid[b]:
            continue
        phase = int(data.phase_id[a])
        if phase != int(data.phase_id[b]):
            continue
        symmetry = phases[phase].proper_symmetry_cartesian
        angle = minimum_disorientation(
            data.orientations[a],
            data.orientations[b],
            symmetry,
        ).angle_deg
        if angle <= threshold_deg:
            uf.union(a, b)

    components: dict[int, list[int]] = {}
    for index in np.flatnonzero(valid):
        root = uf.find(int(index))
        components.setdefault(root, []).append(int(index))

    labels = np.full(data.n_points, -1, dtype=int)
    kept = [
        points for points in components.values()
        if len(points) >= minimum_points
    ]
    kept.sort(key=lambda points: min(points))
    for grain, points in enumerate(kept):
        labels[points] = grain

    return GrainSegmentation(
        grain_id=labels,
        n_grains=len(kept),
        threshold_deg=float(threshold_deg),
        neighbor_graph=neighbor_graph,
    )


@dataclass(frozen=True)
class Grain:
    grain_id: int
    phase_id: int
    point_indices: np.ndarray
    size: int
    centroid: np.ndarray
    mean_orientation: np.ndarray
    gos_deg: float
    maximum_spread_deg: float


def grain_statistics(
    data: EBSDMap,
    segmentation: GrainSegmentation,
    phases: Mapping[int, EBSDPhase],
    *,
    weights: np.ndarray | None = None,
) -> tuple[Grain, ...]:
    grains: list[Grain] = []
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != data.n_points:
            raise ValueError("weights length mismatch")

    for grain_id in range(segmentation.n_grains):
        indices = np.flatnonzero(segmentation.grain_id == grain_id)
        if len(indices) == 0:
            continue
        phase_ids = np.unique(data.phase_id[indices])
        if len(phase_ids) != 1:
            raise AssertionError("grain segmentation mixed different phases")
        phase_id = int(phase_ids[0])
        phase = phases[phase_id]
        local_weights = None if weights is None else weights[indices]
        mean = symmetry_aligned_mean(
            data.orientations[indices],
            phase.proper_symmetry_cartesian,
            weights=local_weights,
        )
        residuals = np.array(
            [
                minimum_disorientation(
                    mean,
                    data.orientations[index],
                    phase.proper_symmetry_cartesian,
                ).angle_deg
                for index in indices
            ],
            dtype=float,
        )
        grains.append(
            Grain(
                grain_id=grain_id,
                phase_id=phase_id,
                point_indices=indices,
                size=len(indices),
                centroid=np.mean(data.coordinates[indices], axis=0),
                mean_orientation=mean,
                gos_deg=float(np.mean(residuals)),
                maximum_spread_deg=float(np.max(residuals)),
            )
        )
    return tuple(grains)


def kernel_average_misorientation(
    data: EBSDMap,
    phases: Mapping[int, EBSDPhase],
    *,
    neighbor_graph: NeighborGraph | None = None,
    maximum_neighbor_misorientation_deg: float | None = 5.0,
) -> np.ndarray:
    if neighbor_graph is None:
        neighbor_graph = build_neighbor_graph(data)
    values: list[list[float]] = [[] for _ in range(data.n_points)]
    for a, b in neighbor_graph.edges:
        a = int(a)
        b = int(b)
        if not data.indexed[a] or not data.indexed[b]:
            continue
        phase = int(data.phase_id[a])
        if phase != int(data.phase_id[b]) or phase not in phases:
            continue
        angle = minimum_disorientation(
            data.orientations[a],
            data.orientations[b],
            phases[phase].proper_symmetry_cartesian,
        ).angle_deg
        if (
            maximum_neighbor_misorientation_deg is not None
            and angle > maximum_neighbor_misorientation_deg
        ):
            continue
        values[a].append(angle)
        values[b].append(angle)

    kam = np.full(data.n_points, np.nan, dtype=float)
    for index, local in enumerate(values):
        if local:
            kam[index] = float(np.mean(local))
    return kam


@dataclass(frozen=True)
class GrainBoundary:
    point_a: int
    point_b: int
    grain_a: int
    grain_b: int
    phase_a: int
    phase_b: int
    same_phase_disorientation_deg: float | None


def extract_boundaries(
    data: EBSDMap,
    segmentation: GrainSegmentation,
    phases: Mapping[int, EBSDPhase],
) -> tuple[GrainBoundary, ...]:
    output: list[GrainBoundary] = []
    labels = segmentation.grain_id
    for a, b in segmentation.neighbor_graph.edges:
        a = int(a)
        b = int(b)
        ga = int(labels[a])
        gb = int(labels[b])
        if ga < 0 or gb < 0 or ga == gb:
            continue
        pa = int(data.phase_id[a])
        pb = int(data.phase_id[b])
        angle = None
        if pa == pb and pa in phases:
            angle = minimum_disorientation(
                data.orientations[a],
                data.orientations[b],
                phases[pa].proper_symmetry_cartesian,
            ).angle_deg
        output.append(
            GrainBoundary(
                point_a=a,
                point_b=b,
                grain_a=ga,
                grain_b=gb,
                phase_a=pa,
                phase_b=pb,
                same_phase_disorientation_deg=angle,
            )
        )
    return tuple(output)
