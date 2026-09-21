from __future__ import annotations

"""Experimental boundary-trace validation for EBSD.

The trace of a crystallographic plane on an observed specimen surface is the
intersection of two planes.  If ``n_plane`` and ``n_surface`` are sample-frame
unit normals,

    t_predicted ∝ n_surface × n_plane.

The measured trace is estimated from the *midpoints of EBSD neighbour edges*
belonging to one grain-pair boundary and a PCA in the physical specimen
surface.  The code reports a line-shape/linearity diagnostic and refuses to
pretend that a strongly curved or under-resolved boundary is one exact line.

Twin-plane hypotheses can be supplied directly or generated from a CTTwin.
For a CT intercorrespondence C_int mapping direct coordinates from side 1 to
side 2, reciprocal plane coordinates obey

    p_2 ∝ C_int^{-T} p_1.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .ebsd import (
    predicted_plane_trace,
    trace_residual_deg,
    undirected_angle_deg,
)
from .ebsd_analysis import GrainSegmentation
from .ebsd_map import EBSDMap, EBSDPhase
from .lattice import plane_normal_cartesian
from .twinning_ct import CTTwin


def _unit(vector: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-15:
        raise ValueError(f"{name} must be nonzero")
    return value / norm


def _canonical_projective_direction(vector: np.ndarray) -> np.ndarray:
    value = _unit(vector, name="direction")
    nonzero = np.flatnonzero(np.abs(value) > 1.0e-12)
    if len(nonzero) and value[int(nonzero[0])] < 0.0:
        value = -value
    return value


def _surface_basis(surface_normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = _unit(surface_normal, name="surface normal")
    axes = np.eye(3)
    reference = axes[int(np.argmin(np.abs(axes @ normal)))]
    e1 = _unit(np.cross(normal, reference), name="surface tangent axis 1")
    e2 = _unit(np.cross(normal, e1), name="surface tangent axis 2")
    return e1, e2


@dataclass(frozen=True)
class BoundaryTraceEstimate:
    grain_a: int
    grain_b: int
    tangent_sample: np.ndarray
    centroid_sample: np.ndarray
    n_boundary_edges: int
    n_unique_midpoints: int
    linearity: float
    projected_span: float
    surface_normal_sample: np.ndarray


def estimate_boundary_trace(
    data: EBSDMap,
    segmentation: GrainSegmentation,
    grain_a: int,
    grain_b: int,
    *,
    surface_normal_sample: np.ndarray,
) -> BoundaryTraceEstimate:
    """Estimate one grain-pair boundary trace from neighbour-edge midpoints."""

    if grain_a == grain_b:
        raise ValueError("boundary requires two distinct grains")
    labels = segmentation.grain_id
    midpoint_list: list[np.ndarray] = []

    low, high = sorted((int(grain_a), int(grain_b)))
    for point_a, point_b in segmentation.neighbor_graph.edges:
        a = int(point_a)
        b = int(point_b)
        pair = sorted((int(labels[a]), int(labels[b])))
        if pair == [low, high]:
            midpoint_list.append(
                0.5 * (data.coordinates[a] + data.coordinates[b])
            )

    if len(midpoint_list) < 2:
        raise ValueError(
            "at least two boundary neighbour edges are required to estimate a trace"
        )

    midpoints = np.asarray(midpoint_list, dtype=float)
    # Remove repeated midpoints before PCA.
    rounded = np.round(midpoints, decimals=12)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    unique = midpoints[np.sort(unique_indices)]
    if len(unique) < 2:
        raise ValueError("boundary trace is under-resolved: fewer than two unique points")

    normal = _unit(surface_normal_sample, name="surface normal")
    e1, e2 = _surface_basis(normal)
    centroid = np.mean(unique, axis=0)
    centered = unique - centroid
    projected = np.column_stack((centered @ e1, centered @ e2))

    covariance = projected.T @ projected / max(len(projected), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    principal = eigenvectors[:, order[0]]

    if float(eigenvalues[0]) <= 1.0e-24:
        raise ValueError("boundary midpoints have no measurable spatial span")

    tangent = _canonical_projective_direction(
        principal[0] * e1 + principal[1] * e2
    )
    total_variance = float(np.sum(eigenvalues))
    linearity = (
        float(eigenvalues[0] / total_variance)
        if total_variance > 0.0
        else 0.0
    )
    coordinate = centered @ tangent
    span = float(np.max(coordinate) - np.min(coordinate))

    return BoundaryTraceEstimate(
        grain_a=low,
        grain_b=high,
        tangent_sample=tangent,
        centroid_sample=centroid,
        n_boundary_edges=len(midpoint_list),
        n_unique_midpoints=len(unique),
        linearity=linearity,
        projected_span=span,
        surface_normal_sample=normal,
    )


@dataclass(frozen=True)
class TwinPlaneHypothesis:
    label: str
    plane_side1_crystal: np.ndarray
    plane_side2_crystal: np.ndarray
    operator_index: int | None = None
    source: str = ""


def twin_plane_hypothesis_from_ct(
    twin: CTTwin,
    *,
    operator_index: int | None = None,
    label: str | None = None,
) -> TwinPlaneHypothesis:
    p1 = np.asarray(twin.plane_m, dtype=float).reshape(3)
    Cint = np.asarray(twin.intercorrespondence, dtype=float).reshape(3, 3)
    if abs(float(np.linalg.det(Cint))) <= 1.0e-14:
        raise ValueError("CT intercorrespondence is singular")
    p2 = np.linalg.solve(Cint.T, p1)
    return TwinPlaneHypothesis(
        label=(
            label
            if label is not None
            else f"{twin.classification}:{twin.rational_element}"
        ),
        plane_side1_crystal=p1,
        plane_side2_crystal=p2,
        operator_index=operator_index,
        source="Cayron CT intercorrespondence",
    )


@dataclass(frozen=True)
class TraceHypothesisResult:
    hypothesis: TwinPlaneHypothesis
    side1_trace_residual_deg: float
    side2_trace_residual_deg: float
    plane_normal_coherence_deg: float
    maximum_trace_residual_deg: float
    score_deg: float


def evaluate_trace_hypothesis(
    estimate: BoundaryTraceEstimate,
    orientation_side1: np.ndarray,
    orientation_side2: np.ndarray,
    product_phase: EBSDPhase,
    hypothesis: TwinPlaneHypothesis,
) -> TraceHypothesisResult:
    """Compare one measured line with the predicted traces from both crystals."""

    surface = estimate.surface_normal_sample
    measured = estimate.tangent_sample

    trace1 = predicted_plane_trace(
        hypothesis.plane_side1_crystal,
        orientation_side1,
        product_phase.lattice,
        surface,
    )
    trace2 = predicted_plane_trace(
        hypothesis.plane_side2_crystal,
        orientation_side2,
        product_phase.lattice,
        surface,
    )
    residual1 = trace_residual_deg(trace1, measured)
    residual2 = trace_residual_deg(trace2, measured)

    n1_crystal = plane_normal_cartesian(
        hypothesis.plane_side1_crystal,
        product_phase.lattice,
    )
    n2_crystal = plane_normal_cartesian(
        hypothesis.plane_side2_crystal,
        product_phase.lattice,
    )
    n1_sample = np.asarray(orientation_side1, dtype=float) @ n1_crystal
    n2_sample = np.asarray(orientation_side2, dtype=float) @ n2_crystal
    coherence = undirected_angle_deg(n1_sample, n2_sample)

    # Max rather than mean: a twin-plane hypothesis is not accepted merely
    # because one crystal side fits a measured trace while the other does not.
    maximum = max(residual1, residual2)
    score = float(np.hypot(maximum, coherence))
    return TraceHypothesisResult(
        hypothesis=hypothesis,
        side1_trace_residual_deg=float(residual1),
        side2_trace_residual_deg=float(residual2),
        plane_normal_coherence_deg=float(coherence),
        maximum_trace_residual_deg=float(maximum),
        score_deg=score,
    )


def rank_trace_hypotheses(
    estimate: BoundaryTraceEstimate,
    orientation_side1: np.ndarray,
    orientation_side2: np.ndarray,
    product_phase: EBSDPhase,
    hypotheses: Sequence[TwinPlaneHypothesis],
    *,
    minimum_linearity: float = 0.90,
) -> tuple[TraceHypothesisResult, ...]:
    if not hypotheses:
        raise ValueError("at least one twin-plane hypothesis is required")
    if not (0.0 <= minimum_linearity <= 1.0):
        raise ValueError("minimum_linearity must lie in [0,1]")
    if estimate.linearity < minimum_linearity:
        raise ValueError(
            "boundary is not sufficiently line-like for a single trace "
            f"comparison: linearity={estimate.linearity:.6g}"
        )
    results = [
        evaluate_trace_hypothesis(
            estimate,
            orientation_side1,
            orientation_side2,
            product_phase,
            hypothesis,
        )
        for hypothesis in hypotheses
    ]
    results.sort(
        key=lambda item: (
            item.score_deg,
            item.maximum_trace_residual_deg,
            item.plane_normal_coherence_deg,
            item.hypothesis.label,
        )
    )
    return tuple(results)
