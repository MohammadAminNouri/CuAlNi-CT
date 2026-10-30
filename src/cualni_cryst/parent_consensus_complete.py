from __future__ import annotations

"""Bounded, symmetry-complete exploratory beta-parent consensus from daughter EBSD.

The prior single-OR parent solver voted with only user-entered OR matrices.
For cubic beta -> monoclinic 18R, a single *reference* OR generates an entire
proper parent-symmetry variant family, modulo proper daughter symmetry.

This solver tests g_parent = g_daughter S_daughter (S_parent R_reference)^T,
uses quaternion consensus, and validates held-out measurements separately.
Nothing here certifies the physical indexed 18R OR or absolute specimen frame.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .orientation_kernel import require_so3
from .parent_variant_audit import (
    parent_generated_variants, daughter_variant_assignment,
)


def reconstruct_parent_full_variant_consensus(
    daughter_orientations: np.ndarray,
    reference_parent_from_daughter: np.ndarray,
    parent_symmetry: tuple[np.ndarray, ...],
    daughter_symmetry: tuple[np.ndarray, ...],
    *,
    weights: np.ndarray | None = None,
    tolerance_deg: float = 3.0,
    max_seeds: int = 512,
    max_iterations: int = 12,
) -> dict:
    """Bounded SO(3) consensus using ALL parent-derived daughter variants.

    Input rotations are crystal->sample and parent<-daughter Cartesian OR.
    The result is a *conditional* single-parent hypothesis for supplied points.
    Do not infer physical OR, indexer correctness, or spatial parent boundaries.
    """
    measured = np.asarray(daughter_orientations, float)
    if measured.ndim != 3 or measured.shape[1:] != (3, 3) or not 2 <= len(measured) <= 256:
        raise ValueError("Provide 2..256 indexed daughter orientation matrices (N,3,3)")
    if not 0 < tolerance_deg <= 15 or not 1 <= max_iterations <= 50 or not 1 <= max_seeds <= 2048:
        raise ValueError("Invalid parent-consensus tolerances or work bounds")
    for i, g in enumerate(measured):
        require_so3(g, tolerance=2e-8, name=f"indexed daughter orientation {i}")
    p_ops = tuple(require_so3(x, tolerance=2e-8, name="parent symmetry") for x in parent_symmetry)
    m_ops = tuple(require_so3(x, tolerance=2e-8, name="daughter symmetry") for x in daughter_symmetry)
    if not p_ops or not m_ops:
        raise ValueError("Proper parent and daughter symmetry groups are required")
    R = require_so3(reference_parent_from_daughter, tolerance=2e-8, name="reference OR")
    variants = np.asarray(parent_generated_variants(R, p_ops, m_ops))
    if weights is None:
        w = np.full(len(measured), 1.0 / len(measured))
    else:
        w = np.asarray(weights, float).reshape(-1)
        if len(w) != len(measured) or np.any(~np.isfinite(w)) or np.any(w<0) or w.sum()<=0:
            raise ValueError("Weights must be finite nonnegative with positive sum")
        w /= w.sum()

    # g_parent = g_daughter @ proper_daughter_symmetry @ (parent_variant_OR)^T
    gM = np.einsum("nij,sjk->nsik", measured, np.asarray(m_ops), optimize=True)
    family = np.einsum("nsij,vjk->nsvik", gM, variants.transpose(0, 2, 1), optimize=True)
    q = Rotation.from_matrix(family.reshape(-1, 3, 3)).as_quat()
    n = len(measured)
    family_q = q.reshape(n, -1, 4)
    flat = family_q.reshape(-1, 4)
    # Include all symmetry representatives for multiple geographically
    # separated sample points, then deterministic evenly spaced seeds.
    first = np.arange(min(3 * family_q.shape[1], len(flat)))
    sampled = np.linspace(0, len(flat)-1, max_seeds, dtype=int)
    seeds = np.unique(np.r_[first, sampled])[:max_seeds]
    best = None
    best_q = None
    for idx in seeds:
        seed = flat[idx]
        cos = np.max(np.abs(np.einsum("nki,i->nk", family_q, seed, optimize=True)),axis=1)
        angles = np.rad2deg(2*np.arccos(np.clip(cos, 0, 1)))
        inside = angles <= tolerance_deg
        key = (-float(w[inside].sum()), float(np.dot(w[inside], angles[inside])), int(idx))
        if best is None or key < best:
            best, best_q = key, seed
    assert best_q is not None
    seed = best_q.copy()
    iters = 0
    for step in range(max_iterations):
        iters = step + 1
        dots = np.einsum("nki,i->nk",family_q,seed,optimize=True)
        ix = np.argmax(np.abs(dots),axis=1)
        align_dot = dots[np.arange(n),ix]
        angles = np.rad2deg(2*np.arccos(np.clip(np.abs(align_dot), 0, 1)))
        take = angles <= tolerance_deg
        if not take.any():
            raise ValueError("All candidate parent votes rejected during refinement")
        chosen = family_q[np.arange(n),ix] * np.where(align_dot[:,None]>=0,1,-1)
        updated = np.sum(w[take,None]*chosen[take],axis=0)
        updated /= np.linalg.norm(updated)
        delta = np.rad2deg(2*np.arccos(np.clip(abs(np.dot(updated,seed)),0,1)))
        seed = updated
        if delta < 1e-7:
            break
    parent = Rotation.from_quat(seed).as_matrix()
    labels = daughter_variant_assignment(
        measured,parent,R,p_ops,m_ops,
        acceptance_deg=tolerance_deg,
    )
    residuals = np.asarray(labels["residual_deg"],float)
    inliers = residuals <= tolerance_deg
    return {
        "parent_crystal_to_sample_rotation": parent.tolist(),
        "parent_quaternion_xyzw": seed.tolist(),
        "n_sampled": n,
        "n_parent_generated_variants": len(variants),
        "proper_daughter_symmetry_count": len(m_ops),
        "proper_parent_symmetry_count": len(p_ops),
        "weighted_inlier_fraction": float(w[inliers].sum()),
        "mean_inlier_residual_deg": (float(np.dot(w[inliers],residuals[inliers])/w[inliers].sum())
                                        if inliers.any() else None),
        "maximum_inlier_residual_deg": float(np.max(residuals[inliers])) if inliers.any() else None,
        "iterations": iters,
        "assigned_variant_labels": labels["variant_index_1based"],
        "full_parent_generated_variant_assignment": labels,
        "status": "SYMMETRY-COMPLETE NUMERICAL PARENT CANDIDATE; NOT an experimentally verified physical beta parent",
        "experimental_OR_certified": False,
        "absolute_parent_frame_certified": False,
    }
