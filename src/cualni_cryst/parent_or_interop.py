from __future__ import annotations

"""Cross-tool beta<-18R OR contract and MTEX-style *boundary weight* audit.

The orientation convention is crystal Cartesian -> sample Cartesian. A user
fitted Q maps daughter crystal Cartesian -> parent crystal Cartesian, so MTEX
parent-to-child p2c is Q.T. This module neither claims the physical OR is known
nor reproduces MTEX's Markov variant-graph clustering.
"""

from itertools import permutations, product
from typing import Sequence

import numpy as np
from scipy.special import erfc
from scipy.spatial.transform import Rotation

from .orientation_kernel import require_so3


def proper_cubic_parent_operations() -> np.ndarray:
    """Exactly 24 proper signed permutation matrices, 432 point group."""
    ops: list[np.ndarray] = []
    for perm in permutations(range(3)):
        for sign in product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3), dtype=float)
            matrix[np.arange(3), perm] = sign
            if np.linalg.det(matrix) > 0:
                ops.append(matrix)
    if len(ops) != 24:
        raise AssertionError("Incorrect proper cubic point group")
    return np.asarray(ops)


def mtex_or_contract(parent_from_daughter: object) -> dict:
    """Export complete unambiguous OR direction, never assign a literature name."""
    Q = require_so3(np.asarray(parent_from_daughter, float),
                    tolerance=2e-8, name="parent<-daughter OR")
    return {
        "Q_parent_from_daughter": Q.tolist(),
        "mtex_p2c_daughter_from_parent": Q.T.tolist(),
        "quaternion_xyzw_parent_from_daughter": Rotation.from_matrix(Q).as_quat().tolist(),
        "rotation_direction": "daughter-crystal to parent-crystal, orthonormal Cartesian",
        "orientation_convention": "all g are crystal->sample; parent g = daughter g @ S_child @ Q.T @ S_parent.T",
        "scientific_status": "input OR only; physical parent not certified",
    }


def frozen_or_boundary_audit(
    daughter_orientation_pairs: np.ndarray,
    parent_from_daughter: np.ndarray,
    parent_symmetry: Sequence[np.ndarray],
    daughter_symmetry: Sequence[np.ndarray],
    *,
    threshold_deg: float = 2.5,
    transition_width_deg: float = 1.5,
) -> dict:
    """Evaluate measured daughter/daughter operators against Q.T S_parent Q.

    For each parent operation, compare all proper left/right daughter
    equivalents, and report MTEX's source-defined edge probability.

    MTEX calcVariantGraph.m uses 0.5*erfc(2*(misfit-threshold)/tolerance),
    which is NOT scipy.ndtr((threshold-misfit)/tolerance). The input
    tolerance is a transition-width parameter, not a Gaussian standard
    deviation. This calculates a boundary-only diagnostic; MTEX also uses
    variant-node weights and neighbour statistics.
    This is NOT calcVariantGraph, which has variant nodes and MCL clustering.

    Parameters
    ----------
    daughter_orientation_pairs : float, (N,2,3,3)
      Indexed daughter crystal-to-sample proper rotations.
    Q : (3,3)
      parent<-daughter proper Cartesian rotation; inverse is MTEX p2c.
    """
    ori = np.asarray(daughter_orientation_pairs, float)
    if ori.ndim != 4 or ori.shape[1:] != (2,3,3) or not 1 <= len(ori) <= 20000:
        raise ValueError("Expected 1..20000 daughter orientation pairs (N,2,3,3)")
    if (not np.isfinite(threshold_deg) or not np.isfinite(transition_width_deg)
            or not 0 < threshold_deg <= 30 or not 0 < transition_width_deg <= 20):
        raise ValueError("Invalid graph probability threshold/tolerance")
    for idx, matrix in enumerate(ori.reshape(-1,3,3)):
        require_so3(matrix, tolerance=2e-8, name=f"indexed daughter orientation {idx}")
    Q = require_so3(np.asarray(parent_from_daughter,float),
                    tolerance=2e-8, name="parent<-daughter OR")
    p = np.asarray(tuple(require_so3(np.asarray(x,float),tolerance=2e-8,
                      name="parent symmetry") for x in parent_symmetry))
    m = np.asarray(tuple(require_so3(np.asarray(x,float),tolerance=2e-8,
                      name="daughter symmetry") for x in daughter_symmetry))
    if not len(p) or not len(m):
        raise ValueError("Proper phase symmetry groups are required")
    relative = np.einsum("nij,nik->njk",ori[:,0],ori[:,1])
    representatives = np.einsum(
        "aij,njk,bkl->nabil",m.transpose(0,2,1),relative,m,optimize=True
    ).reshape(-1,3,3)
    observed = Rotation.from_matrix(representatives).as_quat().reshape(
        len(ori),-1,4
    )
    theoretical = Rotation.from_matrix(Q.T@p@Q).as_quat()
    dots = np.max(np.abs(np.einsum("nsi,pi->nsp",observed,theoretical,optimize=True)),
                  axis=(1,2))
    residual = np.rad2deg(2*np.arccos(np.clip(dots,0,1)))
    weights = 0.5 * erfc(2.0 * (residual - threshold_deg) / transition_width_deg)
    return {
        "n_boundary_pairs":len(ori),
        "median_operator_misfit_deg":float(np.median(residual)),
        "p90_operator_misfit_deg":float(np.percentile(residual,90)),
        "fraction_under_3deg":float(np.mean(residual<3)),
        "operator_misfit_deg":residual.tolist(),
        "mtex_style_probability":weights.tolist(),
        "mtex_style_threshold_deg":float(threshold_deg),
        "mtex_style_transition_width_deg":float(transition_width_deg),
        "mtex_probability_formula": "0.5*erfc(2*(misfit_deg-threshold_deg)/tolerance_deg)",
        "mtex_probability_scope": "unweighted pair-only probability; not MTEX graph or boundary-length weighting",
        "parent_symmetry_operations":len(p),
        "daughter_symmetry_operations":len(m),
        "scientific_status":"MTEX-inspired edge-probability diagnostic, not MTEX MCL variant clustering or physically validated OR",
    }
