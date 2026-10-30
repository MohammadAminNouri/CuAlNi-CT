from __future__ import annotations

"""Physically consistent parent-derived variant sets and fast EBSD assignments.

Parent crystal symmetry generates martensite OR variants:
    R_(A<-M, i) = S_parent,i @ R_(A<-M, reference)
and right multiplication by proper martensite symmetry yields equivalent
descriptions of the SAME variant. These distinctions matter: one reference OR
alone is NOT a complete list of daughter variants.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from .orientation_kernel import require_so3


def parent_generated_variants(
    reference_parent_from_product: np.ndarray,
    parent_symmetry: tuple[np.ndarray, ...],
    product_symmetry: tuple[np.ndarray, ...],
    *, uniqueness_deg: float = 1.0e-5,
) -> tuple[np.ndarray, ...]:
    """Generate inequivalent variants modulo proper daughter symmetry."""
    R = require_so3(reference_parent_from_product, tolerance=2e-8,
                    name="reference parent<-martensite OR")
    if not parent_symmetry or not product_symmetry:
        raise ValueError("Provide complete proper parent and product point groups")
    p = [require_so3(s, tolerance=2e-8, name="parent symmetry") for s in parent_symmetry]
    m = [require_so3(s, tolerance=2e-8, name="product symmetry") for s in product_symmetry]
    unique: list[np.ndarray] = []
    for S in p:
        trial = S @ R
        if any(
            min(Rotation.from_matrix(trial @ M @ known.T).magnitude()
                for M in m) <= np.deg2rad(uniqueness_deg)
            for known in unique
        ):
            continue
        unique.append(trial)
    return tuple(unique)


def daughter_variant_assignment(
    measured_product_orientations: np.ndarray,
    parent_crystal_to_sample: np.ndarray,
    reference_parent_from_product: np.ndarray,
    parent_symmetry: tuple[np.ndarray, ...],
    product_symmetry: tuple[np.ndarray, ...],
    *, acceptance_deg: float = 3.0, minimum_margin_deg: float = 0.5,
) -> dict:
    """Return best-variant residuals/margins without conflating symmetry copies."""
    g = require_so3(parent_crystal_to_sample, tolerance=2e-8,
                    name="candidate parent orientation")
    phases = parent_generated_variants(
        reference_parent_from_product, parent_symmetry, product_symmetry
    )
    array = np.asarray(measured_product_orientations, float)
    if array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ValueError("measured product orientations must be (N,3,3)")
    if not 0 < len(array) <= 100000:
        raise ValueError("Provide 1-100000 measured product orientations")
    variants = np.asarray(phases)
    product_ops = np.asarray(product_symmetry)
    predicted = np.einsum("ij,vjk->vik", g, variants)
    all_reps = np.einsum("vij,sjk->vsik", predicted, product_ops)
    pred_q = Rotation.from_matrix(all_reps.reshape(-1, 3, 3)).as_quat().reshape(
        len(variants), len(product_ops), 4
    )
    observed = Rotation.from_matrix(array).as_quat()
    # q and -q represent the same SO3 rotation; maximize dot over each
    # proper martensite symmetry copy, not over a guessed convention.
    cos_half = np.max(np.abs(
        np.einsum("ni,vsi->nvs", observed, pred_q, optimize=True)
    ), axis=2)
    residual = np.rad2deg(2 * np.arccos(np.clip(cos_half, 0, 1)))
    indices = np.argmin(residual, axis=1)
    minima = residual[np.arange(len(array)), indices]
    if len(variants) > 1:
        second = np.partition(residual, 1, axis=1)[:, 1]
    else:
        second = np.full(len(array), 180.0)
    accepted = minima <= acceptance_deg
    return {
        "parent_generated_variant_count": len(variants),
        "variant_index_1based": np.where(accepted, indices + 1, 0).astype(int).tolist(),
        "residual_deg": minima.astype(float).tolist(),
        "runner_up_margin_deg": (second - minima).astype(float).tolist(),
        "accepted": accepted.tolist(),
        "ambiguous": (accepted & (second - minima < minimum_margin_deg)).tolist(),
        "accepted_fraction": float(np.mean(accepted)),
        "median_residual_deg": float(np.median(minima)),
        "ambiguous_fraction_among_accepted": float(
            np.mean((second - minima)[accepted] < minimum_margin_deg)
        ) if np.any(accepted) else None,
        "status": "conditioned on supplied OR and parent; not independent OR validation",
    }
