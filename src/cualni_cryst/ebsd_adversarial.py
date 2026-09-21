from __future__ import annotations

"""Deterministic adversarial transforms for EBSD validation.

These functions exist to test failure semantics.  They are not data-cleaning
tools and never run automatically on experimental data.
"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.spatial.transform import Rotation

from .ebsd_map import EBSDMap
from .orientation_kernel import require_so3


def _stack(orientations: np.ndarray) -> np.ndarray:
    values = np.asarray(orientations, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise ValueError("orientations must have shape (N,3,3)")
    return values


def apply_global_sample_rotation(
    orientations: np.ndarray,
    rotation_sample: np.ndarray,
) -> np.ndarray:
    """Left-multiply every orientation: crystallographically unidentifiable internally."""

    values = _stack(orientations)
    Q = require_so3(rotation_sample, tolerance=2.0e-8, name="sample rotation")
    return np.einsum("ab,nbc->nac", Q, values)


def apply_crystal_frame_offset(
    orientations: np.ndarray,
    rotation_crystal: np.ndarray,
) -> np.ndarray:
    """Right-multiply every orientation by an explicit crystal-frame offset."""

    values = _stack(orientations)
    Q = require_so3(rotation_crystal, tolerance=2.0e-8, name="crystal offset")
    return np.einsum("nab,bc->nac", values, Q)


def transpose_orientations(orientations: np.ndarray) -> np.ndarray:
    """Deliberately swap crystal->sample with sample->crystal."""

    values = _stack(orientations)
    return np.transpose(values, (0, 2, 1)).copy()


def add_isotropic_orientation_noise(
    orientations: np.ndarray,
    sigma_deg: float,
    *,
    seed: int,
) -> np.ndarray:
    """Left-multiply independent isotropic Gaussian small rotations."""

    if sigma_deg < 0.0:
        raise ValueError("sigma_deg must be nonnegative")
    values = _stack(orientations)
    rng = np.random.default_rng(seed)
    vectors = rng.normal(
        scale=np.deg2rad(sigma_deg),
        size=(len(values), 3),
    )
    noise = Rotation.from_rotvec(vectors).as_matrix()
    return np.einsum("nab,nbc->nac", noise, values)


def inject_random_orientation_outliers(
    orientations: np.ndarray,
    fraction: float,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 <= fraction < 1.0):
        raise ValueError("fraction must lie in [0,1)")
    values = _stack(orientations).copy()
    count = int(round(fraction * len(values)))
    mask = np.zeros(len(values), dtype=bool)
    if count == 0:
        return values, mask
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(values), size=count, replace=False))
    mask[indices] = True
    values[indices] = Rotation.random(
        count,
        random_state=rng,
    ).as_matrix()
    return values, mask


def randomly_unindex_map(
    data: EBSDMap,
    fraction: float,
    *,
    seed: int,
) -> EBSDMap:
    if not (0.0 <= fraction < 1.0):
        raise ValueError("fraction must lie in [0,1)")
    indexed_indices = np.flatnonzero(data.indexed)
    count = int(round(fraction * len(indexed_indices)))
    if count == 0:
        return data

    rng = np.random.default_rng(seed)
    selected = np.sort(
        rng.choice(indexed_indices, size=count, replace=False)
    )
    indexed = data.indexed.copy()
    phase = data.phase_id.copy()
    orientations = data.orientations.copy()
    indexed[selected] = False
    phase[selected] = 0
    orientations[selected] = np.nan
    metadata = dict(data.metadata)
    metadata["adversarial_random_unindex_fraction"] = float(fraction)

    return EBSDMap(
        orientations=orientations,
        phase_id=phase,
        indexed=indexed,
        x=data.x.copy(),
        y=data.y.copy(),
        z=data.z.copy(),
        quality={key: value.copy() for key, value in data.quality.items()},
        metadata=MappingProxyType(metadata),
    )


@dataclass(frozen=True)
class AdversarialInvariant:
    label: str
    expected: str
    reason: str


ADVERSARIAL_INVARIANTS = (
    AdversarialInvariant(
        "global_sample_rotation",
        "internal child/child theory score invariant",
        "all relative crystal misorientations are unchanged by common left multiplication",
    ),
    AdversarialInvariant(
        "crystal_frame_offset",
        "generally detectable against a fixed theoretical OR unless symmetry-equivalent",
        "right multiplication changes the crystal-frame representation relative to theory",
    ),
    AdversarialInvariant(
        "transpose_active_passive",
        "generally detectable",
        "matrix inversion changes the transformation geometry except in special symmetric cases",
    ),
    AdversarialInvariant(
        "missing_points",
        "should reduce support, not invent orientations",
        "unindexed points are excluded rather than filled by default",
    ),
    AdversarialInvariant(
        "random_orientation_outliers",
        "robust reconstruction should reject them as residual outliers",
        "consensus/inlier logic is explicit",
    ),
)
