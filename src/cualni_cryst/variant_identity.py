from __future__ import annotations

"""Symmetry-orbit variant identity with explicit representation-error handling.

Scientific contract
-------------------
Two supplied binary64 pullback metrics can differ in the last bits even when
both represent the same intended crystallographic symmetry orbit. Example:
90-degree lattice angles are evaluated through trig functions; their computed
cosines are not mathematically zero. Treating EVERY such difference as an
exact phase distinction (v1) incorrectly rejects real benchmarks.

The input pullback keys are evaluated as exact binary64 rationals. Comparisons
are *not* exact crystallographic certificates: a key difference smaller than a
conservative floating-input roundoff envelope is classified as a numerical
representation tie, provided the computed stretches ALSO coincide within a
separate, relative backend roundoff envelope. Resolvable differences are
preserved regardless of the loose algebraic or eigenvalue acceptance policy.
Ambiguous disagreement between these two diagnostics raises explicitly.

This is a computational identity convention, NOT a measurement uncertainty
model and NOT evidence that rounded experimental phase parameters have exact
physical symmetry.
"""

from collections.abc import Iterable
from typing import Hashable
import math

import numpy as np
import sympy as sp


def _rational_matrix(value: object) -> sp.Matrix:
    x = np.asarray(value, dtype=float)
    if x.shape != (3, 3) or not np.isfinite(x).all():
        raise ValueError("Orbit identity requires finite 3x3 matrices")
    return sp.Matrix(3, 3, lambda i, j: sp.Rational(*float(x[i, j]).as_integer_ratio()))


def pullback_orbit_keys(
    product_metric: object,
    correspondence: object,
    indexed_parent_symmetries: Iterable[tuple[int, object]],
) -> dict[int, tuple[sp.Rational, ...]]:
    """Binary64-exact pullbacks T_g=g^{-T} C^T G_M C g^{-1}."""
    m = _rational_matrix(product_metric)
    c = _rational_matrix(correspondence)
    h = c.T * m * c
    keys: dict[int, tuple[sp.Rational, ...]] = {}
    for index, operation in indexed_parent_symmetries:
        g = _rational_matrix(operation)
        if g.det() == 0:
            raise ValueError(f"Singular parent symmetry {index}")
        inv = g.inv()
        keys[int(index)] = tuple(inv.T * h * inv)
    return keys


def _key_separation(
    left: Hashable | None, right: Hashable | None,
) -> tuple[float, float] | None:
    """Absolute metric-key discrepancy and metric magnitude, before float rounding.

    Computing `left_i-right_i` AS EXACT RATIONALS before converting avoids
    spuriously zero differences from subtracting individually rounded floats.
    """
    if left is None or right is None:
        return None
    try:
        aa = tuple(left)
        bb = tuple(right)
        if len(aa) != 9 or len(bb) != 9:
            raise ValueError("Metric pullback signatures must have 9 entries")
        differences = [float(sp.Rational(x) - sp.Rational(y)) for x, y in zip(aa, bb)]
        a_values = [float(sp.Rational(x)) for x in aa]
        b_values = [float(sp.Rational(y)) for y in bb]
        discrepancy = math.hypot(*differences)
        magnitude = max(math.hypot(*a_values), math.hypot(*b_values))
        if not math.isfinite(discrepancy) or not math.isfinite(magnitude) or magnitude <= 0:
            raise ValueError("Unresolvable pullback signature magnitude")
        return discrepancy, magnitude
    except (TypeError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError("Invalid pullback identity metadata") from exc


def match_stretch_orbit(
    candidate: object,
    representatives: list[object],
    *,
    generator_index: int,
    identity_key: Hashable | None,
    existing_keys: list[Hashable | None],
    conditioning: float = 1.0,
) -> int | None:
    """Match established orbit member, otherwise return None or raise.

    - Match: both tensor and metric key agree to justified binary64 resolution.
    - Distinct: tensor separation exceeds the known roundoff envelope.
    - Unresolved: signature and tensor diagnostics contradict each other.

    Strict equality of keys is sufficient but NOT required, because a lattice
    metric formed from trig(90 degrees) is not exactly diagonal in binary64.
    """
    current = np.asarray(candidate, dtype=float)
    if current.shape != (3, 3) or not np.isfinite(current).all():
        raise ValueError("Invalid stretch candidate")
    if len(existing_keys) != len(representatives):
        raise ValueError("Missing orbit identity metadata")
    if not np.isfinite(conditioning) or conditioning < 1.0:
        raise ValueError("Invalid metric conditioning")
    if conditioning > 1.0e8:
        raise ValueError("UNRESOLVED_VARIANT_IDENTITY: badly conditioned input metric")

    machine = float(np.finfo(float).eps)
    for idx, obj in enumerate(representatives):
        previous = np.asarray(obj, dtype=float)
        if previous.shape != (3, 3) or not np.isfinite(previous).all():
            raise ValueError("Invalid stored stretch representative")
        distance = float(np.linalg.norm(current - previous, ord="fro"))
        tensor_magnitude = max(1.0, float(np.linalg.norm(current, "fro")),
                               float(np.linalg.norm(previous, "fro")))
        # Split the narrow "safe to merge" representation band from the
        # much wider "cannot confidently resolve" envelope. In particular,
        # do not silently merge a real ~1e-14 perturbation merely because it
        # lies below a generous 256-ulp conservative error estimate.
        merge_bound = 16.0 * machine * tensor_magnitude * conditioning
        uncertainty_bound = 256.0 * machine * tensor_magnitude * conditioning
        if not math.isfinite(distance) or not math.isfinite(uncertainty_bound):
            raise ValueError("UNRESOLVED_VARIANT_IDENTITY: nonfinite comparison")

        key_gap = _key_separation(identity_key, existing_keys[idx])
        if key_gap is not None:
            key_distance, key_magnitude = key_gap
            # Two independent checks: metric-native key and actual stretch.
            # The coefficient is the roundoff budget for finitely many 3x3
            # matrix products/inversions and input cell construction, NOT the
            # caller's algebraic acceptance tolerance.
            key_bound = 32.0 * machine * key_magnitude * conditioning
            same_key_to_precision = key_distance <= key_bound
            if same_key_to_precision:
                if distance <= merge_bound:
                    return idx
                raise ValueError(
                    "UNRESOLVED_VARIANT_IDENTITY: pullbacks overlap input "
                    "roundoff but stretch splitting exceeds safe merge band; "
                    f"generator={generator_index}, tensor_distance={distance:.3e}, "
                    f"merge_bound={merge_bound:.3e}, "
                    f"uncertainty_bound={uncertainty_bound:.3e}"
                )
            if distance <= uncertainty_bound:
                raise ValueError(
                    "UNRESOLVED_VARIANT_IDENTITY: distinct resolvable input "
                    "pullbacks produce indistinguishable float stretches; "
                    f"generator={generator_index}, key_distance={key_distance:.3e}, "
                    f"key_bound={key_bound:.3e}, tensor_distance={distance:.3e}"
                )
            # Both metrics and stretches are decisively distinct.
            continue

        # Backward-compatible direct numerical API without metric provenance:
        # exactly coincident tensors can be merged; otherwise do not claim
        # a near-equality that cannot be independently corroborated.
        if distance == 0.0:
            return idx
        if distance <= uncertainty_bound:
            raise ValueError(
                "UNRESOLVED_VARIANT_IDENTITY: near-identical tensor without "
                f"metric provenance (generator={generator_index}, "
                f"distance={distance:.3e}, bound={uncertainty_bound:.3e})"
            )
    return None
