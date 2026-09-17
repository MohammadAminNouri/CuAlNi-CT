from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def rotation_angle_deg(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=float)
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def minimum_misorientation_deg(
    g1: np.ndarray, g2: np.ndarray, product_symmetry_cart: list[np.ndarray]
) -> float:
    """Minimum crystallographic misorientation.

    Convention: g maps crystal Cartesian coordinates -> sample Cartesian coordinates.
    A symmetry-equivalent orientation is g @ S.
    """
    best = 180.0
    for S1 in product_symmetry_cart:
        A = g1 @ S1
        for S2 in product_symmetry_cart:
            B = g2 @ S2
            D = B.T @ A
            best = min(best, rotation_angle_deg(D))
    return float(best)


@dataclass(frozen=True)
class VariantAssignment:
    variant_index: int | None
    best_deg: float
    second_best_deg: float
    ambiguous: bool


def assign_variant(
    measured_g: np.ndarray,
    predicted_g: list[np.ndarray],
    product_symmetry_cart: list[np.ndarray],
    max_angle_deg: float = 5.0,
    ambiguity_gap_deg: float = 0.5,
) -> VariantAssignment:
    errors = [minimum_misorientation_deg(measured_g, p, product_symmetry_cart) for p in predicted_g]
    order = np.argsort(errors)
    best_i = int(order[0])
    best = float(errors[best_i])
    second = float(errors[int(order[1])]) if len(order) > 1 else 180.0
    if best > max_angle_deg:
        return VariantAssignment(None, best, second, False)
    return VariantAssignment(best_i, best, second, (second - best) < ambiguity_gap_deg)


def plane_trace_direction(plane_normal_sample: np.ndarray, surface_normal_sample: np.ndarray) -> np.ndarray:
    t = np.cross(surface_normal_sample, plane_normal_sample)
    n = np.linalg.norm(t)
    if n < 1e-14:
        raise ValueError("Plane is parallel to sample surface; trace direction is undefined.")
    return t / n


def undirected_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    a = np.asarray(v1, dtype=float); b = np.asarray(v2, dtype=float)
    a /= np.linalg.norm(a); b /= np.linalg.norm(b)
    c = abs(float(a @ b))  # trace is a line, so +/- are equivalent
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
