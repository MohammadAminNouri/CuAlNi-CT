from __future__ import annotations

from itertools import permutations
import numpy as np


def projective_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).reshape(3)
    b = np.asarray(b, float).reshape(3)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(min(np.linalg.norm(a-b), np.linalg.norm(a+b)))


def projective_family_distance(lhs: list[np.ndarray], rhs: list[np.ndarray]) -> float:
    if len(lhs) != len(rhs):
        return float("inf")
    if not lhs:
        return 0.0
    best = float("inf")
    for perm in permutations(range(len(rhs))):
        best = min(best, max(projective_distance(lhs[i], rhs[perm[i]]) for i in range(len(lhs))))
    return best


def relative_matrix_residual(A: np.ndarray, B: np.ndarray) -> float:
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    return float(np.linalg.norm(A-B) / max(np.linalg.norm(A), np.linalg.norm(B), np.finfo(float).tiny))
