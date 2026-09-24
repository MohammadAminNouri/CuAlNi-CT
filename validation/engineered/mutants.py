from __future__ import annotations

"""Deliberately wrong formulas.  The validation system must kill every mutant."""

import numpy as np


def pullback_correct(C, Mm):
    return C.T @ Mm @ C


def pullback_wrong_order(C, Mm):
    return C @ Mm @ C.T


def cmc_correct(Ma, Mm, C):
    return pullback_correct(C, Mm) - Ma


def cmc_wrong_plus_parent(Ma, Mm, C):
    return pullback_correct(C, Mm) + Ma


def plane_map_correct(C, p):
    return np.linalg.inv(C).T @ p


def plane_map_wrong_direct(C, p):
    return C @ p


def smc_correct(Ma, Mm, C):
    Ci = np.linalg.inv(C)
    return np.linalg.inv(Ma) - Ci @ np.linalg.inv(Mm) @ Ci.T


def smc_wrong_no_inverse_C(Ma, Mm, C):
    return np.linalg.inv(Ma) - C @ np.linalg.inv(Mm) @ C.T


def basis_C_correct(C, Pa, Pm):
    return np.linalg.inv(Pm) @ C @ Pa


def basis_C_wrong(C, Pa, Pm):
    return Pm @ C @ np.linalg.inv(Pa)


def cofactor_cc3_correct(U, a, n):
    U2 = U @ U
    return np.trace(U2) - np.linalg.det(U2) - 0.25*(a@a)*(n@n) - 2.0


def cofactor_cc3_wrong_plus_det(U, a, n):
    U2 = U @ U
    return np.trace(U2) + np.linalg.det(U2) - 0.25*(a@a)*(n@n) - 2.0


def middle_stretch_correct(F):
    return np.sort(np.linalg.svd(F, compute_uv=False))[1]


def middle_stretch_wrong_smallest(F):
    return np.sort(np.linalg.svd(F, compute_uv=False))[0]


def plane_norm_squared_correct(p, M):
    return float(p @ np.linalg.inv(M) @ p)


def plane_norm_squared_wrong_direct(p, M):
    return float(p @ M @ p)


def direction_norm_squared_correct(u, M):
    return float(u @ M @ u)


def direction_norm_squared_wrong_reciprocal(u, M):
    return float(u @ np.linalg.inv(M) @ u)


def compatible_signature_correct(q, tol=1e-10):
    q = np.sort(np.asarray(q, float))
    zero = np.abs(q) <= tol
    if np.sum(zero) == 1:
        other = q[~zero]
        return bool(other[0] * other[1] < 0)
    return bool(np.sum(zero) >= 2)


def compatible_signature_wrong_any_zero(q, tol=1e-10):
    return bool(np.any(np.abs(np.asarray(q, float)) <= tol))


def projective_parallel_correct(a, b, tol=1e-10):
    a = np.asarray(a,float)/np.linalg.norm(a)
    b = np.asarray(b,float)/np.linalg.norm(b)
    return bool(min(np.linalg.norm(a-b), np.linalg.norm(a+b)) <= tol)


def projective_parallel_wrong_oriented(a, b, tol=1e-10):
    a = np.asarray(a,float)/np.linalg.norm(a)
    b = np.asarray(b,float)/np.linalg.norm(b)
    return bool(np.linalg.norm(a-b) <= tol)


def volume_ratio_correct(Ma, Mm, C):
    return float(np.sqrt(np.linalg.det(C.T@Mm@C) / np.linalg.det(Ma)))


def volume_ratio_wrong_without_sqrt(Ma, Mm, C):
    return float(np.linalg.det(C.T@Mm@C) / np.linalg.det(Ma))
