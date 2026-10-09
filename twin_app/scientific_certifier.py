from __future__ import annotations

"""Independent matrix-level verification of published PTMC output contracts.

Not an alternative source of habit solutions. This checks calculated rotations,
rank-one tensors, laminate fractions and the middle-stretch criterion from the
*original matrices*. Reported solver residual fields are not used as evidence.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CertificationSummary:
    twin_relations_checked: int
    habit_solutions_checked: int
    maximum_twin_tensor_residual: float
    maximum_habit_tensor_residual: float
    maximum_middle_stretch_residual: float


class ScientificCertificationError(AssertionError):
    """A returned numerical solution violates an independently checked identity."""


def _matrix(value: Any, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ScientificCertificationError(f"{name}: expected finite 3x3 matrix")
    return a


def _vector(value: Any, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ScientificCertificationError(f"{name}: expected finite three-vector")
    return a


def _residual(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, ord="fro") / max(
        float(np.linalg.norm(a, ord="fro")),
        float(np.linalg.norm(b, ord="fro")), 1.0,
    ))


def _proper_rotation(R: np.ndarray, name: str, tol: float) -> None:
    error = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
    det = float(np.linalg.det(R))
    if error > tol or abs(det - 1.0) > tol:
        raise ScientificCertificationError(
            f"{name}: not a proper rotation (orthogonality {error:.3e}, det {det:.16g})"
        )


def certify_ptmc_twinning(report: Any, *, tolerance: float = 1.0e-7) -> CertificationSummary:
    """Check every exact twinning relation and true A/M habit branch.

    A twin: R_t U_j - U_i = a ⊗ n.
    A true habit: Fλ = U_i + λ(a ⊗ n), R_h Fλ - I = b ⊗ m.
    The middle principal stretch of Fλ must be 1.

    It is essential not to treat an absent solution as a failure: this routine
    verifies *present* exact solutions, without inventing solutions for other
    branches or confusing a no-root result with a crash.
    """
    if not (np.isfinite(tolerance) and 0.0 < tolerance <= 1.0e-4):
        raise ValueError("tolerance must lie in (0, 1e-4]")

    variants = {int(v.index): _matrix(v.U, f"variant {v.index} U") for v in report.variants}
    # All variants in one martensitic transformation must have positive
    # stretches and equal volume change (they are symmetry-related).
    for index, U in variants.items():
        if _residual(U, U.T) > tolerance or float(np.linalg.eigvalsh(U).min()) <= 0.0:
            raise ScientificCertificationError(f"variant {index}: stretch is not SPD")
    twins: dict[tuple[int, int, int], tuple[Any, np.ndarray]] = {}
    maximum_twin = 0.0
    maximum_habit = 0.0
    maximum_middle = 0.0

    for twin in report.twin_relations:
        i, j, branch = int(twin.base_variant_index), int(twin.other_variant_index), int(twin.branch)
        key = (i, j, branch)
        if i not in variants or j not in variants or i == j or key in twins:
            raise ScientificCertificationError(f"twin {key}: missing/duplicate/identical stretch indices")
        det_i = float(np.linalg.det(variants[i])); det_j = float(np.linalg.det(variants[j]))
        if abs(det_i-det_j) / max(abs(det_i),abs(det_j),1.0e-15) > tolerance:
            raise ScientificCertificationError(f"twin {key}: variant volume changes disagree")
        R = _matrix(twin.rotation_hat, f"twin {key} R")
        _proper_rotation(R, f"twin {key}", tolerance)
        a = _vector(twin.a, f"twin {key} a")
        n = _vector(twin.n_reference, f"twin {key} n")
        if abs(float(np.linalg.norm(n)) - 1.0) > tolerance:
            raise ScientificCertificationError(f"twin {key}: n is not unit length")
        outer = np.outer(a, n)
        error = _residual(R @ variants[j] - variants[i], outer)
        if error > tolerance:
            raise ScientificCertificationError(f"twin {key}: rank-one tensor residual={error:.3e}")
        maximum_twin = max(maximum_twin, error)
        twins[key] = (twin, outer)

    checked_habits = 0
    for sol in report.solutions:
        if not bool(sol.true_invariant_plane):
            # A dilatational (non-invariant) interface is not a true A/M habit.
            continue
        if sol.other_variant_index is None or sol.twin_branch is None:
            raise ScientificCertificationError("true twinning habit missing pair/branch provenance")
        key = (int(sol.base_variant_index), int(sol.other_variant_index), int(sol.twin_branch))
        if key not in twins:
            raise ScientificCertificationError(f"habit {key}: no exact matching twin branch")
        if sol.other_variant_volume_fraction is None or sol.base_variant_volume_fraction is None:
            raise ScientificCertificationError(f"habit {key}: missing laminate volume fractions")
        lam = float(sol.other_variant_volume_fraction)
        base_fraction = float(sol.base_variant_volume_fraction)
        if (not np.isfinite(lam) or not np.isfinite(base_fraction)
            or lam < -tolerance or lam > 1 + tolerance
            or base_fraction < -tolerance or base_fraction > 1 + tolerance
            or abs(lam + base_fraction - 1) > tolerance):
            raise ScientificCertificationError(f"habit {key}: invalid lambda/fraction normalization")
        F = variants[key[0]] + lam * twins[key][1]
        pre = _matrix(sol.pre_shape_deformation, f"habit {key} Fλ")
        err_pre = _residual(pre, F)
        if err_pre > tolerance:
            raise ScientificCertificationError(f"habit {key}: Fλ not linked to its twin ({err_pre:.3e})")
        R_h = _matrix(sol.habit_rotation, f"habit {key} R_h")
        _proper_rotation(R_h, f"habit {key} R_h", tolerance)
        b = _vector(sol.rank_one_vector, f"habit {key} b")
        m = _vector(sol.habit_normal_parent_cartesian, f"habit {key} m")
        if abs(float(np.linalg.norm(m)) - 1) > tolerance:
            raise ScientificCertificationError(f"habit {key}: m is not a unit normal")
        residual = _residual(R_h @ F - np.eye(3), np.outer(b, m))
        if residual > tolerance:
            raise ScientificCertificationError(f"habit {key}: A/M rank-one tensor residual={residual:.3e}")
        # Spectral criterion: the middle stretch singular value is exactly one.
        middle_error = abs(float(np.linalg.svd(F, compute_uv=False)[1]) - 1.0)
        if middle_error > tolerance:
            raise ScientificCertificationError(f"habit {key}: middle-stretch residual={middle_error:.3e}")
        maximum_habit = max(maximum_habit, residual, err_pre)
        maximum_middle = max(maximum_middle, middle_error)
        checked_habits += 1

    return CertificationSummary(
        twin_relations_checked=len(twins),
        habit_solutions_checked=checked_habits,
        maximum_twin_tensor_residual=maximum_twin,
        maximum_habit_tensor_residual=maximum_habit,
        maximum_middle_stretch_residual=maximum_middle,
    )
