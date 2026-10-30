from __future__ import annotations

"""Evidence-gated β-parent / 18R frame and basis validation.

This module does NOT infer a physical orientation relationship from daughter-only
EBSD. It audits an explicit change of crystallographic basis and checks measured
parent-direction or retained-austenite anchors against a conditional reconstruction.

Conventions
-----------
B: column-wise direct lattice structure matrix, x_cart = B @ u.
P: integer columns of NEW direct basis in OLD direct coordinates:
       B_new(physical) = B_old @ P
g: crystal-Cartesian -> sample Cartesian, proper SO(3).
R: daughter crystal Cartesian -> parent crystal Cartesian.
"""

from dataclasses import dataclass
from fractions import Fraction
import numpy as np
from scipy.spatial.transform import Rotation

from .lattice import Lattice
from .orientation_kernel import require_so3
from .ebsd_analysis import minimum_disorientation


@dataclass(frozen=True)
class BasisAudit:
    integer_matrix_old_from_new: tuple[tuple[float, float, float], ...]
    absolute_volume_factor: int | None
    volume_ratio_new_over_old: float
    integral_basis: bool
    metric_relative_residual: float
    crystal_frame_new_to_old: np.ndarray
    status: str


def _rational_basis(P: object) -> tuple[np.ndarray, bool]:
    """Use exact small fractions, without silently rounding a proposed basis.

    Rational transformations may describe an EBSD pseudocell, but they do not
    prove lattice periodicity or atomic correspondence.
    """
    try:
        rows = np.asarray(P, dtype=object)
        if rows.shape != (3, 3):
            raise ValueError("Basis P must be 3 x 3")
        fractions = []
        for raw in rows.reshape(-1):
            val = Fraction(str(raw)).limit_denominator(12)
            numeric = float(Fraction(str(raw)))
            if (not np.isfinite(numeric) or abs(float(val)-numeric) > 1e-9
                    or abs(numeric) > 100):
                raise ValueError("Basis entries must be small exact rational numbers")
            fractions.append(val)
        values = np.array([float(v) for v in fractions]).reshape(3,3)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError("Invalid small-rational basis matrix P") from exc
    if float(np.linalg.det(values)) <= 1e-10:
        raise ValueError("P must be nonsingular and right-handed (det P > 0)")
    integral = all(v.denominator == 1 for v in fractions)
    return values, integral


def audit_direct_basis(
    old_lattice: Lattice, new_lattice: Lattice, P: object,
    *, relative_tolerance: float = 5.0e-4,
) -> BasisAudit:
    """Require metric-compatible P; never infer atomic correspondence.

    A relaxed tolerance is for a literature metric screen ONLY.
    """
    P, integral = _rational_basis(P)
    G_old = old_lattice.metric()
    G_new = new_lattice.metric()
    predicted = P.T @ G_old @ P
    err = float(np.linalg.norm(predicted - G_new, "fro") /
                max(np.linalg.norm(G_new, "fro"), 1e-15))
    if not np.isfinite(err) or err > relative_tolerance:
        raise ValueError(
            f"Proposed P fails lattice metric equality: relative residual "
            f"{err:.6g} exceeds {relative_tolerance:.6g}"
        )
    # Each cell uses its own Cartesian orthonormal axes as built by Lattice.
    # This rotation takes a vector in the NEW Cartesian frame to the OLD one.
    B_old, B_new = old_lattice.structure_matrix(), new_lattice.structure_matrix()
    T_raw = B_old @ P @ np.linalg.inv(B_new)
    # Different measured and literature cells may have small metric strain.
    # Keep that error in the audit, and use only the polar rotation to express
    # a CONDITIONAL frame comparison. No physical mapping is inferred.
    U, _, Vt = np.linalg.svd(T_raw)
    T = require_so3(U @ Vt, tolerance=2e-8,
                    name="projected crystallographic frame")
    volume_ratio = float(np.linalg.det(P))
    return BasisAudit(
        integer_matrix_old_from_new=tuple(tuple(float(v) for v in row) for row in P),
        absolute_volume_factor=(round(volume_ratio) if integral else None),
        volume_ratio_new_over_old=volume_ratio,
        integral_basis=integral,
        metric_relative_residual=err,
        crystal_frame_new_to_old=T,
        status=(
            "rational subcell candidate; atomic translations and physical OR unverified"
            if not integral or err > 5e-4 else
            "metric-compatible integral basis; atomic correspondence not checked"
        ),
    )


def rebase_orientation_relationship(
    R_parent_old_from_child_old: object,
    parent_change: BasisAudit, child_change: BasisAudit,
) -> np.ndarray:
    """Convert OR from old Cartesian parent/child frames to new frames.

    x_parent_old = R_old @ x_child_old;
    x_parent_old = T_parent @ x_parent_new;
    x_child_old = T_child @ x_child_new;
    hence R_new = T_parent.T @ R_old @ T_child.
    """
    R = require_so3(np.asarray(R_parent_old_from_child_old, float),
                    tolerance=2e-8, name="input OR")
    T_A, T_M = (parent_change.crystal_frame_new_to_old,
                 child_change.crystal_frame_new_to_old)
    return require_so3(T_A.T @ R @ T_M, tolerance=2e-8, name="rebased OR")


def lattice_direction_to_cart(lattice: Lattice, uvw: object) -> np.ndarray:
    u = np.asarray(uvw, float)
    if u.shape != (3,) or not np.isfinite(u).all():
        raise ValueError("Direction must have exactly three finite components")
    v = lattice.structure_matrix() @ u
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        raise ValueError("Zero crystal direction")
    return v / norm


def direction_anchor_residual(
    parent_g: np.ndarray, parent_symmetry: tuple[np.ndarray, ...],
    lattice: Lattice, parent_direction_uvw: object,
    observed_sample_direction: object, *, signed: bool = False,
) -> float:
    """Angle against ONE parent direction; leaves azimuth about that axis unknown."""
    g = require_so3(parent_g, tolerance=2e-8, name="candidate parent")
    d = lattice_direction_to_cart(lattice, parent_direction_uvw)
    sample = np.asarray(observed_sample_direction, dtype=float)
    if sample.shape != (3,) or not np.isfinite(sample).all():
        raise ValueError("Observed specimen direction must be a finite 3-vector")
    if np.linalg.norm(sample) < 1e-12:
        raise ValueError("Observed specimen direction is zero")
    sample /= np.linalg.norm(sample)
    if not parent_symmetry:
        raise ValueError("Known parent symmetry required")
    values = []
    for S in parent_symmetry:
        s = require_so3(S, tolerance=2e-8, name="parent symmetry")
        dot = float(np.dot(g @ s @ d, sample))
        values.append(np.degrees(np.arccos(np.clip(
            dot if signed else abs(dot), -1, 1))))
    return float(min(values))


def parent_orientation_from_two_directions(
    lattice: Lattice, uvw_1: object, uvw_2: object,
    sample_1: object, sample_2: object,
    *, max_pair_angular_mismatch_deg: float = 2.0,
) -> np.ndarray:
    """Build a full calibrated SO3 parent orientation from TWO independent pairs.

    Refuses incompatible angular separation and parallel directions.
    Measurements remain the user's responsibility.
    """
    if not np.isfinite(max_pair_angular_mismatch_deg) or not 0 <= max_pair_angular_mismatch_deg <= 20:
        raise ValueError('Pair angular mismatch tolerance must be finite and within 0..20 degrees')
    d1 = lattice_direction_to_cart(lattice, uvw_1)
    d2 = lattice_direction_to_cart(lattice, uvw_2)
    a = np.asarray(sample_1, float)
    b = np.asarray(sample_2, float)
    for v in (a, b):
        if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-12:
            raise ValueError("Two finite nonzero specimen direction vectors required")
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    if min(np.linalg.norm(np.cross(d1, d2)),
           np.linalg.norm(np.cross(a, b))) < 0.1:
        raise ValueError("Direction pairs are (nearly) collinear; twist is not resolved")
    old_sep = float(np.dot(d1, d2))
    new_sep = float(np.dot(a, b))
    delta = abs(np.degrees(np.arccos(np.clip(old_sep, -1, 1))) -
                np.degrees(np.arccos(np.clip(new_sep, -1, 1))))
    if delta > max_pair_angular_mismatch_deg:
        raise ValueError(f"Anchor pair internal angle inconsistent by {delta:.4g}°")
    # Wahba least-squares fit: share small calibration/noise errors between both
    # measured axes rather than forcing the first pair to agree exactly.
    # Strict pair separation and handedness tests were done above.
    estimated, _ = Rotation.align_vectors(
        np.stack((a, b)), np.stack((d1, d2)),
    )
    result = require_so3(
        estimated.as_matrix(), tolerance=2e-8, name="two-direction parent anchor",
    )
    fits = np.stack((result @ d1, result @ d2))
    observed = np.stack((a, b))
    per_axis = np.rad2deg(np.arccos(np.clip(
        np.sum(fits * observed, axis=1), -1.0, 1.0,
    )))
    # A sound alignment must satisfy BOTH original experimental constraints.
    if np.max(per_axis) > max_pair_angular_mismatch_deg + 1e-7:
        raise ValueError("Best-fit parent anchor is inconsistent with measured directions")
    return result


def compare_parent_to_anchor(
    candidate_parent_g: np.ndarray, anchor_parent_g: np.ndarray,
    parent_symmetry: tuple[np.ndarray, ...],
) -> float:
    """Nearest cubic-equivalent orientation error (degrees) to measured parent."""
    return float(minimum_disorientation(
        require_so3(candidate_parent_g, tolerance=2e-8, name="candidate parent"),
        require_so3(anchor_parent_g, tolerance=2e-8, name="measured parent"),
        parent_symmetry,
    ).angle_deg)


def audit_anchor_status(
    *, measured_parent_orientation: bool, independent_direction_count: int,
    indexed_parent_phase_present: bool, verified_change_of_basis: bool,
    scan_frame_verified: bool,
) -> dict[str, object]:
    """Explicit refusal to certify a daughter-only physically unique parent."""
    has_absolute = bool(measured_parent_orientation or independent_direction_count >= 2)
    blockers = []
    if not has_absolute:
        blockers.append("no calibrated absolute parent orientation or two noncollinear parent directions")
    if not verified_change_of_basis:
        blockers.append("conventional-to-indexed 18R basis not experimentally verified")
    if not scan_frame_verified:
        blockers.append("Oxford Euler/map/specimen frame alignment not independently verified")
    if not indexed_parent_phase_present and not has_absolute:
        blockers.append("no indexed retained/reverted parent phase")
    return {
        "physical_parent_or_certified": False,
        "declared_prerequisites_present": not blockers,
        "numerical_candidate_allowed": True,
        "independent_absolute_anchor_available": has_absolute,
        "blocking_requirements": blockers,
        "note": "User-declared external evidence is not authenticated; daughter misorientations alone never certify a unique physical OR.",
    }


def parent_fit_in_raw_oxford_frame(fit: dict) -> np.ndarray:
    """Undo an explicitly calibrated specimen-frame change for H5OINA replay.

    The AR fitting code transforms raw Oxford g to corrected g' = H @ g.
    Forward/round-trip replay deliberately uses raw indexed g, so a parent
    inferred in the corrected frame must first be converted H.T @ g_parent'.
    This does not infer H or certify the parent orientation.
    """
    g_corrected = require_so3(
        np.asarray(fit["parent_crystal_to_sample_rotation"], float),
        tolerance=2e-8, name="AR fitted parent",
    )
    correction = fit.get("explicit_frame_correction_matrix")
    if correction is None:
        return g_corrected
    H = require_so3(np.asarray(correction, float),
                    tolerance=2e-8, name="AR specimen-frame correction")
    return require_so3(H.T @ g_corrected, tolerance=2e-8,
                       name="parent in raw Oxford CS1")
