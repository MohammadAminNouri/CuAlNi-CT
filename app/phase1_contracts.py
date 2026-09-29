from __future__ import annotations

"""Urgent scientific presentation contracts added in Phase 1.

This module does not replace CT/Ball--James/PTMC solvers.  It makes the
scientific state, geometry, provenance and error semantics explicit at the UI
boundary and provides the immutable Cu--Al--Ni regression input used by tests.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


ZERO_ROTATION_TOL_DEG = 1.0e-10


@dataclass(frozen=True)
class FrozenCuAlNiOracle:
    parent_a: float = 5.812
    product_a: float = 4.392
    product_b: float = 5.360
    product_c: float = 12.940
    product_beta_deg: float = 96.35
    correspondence_m_from_a: tuple[tuple[str, str, str], ...] = (
        ("0", "1", "1"),
        ("1", "0", "0"),
        ("0", "1/3", "-1/3"),
    )
    lambda2: float = 0.9980600225927657
    ct_nearest_zero_residual: float = 0.003876191302128462
    stretch_variants: int = 12
    operator_classes: int = 8
    exact_ct_am: bool = False
    exact_bj_am: bool = False
    generic_ct_bj_mm_assignments: int = 16
    authoritative_mm_relations: int = 8
    native_ptmc_branches: int = 384
    closest_assigned_ct_ptmc_or_deg: float = 0.10868836866302703
    full_cofactor_solutions: int = 0
    ct_supercompatibility_status: str = "not evaluable"


def frozen_cualni_oracle() -> FrozenCuAlNiOracle:
    """The exact canonical state behind the frozen project checkpoint.

    The lattice/correspondence inputs were recovered from the saved unified
    report, not reverse-fitted to only one scalar output.  The exact 1/3 entries
    are retained as rational strings so round-off cannot redefine C.
    """

    return FrozenCuAlNiOracle()


def _cell_metric(a: float, b: float, c: float, alpha_deg: float, beta_deg: float, gamma_deg: float) -> np.ndarray:
    """Conventional direct metric from six lattice parameters."""
    ar, br, gr = np.radians([alpha_deg, beta_deg, gamma_deg])
    return np.array([
        [a*a, a*b*np.cos(gr), a*c*np.cos(br)],
        [a*b*np.cos(gr), b*b, b*c*np.cos(ar)],
        [a*c*np.cos(br), b*c*np.cos(ar), c*c],
    ], dtype=float)


def frozen_cualni_metric_regression() -> dict[str, Any]:
    """Independent metric-only reconstruction of the frozen A→M checkpoint.

    This deliberately does not call CT/BJ/PTMC code. It verifies that the
    canonical lattice + exact C reproduce the generalized metric spectrum
    behind the saved report. Full branch-count/oracle tests remain separate.
    """
    o = frozen_cualni_oracle()
    MA = _cell_metric(o.parent_a, o.parent_a, o.parent_a, 90.0, 90.0, 90.0)
    MM = _cell_metric(o.product_a, o.product_b, o.product_c, 90.0, o.product_beta_deg, 90.0)
    from fractions import Fraction
    C = np.array([[float(Fraction(x)) for x in row] for row in o.correspondence_m_from_a], dtype=float)
    vals, vecs = np.linalg.eigh(MA)
    MA_mhalf = (vecs * (1.0 / np.sqrt(vals))) @ vecs.T
    Ghat = MA_mhalf @ C.T @ MM @ C @ MA_mhalf
    mu = np.linalg.eigvalsh(Ghat)
    lambdas = np.sqrt(mu)
    return {
        "parent_metric": MA,
        "product_metric": MM,
        "normalized_metric": Ghat,
        "mu": mu,
        "lambdas": lambdas,
        "lambda2": float(lambdas[1]),
        "ct_nearest_zero_residual": float(np.min(np.abs(mu - 1.0))),
    }


def point_group_orders(point_group_symbol: str) -> dict[str, int]:
    from cualni_cryst.point_groups import point_group_operations
    operations = point_group_operations(point_group_symbol)
    full = len(operations)
    proper = sum(abs(float(np.linalg.det(np.asarray(item, dtype=float))) - 1.0) <= 1.0e-10 for item in operations)
    return {"full": full, "proper": proper}


def proper_rotation_order(point_group_symbol: str) -> int:
    """Count det(+1) operators in the registered full crystallographic group."""

    from cualni_cryst.point_groups import point_group_operations

    operations = point_group_operations(point_group_symbol)
    return sum(abs(float(np.linalg.det(np.asarray(item, dtype=float))) - 1.0) <= 1.0e-10 for item in operations)


def metric_geometry_audit(metric: Any, direction: Any, plane: Any) -> dict[str, Any]:
    """Return direct/reciprocal geometry without a cubic shortcut.

    With direct metric M and coefficients u=[uvw], p=(hkl):
      ||u||^2 = u^T M u
      ||p||_*^2 = p^T M^-1 p
      physical direct vector can be B u with B^T B=M
      physical plane normal can be B^-T p.
    """

    M = np.asarray(metric, dtype=float).reshape(3, 3)
    u = np.asarray(direction, dtype=float).reshape(3)
    p = np.asarray(plane, dtype=float).reshape(3)
    if not np.all(np.isfinite(M)) or not np.all(np.isfinite(u)) or not np.all(np.isfinite(p)):
        raise ValueError("Metric, direction and plane coefficients must be finite.")
    if np.linalg.norm(M - M.T, ord="fro") > 1.0e-10:
        raise ValueError("Direct metric must be symmetric.")
    eig = np.linalg.eigvalsh(M)
    if float(np.min(eig)) <= 0.0:
        raise ValueError("Direct metric is not positive definite.")
    Minv = np.linalg.inv(M)
    # Symmetric positive square root: B^T B=M.  For symmetric B, B^2=M.
    vals, vecs = np.linalg.eigh(M)
    B = (vecs * np.sqrt(vals)) @ vecs.T
    r = B @ u
    n = np.linalg.solve(B.T, p)
    nr = float(np.linalg.norm(r))
    nn = float(np.linalg.norm(n))
    incidence = float(abs(np.dot(n / nn, r / nr))) if nr > 0 and nn > 0 else float("nan")
    return {
        "direction_squared_metric": float(u @ M @ u),
        "plane_squared_reciprocal_metric": float(p @ Minv @ p),
        "direction_cartesian": r,
        "plane_normal_cartesian": n,
        "direction_plane_normal_projective_cosine": incidence,
        "metric_inverse_residual": float(np.linalg.norm(M @ Minv - np.eye(3), ord="fro")),
    }


def axis_angle_display(axis: Any, angle_deg: float, *, tolerance_deg: float = ZERO_ROTATION_TOL_DEG) -> dict[str, Any]:
    """Never attach physical meaning to the arbitrary axis of the identity rotation."""

    angle = float(angle_deg)
    if abs(angle) <= tolerance_deg:
        return {
            "angle_deg": 0.0,
            "axis": None,
            "axis_text": "undefined for zero rotation (any displayed axis would be conventional only)",
            "zero_rotation": True,
        }
    arr = np.asarray(axis, dtype=float).reshape(3)
    return {
        "angle_deg": angle,
        "axis": tuple(float(v) for v in arr),
        "axis_text": str(tuple(float(v) for v in arr)),
        "zero_rotation": False,
    }


def orientation_provenance_label(state: Mapping[str, Any] | None, analysis: Mapping[str, Any] | None = None) -> str:
    """Map stored orientation provenance to a scientifically precise UI label."""

    state = state or {}
    analysis = analysis or {}
    fields = " ".join(
        str(value)
        for value in (
            state.get("theory_origin"),
            state.get("origin"),
            state.get("definition"),
            state.get("orientation_id"),
            analysis.get("origin_note"),
            analysis.get("source"),
        )
        if value is not None
    ).lower()
    if "polar" in fields:
        return "calculated polar-rotation candidate"
    if "closing" in fields or "cayron" in fields or "ct_" in fields:
        return "Cayron CT closing-gap OR"
    if "ptmc" in fields:
        return "PTMC-derived OR"
    if "experiment" in fields or "ebsd" in fields or "measured" in fields:
        return "experimental OR"
    if "parallel" in fields:
        return "parallelism-derived OR"
    if "euler" in fields:
        return "Euler-derived OR"
    if "manual" in fields or "matrix" in fields or "user" in fields:
        return "manual rotation"
    if "ball" in fields or "james" in fields:
        return "Ball–James-derived OR comparator"
    return "stored physical OR; inspect Full audit for its exact provenance"


def classify_scientific_error(exc: BaseException) -> dict[str, str]:
    """Specific non-destructive scientific error taxonomy for professor-facing UI."""

    info = getattr(exc, "info", None)
    code = str(getattr(info, "code", ""))
    message = str(getattr(info, "message", exc))
    detail = str(getattr(info, "technical_detail", ""))
    text = f"{code} {message} {detail}".lower()

    if "stale" in text or code == "STALE_RESULTS":
        title = "Stale calculated state"
        hint = "Recalculate the current draft. The previous successful result is retained only as stale provenance."
    elif "singular" in text and ("correspond" in text or "matrix" in text):
        title = "Singular correspondence"
        hint = "Correct C(M←A); it must be invertible. The draft and previous calculated state were not erased."
    elif "positive definite" in text or "not spd" in text or "non-spd" in text or "cholesky" in text:
        title = "Non-SPD lattice metric"
        hint = "Correct the cell/metric so it is symmetric positive definite. No metric was silently repaired."
    elif "rational" in text or "fraction" in text or "p/q" in text or "matrix entry" in text:
        title = "Malformed exact rational input"
        hint = "Use an integer, decimal, or exact p/q entry. Existing draft values remain available for correction."
    elif "so(3)" in text or "rotation" in text or "orientation" in text or "determinant" in text and "1" in text:
        title = "Invalid orientation relationship"
        hint = "Provide a proper physical rotation or explicitly request the available SO(3) projection route; C is never substituted for R."
    elif "unsupported" in text or "not implemented" in text or "not applicable" in text:
        title = "Unsupported scientific state"
        hint = "This state cannot be evaluated by the requested theory branch. It is not classified as a failed physical criterion."
    else:
        title = str(getattr(info, "title", "Scientific calculation error")) or "Scientific calculation error"
        hint = str(getattr(info, "hint", "Correct the stated inconsistency and recalculate; no scientific meaning was silently repaired."))

    return {
        "title": title,
        "message": message,
        "hint": hint,
        "code": code or "SCIENTIFIC_ERROR",
        "technical_detail": detail,
    }


def physical_closing_gap_label(row: Any) -> str:
    """Physical CT closing-gap branch label while preserving raw IDs for audit."""

    metadata = getattr(row, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    operator = metadata.get("operator_index")
    twin = metadata.get("twin_index")
    candidate = metadata.get("candidate_index")
    kind = str(metadata.get("twin_kind") or "").strip().upper()
    classification = str(metadata.get("twin_classification") or "").strip().lower()
    reps = metadata.get("twin_representations") or ()
    try:
        idx = int(candidate)
        branch = chr(ord("A") + idx) if 0 <= idx < 26 else str(idx)
    except (TypeError, ValueError):
        branch = "?"
    twin_kind = {"I": "Type-I", "II": "Type-II"}.get(kind, "CT")
    qualifiers: list[str] = []
    if classification:
        qualifiers.append(classification)
    if isinstance(reps, (list, tuple)) and len(reps) > 1:
        qualifiers.append("representable as " + "/".join(f"Type-{x}" for x in reps))
    tail = f" · {'; '.join(qualifiers)}" if qualifiers else ""
    op_text = "?" if operator is None else str(operator)
    twin_text = "?" if twin is None else str(twin)
    return f"{twin_kind} twin · operator {op_text} · twin {twin_text} · sign branch {branch}{tail}"


def pole_series_metadata(*, label: str, phase_id: str, object_kind: str, reference_phase_id: str, provenance: str, source: str, exact_status: str) -> dict[str, str]:
    return {
        "series": label,
        "phase": phase_id,
        "object type": object_kind,
        "reference frame": f"{reference_phase_id} Cartesian reference frame",
        "projective convention": "unoriented pole; v ~ -v; upper hemisphere",
        "OR provenance": provenance,
        "theory source": source,
        "scientific status": exact_status,
    }


def frozen_cualni_metric_regression() -> dict[str, Any]:
    """Independently reconstruct the frozen metric spectrum from canonical inputs.

    This is deliberately small and equation-level: it is not a replacement for
    CalculationService.  It guards the frozen fixture itself against accidental
    transcription errors before the heavier application/unified-theory test runs.
    """

    o = frozen_cualni_oracle()
    beta = np.deg2rad(o.product_beta_deg)
    M_a = np.eye(3) * o.parent_a**2
    M_m = np.array(
        [
            [o.product_a**2, 0.0, o.product_a * o.product_c * np.cos(beta)],
            [0.0, o.product_b**2, 0.0],
            [o.product_a * o.product_c * np.cos(beta), 0.0, o.product_c**2],
        ],
        dtype=float,
    )
    C = np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0 / 3.0, -1.0 / 3.0]],
        dtype=float,
    )
    pulled = C.T @ M_m @ C
    # Parent is cubic here, so M_A^{-1/2}=I/a0 exactly in this basis.
    ghat = pulled / (o.parent_a**2)
    mu = np.linalg.eigvalsh(ghat)
    lambdas = np.sqrt(mu)
    eta = mu - 1.0
    return {
        "M_A": M_a,
        "M_M": M_m,
        "C_M_from_A": C,
        "generalized_mu": mu,
        "lambdas": lambdas,
        "lambda2": float(lambdas[1]),
        "ct_nearest_zero_residual": float(np.min(np.abs(eta))),
    }
