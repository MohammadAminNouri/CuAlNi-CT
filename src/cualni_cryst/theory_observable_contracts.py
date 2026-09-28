from __future__ import annotations

"""Explicit observable contracts for cross-theory martensite comparisons.

This module is intentionally solver-independent: it calls neither CT,
Ball--James nor PTMC.  It does, however, encode the explicit producer contracts
used by the unified comparison layer so that a vector is never reinterpreted
from its field name or numerical appearance.  Its purpose is to prevent a
specific class of scientific bug: treating vectors with similar names as the
same physical observable when they are defined by different equations.

For exact A/M compatibility, Cayron CT supplies the source-native vector
``d`` through ``d = SMC m``.  In a common orthonormal parent frame with a unit
habit normal ``n``, that physicalizes to

    d = (I - U^{-2}) n.

Ball--James/PTMC rank-one compatibility instead uses

    R U = I + b ⊗ n

(or, for PTMC after a lattice-invariant deformation, ``R F_pre = I+b⊗n``).
The CT vector ``d`` and the rank-one jump ``b`` are not numerically identical in
general.  When the state is an exact single-variant IPS and ``J=det(U)>0``, the
Sherman--Morrison identity gives the downstream algebraic bridge

    b = J^2 d - J(J-1)n.

This bridge is *not* a replacement CT equation.  It is a comparison-layer
identity derived from the two independently defined representations.  Native
outputs must remain native; conversion is performed only when an exact
rank-one observable is explicitly requested.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

Array = np.ndarray


class ShapeVectorRole(str, Enum):
    """Meaning of a stored A/M shape/shear vector.

    ``UNSPECIFIED`` is fail-safe: a vector carrying this role must not enter an
    exact rank-one equivalence test.
    """

    UNSPECIFIED = "unspecified"
    CT_SMC_IPS_D = "ct_smc_ips_d"
    RANK_ONE_PARENT_IDENTITY_B = "rank_one_parent_identity_b"
    RANK_ONE_DILATED_IDENTITY_B = "rank_one_dilated_identity_b"


@dataclass(frozen=True)
class AMRankOneBridgeAudit:
    """Independent checks for the CT-SMC -> rank-one representation bridge."""

    volume_ratio: float
    rank_one_shape_cartesian: Array
    smc_from_stretch_cartesian: Array
    reconstructed_smc_cartesian: Array
    source_smc_relative_residual: float
    roundtrip_relative_residual: float
    cauchy_green_relative_residual: float
    rotation_orthogonality_residual: float
    rotation_determinant_residual: float

    @property
    def maximum_residual(self) -> float:
        return max(
            self.source_smc_relative_residual,
            self.roundtrip_relative_residual,
            self.cauchy_green_relative_residual,
            self.rotation_orthogonality_residual,
            self.rotation_determinant_residual,
        )


def canonical_shape_vector_role(value: ShapeVectorRole | str | None) -> str | None:
    if value is None:
        return None
    return ShapeVectorRole(value).value


def infer_native_shape_vector_role(
    *,
    theory: object,
    prediction_kind: object,
    exact: bool | None,
    metadata: dict[str, object] | None = None,
) -> ShapeVectorRole:
    """Infer the *documented native* vector role from the producer contract.

    This is deliberately fail-closed.  It does not inspect vector values or
    guess semantics from a field name.  The mapping is keyed only by the
    theory/prediction producer contract already frozen by ``theory_unified``.

    PTMC rows are additionally cross-checked against their explicit
    ``dilatational_factor``/``true_invariant_plane`` metadata when present.
    Any contradiction raises rather than silently reclassifying the vector.
    Experimental rows are UNSPECIFIED unless a future producer explicitly
    supplies a ``shape_vector_role`` metadata value.
    """

    def token(value: object) -> str:
        raw = getattr(value, "value", value)
        return str(raw)

    theory_token = token(theory)
    kind_token = token(prediction_kind)
    meta = {} if metadata is None else dict(metadata)

    explicit = meta.get("shape_vector_role")
    if explicit is not None:
        return ShapeVectorRole(canonical_shape_vector_role(explicit))

    if theory_token == "cayron_ct" and kind_token in {
        "ct_am_habit",
        "ct_supercompatibility",
    }:
        return ShapeVectorRole.CT_SMC_IPS_D

    if theory_token == "ball_james" and kind_token == "ball_james_am":
        if exact is not True:
            raise ValueError(
                "Ball--James A/M producer contract requires exact=True for its "
                "RU-I=b⊗n rank-one vector"
            )
        return ShapeVectorRole.RANK_ONE_PARENT_IDENTITY_B

    if theory_token == "ptmc" and kind_token == "ptmc_habit":
        declared_true = meta.get("true_invariant_plane")
        delta = meta.get("dilatational_factor")
        if declared_true is not None and bool(declared_true) != bool(exact):
            raise ValueError(
                "PTMC row contract is inconsistent: exact flag and "
                "true_invariant_plane metadata disagree"
            )
        if delta is not None:
            delta_f = float(delta)
            if not np.isfinite(delta_f) or delta_f <= 0.0:
                raise ValueError("PTMC dilatational_factor must be finite and positive")
            is_identity = abs(delta_f - 1.0) <= 1.0e-12
            if exact is True and not is_identity:
                raise ValueError(
                    "PTMC row marked exact=True but dilatational_factor != 1; "
                    "this is not a parent-identity IPS contract"
                )
            if exact is False and is_identity and declared_true is not False:
                raise ValueError(
                    "PTMC row marked exact=False at dilatational_factor=1 without "
                    "an explicit non-IPS declaration"
                )
        return (
            ShapeVectorRole.RANK_ONE_PARENT_IDENTITY_B
            if exact is True
            else ShapeVectorRole.RANK_ONE_DILATED_IDENTITY_B
        )

    return ShapeVectorRole.UNSPECIFIED


def _finite_vector(value: Array, *, name: str) -> Array:
    vector = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _unit_normal(value: Array, *, tolerance: float = 1.0e-10) -> Array:
    normal = _finite_vector(value, name="habit normal")
    norm = float(np.linalg.norm(normal))
    if norm <= np.finfo(float).tiny:
        raise ValueError("habit normal must be nonzero")
    if abs(norm - 1.0) > tolerance:
        raise ValueError(
            "A/M rank-one conversion requires an explicitly unit Cartesian habit "
            f"normal; received norm={norm:.16g}"
        )
    return normal


def _relative(first: Array, second: Array) -> float:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), np.finfo(float).tiny)
    return float(np.linalg.norm(a - b) / scale)


def _validated_stretch(U: Array) -> Array:
    value = np.asarray(U, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(value)):
        raise ValueError("U must be finite")
    scale = max(float(np.linalg.norm(value, ord="fro")), np.finfo(float).tiny)
    asymmetry = float(np.linalg.norm(value - value.T, ord="fro") / scale)
    if asymmetry > 1.0e-10:
        raise ValueError(f"U must be symmetric; relative asymmetry={asymmetry:.3e}")
    value = 0.5 * (value + value.T)
    eigenvalues = np.linalg.eigvalsh(value)
    if float(np.min(eigenvalues)) <= 0.0:
        raise ValueError(f"U must be positive definite; eigenvalues={eigenvalues}")
    return value


def _positive_volume_ratio(value: float) -> float:
    J = float(value)
    if not np.isfinite(J) or J <= 0.0:
        raise ValueError("volume ratio must be finite and strictly positive")
    return J


def smc_vector_from_stretch_cartesian(
    stretch: Array,
    unit_habit_normal_cartesian: Array,
) -> Array:
    """Evaluate ``d=(I-U^{-2})n`` directly in one orthonormal parent frame."""

    U = _validated_stretch(stretch)
    n = _unit_normal(unit_habit_normal_cartesian)
    U2 = U @ U
    return n - np.linalg.solve(U2, n)


def smc_vector_from_rank_one_shape_cartesian(
    rank_one_shape_cartesian: Array,
    unit_habit_normal_cartesian: Array,
    volume_ratio: float,
) -> Array:
    """Inverse bridge from ``b`` to the CT-SMC physical vector ``d``."""

    b = _finite_vector(rank_one_shape_cartesian, name="rank-one shape vector")
    n = _unit_normal(unit_habit_normal_cartesian)
    J = _positive_volume_ratio(volume_ratio)
    return ((J - 1.0) / J) * n + b / (J * J)


def rank_one_shape_from_smc_cartesian(
    smc_vector_cartesian: Array,
    unit_habit_normal_cartesian: Array,
    volume_ratio: float,
) -> Array:
    """Convert an exact CT-SMC vector to the ``RU-I=b⊗n`` rank-one gauge."""

    d = _finite_vector(smc_vector_cartesian, name="CT SMC vector")
    n = _unit_normal(unit_habit_normal_cartesian)
    J = _positive_volume_ratio(volume_ratio)
    return (J * J) * d - J * (J - 1.0) * n


def audit_smc_rank_one_bridge(
    smc_vector_cartesian: Array,
    unit_habit_normal_cartesian: Array,
    stretch: Array,
) -> AMRankOneBridgeAudit:
    """Cross-check the bridge against the deformation, not against itself only.

    Besides the algebraic round-trip, this independently checks:

    * the source CT vector against ``(I-U^{-2})n``;
    * the reconstructed ``F=I+b⊗n`` against ``U^2`` through ``F^T F``;
    * the implied ``R=F U^{-1}`` against SO(3).

    A wrong frame, normal gauge, source vector, or bridge formula therefore
    cannot pass merely because forward/inverse conversion functions share the
    same mistake.
    """

    U = _validated_stretch(stretch)
    d = _finite_vector(smc_vector_cartesian, name="CT SMC vector")
    n = _unit_normal(unit_habit_normal_cartesian)
    J = _positive_volume_ratio(float(np.linalg.det(U)))

    d_from_U = smc_vector_from_stretch_cartesian(U, n)
    b = rank_one_shape_from_smc_cartesian(d, n, J)
    d_roundtrip = smc_vector_from_rank_one_shape_cartesian(b, n, J)

    F = np.eye(3) + np.outer(b, n)
    U2 = U @ U
    cauchy = _relative(F.T @ F, U2)
    R = F @ np.linalg.inv(U)
    orthogonality = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
    determinant_residual = abs(float(np.linalg.det(R)) - 1.0)

    return AMRankOneBridgeAudit(
        volume_ratio=J,
        rank_one_shape_cartesian=b,
        smc_from_stretch_cartesian=d_from_U,
        reconstructed_smc_cartesian=d_roundtrip,
        source_smc_relative_residual=_relative(d, d_from_U),
        roundtrip_relative_residual=_relative(d, d_roundtrip),
        cauchy_green_relative_residual=cauchy,
        rotation_orthogonality_residual=orthogonality,
        rotation_determinant_residual=determinant_residual,
    )
