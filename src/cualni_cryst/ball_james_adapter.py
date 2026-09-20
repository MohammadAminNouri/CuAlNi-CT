from __future__ import annotations

"""Independent Ball--James / nonlinear-elasticity transformation adapter.

This module deliberately does not import Cayron CT, CMC, SMC, PTMC, or the
existing compatibility-atlas implementation.  It consumes only the physical
crystallographic state (metrics, correspondence and parent symmetry) and
constructs nonlinear-elasticity predictions independently.

Conventions
-----------
* Correspondence C maps parent direct crystal coordinates to product direct
  crystal coordinates: u_M = C @ u_A.
* U and all rotations in this module are expressed in the unique parent
  metric-whitened Cartesian frame B_A = M_A^(1/2).
* Rank-one equations are written

      R F_other - F_base = a \\otimes n.

  For an A/M interface F_base = I and F_other = U_i.  For an M/M twin,
  F_base = U_j and F_other = U_i.
* ``n`` is normalized to unit Euclidean length in the parent orthonormal
  frame; ``a`` is rescaled so that a \\otimes n is unchanged.

The mathematical sources are Ball & James (1987), James & Hane (2000,
Proposition 1 and Eqs. 15--18), and Chen et al. (2013, Theorem 2).
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.linalg import eigh

from .correspondence import Correspondence
from .lattice import Lattice
from .numerics import DEFAULT_NUMERICAL_POLICY, NumericalPolicy
from .point_groups import point_group_operations

Array = np.ndarray


def _relative_residual(lhs: Array, rhs: Array) -> float:
    left = np.asarray(lhs, dtype=float)
    right = np.asarray(rhs, dtype=float)
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1.0)
    return float(np.linalg.norm(left - right) / scale)


def _validated_spd(matrix: Array, *, name: str) -> Array:
    value = np.asarray(matrix, dtype=float)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    symmetric = 0.5 * (value + value.T)
    if _relative_residual(value, symmetric) > 1.0e-12:
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if float(np.min(eigenvalues)) <= 0.0:
        raise ValueError(f"{name} must be positive definite; eigenvalues={eigenvalues}")
    return symmetric


def _validated_invertible(matrix: Array, *, name: str) -> Array:
    value = np.asarray(matrix, dtype=float)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    determinant = float(np.linalg.det(value))
    if abs(determinant) <= 1.0e-14:
        raise ValueError(f"{name} must be invertible")
    return value


def _spd_sqrt(matrix: Array) -> Array:
    value = _validated_spd(matrix, name="SPD matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


def _spd_inv_sqrt(matrix: Array) -> Array:
    value = _validated_spd(matrix, name="SPD matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    return (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T


def _proper_rotation_residual(rotation: Array) -> tuple[float, float, float]:
    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    orthogonality = float(np.linalg.norm(matrix.T @ matrix - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(matrix))
    maximum = max(orthogonality, abs(determinant - 1.0))
    return maximum, orthogonality, determinant


def _unit(vector: Array, *, name: str) -> Array:
    value = np.asarray(vector, dtype=float).reshape(3)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-15:
        raise ValueError(f"{name} must be nonzero")
    return value / norm


def _cofactor_matrix(matrix: Array) -> Array:
    """Classical 3x3 cofactor matrix, valid also for singular matrices."""

    value = np.asarray(matrix, dtype=float).reshape(3, 3)
    c1, c2, c3 = value[:, 0], value[:, 1], value[:, 2]
    return np.column_stack((np.cross(c2, c3), np.cross(c3, c1), np.cross(c1, c2)))


def _projective_angle_deg(first: Array, second: Array) -> float:
    left = _unit(first, name="first vector")
    right = _unit(second, name="second vector")
    sine = float(np.linalg.norm(np.cross(left, right)))
    cosine = abs(float(left @ right))
    return float(np.degrees(np.arctan2(sine, cosine)))


@dataclass(frozen=True)
class MetricStretchSpectrum:
    """Metric-native generalized-eigenvalue spectrum.

    The crystal-coordinate eigenvectors satisfy

        C^T M_M C v_i = mu_i M_A v_i,
        lambda_i = sqrt(mu_i),

    with V^T M_A V = I.
    """

    mu: Array
    lambdas: Array
    eigenvectors_crystal: Array
    metric_orthonormality_residual: float
    eigen_equation_residual: float


@dataclass(frozen=True)
class CofactorMargins:
    lambdas: Array
    middle_eigenvector: Array
    cc1_lambda2_minus_one: float
    cc2_value: float
    cc2_product_diagnostic: float
    cc3_margin: float
    cc1_satisfied: bool
    cc2_satisfied: bool
    cc3_satisfied: bool
    all_satisfied: bool
    sampled_all_fraction_max_lambda2_residual: float
    sampled_fraction_count: int


@dataclass(frozen=True)
class RankOneBranch:
    branch: int
    rotation: Array
    a: Array
    n: Array
    eigenvalues_relative_cauchy_green: Array
    outer_product_residual: float
    rotation_orthogonality_residual: float
    rotation_determinant: float
    maximum_residual: float


@dataclass(frozen=True)
class StretchVariant:
    index: int
    U: Array
    generating_parent_symmetry_indices: tuple[int, ...]
    lambdas: Array
    determinant: float
    symmetry_residual: float


@dataclass(frozen=True)
class AusteniteMartensiteBranch:
    variant_index: int
    branch: int
    rotation: Array
    shape_strain: Array
    habit_normal_parent_cartesian: Array
    habit_plane_parent_crystal: Array
    shape_strain_parent_crystal: Array
    rank_one_residual: float
    rotation_residual: float
    rotation_determinant: float


@dataclass(frozen=True)
class MallardMatch:
    parent_symmetry_index: int
    kind: str
    rotation_residual_to_general: float
    outer_product_residual_to_general: float
    shear_relative_residual: float
    mallard_rank_one_residual: float


@dataclass(frozen=True)
class MartensiteTwinBranch:
    base_variant_index: int
    other_variant_index: int
    branch: int
    rotation: Array
    a: Array
    n_reference: Array
    deformed_twin_plane_normal: Array
    shear_magnitude: float
    rank_one_residual: float
    rotation_residual: float
    rotation_determinant: float
    cofactor: CofactorMargins
    mallard_matches: tuple[MallardMatch, ...]
    compound_by_multiple_twofolds: bool


@dataclass(frozen=True)
class BallJamesAudit:
    pulled_back_metric_residual: float
    metric_native_spectrum_residual: float
    maximum_parent_symmetry_so3_residual: float
    maximum_variant_spectrum_residual: float
    maximum_austenite_rank_one_residual: float
    maximum_martensite_rank_one_residual: float
    maximum_mallard_match_residual: float

    @property
    def maximum_residual(self) -> float:
        return max(
            self.pulled_back_metric_residual,
            self.metric_native_spectrum_residual,
            self.maximum_parent_symmetry_so3_residual,
            self.maximum_variant_spectrum_residual,
            self.maximum_austenite_rank_one_residual,
            self.maximum_martensite_rank_one_residual,
            self.maximum_mallard_match_residual,
        )


@dataclass(frozen=True)
class BallJamesReport:
    transformation_id: str
    parent_phase_id: str
    product_phase_id: str
    U0: Array
    metric_spectrum: MetricStretchSpectrum
    variants: tuple[StretchVariant, ...]
    austenite_martensite_branches: tuple[AusteniteMartensiteBranch, ...]
    martensite_twin_branches: tuple[MartensiteTwinBranch, ...]
    full_parent_symmetry_order: int
    proper_parent_symmetry_order: int
    full_product_symmetry_order: int
    proper_product_symmetry_order: int
    audit: BallJamesAudit
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation without changing raw values."""

        def convert(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, np.generic):
                return value.item()
            return value

        return convert(asdict(self))


def metric_native_stretch_spectrum(
    M_a: Array,
    M_m: Array,
    correspondence: Array,
) -> MetricStretchSpectrum:
    """Solve the generalized metric eigenproblem independently of CT."""

    parent = _validated_spd(M_a, name="M_a")
    product = _validated_spd(M_m, name="M_m")
    C = _validated_invertible(correspondence, name="correspondence")
    pulled = 0.5 * (C.T @ product @ C + (C.T @ product @ C).T)
    mu, vectors = eigh(pulled, parent)
    order = np.argsort(mu)
    mu = np.asarray(mu[order], dtype=float)
    vectors = np.asarray(vectors[:, order], dtype=float)
    if float(np.min(mu)) <= 0.0:
        raise ValueError(f"Generalized stretch eigenvalues must be positive; mu={mu}")
    scale = max(float(np.linalg.norm(pulled)), 1.0)
    equation = float(
        np.linalg.norm(pulled @ vectors - parent @ vectors @ np.diag(mu)) / scale
    )
    metric_orthogonality = float(
        np.linalg.norm(vectors.T @ parent @ vectors - np.eye(3), ord="fro")
    )
    return MetricStretchSpectrum(
        mu=mu,
        lambdas=np.sqrt(mu),
        eigenvectors_crystal=vectors,
        metric_orthonormality_residual=metric_orthogonality,
        eigen_equation_residual=equation,
    )


def stretch_from_crystallographic_state(
    M_a: Array,
    M_m: Array,
    correspondence: Array,
) -> tuple[Array, float]:
    r"""Construct the transformation stretch in the parent symmetric frame.

    .. math::

        U^2=M_A^{-1/2} C^T M_M C M_A^{-1/2}.

    Returns ``(U, residual)`` where the residual checks ``U^2`` against the
    independently assembled pulled-back product metric.
    """

    parent = _validated_spd(M_a, name="M_a")
    product = _validated_spd(M_m, name="M_m")
    C = _validated_invertible(correspondence, name="correspondence")
    W = _spd_inv_sqrt(parent)
    target = W @ (C.T @ product @ C) @ W
    target = 0.5 * (target + target.T)
    U = _spd_sqrt(target)
    residual = _relative_residual(U @ U, target)
    return U, residual


def parent_symmetry_rotations(
    M_a: Array,
    parent_symmetry_matrices: list[Array] | tuple[Array, ...],
    *,
    tolerance: float = 1.0e-9,
) -> tuple[tuple[int, Array], ...]:
    """Convert proper parent crystallographic symmetries to the U Cartesian frame."""

    parent = _validated_spd(M_a, name="M_a")
    B = _spd_sqrt(parent)
    B_inv = np.linalg.inv(B)
    output: list[tuple[int, Array]] = []
    for index, operation in enumerate(parent_symmetry_matrices):
        g = _validated_invertible(operation, name=f"parent symmetry[{index}]")
        metric_residual = _relative_residual(g.T @ parent @ g, parent)
        if metric_residual > tolerance:
            raise ValueError(
                f"parent symmetry[{index}] does not preserve M_a; "
                f"residual={metric_residual:.3e}"
            )
        if float(np.linalg.det(g)) <= 0.0:
            continue
        Q = B @ g @ B_inv
        maximum, _, determinant = _proper_rotation_residual(Q)
        if maximum > tolerance:
            raise ValueError(
                f"parent symmetry[{index}] is not SO(3) in the parent frame; "
                f"residual={maximum:.3e}, det={determinant:.12g}"
            )
        output.append((index, Q))
    if not output:
        raise ValueError("No proper parent symmetry rotations were supplied")
    return tuple(output)


def _principal_stretches(U: Array) -> tuple[Array, Array]:
    value = _validated_spd(U, name="U")
    lambdas, vectors = np.linalg.eigh(value)
    order = np.argsort(lambdas)
    return np.asarray(lambdas[order]), np.asarray(vectors[:, order])


def generate_stretch_variants(
    U0: Array,
    proper_parent_rotations: tuple[tuple[int, Array], ...],
    *,
    tolerance: float = 1.0e-9,
) -> tuple[StretchVariant, ...]:
    """Generate the distinct parent-symmetry orbit U_i = Q U_0 Q^T."""

    base = _validated_spd(U0, name="U0")
    matrices: list[Array] = []
    provenance: list[list[int]] = []
    residuals: list[float] = []
    for symmetry_index, rotation in proper_parent_rotations:
        Q = np.asarray(rotation, dtype=float).reshape(3, 3)
        candidate = Q @ base @ Q.T
        candidate = 0.5 * (candidate + candidate.T)
        match = next(
            (
                index
                for index, existing in enumerate(matrices)
                if np.linalg.norm(candidate - existing, ord="fro") <= tolerance
            ),
            None,
        )
        if match is None:
            matrices.append(candidate)
            provenance.append([int(symmetry_index)])
            residuals.append(_relative_residual(candidate, Q @ base @ Q.T))
        else:
            provenance[match].append(int(symmetry_index))

    base_lambdas, _ = _principal_stretches(base)
    variants: list[StretchVariant] = []
    for index, matrix in enumerate(matrices):
        lambdas, _ = _principal_stretches(matrix)
        spectrum_residual = float(np.max(np.abs(lambdas - base_lambdas)))
        variants.append(
            StretchVariant(
                index=index,
                U=matrix,
                generating_parent_symmetry_indices=tuple(provenance[index]),
                lambdas=lambdas,
                determinant=float(np.linalg.det(matrix)),
                symmetry_residual=max(residuals[index], spectrum_residual),
            )
        )
    return tuple(variants)


def analytical_rank_one_connections(
    F_other: Array,
    F_base: Array,
    *,
    eigenvalue_tolerance: float = 1.0e-8,
) -> tuple[RankOneBranch, ...]:
    r"""Solve ``R F_other - F_base = a \\otimes n`` using Proposition 1.

    Set ``A=F_other F_base^{-1}`` and ``C=A^T A``.  For non-trivial C,
    necessary and sufficient compatibility is ``mu_1>0`` and ``mu_2=1``.
    Both kappa=±1 solutions are returned.
    """

    other = _validated_invertible(F_other, name="F_other")
    base = _validated_invertible(F_base, name="F_base")
    if float(np.linalg.det(other)) <= 0.0 or float(np.linalg.det(base)) <= 0.0:
        raise ValueError(
            "Ball-James deformation gradients must be orientation preserving"
        )

    A = other @ np.linalg.inv(base)
    relative = 0.5 * (A.T @ A + (A.T @ A).T)
    mu, vectors = np.linalg.eigh(relative)
    order = np.argsort(mu)
    mu = np.asarray(mu[order], dtype=float)
    vectors = np.asarray(vectors[:, order], dtype=float)
    mu1, mu2, mu3 = (float(value) for value in mu)

    if mu1 <= 0.0 or abs(mu2 - 1.0) > eigenvalue_tolerance:
        return ()
    if np.linalg.norm(relative - np.eye(3), ord="fro") <= eigenvalue_tolerance:
        return ()
    if mu3 - mu1 <= eigenvalue_tolerance:
        return ()

    # The ordering and mu2≈1 imply mu1<=1<=mu3 up to tolerance.  Clamp tiny
    # negative round-off only after explicitly checking the theoretical signs.
    if mu1 > 1.0 + eigenvalue_tolerance or mu3 < 1.0 - eigenvalue_tolerance:
        return ()
    one_minus_mu1 = max(0.0, 1.0 - mu1)
    mu3_minus_one = max(0.0, mu3 - 1.0)
    denominator = float(np.sqrt(mu3 - mu1))
    e1 = vectors[:, 0]
    e3 = vectors[:, 2]

    branches: list[RankOneBranch] = []
    for branch in (-1, 1):
        b = (
            np.sqrt(mu3 * one_minus_mu1) * e1
            + branch * np.sqrt(mu1 * mu3_minus_one) * e3
        ) / denominator
        m = (
            (np.sqrt(mu3) - np.sqrt(mu1))
            / denominator
            * (-np.sqrt(one_minus_mu1) * e1 + branch * np.sqrt(mu3_minus_one) * e3)
        )
        R = (np.eye(3) + np.outer(b, m)) @ np.linalg.inv(A)
        n = base.T @ m
        n_norm = float(np.linalg.norm(n))
        if n_norm <= 1.0e-15:
            raise AssertionError(
                "Proposition-1 branch produced a zero interface normal"
            )
        n_unit = n / n_norm
        a_scaled = b * n_norm

        difference = R @ other - base
        outer = np.outer(a_scaled, n_unit)
        outer_residual = _relative_residual(difference, outer)
        maximum_rotation, orthogonality, determinant = _proper_rotation_residual(R)
        maximum = max(outer_residual, maximum_rotation)
        branches.append(
            RankOneBranch(
                branch=branch,
                rotation=R,
                a=a_scaled,
                n=n_unit,
                eigenvalues_relative_cauchy_green=mu.copy(),
                outer_product_residual=outer_residual,
                rotation_orthogonality_residual=orthogonality,
                rotation_determinant=determinant,
                maximum_residual=maximum,
            )
        )
    return tuple(branches)


def evaluate_cofactor_conditions(
    U_base: Array,
    a: Array,
    n: Array,
    *,
    tolerance: float = 1.0e-8,
    fraction_samples: int = 101,
) -> CofactorMargins:
    r"""Evaluate Chen--Srivastava--Dabade--James CC1--CC3.

    CC1: lambda_2 - 1 = 0
    CC2: a . U cof(U^2-I) n = 0
    CC3: tr(U^2) - det(U^2) - |a|^2 |n|^2 / 4 - 2 >= 0

    The minus sign in CC3 is the sign in Theorem 2 and its proof in the 2013
    paper.  The abstract contains an inconsistent plus sign and is not used.
    """

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if fraction_samples < 2:
        raise ValueError("fraction_samples must be >= 2")

    U = _validated_spd(U_base, name="U_base")
    shear = np.asarray(a, dtype=float).reshape(3)
    normal = np.asarray(n, dtype=float).reshape(3)
    lambdas, vectors = _principal_stretches(U)
    middle = vectors[:, 1]
    cc1 = float(lambdas[1] - 1.0)
    A = U @ U - np.eye(3)
    cc2 = float(shear @ U @ _cofactor_matrix(A) @ normal)
    cc2_product = float((shear @ middle) * (normal @ middle))
    U2 = U @ U
    cc3 = float(
        np.trace(U2)
        - np.linalg.det(U2)
        - 0.25 * float(shear @ shear) * float(normal @ normal)
        - 2.0
    )

    sampled: list[float] = []
    for fraction in np.linspace(0.0, 1.0, fraction_samples):
        laminate = U + float(fraction) * np.outer(shear, normal)
        singular_values = np.sort(np.linalg.svd(laminate, compute_uv=False))
        sampled.append(abs(float(singular_values[1] - 1.0)))

    c1 = abs(cc1) <= tolerance
    c2 = abs(cc2) <= tolerance
    c3 = cc3 >= -tolerance
    return CofactorMargins(
        lambdas=lambdas,
        middle_eigenvector=middle,
        cc1_lambda2_minus_one=cc1,
        cc2_value=cc2,
        cc2_product_diagnostic=cc2_product,
        cc3_margin=cc3,
        cc1_satisfied=c1,
        cc2_satisfied=c2,
        cc3_satisfied=c3,
        all_satisfied=bool(c1 and c2 and c3),
        sampled_all_fraction_max_lambda2_residual=max(sampled),
        sampled_fraction_count=fraction_samples,
    )


def _twofold_axis(rotation: Array, *, tolerance: float) -> Array | None:
    Q = np.asarray(rotation, dtype=float).reshape(3, 3)
    maximum, _, determinant = _proper_rotation_residual(Q)
    if maximum > tolerance or abs(determinant - 1.0) > tolerance:
        return None
    if np.linalg.norm(Q @ Q - np.eye(3), ord="fro") > 10.0 * tolerance:
        return None
    if abs(float(np.trace(Q)) + 1.0) > 10.0 * tolerance:
        return None
    _, _, vh = np.linalg.svd(Q - np.eye(3))
    return _unit(vh[-1], name="twofold axis")


def mallard_law_twins(
    U_base: Array,
    U_other: Array,
    twofold_rotation: Array,
    *,
    tolerance: float = 1.0e-8,
) -> tuple[tuple[str, RankOneBranch, float], ...]:
    """Return the Type-I/II Mallard solutions and physical twin shear.

    The equation is ``R U_other - U_base = a \\otimes n``.  ``U_other`` must
    equal ``Q U_base Q^T`` for the supplied parent 180-degree rotation Q.
    """

    base = _validated_spd(U_base, name="U_base")
    other = _validated_spd(U_other, name="U_other")
    Q = np.asarray(twofold_rotation, dtype=float).reshape(3, 3)
    axis = _twofold_axis(Q, tolerance=tolerance)
    if axis is None:
        raise ValueError("Mallard law requires a proper parent 180-degree rotation")
    relation_residual = _relative_residual(other, Q @ base @ Q.T)
    if relation_residual > tolerance:
        raise ValueError(
            "U_other is not related to U_base by the supplied twofold rotation; "
            f"residual={relation_residual:.3e}"
        )

    # James & Hane Eqs. (16)--(17), with U_j = U_base.
    x = np.linalg.solve(base, axis)
    x2 = float(x @ x)
    a_i = 2.0 * (x / x2 - base @ axis)
    n_i = axis.copy()
    r_i = (-np.eye(3) + 2.0 * np.outer(x, x) / x2) @ Q

    y = base @ axis
    y2 = float(y @ y)
    n0 = axis - (base @ base @ axis) / y2
    n0_norm = float(np.linalg.norm(n0))

    raw: list[tuple[str, Array, Array, Array]] = [("I", r_i, a_i, n_i)]
    if n0_norm > tolerance:
        n_ii = n0 / n0_norm
        a_ii = 2.0 * n0_norm * y
        r_ii = (-np.eye(3) + 2.0 * np.outer(y, y) / y2) @ Q
        raw.append(("II", r_ii, a_ii, n_ii))

    output: list[tuple[str, RankOneBranch, float]] = []
    for kind, rotation, a_vector, normal in raw:
        normal = _unit(normal, name=f"Mallard {kind} normal")
        difference = rotation @ other - base
        outer = np.outer(a_vector, normal)
        outer_residual = _relative_residual(difference, outer)
        maximum_rotation, orthogonality, determinant = _proper_rotation_residual(
            rotation
        )
        rank_branch = RankOneBranch(
            branch=0,
            rotation=rotation,
            a=a_vector,
            n=normal,
            eigenvalues_relative_cauchy_green=np.linalg.eigvalsh(
                (other @ np.linalg.inv(base)).T @ (other @ np.linalg.inv(base))
            ),
            outer_product_residual=outer_residual,
            rotation_orthogonality_residual=orthogonality,
            rotation_determinant=determinant,
            maximum_residual=max(outer_residual, maximum_rotation),
        )
        shear = float(
            np.linalg.norm(np.linalg.solve(base, normal)) * np.linalg.norm(a_vector)
        )
        output.append((kind, rank_branch, shear))
    return tuple(output)


def _match_variant(
    U: Array, variants: tuple[StretchVariant, ...], tolerance: float
) -> int:
    residuals = [float(np.linalg.norm(U - item.U, ord="fro")) for item in variants]
    index = int(np.argmin(residuals))
    if residuals[index] > tolerance:
        raise AssertionError(
            "Parent-symmetry image of a registered stretch did not map back to "
            f"the generated variant family; best residual={residuals[index]:.3e}"
        )
    return index


def _build_mallard_index(
    variants: tuple[StretchVariant, ...],
    proper_parent_rotations: tuple[tuple[int, Array], ...],
    *,
    tolerance: float,
) -> dict[tuple[int, int], list[tuple[int, str, RankOneBranch, float]]]:
    index: dict[tuple[int, int], list[tuple[int, str, RankOneBranch, float]]] = {}
    for base_variant in variants:
        for symmetry_index, Q in proper_parent_rotations:
            if _twofold_axis(Q, tolerance=tolerance) is None:
                continue
            transformed = Q @ base_variant.U @ Q.T
            other_index = _match_variant(transformed, variants, 100.0 * tolerance)
            if other_index == base_variant.index:
                continue
            solutions = mallard_law_twins(
                base_variant.U,
                variants[other_index].U,
                Q,
                tolerance=100.0 * tolerance,
            )
            for kind, branch, shear in solutions:
                index.setdefault((base_variant.index, other_index), []).append(
                    (symmetry_index, kind, branch, shear)
                )
    return index


def analyze_ball_james(
    M_a: Array,
    M_m: Array,
    correspondence: Array,
    parent_symmetry_matrices: list[Array] | tuple[Array, ...],
    *,
    transformation_id: str = "",
    parent_phase_id: str = "",
    product_phase_id: str = "",
    full_product_symmetry_order: int = 0,
    proper_product_symmetry_order: int = 0,
    algebraic_tolerance: float = 1.0e-10,
    eigenvalue_tolerance: float = 1.0e-8,
    rank_one_tolerance: float = 1.0e-8,
    cofactor_tolerance: float = 1.0e-8,
    fraction_samples: int = 101,
) -> BallJamesReport:
    """Run the complete independent Ball--James/cofactor analysis."""

    for name, value in {
        "algebraic_tolerance": algebraic_tolerance,
        "eigenvalue_tolerance": eigenvalue_tolerance,
        "rank_one_tolerance": rank_one_tolerance,
        "cofactor_tolerance": cofactor_tolerance,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")

    parent = _validated_spd(M_a, name="M_a")
    product = _validated_spd(M_m, name="M_m")
    C = _validated_invertible(correspondence, name="correspondence")
    metric_spectrum = metric_native_stretch_spectrum(parent, product, C)
    U0, pullback_residual = stretch_from_crystallographic_state(parent, product, C)
    u_lambdas, _ = _principal_stretches(U0)
    spectrum_residual = float(np.max(np.abs(u_lambdas - metric_spectrum.lambdas)))

    proper_rotations = parent_symmetry_rotations(
        parent,
        parent_symmetry_matrices,
        tolerance=max(algebraic_tolerance, 1.0e-9),
    )
    variants = generate_stretch_variants(
        U0,
        proper_rotations,
        tolerance=max(algebraic_tolerance, 1.0e-9),
    )
    B_parent = _spd_sqrt(parent)
    B_parent_inv = np.linalg.inv(B_parent)

    am_branches: list[AusteniteMartensiteBranch] = []
    for variant in variants:
        branches = analytical_rank_one_connections(
            variant.U,
            np.eye(3),
            eigenvalue_tolerance=eigenvalue_tolerance,
        )
        for branch in branches:
            habit_plane_crystal = B_parent @ branch.n
            shape_strain_crystal = B_parent_inv @ branch.a
            am_branches.append(
                AusteniteMartensiteBranch(
                    variant_index=variant.index,
                    branch=branch.branch,
                    rotation=branch.rotation,
                    shape_strain=branch.a,
                    habit_normal_parent_cartesian=branch.n,
                    habit_plane_parent_crystal=habit_plane_crystal,
                    shape_strain_parent_crystal=shape_strain_crystal,
                    rank_one_residual=branch.outer_product_residual,
                    rotation_residual=max(
                        branch.rotation_orthogonality_residual,
                        abs(branch.rotation_determinant - 1.0),
                    ),
                    rotation_determinant=branch.rotation_determinant,
                )
            )

    mallard_index = _build_mallard_index(
        variants,
        proper_rotations,
        tolerance=max(rank_one_tolerance, 1.0e-9),
    )

    twin_branches: list[MartensiteTwinBranch] = []
    for base_variant in variants:
        for other_variant in variants:
            if base_variant.index == other_variant.index:
                continue
            general = analytical_rank_one_connections(
                other_variant.U,
                base_variant.U,
                eigenvalue_tolerance=eigenvalue_tolerance,
            )
            routes = mallard_index.get((base_variant.index, other_variant.index), [])
            for branch in general:
                deformed_normal = _unit(
                    np.linalg.solve(base_variant.U, branch.n),
                    name="deformed twin-plane normal",
                )
                shear = float(
                    np.linalg.norm(np.linalg.solve(base_variant.U, branch.n))
                    * np.linalg.norm(branch.a)
                )
                cofactor = evaluate_cofactor_conditions(
                    base_variant.U,
                    branch.a,
                    branch.n,
                    tolerance=cofactor_tolerance,
                    fraction_samples=fraction_samples,
                )

                matches: list[MallardMatch] = []
                for symmetry_index, kind, mallard_branch, mallard_shear in routes:
                    rotation_residual = _relative_residual(
                        branch.rotation,
                        mallard_branch.rotation,
                    )
                    outer_residual = _relative_residual(
                        np.outer(branch.a, branch.n),
                        np.outer(mallard_branch.a, mallard_branch.n),
                    )
                    shear_residual = abs(shear - mallard_shear) / max(
                        abs(shear), abs(mallard_shear), 1.0e-15
                    )
                    if (
                        min(rotation_residual, outer_residual)
                        <= 100.0 * rank_one_tolerance
                    ):
                        matches.append(
                            MallardMatch(
                                parent_symmetry_index=symmetry_index,
                                kind=kind,
                                rotation_residual_to_general=rotation_residual,
                                outer_product_residual_to_general=outer_residual,
                                shear_relative_residual=float(shear_residual),
                                mallard_rank_one_residual=mallard_branch.maximum_residual,
                            )
                        )

                unique_twofolds = {match.parent_symmetry_index for match in matches}
                twin_branches.append(
                    MartensiteTwinBranch(
                        base_variant_index=base_variant.index,
                        other_variant_index=other_variant.index,
                        branch=branch.branch,
                        rotation=branch.rotation,
                        a=branch.a,
                        n_reference=branch.n,
                        deformed_twin_plane_normal=deformed_normal,
                        shear_magnitude=shear,
                        rank_one_residual=branch.outer_product_residual,
                        rotation_residual=max(
                            branch.rotation_orthogonality_residual,
                            abs(branch.rotation_determinant - 1.0),
                        ),
                        rotation_determinant=branch.rotation_determinant,
                        cofactor=cofactor,
                        mallard_matches=tuple(matches),
                        compound_by_multiple_twofolds=len(unique_twofolds) > 1,
                    )
                )

    symmetry_residuals = [
        _proper_rotation_residual(rotation)[0] for _, rotation in proper_rotations
    ]
    variant_residuals = [variant.symmetry_residual for variant in variants]
    am_residuals = [
        max(branch.rank_one_residual, branch.rotation_residual)
        for branch in am_branches
    ]
    twin_residuals = [
        max(branch.rank_one_residual, branch.rotation_residual)
        for branch in twin_branches
    ]
    mallard_residuals = [
        max(
            match.rotation_residual_to_general,
            match.outer_product_residual_to_general,
            match.shear_relative_residual,
            match.mallard_rank_one_residual,
        )
        for twin in twin_branches
        for match in twin.mallard_matches
    ]

    audit = BallJamesAudit(
        pulled_back_metric_residual=pullback_residual,
        metric_native_spectrum_residual=spectrum_residual,
        maximum_parent_symmetry_so3_residual=max(symmetry_residuals, default=0.0),
        maximum_variant_spectrum_residual=max(variant_residuals, default=0.0),
        maximum_austenite_rank_one_residual=max(am_residuals, default=0.0),
        maximum_martensite_rank_one_residual=max(twin_residuals, default=0.0),
        maximum_mallard_match_residual=max(mallard_residuals, default=0.0),
    )

    notes = (
        "This report is generated independently from Cayron CT/CMC/SMC/PTMC.",
        (
            "No absence of an exact A/M branch should be interpreted as experimental "
            "incompatibility without considering lattice-parameter uncertainty."
        ),
        (
            "M/M compatibility is solved for every ordered distinct stretch-variant pair; "
            "Mallard matches are an independent symmetry-based cross-check, not the generator "
            "of the general rank-one solutions."
        ),
        (
            "Cofactor CC3 uses the minus-det sign in Chen et al. 2013 Theorem 2/proof; "
            "the paper abstract contains an inconsistent plus sign."
        ),
    )

    return BallJamesReport(
        transformation_id=transformation_id,
        parent_phase_id=parent_phase_id,
        product_phase_id=product_phase_id,
        U0=U0,
        metric_spectrum=metric_spectrum,
        variants=variants,
        austenite_martensite_branches=tuple(am_branches),
        martensite_twin_branches=tuple(twin_branches),
        full_parent_symmetry_order=len(parent_symmetry_matrices),
        proper_parent_symmetry_order=len(proper_rotations),
        full_product_symmetry_order=int(full_product_symmetry_order),
        proper_product_symmetry_order=int(proper_product_symmetry_order),
        audit=audit,
        notes=notes,
    )


@dataclass(frozen=True)
class BallJamesCrystalInput:
    """PTCLab-style crystal/transformation input for Ball--James analysis.

    Users provide the two physical crystal cells, crystallographic point-group
    symbols, and the lattice correspondence.  Metrics, full symmetry matrices,
    proper SO(3) subgroups, stretches and variant families are derived
    internally.  This keeps the public input close to PTCLab while preserving
    the rigorous metric-native backend.

    Nonstandard point-group settings should continue to use :class:`ProjectState`
    with explicit symmetry matrices; this convenience class intentionally uses
    only the conventional settings in :mod:`cualni_cryst.point_groups`.
    """

    parent_lattice: Lattice
    product_lattice: Lattice
    correspondence: Correspondence | Array
    parent_point_group: str
    product_point_group: str
    transformation_id: str = "manual_ball_james"
    parent_phase_id: str = "parent"
    product_phase_id: str = "product"

    @staticmethod
    def _correspondence_array(value: Correspondence | Array) -> Array:
        if isinstance(value, Correspondence):
            return np.asarray(value.C_M_from_A, dtype=float)
        return _validated_invertible(value, name="correspondence")

    @staticmethod
    def _validated_operations(
        symbol: str,
        metric: Array,
        *,
        tolerance: float,
    ) -> tuple[Array, ...]:
        operations = tuple(
            np.asarray(operation, dtype=float)
            for operation in point_group_operations(symbol)
        )
        scale = max(float(np.linalg.norm(metric)), 1.0)
        residuals = [
            float(np.linalg.norm(operation.T @ metric @ operation - metric) / scale)
            for operation in operations
        ]
        maximum = max(residuals, default=0.0)
        if maximum > tolerance:
            raise ValueError(
                f"Point group {symbol!r} is incompatible with the supplied lattice "
                f"metric in its conventional setting; maximum residual={maximum:.3e}. "
                "For a nonstandard setting, use ProjectState with explicit symmetry matrices."
            )
        return operations

    def analyze(
        self,
        *,
        numerical_policy: NumericalPolicy = DEFAULT_NUMERICAL_POLICY,
        fraction_samples: int = 101,
    ) -> BallJamesReport:
        """Analyze the supplied crystal pair without exposing backend matrices."""

        parent_metric = self.parent_lattice.metric()
        product_metric = self.product_lattice.metric()
        parent_symmetry = self._validated_operations(
            self.parent_point_group,
            parent_metric,
            tolerance=numerical_policy.representation,
        )
        product_symmetry = self._validated_operations(
            self.product_point_group,
            product_metric,
            tolerance=numerical_policy.representation,
        )
        proper_product = sum(
            float(np.linalg.det(operation)) > 0.0 for operation in product_symmetry
        )
        return analyze_ball_james(
            parent_metric,
            product_metric,
            self._correspondence_array(self.correspondence),
            parent_symmetry,
            transformation_id=self.transformation_id,
            parent_phase_id=self.parent_phase_id,
            product_phase_id=self.product_phase_id,
            full_product_symmetry_order=len(product_symmetry),
            proper_product_symmetry_order=proper_product,
            algebraic_tolerance=numerical_policy.algebraic,
            eigenvalue_tolerance=numerical_policy.exact_eigenvalue,
            rank_one_tolerance=numerical_policy.rank_one,
            cofactor_tolerance=numerical_policy.exact_eigenvalue,
            fraction_samples=fraction_samples,
        )


def analyze_ball_james_crystals(
    parent_lattice: Lattice,
    product_lattice: Lattice,
    correspondence: Correspondence | Array,
    *,
    parent_point_group: str,
    product_point_group: str,
    numerical_policy: NumericalPolicy = DEFAULT_NUMERICAL_POLICY,
    fraction_samples: int = 101,
    transformation_id: str = "manual_ball_james",
    parent_phase_id: str = "parent",
    product_phase_id: str = "product",
) -> BallJamesReport:
    """Convenience API matching the crystal-first workflow used by PTCLab."""

    return BallJamesCrystalInput(
        parent_lattice=parent_lattice,
        product_lattice=product_lattice,
        correspondence=correspondence,
        parent_point_group=parent_point_group,
        product_point_group=product_point_group,
        transformation_id=transformation_id,
        parent_phase_id=parent_phase_id,
        product_phase_id=product_phase_id,
    ).analyze(
        numerical_policy=numerical_policy,
        fraction_samples=fraction_samples,
    )


class BallJamesAdapter:
    """Project-bound independent nonlinear-elasticity adapter.

    The adapter intentionally imports project/representation classes only when
    instantiated, keeping the numerical engine above usable as a standalone
    scientific kernel.
    """

    def __init__(self, project: Any, transformation_id: str) -> None:
        project.validate().assert_passed()
        self.project = project
        self.transformation_id = transformation_id
        self.transformation = project.transformation(transformation_id)
        self.parent = project.phase(self.transformation.parent_phase_id)
        self.product = project.phase(self.transformation.product_phase_id)

    def analyze(self, *, fraction_samples: int = 101) -> BallJamesReport:
        policy = self.project.numerical_policy
        parent_symmetry = tuple(self.parent.symmetry_matrices())
        if not parent_symmetry:
            raise ValueError(
                "Ball-James variant analysis requires explicit parent symmetry operators; "
                "identity-only symmetry must be supplied explicitly if that is intended."
            )
        product_symmetry = tuple(self.product.symmetry_matrices())
        proper_product = sum(
            float(np.linalg.det(operation)) > 0.0 for operation in product_symmetry
        )
        C = np.asarray(self.transformation.correspondence.C_M_from_A, dtype=float)
        return analyze_ball_james(
            self.parent.lattice.metric(),
            self.product.lattice.metric(),
            C,
            parent_symmetry,
            transformation_id=self.transformation_id,
            parent_phase_id=self.parent.phase_id,
            product_phase_id=self.product.phase_id,
            full_product_symmetry_order=len(product_symmetry),
            proper_product_symmetry_order=proper_product,
            algebraic_tolerance=policy.algebraic,
            eigenvalue_tolerance=policy.exact_eigenvalue,
            rank_one_tolerance=policy.rank_one,
            cofactor_tolerance=policy.exact_eigenvalue,
            fraction_samples=fraction_samples,
        )
