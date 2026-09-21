from __future__ import annotations

"""Physical-equivalence and independent-theory validation for CT.

This module is intentionally a *validation layer*.  It does not alter the
Correspondence Theory (CT), Ball-James, Mallard, groupoid, or stretch equations.

Two questions are answered independently:

1. Coordinate covariance:
   If parent and product crystal bases are replaced by exact unimodular bases,
   does CT predict the same physical operators, planes, directions,
   classifications, intercorrespondences and shears after pull-back?

2. Theory cross-validation:
   For every non-trivial twofold-related stretch pair, do:
       - Cayron CT,
       - Mallard's law, and
       - the generic Ball-James rank-one solver
   independently recover the same physical rank-one relation?

Conventions
-----------
A crystallographic basis change is written

    B'_A = B_A P_A,       B'_M = B_M P_M,

so direct coordinates transform as

    u'_A = P_A^{-1} u_A,  u'_M = P_M^{-1} u_M,

plane covectors as

    p'_A = P_A^T p_A,     p'_M = P_M^T p_M,

metrics as

    M'_A = P_A^T M_A P_A,  M'_M = P_M^T M_M P_M,

symmetry matrices as

    g'_A = P_A^{-1} g_A P_A,

and the package correspondence ``u_M = C_M_from_A u_A`` as

    C' = P_M^{-1} C P_A.

All discrete group comparisons below remain exact SymPy comparisons. Floating
arithmetic is used only for metric geometry / rank-one residuals.
"""

from dataclasses import dataclass
from itertools import permutations

import numpy as np
import sympy as sp

from .ball_james import analytical_rank_one_connections, mallard_law_twins
from .correspondence import Correspondence
from .group_theory import correspondence_groupoid
from .lattice import (
    metric_sqrt,
    plane_to_unit_normal,
    transform_matrix_to_metric_orthonormal,
)
from .stretch import stretch_from_metrics
from .symmetry import matrix_key
from .twinning_ct import classify_parent_order_two_isometry, twins_from_operator


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-15:
        raise ValueError("Cannot normalize a zero vector")
    return value / norm


def _as_spd_metric(metric: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(metric, dtype=float)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    value = 0.5 * (value + value.T)
    eig = np.linalg.eigvalsh(value)
    if float(np.min(eig)) <= 0.0:
        raise ValueError(f"{name} must be positive definite; eigenvalues={eig}")
    return value


def _as_unimodular_basis(P: sp.Matrix, *, name: str) -> sp.Matrix:
    matrix = sp.Matrix(P)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must be 3x3")
    if any(entry.is_integer is not True for entry in matrix):
        raise ValueError(f"{name} must have exact integer entries")
    determinant = sp.simplify(matrix.det())
    if determinant not in (sp.Integer(1), sp.Integer(-1)):
        raise ValueError(
            f"{name} must be unimodular (det = +/-1); det={determinant}"
        )
    return matrix


def metric_projective_residual(
    lhs: np.ndarray,
    rhs: np.ndarray,
    metric: np.ndarray,
    *,
    reciprocal: bool,
) -> float:
    """Stable projective chord residual in a direct/reciprocal metric.

    ``v`` and ``-v`` are the same crystallographic line/plane normal.  The
    vectors are embedded with the appropriate positive-definite Gram matrix,
    normalized, then compared by ``min(||u-v||, ||u+v||)``.

    Unlike ``sqrt(1-cos(theta)^2)``, this remains well conditioned as
    ``theta -> 0``.
    """

    M = _as_spd_metric(metric, name="metric")
    G = np.linalg.solve(M, np.eye(3)) if reciprocal else M
    G = 0.5 * (G + G.T)
    L = np.linalg.cholesky(G)

    def embedded_unit(value: np.ndarray) -> np.ndarray:
        physical = L.T @ np.asarray(value, dtype=float).reshape(3)
        return _unit(physical)

    a = embedded_unit(lhs)
    b = embedded_unit(rhs)
    return float(min(np.linalg.norm(a - b), np.linalg.norm(a + b)))


def projective_angle_deg(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Acute projective angle in an orthonormal physical frame."""

    a = _unit(lhs)
    b = _unit(rhs)
    cross = float(np.linalg.norm(np.cross(a, b)))
    dot = abs(float(a @ b))
    return float(np.degrees(np.arctan2(cross, dot)))


def _matrix_relative_residual(lhs: np.ndarray, rhs: np.ndarray) -> float:
    left = np.asarray(lhs, dtype=float)
    right = np.asarray(rhs, dtype=float)
    scale = max(float(np.linalg.norm(left, ord="fro")),
                float(np.linalg.norm(right, ord="fro")), 1.0)
    return float(np.linalg.norm(left - right, ord="fro") / scale)


def _family_signature(family: list[sp.Matrix] | tuple[sp.Matrix, ...]) -> frozenset:
    return frozenset(matrix_key(sp.Matrix(item)) for item in family)


def _pulled_family_signature(
    family: list[sp.Matrix] | tuple[sp.Matrix, ...],
    P: sp.Matrix,
) -> frozenset:
    Pi = P.inv()
    return frozenset(
        matrix_key(sp.simplify(P * sp.Matrix(item) * Pi))
        for item in family
    )


@dataclass(frozen=True)
class BasisTransformedState:
    parent_metric: np.ndarray
    product_metric: np.ndarray
    parent_group: tuple[sp.Matrix, ...]
    product_group: tuple[sp.Matrix, ...]
    correspondence: Correspondence
    P_A: sp.Matrix
    P_M: sp.Matrix


def transform_crystallographic_state(
    M_a: np.ndarray,
    M_m: np.ndarray,
    parent_group: list[sp.Matrix] | tuple[sp.Matrix, ...],
    product_group: list[sp.Matrix] | tuple[sp.Matrix, ...],
    correspondence: Correspondence,
    P_A: sp.Matrix,
    P_M: sp.Matrix,
) -> BasisTransformedState:
    """Re-express one physical transformation in exact new lattice bases."""

    M_a = _as_spd_metric(M_a, name="M_a")
    M_m = _as_spd_metric(M_m, name="M_m")
    P_A = _as_unimodular_basis(P_A, name="P_A")
    P_M = _as_unimodular_basis(P_M, name="P_M")

    Paf = np.asarray(P_A, dtype=float)
    Pmf = np.asarray(P_M, dtype=float)

    M_a_prime = Paf.T @ M_a @ Paf
    M_m_prime = Pmf.T @ M_m @ Pmf

    P_ai = P_A.inv()
    P_mi = P_M.inv()
    parent_prime = tuple(
        sp.simplify(P_ai * sp.Matrix(g) * P_A) for g in parent_group
    )
    product_prime = tuple(
        sp.simplify(P_mi * sp.Matrix(g) * P_M) for g in product_group
    )
    C_prime = Correspondence(
        sp.simplify(P_mi * correspondence.C_M_from_A * P_A),
        label=f"{correspondence.label} [basis transformed]",
        source=correspondence.source,
        derivation=correspondence.derivation,
    )

    return BasisTransformedState(
        parent_metric=M_a_prime,
        product_metric=M_m_prime,
        parent_group=parent_prime,
        product_group=product_prime,
        correspondence=C_prime,
        P_A=P_A,
        P_M=P_M,
    )


@dataclass(frozen=True)
class TwinCovarianceResidual:
    kind: str
    classification: str
    classification_match: bool
    representations_match: bool
    shear_abs_residual: float
    shear_rel_residual: float
    parent_plane_residual: float
    parent_direction_residual: float
    product_plane_residual: float
    product_direction_residual: float
    intercorrespondence_residual: float

    @property
    def max_geometry_residual(self) -> float:
        return max(
            self.parent_plane_residual,
            self.parent_direction_residual,
            self.product_plane_residual,
            self.product_direction_residual,
        )


@dataclass(frozen=True)
class PhysicalEquivalenceReport:
    subgroup_exact: bool
    variant_partition_exact: bool
    operator_partition_exact: bool
    adjacency_exact: bool
    twin_provenance_exact: bool
    twin_residuals: tuple[TwinCovarianceResidual, ...]
    geometry_tolerance: float
    shear_relative_tolerance: float
    matrix_tolerance: float

    @property
    def max_geometry_residual(self) -> float:
        return max(
            (item.max_geometry_residual for item in self.twin_residuals),
            default=0.0,
        )

    @property
    def max_shear_relative_residual(self) -> float:
        return max(
            (item.shear_rel_residual for item in self.twin_residuals),
            default=0.0,
        )

    @property
    def max_intercorrespondence_residual(self) -> float:
        return max(
            (item.intercorrespondence_residual for item in self.twin_residuals),
            default=0.0,
        )

    @property
    def classification_exact(self) -> bool:
        return all(
            item.classification_match and item.representations_match
            for item in self.twin_residuals
        )

    @property
    def success(self) -> bool:
        return (
            self.subgroup_exact
            and self.variant_partition_exact
            and self.operator_partition_exact
            and self.adjacency_exact
            and self.twin_provenance_exact
            and self.classification_exact
            and self.max_geometry_residual <= self.geometry_tolerance
            and self.max_shear_relative_residual <= self.shear_relative_tolerance
            and self.max_intercorrespondence_residual <= self.matrix_tolerance
        )


def validate_ct_basis_covariance(
    M_a: np.ndarray,
    M_m: np.ndarray,
    parent_group: list[sp.Matrix] | tuple[sp.Matrix, ...],
    product_group: list[sp.Matrix] | tuple[sp.Matrix, ...],
    correspondence: Correspondence,
    P_A: sp.Matrix,
    P_M: sp.Matrix,
    *,
    geometry_tolerance: float = 2.0e-9,
    shear_relative_tolerance: float = 2.0e-10,
    matrix_tolerance: float = 2.0e-10,
) -> PhysicalEquivalenceReport:
    """Validate complete CT physical equivalence under exact basis changes.

    This is deliberately stronger than comparing counts and shears.  It checks:

    - exact correspondence subgroup after pull-back;
    - exact variant-coset partition after pull-back;
    - exact double-coset/operator partition after pull-back;
    - the complete variant-to-variant operator adjacency after relabelling by
      exact physical operator signatures;
    - every CT twin's exact parent-symmetry provenance;
    - Type-I/Type-II/compound classification;
    - parent/product planes and directions after covariant pull-back;
    - shear magnitude;
    - martensite-space intercorrespondence.
    """

    M_a = _as_spd_metric(M_a, name="M_a")
    M_m = _as_spd_metric(M_m, name="M_m")
    transformed = transform_crystallographic_state(
        M_a,
        M_m,
        parent_group,
        product_group,
        correspondence,
        P_A,
        P_M,
    )

    base = correspondence_groupoid(
        list(parent_group), list(product_group), correspondence
    )
    prime = correspondence_groupoid(
        list(transformed.parent_group),
        list(transformed.product_group),
        transformed.correspondence,
    )

    subgroup_exact = (
        _family_signature(base.subgroup)
        == _pulled_family_signature(prime.subgroup, transformed.P_A)
    )

    base_variant_sigs = [_family_signature(item) for item in base.variants]
    prime_variant_sigs = [
        _pulled_family_signature(item, transformed.P_A)
        for item in prime.variants
    ]
    variant_partition_exact = set(base_variant_sigs) == set(prime_variant_sigs)

    base_operator_sigs = [_family_signature(item) for item in base.operators]
    prime_operator_sigs = [
        _pulled_family_signature(item, transformed.P_A)
        for item in prime.operators
    ]
    operator_partition_exact = set(base_operator_sigs) == set(prime_operator_sigs)

    adjacency_exact = False
    if variant_partition_exact and operator_partition_exact:
        prime_variant_index = {
            signature: index for index, signature in enumerate(prime_variant_sigs)
        }
        adjacency_exact = True
        for i, sig_i in enumerate(base_variant_sigs):
            ip = prime_variant_index[sig_i]
            for j, sig_j in enumerate(base_variant_sigs):
                jp = prime_variant_index[sig_j]
                base_operator_signature = base_operator_sigs[base.adjacency[i][j]]
                prime_operator_signature = prime_operator_sigs[prime.adjacency[ip][jp]]
                if base_operator_signature != prime_operator_signature:
                    adjacency_exact = False
                    break
            if not adjacency_exact:
                break

    prime_operator_index = {
        signature: index for index, signature in enumerate(prime_operator_sigs)
    }

    Paf = np.asarray(transformed.P_A, dtype=float)
    Pmf = np.asarray(transformed.P_M, dtype=float)
    PaiT = np.asarray(transformed.P_A.inv().T, dtype=float)
    PmiT = np.asarray(transformed.P_M.inv().T, dtype=float)
    Pmi = np.asarray(transformed.P_M.inv(), dtype=float)

    twin_residuals: list[TwinCovarianceResidual] = []
    twin_provenance_exact = operator_partition_exact

    if operator_partition_exact:
        for base_index, base_operator in enumerate(base.operators):
            signature = base_operator_sigs[base_index]
            prime_index = prime_operator_index[signature]
            prime_operator = prime.operators[prime_index]

            base_twins = twins_from_operator(
                base_operator, M_a, M_m, correspondence
            )
            prime_twins = twins_from_operator(
                prime_operator,
                transformed.parent_metric,
                transformed.product_metric,
                transformed.correspondence,
            )

            base_by_provenance = {
                (twin.kind, matrix_key(twin.parent_symmetry)): twin
                for twin in base_twins
            }
            prime_by_provenance = {}
            for twin in prime_twins:
                pulled_parent_symmetry = sp.simplify(
                    transformed.P_A
                    * twin.parent_symmetry
                    * transformed.P_A.inv()
                )
                key = (twin.kind, matrix_key(pulled_parent_symmetry))
                if key in prime_by_provenance:
                    raise AssertionError(
                        "Basis-transformed CT output has duplicate exact "
                        f"parent-symmetry provenance for {key!r}"
                    )
                prime_by_provenance[key] = twin

            if set(base_by_provenance) != set(prime_by_provenance):
                twin_provenance_exact = False
                continue

            for key, reference in base_by_provenance.items():
                candidate = prime_by_provenance[key]

                plane_a = PaiT @ candidate.plane_a
                direction_a = Paf @ candidate.direction_a
                plane_m = PmiT @ candidate.plane_m
                direction_m = Pmf @ candidate.direction_m
                Cint = Pmf @ candidate.intercorrespondence @ Pmi

                shear_scale = max(
                    abs(float(reference.shear)),
                    abs(float(candidate.shear)),
                    1.0e-15,
                )
                shear_abs = abs(float(reference.shear) - float(candidate.shear))

                twin_residuals.append(
                    TwinCovarianceResidual(
                        kind=reference.kind,
                        classification=reference.classification,
                        classification_match=(
                            reference.classification == candidate.classification
                        ),
                        representations_match=(
                            reference.representations == candidate.representations
                        ),
                        shear_abs_residual=shear_abs,
                        shear_rel_residual=shear_abs / shear_scale,
                        parent_plane_residual=metric_projective_residual(
                            reference.plane_a,
                            plane_a,
                            M_a,
                            reciprocal=True,
                        ),
                        parent_direction_residual=metric_projective_residual(
                            reference.direction_a,
                            direction_a,
                            M_a,
                            reciprocal=False,
                        ),
                        product_plane_residual=metric_projective_residual(
                            reference.plane_m,
                            plane_m,
                            M_m,
                            reciprocal=True,
                        ),
                        product_direction_residual=metric_projective_residual(
                            reference.direction_m,
                            direction_m,
                            M_m,
                            reciprocal=False,
                        ),
                        intercorrespondence_residual=_matrix_relative_residual(
                            reference.intercorrespondence, Cint
                        ),
                    )
                )

    return PhysicalEquivalenceReport(
        subgroup_exact=subgroup_exact,
        variant_partition_exact=variant_partition_exact,
        operator_partition_exact=operator_partition_exact,
        adjacency_exact=adjacency_exact,
        twin_provenance_exact=twin_provenance_exact,
        twin_residuals=tuple(twin_residuals),
        geometry_tolerance=float(geometry_tolerance),
        shear_relative_tolerance=float(shear_relative_tolerance),
        matrix_tolerance=float(matrix_tolerance),
    )


def _physical_axis_of_twofold(q: sp.Matrix, M_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Q = transform_matrix_to_metric_orthonormal(
        np.asarray(q, dtype=float), M_a
    )
    orth = float(np.linalg.norm(Q.T @ Q - np.eye(3), ord="fro"))
    det = float(abs(np.linalg.det(Q) - 1.0))
    if max(orth, det) > 1.0e-9:
        raise AssertionError(
            "Metric-preserving parent twofold did not become a proper "
            f"orthogonal rotation; orth={orth:.3e}, det={det:.3e}"
        )
    _, _, vh = np.linalg.svd(Q - np.eye(3))
    axis = _unit(vh[-1])
    if float(np.linalg.norm(Q @ axis - axis)) > 1.0e-9:
        raise AssertionError("Failed to recover physical twofold axis")
    return Q, axis


def _axis_key(axis: np.ndarray, decimals: int = 11) -> tuple[float, float, float]:
    value = _unit(axis)
    nz = np.flatnonzero(np.abs(value) > 10.0 ** (-(decimals - 2)))
    if len(nz) and value[nz[0]] < 0.0:
        value = -value
    return tuple(float(x) for x in np.round(value, decimals))


def _parent_plane_hat(p_a: np.ndarray, M_a: np.ndarray) -> np.ndarray:
    normal_crystal = plane_to_unit_normal(p_a, M_a)
    return _unit(metric_sqrt(M_a) @ normal_crystal)


def _parent_direction_hat(u_a: np.ndarray, M_a: np.ndarray) -> np.ndarray:
    return _unit(metric_sqrt(M_a) @ np.asarray(u_a, dtype=float).reshape(3))


def _shear_from_rank_one(
    a: np.ndarray,
    n: np.ndarray,
    Uj: np.ndarray,
) -> float:
    # Invariant under the non-unique scaling a -> rho a, n -> n/rho.
    return float(
        np.linalg.norm(np.asarray(a, dtype=float).reshape(3))
        * np.linalg.norm(
            np.linalg.solve(
                np.asarray(Uj, dtype=float).T,
                np.asarray(n, dtype=float).reshape(3),
            )
        )
    )


@dataclass(frozen=True)
class TheoryRelationResidual:
    operator_index: int
    compound_expected_from_twofold_generators: bool
    ct_classification_match: bool
    ct_type_i_vs_mallard_shear_rel: float
    ct_type_ii_vs_mallard_shear_rel: float
    ct_type_i_plane_angle_deg: float
    ct_type_i_direction_angle_deg: float
    ct_type_ii_plane_angle_deg: float
    ct_type_ii_direction_angle_deg: float
    mallard_type_i_rank_one_residual: float
    mallard_type_ii_rank_one_residual: float
    ball_james_type_i_rank_one_residual: float
    ball_james_type_ii_rank_one_residual: float
    ball_james_type_i_vs_mallard_rotation_residual: float
    ball_james_type_ii_vs_mallard_rotation_residual: float
    ball_james_type_i_vs_mallard_plane_angle_deg: float
    ball_james_type_ii_vs_mallard_plane_angle_deg: float
    ball_james_type_i_vs_mallard_direction_angle_deg: float
    ball_james_type_ii_vs_mallard_direction_angle_deg: float
    ball_james_type_i_vs_mallard_shear_rel: float
    ball_james_type_ii_vs_mallard_shear_rel: float

    @property
    def max_ct_geometry_angle_deg(self) -> float:
        return max(
            self.ct_type_i_plane_angle_deg,
            self.ct_type_i_direction_angle_deg,
            self.ct_type_ii_plane_angle_deg,
            self.ct_type_ii_direction_angle_deg,
        )

    @property
    def max_ball_james_geometry_angle_deg(self) -> float:
        return max(
            self.ball_james_type_i_vs_mallard_plane_angle_deg,
            self.ball_james_type_ii_vs_mallard_plane_angle_deg,
            self.ball_james_type_i_vs_mallard_direction_angle_deg,
            self.ball_james_type_ii_vs_mallard_direction_angle_deg,
        )

    @property
    def max_shear_relative_residual(self) -> float:
        return max(
            self.ct_type_i_vs_mallard_shear_rel,
            self.ct_type_ii_vs_mallard_shear_rel,
            self.ball_james_type_i_vs_mallard_shear_rel,
            self.ball_james_type_ii_vs_mallard_shear_rel,
        )


@dataclass(frozen=True)
class TheoryCrossValidationReport:
    relations: tuple[TheoryRelationResidual, ...]
    ct_coverage_complete: bool
    geometry_angle_tolerance_deg: float
    shear_relative_tolerance: float
    rank_one_tolerance: float
    rotation_tolerance: float

    @property
    def relation_count(self) -> int:
        return len(self.relations)

    @property
    def max_ct_geometry_angle_deg(self) -> float:
        return max(
            (item.max_ct_geometry_angle_deg for item in self.relations),
            default=0.0,
        )

    @property
    def max_ball_james_geometry_angle_deg(self) -> float:
        return max(
            (item.max_ball_james_geometry_angle_deg for item in self.relations),
            default=0.0,
        )

    @property
    def max_shear_relative_residual(self) -> float:
        return max(
            (item.max_shear_relative_residual for item in self.relations),
            default=0.0,
        )

    @property
    def max_rank_one_residual(self) -> float:
        values = []
        for item in self.relations:
            values.extend(
                [
                    item.mallard_type_i_rank_one_residual,
                    item.mallard_type_ii_rank_one_residual,
                    item.ball_james_type_i_rank_one_residual,
                    item.ball_james_type_ii_rank_one_residual,
                ]
            )
        return max(values, default=0.0)

    @property
    def max_rotation_residual(self) -> float:
        return max(
            (
                max(
                    item.ball_james_type_i_vs_mallard_rotation_residual,
                    item.ball_james_type_ii_vs_mallard_rotation_residual,
                )
                for item in self.relations
            ),
            default=0.0,
        )

    @property
    def classification_exact(self) -> bool:
        return all(item.ct_classification_match for item in self.relations)

    @property
    def success(self) -> bool:
        return (
            self.relation_count > 0
            and self.ct_coverage_complete
            and self.classification_exact
            and self.max_ct_geometry_angle_deg
            <= self.geometry_angle_tolerance_deg
            and self.max_ball_james_geometry_angle_deg
            <= self.geometry_angle_tolerance_deg
            and self.max_shear_relative_residual
            <= self.shear_relative_tolerance
            and self.max_rank_one_residual <= self.rank_one_tolerance
            and self.max_rotation_residual <= self.rotation_tolerance
        )


def _independent_generating_axes(
    U0: np.ndarray,
    Uj: np.ndarray,
    parent_group: list[sp.Matrix] | tuple[sp.Matrix, ...],
    M_a: np.ndarray,
    *,
    tol: float,
) -> tuple[tuple[float, float, float], ...]:
    axes = {}
    for q in parent_group:
        if classify_parent_order_two_isometry(q, M_a, tol=1.0e-9) != "twofold":
            continue
        Q, axis = _physical_axis_of_twofold(q, M_a)
        candidate = Q @ U0 @ Q.T
        if np.linalg.norm(candidate - Uj, ord="fro") <= tol:
            axes[_axis_key(axis)] = None
    return tuple(sorted(axes))


def _rank_one_pair_metrics(
    analytical,
    mallard,
    Uj: np.ndarray,
) -> tuple[float, float, float, float]:
    rotation_residual = float(
        np.linalg.norm(
            np.asarray(analytical.rotation, dtype=float)
            - np.asarray(mallard.rotation, dtype=float),
            ord="fro",
        )
    )
    plane_angle = projective_angle_deg(analytical.n, mallard.n)
    direction_angle = projective_angle_deg(analytical.a, mallard.a)
    shear_a = _shear_from_rank_one(analytical.a, analytical.n, Uj)
    shear_m = float(mallard.shear_magnitude)
    shear_rel = abs(shear_a - shear_m) / max(abs(shear_a), abs(shear_m), 1.0e-15)
    return rotation_residual, plane_angle, direction_angle, shear_rel


def _best_ball_james_assignment(
    analytical,
    mallard_by_kind: dict,
    Uj: np.ndarray,
):
    """One-to-one match of the generic Ball-James branches to Mallard I/II."""

    kinds = ("I", "II")
    if len(analytical) != 2 or set(mallard_by_kind) != set(kinds):
        raise AssertionError(
            "Expected exactly two generic Ball-James and two Mallard branches"
        )

    best = None
    for perm in permutations(range(2)):
        metrics = [
            _rank_one_pair_metrics(
                analytical[perm[index]],
                mallard_by_kind[kind],
                Uj,
            )
            for index, kind in enumerate(kinds)
        ]
        # Rotations are scale-independent and give a particularly clean branch
        # identifier. Geometry is included as a tie-breaker.
        score = max(
            max(item[0], np.deg2rad(item[1]), np.deg2rad(item[2]))
            for item in metrics
        )
        if best is None or score < best[0]:
            best = (score, perm, metrics)

    assert best is not None
    _, perm, metrics = best
    return (
        (analytical[perm[0]], metrics[0]),
        (analytical[perm[1]], metrics[1]),
    )


def cross_validate_ct_mallard_ball_james(
    M_a: np.ndarray,
    M_m: np.ndarray,
    parent_group: list[sp.Matrix] | tuple[sp.Matrix, ...],
    product_group: list[sp.Matrix] | tuple[sp.Matrix, ...],
    correspondence: Correspondence,
    *,
    relation_tolerance: float = 2.0e-9,
    geometry_angle_tolerance_deg: float = 2.0e-6,
    shear_relative_tolerance: float = 2.0e-9,
    rank_one_tolerance: float = 2.0e-9,
    rotation_tolerance: float = 2.0e-8,
) -> TheoryCrossValidationReport:
    """Cross-validate CT against independent Mallard and Ball-James solvers.

    The three paths are kept independent:

    * CT obtains twin elements from parent symmetry + correspondence + metrics.
    * Mallard obtains Type-I/II rank-one solutions from ``U0``, ``Uj`` and a
      physical 180-degree symmetry axis.
    * the generic Ball-James solver receives only ``U0`` and ``Uj`` and solves
      the middle-eigenvalue compatibility problem analytically.

    CT outputs are never used to construct ``Uj`` or either independent
    rank-one solution.
    """

    M_a = _as_spd_metric(M_a, name="M_a")
    M_m = _as_spd_metric(M_m, name="M_m")

    groupoid = correspondence_groupoid(
        list(parent_group), list(product_group), correspondence
    )
    U0 = stretch_from_metrics(M_a, M_m, correspondence)

    all_raw_ct_keys = set()
    covered_ct_keys = set()
    residuals: list[TheoryRelationResidual] = []

    for operator_index, operator in enumerate(groupoid.operators):
        raw_twins = twins_from_operator(operator, M_a, M_m, correspondence)
        raw_by_key = {
            (twin.kind, matrix_key(twin.parent_symmetry)): twin
            for twin in raw_twins
        }
        all_raw_ct_keys.update(
            (operator_index, key) for key in raw_by_key
        )

        twofold_entries = []
        for q in operator:
            if classify_parent_order_two_isometry(q, M_a, tol=1.0e-9) != "twofold":
                continue
            Q, axis = _physical_axis_of_twofold(q, M_a)
            Uj = Q @ U0 @ Q.T
            Uj = 0.5 * (Uj + Uj.T)
            if np.linalg.norm(Uj - U0, ord="fro") <= relation_tolerance:
                continue
            twofold_entries.append((sp.Matrix(q), Q, axis, Uj))

        for q, _, axis, Uj in twofold_entries:
            mirror = -q
            operator_keys = {matrix_key(item) for item in operator}
            if matrix_key(mirror) not in operator_keys:
                raise AssertionError(
                    "The requested exact CT I/II cross-validation requires "
                    "the reflection -Q to lie in the same double coset as Q. "
                    f"This failed in operator {operator_index}."
                )

            key_ii = ("II", matrix_key(q))
            key_i = ("I", matrix_key(mirror))
            if key_ii not in raw_by_key or key_i not in raw_by_key:
                raise AssertionError(
                    "CT did not return the expected non-zero Type-I/II pair "
                    f"for operator {operator_index}."
                )

            ct_i = raw_by_key[key_i]
            ct_ii = raw_by_key[key_ii]
            covered_ct_keys.add((operator_index, key_i))
            covered_ct_keys.add((operator_index, key_ii))

            generator_axes = _independent_generating_axes(
                U0,
                Uj,
                parent_group,
                M_a,
                tol=relation_tolerance,
            )
            compound_expected = len(generator_axes) >= 2
            expected_i = "compound" if compound_expected else "type_I"
            expected_ii = "compound" if compound_expected else "type_II"
            classification_match = (
                ct_i.classification == expected_i
                and ct_ii.classification == expected_ii
            )

            mallard_by_kind = {
                item.kind: item
                for item in mallard_law_twins(
                    U0, Uj, axis, tol=max(relation_tolerance, 1.0e-9)
                )
            }
            if set(mallard_by_kind) != {"I", "II"}:
                raise AssertionError(
                    f"Mallard law did not return both branches in operator "
                    f"{operator_index}: {set(mallard_by_kind)}"
                )

            analytical = analytical_rank_one_connections(
                U0, Uj, tol=max(relation_tolerance, 1.0e-9)
            )
            (bj_i, bj_i_metrics), (bj_ii, bj_ii_metrics) = (
                _best_ball_james_assignment(
                    analytical,
                    mallard_by_kind,
                    Uj,
                )
            )

            ct_i_plane = _parent_plane_hat(ct_i.plane_a, M_a)
            ct_ii_plane = _parent_plane_hat(ct_ii.plane_a, M_a)
            ct_i_ref_dir = _parent_direction_hat(ct_i.direction_a, M_a)
            ct_ii_ref_dir = _parent_direction_hat(ct_ii.direction_a, M_a)
            ct_i_current_dir = _unit(Uj @ ct_i_ref_dir)
            ct_ii_current_dir = _unit(Uj @ ct_ii_ref_dir)

            mallard_i = mallard_by_kind["I"]
            mallard_ii = mallard_by_kind["II"]

            def shear_rel(ct_shear: float, independent_shear: float) -> float:
                return abs(float(ct_shear) - float(independent_shear)) / max(
                    abs(float(ct_shear)),
                    abs(float(independent_shear)),
                    1.0e-15,
                )

            residuals.append(
                TheoryRelationResidual(
                    operator_index=operator_index,
                    compound_expected_from_twofold_generators=compound_expected,
                    ct_classification_match=classification_match,
                    ct_type_i_vs_mallard_shear_rel=shear_rel(
                        ct_i.shear, mallard_i.shear_magnitude
                    ),
                    ct_type_ii_vs_mallard_shear_rel=shear_rel(
                        ct_ii.shear, mallard_ii.shear_magnitude
                    ),
                    ct_type_i_plane_angle_deg=projective_angle_deg(
                        ct_i_plane, mallard_i.n
                    ),
                    ct_type_i_direction_angle_deg=projective_angle_deg(
                        ct_i_current_dir, mallard_i.a
                    ),
                    ct_type_ii_plane_angle_deg=projective_angle_deg(
                        ct_ii_plane, mallard_ii.n
                    ),
                    ct_type_ii_direction_angle_deg=projective_angle_deg(
                        ct_ii_current_dir, mallard_ii.a
                    ),
                    mallard_type_i_rank_one_residual=float(mallard_i.residual),
                    mallard_type_ii_rank_one_residual=float(mallard_ii.residual),
                    ball_james_type_i_rank_one_residual=float(bj_i.residual),
                    ball_james_type_ii_rank_one_residual=float(bj_ii.residual),
                    ball_james_type_i_vs_mallard_rotation_residual=float(
                        bj_i_metrics[0]
                    ),
                    ball_james_type_ii_vs_mallard_rotation_residual=float(
                        bj_ii_metrics[0]
                    ),
                    ball_james_type_i_vs_mallard_plane_angle_deg=float(
                        bj_i_metrics[1]
                    ),
                    ball_james_type_ii_vs_mallard_plane_angle_deg=float(
                        bj_ii_metrics[1]
                    ),
                    ball_james_type_i_vs_mallard_direction_angle_deg=float(
                        bj_i_metrics[2]
                    ),
                    ball_james_type_ii_vs_mallard_direction_angle_deg=float(
                        bj_ii_metrics[2]
                    ),
                    ball_james_type_i_vs_mallard_shear_rel=float(
                        bj_i_metrics[3]
                    ),
                    ball_james_type_ii_vs_mallard_shear_rel=float(
                        bj_ii_metrics[3]
                    ),
                )
            )

    return TheoryCrossValidationReport(
        relations=tuple(residuals),
        ct_coverage_complete=(all_raw_ct_keys == covered_ct_keys),
        geometry_angle_tolerance_deg=float(geometry_angle_tolerance_deg),
        shear_relative_tolerance=float(shear_relative_tolerance),
        rank_one_tolerance=float(rank_one_tolerance),
        rotation_tolerance=float(rotation_tolerance),
    )
