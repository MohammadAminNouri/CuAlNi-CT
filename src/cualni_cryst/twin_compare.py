from __future__ import annotations

"""Operator-by-operator CT vs Ball-James/Mallard comparison for DO3 -> 6M.

This module compares genuinely independent constructions:

* Cayron CT: parent symmetry + correspondence + martensite metric.
* Ball-James/Mallard: stretch variants + rank-one compatibility.

A crucial coordinate distinction is kept explicit.  CT stores its parent-space
plane covector and parent-space direct shear direction in the austenite
reference coordinates.  In the Ball-James rank-one equation

    R U_i - U_j = a ⊗ n,

``n`` is a reference-configuration interface normal, whereas ``a`` is the
shear vector in the deformed/current configuration.  Therefore:

* CT plane_a is compared with Ball-James n after metric-to-orthonormal mapping.
* CT direction_a is first carried by U_j and then compared with Ball-James a.

Without this U_j mapping the two direction vectors are different coordinate
objects and a direct angular comparison would be scientifically wrong.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import sympy as sp

from .ball_james import MallardTwinSolution, mallard_law_twins
from .cualni_models import TransformationBranch
from .group_theory import correspondence_groupoid
from .lattice import metric_sqrt, plane_to_unit_normal
from .stretch import generate_stretch_variants, stretch_from_metrics
from .symmetry import classify_symmetry, matrix_key
from .twinning_ct import (
    CTTwin,
    type_i_from_parent_reflection,
    type_ii_from_parent_twofold,
)


def _unit(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(x))
    if n <= 1e-15:
        raise ValueError("Cannot normalize a zero vector")
    return x / n


def projective_angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Acute angle between unoriented lines/normals, robust near 0 and 180 degrees."""

    a = _unit(u)
    b = _unit(v)
    cross = float(np.linalg.norm(np.cross(a, b)))
    dot = abs(float(a @ b))
    return float(np.degrees(np.arctan2(cross, dot)))


def _parent_direct_to_orthonormal(u_a: np.ndarray, M_a: np.ndarray) -> np.ndarray:
    """Physical unit direction corresponding to parent crystallographic coordinates."""

    return _unit(metric_sqrt(M_a) @ np.asarray(u_a, dtype=float).reshape(3))


def _parent_plane_to_orthonormal_normal(
    p_a: np.ndarray, M_a: np.ndarray
) -> np.ndarray:
    """Physical unit normal corresponding to a parent reciprocal covector."""

    n_crystal = plane_to_unit_normal(p_a, M_a)
    return _unit(metric_sqrt(M_a) @ n_crystal)


def _twofold_axis(q: sp.Matrix) -> np.ndarray:
    info = classify_symmetry(q)
    if info.kind != "rotation" or info.order != 2 or info.axis is None:
        raise ValueError("Matrix is not a proper twofold parent rotation")
    return _unit(np.asarray(info.axis, dtype=float))


def _axis_key(axis: np.ndarray, decimals: int = 12) -> tuple[float, float, float]:
    """Canonical projective-axis key: e and -e are the same twofold axis."""

    a = _unit(axis)
    nz = np.flatnonzero(np.abs(a) > 10 ** (-(decimals - 2)))
    if len(nz) and a[nz[0]] < 0:
        a = -a
    return tuple(float(x) for x in np.round(a, decimals))


def _target_index(Uj: np.ndarray, variants: list[np.ndarray], tol: float) -> int:
    residuals = np.array(
        [np.linalg.norm(Uj - v, ord="fro") for v in variants], dtype=float
    )
    i = int(np.argmin(residuals))
    if float(residuals[i]) > tol:
        raise AssertionError(
            f"Target stretch not found in 12-variant family; residual={residuals[i]:.3e}"
        )
    return i


def _rotation_quality(sol: MallardTwinSolution) -> tuple[float, float]:
    R = np.asarray(sol.rotation, dtype=float)
    orth = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
    det = float(abs(np.linalg.det(R) - 1.0))
    return orth, det


@dataclass(frozen=True)
class TwinRelationComparison:
    operator_index: int
    target_stretch_index: int
    representative_twofold: sp.Matrix
    generator_axes: tuple[tuple[float, float, float], ...]
    compound: bool
    ct_type_i: CTTwin
    ct_type_ii: CTTwin
    bj_type_i: MallardTwinSolution
    bj_type_ii: MallardTwinSolution
    ct_type_i_vs_ii_shear_abs: float
    type_i_shear_abs_residual: float
    type_i_shear_rel_residual: float
    type_i_plane_angle_deg: float
    type_i_direction_angle_deg: float
    type_ii_shear_abs_residual: float
    type_ii_shear_rel_residual: float
    type_ii_plane_angle_deg: float
    type_ii_direction_angle_deg: float
    bj_type_i_orthogonality_residual: float
    bj_type_i_determinant_residual: float
    bj_type_ii_orthogonality_residual: float
    bj_type_ii_determinant_residual: float

    @property
    def shear(self) -> float:
        return float(self.ct_type_i.shear)

    @property
    def max_geometry_angle_deg(self) -> float:
        return max(
            self.type_i_plane_angle_deg,
            self.type_i_direction_angle_deg,
            self.type_ii_plane_angle_deg,
            self.type_ii_direction_angle_deg,
        )

    @property
    def max_relative_shear_residual(self) -> float:
        return max(
            self.type_i_shear_rel_residual,
            self.type_ii_shear_rel_residual,
        )

    @property
    def max_rank_one_residual(self) -> float:
        return max(self.bj_type_i.residual, self.bj_type_ii.residual)


@dataclass(frozen=True)
class OperatorTwinSummary:
    operator_index: int
    size: int
    classification: str
    symmetry_kinds: tuple[tuple[str, int], ...]
    proper_rotation_orders: tuple[int, ...]
    relations: tuple[TwinRelationComparison, ...]
    note: str


@dataclass(frozen=True)
class TwinAtlas:
    operators: tuple[OperatorTwinSummary, ...]
    n_exact_relations: int

    @property
    def max_geometry_angle_deg(self) -> float:
        values = [
            rel.max_geometry_angle_deg
            for op in self.operators
            for rel in op.relations
        ]
        return max(values, default=0.0)

    @property
    def max_relative_shear_residual(self) -> float:
        values = [
            rel.max_relative_shear_residual
            for op in self.operators
            for rel in op.relations
        ]
        return max(values, default=0.0)

    @property
    def max_rank_one_residual(self) -> float:
        values = [
            rel.max_rank_one_residual
            for op in self.operators
            for rel in op.relations
        ]
        return max(values, default=0.0)


def _group_twofolds_by_target(
    twofolds: list[sp.Matrix],
    U0: np.ndarray,
    tol: float,
) -> list[tuple[np.ndarray, list[sp.Matrix]]]:
    groups: list[tuple[np.ndarray, list[sp.Matrix]]] = []
    for q_sp in twofolds:
        q = np.asarray(q_sp, dtype=float)
        Uj = q @ U0 @ q.T
        for known_Uj, generators in groups:
            if np.linalg.norm(Uj - known_Uj, ord="fro") <= tol:
                generators.append(q_sp)
                break
        else:
            groups.append((Uj, [q_sp]))
    return groups


def _all_generating_axes(
    U0: np.ndarray,
    Uj: np.ndarray,
    parent_group: list[sp.Matrix],
    tol: float,
) -> tuple[tuple[float, float, float], ...]:
    axes: dict[tuple[float, float, float], None] = {}
    for q in parent_group:
        info = classify_symmetry(q)
        if info.kind != "rotation" or info.order != 2:
            continue
        qn = np.asarray(q, dtype=float)
        if np.linalg.norm(qn @ U0 @ qn.T - Uj, ord="fro") <= tol:
            axes[_axis_key(_twofold_axis(q))] = None
    return tuple(sorted(axes))


def _compare_one_relation(
    operator_index: int,
    operator: list[sp.Matrix],
    q_sp: sp.Matrix,
    U0: np.ndarray,
    Uj: np.ndarray,
    all_stretches: list[np.ndarray],
    parent_group: list[sp.Matrix],
    M_a: np.ndarray,
    M_m: np.ndarray,
    branch: TransformationBranch,
    tol: float,
) -> TwinRelationComparison:
    e = _twofold_axis(q_sp)

    # For a twofold Q, -Q is the reflection in the plane normal to the same
    # axis. Because inversion belongs to H_C for this reference branch, the
    # reflection and Q occur in the same double-coset operator.
    mirror = -sp.Matrix(q_sp)
    op_keys = {matrix_key(x) for x in operator}
    if matrix_key(mirror) not in op_keys:
        raise AssertionError(
            "Expected reflection partner -Q in the same operator; "
            "the current DO3->6M groupoid no longer satisfies this assumption."
        )
    if classify_symmetry(mirror).kind != "reflection":
        raise AssertionError("-Q is not classified as a reflection")

    ct_i = type_i_from_parent_reflection(
        mirror, M_a, M_m, branch.correspondence
    )
    ct_ii = type_ii_from_parent_twofold(
        q_sp, M_a, M_m, branch.correspondence
    )

    bj = {x.kind: x for x in mallard_law_twins(U0, Uj, e)}
    if set(bj) != {"I", "II"}:
        raise AssertionError(f"Mallard law did not produce both branches: {set(bj)}")
    bj_i = bj["I"]
    bj_ii = bj["II"]

    # Ball-James n is a reference normal. CT plane_a is a reciprocal
    # parent covector representing that same reference plane.
    ct_i_plane_hat = _parent_plane_to_orthonormal_normal(ct_i.plane_a, M_a)
    ct_ii_plane_hat = _parent_plane_to_orthonormal_normal(ct_ii.plane_a, M_a)

    # Ball-James a is a deformed/current shear direction. CT direction_a is
    # stored in the parent reference coordinates, so Cauchy-Born mapping by Uj
    # is required before comparison.
    ct_i_ref_direction_hat = _parent_direct_to_orthonormal(ct_i.direction_a, M_a)
    ct_ii_ref_direction_hat = _parent_direct_to_orthonormal(ct_ii.direction_a, M_a)
    ct_i_deformed_direction_hat = _unit(Uj @ ct_i_ref_direction_hat)
    ct_ii_deformed_direction_hat = _unit(Uj @ ct_ii_ref_direction_hat)

    def shear_residual(ct_shear: float, bj_shear: float) -> tuple[float, float]:
        absolute = abs(float(ct_shear) - float(bj_shear))
        relative = absolute / max(
            abs(float(ct_shear)), abs(float(bj_shear)), 1e-15
        )
        return absolute, relative

    i_abs, i_rel = shear_residual(ct_i.shear, bj_i.shear_magnitude)
    ii_abs, ii_rel = shear_residual(ct_ii.shear, bj_ii.shear_magnitude)
    orth_i, det_i = _rotation_quality(bj_i)
    orth_ii, det_ii = _rotation_quality(bj_ii)

    generator_axes = _all_generating_axes(
        U0, Uj, parent_group, tol
    )

    return TwinRelationComparison(
        operator_index=operator_index,
        target_stretch_index=_target_index(Uj, all_stretches, tol),
        representative_twofold=sp.Matrix(q_sp),
        generator_axes=generator_axes,
        compound=len(generator_axes) >= 2,
        ct_type_i=ct_i,
        ct_type_ii=ct_ii,
        bj_type_i=bj_i,
        bj_type_ii=bj_ii,
        ct_type_i_vs_ii_shear_abs=abs(ct_i.shear - ct_ii.shear),
        type_i_shear_abs_residual=i_abs,
        type_i_shear_rel_residual=i_rel,
        type_i_plane_angle_deg=projective_angle_deg(
            ct_i_plane_hat, bj_i.n
        ),
        type_i_direction_angle_deg=projective_angle_deg(
            ct_i_deformed_direction_hat, bj_i.a
        ),
        type_ii_shear_abs_residual=ii_abs,
        type_ii_shear_rel_residual=ii_rel,
        type_ii_plane_angle_deg=projective_angle_deg(
            ct_ii_plane_hat, bj_ii.n
        ),
        type_ii_direction_angle_deg=projective_angle_deg(
            ct_ii_deformed_direction_hat, bj_ii.a
        ),
        bj_type_i_orthogonality_residual=orth_i,
        bj_type_i_determinant_residual=det_i,
        bj_type_ii_orthogonality_residual=orth_ii,
        bj_type_ii_determinant_residual=det_ii,
    )


def build_twin_atlas(
    branch: TransformationBranch,
    M_a: np.ndarray,
    M_m: np.ndarray,
    *,
    tol: float = 1e-10,
) -> TwinAtlas:
    """Build the exact-operator twin atlas and CT-vs-Mallard cross-check.

    Classification used here:

    ``neutral``
        The double coset contains identity; no M/M boundary is reported.

    ``compound``
        At least two distinct parent twofold axes generate the same stretch pair.
        This follows the Ball-James/Chen definition rather than the weaker
        criterion "the operator contains both a reflection and a twofold".

    ``type_I_II``
        A nontrivial twofold-generated stretch pair has the ordinary two
        Mallard solutions, independently reproduced by CT Type I and Type II.

    ``weak_candidate_non_twofold``
        No parent twofold relation exists in the operator, but proper n-fold
        rotations with n != 2 are present. CT says no fully compatible plane is
        generated by that symmetry alone; weak-plane analysis is a later step.
    """

    parent = list(branch.parent_point_group)
    product = list(branch.product_point_group)
    groupoid = correspondence_groupoid(parent, product, branch.correspondence)

    U0 = stretch_from_metrics(M_a, M_m, branch.correspondence)
    proper = [q for q in parent if int(q.det()) == 1]
    all_stretches = generate_stretch_variants(
        U0, [np.asarray(q, dtype=float) for q in proper], tol=tol
    )

    summaries: list[OperatorTwinSummary] = []
    exact_relation_count = 0

    identity_key = matrix_key(sp.eye(3))

    for operator_index, operator in enumerate(groupoid.operators):
        kinds = Counter(classify_symmetry(x).kind for x in operator)
        proper_rotation_orders = tuple(
            sorted(
                {
                    classify_symmetry(x).order
                    for x in operator
                    if classify_symmetry(x).kind == "rotation"
                }
            )
        )

        if identity_key in {matrix_key(x) for x in operator}:
            summaries.append(
                OperatorTwinSummary(
                    operator_index,
                    len(operator),
                    "neutral",
                    tuple(sorted(kinds.items())),
                    proper_rotation_orders,
                    (),
                    "Identity operator / self relation; zero-shear subgroup symmetries are not M/M twins.",
                )
            )
            continue

        twofolds = [
            x
            for x in operator
            if classify_symmetry(x).kind == "rotation"
            and classify_symmetry(x).order == 2
        ]

        relations: list[TwinRelationComparison] = []
        if twofolds:
            for Uj, q_generators_in_operator in _group_twofolds_by_target(
                twofolds, U0, tol
            ):
                q_sp = q_generators_in_operator[0]
                rel = _compare_one_relation(
                    operator_index,
                    operator,
                    q_sp,
                    U0,
                    Uj,
                    all_stretches,
                    parent,
                    M_a,
                    M_m,
                    branch,
                    tol,
                )
                relations.append(rel)
            exact_relation_count += len(relations)
            classification = (
                "compound"
                if relations and all(x.compound for x in relations)
                else "type_I_II"
            )
            note = (
                "Twofold-generated exact M/M rank-one relations. "
                "CT and Mallard are calculated independently and compared after "
                "putting plane normals and shear directions in the same physical frame."
            )
        else:
            non_twofold_orders = tuple(x for x in proper_rotation_orders if x != 2)
            classification = (
                "weak_candidate_non_twofold"
                if non_twofold_orders
                else "unclassified"
            )
            relations = []
            note = (
                "No exact twofold-generated Type-I/II relation is asserted. "
                "Parent rotations of order n != 2 are retained only as CT weak-junction candidates."
                if classification == "weak_candidate_non_twofold"
                else "No Type-I/II classification obtained from the present exact symmetry test."
            )

        summaries.append(
            OperatorTwinSummary(
                operator_index,
                len(operator),
                classification,
                tuple(sorted(kinds.items())),
                proper_rotation_orders,
                tuple(relations),
                note,
            )
        )

    return TwinAtlas(tuple(summaries), exact_relation_count)
