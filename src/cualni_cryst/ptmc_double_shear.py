
from __future__ import annotations

"""Auditable two-LIS PTMC compatibility.

PTCLab's manual describes a double-shear PTMC route with two lattice-invariant
shear systems and one additional degree of freedom fixed by prescribing the
second shear.  The manual does not publish the source equations.  This module
therefore makes the composition law explicit instead of silently claiming one.

``additive_laminate`` extends the repository's validated single-shear PTMC
affine predeformation directly:

    F_pre = U + p1 a1⊗n1 + p2 a2⊗n2.

The parameters are called three-variant volume fractions only when both rank-one
increments carry explicit common-base variant provenance; otherwise they remain
generic coefficients.

``sequential_simple_shear`` evaluates explicitly composed finite shear
generators derived from the same rank-one increments.  If
``A_i=a_i⊗n_i`` is the deformation-gradient increment relative to the base
variant ``U``, its spatial multiplicative generator is ``K_i=A_i U^-1``.
Therefore

    F_pre = (I + f2 K2)(I + f1 K1) U.

This definition has the required single-shear limit
``F_pre(f2=0)=U+f1 A1``.

For a fixed second parameter, every binary64-exact real root in the explicitly
configured first-parameter domain of ``det(F_pre.T F_pre - delta^2 I)=0`` is
isolated symbolically and then checked
against the middle singular value.  Habit branches are solved independently by
the existing Ball--James rank-one routine.  This is not the special Kelly
screw-dislocation construction.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import sympy as sp

from .ball_james import AnalyticalRankOneSolution, analytical_rank_one_connections


class DoubleShearComposition(str, Enum):
    ADDITIVE_LAMINATE = "additive_laminate"
    SEQUENTIAL_SIMPLE_SHEAR = "sequential_simple_shear"


class DoubleShearSystemSource(str, Enum):
    UNSPECIFIED = "unspecified_rank_one_increment"
    VARIANT_RANK_ONE_INCREMENT = "variant_rank_one_increment"


class DoubleShearParameterSemantics(str, Enum):
    GENERIC_COEFFICIENTS = "generic_coefficients"
    THREE_VARIANT_FRACTIONS = "three_variant_fractions"
    FINITE_SHEAR_MULTIPLIERS = "finite_shear_multipliers"


@dataclass(frozen=True)
class DoubleShearSystem:
    a: tuple[float, float, float]
    n: tuple[float, float, float]
    label: str = ""
    source: DoubleShearSystemSource = DoubleShearSystemSource.UNSPECIFIED
    base_variant_index: int | None = None
    other_variant_index: int | None = None
    source_rank_one_residual: float | None = None

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        a = np.asarray(self.a, dtype=float).reshape(3)
        n = np.asarray(self.n, dtype=float).reshape(3)
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(n)):
            raise ValueError("Double-shear a and n must be finite")
        if float(np.linalg.norm(a)) <= 1.0e-15:
            raise ValueError("Double-shear a must be nonzero")
        if float(np.linalg.norm(n)) <= 1.0e-15:
            raise ValueError("Double-shear n must be nonzero")
        return a, n


@dataclass(frozen=True)
class DoubleShearHabitBranch:
    """One habit rank-one branch with a unit physical parent normal.

    ``AnalyticalRankOneSolution`` has a gauge freedom a⊗n = (c a)⊗(n/c).
    This public form fixes |n|=1 and carries the compensating scale in the
    shape vector, making shape magnitudes directly comparable with PTMC/CT
    outputs expressed against unit habit normals.
    """

    branch: int
    rotation: np.ndarray
    shape_vector_parent_cartesian: np.ndarray
    habit_normal_parent_cartesian: np.ndarray
    shape_vector_magnitude: float
    residual: float
    eigenvalues_C: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "rotation": self.rotation.tolist(),
            "shape_vector_parent_cartesian": self.shape_vector_parent_cartesian.tolist(),
            "habit_normal_parent_cartesian": self.habit_normal_parent_cartesian.tolist(),
            "shape_vector_magnitude": self.shape_vector_magnitude,
            "residual": self.residual,
            "eigenvalues_C": self.eigenvalues_C.tolist(),
        }


@dataclass(frozen=True)
class DoubleShearSolution:
    first_parameter: float
    second_parameter: float
    composition: DoubleShearComposition
    pre_shape_deformation: np.ndarray
    habit_connections: tuple[DoubleShearHabitBranch, ...]
    middle_stretch_residual: float
    determinant_residual: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_parameter": self.first_parameter,
            "second_parameter": self.second_parameter,
            "composition": self.composition.value,
            "pre_shape_deformation": self.pre_shape_deformation.tolist(),
            "middle_stretch_residual": self.middle_stretch_residual,
            "determinant_residual": self.determinant_residual,
            "habit_connections": [item.to_dict() for item in self.habit_connections],
        }


@dataclass(frozen=True)
class DoubleShearContinuum:
    second_parameter: float
    composition: DoubleShearComposition
    first_parameter_domain: tuple[float, float]
    note: str


@dataclass(frozen=True)
class DoubleShearReport:
    composition: DoubleShearComposition
    parameter_semantics: DoubleShearParameterSemantics
    dilatational_factor: float
    first_system: DoubleShearSystem
    second_system: DoubleShearSystem
    fixed_second_parameter: float
    first_parameter_domain: tuple[float, float]
    polynomial_coefficients_ascending: tuple[float, ...]
    polynomial_verification_residual: float
    solutions: tuple[DoubleShearSolution, ...]
    continuum: DoubleShearContinuum | None
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "composition": self.composition.value,
            "parameter_semantics": self.parameter_semantics.value,
            "dilatational_factor": self.dilatational_factor,
            "first_system": {
                "a": list(self.first_system.a),
                "n": list(self.first_system.n),
                "label": self.first_system.label,
                "source": self.first_system.source.value,
                "base_variant_index": self.first_system.base_variant_index,
                "other_variant_index": self.first_system.other_variant_index,
                "source_rank_one_residual": self.first_system.source_rank_one_residual,
            },
            "second_system": {
                "a": list(self.second_system.a),
                "n": list(self.second_system.n),
                "label": self.second_system.label,
                "source": self.second_system.source.value,
                "base_variant_index": self.second_system.base_variant_index,
                "other_variant_index": self.second_system.other_variant_index,
                "source_rank_one_residual": self.second_system.source_rank_one_residual,
            },
            "fixed_second_parameter": self.fixed_second_parameter,
            "first_parameter_domain": list(self.first_parameter_domain),
            "polynomial_coefficients_ascending": list(
                self.polynomial_coefficients_ascending
            ),
            "polynomial_verification_residual": self.polynomial_verification_residual,
            "solutions": [item.to_dict() for item in self.solutions],
            "continuum": (
                None
                if self.continuum is None
                else {
                    "second_parameter": self.continuum.second_parameter,
                    "composition": self.continuum.composition.value,
                    "first_parameter_domain": list(
                        self.continuum.first_parameter_domain
                    ),
                    "note": self.continuum.note,
                }
            ),
            "notes": list(self.notes),
        }


def _validate_U(U: np.ndarray) -> np.ndarray:
    value = np.asarray(U, dtype=float)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError("U must be a finite 3x3 matrix")
    if float(np.linalg.det(value)) <= 0.0:
        raise ValueError("U must have positive determinant")
    return value


def _predeformation(
    U: np.ndarray,
    first: DoubleShearSystem,
    second: DoubleShearSystem,
    f1: float,
    f2: float,
    composition: DoubleShearComposition,
) -> np.ndarray:
    a1, n1 = first.arrays()
    a2, n2 = second.arrays()
    A1 = np.outer(a1, n1)
    A2 = np.outer(a2, n2)
    if composition is DoubleShearComposition.ADDITIVE_LAMINATE:
        return U + float(f1) * A1 + float(f2) * A2
    if composition is DoubleShearComposition.SEQUENTIAL_SIMPLE_SHEAR:
        U_inv = np.linalg.inv(U)
        K1 = A1 @ U_inv
        K2 = A2 @ U_inv
        return (
            (np.eye(3) + float(f2) * K2)
            @ (np.eye(3) + float(f1) * K1)
            @ U
        )
    raise AssertionError(f"Unhandled composition {composition}")


def _rat(value: float) -> sp.Rational:
    numerator, denominator = float(value).as_integer_ratio()
    return sp.Rational(numerator, denominator)


def _sp_matrix(A: np.ndarray) -> sp.Matrix:
    return sp.Matrix([[_rat(x) for x in row] for row in np.asarray(A, dtype=float)])


def _exact_polynomial(
    U: np.ndarray,
    first: DoubleShearSystem,
    second: DoubleShearSystem,
    fixed_second: float,
    composition: DoubleShearComposition,
    delta: float,
) -> sp.Poly:
    f = sp.symbols("f")
    Us = _sp_matrix(U)
    a1, n1 = first.arrays()
    a2, n2 = second.arrays()
    A1 = sp.Matrix([_rat(x) for x in a1]) * sp.Matrix(
        [_rat(x) for x in n1]
    ).T
    A2 = sp.Matrix([_rat(x) for x in a2]) * sp.Matrix(
        [_rat(x) for x in n2]
    ).T
    f2 = _rat(fixed_second)
    delta_s = _rat(delta)

    if composition is DoubleShearComposition.ADDITIVE_LAMINATE:
        F = Us + f * A1 + f2 * A2
    else:
        U_inv = Us.inv()
        K1 = A1 * U_inv
        K2 = A2 * U_inv
        F = (sp.eye(3) + f2 * K2) * (sp.eye(3) + f * K1) * Us
    expression = sp.expand((F.T * F - delta_s**2 * sp.eye(3)).det())
    return sp.Poly(expression, f)


def _real_roots(poly: sp.Poly, *, digits: int = 35) -> tuple[float, ...]:
    if poly.is_zero:
        return tuple()
    if poly.degree() <= 0:
        return tuple()
    eps = sp.Rational(1, 10**digits)
    roots = []
    for (lo, hi), multiplicity in sp.polys.polytools.intervals(poly, eps=eps):
        root = float((lo + hi) / 2)
        roots.extend([root] * int(multiplicity))
    return tuple(roots)


def _middle_singular_value(F: np.ndarray) -> float:
    return float(np.sort(np.linalg.svd(np.asarray(F, dtype=float), compute_uv=False))[1])


def _validated_parameter_domain(domain: tuple[float, float]) -> tuple[float, float]:
    if len(domain) != 2:
        raise ValueError("first_parameter_domain must contain exactly two bounds")
    lower, upper = map(float, domain)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("first_parameter_domain must be a finite ordered interval")
    return lower, upper


def _effective_parameter_domain(
    composition: DoubleShearComposition,
    fixed_second: float,
    semantics: DoubleShearParameterSemantics,
    requested: tuple[float, float],
) -> tuple[float, float]:
    lower, upper = _validated_parameter_domain(requested)
    if (
        composition is DoubleShearComposition.ADDITIVE_LAMINATE
        and semantics is DoubleShearParameterSemantics.THREE_VARIANT_FRACTIONS
    ):
        if not 0.0 <= fixed_second <= 1.0:
            raise ValueError(
                "fixed_second_parameter must lie in [0,1] for three_variant_fractions"
            )
        lower = max(lower, 0.0)
        upper = min(upper, 1.0 - fixed_second)
        if lower > upper:
            raise ValueError(
                "first_parameter_domain does not intersect the admissible three-variant "
                "simplex at the fixed second fraction"
            )
    return lower, upper


def _validate_fraction_provenance(
    first: DoubleShearSystem,
    second: DoubleShearSystem,
    *,
    tolerance: float,
) -> None:
    systems = (first, second)
    if any(item.source is not DoubleShearSystemSource.VARIANT_RANK_ONE_INCREMENT for item in systems):
        raise ValueError(
            "three_variant_fractions requires both rank-one increments to be explicitly "
            "provenanced as variant_rank_one_increment"
        )
    if any(item.base_variant_index is None or item.other_variant_index is None for item in systems):
        raise ValueError(
            "three_variant_fractions requires base_variant_index and other_variant_index "
            "for both LIS systems"
        )
    if first.base_variant_index != second.base_variant_index:
        raise ValueError("three_variant_fractions requires one common base variant")
    if first.other_variant_index == second.other_variant_index:
        raise ValueError("three_variant_fractions requires two distinct other variants")
    for item in systems:
        residual = item.source_rank_one_residual
        if residual is None or not np.isfinite(residual):
            raise ValueError(
                "three_variant_fractions requires a finite source_rank_one_residual "
                "for each LIS system"
            )
        if float(residual) > max(tolerance, 1.0e-10):
            raise ValueError(
                "A supplied variant rank-one increment fails its provenance audit: "
                f"residual={float(residual):.3e}"
            )


def _validate_distinct_systems(
    first: DoubleShearSystem,
    second: DoubleShearSystem,
    *,
    tolerance: float,
) -> None:
    a1, n1 = first.arrays()
    a2, n2 = second.arrays()
    A1 = np.outer(a1, n1).reshape(-1)
    A2 = np.outer(a2, n2).reshape(-1)
    u1 = A1 / np.linalg.norm(A1)
    u2 = A2 / np.linalg.norm(A2)
    projective_separation = min(
        float(np.linalg.norm(u1 - u2)),
        float(np.linalg.norm(u1 + u2)),
    )
    if projective_separation <= max(tolerance, 1.0e-12):
        raise ValueError(
            "The two lattice-invariant shear increments are projectively "
            "identical; double-shear PTMC requires two distinct systems"
        )


def solve_double_shear_ptmc(
    U: np.ndarray,
    first: DoubleShearSystem,
    second: DoubleShearSystem,
    *,
    fixed_second_parameter: float,
    composition: DoubleShearComposition | str = (
        DoubleShearComposition.ADDITIVE_LAMINATE
    ),
    parameter_semantics: DoubleShearParameterSemantics | str | None = None,
    first_parameter_domain: tuple[float, float] = (0.0, 1.0),
    dilatational_factor: float = 1.0,
    tolerance: float = 1.0e-8,
) -> DoubleShearReport:
    """Solve the remaining first-shear parameter after explicitly fixing the second."""

    U = _validate_U(U)
    mode = DoubleShearComposition(composition)
    if parameter_semantics is None:
        semantics = (
            DoubleShearParameterSemantics.GENERIC_COEFFICIENTS
            if mode is DoubleShearComposition.ADDITIVE_LAMINATE
            else DoubleShearParameterSemantics.FINITE_SHEAR_MULTIPLIERS
        )
    else:
        semantics = DoubleShearParameterSemantics(parameter_semantics)
    if mode is DoubleShearComposition.SEQUENTIAL_SIMPLE_SHEAR and semantics is not DoubleShearParameterSemantics.FINITE_SHEAR_MULTIPLIERS:
        raise ValueError(
            "sequential_simple_shear parameters are finite shear multipliers; "
            "other parameter semantics are not admissible"
        )
    if mode is DoubleShearComposition.ADDITIVE_LAMINATE and semantics is DoubleShearParameterSemantics.FINITE_SHEAR_MULTIPLIERS:
        raise ValueError(
            "additive_laminate does not use finite-shear-multiplier semantics"
        )
    f2 = float(fixed_second_parameter)
    delta = float(dilatational_factor)
    if not np.isfinite(f2):
        raise ValueError("fixed_second_parameter must be finite")
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("dilatational_factor must be finite and positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    _validate_distinct_systems(first, second, tolerance=tolerance)
    if semantics is DoubleShearParameterSemantics.THREE_VARIANT_FRACTIONS:
        _validate_fraction_provenance(first, second, tolerance=tolerance)
    lower, upper = _effective_parameter_domain(
        mode, f2, semantics, first_parameter_domain
    )

    poly = _exact_polynomial(U, first, second, f2, mode, delta)
    coefficients = tuple(float(item) for item in reversed(poly.all_coeffs()))
    if poly.is_zero:
        return DoubleShearReport(
            composition=mode,
            parameter_semantics=semantics,
            dilatational_factor=delta,
            first_system=first,
            second_system=second,
            fixed_second_parameter=f2,
            first_parameter_domain=(lower, upper),
            polynomial_coefficients_ascending=tuple(),
            polynomial_verification_residual=0.0,
            solutions=tuple(),
            continuum=DoubleShearContinuum(
                second_parameter=f2,
                composition=mode,
                first_parameter_domain=(lower, upper),
                note=(
                    "The exact binary64 compatibility determinant is identically "
                    "zero in the first parameter. The family is retained as a continuum."
                ),
            ),
            notes=_notes(mode, semantics),
        )

    roots = list(_real_roots(poly))
    # Exact endpoints are included even if interval conversion does not return them.
    roots.extend([lower, upper])
    merge = max(64.0 * np.finfo(float).eps, 0.1 * tolerance)
    accepted: list[float] = []
    for root in roots:
        if root < lower - tolerance or root > upper + tolerance:
            continue
        f1 = min(upper, max(lower, float(root)))
        F = _predeformation(U, first, second, f1, f2, mode)
        middle = abs(_middle_singular_value(F) - delta)
        determinant = abs(float(np.linalg.det(F.T @ F - delta**2 * np.eye(3))))
        if middle <= tolerance and not any(abs(f1 - old) <= merge for old in accepted):
            accepted.append(f1)

    accepted.sort()
    solutions = []
    polynomial_scale = max(
        max((abs(value) for value in coefficients), default=0.0),
        1.0,
    )
    max_poly_residual = 0.0
    for f1 in accepted:
        F = _predeformation(U, first, second, f1, f2, mode)
        middle = abs(_middle_singular_value(F) - delta)
        determinant = abs(float(np.linalg.det(F.T @ F - delta**2 * np.eye(3))))
        polynomial_value = abs(float(poly.eval(_rat(f1)))) / polynomial_scale
        max_poly_residual = max(max_poly_residual, polynomial_value)
        raw_habits = analytical_rank_one_connections(
            F,
            delta * np.eye(3),
            tol=tolerance,
        )
        habits_list: list[DoubleShearHabitBranch] = []
        for habit in raw_habits:
            normal_norm = float(np.linalg.norm(habit.n))
            if normal_norm <= 1.0e-15:
                raise AssertionError("Double-shear habit solver returned a zero normal")
            normal = np.asarray(habit.n, dtype=float) / normal_norm
            shape = np.asarray(habit.a, dtype=float) * normal_norm
            habits_list.append(
                DoubleShearHabitBranch(
                    branch=int(habit.branch),
                    rotation=np.asarray(habit.rotation, dtype=float),
                    shape_vector_parent_cartesian=shape,
                    habit_normal_parent_cartesian=normal,
                    shape_vector_magnitude=float(np.linalg.norm(shape)),
                    residual=float(habit.residual),
                    eigenvalues_C=np.asarray(habit.eigenvalues_C, dtype=float),
                )
            )
        habits = tuple(habits_list)
        solutions.append(
            DoubleShearSolution(
                first_parameter=f1,
                second_parameter=f2,
                composition=mode,
                pre_shape_deformation=F,
                habit_connections=habits,
                middle_stretch_residual=middle,
                determinant_residual=determinant,
            )
        )

    return DoubleShearReport(
        composition=mode,
        parameter_semantics=semantics,
        dilatational_factor=delta,
        first_system=first,
        second_system=second,
        fixed_second_parameter=f2,
        first_parameter_domain=(lower, upper),
        polynomial_coefficients_ascending=coefficients,
        polynomial_verification_residual=max_poly_residual,
        solutions=tuple(solutions),
        continuum=None,
        notes=_notes(mode, semantics),
    )


def _notes(
    mode: DoubleShearComposition,
    semantics: DoubleShearParameterSemantics,
) -> tuple[str, ...]:
    if semantics is DoubleShearParameterSemantics.THREE_VARIANT_FRACTIONS:
        parameter_note = (
            "f1 and f2 are interpreted as three-variant laminate fractions only because "
            "both LIS increments carry validated common-base variant provenance; the "
            "simplex f1>=0, f2>=0, f1+f2<=1 is enforced."
        )
    elif semantics is DoubleShearParameterSemantics.GENERIC_COEFFICIENTS:
        parameter_note = (
            "The additive parameters are numerical coefficients on the supplied rank-one "
            "increments over the explicitly configured first-parameter domain. They are not "
            "called phase or variant volume fractions without common-base variant provenance."
        )
    else:
        parameter_note = (
            "For sequential_simple_shear, A_i=a_i⊗n_i is converted to K_i=A_i U^-1. "
            "The parameters are finite-shear multipliers, not laminate volume fractions."
        )
    return (
        (
            "Two LIS systems and the fixed second parameter are explicit inputs; "
            "the solver never invents a second shear system."
        ),
        (
            "All exact-binary64 roots in the remaining parameter are retained, "
            "including legitimate zero-solution and continuum states."
        ),
        parameter_note,
        (
            "Habit-plane rank-one branches are obtained independently from the "
            "existing Ball-James solver after the two-LIS predeformation is built."
        ),
        (
            "This is the explicit "
            + mode.value
            + " composition; it is not the special Kelly screw-dislocation construction."
        ),
    )
