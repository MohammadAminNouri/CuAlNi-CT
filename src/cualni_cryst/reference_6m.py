from __future__ import annotations

"""Truth-locked crystallographic definition of the DO3 -> 6M reference branch.

This module deliberately separates three things that are easy to conflate:

1. the physical long-period martensite,
2. the chosen 6M unit-cell/basis representation,
3. one symmetry-equivalent reference correspondence variant.

Coordinate convention used throughout the package:

    u_M = C_M_from_A @ u_A

for direct-space crystallographic coordinates. Consequently

    C_A_from_M = C_M_from_A^{-1}

has as its columns the daughter basis vectors expressed in parent coordinates.
Plane covectors transform by inverse transpose.

The reference basis is reconstructed from the 6M geometry summarized by
James & Hane (Acta Materialia 48, 2000) following Otsuka et al. The matrix is
therefore a source-constrained reconstruction, not a verbatim matrix quote.
"""

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class CrystallographicBasisConvention:
    """Explicit parent/product basis and angle metadata."""

    parent_basis_labels: tuple[str, str, str]
    daughter_basis_labels: tuple[str, str, str]
    parent_axis_order: tuple[str, str, str]
    daughter_axis_order: tuple[str, str, str]
    source_angle_symbol: str
    internal_angle_symbol: str
    non_right_angle_between: tuple[str, str]
    unique_axis: str
    notes: str


DO3_6M_BASIS_CONVENTION = CrystallographicBasisConvention(
    parent_basis_labels=("a_A", "b_A", "c_A"),
    daughter_basis_labels=("a_6M", "b_6M", "c_6M"),
    parent_axis_order=("[100]_A", "[010]_A", "[001]_A"),
    daughter_axis_order=("[100]_6M", "[010]_6M", "[001]_6M"),
    source_angle_symbol="gamma (James-Hane source notation)",
    internal_angle_symbol="beta",
    non_right_angle_between=("a_6M", "c_6M"),
    unique_axis="b_6M",
    notes=(
        "James & Hane state that the monoclinic non-right angle is the angle "
        "between martensite a and c. This package uses the conventional unique-b "
        "symbol beta for that same physical angle. The symbol change is metadata, "
        "not a basis change, unit conversion, or normalization."
    ),
)


# Chosen reference 6M basis in cubic DO3 parent coordinates.
#
#   a_6M = 1/2 [0  1  1]_A
#   b_6M =     [1  0  0]_A
#   c_6M = 3/2 [0  1 -1]_A
#
# This reference choice reproduces the James-Hane cube-edge variant with the
# martensite b-axis along the parent [100] cube edge.
#
# The factors are fixed by the cell geometry: the corresponding parent
# translation lengths are a0/sqrt(2), a0, and 3*a0/sqrt(2), which yield the
# James-Hane dimensionless stretches sqrt(2)a/a0, b/a0, and sqrt(2)c/(3a0).
# The 3/2 factor is a long-period translation; it is NOT a normalization.
A6M_IN_A = sp.Matrix([0, sp.Rational(1, 2), sp.Rational(1, 2)])
B6M_IN_A = sp.Matrix([1, 0, 0])
C6M_IN_A = sp.Matrix([0, sp.Rational(3, 2), -sp.Rational(3, 2)])


def daughter_basis_in_parent_coordinates() -> sp.Matrix:
    """Return C_A_from_M for the selected reference variant.

    Columns are a_6M, b_6M, c_6M expressed in the parent cubic basis.
    """

    return sp.Matrix.hstack(A6M_IN_A, B6M_IN_A, C6M_IN_A)


def derive_reference_correspondence_M_from_A() -> sp.Matrix:
    """Derive the exact reference direct-space correspondence."""

    return sp.simplify(daughter_basis_in_parent_coordinates().inv())


C_REF_A_FROM_M = daughter_basis_in_parent_coordinates()
C_REF_M_FROM_A = derive_reference_correspondence_M_from_A()


# Alternative matrix that appeared in earlier exploratory notes. It is retained
# only as an audit target. It is NOT an independent material law.
C_EXPLORATORY_ALT_M_FROM_A = sp.Matrix(
    [
        [1, 0, 1],
        [0, 1, 0],
        [-sp.Rational(1, 3), 0, sp.Rational(1, 3)],
    ]
)


def parent_axis_permutation_relating_reference_to_alt() -> sp.Matrix:
    """Return proper cubic P satisfying C_alt = C_ref P exactly."""

    P = sp.simplify(C_REF_M_FROM_A.inv() * C_EXPLORATORY_ALT_M_FROM_A)
    if sp.simplify(C_REF_M_FROM_A * P - C_EXPLORATORY_ALT_M_FROM_A) != sp.zeros(3):
        raise AssertionError("Alternative correspondence is not related by the computed parent permutation")
    if sp.simplify(P.T * P - sp.eye(3)) != sp.zeros(3):
        raise AssertionError("Computed relation is not orthogonal")
    if sp.simplify(P.det() - 1) != 0:
        raise AssertionError("Computed relation is not a proper cubic rotation/permutation")
    return P


def general_monoclinic_metric_symbolic(
    a: sp.Expr,
    b: sp.Expr,
    c: sp.Expr,
    alpha: sp.Expr,
    beta: sp.Expr,
    gamma: sp.Expr,
) -> sp.Matrix:
    """General direct metric before choosing a monoclinic setting."""

    return sp.Matrix(
        [
            [a**2, a * b * sp.cos(gamma), a * c * sp.cos(beta)],
            [a * b * sp.cos(gamma), b**2, b * c * sp.cos(alpha)],
            [a * c * sp.cos(beta), b * c * sp.cos(alpha), c**2],
        ]
    )


def unique_b_6m_metric_symbolic(
    a: sp.Expr, b: sp.Expr, c: sp.Expr, beta: sp.Expr
) -> sp.Matrix:
    """6M unique-b metric obtained by alpha=gamma=pi/2."""

    return sp.simplify(
        general_monoclinic_metric_symbolic(
            a, b, c, sp.pi / 2, beta, sp.pi / 2
        )
    )


def pulled_back_reference_metric_symbolic(
    a: sp.Expr, b: sp.Expr, c: sp.Expr, beta: sp.Expr
) -> sp.Matrix:
    """Exact G_C = C^T M_6M C for the selected reference variant."""

    M = unique_b_6m_metric_symbolic(a, b, c, beta)
    C = C_REF_M_FROM_A
    return sp.simplify(C.T * M * C)


def reference_parent_translation_lengths(a0: sp.Expr) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Parent lengths associated with the selected a_6M,b_6M,c_6M translations."""

    return (
        sp.simplify(a0 / sp.sqrt(2)),
        sp.simplify(a0),
        sp.simplify(3 * a0 / sp.sqrt(2)),
    )
