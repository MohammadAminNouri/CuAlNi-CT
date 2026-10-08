from __future__ import annotations

"""Pure input helpers for the standalone twin-family workbench.

The scientific backend always receives the canonical correspondence

    u_M = C_(M<-A) u_A.

The UI may accept either A -> M or M -> A.  When the reverse direction is
selected, this module performs an exact symbolic inverse before the state is
submitted to the existing scientific backend.  No floating approximation is
introduced at this boundary.
"""

from fractions import Fraction
from typing import Sequence

import sympy as sp


MatrixRows = tuple[
    tuple[str, str, str],
    tuple[str, str, str],
    tuple[str, str, str],
]


def _exact_scalar(value: object) -> sp.Rational:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid correspondence entries")
    text = str(value).strip()
    if not text:
        raise ValueError("Correspondence entries must not be empty")
    try:
        fraction = Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(
            f"Invalid exact correspondence entry {text!r}; use an integer, decimal, or p/q"
        ) from exc
    return sp.Rational(fraction.numerator, fraction.denominator)


def exact_matrix(rows: Sequence[Sequence[object]]) -> sp.Matrix:
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("Correspondence must contain exactly 3 rows and 3 columns")
    matrix = sp.Matrix([[_exact_scalar(value) for value in row] for row in rows])
    determinant = sp.simplify(matrix.det())
    if determinant == 0:
        raise ValueError("Correspondence matrix must be invertible")
    return matrix


def _rows(matrix: sp.Matrix) -> MatrixRows:
    simplified = sp.Matrix(matrix).applyfunc(sp.simplify)
    return tuple(
        tuple(str(simplified[i, j]) for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


def canonical_correspondence_rows(
    entered_rows: Sequence[Sequence[object]],
    *,
    direction: str,
) -> MatrixRows:
    """Return exact canonical C_(M<-A) rows from either input direction.

    Parameters
    ----------
    entered_rows:
        User-entered exact 3x3 matrix.
    direction:
        ``"A_TO_M"`` means the entered matrix acts as ``u_M = C u_A`` and is
        already canonical. ``"M_TO_A"`` means ``u_A = C u_M``; the exact inverse
        is returned.
    """

    matrix = exact_matrix(entered_rows)
    key = str(direction).strip().upper()
    if key == "A_TO_M":
        canonical = matrix
    elif key == "M_TO_A":
        canonical = sp.simplify(matrix.inv())
    else:
        raise ValueError("direction must be 'A_TO_M' or 'M_TO_A'")
    return _rows(canonical)


def displayed_relation(direction: str) -> tuple[str, str]:
    key = str(direction).strip().upper()
    if key == "A_TO_M":
        return "A → M", "u_M = C_(M←A) u_A"
    if key == "M_TO_A":
        return "M → A", "u_A = C_(A←M) u_M"
    raise ValueError("direction must be 'A_TO_M' or 'M_TO_A'")
