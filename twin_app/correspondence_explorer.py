from __future__ import annotations

"""Exact, material-independent correspondence action for an educational UI.

C maps parent *direct-coordinate column vectors* to daughter direct-coordinate
column vectors. These are not shared-frame Cartesian deformations or ORs.
"""

from dataclasses import dataclass
from typing import Sequence

import sympy as sp

from .input_logic import exact_matrix


@dataclass(frozen=True)
class CorrespondenceAction:
    matrix: sp.Matrix
    parent: tuple[sp.Rational, sp.Rational, sp.Rational]
    daughter: tuple[sp.Rational, sp.Rational, sp.Rational]
    determinant: sp.Rational


def _rational(value: object) -> sp.Rational:
    # Match existing rational grammar (integer, decimal, or exact p/q).
    return sp.Rational(str(value).strip())


def calculate_action(
    matrix_rows: Sequence[Sequence[object]],
    parent_vector: Sequence[object],
) -> CorrespondenceAction:
    """C @ u_parent, with no approximations, hidden alloy cases or rounding."""
    C = exact_matrix(matrix_rows)  # also rejects singular or malformed input
    if len(parent_vector) != 3:
        raise ValueError("The parent direction must contain exactly 3 entries")
    vec = sp.Matrix([_rational(v) for v in parent_vector])
    output = C * vec
    # SymPy Rational matrices guarantee exact reversibility for this mapping.
    assert C.inv() * output == vec
    return CorrespondenceAction(
        matrix=C,
        parent=tuple(vec),
        daughter=tuple(output),
        determinant=sp.Rational(C.det()),
    )


def images_of_parent_basis(matrix_rows: Sequence[Sequence[object]]) -> tuple[CorrespondenceAction, ...]:
    basis = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    return tuple(calculate_action(matrix_rows, vec) for vec in basis)


def vector_text(vector: Sequence[object]) -> str:
    return "[" + ", ".join(str(x) for x in vector) + "]"
