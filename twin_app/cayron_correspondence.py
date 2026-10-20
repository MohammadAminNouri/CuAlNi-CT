from __future__ import annotations

"""Exact crystallographic correspondence using Cayron's paired-supercell method.

Cayron (2019), Acta Cryst. A75, 411-437, Sections 2.3 and 3:
the columns of P and Q contain three corresponding direct lattice vectors,
respectively expressed in the source and destination crystal bases.
C_(destination <- source) = Q * P**(-1), NOT a rotation or distortion F.
The existing science engine always expects C_(M <- A).
"""

from dataclasses import dataclass
from typing import Sequence

import sympy as sp

from .input_logic import canonical_correspondence_rows, exact_matrix, MatrixRows, _exact_scalar


@dataclass(frozen=True)
class PairedSupercell:
    """Fully exact entered supercell and calculated canonical correspondence."""

    source_columns: MatrixRows
    destination_columns: MatrixRows
    source_to_destination: MatrixRows
    canonical_M_from_A: MatrixRows
    direction: str


def parse_triplet(text: object) -> tuple[str, str, str]:
    """Read three exact direct-lattice components; no floating approximation."""
    value = str(text).strip().strip("[]()").strip()
    values = value.split(",") if "," in value else value.split()
    if len(values) != 3:
        raise ValueError("Each lattice vector needs three components: u,v,w")
    result = []
    for component in values:
        component = component.strip()
        if len(component) > 80:
            raise ValueError("A lattice-vector component is too long")
        # Use the same exact rational parser and validation as matrix input.
        result.append(str(_exact_scalar(component)))
    return tuple(result)  # type: ignore[return-value]


def _columns(vectors: Sequence[Sequence[object]], *, label: str) -> sp.Matrix:
    if len(vectors) != 3 or any(len(v) != 3 for v in vectors):
        raise ValueError(f"{label}: provide exactly three 3-component lattice vectors")
    matrix = sp.Matrix.hstack(*(
        sp.Matrix([_exact_scalar(component) for component in vector])
        for vector in vectors
    ))
    if matrix.det() == 0:
        raise ValueError(f"{label}: the three vectors are linearly dependent")
    return matrix


def _rows(matrix: sp.Matrix) -> MatrixRows:
    return tuple(tuple(str(matrix[i, j]) for j in range(3)) for i in range(3))  # type: ignore[return-value]


def calculate_from_paired_vectors(
    source_vectors: Sequence[Sequence[object]],
    destination_vectors: Sequence[Sequence[object]],
    *,
    direction: str,
) -> PairedSupercell:
    """Exact Cayron equation (9), with automatic exact reverse conversion."""
    P = _columns(source_vectors, label="Source supercell")
    Q = _columns(destination_vectors, label="Destination supercell")
    D = Q * P.inv()
    # Reject invalid/singular matrices, and preserve exact rational strings.
    canonical = canonical_correspondence_rows(_rows(D), direction=direction)
    C = exact_matrix(canonical)
    if direction == "A_TO_M":
        if C * P != Q:
            raise AssertionError("Supercell mapping not preserved")
    elif direction == "M_TO_A":
        if C.inv() * P != Q:
            raise AssertionError("Reverse supercell mapping not preserved")
    else:
        raise ValueError("Direction must be A_TO_M or M_TO_A")
    return PairedSupercell(_rows(P), _rows(Q), _rows(D), canonical, direction)


def entered_matrix_from_canonical(
    canonical_rows: Sequence[Sequence[object]], *, direction: str
) -> MatrixRows:
    """Exact matrix acting SOURCE -> DESTINATION in the selected direction."""
    C = exact_matrix(canonical_rows)
    if direction == "A_TO_M":
        return _rows(C)
    if direction == "M_TO_A":
        return _rows(C.inv())
    raise ValueError("Direction must be A_TO_M or M_TO_A")


def direction_labels(direction: str) -> tuple[str, str]:
    """Left-side source and right-side destination for both directions."""
    if direction == "A_TO_M":
        return "Parent A", "Product M"
    if direction == "M_TO_A":
        return "Product M", "Parent A"
    raise ValueError("Direction must be A_TO_M or M_TO_A")


def seed_paired_vectors(
    canonical_rows: Sequence[Sequence[object]], *, direction: str
) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    """Lossless representation of any invertible rational C as three pairs.

    Identity source vectors are a convenient exact basis, not an assertion that
    those are the shortest physical lattice vectors or that centering is known.
    """
    D = exact_matrix(entered_matrix_from_canonical(canonical_rows, direction=direction))
    identity = tuple(",".join(str(sp.eye(3)[i, j]) for i in range(3)) for j in range(3))
    images = tuple(",".join(str(D[i, j]) for i in range(3)) for j in range(3))
    return identity, images


def plane_image(
    canonical_rows: Sequence[Sequence[object]], parent_plane: Sequence[object]
) -> tuple[str, str, str]:
    """Exact reciprocal-plane map h_M = C_(M<-A)^(-T) h_A."""
    from .input_logic import _exact_scalar
    if len(parent_plane) != 3:
        raise ValueError("A reciprocal plane requires three Miller components")
    C = exact_matrix(canonical_rows)
    image = C.inv().T * sp.Matrix([_exact_scalar(x) for x in parent_plane])
    return tuple(str(item) for item in image)  # type: ignore[return-value]
