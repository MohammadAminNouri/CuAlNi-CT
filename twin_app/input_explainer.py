from __future__ import annotations

"""Pure, exact, scientifically cautious input explanations.

Matrix columns map parent coordinate basis vectors into the product coordinate
basis. A correspondence is NOT an orientation relationship or rigid rotation.
"""

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import sympy as sp

from .input_logic import canonical_correspondence_rows, exact_matrix


@dataclass(frozen=True)
class CorrespondencePreview:
    direction: str
    canonical_matrix: tuple[tuple[str, str, str], ...]
    basis_mappings: tuple[tuple[str, str], ...]
    determinant: str
    notes: tuple[str, ...]


def preview_correspondence(
    entered_rows: Sequence[Sequence[object]], *, direction: str
) -> CorrespondencePreview:
    canonical = canonical_correspondence_rows(entered_rows, direction=direction)
    matrix = exact_matrix(canonical)
    columns = tuple(
        (f"Parent [{','.join('1' if i == j else '0' for i in range(3))}]",
         "Product [" + ", ".join(str(matrix[i, j]) for i in range(3)) + "]")
        for j in range(3)
    )
    return CorrespondencePreview(
        direction="A → M" if direction == "A_TO_M" else "M → A (exactly inverted)",
        canonical_matrix=canonical,
        basis_mappings=columns,
        determinant=str(sp.factor(matrix.det())),
        notes=(
            "Each column describes the image of one parent basis direction, expressed in product fractional coordinates.",
            "This is a lattice correspondence, not an orientation relationship, rotation or physical deformation gradient.",
            "Point-group labels and lattice metrics must use the same conventional basis settings as this matrix.",
        ),
    )


def validate_metric_parameters(
    lengths: Sequence[float], angles_deg: Sequence[float], *, near_singular: float = 1e-9
) -> tuple[float, float]:
    """Return cell volume and normalized Gram condition diagnostic.

    Raises on impossible or numerically unsafe cells. Uses no assumed atomic basis.
    """
    if len(lengths) != 3 or len(angles_deg) != 3:
        raise ValueError("Cell requires three lengths and three angles")
    l = np.asarray(lengths, dtype=float)
    ang = np.asarray(angles_deg, dtype=float)
    if not (np.isfinite(l).all() and np.isfinite(ang).all()):
        raise ValueError("All cell parameters must be finite")
    if (l <= 0).any() or (ang <= 0).any() or (ang >= 180).any():
        raise ValueError("Lengths must be positive and angles must lie strictly between 0° and 180°")
    alpha, beta, gamma = np.deg2rad(ang)
    ca, cb, cg = np.cos((alpha, beta, gamma))
    gram_normalized = np.array([[1., cg, cb], [cg, 1., ca], [cb, ca, 1.]])
    eig = np.linalg.eigvalsh(gram_normalized)
    if eig[0] <= near_singular:
        raise ValueError("The cell metric is singular or too close to singular for stable calculations")
    volume = float(np.prod(l) * np.sqrt(float(np.linalg.det(gram_normalized))))
    return volume, float(eig[-1] / eig[0])

_LENGTH_IN_ANGSTROM = {
    "angstrom": 1.0,
    "nanometer": 10.0,
    "picometer": 0.01,
    "micrometer": 1.0e4,
    "meter": 1.0e10,
}


def convert_cell_lengths(
    lengths: Sequence[float], *, from_unit: str, to_unit: str
) -> tuple[float, ...]:
    """Convert lengths so changing a UI unit never changes physical geometry."""
    if from_unit not in _LENGTH_IN_ANGSTROM or to_unit not in _LENGTH_IN_ANGSTROM:
        raise ValueError("Unsupported length unit")
    scale = _LENGTH_IN_ANGSTROM[from_unit] / _LENGTH_IN_ANGSTROM[to_unit]
    result = tuple(float(value)*scale for value in lengths)
    if not result or any(not np.isfinite(x) or x <= 0 for x in result):
        raise ValueError("Converted lengths must remain finite and positive")
    return result
