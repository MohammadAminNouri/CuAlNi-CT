from __future__ import annotations

"""Describe independent M/M rank-one solutions without renaming their physics.

The angle measures *unoriented physical parent-Cartesian interface normals*.
It is not the crystallographic Type-I/II identifier and not an A/M habit angle.
"""

from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np


@dataclass(frozen=True)
class InterfaceComparison:
    normal_separation_deg: float
    relative_tensor_distance: float
    narrative: str


def compare_interface_geometries(first: Any, second: Any) -> InterfaceComparison:
    def data(obj: Any) -> tuple[np.ndarray, np.ndarray]:
        a = np.asarray(obj.a_parent_cartesian, dtype=float)
        n = np.asarray(obj.n_parent_cartesian, dtype=float)
        if a.shape != (3,) or n.shape != (3,) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(n)):
            raise ValueError('Cannot compare invalid twin vectors')
        if np.linalg.norm(a) < 1e-15 or np.linalg.norm(n) < 1e-15:
            raise ValueError('Twin vectors cannot be zero')
        return a, n
    a1, n1 = data(first)
    a2, n2 = data(second)
    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(n1, n2)) / (np.linalg.norm(n1) * np.linalg.norm(n2)), 0, 1))))
    t1, t2 = np.outer(a1, n1), np.outer(a2, n2)
    difference = float(np.linalg.norm(t1 - t2) / max(np.linalg.norm(t1), np.linalg.norm(t2), 1e-15))
    if angle <= 0.1:
        message = ('Their interface-plane normals are essentially parallel, but their rank-one '
                   'tensors differ. The alternatives therefore differ in their deformation, '
                   'not simply in a visually rotated interface plane.')
    else:
        message = (f'The two twin-plane normals are separated by {angle:.1f}° in the same '
                   'parent Cartesian frame (ignoring reversal of a plane normal). '
                   'They are alternative geometric solutions for the same martensite couple.')
    return InterfaceComparison(angle, difference, message)


def interface_choice_description(interface: Any) -> str:
    """Scientific status, not a fabricated Type-I/II classification."""
    branch = interface.representative
    kinds = set(interface.classifications)
    if len(kinds) > 1:
        return 'Twin type unresolved — source classifications disagree'
    if 'unresolved' in str(branch.classification).lower() or 'unresolved' in str(branch.classification_status).lower():
        return 'Twin geometry calculated; Type-I/II check unresolved'
    return str(branch.classification)
