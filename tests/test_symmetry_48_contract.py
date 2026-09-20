import numpy as np
import sympy as sp

from cualni_cryst.lattice import Lattice
from cualni_cryst.symmetry import (
    cubic_full_m3m,
    cubic_m3m_inventory,
    cubic_proper_rotations,
)


def _metric_residual(operations, M):
    scale = max(float(np.linalg.norm(M)), 1.0)
    return max(
        float(
            np.linalg.norm(np.asarray(g, float).T @ M @ np.asarray(g, float) - M)
            / scale
        )
        for g in operations
    )


def test_cubic_m3m_full_48_inventory_is_locked():
    inv = cubic_m3m_inventory()
    assert inv.total == 48
    assert inv.proper == 24
    assert inv.improper == 24
    assert inv.identity == 1
    assert inv.inversion == 1
    assert inv.reflections == 9
    assert inv.proper_twofold_rotations == 9
    assert inv.proper_threefold_rotations == 8
    assert inv.proper_fourfold_rotations == 6
    assert inv.rotoinversions_order4 == 6
    assert inv.rotoinversions_order6 == 8


def test_full_48_and_proper_24_are_distinct_but_both_metric_symmetries():
    full = tuple(cubic_full_m3m())
    proper = tuple(cubic_proper_rotations())
    assert len(full) == 48
    assert len(proper) == 24
    assert all(sp.det(g) == 1 for g in proper)

    M = Lattice.cubic(5.83, length_unit="angstrom").metric()
    assert _metric_residual(full, M) < 1e-12
    assert _metric_residual(proper, M) < 1e-12
