
import numpy as np

from cualni_cryst.invariant_line import (
    solve_direct_invariant_line,
    solve_reciprocal_invariant_line,
)
from cualni_cryst.lattice import Lattice


def test_identical_direct_planar_correspondence_has_exact_invariant_branch():
    lattice = Lattice.cubic(3.0)
    report = solve_direct_invariant_line(
        lattice,
        lattice,
        parent_plane=(0, 0, 1),
        product_plane=(0, 0, 1),
        parent_vectors=((1, 0, 0), (0, 1, 0)),
        product_vectors=((1, 0, 0), (0, 1, 0)),
    )
    assert report.solutions
    assert min(item.invariant_residual for item in report.solutions) < 1e-12
    assert min(item.determinant_residual for item in report.solutions) < 1e-12
    assert min(abs(item.rotation_about_constraint_deg) for item in report.solutions) < 1e-10


def test_identical_reciprocal_correspondence_has_exact_invariant_branch():
    lattice = Lattice.cubic(3.0)
    report = solve_reciprocal_invariant_line(
        lattice,
        lattice,
        parent_zone_axis=(0, 0, 1),
        product_zone_axis=(0, 0, 1),
        parent_g_vectors=((1, 0, 0), (0, 1, 0)),
        product_g_vectors=((1, 0, 0), (0, 1, 0)),
    )
    assert report.solutions
    assert min(item.invariant_residual for item in report.solutions) < 1e-12
    assert min(item.determinant_residual for item in report.solutions) < 1e-12
