import numpy as np
from cualni_cryst.lattice import Lattice, normalize_direct, normalize_plane, plane_to_unit_normal


def test_cubic_metric():
    M = Lattice.cubic(5.0).metric()
    assert np.allclose(M, 25.0 * np.eye(3))


def test_monoclinic_metric():
    M = Lattice.monoclinic_unique_b(4, 5, 6, 100).metric()
    assert np.isclose(M[0, 2], 4*6*np.cos(np.deg2rad(100)))
    assert np.isclose(M[1, 0], 0)


def test_direct_and_plane_normalization():
    M = Lattice.monoclinic_unique_b(4, 5, 6, 100).metric()
    u = normalize_direct([1,2,3], M)
    assert np.isclose(u @ M @ u, 1.0)
    p = normalize_plane([1,0,1], M)
    assert np.isclose(p @ np.linalg.inv(M) @ p, 1.0)
    n = plane_to_unit_normal(p, M)
    assert np.isclose(n @ M @ n, 1.0)
