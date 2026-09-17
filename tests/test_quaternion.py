import numpy as np

from cualni_cryst.lattice import Lattice
from cualni_cryst.quaternion import crystallographic_quaternion_product


def test_cartesian_limit_matches_hamilton_product():
    M = Lattice.cubic(1.0).metric()
    q1 = (0.8, np.array([0.6,0,0]))
    q2 = (0.8, np.array([0,0.6,0]))
    q = crystallographic_quaternion_product(q1, q2, M)
    assert np.isclose(q.scalar, 0.64)
    assert np.allclose(q.vector, [0.48,0.48,0.36])
