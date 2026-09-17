import numpy as np
import sympy as sp
from cualni_cryst.lattice import Lattice
from cualni_cryst.correspondence import Correspondence
from cualni_cryst.ct import cmc, normalized_cmc
from cualni_cryst.stretch import stretch_from_metrics, principal_stretches


def test_cmc_stretch_bridge_cubic_parent():
    MA = Lattice.cubic(2.0).metric()
    # diagonal daughter metric with identity correspondence is enough to test the algebraic bridge
    MM = np.diag([1.8**2, 2.0**2, 2.3**2])
    C = Correspondence(sp.eye(3))
    U = stretch_from_metrics(MA, MM, C)
    lam, _ = principal_stretches(U)
    D = normalized_cmc(MA, MM, C)
    d = np.linalg.eigvalsh(D)
    assert np.allclose(np.sort(d), np.sort(lam**2 - 1.0), atol=1e-10)
    assert np.allclose(cmc(MA, MM, C), 4.0 * D, atol=1e-10)
