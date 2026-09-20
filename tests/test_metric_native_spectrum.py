import numpy as np
import sympy as sp

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.lattice import Lattice
from cualni_cryst.stretch import (
    metric_native_stretch_spectrum,
    principal_stretches,
    stretch_from_metrics,
)


def test_metric_native_generalized_eigenproblem_matches_whitened_stretch_non_cubic():
    parent = Lattice(
        3.2,
        4.1,
        5.3,
        77.0,
        103.0,
        66.0,
        length_unit="angstrom",
    )
    product = Lattice(
        2.9,
        4.4,
        5.7,
        81.0,
        97.0,
        72.0,
        length_unit="angstrom",
    )
    C = Correspondence(
        sp.Matrix([[1, 1, 0], [0, 1, 1], [1, 0, 1]]),
        label="generic non-cubic regression",
    )

    native = metric_native_stretch_spectrum(
        parent.metric(),
        product.metric(),
        C,
    )
    U = stretch_from_metrics(parent.metric(), product.metric(), C)
    lambdas, _ = principal_stretches(U)

    assert np.allclose(native.lambdas, lambdas, atol=1e-11, rtol=1e-11)
    assert native.metric_orthonormality_residual < 1e-10
    assert native.eigen_equation_residual < 1e-10


def test_metric_native_spectrum_rejects_singular_or_non_spd_inputs():
    C = Correspondence(sp.eye(3))
    good = np.eye(3)

    bad = np.diag([1.0, 1.0, -1.0])
    try:
        metric_native_stretch_spectrum(bad, good, C)
    except ValueError:
        pass
    else:
        raise AssertionError("non-SPD parent metric must be rejected")
