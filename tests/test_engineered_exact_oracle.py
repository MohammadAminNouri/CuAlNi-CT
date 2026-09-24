from __future__ import annotations

import numpy as np
import sympy as sp

from validation.engineered.exact_oracle import (
    exact_rational_case,
    exact_generalized_characteristic,
    expected_generalized_characteristic,
    exact_cmc_determinant,
)
from cualni_cryst.correspondence import Correspondence
from cualni_cryst.stretch import metric_native_stretch_spectrum
from cualni_cryst.ct import cmc, analyze_cmc


def test_exact_rational_oracle_generalized_characteristic_matches_physical_singular_values():
    families = [
        (sp.Rational(4,5), sp.Rational(1,1), sp.Rational(6,5)),
        (sp.Rational(3,4), sp.Rational(1,1), sp.Rational(5,4)),
        (sp.Rational(9,10), sp.Rational(1,1), sp.Rational(11,10)),
        (sp.Rational(4,5), sp.Rational(9,10), sp.Rational(6,5)),
    ]
    for lambdas in families:
        case = exact_rational_case(lambdas)
        assert sp.factor(
            exact_generalized_characteristic(case)
            - expected_generalized_characteristic(case)
        ) == 0

        corr = Correspondence(case.C)
        Ma = np.asarray(case.M_parent, float)
        Mm = np.asarray(case.M_product, float)
        spectrum = metric_native_stretch_spectrum(Ma, Mm, corr)
        np.testing.assert_allclose(
            spectrum.lambdas,
            np.array([float(x) for x in sorted(lambdas)]),
            atol=3e-12, rtol=3e-12,
        )


def test_exact_cmc_determinant_is_zero_iff_a_unit_principal_stretch_is_constructed():
    compatible = exact_rational_case((sp.Rational(4,5), sp.Integer(1), sp.Rational(6,5)))
    incompatible = exact_rational_case((sp.Rational(4,5), sp.Rational(9,10), sp.Rational(6,5)))
    assert exact_cmc_determinant(compatible) == 0
    assert exact_cmc_determinant(incompatible) != 0

    corr = Correspondence(compatible.C)
    ana = analyze_cmc(
        np.asarray(compatible.M_parent,float),
        np.asarray(compatible.M_product,float),
        corr, tol=1e-11,
    )
    assert ana.exact_compatible
    assert ana.degeneracy_order == 1
