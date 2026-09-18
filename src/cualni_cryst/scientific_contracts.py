from __future__ import annotations

"""Cross-theory mathematical consistency contracts.

This module does not add a new phase-transformation theory.  It verifies that
the independently implemented representations/theories already in CuAlNi-CT
satisfy the identities that *must* connect them when they are fed the same
physical crystallographic state.

The central contracts are:

1. normalized CMC is the parent-metric congruence of dimensional CMC;
2. normalized CMC = U^2 - I;
3. eigenvalues(normalized CMC) = lambda_i^2 - 1;
4. Cayron exact A/M metric compatibility and Ball-James lambda_2=1 give the
   same exact numerical classification for a single variant;
5. parent-whitened SMC = I - U^-2;
6. all supported Cartesian frame conventions describe the same physical map.

Passing these contracts does not prove that a theory matches experiment.
It proves that the software's implementations of the connected mathematical
objects are mutually coherent.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .correspondence import Correspondence
from .ct import (
    analyze_cmc,
    cmc,
    normalized_cmc,
    normalized_correspondence_metric,
    smc,
)
from .lattice import Lattice, metric_inv_sqrt, metric_sqrt
from .numerics import DEFAULT_NUMERICAL_POLICY, NumericalPolicy, classify_residual
from .representation import CartesianConvention, RepresentationBridge
from .stretch import principal_stretches, stretch_from_metrics


def _relative_residual(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    scale = max(float(np.linalg.norm(lhs)), float(np.linalg.norm(rhs)), 1.0)
    return float(np.linalg.norm(lhs - rhs) / scale)


@dataclass(frozen=True)
class TheoryConsistencyAudit:
    """Audit result for one parent/product/correspondence state."""

    normalized_cmc_from_dimensional_residual: float
    cmc_vs_stretch_residual: float
    cmc_eigenvalue_vs_lambda_residual: float
    lambda2_residual: float
    cmc_nearest_zero_residual: float
    exact_compatibility_agreement: bool
    smc_duality_residual: float
    representation_max_residual: float
    representation_pairs_checked: int
    maximum_algebraic_residual: float
    passed: bool
    policy: NumericalPolicy

    @property
    def algebraic_class(self) -> str:
        return classify_residual(
            self.maximum_algebraic_residual,
            self.policy.algebraic,
        ).value

    @property
    def representation_class(self) -> str:
        return classify_residual(
            self.representation_max_residual,
            self.policy.representation,
        ).value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assert_passed(self) -> None:
        if not self.passed:
            raise AssertionError(
                "Cross-theory consistency audit failed: "
                f"algebraic={self.maximum_algebraic_residual:.3e}, "
                f"representation={self.representation_max_residual:.3e}, "
                f"exact_classification_agreement={self.exact_compatibility_agreement}"
            )


def audit_core_theory_consistency(
    parent: Lattice,
    product: Lattice,
    correspondence: Correspondence,
    *,
    policy: NumericalPolicy = DEFAULT_NUMERICAL_POLICY,
) -> TheoryConsistencyAudit:
    """Audit mathematical identities linking CT, stretch theory and Cartesian frames.

    This function is deliberately generic: it does not assume cubic parent,
    monoclinic product, Cu-Al-Ni, or a particular literature parameter set.
    """

    M_a = parent.metric()
    M_m = product.metric()

    # --- Contract 1: dimensional CMC -> normalized CMC ---------------------
    W = metric_inv_sqrt(M_a)
    cmc_dim = cmc(M_a, M_m, correspondence)
    D = normalized_cmc(M_a, M_m, correspondence)
    D_from_dimensional = W @ cmc_dim @ W
    normalized_cmc_from_dimensional_residual = _relative_residual(
        D,
        D_from_dimensional,
    )

    # --- Contracts 2-3: normalized CMC <-> stretch ------------------------
    U = stretch_from_metrics(M_a, M_m, correspondence)
    U2_minus_I = U.T @ U - np.eye(3)
    cmc_vs_stretch_residual = _relative_residual(D, U2_minus_I)

    lambdas, _ = principal_stretches(U)
    q = np.sort(np.linalg.eigvalsh(0.5 * (D + D.T)))
    q_from_lambda = np.sort(lambdas**2 - 1.0)
    cmc_eigenvalue_vs_lambda_residual = _relative_residual(q, q_from_lambda)

    # --- Contract 4: exact CT A/M classification <-> lambda_2 = 1 --------
    ct_analysis = analyze_cmc(
        M_a,
        M_m,
        correspondence,
        tol=policy.exact_eigenvalue,
    )
    lambda2_residual = float(abs(lambdas[1] - 1.0))
    bj_exact = bool(lambda2_residual <= policy.exact_eigenvalue)
    exact_compatibility_agreement = bool(ct_analysis.exact_compatible == bj_exact)

    # The nearest normalized-CMC zero residual is reported, never hidden.
    cmc_nearest_zero_residual = float(ct_analysis.nearest_zero_residual)

    # --- Contract 5: SMC duality ------------------------------------------
    #
    # Ghat = M_A^-1/2 C^T M_M C M_A^-1/2 = U^2
    #
    # S M_C S = I - Ghat^-1
    #
    # with S = M_A^(1/2).
    S = metric_sqrt(M_a)
    SMC = smc(M_a, M_m, correspondence)
    Ghat = normalized_correspondence_metric(M_a, M_m, correspondence)
    lhs_smc = S @ SMC @ S
    rhs_smc = np.eye(3) - np.linalg.inv(Ghat)
    smc_duality_residual = _relative_residual(lhs_smc, rhs_smc)

    # --- Contract 6: all supported Cartesian representations --------------
    representation_residuals: list[float] = []
    for parent_convention in CartesianConvention:
        for product_convention in CartesianConvention:
            bridge = RepresentationBridge(
                parent,
                product,
                correspondence,
                parent_convention,
                product_convention,
            )
            representation_residuals.append(bridge.audit().maximum_residual)

    representation_max_residual = max(representation_residuals, default=0.0)
    representation_pairs_checked = len(representation_residuals)

    algebraic_residuals = (
        normalized_cmc_from_dimensional_residual,
        cmc_vs_stretch_residual,
        cmc_eigenvalue_vs_lambda_residual,
        smc_duality_residual,
    )
    maximum_algebraic_residual = max(algebraic_residuals)

    passed = bool(
        maximum_algebraic_residual <= policy.algebraic
        and representation_max_residual <= policy.representation
        and exact_compatibility_agreement
    )

    return TheoryConsistencyAudit(
        normalized_cmc_from_dimensional_residual=(
            normalized_cmc_from_dimensional_residual
        ),
        cmc_vs_stretch_residual=cmc_vs_stretch_residual,
        cmc_eigenvalue_vs_lambda_residual=cmc_eigenvalue_vs_lambda_residual,
        lambda2_residual=lambda2_residual,
        cmc_nearest_zero_residual=cmc_nearest_zero_residual,
        exact_compatibility_agreement=exact_compatibility_agreement,
        smc_duality_residual=smc_duality_residual,
        representation_max_residual=representation_max_residual,
        representation_pairs_checked=representation_pairs_checked,
        maximum_algebraic_residual=maximum_algebraic_residual,
        passed=passed,
        policy=policy,
    )
