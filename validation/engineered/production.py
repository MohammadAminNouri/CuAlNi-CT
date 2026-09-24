from __future__ import annotations

"""Thin production adapter. Keeping imports here makes oracle separation auditable."""

from dataclasses import dataclass
import numpy as np
import sympy as sp

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.ct import (
    analyze_cmc,
    cmc,
    habit_planes_from_analysis,
    normalized_correspondence_metric,
    smc,
)
from cualni_cryst.stretch import stretch_from_metrics
from cualni_cryst.ball_james import single_variant_austenite_habit_solutions
from cualni_cryst.rank_one import solve_rank_one_connection
from cualni_cryst.cofactor import evaluate_cofactor_conditions
from cualni_cryst.ptmc import ptmc_volume_fractions
from cualni_cryst.numerics import DEFAULT_NUMERICAL_POLICY

from .model import CrystalEncoding


@dataclass(frozen=True)
class ProductionResult:
    correspondence: Correspondence
    normalized_metric: np.ndarray
    stretch: np.ndarray
    lambdas: np.ndarray
    mu: np.ndarray
    generalized_eigenvectors: np.ndarray
    generalized_eigen_equation_residual: float
    generalized_metric_orthonormality_residual: float
    cmc_dimensional: np.ndarray
    smc_dimensional: np.ndarray
    ct_analysis: object
    ct_planes_crystal: tuple[np.ndarray, ...]
    bj_solutions: tuple[object, ...]
    tolerance: float


def correspondence(case: CrystalEncoding) -> Correspondence:
    """Represent the actual IEEE-754 correspondence entries exactly in SymPy."""
    entries = []
    for i in range(3):
        row = []
        for j in range(3):
            value = float(case.C_m_from_a[i, j])
            # Preserve exact integers as integers for readability; all other
            # values are exact rationals of their binary64 bit patterns.
            if value.is_integer():
                row.append(sp.Integer(int(value)))
            else:
                num, den = value.as_integer_ratio()
                row.append(sp.Rational(num, den))
        entries.append(row)
    return Correspondence(sp.Matrix(entries), label=case.case_id)


def evaluate(
    case: CrystalEncoding,
    *,
    tol: float = DEFAULT_NUMERICAL_POLICY.exact_eigenvalue,
) -> ProductionResult:
    corr = correspondence(case)
    G = normalized_correspondence_metric(case.M_parent, case.M_product, corr)
    U = stretch_from_metrics(case.M_parent, case.M_product, corr)

    # Compute CMC once and reuse its generalized eigensystem for both the
    # spectrum and habit-plane construction.  This avoids repeated precision
    # escalation and keeps all downstream decisions tied to the same solve.
    analysis = analyze_cmc(case.M_parent, case.M_product, corr, tol=tol)
    if analysis.generalized_mu is None or analysis.eigenvectors_crystal is None:
        raise RuntimeError("CMC analysis did not retain its generalized eigensystem")
    mu = np.asarray(analysis.generalized_mu, float)
    V = np.asarray(analysis.eigenvectors_crystal, float)
    planes = tuple(habit_planes_from_analysis(case.M_parent, analysis, tol))
    bj = tuple(single_variant_austenite_habit_solutions(U, tol=tol))

    return ProductionResult(
        correspondence=corr,
        normalized_metric=G,
        stretch=U,
        lambdas=np.sqrt(mu),
        mu=mu,
        generalized_eigenvectors=V,
        generalized_eigen_equation_residual=float(analysis.generalized_eigen_residual),
        generalized_metric_orthonormality_residual=float(
            analysis.metric_orthonormality_residual
        ),
        cmc_dimensional=cmc(case.M_parent, case.M_product, corr),
        smc_dimensional=smc(case.M_parent, case.M_product, corr),
        ct_analysis=analysis,
        ct_planes_crystal=planes,
        bj_solutions=bj,
        tolerance=float(tol),
    )


def numerical_rank_one(F1: np.ndarray, F2: np.ndarray):
    return solve_rank_one_connection(F1, F2)


def cofactor(U: np.ndarray, a: np.ndarray, n: np.ndarray):
    return evaluate_cofactor_conditions(U, a, n)


def ptmc_roots(U: np.ndarray, a: np.ndarray, n: np.ndarray):
    return ptmc_volume_fractions(U, a, n)
