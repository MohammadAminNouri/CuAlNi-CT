from __future__ import annotations

"""Thin production adapter.  Keeping imports here makes oracle separation auditable."""

from dataclasses import dataclass
import numpy as np
import sympy as sp

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.ct import (
    analyze_cmc,
    cmc,
    habit_planes_from_cmc,
    normalized_correspondence_metric,
    smc,
)
from cualni_cryst.stretch import metric_native_stretch_spectrum, stretch_from_metrics
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
    entries = [
        [sp.Rational(str(float(case.C_m_from_a[i, j]))) for j in range(3)]
        for i in range(3)
    ]
    for i in range(3):
        for j in range(3):
            value = case.C_m_from_a[i, j]
            if abs(value - round(value)) < 1e-12:
                entries[i][j] = sp.Integer(int(round(value)))
    return Correspondence(sp.Matrix(entries), label=case.case_id)


def evaluate(
    case: CrystalEncoding,
    *,
    tol: float = DEFAULT_NUMERICAL_POLICY.exact_eigenvalue,
) -> ProductionResult:
    corr = correspondence(case)
    G = normalized_correspondence_metric(case.M_parent, case.M_product, corr)
    U = stretch_from_metrics(case.M_parent, case.M_product, corr)
    spectrum = metric_native_stretch_spectrum(case.M_parent, case.M_product, corr)
    analysis = analyze_cmc(case.M_parent, case.M_product, corr, tol=tol)
    planes = tuple(habit_planes_from_cmc(case.M_parent, case.M_product, corr, tol=tol))
    bj = tuple(single_variant_austenite_habit_solutions(U, tol=tol))
    return ProductionResult(
        correspondence=corr,
        normalized_metric=G,
        stretch=U,
        lambdas=spectrum.lambdas,
        mu=spectrum.mu,
        generalized_eigenvectors=spectrum.eigenvectors_crystal,
        generalized_eigen_equation_residual=spectrum.eigen_equation_residual,
        generalized_metric_orthonormality_residual=spectrum.metric_orthonormality_residual,
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
