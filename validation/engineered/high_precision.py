from __future__ import annotations

"""High-precision differential helpers using exact IEEE-754 binary64 inputs."""

import numpy as np
import sympy as sp

from .certified_binary64 import canonical_metric_binary64


def _rat(x: float) -> sp.Rational:
    num, den = float(x).as_integer_ratio()
    return sp.Rational(num, den)


def generalized_eigen_polynomial_high_precision(
    Ma: np.ndarray, Mm: np.ndarray, C: np.ndarray, digits: int = 80
) -> list[complex]:
    mu = sp.symbols("mu")
    Ma = canonical_metric_binary64(Ma)
    Mm = canonical_metric_binary64(Mm)
    A = sp.Matrix([[_rat(x) for x in row] for row in Ma])
    M = sp.Matrix([[_rat(x) for x in row] for row in Mm])
    X = sp.Matrix([[_rat(x) for x in row] for row in C])
    poly = sp.Poly(sp.expand((X.T * M * X - mu * A).det()), mu)
    roots = sp.nroots(poly, n=digits, maxsteps=500)
    vals = sorted([complex(r) for r in roots], key=lambda z: z.real)
    return vals


def conditioning_diagnostic(Ma, Mm, C, production_mu, digits: int = 80) -> dict:
    roots = generalized_eigen_polynomial_high_precision(Ma, Mm, C, digits=digits)
    max_imag = max(abs(z.imag) for z in roots)
    hp_mu = np.array([z.real for z in roots], dtype=float)
    prod_mu = np.sort(np.asarray(production_mu, float))
    hp_mu = np.sort(hp_mu)
    rel = np.abs(prod_mu - hp_mu) / np.maximum(np.abs(hp_mu), np.finfo(float).tiny)
    Ma = canonical_metric_binary64(Ma)
    Mm = canonical_metric_binary64(Mm)
    G = C.T @ Mm @ C

    backward = []
    scale = max(np.linalg.norm(G), np.linalg.norm(Ma), 1.0)
    for mu in prod_mu:
        backward.append(float(np.linalg.svd(G - mu * Ma, compute_uv=False)[-1] / scale))

    return {
        "hp_roots": roots,
        "hp_mu": hp_mu,
        "production_mu": prod_mu,
        "max_imag": float(max_imag),
        "max_relative_forward_error": float(np.max(rel)),
        "relative_forward_errors": rel,
        "max_pencil_backward_residual": float(max(backward)),
        "pencil_backward_residuals": np.array(backward),
        "condition_Ma": float(np.linalg.cond(Ma)),
        "condition_Mm": float(np.linalg.cond(Mm)),
        "condition_C": float(np.linalg.cond(C)),
    }
