from __future__ import annotations

"""Cayron 2026 / Zhang equivalent CC2 criterion for one GENERATOR.

Acta Materialia 316 (2026) 122399, page 2, paragraph after SC1–SC3:
Type I: || U^(-1) e || = 1, e normal to the Type-I twinning plane.
Type II: || U e || = 1, e the parent twofold rotation axis.

This is an additional independent CC2-equivalent metric diagnostic, NOT
a replacement for the full Ball–James CC2 rank-one calculation, CC3
inequality, or source-confirmed matching to a physical twin in the other
theory. It is meaningful with the correct reference configuration.
"""
from typing import Any
import numpy as np


def cayron_generator_cc2_equivalent(
    parent_metric: Any, product_metric: Any,
    correspondence_M_from_A: Any,
    construction_route: str, rational_element_parent: Any,
) -> dict:
    MA=np.asarray(parent_metric,float)
    MM=np.asarray(product_metric,float)
    C=np.asarray(correspondence_M_from_A,float)
    if any(m.shape!=(3,3) or not np.isfinite(m).all()
           for m in (MA,MM,C)):
        raise ValueError("Expected finite 3x3 metrics and correspondence")
    if min(np.linalg.eigvalsh(MA))<=0 or min(np.linalg.eigvalsh(MM))<=0:
        raise ValueError("Parent and product metrics must be positive definite")
    if abs(np.linalg.det(C))<1e-12:
        raise ValueError("Singular correspondence")
    if construction_route not in ("I","II"):
        raise ValueError("Only exact Type-I/II twin generators support this condition")
    e=np.asarray(rational_element_parent,float).reshape(3)
    if not np.isfinite(e).all() or np.linalg.norm(e)<1e-14:
        raise ValueError("Invalid reference rational element")
    # A.T A = MA; the physical stretch U acts in this orthonormalized
    # reference frame. The product metric pulled back by C is G.
    A=np.linalg.cholesky(MA).T
    G=C.T@MM@C
    H=np.linalg.solve(A.T,G)@np.linalg.inv(A)
    H=(H+H.T)/2
    if np.linalg.eigvalsh(H)[0]<=0:
        raise ValueError("Nonpositive generalized stretch metric")
    if construction_route=="I":
        # e is a reciprocal plane normal in A crystal coordinates.
        direction=np.linalg.solve(MA,e)
        direction=direction/np.sqrt(float(e@direction))
    else:
        # e is the direct crystallographic twofold rotation axis.
        direction=e/np.sqrt(float(e@MA@e))
    v=A@direction
    v=v/np.linalg.norm(v)
    value=float(v@np.linalg.solve(H,v)) if construction_route=="I" else float(v@H@v)
    lambdas=np.sqrt(np.linalg.eigvalsh(H))
    return {
        "route":construction_route,
        "criterion":(
            "Type-I ||U^-1 e||^2 = 1"
            if construction_route=="I" else "Type-II ||U e||^2 = 1"
        ),
        "squared_norm":value,
        "squared_norm_minus_one":value-1.,
        "absolute_squared_norm_gap":abs(value-1.),
        "reference_unit_cartesian":v.tolist(),
        "lambda2_minus_one":float(lambdas[1]-1.),
        "source":"Cayron 2026 Eqs 1–3 and page 2 Zhang-equivalent CCI/CCII",
        "CC3_verified":False,
        "BJ_physical_branch_assignment_verified":False,
        "scope":(
            "Equivalent CC2 metric condition for this selected exact "
            "Type-I mirror or Type-II twofold axis. The square is used "
            "to avoid unnecessary square roots. CC1 and CC3 must "
            "also hold, and the physical CT/BJ twin assignment is separate."
        ),
    }
