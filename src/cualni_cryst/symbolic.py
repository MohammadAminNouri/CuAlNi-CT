from __future__ import annotations

"""Exact symbolic derivations for the reference Cu-Al-Ni transformation branches.

This module exists so the repository can show the algebra to a human reader, not
only return floating-point answers. Every expression here follows from an
explicit correspondence matrix and a conventional lattice metric.
"""

import sympy as sp

from .cualni_models import C_DO3_TO_2H, C_DO3_TO_6M


def monoclinic_metric_symbolic(a, b, c, beta):
    """Monoclinic unique-b direct metric, beta is a SymPy angle expression."""
    return sp.Matrix(
        [
            [a**2, 0, a * c * sp.cos(beta)],
            [0, b**2, 0],
            [a * c * sp.cos(beta), 0, c**2],
        ]
    )


def orthorhombic_metric_symbolic(a, b, c):
    return sp.diag(a**2, b**2, c**2)


def pulled_back_metric(C_m_from_a: sp.Matrix, M_m: sp.Matrix) -> sp.Matrix:
    C = sp.Matrix(C_m_from_a)
    return sp.simplify(C.T * M_m * C)


def do3_to_6m_pulled_metric():
    """Return exact G=C^T M_6M C and normalized CMC for symbolic parameters."""
    a0, a, b, c, beta = sp.symbols("a0 a b c beta", positive=True, real=True)
    MM = monoclinic_metric_symbolic(a, b, c, beta)
    G = sp.simplify(pulled_back_metric(C_DO3_TO_6M, MM))
    D = sp.simplify(G / a0**2 - sp.eye(3))
    return (a0, a, b, c, beta), G, D


def do3_to_2h_pulled_metric():
    """Return exact G=C^T M_2H C and normalized CMC for symbolic parameters."""
    a0, a, b, c = sp.symbols("a0 a b c", positive=True, real=True)
    MM = orthorhombic_metric_symbolic(a, b, c)
    G = sp.simplify(pulled_back_metric(C_DO3_TO_2H, MM))
    D = sp.simplify(G / a0**2 - sp.eye(3))
    return (a0, a, b, c), G, D


def pretty_symbolic_reference() -> str:
    _, G6, D6 = do3_to_6m_pulled_metric()
    _, G2, D2 = do3_to_2h_pulled_metric()
    return "\n".join(
        [
            "DO3 -> 6M: G = C^T M_M C",
            sp.pretty(G6),
            "",
            "DO3 -> 6M: D = G/a0^2 - I",
            sp.pretty(D6),
            "",
            "DO3 -> 2H: G = C^T M_M C",
            sp.pretty(G2),
            "",
            "DO3 -> 2H: D = G/a0^2 - I",
            sp.pretty(D2),
        ]
    )
