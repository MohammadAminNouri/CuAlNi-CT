from __future__ import annotations

"""Reference formulas transcribed from James & Hane (Acta Mater. 48, 2000).

The functions here are deliberately small and explicit.  They are used as an
*independent reference implementation* against which metric/correspondence
calculations are unit-tested.  This is important: the code should not validate
itself by comparing two functions that share the same derivation.
"""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class CubeEdgeParameters:
    alpha: float
    beta: float
    gamma: float
    rho: float
    sigma: float
    tau: float
    denominator: float


def cube_edge_6m_parameters(a0: float, a: float, b: float, c: float, beta_deg: float) -> CubeEdgeParameters:
    r"""James-Hane Eq. (10) parameters for cubic -> monoclinic 6M cube-edge variants.

    The 6M lattice convention used in their Section 3 gives

        alpha = sqrt(2) a/a0
        beta  = b/a0
        gamma = sqrt(2) c/(3 a0)

    and theta (= beta monoclinic angle in modern notation) is the angle between
    the martensite a and c axes.
    """
    if min(a0, a, b, c) <= 0:
        raise ValueError("Lattice lengths must be positive")
    th = math.radians(beta_deg)
    alpha = math.sqrt(2.0) * a / a0
    beta = b / a0
    gamma = math.sqrt(2.0) * c / (3.0 * a0)
    D = math.sqrt(alpha**2 + gamma**2 + 2.0 * alpha * gamma * math.sin(th))
    rho = (alpha**2 + gamma**2 + 2.0 * alpha * gamma * (math.sin(th) + math.cos(th))) / (2.0 * D)
    sigma = (alpha**2 - gamma**2) / (2.0 * D)
    tau = (alpha**2 + gamma**2 + 2.0 * alpha * gamma * (math.sin(th) - math.cos(th))) / (2.0 * D)
    return CubeEdgeParameters(alpha, beta, gamma, rho, sigma, tau, D)


def cube_edge_6m_reference_U(a0: float, a: float, b: float, c: float, beta_deg: float) -> np.ndarray:
    p = cube_edge_6m_parameters(a0, a, b, c, beta_deg)
    return np.array([[p.beta, 0.0, 0.0], [0.0, p.rho, p.sigma], [0.0, p.sigma, p.tau]])


def cube_edge_6m_variants(a0: float, a: float, b: float, c: float, beta_deg: float) -> list[np.ndarray]:
    """The twelve cube-edge stretch variants of James-Hane Eq. (10)."""
    p = cube_edge_6m_parameters(a0, a, b, c, beta_deg)
    B, r, s, t = p.beta, p.rho, p.sigma, p.tau
    return [
        np.array([[B,0,0],[0,r, s],[0, s,t]], float),
        np.array([[B,0,0],[0,r,-s],[0,-s,t]], float),
        np.array([[B,0,0],[0,t, s],[0, s,r]], float),
        np.array([[B,0,0],[0,t,-s],[0,-s,r]], float),
        np.array([[r,0, s],[0,B,0],[ s,0,t]], float),
        np.array([[r,0,-s],[0,B,0],[-s,0,t]], float),
        np.array([[t,0, s],[0,B,0],[ s,0,r]], float),
        np.array([[t,0,-s],[0,B,0],[-s,0,r]], float),
        np.array([[r, s,0],[ s,t,0],[0,0,B]], float),
        np.array([[r,-s,0],[-s,t,0],[0,0,B]], float),
        np.array([[t, s,0],[ s,r,0],[0,0,B]], float),
        np.array([[t,-s,0],[-s,r,0],[0,0,B]], float),
    ]


@dataclass(frozen=True)
class OrthorhombicParameters:
    alpha: float
    beta: float
    gamma: float


def orthorhombic_2h_parameters(a0: float, a: float, b: float, c: float) -> OrthorhombicParameters:
    r"""Stretch ratios for the Cu-Al-Ni cubic->orthorhombic face-diagonal family.

    In the correspondence used by the Eq. (9) family, martensite a and c lie
    along parent face diagonals and martensite b along a parent cube edge, hence

        alpha = sqrt(2) a/a0, beta = b/a0, gamma = sqrt(2) c/a0.

    This is also independently verified in the package by pulling the daughter
    metric back with the source-derived 2H correspondence.
    """
    if min(a0, a, b, c) <= 0:
        raise ValueError("Lattice lengths must be positive")
    return OrthorhombicParameters(math.sqrt(2)*a/a0, b/a0, math.sqrt(2)*c/a0)


def orthorhombic_2h_reference_U(a0: float, a: float, b: float, c: float) -> np.ndarray:
    p = orthorhombic_2h_parameters(a0, a, b, c)
    apg = 0.5*(p.alpha+p.gamma)
    amg = 0.5*(p.alpha-p.gamma)
    return np.array([[p.beta,0,0],[0,apg,amg],[0,amg,apg]], float)


def orthorhombic_2h_variants(a0: float, a: float, b: float, c: float) -> list[np.ndarray]:
    """Six face-diagonal cubic->orthorhombic variants, James-Hane Eq. (9)."""
    p = orthorhombic_2h_parameters(a0,a,b,c)
    B = p.beta; A = 0.5*(p.alpha+p.gamma); D = 0.5*(p.alpha-p.gamma)
    return [
        np.array([[B,0,0],[0,A, D],[0, D,A]],float),
        np.array([[B,0,0],[0,A,-D],[0,-D,A]],float),
        np.array([[A,0, D],[0,B,0],[ D,0,A]],float),
        np.array([[A,0,-D],[0,B,0],[-D,0,A]],float),
        np.array([[A, D,0],[ D,A,0],[0,0,B]],float),
        np.array([[A,-D,0],[-D,A,0],[0,0,B]],float),
    ]


def m18r_to_6m(a: float, c_m18r: float, beta_m18r_deg: float) -> tuple[float, float]:
    r"""James-Hane Eq. (24): convert M18R c, beta to the corresponding 6M cell.

    a and b are unchanged by this cell change.  This is a change of cell, NOT a
    normalization and NOT a change of length units.
    Returns ``(c_6m, beta_6m_deg)``.
    """
    th = math.radians(beta_m18r_deg)
    th6 = th + math.atan2(a, c_m18r * math.sin(th))
    c6 = c_m18r * math.sin(th) / (3.0 * math.sin(th6))
    return c6, math.degrees(th6)


def six_m_exact_compatibility_angles(a0: float, a: float, c6m: float) -> tuple[float, float]:
    r"""James-Hane Eq. (25) exact single-variant A/M compatibility angles.

    Returns the acute and obtuse solutions in degrees.  Real solutions require
    the right-hand side of cos^2(theta) to lie in [0,1].
    """
    alpha = math.sqrt(2.0)*a/a0
    gamma = math.sqrt(2.0)*c6m/(3.0*a0)
    rhs = ((1-alpha**2)*(1-gamma**2))/(alpha**2*gamma**2)
    if rhs < -1e-12 or rhs > 1+1e-12:
        raise ValueError(f"No real exact-compatibility beta for rhs={rhs}")
    rhs = min(1.0,max(0.0,rhs))
    acute = math.degrees(math.acos(math.sqrt(rhs)))
    return acute, 180.0-acute
