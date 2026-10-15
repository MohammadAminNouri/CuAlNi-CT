"""Independent published Cu-Al-Ni 6M metric diagnostics.

James & Hane, Acta Materialia 48 (2000), 197-222, eqs (24)-(25), Table 4.
https://aem.umn.edu/~james/research/Publications_Files/PDF/67.pdf

These algebraic diagnostics do NOT manufacture a correspondence, OR, twin,
habit plane, phase identity or experimental validation. Inputs are physical
lattice lengths in the same unit, angles in degrees; the parent convention
in eq (25) is DO3 (a0), NOT an interchangeable B2 subcell.
"""
from __future__ import annotations
from math import acos, atan, cos, degrees, isfinite, radians, sin, sqrt
from typing import Any

class LiteratureMetricError(ValueError):
    pass


def _positive(name: str, value: float) -> float:
    x = float(value)
    if not isfinite(x) or x <= 0:
        raise LiteratureMetricError(f"{name} must be finite and strictly positive")
    return x


def m18r_to_6m_james_hane(*, a: float, c_m18r: float,
                          theta_m18r_deg: float) -> dict[str, Any]:
    """Eq (24): change of monoclinic M18R to 6M *cell descriptions*.

    This is NOT the generic unique-b / gamma-axis permutation, not an
    independently verified M18R correspondence and not identification of 6R.
    """
    a = _positive("M18R a", a)
    c = _positive("M18R c", c_m18r)
    theta = float(theta_m18r_deg)
    if not isfinite(theta) or not 0 < theta < 180:
        raise LiteratureMetricError("M18R angle must be within (0,180) degrees")
    s = sin(radians(theta))
    angle6 = theta + degrees(atan(a / (c * s)))
    if not 0 < angle6 < 180 or not isfinite(angle6):
        raise LiteratureMetricError("M18R to 6M cell relation gives invalid angle")
    c6 = c * s / (3 * sin(radians(angle6)))
    if not isfinite(c6) or c6 <= 0:
        raise LiteratureMetricError("M18R to 6M gives an invalid c axis")
    return {"c_6m": c6, "theta_6m_deg": angle6,
            "invariant_layer_height_ratio": c6*sin(radians(angle6)) /
                                            (c*s/3),
            "source_equation": "James-Hane-2000-Eq24",
            "requires_transformed_correspondence": True,
            "experimental_phase_identity_verified": False}


def six_m_compatibility_james_hane(*, parent_do3_a0: float, a_6m: float,
                                    c_6m: float, theta_6m_deg: float) -> dict[str, Any]:
    """Eq (25) (DO3 6M cube-edge): cos²(theta)=(1-alpha²)(1-gamma²)/(alpha² gamma²).

    This is the single-variant A/M geometric compatibility condition in the
    PAPER'S 6M correspondence, not a generic criterion for arbitrary C, 18R
    settings, twinning mode, habit type, or measured EBSD.
    """
    a0 = _positive("DO3 parent a0", parent_do3_a0)
    a = _positive("6M a", a_6m)
    c = _positive("6M c", c_6m)
    theta = float(theta_6m_deg)
    if not isfinite(theta) or not 0 < theta < 180:
        raise LiteratureMetricError("6M monoclinic angle invalid")
    alpha = sqrt(2) * a/a0
    gamma = sqrt(2) * c/(3*a0)
    rhs = ((1-alpha**2)*(1-gamma**2))/(alpha**2*gamma**2)
    measured = cos(radians(theta))**2
    mathematically_admissible = 0 <= rhs <= 1
    predicted_theta = (
        degrees(acos(-sqrt(rhs))) if mathematically_admissible and theta >= 90 else
        degrees(acos(sqrt(rhs))) if mathematically_admissible else None
    )
    return {
        "source_equation": "James-Hane-2000-Eq25",
        "parent_setting": "DO3",
        "alpha_dimensionless": alpha, "gamma_dimensionless": gamma,
        "measured_cos2_theta": measured, "required_cos2_theta": rhs,
        "cos2_mismatch": measured-rhs,
        "admissible_compatibility_angle": mathematically_admissible,
        "nearest_branch_compatible_theta_deg": predicted_theta,
        "observed_minus_compatible_angle_deg":
            theta-predicted_theta if predicted_theta is not None else None,
        "exactly_compatible_with_reported_rounded_input":
            mathematically_admissible and abs(measured-rhs) <= 1e-10,
        "twinning_or_habit_solutions_computed": False,
        "source_correspondence_not_loaded": True,
        "experimental_verification": "NOT_ESTABLISHED",
    }


def cube_edge_stretch_james_hane(*, parent_do3_a0: float,
                                  a_6m: float,b_6m: float,c_6m: float,
                                  theta_6m_deg: float):
    """Eq (10) of James-Hane: computed U1 tensor (cube-edge geometry).

    Returns a Cartesian stretch matrix in the SOURCE'S cubic gauge, NOT a
    full atomic correspondence, an EBSD orientation relationship or a twin.
    In particular, this must not be substituted for an unknown experimental C.
    """
    from math import sin,cos,radians
    import numpy as np
    a0=_positive("DO3 parent a0",parent_do3_a0)
    a=_positive("6M a",a_6m)
    b=_positive("6M b",b_6m)
    c=_positive("6M c",c_6m)
    theta=float(theta_6m_deg)
    if not isfinite(theta) or not 0<theta<180:
        raise LiteratureMetricError("6M monoclinic angle invalid")
    alpha=sqrt(2)*a/a0
    beta=b/a0
    gamma=sqrt(2)*c/(3*a0)
    sine=sin(radians(theta));cosine=cos(radians(theta))
    length=sqrt(alpha**2+gamma**2+2*alpha*gamma*sine)
    rho=(alpha**2+gamma**2+2*alpha*gamma*(sine+cosine))/(2*length)
    sigma=(alpha**2-gamma**2)/(2*length)
    tau=(alpha**2+gamma**2+2*alpha*gamma*(sine-cosine))/(2*length)
    U=np.array(((beta,0.,0.),(0.,rho,sigma),(0.,sigma,tau)),dtype=float)
    if np.linalg.eigvalsh(U)[0] <= 0:
        raise LiteratureMetricError("Calculated source-gauge stretch is not positive definite")
    return U


def proper_cubic_stretch_orbit(U,*,atol=1e-10):
    """Calculate the proper cubic parent conjugacy orbit, without preset counts.

    Finite cubic rotations act on the *same* source-gauge stretch. The number of
    physically different tensor variants is derived; no answer table is used.
    """
    from itertools import permutations,product
    import numpy as np
    X=np.asarray(U,float)
    if X.shape!=(3,3) or not np.isfinite(X).all() or not np.allclose(X,X.T,atol=1e-13,rtol=0):
        raise LiteratureMetricError("Finite symmetric 3x3 stretch required")
    if np.linalg.eigvalsh(X)[0]<=0:
        raise LiteratureMetricError("Proper cubic orbit requires a positive stretch")
    if not np.isfinite(atol) or not 0<atol<=1e-5:
        raise LiteratureMetricError("Invalid equality tolerance")
    unique=[]
    for ordering in permutations(range(3)):
        for signs in product((-1,1),repeat=3):
            Q=np.zeros((3,3))
            for row,column in enumerate(ordering):
                Q[row,column]=signs[row]
            if round(np.linalg.det(Q))!=1:
                continue
            V=Q@X@Q.T
            if not any(np.linalg.norm(V-old,ord="fro") <= atol for old in unique):
                unique.append(V)
    return tuple(unique)
