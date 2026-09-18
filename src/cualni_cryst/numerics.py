from __future__ import annotations

"""Central numerical policy for crystallographic calculations.

These tolerances are *computational* defaults, not experimental uncertainties
and not material-science acceptance criteria.  Physical uncertainty must be
stored separately in provenance / measurement data.

The purpose of this module is to eliminate hidden, inconsistent magic numbers
from cross-theory validation and future GUI classification.
"""

from dataclasses import dataclass
from enum import Enum


class ResidualClass(str, Enum):
    """Numerical classification relative to an explicit tolerance policy."""

    PASS = "PASS"
    NEAR = "NEAR"
    FAIL = "FAIL"


@dataclass(frozen=True)
class NumericalPolicy:
    """Named numerical tolerances used by scientific consistency audits.

    Parameters
    ----------
    algebraic
        Relative residual for exact algebraic identities represented in
        floating-point arithmetic.
    representation
        Residual allowed for Metric <-> Cartesian parity checks.
    exact_eigenvalue
        Absolute tolerance used when deciding whether a theoretically exact
        eigenvalue condition such as q_i=0 or lambda_2=1 is satisfied
        numerically.
    rank_one
        Numerical residual for rank-one compatibility solvers.
    projective_angle_deg
        Angular tolerance used for parallel/antiparallel plane or direction
        comparisons.

    Notes
    -----
    These values do not encode experimental precision.  A measured lattice
    parameter with uncertainty must never be declared physically "exact"
    merely because a floating-point residual falls below one of these values.
    """

    algebraic: float = 1.0e-10
    representation: float = 1.0e-9
    exact_eigenvalue: float = 1.0e-8
    rank_one: float = 1.0e-8
    projective_angle_deg: float = 1.0e-7

    def __post_init__(self) -> None:
        values = {
            "algebraic": self.algebraic,
            "representation": self.representation,
            "exact_eigenvalue": self.exact_eigenvalue,
            "rank_one": self.rank_one,
            "projective_angle_deg": self.projective_angle_deg,
        }
        bad = {name: value for name, value in values.items() if value <= 0.0}
        if bad:
            raise ValueError(f"All numerical tolerances must be positive: {bad}")


DEFAULT_NUMERICAL_POLICY = NumericalPolicy()


def classify_residual(
    residual: float,
    tolerance: float,
    *,
    near_factor: float = 100.0,
) -> ResidualClass:
    """Classify a non-negative residual without hiding the raw value.

    PASS
        residual <= tolerance
    NEAR
        tolerance < residual <= near_factor * tolerance
    FAIL
        residual > near_factor * tolerance

    The returned class is only a display/diagnostic aid.  Callers must retain
    and report the raw residual and the exact tolerance used.
    """

    if residual < 0.0:
        raise ValueError("Residual must be non-negative")
    if tolerance <= 0.0:
        raise ValueError("Tolerance must be positive")
    if near_factor <= 1.0:
        raise ValueError("near_factor must be > 1")

    if residual <= tolerance:
        return ResidualClass.PASS
    if residual <= near_factor * tolerance:
        return ResidualClass.NEAR
    return ResidualClass.FAIL
