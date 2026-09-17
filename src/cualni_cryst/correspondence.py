from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class Correspondence:
    """Lattice correspondence with an explicit mapping convention.

    Physical meaning:
        A parent direct-space direction u_A is transformed to the daughter basis by
        u_M = C_M_from_A @ u_A. Plane covectors are reciprocal-space objects and obey
        p_M = C_M_from_A^{-T} p_A.

    Equation:
        C_M_from_A : A -> M
        C_A_from_M = C_M_from_A^{-1}

    Coordinate convention:
        This package keeps the direct-space and reciprocal-space transforms separate.
        A bare matrix symbol is never used for both operations; the explicit names
        below are required in all scientific code paths.

    Input units:
        dimensionless lattice transformation matrix.

    Output units:
        dimensionless lattice transformation matrix.

    Source/theory:
        This is the exact basis transformation used for the DO3 -> 6M reference branch.
        It is checked against the independently published James-Hane stretch family.

    Limitations:
        The code stores the lattice relation itself, not a claim about the full EBSD
        sample orientation or a phase-specific fitted parameter set.
    """

    C_m_from_a: sp.Matrix
    label: str = ""
    source: str = ""
    derivation: str = ""

    def __post_init__(self) -> None:
        C = sp.Matrix(self.C_m_from_a)
        if C.shape != (3, 3) or sp.simplify(C.det()) == 0:
            raise ValueError("Correspondence must be an invertible 3x3 matrix.")
        object.__setattr__(self, "C_m_from_a", C)

    @property
    def C_M_from_A(self) -> sp.Matrix:
        """Explicit direct-space parent-to-daughter basis matrix."""
        return sp.simplify(self.C_m_from_a)

    @property
    def C_A_from_M(self) -> sp.Matrix:
        """Explicit daughter-to-parent basis matrix, the exact inverse."""
        return sp.simplify(self.C_m_from_a.inv())

    @property
    def C_a_from_m(self) -> sp.Matrix:
        return sp.simplify(self.C_m_from_a.inv())

    def map_direction_A_to_M(self, u_A) -> sp.Matrix:
        """Transform a direct-space parent direction into daughter coordinates."""
        return sp.simplify(self.C_M_from_A * sp.Matrix(u_A))

    def map_direction_a_to_m(self, u_a) -> sp.Matrix:
        return self.map_direction_A_to_M(u_a)

    def map_direction_M_to_A(self, u_M) -> sp.Matrix:
        """Transform a direct-space daughter direction back to parent coordinates."""
        return sp.simplify(self.C_A_from_M * sp.Matrix(u_M))

    def map_direction_m_to_a(self, u_m) -> sp.Matrix:
        return self.map_direction_M_to_A(u_m)

    def map_plane_A_to_M(self, p_A) -> sp.Matrix:
        """Transform a reciprocal-space parent plane covector to daughter coordinates."""
        return sp.simplify(self.C_M_from_A.inv().T * sp.Matrix(p_A))

    def map_plane_a_to_m(self, p_a) -> sp.Matrix:
        return self.map_plane_A_to_M(p_a)

    def map_plane_M_to_A(self, p_M) -> sp.Matrix:
        """Transform a reciprocal-space daughter plane covector to parent coordinates."""
        return sp.simplify(self.C_M_from_A.T * sp.Matrix(p_M))

    def map_plane_m_to_a(self, p_m) -> sp.Matrix:
        return self.map_plane_M_to_A(p_m)

    def pullback_product_metric(self, M_m: np.ndarray) -> np.ndarray:
        C = np.array(self.C_m_from_a, dtype=float)
        return C.T @ np.asarray(M_m, dtype=float) @ C

    def intercorrespondence_from_parent_symmetry(self, g_a: sp.Matrix) -> sp.Matrix:
        """C_int = C_{M<-A} g_A C_{A<-M}; acts in martensite coordinates."""
        return sp.simplify(self.C_M_from_A * sp.Matrix(g_a) * self.C_A_from_M)

    def verify_direction_pair(self, u_a, u_m) -> bool:
        lhs = self.map_direction_a_to_m(u_a)
        rhs = sp.Matrix(u_m)
        if rhs == sp.zeros(3, 1):
            raise ValueError("Target direction cannot be zero")
        # Parallelism, not necessarily identical integer scaling.
        return sp.Matrix(lhs).cross(rhs) == sp.zeros(3, 1)


def validate_correspondence_reference(
    corr: Correspondence,
    *,
    u_A=None,
    u_M=None,
    p_A=None,
    p_M=None,
) -> dict[str, object]:
    """Audit a correspondence reference against the direct/reciprocal convention.

    This is intentionally a minimal reference validator: it checks the exact inverse,
    determinant, and first-principles direct/reciprocal round-trips using the
    identities p^T u = 0 and C^{-T}.
    """
    C = corr.C_M_from_A
    Ci = corr.C_A_from_M
    checks: dict[str, object] = {
        "determinant_nonzero": sp.simplify(C.det()) != 0,
        "matrix_inverse_exact": sp.simplify(C * Ci - sp.eye(3)) == sp.zeros(3),
        "matrix_inverse_exact_reverse": sp.simplify(Ci * C - sp.eye(3)) == sp.zeros(3),
    }

    if u_A is not None and u_M is not None:
        uA = sp.Matrix(u_A)
        uM = sp.Matrix(u_M)
        checks["direction_round_trip"] = sp.simplify(corr.map_direction_M_to_A(corr.map_direction_A_to_M(uA)) - uA) == sp.zeros(3, 1)
        checks["direction_mapping"] = sp.simplify(corr.map_direction_A_to_M(uA) - uM) == sp.zeros(3, 1)

    if p_A is not None and p_M is not None:
        pA = sp.Matrix(p_A)
        pM = sp.Matrix(p_M)
        checks["plane_round_trip"] = sp.simplify(corr.map_plane_M_to_A(corr.map_plane_A_to_M(pA)) - pA) == sp.zeros(3, 1)
        checks["plane_mapping"] = sp.simplify(corr.map_plane_A_to_M(pA) - pM) == sp.zeros(3, 1)

    if u_A is not None and p_A is not None:
        uA = sp.Matrix(u_A)
        pA = sp.Matrix(p_A)
        checks["incidence_before"] = sp.simplify((pA.T * uA)[0]) == 0
        uM = corr.map_direction_A_to_M(uA)
        pM = corr.map_plane_A_to_M(pA)
        checks["incidence_after"] = sp.simplify((pM.T * uM)[0]) == 0

    return checks
