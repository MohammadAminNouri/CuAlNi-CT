from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import sympy as sp


@dataclass(frozen=True)
class Correspondence:
    """Lattice correspondence with an explicit internal convention.

    Internal convention:
        u_M = C_m_from_a @ u_A

    This matches the action described by Cayron for the matrix he denotes C^{M->A}:
    a crystallographic direction of austenite becomes a martensite direction.

    Because literature superscripts can be counter-intuitive, this class never uses a bare "C" in
    public attributes: the mapping direction is written into the name.
    """

    C_m_from_a: sp.Matrix
    label: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        C = sp.Matrix(self.C_m_from_a)
        if C.shape != (3, 3) or C.det() == 0:
            raise ValueError("Correspondence must be an invertible 3x3 matrix.")
        object.__setattr__(self, "C_m_from_a", C)

    @property
    def C_a_from_m(self) -> sp.Matrix:
        return self.C_m_from_a.inv()

    def map_direction_a_to_m(self, u_a) -> sp.Matrix:
        return self.C_m_from_a * sp.Matrix(u_a)

    def map_direction_m_to_a(self, u_m) -> sp.Matrix:
        return self.C_a_from_m * sp.Matrix(u_m)

    def map_plane_a_to_m(self, p_a) -> sp.Matrix:
        """Plane covector transform p_M = C^{-T} p_A."""
        return self.C_m_from_a.inv().T * sp.Matrix(p_a)

    def map_plane_m_to_a(self, p_m) -> sp.Matrix:
        return self.C_m_from_a.T * sp.Matrix(p_m)

    def pullback_product_metric(self, M_m: np.ndarray) -> np.ndarray:
        C = np.array(self.C_m_from_a, dtype=float)
        return C.T @ np.asarray(M_m, dtype=float) @ C

    def intercorrespondence_from_parent_symmetry(self, g_a: sp.Matrix) -> sp.Matrix:
        """C_int = C_{M<-A} g_A C_{A<-M}; acts in martensite coordinates."""
        return sp.simplify(self.C_m_from_a * sp.Matrix(g_a) * self.C_a_from_m)
