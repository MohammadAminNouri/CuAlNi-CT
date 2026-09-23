from __future__ import annotations

"""Metric <-> Cartesian representation bridge.

The crystallographic metric remains the source of truth. Cartesian frames are
explicit representations of the same lattice, never a replacement for it.

Three conventions are supported:

- LEGACY_A_X_B_XY:
  the package's historical structure matrix (a || x, b in xy).
- PTCLAB_A_X_C_XZ:
  PTCLab section 2.2 convention (a || x, c in xz).
- SYMMETRIC_METRIC:
  the unique symmetric positive square root M^(1/2).

For every convention B satisfies B.T @ B = M. Therefore direct vectors,
reciprocal plane covectors, lengths, angles, incidence, deformation gradients,
and principal stretches can be cross-checked representation-independently.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .correspondence import Correspondence
from .lattice import Lattice, metric_sqrt
from .stretch import positive_definite_sqrt


class CartesianConvention(str, Enum):
    """Named orthonormal embeddings of one crystallographic metric."""

    LEGACY_A_X_B_XY = "legacy_a_x_b_xy"
    PTCLAB_A_X_C_XZ = "ptclab_a_x_c_xz"
    SYMMETRIC_METRIC = "symmetric_metric"


def _cosd(x: float) -> float:
    return float(np.cos(np.deg2rad(x)))


def _sind(x: float) -> float:
    return float(np.sin(np.deg2rad(x)))


def _relative_residual(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    scale = max(float(np.linalg.norm(lhs)), float(np.linalg.norm(rhs)), 1.0)
    return float(np.linalg.norm(lhs - rhs) / scale)


DEFAULT_AUDIT_U_PARENT = np.array([1.0, 2.0, -1.0], dtype=float)
DEFAULT_AUDIT_P_PARENT = np.array([2.0, -1.0, 1.0], dtype=float)


def _ptclab_basis(lattice: Lattice) -> np.ndarray:
    """Return PTCLab's x || a, c in xz Cartesian convention.

    PTCLab User Manual section 2.2 explicitly states that it adopts the
    convention x || e1 and e3 in the xz plane.

    Columns are the physical Cartesian basis vectors a, b, c.
    """

    a, b, c = lattice.a, lattice.b, lattice.c
    ca = _cosd(lattice.alpha_deg)
    cb = _cosd(lattice.beta_deg)
    cg = _cosd(lattice.gamma_deg)
    sb = _sind(lattice.beta_deg)

    if abs(sb) <= 1e-14:
        raise ValueError(
            "PTCLab x||a, c-in-xz convention is singular when beta is 0 or 180 degrees"
        )

    a_vec = np.array([a, 0.0, 0.0], dtype=float)
    c_vec = np.array([c * cb, 0.0, c * sb], dtype=float)

    b_x = b * cg
    b_z = b * (ca - cb * cg) / sb
    b_y_sq = b * b - b_x * b_x - b_z * b_z
    if b_y_sq < -1e-10 * max(b * b, 1.0):
        raise ValueError(
            "Invalid lattice geometry in PTCLab Cartesian construction: "
            f"computed b_y^2={b_y_sq}"
        )
    b_vec = np.array([b_x, np.sqrt(max(0.0, b_y_sq)), b_z], dtype=float)

    return np.column_stack((a_vec, b_vec, c_vec))


def basis_matrix(
    lattice: Lattice,
    convention: CartesianConvention,
) -> np.ndarray:
    """Construct B with x_cart = B x_crystal and B.T @ B = M."""

    convention = CartesianConvention(convention)
    if convention is CartesianConvention.LEGACY_A_X_B_XY:
        B = lattice.structure_matrix()
    elif convention is CartesianConvention.PTCLAB_A_X_C_XZ:
        B = _ptclab_basis(lattice)
    elif convention is CartesianConvention.SYMMETRIC_METRIC:
        B = metric_sqrt(lattice.metric())
    else:  # pragma: no cover - Enum makes this defensive only.
        raise ValueError(f"Unsupported Cartesian convention: {convention}")

    M = lattice.metric()
    residual = _relative_residual(B.T @ B, M)
    if residual > 1e-10:
        raise AssertionError(
            f"Cartesian basis does not reproduce the metric; residual={residual:.3e}"
        )
    if float(np.linalg.det(B)) <= 0.0:
        raise ValueError("Cartesian basis must be right-handed")
    return np.asarray(B, dtype=float)


@dataclass(frozen=True)
class CartesianFrame:
    """One explicit Cartesian realization of a crystallographic lattice."""

    lattice: Lattice
    convention: CartesianConvention

    @property
    def B(self) -> np.ndarray:
        return basis_matrix(self.lattice, self.convention)

    @property
    def B_inv(self) -> np.ndarray:
        return np.linalg.inv(self.B)

    @property
    def B_inv_T(self) -> np.ndarray:
        return self.B_inv.T

    def direct_to_cartesian(
        self,
        u_crystal: np.ndarray,
        *,
        normalize: bool = False,
    ) -> np.ndarray:
        """Map direct coordinates [uvw] to physical Cartesian coordinates."""

        u = np.asarray(u_crystal, dtype=float).reshape(3)
        v = self.B @ u
        if normalize:
            n = float(np.linalg.norm(v))
            if n <= 1e-15:
                raise ValueError("Cannot normalize a zero direct vector")
            v = v / n
        return v

    def direct_from_cartesian(self, v_cart: np.ndarray) -> np.ndarray:
        """Recover direct crystallographic coordinates from Cartesian coordinates."""

        return self.B_inv @ np.asarray(v_cart, dtype=float).reshape(3)

    def plane_to_cartesian(
        self,
        p_crystal: np.ndarray,
        *,
        normalize: bool = False,
    ) -> np.ndarray:
        """Map reciprocal plane covector (hkl) to its Cartesian normal vector."""

        p = np.asarray(p_crystal, dtype=float).reshape(3)
        g = self.B_inv_T @ p
        if normalize:
            n = float(np.linalg.norm(g))
            if n <= 1e-15:
                raise ValueError("Cannot normalize a zero plane covector")
            g = g / n
        return g

    def plane_from_cartesian(self, g_cart: np.ndarray) -> np.ndarray:
        """Recover reciprocal plane coordinates from a Cartesian normal vector."""

        return self.B.T @ np.asarray(g_cart, dtype=float).reshape(3)

    def operator_to_cartesian(self, g_crystal: np.ndarray) -> np.ndarray:
        """Represent a same-lattice direct-space operator in this Cartesian frame."""

        g = np.asarray(g_crystal, dtype=float).reshape(3, 3)
        return self.B @ g @ self.B_inv

    def direct_length(self, u_crystal: np.ndarray) -> float:
        return float(np.linalg.norm(self.direct_to_cartesian(u_crystal)))

    def reciprocal_length(self, p_crystal: np.ndarray) -> float:
        return float(np.linalg.norm(self.plane_to_cartesian(p_crystal)))

    def plane_spacing(self, p_crystal: np.ndarray) -> float:
        """Return d=1/|g| in the no-2pi reciprocal convention used by the package."""

        g_norm = self.reciprocal_length(p_crystal)
        if g_norm <= 1e-15:
            raise ValueError("Plane spacing is undefined for the zero covector")
        return 1.0 / g_norm


def frame_rotation(
    source: CartesianFrame,
    target: CartesianFrame,
    *,
    tol: float = 1e-10,
) -> np.ndarray:
    """Return proper rotation Q mapping Cartesian coordinates source -> target.

    This requires both frames to represent the *same* crystallographic metric.
    """

    if _relative_residual(source.lattice.metric(), target.lattice.metric()) > tol:
        raise ValueError("Frame rotation requires identical lattice metrics")

    Q = target.B @ source.B_inv
    ortho_res = _relative_residual(Q.T @ Q, np.eye(3))
    det_q = float(np.linalg.det(Q))
    if ortho_res > tol or abs(det_q - 1.0) > 10.0 * tol:
        raise AssertionError(
            "Two valid right-handed embeddings of one metric must differ by a "
            f"proper rotation; orthogonality={ortho_res:.3e}, det={det_q:.12g}"
        )
    return Q


@dataclass(frozen=True)
class RepresentationParity:
    """Numerical audit proving metric and Cartesian representations agree."""

    parent_metric_residual: float
    product_metric_residual: float
    direct_norm_residual: float
    reciprocal_norm_residual: float
    pairing_residual: float
    correspondence_direction_residual: float
    correspondence_plane_residual: float
    right_cauchy_green_residual: float
    principal_stretch_residual: float
    polar_orthogonality_residual: float
    polar_det_residual: float

    @property
    def maximum_residual(self) -> float:
        return max(
            self.parent_metric_residual,
            self.product_metric_residual,
            self.direct_norm_residual,
            self.reciprocal_norm_residual,
            self.pairing_residual,
            self.correspondence_direction_residual,
            self.correspondence_plane_residual,
            self.right_cauchy_green_residual,
            self.principal_stretch_residual,
            self.polar_orthogonality_residual,
            self.polar_det_residual,
        )

    def assert_within(self, tol: float = 1e-10) -> None:
        if self.maximum_residual > tol:
            raise AssertionError(
                "Metric/Cartesian parity failed: "
                f"maximum residual={self.maximum_residual:.3e} > {tol:.3e}"
            )


@dataclass(frozen=True)
class RepresentationBridge:
    """One physical transformation exposed in metric and Cartesian form.

    ``correspondence`` follows the package convention u_M = C_M_from_A u_A.
    ``F_cart`` is the physical deformation gradient between the selected
    Cartesian frames:

        F_cart = B_M C_M_from_A B_A^{-1}

    The frame choice can change matrix entries, Euler angles, and plotted
    coordinates, but not physical scalar invariants such as principal stretches.
    """

    parent: Lattice
    product: Lattice
    correspondence: Correspondence
    parent_convention: CartesianConvention = CartesianConvention.SYMMETRIC_METRIC
    product_convention: CartesianConvention = CartesianConvention.SYMMETRIC_METRIC

    @property
    def parent_frame(self) -> CartesianFrame:
        return CartesianFrame(self.parent, self.parent_convention)

    @property
    def product_frame(self) -> CartesianFrame:
        return CartesianFrame(self.product, self.product_convention)

    @property
    def C(self) -> np.ndarray:
        return np.asarray(self.correspondence.C_M_from_A, dtype=float)

    def pulled_back_product_metric(self) -> np.ndarray:
        """Return G_C = C.T M_M C in parent crystallographic coordinates."""

        return self.C.T @ self.product.metric() @ self.C

    def deformation_gradient_cartesian(self) -> np.ndarray:
        """Return the physical F = B_M C B_A^-1.

        Both lattice embeddings are required to be right-handed. Therefore
        sign(det(F)) == sign(det(C)). A negative determinant would make the
        polar factor an improper orthogonal transformation (det(R)=-1), not a
        physically admissible martensitic deformation rotation. Refuse such a
        map explicitly instead of letting it surface later as a cryptic
        polar-det residual of 2.
        """

        F = self.product_frame.B @ self.C @ self.parent_frame.B_inv
        det_f = float(np.linalg.det(F))
        if not np.isfinite(det_f):
            raise ValueError(
                "Physical deformation gradient has a non-finite determinant."
            )
        if det_f <= 0.0:
            raise ValueError(
                "Physical deformation gradient must preserve handedness "
                f"(det(F) > 0); got det(F)={det_f:.12g}. "
                "Check the correspondence direction and basis handedness. "
                "The code will not silently flip an axis."
            )
        return F

    def right_cauchy_green_cartesian(self) -> np.ndarray:
        F = self.deformation_gradient_cartesian()
        return F.T @ F

    def right_cauchy_green_from_metric(self) -> np.ndarray:
        """Return the Cartesian congruence of C.T M_M C."""

        B_ai = self.parent_frame.B_inv
        return B_ai.T @ self.pulled_back_product_metric() @ B_ai

    def right_stretch_cartesian(self) -> np.ndarray:
        return positive_definite_sqrt(self.right_cauchy_green_cartesian())

    def polar_rotation_cartesian(self) -> np.ndarray:
        F = self.deformation_gradient_cartesian()
        U = self.right_stretch_cartesian()
        return F @ np.linalg.inv(U)

    def principal_stretches(self) -> np.ndarray:
        """Return sorted singular values of the physical deformation gradient."""

        return np.sort(np.linalg.svd(self.deformation_gradient_cartesian(), compute_uv=False))

    def map_parent_direction_two_ways(
        self,
        u_parent: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return product physical vector via crystallography and via F."""

        u_a = np.asarray(u_parent, dtype=float).reshape(3)
        u_m = self.C @ u_a
        via_crystal = self.product_frame.direct_to_cartesian(u_m)
        via_deformation = (
            self.deformation_gradient_cartesian()
            @ self.parent_frame.direct_to_cartesian(u_a)
        )
        return via_crystal, via_deformation

    def map_parent_plane_two_ways(
        self,
        p_parent: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return product plane normal via C^-T and via F^-T."""

        p_a = np.asarray(p_parent, dtype=float).reshape(3)
        p_m = np.linalg.inv(self.C).T @ p_a
        via_crystal = self.product_frame.plane_to_cartesian(p_m)
        via_deformation = (
            np.linalg.inv(self.deformation_gradient_cartesian()).T
            @ self.parent_frame.plane_to_cartesian(p_a)
        )
        return via_crystal, via_deformation

    def audit(
        self,
        *,
        u_parent: np.ndarray | None = None,
        p_parent: np.ndarray | None = None,
    ) -> RepresentationParity:
        """Run representation-independent identities on one transformation."""

        if u_parent is None:
            u_parent = DEFAULT_AUDIT_U_PARENT
        if p_parent is None:
            p_parent = DEFAULT_AUDIT_P_PARENT

        u = np.asarray(u_parent, dtype=float).reshape(3)
        p = np.asarray(p_parent, dtype=float).reshape(3)

        M_a = self.parent.metric()
        M_m = self.product.metric()
        B_a = self.parent_frame.B
        B_m = self.product_frame.B

        parent_metric_residual = _relative_residual(B_a.T @ B_a, M_a)
        product_metric_residual = _relative_residual(B_m.T @ B_m, M_m)

        u_cart = B_a @ u
        metric_u2 = float(u @ M_a @ u)
        cart_u2 = float(u_cart @ u_cart)
        direct_norm_residual = abs(metric_u2 - cart_u2) / max(
            abs(metric_u2), abs(cart_u2), 1.0
        )

        g_cart = np.linalg.inv(B_a).T @ p
        reciprocal_p2 = float(p @ np.linalg.inv(M_a) @ p)
        cart_g2 = float(g_cart @ g_cart)
        reciprocal_norm_residual = abs(reciprocal_p2 - cart_g2) / max(
            abs(reciprocal_p2), abs(cart_g2), 1.0
        )

        pairing_metric = float(p @ u)
        pairing_cart = float(g_cart @ u_cart)
        pairing_residual = abs(pairing_metric - pairing_cart) / max(
            abs(pairing_metric), abs(pairing_cart), 1.0
        )

        d1, d2 = self.map_parent_direction_two_ways(u)
        correspondence_direction_residual = _relative_residual(d1, d2)

        n1, n2 = self.map_parent_plane_two_ways(p)
        correspondence_plane_residual = _relative_residual(n1, n2)

        c_cart = self.right_cauchy_green_cartesian()
        c_metric = self.right_cauchy_green_from_metric()
        right_cauchy_green_residual = _relative_residual(c_cart, c_metric)

        principal_from_f = np.sort(
            np.linalg.svd(self.deformation_gradient_cartesian(), compute_uv=False)
        )

        # In the symmetric parent/product frames this is exactly the metric-whitened
        # stretch problem. Singular values are frame-invariant, so we may compare
        # against the same physical map expressed in symmetric frames.
        symmetric_bridge = RepresentationBridge(
            self.parent,
            self.product,
            self.correspondence,
            CartesianConvention.SYMMETRIC_METRIC,
            CartesianConvention.SYMMETRIC_METRIC,
        )
        principal_symmetric = symmetric_bridge.principal_stretches()
        principal_stretch_residual = _relative_residual(
            principal_from_f, principal_symmetric
        )

        R = self.polar_rotation_cartesian()
        polar_orthogonality_residual = _relative_residual(R.T @ R, np.eye(3))
        polar_det_residual = abs(float(np.linalg.det(R)) - 1.0)

        return RepresentationParity(
            parent_metric_residual=parent_metric_residual,
            product_metric_residual=product_metric_residual,
            direct_norm_residual=float(direct_norm_residual),
            reciprocal_norm_residual=float(reciprocal_norm_residual),
            pairing_residual=float(pairing_residual),
            correspondence_direction_residual=correspondence_direction_residual,
            correspondence_plane_residual=correspondence_plane_residual,
            right_cauchy_green_residual=right_cauchy_green_residual,
            principal_stretch_residual=principal_stretch_residual,
            polar_orthogonality_residual=polar_orthogonality_residual,
            polar_det_residual=float(polar_det_residual),
        )


def compare_convention_pairs(
    parent: Lattice,
    product: Lattice,
    correspondence: Correspondence,
) -> dict[str, np.ndarray]:
    """Return principal stretches for all supported frame-convention pairs.

    A future GUI can expose this directly as a scientific parity panel.
    """

    out: dict[str, np.ndarray] = {}
    for parent_convention in CartesianConvention:
        for product_convention in CartesianConvention:
            bridge = RepresentationBridge(
                parent,
                product,
                correspondence,
                parent_convention,
                product_convention,
            )
            key = f"{parent_convention.value}->{product_convention.value}"
            out[key] = bridge.principal_stretches()
    return out
