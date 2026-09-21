from __future__ import annotations

"""Exact higher-order CT operators and metric-native axial weak twins.

This module is an additive validation/analysis layer.  It does not alter the
verified Type-I/Type-II CT equations in :mod:`twinning_ct`, nor the published
reticular weak-twin equations in :mod:`weak_twins`.

Scientific separation
---------------------
For one parent symmetry element ``g_A`` and package correspondence

    u_M = C_{M<-A} u_A,

the product intercorrespondence is

    C_int = C_{M<-A} g_A C_{A<-M}.

The exact route is decided *before* any weak-plane search:

* mirror (order 2, det=-1, tr=+1)     -> classical Type-I route;
* proper twofold (order 2, det=+1)    -> classical Type-II route;
* proper 3/4/6-fold with unique +1
  axis, in a double-coset containing no
  classical representative             -> axial weak-twin route;
* higher-order improper operations      -> reported, never silently converted
  to an axial weak rotation.

The double-coset guard is essential: a higher-order representative may occur
inside an operator class that already contains an exact mirror/twofold route.
Such a class is classical, not a weak-twin class.

The numerical geometry is metric-native.  Exact finite-group/order/index
questions are kept in SymPy arithmetic; floating arithmetic is used only for
positive-definite metric geometry.
"""

from dataclasses import dataclass
from math import gcd
from typing import Iterable, Literal, Sequence

import numpy as np
import sympy as sp
from sympy.matrices.normalforms import smith_normal_form
from sympy.polys.domains import ZZ

from .correspondence import Correspondence
from .group_theory import GroupoidResult, correspondence_groupoid
from .symmetry import matrix_key
from .weak_twins import (
    AxialWeakTwinResult,
    BravaisNodeBasis,
    enumerate_ct_constrained_weak_planes,
    primitive_integer_direction,
)

WeakRoute = Literal[
    "identity",
    "type_I_reflection",
    "type_II_twofold",
    "inversion",
    "axial_weak_rotation",
    "improper_higher_order",
    "unsupported_finite_order",
]

OperatorRoute = Literal[
    "identity",
    "classical_exact",
    "axial_weak",
    "unsupported",
]


def _relative_residual(lhs: np.ndarray, rhs: np.ndarray) -> float:
    a = np.asarray(lhs, dtype=float)
    b = np.asarray(rhs, dtype=float)
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0)
    return float(np.linalg.norm(a - b) / scale)


def _validated_metric(metric: np.ndarray, *, name: str) -> np.ndarray:
    M = np.asarray(metric, dtype=float)
    if M.shape != (3, 3) or not np.all(np.isfinite(M)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    S = 0.5 * (M + M.T)
    if _relative_residual(M, S) > 1.0e-12:
        raise ValueError(f"{name} must be symmetric")
    eig = np.linalg.eigvalsh(S)
    if float(np.min(eig)) <= 0.0:
        raise ValueError(f"{name} must be positive definite; eigenvalues={eig}")
    return S


def _metric_basis(metric: np.ndarray) -> np.ndarray:
    """Return B with B.T @ B == M."""

    return np.linalg.cholesky(_validated_metric(metric, name="metric")).T


def _rational_matrix(
    value: object,
    *,
    name: str = "matrix",
    tolerance: float = 1.0e-12,
    max_denominator: int = 4096,
) -> sp.Matrix:
    """Safely recover an exact rational 3x3 matrix.

    Exact SymPy rationals/integers are preserved.  A floating entry is accepted
    only when a modest-denominator rational reproduces it within ``tolerance``.
    No ``nsimplify`` guess is made silently.
    """

    source = sp.Matrix(value)
    if source.shape != (3, 3):
        raise ValueError(f"{name} must be 3x3; got {source.shape}")

    exact: list[sp.Rational] = []
    for entry in source:
        if entry.is_Rational:
            exact.append(sp.Rational(entry))
            continue
        numeric = float(entry)
        if not np.isfinite(numeric):
            raise ValueError(f"{name} contains non-finite entries")
        candidate = sp.Rational(str(numeric)).limit_denominator(max_denominator)
        if abs(float(candidate) - numeric) > tolerance:
            raise ValueError(
                f"{name} entry {numeric:.16g} is not safely recoverable as "
                f"a rational with denominator <= {max_denominator}"
            )
        exact.append(candidate)
    return sp.Matrix(3, 3, exact)


def _projective_integer_direction(vector: object) -> tuple[int, int, int]:
    """Canonical unoriented integer representative of a rational axis."""

    values = list(primitive_integer_direction(sp.Matrix(vector)))
    first = next((item for item in values if item != 0), None)
    if first is None:
        raise ValueError("axis must be nonzero")
    if first < 0:
        values = [-item for item in values]
    return tuple(values)  # type: ignore[return-value]


def _projective_integer_plane(vector: object) -> tuple[int, int, int]:
    values = [sp.Rational(item) for item in sp.Matrix(vector)]
    if len(values) != 3 or all(item == 0 for item in values):
        raise ValueError("plane must be a nonzero 3-vector")
    denominator = 1
    for item in values:
        denominator = int(sp.ilcm(denominator, int(item.q)))
    integers = [int(item * denominator) for item in values]
    divisor = 0
    for item in integers:
        divisor = gcd(divisor, abs(item))
    integers = [item // divisor for item in integers]
    first = next(item for item in integers if item != 0)
    if first < 0:
        integers = [-item for item in integers]
    return tuple(integers)  # type: ignore[return-value]


def _exact_zero(expression: sp.Expr) -> bool:
    """Robust exact-zero predicate for rational and algebraic/trig entries."""

    simplified = sp.simplify(expression)
    if simplified == 0:
        return True
    # Explicit trig reduction is needed for exact non-crystallographic audit
    # inputs such as cos(2*pi/5).  This is never a floating comparison.
    reduced = sp.simplify(sp.trigsimp(sp.expand_trig(simplified)))
    return reduced == 0


def _exact_identity(matrix: sp.Matrix) -> bool:
    value = sp.Matrix(matrix)
    if value.shape != (3, 3):
        return False
    return all(
        _exact_zero(value[i, j] - (1 if i == j else 0))
        for i in range(3)
        for j in range(3)
    )


def exact_matrix_order(matrix: sp.Matrix, *, maximum_order: int = 24) -> int:
    """Return the exact finite order of a 3x3 matrix.

    No eigenvalue rounding is used.  Rational crystallographic matrices take
    the fast SymPy path; algebraic/trigonometric exact matrices receive an
    elementwise exact reduction before being rejected.
    """

    if maximum_order < 1:
        raise ValueError("maximum_order must be >= 1")
    G = sp.Matrix(matrix)
    if G.shape != (3, 3) or _exact_zero(sp.simplify(G.det())):
        raise ValueError("matrix must be an invertible exact 3x3 matrix")

    power = sp.eye(3)
    for order in range(1, maximum_order + 1):
        power = sp.simplify(power * G)
        if _exact_identity(power):
            return order
    raise ValueError(
        f"matrix order exceeds {maximum_order} or the matrix is not finite-order"
    )


def _metric_preservation_residual(
    operation: sp.Matrix,
    metric: np.ndarray,
) -> float:
    M = _validated_metric(metric, name="parent metric")
    G = np.asarray(operation, dtype=float)
    return _relative_residual(G.T @ M @ G, M)


@dataclass(frozen=True)
class SymmetryElementAudit:
    order: int
    determinant_exact: str
    trace_exact: str
    route: WeakRoute
    axis_parent: tuple[int, int, int] | None
    metric_preservation_residual: float

    @property
    def weak_eligible(self) -> bool:
        return self.route == "axial_weak_rotation"

    @property
    def classical_exact(self) -> bool:
        return self.route in {"type_I_reflection", "type_II_twofold"}


def audit_parent_symmetry_element(
    operation: sp.Matrix,
    metric_parent: np.ndarray,
    *,
    tolerance: float = 1.0e-9,
) -> SymmetryElementAudit:
    """Classify one exact crystallographic parent operation."""

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    G = sp.Matrix(operation)
    if G.shape != (3, 3):
        raise ValueError("parent symmetry operation must be 3x3")

    residual = _metric_preservation_residual(G, metric_parent)
    if residual > tolerance:
        raise ValueError(
            "parent operation does not preserve the supplied parent metric: "
            f"residual={residual:.3e}"
        )

    order = exact_matrix_order(G)
    determinant = sp.simplify(G.det())
    trace = sp.simplify(sp.trace(G))
    axis: tuple[int, int, int] | None = None

    if order == 1:
        route: WeakRoute = "identity"
    elif order == 2 and determinant == -1 and trace == 1:
        route = "type_I_reflection"
    elif order == 2 and determinant == 1 and trace == -1:
        route = "type_II_twofold"
    elif order == 2 and G == -sp.eye(3):
        route = "inversion"
    elif determinant == 1 and order in {3, 4, 6}:
        nullspace = (G - sp.eye(3)).nullspace()
        if len(nullspace) != 1:
            raise ValueError(
                "proper higher-order crystallographic rotation must have one "
                f"unique +1 axis; nullity={len(nullspace)}"
            )
        axis = _projective_integer_direction(nullspace[0])
        route = "axial_weak_rotation"
    elif determinant == -1 and order > 2:
        route = "improper_higher_order"
    else:
        route = "unsupported_finite_order"

    return SymmetryElementAudit(
        order=order,
        determinant_exact=str(determinant),
        trace_exact=str(trace),
        route=route,
        axis_parent=axis,
        metric_preservation_residual=residual,
    )


@dataclass(frozen=True)
class OperatorRouteAudit:
    operator_index: int
    size: int
    contains_identity: bool
    route: OperatorRoute
    element_audits: tuple[SymmetryElementAudit, ...]
    classical_element_indices: tuple[int, ...]
    weak_element_indices: tuple[int, ...]

    @property
    def is_polar_weak_operator(self) -> bool:
        return self.route == "axial_weak"


def audit_operator_route(
    operator_index: int,
    operator: Sequence[sp.Matrix],
    metric_parent: np.ndarray,
) -> OperatorRouteAudit:
    """Audit one CT double-coset before selecting any weak-twin route."""

    audits = tuple(
        audit_parent_symmetry_element(item, metric_parent) for item in operator
    )
    classical = tuple(
        i for i, item in enumerate(audits) if item.classical_exact
    )
    weak = tuple(i for i, item in enumerate(audits) if item.weak_eligible)
    contains_identity = any(item.route == "identity" for item in audits)

    if contains_identity:
        route: OperatorRoute = "identity"
    elif classical:
        # Critical rule: a higher-order representative inside a class that also
        # contains an exact mirror/twofold does not turn the class into a weak
        # twin operator.
        route = "classical_exact"
    elif weak:
        route = "axial_weak"
    else:
        route = "unsupported"

    return OperatorRouteAudit(
        operator_index=int(operator_index),
        size=len(operator),
        contains_identity=contains_identity,
        route=route,
        element_audits=audits,
        classical_element_indices=classical,
        weak_element_indices=weak,
    )


@dataclass(frozen=True)
class SmithIndexAudit:
    denominator: int
    smith_invariants: tuple[int, int, int]
    index: int


def _smith_domain_index(C: sp.Matrix) -> SmithIndexAudit:
    denominator = 1
    for value in C:
        denominator = int(
            sp.ilcm(denominator, int(sp.Rational(value).q))
        )
    numerator = (denominator * C).applyfunc(int)
    smith = smith_normal_form(numerator, domain=ZZ)
    invariants = tuple(abs(int(smith[i, i])) for i in range(3))

    index = 1
    for invariant in invariants:
        index *= denominator // gcd(denominator, invariant)

    return SmithIndexAudit(
        denominator=int(denominator),
        smith_invariants=invariants,  # type: ignore[arg-type]
        index=int(index),
    )


@dataclass(frozen=True)
class GeneralizedTwinLatticeAudit:
    determinant_exact: str
    equal_volume: bool
    domain: SmithIndexAudit
    codomain: SmithIndexAudit
    q_g: int | None
    note: str


def generalized_twin_lattice_audit(
    correspondence: object,
) -> GeneralizedTwinLatticeAudit:
    """Audit the exact common-sublattice indices of a rational map.

    For an equal-volume same-lattice twin, both sides must give the same index
    and that common value is Cayron's generalized twin index ``q_g``.

    For an arbitrary rational non-equal-volume map the two indices remain
    useful mathematical invariants, but the function intentionally returns
    ``q_g=None`` instead of mislabelling one side as a twin index.
    """

    C = _rational_matrix(correspondence, name="correspondence")
    determinant = sp.simplify(C.det())
    if determinant == 0:
        raise ValueError("correspondence must be invertible")

    domain = _smith_domain_index(C)
    codomain = _smith_domain_index(sp.simplify(C.inv()))
    equal_volume = sp.Abs(determinant) == 1

    q_g: int | None = None
    if equal_volume:
        if domain.index != codomain.index:
            raise AssertionError(
                "equal-volume rational correspondence has inconsistent "
                "domain/codomain common-sublattice indices"
            )
        q_g = domain.index

    return GeneralizedTwinLatticeAudit(
        determinant_exact=str(determinant),
        equal_volume=bool(equal_volume),
        domain=domain,
        codomain=codomain,
        q_g=q_g,
        note=(
            "q_g is reported only for an equal-volume rational same-lattice "
            "correspondence. Non-equal-volume maps retain separate domain and "
            "codomain common-sublattice indices."
        ),
    )


@dataclass(frozen=True)
class CorrespondenceGeometry:
    determinant: float
    singular_values: tuple[float, float, float]
    generalized_strain_squared: float
    generalized_strain: float | None
    metric_trace_residual: float
    equal_volume_residual: float


def generalized_correspondence_geometry(
    metric: np.ndarray,
    correspondence: object,
) -> CorrespondenceGeometry:
    """Metric-native geometry of a same-phase rational correspondence."""

    M = _validated_metric(metric, name="metric")
    C_exact = _rational_matrix(correspondence, name="correspondence")
    C = np.asarray(C_exact, dtype=float)
    B = _metric_basis(M)
    physical = B @ C @ np.linalg.inv(B)
    singular = np.linalg.svd(physical, compute_uv=False)
    singular = np.sort(np.asarray(singular, dtype=float))[::-1]

    squared = float(np.sum(physical * physical) - 3.0)
    scale = max(float(np.sum(physical * physical)), 3.0, 1.0)
    roundoff = 512.0 * np.finfo(float).eps * scale
    strain = None if squared < -roundoff else float(np.sqrt(max(0.0, squared)))

    metric_squared = float(
        np.trace(M @ C @ np.linalg.inv(M) @ C.T) - 3.0
    )
    metric_residual = abs(metric_squared - squared) / max(
        abs(metric_squared), abs(squared), 1.0
    )

    determinant = float(np.linalg.det(physical))
    return CorrespondenceGeometry(
        determinant=determinant,
        singular_values=tuple(float(item) for item in singular),
        generalized_strain_squared=squared,
        generalized_strain=strain,
        metric_trace_residual=float(metric_residual),
        equal_volume_residual=abs(abs(determinant) - 1.0),
    )


@dataclass(frozen=True)
class DistortionGeometry:
    determinant: float
    generalized_shear: float
    displacement_gradient_singular_values: tuple[float, float, float]
    numerical_rank: int
    best_rank_one_relative_residual: float
    simple_shear_compatible: bool


def generalized_distortion_geometry(
    metric: np.ndarray,
    distortion: np.ndarray,
    *,
    rank_tolerance: float = 1.0e-10,
) -> DistortionGeometry:
    """Audit how far a weak distortion is from a conventional rank-one shear."""

    if rank_tolerance <= 0.0:
        raise ValueError("rank_tolerance must be positive")
    M = _validated_metric(metric, name="metric")
    F = np.asarray(distortion, dtype=float)
    if F.shape != (3, 3) or not np.all(np.isfinite(F)):
        raise ValueError("distortion must be a finite 3x3 matrix")

    B = _metric_basis(M)
    F_hat = B @ F @ np.linalg.inv(B)
    H = F_hat - np.eye(3)
    singular = np.linalg.svd(H, compute_uv=False)
    singular = np.sort(np.asarray(singular, dtype=float))[::-1]
    norm = float(np.linalg.norm(singular))
    threshold = rank_tolerance * max(float(singular[0]), 1.0)
    numerical_rank = int(np.sum(singular > threshold))
    tail = float(np.linalg.norm(singular[1:]))
    rank_one_residual = 0.0 if norm <= 1.0e-15 else tail / norm
    determinant = float(np.linalg.det(F_hat))

    return DistortionGeometry(
        determinant=determinant,
        generalized_shear=float(norm),
        displacement_gradient_singular_values=tuple(
            float(item) for item in singular
        ),
        numerical_rank=numerical_rank,
        best_rank_one_relative_residual=rank_one_residual,
        simple_shear_compatible=bool(
            rank_one_residual <= 1.0e-8
            and abs(determinant - 1.0) <= 1.0e-8
        ),
    )


@dataclass(frozen=True)
class WeakPlaneGeometry:
    intraplanar_singular_values: tuple[float, float]
    intraplanar_distortion: float
    maximum_principal_intraplanar_strain: float
    intraplanar_area_ratio: float
    incidence_residual: float
    reticular_isometry_residual: float
    reticular_determinant: float
    proper_reticular_angle_deg: float | None
    distortion: DistortionGeometry


def _stable_rotation_angle_deg(Q: np.ndarray) -> float:
    skew = 0.5 * np.array(
        [
            Q[2, 1] - Q[1, 2],
            Q[0, 2] - Q[2, 0],
            Q[1, 0] - Q[0, 1],
        ],
        dtype=float,
    )
    sine = float(np.linalg.norm(skew))
    cosine = float(np.clip((np.trace(Q) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arctan2(sine, cosine)))


def weak_plane_geometry(
    metric: np.ndarray,
    weak_twin: AxialWeakTwinResult,
) -> WeakPlaneGeometry:
    """Metric-native intrinsic distortion of the selected weak-plane branch.

    A physical orthonormal basis of the source plane is built from the invariant
    axis and the transverse in-plane direction.  The two singular values of
    ``F`` restricted to that plane quantify *intrinsic* plane distortion.  This
    is basis-invariant and does not depend on Miller-index scaling.
    """

    M = _validated_metric(metric, name="metric")
    B = _metric_basis(M)
    B_inv = np.linalg.inv(B)

    u = np.asarray(weak_twin.axis_conventional, dtype=float)
    p = np.asarray(weak_twin.plane1_conventional, dtype=float)
    axis = B @ u
    axis /= float(np.linalg.norm(axis))
    normal = np.linalg.solve(B.T, p)
    normal /= float(np.linalg.norm(normal))

    incidence = abs(float(axis @ normal))
    if incidence > 2.0e-9:
        raise ValueError(
            "weak-twin invariant axis is not incident in its source plane: "
            f"residual={incidence:.3e}"
        )
    transverse = np.cross(normal, axis)
    transverse /= float(np.linalg.norm(transverse))
    E = np.column_stack((axis, transverse))

    F = np.asarray(weak_twin.selected.distortion_F1, dtype=float)
    F_hat = B @ F @ B_inv
    mapped = F_hat @ E
    sigma = np.linalg.svd(mapped, compute_uv=False)
    sigma = np.sort(np.asarray(sigma, dtype=float))[::-1]
    intrinsic = float(np.linalg.norm(sigma - 1.0))

    T = np.asarray(weak_twin.selected.T_2_from_1, dtype=float)
    T_hat = B @ T @ B_inv
    isometry = float(np.linalg.norm(T_hat.T @ T_hat - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(T_hat))
    angle = (
        _stable_rotation_angle_deg(T_hat)
        if determinant > 0.0 and isometry <= 2.0e-8
        else None
    )

    return WeakPlaneGeometry(
        intraplanar_singular_values=tuple(float(item) for item in sigma),
        intraplanar_distortion=intrinsic,
        maximum_principal_intraplanar_strain=float(
            np.max(np.abs(sigma - 1.0))
        ),
        intraplanar_area_ratio=float(np.prod(sigma)),
        incidence_residual=incidence,
        reticular_isometry_residual=isometry,
        reticular_determinant=determinant,
        proper_reticular_angle_deg=angle,
        distortion=generalized_distortion_geometry(M, F),
    )


@dataclass(frozen=True)
class RankedWeakPlane:
    result: AxialWeakTwinResult
    geometry: WeakPlaneGeometry
    plane_complexity: int

    @property
    def plane_pair(
        self,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return (self.result.plane1_primitive, self.result.plane2_primitive)


def enumerate_ranked_weak_planes(
    metric_conventional: np.ndarray,
    correspondence_conventional: object,
    axis_conventional: Iterable[float | int],
    *,
    node_basis: BravaisNodeBasis,
    max_plane_index: int = 6,
    maximum_generalized_shear: float | None = None,
    maximum_intraplanar_distortion: float | None = None,
) -> tuple[RankedWeakPlane, ...]:
    """Enumerate CT-constrained weak planes and rank by intrinsic plane strain."""

    if maximum_intraplanar_distortion is not None:
        if maximum_intraplanar_distortion <= 0.0:
            raise ValueError("maximum_intraplanar_distortion must be positive")

    raw = enumerate_ct_constrained_weak_planes(
        metric_conventional,
        correspondence_conventional,
        axis_conventional,
        node_basis=node_basis,
        max_plane_index=max_plane_index,
        maximum_generalized_shear=maximum_generalized_shear,
    )
    ranked: list[RankedWeakPlane] = []
    for item in raw:
        geometry = weak_plane_geometry(metric_conventional, item)
        if (
            maximum_intraplanar_distortion is not None
            and geometry.intraplanar_distortion
            > maximum_intraplanar_distortion
        ):
            continue
        complexity = (
            sum(abs(value) for value in item.plane1_primitive)
            + sum(abs(value) for value in item.plane2_primitive)
        )
        ranked.append(
            RankedWeakPlane(
                result=item,
                geometry=geometry,
                plane_complexity=int(complexity),
            )
        )

    ranked.sort(
        key=lambda item: (
            item.geometry.intraplanar_distortion,
            item.plane_complexity,
            item.result.selected.generalized_shear,
            item.result.plane1_primitive,
            item.result.plane2_primitive,
        )
    )
    return tuple(ranked)


@dataclass(frozen=True)
class WeakElementAnalysis:
    parent_operation: sp.Matrix
    element_audit: SymmetryElementAudit
    parent_axis: tuple[int, int, int]
    product_axis: tuple[int, int, int]
    intercorrespondence_exact: sp.Matrix
    intercorrespondence_order: int
    lattice_audit: GeneralizedTwinLatticeAudit
    correspondence_geometry: CorrespondenceGeometry
    ranked_weak_planes: tuple[RankedWeakPlane, ...]


def analyze_higher_order_element(
    parent_operation: sp.Matrix,
    metric_parent: np.ndarray,
    metric_product: np.ndarray,
    correspondence: Correspondence,
    *,
    product_node_basis: BravaisNodeBasis,
    max_plane_index: int = 6,
    maximum_generalized_shear: float | None = None,
    maximum_intraplanar_distortion: float | None = None,
) -> WeakElementAnalysis:
    """Derive a higher-order CT weak route from first principles."""

    element = audit_parent_symmetry_element(parent_operation, metric_parent)
    if not element.weak_eligible or element.axis_parent is None:
        raise ValueError(
            "axial weak route requires a proper crystallographic 3/4/6-fold "
            f"parent rotation; classified as {element.route!r}"
        )

    C = correspondence.C_M_from_A
    G = sp.Matrix(parent_operation)
    axis_parent_exact = sp.Matrix(element.axis_parent)
    axis_product_exact = sp.simplify(C * axis_parent_exact)
    axis_product = _projective_integer_direction(axis_product_exact)

    C_int = sp.simplify(C * G * C.inv())
    inter_order = exact_matrix_order(C_int)
    if inter_order != element.order:
        raise AssertionError(
            "intercorrespondence order is not preserved by conjugation: "
            f"parent={element.order}, product={inter_order}"
        )
    if sp.simplify(C_int.det()) != sp.simplify(G.det()):
        raise AssertionError("intercorrespondence determinant changed under conjugation")
    if sp.simplify(C_int * axis_product_exact - axis_product_exact) != sp.zeros(3, 1):
        raise AssertionError("derived product axis is not invariant under C_int")

    lattice_audit = generalized_twin_lattice_audit(C_int)
    if not lattice_audit.equal_volume or lattice_audit.q_g is None:
        raise ValueError(
            "same-phase axial weak twin requires an equal-volume "
            "intercorrespondence"
        )

    geometry = generalized_correspondence_geometry(metric_product, C_int)
    if geometry.generalized_strain is None:
        raise ValueError(
            "Cayron generalized strain is not real for this intercorrespondence"
        )

    weak_planes = enumerate_ranked_weak_planes(
        metric_product,
        C_int,
        axis_product_exact,
        node_basis=product_node_basis,
        max_plane_index=max_plane_index,
        maximum_generalized_shear=maximum_generalized_shear,
        maximum_intraplanar_distortion=maximum_intraplanar_distortion,
    )

    return WeakElementAnalysis(
        parent_operation=G,
        element_audit=element,
        parent_axis=_projective_integer_direction(axis_parent_exact),
        product_axis=axis_product,
        intercorrespondence_exact=C_int,
        intercorrespondence_order=inter_order,
        lattice_audit=lattice_audit,
        correspondence_geometry=geometry,
        ranked_weak_planes=weak_planes,
    )


@dataclass(frozen=True)
class WeakOperatorAnalysis:
    operator_audit: OperatorRouteAudit
    weak_elements: tuple[WeakElementAnalysis, ...]


@dataclass(frozen=True)
class HigherOrderGroupoidReport:
    groupoid: GroupoidResult
    operators: tuple[WeakOperatorAnalysis, ...]

    @property
    def weak_operators(self) -> tuple[WeakOperatorAnalysis, ...]:
        return tuple(
            item for item in self.operators
            if item.operator_audit.is_polar_weak_operator
        )


def analyze_non_twofold_groupoid(
    parent_group: Sequence[sp.Matrix],
    product_group: Sequence[sp.Matrix],
    metric_parent: np.ndarray,
    metric_product: np.ndarray,
    correspondence: Correspondence,
    *,
    product_node_basis: BravaisNodeBasis,
    max_plane_index: int = 6,
    maximum_generalized_shear: float | None = None,
    maximum_intraplanar_distortion: float | None = None,
    eligible_orders: Iterable[int] = (3, 4, 6),
    max_weak_planes_per_element: int | None = None,
) -> HigherOrderGroupoidReport:
    """Analyze every CT operator, but run weak search only on genuinely polar ones."""

    allowed_orders = frozenset(int(item) for item in eligible_orders)
    if not allowed_orders or not allowed_orders.issubset({3, 4, 6}):
        raise ValueError("eligible_orders must be a nonempty subset of {3,4,6}")
    if max_weak_planes_per_element is not None and max_weak_planes_per_element < 1:
        raise ValueError("max_weak_planes_per_element must be >= 1 or None")

    groupoid = correspondence_groupoid(
        list(parent_group), list(product_group), correspondence
    )
    output: list[WeakOperatorAnalysis] = []

    for operator_index, operator in enumerate(groupoid.operators):
        audit = audit_operator_route(operator_index, operator, metric_parent)
        weak_elements: list[WeakElementAnalysis] = []

        if audit.is_polar_weak_operator:
            for element_index in audit.weak_element_indices:
                element_audit = audit.element_audits[element_index]
                if element_audit.order not in allowed_orders:
                    continue
                analysis = analyze_higher_order_element(
                    operator[element_index],
                    metric_parent,
                    metric_product,
                    correspondence,
                    product_node_basis=product_node_basis,
                    max_plane_index=max_plane_index,
                    maximum_generalized_shear=maximum_generalized_shear,
                    maximum_intraplanar_distortion=maximum_intraplanar_distortion,
                )
                if max_weak_planes_per_element is not None:
                    analysis = WeakElementAnalysis(
                        parent_operation=analysis.parent_operation,
                        element_audit=analysis.element_audit,
                        parent_axis=analysis.parent_axis,
                        product_axis=analysis.product_axis,
                        intercorrespondence_exact=analysis.intercorrespondence_exact,
                        intercorrespondence_order=analysis.intercorrespondence_order,
                        lattice_audit=analysis.lattice_audit,
                        correspondence_geometry=analysis.correspondence_geometry,
                        ranked_weak_planes=analysis.ranked_weak_planes[
                            :max_weak_planes_per_element
                        ],
                    )
                weak_elements.append(analysis)

        output.append(
            WeakOperatorAnalysis(
                operator_audit=audit,
                weak_elements=tuple(weak_elements),
            )
        )

    return HigherOrderGroupoidReport(
        groupoid=groupoid,
        operators=tuple(output),
    )
