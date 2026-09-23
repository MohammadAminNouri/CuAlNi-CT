from __future__ import annotations

"""Human-facing crystallographic input facade.

This module is deliberately not another theory implementation. It translates
quantities a crystallographer normally knows -- unit-cell parameters, point
groups, basis correspondences and plane/direction parallelisms -- into the
repository's explicit scientific state.

Normal users never need to choose a metric tensor or a Cartesian embedding.
Those remain internal representations and are cross-checked automatically.

A correspondence matrix is never accepted without a declared mapping direction.
This prevents C_(M<-A) versus C_(A<-M) convention errors at the UI boundary.
"""

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import sympy as sp

from .calculation_service import (
    CalculationKind,
    CalculationService,
    TransformationBundle,
)
from .lattice import Lattice
from .orientation import OrientationService
from .point_groups import resolve_point_group
from .project_io import LoadedProject, project_from_dict
from .representation import CartesianConvention, RepresentationBridge


class MatrixMapping(str, Enum):
    PARENT_TO_PRODUCT = "parent_to_product"
    PRODUCT_TO_PARENT = "product_to_parent"


def _exact_scalar(value: object) -> sp.Rational:
    if isinstance(value, bool):
        raise TypeError("Boolean is not a crystallographic scalar.")
    if isinstance(value, (int, np.integer)):
        return sp.Rational(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Crystallographic scalar must be finite.")
        return sp.Rational(str(number))
    if isinstance(value, str):
        text = value.strip().replace("−", "-")
        if not text:
            raise ValueError("Empty crystallographic scalar.")
        try:
            fraction = Fraction(text)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(
                f"Scalar {value!r} must be an integer, decimal or exact p/q."
            ) from exc
        return sp.Rational(fraction.numerator, fraction.denominator)
    if isinstance(value, sp.Rational):
        return sp.Rational(value)
    raise TypeError(
        f"Unsupported crystallographic scalar type: {type(value).__name__}."
    )


def _tokenize_indices(text: str) -> list[str]:
    value = text.strip().replace("−", "-")
    if value and value[0] in "[(" and value[-1] in "])":
        value = value[1:-1].strip()
    value = value.replace(",", " ")
    return [part for part in value.split() if part]


def exact_vector3(value: object, *, name: str = "vector") -> sp.Matrix:
    if isinstance(value, str):
        items: Sequence[object] = _tokenize_indices(value)
    elif isinstance(value, np.ndarray):
        items = np.asarray(value, dtype=object).reshape(-1).tolist()
    elif isinstance(value, Sequence):
        items = list(value)
    else:
        raise TypeError(f"{name} must be a 3-vector or index string.")

    if len(items) != 3:
        raise ValueError(f"{name} must contain exactly three indices; got {len(items)}.")
    vector = sp.Matrix([_exact_scalar(item) for item in items])
    if vector == sp.zeros(3, 1):
        raise ValueError(f"{name} cannot be the zero vector.")
    return vector


def exact_matrix3(value: object, *, name: str = "matrix") -> sp.Matrix:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a nested 3x3 sequence.")
    rows = list(value)
    if len(rows) != 3:
        raise ValueError(f"{name} must have exactly three rows.")
    parsed: list[list[sp.Rational]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TypeError(f"{name} row {i} must be a sequence.")
        entries = list(row)
        if len(entries) != 3:
            raise ValueError(f"{name} row {i} must contain exactly three entries.")
        parsed.append([_exact_scalar(item) for item in entries])
    matrix = sp.Matrix(parsed)
    if sp.simplify(matrix.det()) == 0:
        raise ValueError(f"{name} must be invertible.")
    return matrix


def _matrix_as_json(matrix: sp.Matrix) -> list[list[object]]:
    out: list[list[object]] = []
    for i in range(3):
        row: list[object] = []
        for j in range(3):
            item = sp.Rational(sp.simplify(matrix[i, j]))
            if item.q == 1:
                row.append(int(item.p))
            else:
                row.append(f"{int(item.p)}/{int(item.q)}")
        out.append(row)
    return out


@dataclass(frozen=True)
class CrystalInput:
    """PTCLab-style phase definition using ordinary unit-cell parameters."""

    label: str
    point_group: str
    a: float
    b: float | None = None
    c: float | None = None
    alpha_deg: float = 90.0
    beta_deg: float = 90.0
    gamma_deg: float = 90.0
    length_unit: str = "angstrom"

    def __post_init__(self) -> None:
        label = self.label.strip()
        symbol = self.point_group.strip()
        if not label:
            raise ValueError("Crystal label must be non-empty.")
        if not symbol:
            raise ValueError("point_group must be supplied explicitly.")

        b = self.a if self.b is None else self.b
        c = self.a if self.c is None else self.c
        lattice = Lattice(
            a=float(self.a),
            b=float(b),
            c=float(c),
            alpha_deg=float(self.alpha_deg),
            beta_deg=float(self.beta_deg),
            gamma_deg=float(self.gamma_deg),
            label=label,
            length_unit=self.length_unit,
        )
        resolve_point_group(symbol)

        object.__setattr__(self, "label", label)
        object.__setattr__(self, "point_group", symbol)
        object.__setattr__(self, "b", lattice.b)
        object.__setattr__(self, "c", lattice.c)

    @property
    def lattice(self) -> Lattice:
        return Lattice(
            a=float(self.a),
            b=float(self.b),
            c=float(self.c),
            alpha_deg=float(self.alpha_deg),
            beta_deg=float(self.beta_deg),
            gamma_deg=float(self.gamma_deg),
            label=self.label,
            length_unit=self.length_unit,
        )

    @classmethod
    def cubic(cls, label: str, point_group: str, a: float, **kwargs: Any) -> "CrystalInput":
        return cls(label=label, point_group=point_group, a=a, b=a, c=a, **kwargs)

    @classmethod
    def tetragonal(
        cls, label: str, point_group: str, a: float, c: float, **kwargs: Any
    ) -> "CrystalInput":
        return cls(label=label, point_group=point_group, a=a, b=a, c=c, **kwargs)

    @classmethod
    def orthorhombic(
        cls, label: str, point_group: str, a: float, b: float, c: float, **kwargs: Any
    ) -> "CrystalInput":
        return cls(label=label, point_group=point_group, a=a, b=b, c=c, **kwargs)

    @classmethod
    def hexagonal(
        cls, label: str, point_group: str, a: float, c: float, **kwargs: Any
    ) -> "CrystalInput":
        return cls(
            label=label, point_group=point_group, a=a, b=a, c=c,
            alpha_deg=90.0, beta_deg=90.0, gamma_deg=120.0, **kwargs
        )

    @classmethod
    def monoclinic_unique_b(
        cls,
        label: str,
        point_group: str,
        a: float,
        b: float,
        c: float,
        beta_deg: float,
        **kwargs: Any,
    ) -> "CrystalInput":
        return cls(
            label=label, point_group=point_group, a=a, b=b, c=c,
            alpha_deg=90.0, beta_deg=beta_deg, gamma_deg=90.0, **kwargs
        )

    def _phase_payload(self, phase_id: str) -> dict[str, object]:
        return {
            "phase_id": phase_id,
            "label": self.label,
            "physical_phase": self.label,
            "cell_representation": "user_cell",
            "cell": {
                "a": float(self.a),
                "b": float(self.b),
                "c": float(self.c),
                "alpha_deg": float(self.alpha_deg),
                "beta_deg": float(self.beta_deg),
                "gamma_deg": float(self.gamma_deg),
                "length_unit": self.length_unit,
            },
            "point_group": self.point_group,
        }


@dataclass(frozen=True)
class BasisVectorMapping:
    """One exact direct-space mapping u_parent -> u_product."""

    parent: tuple[object, object, object] | str
    product: tuple[object, object, object] | str

    @property
    def parent_vector(self) -> sp.Matrix:
        return exact_vector3(self.parent, name="parent basis vector")

    @property
    def product_vector(self) -> sp.Matrix:
        return exact_vector3(self.product, name="product basis vector")


@dataclass(frozen=True)
class CorrespondenceAudit:
    input_mapping: str
    determinant_exact: str
    orientation_preserving: bool
    inverse_roundtrip_exact: bool
    basis_mapping_count: int
    all_basis_mappings_exact: bool
    dual_plane_roundtrip_exact: bool

    @property
    def passed(self) -> bool:
        return (
            self.orientation_preserving
            and self.inverse_roundtrip_exact
            and self.all_basis_mappings_exact
            and self.dual_plane_roundtrip_exact
        )

    def assert_passed(self) -> None:
        if self.passed:
            return

        failures: list[str] = []
        if not self.orientation_preserving:
            failures.append(
                "det(C_M_from_A) must be positive for a physical "
                "orientation-preserving deformation"
            )
        if not self.inverse_roundtrip_exact:
            failures.append("exact inverse round-trip failed")
        if not self.all_basis_mappings_exact:
            failures.append("one or more verification basis mappings are inconsistent")
        if not self.dual_plane_roundtrip_exact:
            failures.append("direct/reciprocal dual plane round-trip failed")

        raise ValueError(
            "Invalid physical correspondence: " + "; ".join(failures)
        )


@dataclass(frozen=True)
class CorrespondenceInput:
    """A correspondence that cannot be direction-ambiguous."""

    raw_matrix: sp.Matrix
    matrix_maps: MatrixMapping
    verification_mappings: tuple[BasisVectorMapping, ...] = ()
    label: str = "user correspondence"

    @classmethod
    def from_matrix(
        cls,
        matrix: object,
        *,
        matrix_maps: MatrixMapping | str,
        verification_mappings: Iterable[BasisVectorMapping] = (),
        label: str = "user correspondence",
    ) -> "CorrespondenceInput":
        try:
            direction = MatrixMapping(matrix_maps)
        except Exception as exc:
            raise ValueError(
                "matrix_maps must explicitly be 'parent_to_product' or "
                "'product_to_parent'. A bare correspondence matrix is unsafe."
            ) from exc
        return cls(
            exact_matrix3(matrix, name="correspondence matrix"),
            direction,
            tuple(verification_mappings),
            label.strip() or "user correspondence",
        )

    @classmethod
    def from_basis_mappings(
        cls,
        mappings: Sequence[BasisVectorMapping],
        *,
        label: str = "basis-derived correspondence",
    ) -> "CorrespondenceInput":
        if len(mappings) != 3:
            raise ValueError(
                "Exactly three independent basis-vector mappings are required "
                "to derive a 3-D correspondence."
            )
        parent = sp.Matrix.hstack(*(item.parent_vector for item in mappings))
        product = sp.Matrix.hstack(*(item.product_vector for item in mappings))
        if sp.simplify(parent.det()) == 0:
            raise ValueError("Parent basis vectors are linearly dependent.")
        if sp.simplify(product.det()) == 0:
            raise ValueError("Product basis vectors are linearly dependent.")
        matrix = sp.simplify(product * parent.inv())
        return cls(matrix, MatrixMapping.PARENT_TO_PRODUCT, tuple(mappings), label)

    @classmethod
    def from_user_payload(cls, payload: Mapping[str, object]) -> "CorrespondenceInput":
        if not isinstance(payload, Mapping):
            raise TypeError("Correspondence input must be an object.")
        has_matrix = "matrix" in payload
        has_pairs = "basis_mappings" in payload
        if has_matrix == has_pairs:
            raise ValueError("Supply exactly one of 'matrix' or 'basis_mappings'.")

        allowed = (
            {"matrix", "matrix_maps", "verification_mappings", "label"}
            if has_matrix
            else {"basis_mappings", "label"}
        )
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                "Unknown correspondence input fields are refused rather than "
                f"silently ignored: {sorted(str(x) for x in unknown)}"
            )

        label = str(payload.get("label", "user correspondence"))
        if has_matrix:
            if "matrix_maps" not in payload:
                raise ValueError(
                    "A correspondence matrix is ambiguous without 'matrix_maps'. "
                    "Choose parent_to_product or product_to_parent."
                )
            verifications: list[BasisVectorMapping] = []
            raw_verify = payload.get("verification_mappings", [])
            if not isinstance(raw_verify, Sequence):
                raise TypeError("verification_mappings must be a list.")
            for item in raw_verify:
                if not isinstance(item, Mapping):
                    raise TypeError("Each verification mapping must be an object.")
                if set(item) != {"parent", "product"}:
                    raise ValueError(
                        "Each verification mapping requires exactly parent and product."
                    )
                verifications.append(
                    BasisVectorMapping(item["parent"], item["product"])  # type: ignore[arg-type]
                )
            return cls.from_matrix(
                payload["matrix"],
                matrix_maps=str(payload["matrix_maps"]),
                verification_mappings=verifications,
                label=label,
            )

        raw = payload["basis_mappings"]
        if not isinstance(raw, Sequence):
            raise TypeError("basis_mappings must be a list.")
        mappings: list[BasisVectorMapping] = []
        for item in raw:
            if not isinstance(item, Mapping) or set(item) != {"parent", "product"}:
                raise ValueError(
                    "Each basis mapping requires exactly parent and product."
                )
            mappings.append(
                BasisVectorMapping(item["parent"], item["product"])  # type: ignore[arg-type]
            )
        return cls.from_basis_mappings(mappings, label=label)

    @property
    def C_product_from_parent(self) -> sp.Matrix:
        if self.matrix_maps is MatrixMapping.PARENT_TO_PRODUCT:
            return sp.simplify(self.raw_matrix)
        return sp.simplify(self.raw_matrix.inv())

    def audit(self) -> CorrespondenceAudit:
        C = self.C_product_from_parent
        Ci = sp.simplify(C.inv())
        inverse_ok = (
            sp.simplify(C * Ci - sp.eye(3)) == sp.zeros(3)
            and sp.simplify(Ci * C - sp.eye(3)) == sp.zeros(3)
        )

        mappings_ok = True
        for mapping in self.verification_mappings:
            residual = sp.simplify(C * mapping.parent_vector - mapping.product_vector)
            if residual != sp.zeros(3, 1):
                mappings_ok = False
                break

        p_parent = sp.Matrix([2, -1, 3])
        p_product = sp.simplify(C.inv().T * p_parent)
        p_roundtrip = sp.simplify(C.T * p_product)
        dual_ok = p_roundtrip == p_parent

        determinant = sp.Rational(sp.simplify(C.det()))
        orientation_preserving = bool(determinant > 0)

        return CorrespondenceAudit(
            input_mapping=self.matrix_maps.value,
            determinant_exact=str(determinant),
            orientation_preserving=orientation_preserving,
            inverse_roundtrip_exact=inverse_ok,
            basis_mapping_count=len(self.verification_mappings),
            all_basis_mappings_exact=mappings_ok,
            dual_plane_roundtrip_exact=dual_ok,
        )


@dataclass(frozen=True)
class RepresentationAuditSummary:
    convention_pairs_checked: int
    maximum_internal_parity_residual: float
    maximum_principal_stretch_disagreement: float

    @property
    def passed(self) -> bool:
        return (
            self.maximum_internal_parity_residual <= 1.0e-9
            and self.maximum_principal_stretch_disagreement <= 1.0e-9
        )

    def assert_passed(self) -> None:
        if not self.passed:
            raise AssertionError(f"Representation parity failed: {self}")


@dataclass(frozen=True)
class UserAnalysis:
    subgroup_order: int
    variant_count: int
    operator_count: int
    principal_stretches: tuple[float, float, float]
    lambda2_minus_one: float
    ct_exact_austenite_martensite_compatibility: bool
    ball_james_lambda2_exact: bool
    representation_audit: RepresentationAuditSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "subgroup_order": self.subgroup_order,
            "variant_count": self.variant_count,
            "operator_count": self.operator_count,
            "principal_stretches": list(self.principal_stretches),
            "lambda2_minus_one": self.lambda2_minus_one,
            "ct_exact_austenite_martensite_compatibility": (
                self.ct_exact_austenite_martensite_compatibility
            ),
            "ball_james_lambda2_exact": self.ball_james_lambda2_exact,
            "representation_consistent": self.representation_audit.passed,
            "advanced": {
                "cartesian_convention_pairs_crosschecked": (
                    self.representation_audit.convention_pairs_checked
                ),
                "maximum_internal_parity_residual": (
                    self.representation_audit.maximum_internal_parity_residual
                ),
                "maximum_principal_stretch_disagreement": (
                    self.representation_audit.maximum_principal_stretch_disagreement
                ),
            },
        }


@dataclass(frozen=True)
class TransformationInput:
    """Normal-user transformation input; no metric tensor or Cartesian frame required."""

    parent: CrystalInput
    product: CrystalInput
    correspondence: CorrespondenceInput
    transformation_id: str = "A_to_M"

    def compile(self) -> "CompiledTransformation":
        audit = self.correspondence.audit()
        audit.assert_passed()
        C = self.correspondence.C_product_from_parent

        payload: dict[str, object] = {
            "schema_version": 1,
            "project_id": "user_project",
            "title": "user_project",
            "phases": [
                self.parent._phase_payload("A"),
                self.product._phase_payload("M"),
            ],
            "transformations": [
                {
                    "transformation_id": self.transformation_id,
                    "label": self.transformation_id,
                    "parent_phase_id": "A",
                    "product_phase_id": "M",
                    "correspondence_M_from_A": _matrix_as_json(C),
                    "correspondence_label": self.correspondence.label,
                }
            ],
            "orientations": [],
        }
        loaded = project_from_dict(payload, source="user_input")
        loaded.project.validate().assert_passed()
        return CompiledTransformation(
            loaded=loaded,
            transformation_id=self.transformation_id,
            correspondence_audit=audit,
        )


@dataclass(frozen=True)
class OrientationParallelisms:
    parent_plane: str
    product_plane: str
    parent_direction: str
    product_direction: str
    unoriented_directions: bool = True


@dataclass(frozen=True)
class OrientationCandidateSummary:
    first_parallelism_residual_deg: float
    second_parallelism_residual_deg: float
    intersection_order: int
    variant_count: int
    operator_count: int


@dataclass(frozen=True)
class OrientationAnalysis:
    candidate_count: int
    candidates: tuple[OrientationCandidateSummary, ...]


@dataclass(frozen=True)
class CompiledTransformation:
    loaded: LoadedProject
    transformation_id: str
    correspondence_audit: CorrespondenceAudit

    @property
    def project(self):
        return self.loaded.project

    def representation_audit(self, *, tolerance: float = 1.0e-9) -> RepresentationAuditSummary:
        transformation = self.project.transformation(self.transformation_id)
        parent = self.project.phase(transformation.parent_phase_id)
        product = self.project.phase(transformation.product_phase_id)

        conventions = tuple(CartesianConvention)
        principal_sets: list[np.ndarray] = []
        max_internal = 0.0

        for parent_convention in conventions:
            for product_convention in conventions:
                bridge = RepresentationBridge(
                    parent.lattice,
                    product.lattice,
                    transformation.correspondence,
                    parent_convention,
                    product_convention,
                )
                parity = bridge.audit()
                if not math.isfinite(parity.maximum_residual):
                    raise AssertionError(
                        "Representation audit produced a non-finite residual."
                    )
                max_internal = max(max_internal, parity.maximum_residual)
                principal_sets.append(bridge.principal_stretches())

        baseline = principal_sets[0]
        max_stretch = 0.0
        for current in principal_sets[1:]:
            scale = max(
                float(np.linalg.norm(baseline)),
                float(np.linalg.norm(current)),
                1.0,
            )
            max_stretch = max(
                max_stretch,
                float(np.linalg.norm(current - baseline) / scale),
            )

        result = RepresentationAuditSummary(
            convention_pairs_checked=len(principal_sets),
            maximum_internal_parity_residual=max_internal,
            maximum_principal_stretch_disagreement=max_stretch,
        )
        if max(
            result.maximum_internal_parity_residual,
            result.maximum_principal_stretch_disagreement,
        ) > tolerance:
            raise AssertionError(
                "The physical transformation changed with Cartesian convention: "
                f"{result}"
            )
        return result

    def bundle(self) -> TransformationBundle:
        service = CalculationService(self.project)
        value = service.compute(
            CalculationKind.TRANSFORMATION_BUNDLE,
            self.transformation_id,
        )
        if not isinstance(value, TransformationBundle):
            raise TypeError("Calculation service returned an unexpected bundle type.")
        return value

    def analyze(self) -> UserAnalysis:
        representation = self.representation_audit()
        bundle = self.bundle()
        principal = tuple(float(x) for x in bundle.metric.principal_stretches)
        if len(principal) != 3:
            raise AssertionError(
                "Principal-stretch spectrum must contain three values."
            )
        return UserAnalysis(
            subgroup_order=bundle.topology.subgroup_order,
            variant_count=bundle.topology.n_variants,
            operator_count=bundle.topology.n_operators,
            principal_stretches=principal,  # type: ignore[arg-type]
            lambda2_minus_one=float(principal[1] - 1.0),
            ct_exact_austenite_martensite_compatibility=(
                bundle.am_compatibility.ct_exact
            ),
            ball_james_lambda2_exact=(
                bundle.am_compatibility.ball_james_lambda2_exact
            ),
            representation_audit=representation,
        )

    def analyze_orientation(self, relation: OrientationParallelisms) -> OrientationAnalysis:
        service = OrientationService(self.project)
        report = service.state_from_parallelisms(
            "A",
            "M",
            relation.parent_plane,
            relation.product_plane,
            relation.parent_direction,
            relation.product_direction,
            orientation_id="user_or",
            label="user_or",
            unoriented_directions=relation.unoriented_directions,
        )
        summaries: list[OrientationCandidateSummary] = []
        for candidate in report.candidates:
            topology = service.topology(candidate.state)
            audit = topology.audit
            summaries.append(
                OrientationCandidateSummary(
                    first_parallelism_residual_deg=float(
                        candidate.first_parallelism_residual_deg
                    ),
                    second_parallelism_residual_deg=float(
                        candidate.second_parallelism_residual_deg
                    ),
                    intersection_order=int(
                        audit.full_orientation_intersection_order
                    ),
                    variant_count=int(audit.full_orientation_variant_count),
                    operator_count=int(audit.full_orientation_operator_count),
                )
            )
        return OrientationAnalysis(len(summaries), tuple(summaries))
