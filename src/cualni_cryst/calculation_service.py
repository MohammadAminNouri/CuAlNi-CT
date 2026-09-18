from __future__ import annotations

"""Dependency-aware crystallographic calculation service.

This module is the boundary between immutable scientific input state and
derived calculations.  It is designed for an interactive application, but it
contains no GUI code.

Two principles drive the implementation:

1. cache keys are based only on the inputs a calculation actually depends on;
2. discrete crystallographic topology is kept separate from metric-dependent
   geometry.

Therefore, for example, changing a monoclinic beta angle invalidates CMC,
stretch, habit-plane and representation calculations, but does not invalidate
the exact correspondence cosets/double-cosets when symmetry and correspondence
remain unchanged.
"""

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

import numpy as np
import sympy as sp

from .ball_james import (
    AnalyticalRankOneSolution,
    single_variant_austenite_habit_solutions,
)
from .ct import CTAMResult, analyze_austenite_martensite, cmc, normalized_cmc, smc
from .group_theory import GroupoidResult, correspondence_groupoid
from .project_state import (
    PhaseState,
    ProjectState,
    ProjectValidationReport,
    TransformationState,
)
from .representation import CartesianConvention, CartesianFrame
from .scientific_contracts import TheoryConsistencyAudit, audit_core_theory_consistency
from .stretch import (
    generate_stretch_variants,
    principal_stretches,
    stretch_from_metrics,
)


class CalculationKind(str, Enum):
    """Built-in calculation nodes in dependency order."""

    PROJECT_VALIDATION = "project_validation"
    DISCRETE_TOPOLOGY = "discrete_topology"
    METRIC_CORE = "metric_core"
    REPRESENTATION = "representation"
    AM_COMPATIBILITY = "am_compatibility"
    SCIENTIFIC_CONTRACTS = "scientific_contracts"
    STRETCH_VARIANTS = "stretch_variants"
    TRANSFORMATION_BUNDLE = "transformation_bundle"


TRANSFORMATION_KINDS = (
    CalculationKind.DISCRETE_TOPOLOGY,
    CalculationKind.METRIC_CORE,
    CalculationKind.REPRESENTATION,
    CalculationKind.AM_COMPATIBILITY,
    CalculationKind.SCIENTIFIC_CONTRACTS,
    CalculationKind.STRETCH_VARIANTS,
    CalculationKind.TRANSFORMATION_BUNDLE,
)


DEPENDENCIES: dict[CalculationKind, tuple[CalculationKind, ...]] = {
    CalculationKind.PROJECT_VALIDATION: (),
    CalculationKind.DISCRETE_TOPOLOGY: (),
    CalculationKind.METRIC_CORE: (),
    CalculationKind.REPRESENTATION: (CalculationKind.METRIC_CORE,),
    CalculationKind.AM_COMPATIBILITY: (CalculationKind.METRIC_CORE,),
    CalculationKind.SCIENTIFIC_CONTRACTS: (CalculationKind.METRIC_CORE,),
    CalculationKind.STRETCH_VARIANTS: (CalculationKind.METRIC_CORE,),
    CalculationKind.TRANSFORMATION_BUNDLE: (
        CalculationKind.DISCRETE_TOPOLOGY,
        CalculationKind.METRIC_CORE,
        CalculationKind.REPRESENTATION,
        CalculationKind.AM_COMPATIBILITY,
        CalculationKind.SCIENTIFIC_CONTRACTS,
        CalculationKind.STRETCH_VARIANTS,
    ),
}


def dependency_order(target: CalculationKind) -> tuple[CalculationKind, ...]:
    """Return a deterministic topological order ending at ``target``."""

    visiting: set[CalculationKind] = set()
    visited: set[CalculationKind] = set()
    order: list[CalculationKind] = []

    def visit(node: CalculationKind) -> None:
        if node in visited:
            return
        if node in visiting:
            raise RuntimeError(f"Calculation dependency cycle detected at {node.value}")
        visiting.add(node)
        for dependency in DEPENDENCIES[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)
        order.append(node)

    visit(target)
    return tuple(order)


def _canonical(value: object) -> object:
    """Canonical JSON-safe representation used only for cache fingerprints."""

    if isinstance(value, float):
        return {"__float_hex__": value.hex()}
    if isinstance(value, np.floating):
        return {"__float_hex__": float(value).hex()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, dict):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _digest(payload: object) -> str:
    encoded = json.dumps(
        _canonical(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _lattice_payload(phase: PhaseState) -> dict[str, object]:
    lattice = phase.lattice
    return {
        "phase_id": phase.phase_id,
        "basis": {
            "phase_id": phase.basis.phase_id,
            "basis_id": phase.basis.basis_id,
            "cell_representation": phase.basis.cell_representation,
        },
        "cell_representation": phase.cell_representation,
        "lattice": (
            lattice.a,
            lattice.b,
            lattice.c,
            lattice.alpha_deg,
            lattice.beta_deg,
            lattice.gamma_deg,
        ),
    }


def _correspondence_payload(transformation: TransformationState) -> list[list[str]]:
    matrix = transformation.correspondence.C_M_from_A
    return [[str(sp.simplify(matrix[i, j])) for j in range(3)] for i in range(3)]


def _symmetry_payload(phase: PhaseState) -> list[list[list[float]]]:
    return [
        [[float(value) for value in row] for row in operator]
        for operator in phase.symmetry_operators
    ]


def _policy_payload(project: ProjectState) -> dict[str, float]:
    policy = project.numerical_policy
    return {
        "algebraic": policy.algebraic,
        "representation": policy.representation,
        "exact_eigenvalue": policy.exact_eigenvalue,
        "rank_one": policy.rank_one,
        "projective_angle_deg": policy.projective_angle_deg,
    }


def _exact_symmetry_group(
    phase: PhaseState,
    *,
    tolerance: float,
) -> list[sp.Matrix]:
    """Recover exact crystallographic operators from validated state matrices.

    Point-group operators in a crystallographic lattice basis are exact
    unimodular matrices.  ``PhaseState`` currently stores a frozen numerical
    copy for serialization.  Here each entry is rationalized and then checked
    against the stored matrix before it is admitted to exact group theory.

    No approximate matrix is silently accepted into a coset calculation.
    """

    out: list[sp.Matrix] = []
    for index, operator in enumerate(phase.symmetry_matrices()):
        exact = sp.Matrix(
            [
                [
                    sp.nsimplify(
                        float(operator[i, j]),
                        tolerance=tolerance,
                        rational=True,
                    )
                    for j in range(3)
                ]
                for i in range(3)
            ]
        )
        reconstructed = np.asarray(exact, dtype=float)
        residual = float(
            np.linalg.norm(reconstructed - operator)
            / max(float(np.linalg.norm(operator)), 1.0)
        )
        if residual > tolerance:
            raise ValueError(
                f"Could not recover exact symmetry operator {index} for "
                f"{phase.phase_id!r}; residual={residual:.3e}"
            )
        determinant = sp.simplify(exact.det())
        if determinant not in {sp.Integer(-1), sp.Integer(1)}:
            raise ValueError(
                f"Recovered symmetry operator {index} for {phase.phase_id!r} "
                f"has exact determinant {determinant}, expected ±1"
            )
        out.append(exact)
    return out


@dataclass(frozen=True)
class MetricCoreResult:
    """Common metric/stretches used by several theories."""

    parent_metric: np.ndarray
    product_metric: np.ndarray
    pulled_product_metric: np.ndarray
    cmc_dimensional: np.ndarray
    cmc_normalized: np.ndarray
    smc_dimensional: np.ndarray
    stretch: np.ndarray
    principal_stretches: np.ndarray
    principal_axes: np.ndarray
    lambda2_residual: float

    def to_dict(self) -> dict[str, object]:
        return {
            "parent_metric": self.parent_metric.tolist(),
            "product_metric": self.product_metric.tolist(),
            "pulled_product_metric": self.pulled_product_metric.tolist(),
            "cmc_dimensional": self.cmc_dimensional.tolist(),
            "cmc_normalized": self.cmc_normalized.tolist(),
            "smc_dimensional": self.smc_dimensional.tolist(),
            "stretch": self.stretch.tolist(),
            "principal_stretches": self.principal_stretches.tolist(),
            "principal_axes": self.principal_axes.tolist(),
            "lambda2_residual": self.lambda2_residual,
        }


@dataclass(frozen=True)
class DiscreteTopologyResult:
    """Exact correspondence topology, independent of lattice parameters."""

    groupoid: GroupoidResult
    parent_group_order: int
    product_group_order: int

    @property
    def subgroup_order(self) -> int:
        return len(self.groupoid.subgroup)

    @property
    def n_variants(self) -> int:
        return self.groupoid.n_variants

    @property
    def n_operators(self) -> int:
        return self.groupoid.n_operators

    def to_dict(self) -> dict[str, object]:
        return {
            "parent_group_order": self.parent_group_order,
            "product_group_order": self.product_group_order,
            "subgroup_order": self.subgroup_order,
            "n_variants": self.n_variants,
            "n_operators": self.n_operators,
            "burnside_count": self.groupoid.burnside_count,
            "operator_summaries": [
                {
                    "index": summary.index,
                    "size": summary.size,
                    "inverse_index": summary.inverse_index,
                    "ambivalent": summary.ambivalent,
                    "symmetry_kinds": list(summary.symmetry_kinds),
                }
                for summary in self.groupoid.summaries
            ],
            "adjacency": self.groupoid.adjacency,
            "inverse_operators": self.groupoid.inverse_operators,
        }


@dataclass(frozen=True)
class RepresentationResult:
    """Selected Cartesian representation and parity audit."""

    parent_convention: str
    product_convention: str
    deformation_gradient: np.ndarray
    right_stretch: np.ndarray
    polar_rotation: np.ndarray
    principal_stretches: np.ndarray
    maximum_parity_residual: float

    def to_dict(self) -> dict[str, object]:
        return {
            "parent_convention": self.parent_convention,
            "product_convention": self.product_convention,
            "deformation_gradient": self.deformation_gradient.tolist(),
            "right_stretch": self.right_stretch.tolist(),
            "polar_rotation": self.polar_rotation.tolist(),
            "principal_stretches": self.principal_stretches.tolist(),
            "maximum_parity_residual": self.maximum_parity_residual,
        }


@dataclass(frozen=True)
class AMCompatibilityResult:
    """Single-variant A/M comparison of CT and Ball-James geometry."""

    ct: CTAMResult
    ball_james_solutions: tuple[AnalyticalRankOneSolution, ...]
    ct_exact: bool
    ball_james_lambda2_exact: bool
    exact_classification_agreement: bool
    ct_bj_habit_normal_angles_deg: np.ndarray

    @property
    def minimum_habit_normal_angle_deg(self) -> float | None:
        if self.ct_bj_habit_normal_angles_deg.size == 0:
            return None
        return float(np.min(self.ct_bj_habit_normal_angles_deg))

    def to_dict(self) -> dict[str, object]:
        return {
            "ct_exact": self.ct_exact,
            "ct_degeneracy_order": self.ct.analysis.degeneracy_order,
            "ct_reason": self.ct.analysis.reason,
            "ct_nearest_zero_residual": self.ct.analysis.nearest_zero_residual,
            "ct_exact_habit_plane_count": len(self.ct.exact_habit_planes),
            "ct_approximate_residual": self.ct.approximate.residual,
            "ct_approximate_admissible_signature": (
                self.ct.approximate.admissible_signature
            ),
            "ball_james_lambda2_exact": self.ball_james_lambda2_exact,
            "ball_james_solution_count": len(self.ball_james_solutions),
            "ball_james_rank_one_residuals": [
                solution.residual for solution in self.ball_james_solutions
            ],
            "exact_classification_agreement": self.exact_classification_agreement,
            "ct_bj_habit_normal_angles_deg": (
                self.ct_bj_habit_normal_angles_deg.tolist()
            ),
            "minimum_habit_normal_angle_deg": self.minimum_habit_normal_angle_deg,
        }


@dataclass(frozen=True)
class StretchVariantsResult:
    """Metric-dependent stretch variants generated in an orthonormal frame."""

    proper_parent_rotation_count: int
    variants: tuple[np.ndarray, ...]

    @property
    def n_variants(self) -> int:
        return len(self.variants)

    def to_dict(self) -> dict[str, object]:
        return {
            "proper_parent_rotation_count": self.proper_parent_rotation_count,
            "n_variants": self.n_variants,
            "variants": [variant.tolist() for variant in self.variants],
        }


@dataclass(frozen=True)
class TransformationBundle:
    """One UI-ready aggregate without making the UI perform crystallography."""

    transformation_id: str
    topology: DiscreteTopologyResult
    metric: MetricCoreResult
    representation: RepresentationResult
    am_compatibility: AMCompatibilityResult
    contracts: TheoryConsistencyAudit
    stretch_variants: StretchVariantsResult

    def to_dict(self) -> dict[str, object]:
        return {
            "transformation_id": self.transformation_id,
            "topology": self.topology.to_dict(),
            "metric": self.metric.to_dict(),
            "representation": self.representation.to_dict(),
            "am_compatibility": self.am_compatibility.to_dict(),
            "contracts": self.contracts.to_dict(),
            "stretch_variants": self.stretch_variants.to_dict(),
        }


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    entries: int


@dataclass(frozen=True)
class InvalidationReport:
    """Which cached scientific domains must change for a proposed new state."""

    transformation_id: str
    changed: tuple[CalculationKind, ...]
    unchanged: tuple[CalculationKind, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "transformation_id": self.transformation_id,
            "changed": [node.value for node in self.changed],
            "unchanged": [node.value for node in self.unchanged],
        }


class CalculationService:
    """Dependency-aware calculation/cache service for an immutable ProjectState."""

    def __init__(self, project: ProjectState):
        self._project = project
        self._cache: dict[tuple[str, str, str], object] = {}
        self._hits = 0
        self._misses = 0

    @property
    def project(self) -> ProjectState:
        return self._project

    def set_project(self, project: ProjectState) -> None:
        """Switch state without destroying reusable fingerprint-addressed cache."""

        self._project = project

    def clear_cache(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def cache_stats(self) -> CacheStats:
        return CacheStats(self._hits, self._misses, len(self._cache))

    def _endpoints(
        self,
        transformation_id: str,
        *,
        project: ProjectState | None = None,
    ) -> tuple[TransformationState, PhaseState, PhaseState]:
        state = project or self._project
        transformation = state.transformation(transformation_id)
        parent = state.phase(transformation.parent_phase_id)
        product = state.phase(transformation.product_phase_id)
        return transformation, parent, product

    def fingerprint(
        self,
        kind: CalculationKind,
        transformation_id: str = "",
        *,
        project: ProjectState | None = None,
    ) -> str:
        """Fingerprint only the scientific inputs relevant to ``kind``."""

        state = project or self._project

        if kind is CalculationKind.PROJECT_VALIDATION:
            return _digest(state.to_dict())

        if not transformation_id:
            raise ValueError(f"{kind.value} requires transformation_id")

        transformation, parent, product = self._endpoints(
            transformation_id,
            project=state,
        )

        topology_payload = {
            "parent_symmetry": _symmetry_payload(parent),
            "product_symmetry": _symmetry_payload(product),
            "correspondence": _correspondence_payload(transformation),
        }
        metric_payload = {
            "parent": _lattice_payload(parent),
            "product": _lattice_payload(product),
            "correspondence": _correspondence_payload(transformation),
        }

        if kind is CalculationKind.DISCRETE_TOPOLOGY:
            return _digest(topology_payload)
        if kind is CalculationKind.METRIC_CORE:
            return _digest(metric_payload)
        if kind is CalculationKind.REPRESENTATION:
            return _digest(
                {
                    "metric": metric_payload,
                    "parent_convention": transformation.parent_cartesian_convention,
                    "product_convention": transformation.product_cartesian_convention,
                }
            )
        if kind is CalculationKind.AM_COMPATIBILITY:
            return _digest(
                {
                    "metric": metric_payload,
                    "exact_eigenvalue": state.numerical_policy.exact_eigenvalue,
                    "rank_one": state.numerical_policy.rank_one,
                }
            )
        if kind is CalculationKind.SCIENTIFIC_CONTRACTS:
            return _digest({"metric": metric_payload, "policy": _policy_payload(state)})
        if kind is CalculationKind.STRETCH_VARIANTS:
            return _digest(
                {
                    "metric": metric_payload,
                    "parent_symmetry": _symmetry_payload(parent),
                    "algebraic_tolerance": state.numerical_policy.algebraic,
                    "representation_tolerance": state.numerical_policy.representation,
                }
            )
        if kind is CalculationKind.TRANSFORMATION_BUNDLE:
            return _digest(
                {
                    dependency.value: self.fingerprint(
                        dependency,
                        transformation_id,
                        project=state,
                    )
                    for dependency in DEPENDENCIES[kind]
                }
            )
        raise ValueError(f"Unsupported calculation kind: {kind}")

    def impact(
        self,
        proposed_project: ProjectState,
        transformation_id: str,
    ) -> InvalidationReport:
        """Compare current/proposed states without performing calculations."""

        changed: list[CalculationKind] = []
        unchanged: list[CalculationKind] = []

        for kind in TRANSFORMATION_KINDS:
            current = self.fingerprint(kind, transformation_id)
            proposed = self.fingerprint(
                kind,
                transformation_id,
                project=proposed_project,
            )
            (changed if current != proposed else unchanged).append(kind)

        return InvalidationReport(
            transformation_id,
            tuple(changed),
            tuple(unchanged),
        )

    def compute(
        self,
        kind: CalculationKind,
        transformation_id: str = "",
    ) -> object:
        """Compute one node, reusing cache entries with identical fingerprints."""

        fingerprint = self.fingerprint(kind, transformation_id)
        key = (kind.value, transformation_id, fingerprint)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]

        self._misses += 1
        result = self._compute_uncached(kind, transformation_id)
        self._cache[key] = result
        return result

    def _validated_project(self) -> ProjectValidationReport:
        report = self.compute(CalculationKind.PROJECT_VALIDATION)
        if not isinstance(report, ProjectValidationReport):
            raise TypeError("Internal project-validation result has wrong type")
        report.assert_passed()
        return report

    def _metric_result(self, transformation_id: str) -> MetricCoreResult:
        result = self.compute(CalculationKind.METRIC_CORE, transformation_id)
        if not isinstance(result, MetricCoreResult):
            raise TypeError("Internal metric-core result has wrong type")
        return result

    def _compute_uncached(
        self,
        kind: CalculationKind,
        transformation_id: str,
    ) -> object:
        if kind is CalculationKind.PROJECT_VALIDATION:
            return self._project.validate()

        self._validated_project()
        transformation, parent, product = self._endpoints(transformation_id)

        if kind is CalculationKind.DISCRETE_TOPOLOGY:
            return self._compute_topology(transformation, parent, product)
        if kind is CalculationKind.METRIC_CORE:
            return self._compute_metric_core(transformation, parent, product)
        if kind is CalculationKind.REPRESENTATION:
            return self._compute_representation(transformation_id)
        if kind is CalculationKind.AM_COMPATIBILITY:
            return self._compute_am_compatibility(
                transformation,
                parent,
                product,
                transformation_id,
            )
        if kind is CalculationKind.SCIENTIFIC_CONTRACTS:
            return audit_core_theory_consistency(
                parent.lattice,
                product.lattice,
                transformation.correspondence,
                policy=self._project.numerical_policy,
            )
        if kind is CalculationKind.STRETCH_VARIANTS:
            return self._compute_stretch_variants(parent, transformation_id)
        if kind is CalculationKind.TRANSFORMATION_BUNDLE:
            return self._compute_bundle(transformation_id)

        raise ValueError(f"Unsupported calculation kind: {kind}")

    def _compute_topology(
        self,
        transformation: TransformationState,
        parent: PhaseState,
        product: PhaseState,
    ) -> DiscreteTopologyResult:
        tolerance = self._project.numerical_policy.algebraic
        parent_group = _exact_symmetry_group(parent, tolerance=tolerance)
        product_group = _exact_symmetry_group(product, tolerance=tolerance)
        groupoid = correspondence_groupoid(
            parent_group,
            product_group,
            transformation.correspondence,
        )
        return DiscreteTopologyResult(
            groupoid,
            len(parent_group),
            len(product_group),
        )

    @staticmethod
    def _compute_metric_core(
        transformation: TransformationState,
        parent: PhaseState,
        product: PhaseState,
    ) -> MetricCoreResult:
        M_a = parent.lattice.metric()
        M_m = product.lattice.metric()
        correspondence = transformation.correspondence
        pulled = correspondence.pullback_product_metric(M_m)
        U = stretch_from_metrics(M_a, M_m, correspondence)
        lambdas, axes = principal_stretches(U)

        return MetricCoreResult(
            parent_metric=M_a,
            product_metric=M_m,
            pulled_product_metric=pulled,
            cmc_dimensional=cmc(M_a, M_m, correspondence),
            cmc_normalized=normalized_cmc(M_a, M_m, correspondence),
            smc_dimensional=smc(M_a, M_m, correspondence),
            stretch=U,
            principal_stretches=lambdas,
            principal_axes=axes,
            lambda2_residual=float(abs(lambdas[1] - 1.0)),
        )

    def _compute_representation(
        self,
        transformation_id: str,
    ) -> RepresentationResult:
        transformation = self._project.transformation(transformation_id)
        bridge = self._project.bridge(transformation_id)
        parity = bridge.audit()
        parity.assert_within(self._project.numerical_policy.representation)

        return RepresentationResult(
            parent_convention=transformation.parent_cartesian_convention.value,
            product_convention=transformation.product_cartesian_convention.value,
            deformation_gradient=bridge.deformation_gradient_cartesian(),
            right_stretch=bridge.right_stretch_cartesian(),
            polar_rotation=bridge.polar_rotation_cartesian(),
            principal_stretches=bridge.principal_stretches(),
            maximum_parity_residual=parity.maximum_residual,
        )

    def _compute_am_compatibility(
        self,
        transformation: TransformationState,
        parent: PhaseState,
        product: PhaseState,
        transformation_id: str,
    ) -> AMCompatibilityResult:
        policy = self._project.numerical_policy
        metric = self._metric_result(transformation_id)
        ct_result = analyze_austenite_martensite(
            metric.parent_metric,
            metric.product_metric,
            transformation.correspondence,
            tol=policy.exact_eigenvalue,
        )

        lambda2_exact = bool(metric.lambda2_residual <= policy.exact_eigenvalue)
        squared_tolerance = max(
            policy.rank_one,
            2.0 * policy.exact_eigenvalue + policy.exact_eigenvalue**2,
        )
        bj_solutions = tuple(
            single_variant_austenite_habit_solutions(
                metric.stretch,
                tol=squared_tolerance,
            )
        )

        symmetric_frame = CartesianFrame(
            parent.lattice,
            CartesianConvention.SYMMETRIC_METRIC,
        )
        ct_normals = [
            symmetric_frame.plane_to_cartesian(plane, normalize=True)
            for plane in ct_result.exact_habit_planes
        ]
        bj_normals = [
            np.asarray(solution.n, dtype=float)
            / max(float(np.linalg.norm(solution.n)), 1.0e-15)
            for solution in bj_solutions
        ]

        angles = np.empty((len(ct_normals), len(bj_normals)), dtype=float)
        for i, ct_normal in enumerate(ct_normals):
            for j, bj_normal in enumerate(bj_normals):
                cosine = float(np.clip(abs(ct_normal @ bj_normal), 0.0, 1.0))
                angles[i, j] = float(np.rad2deg(np.arccos(cosine)))

        return AMCompatibilityResult(
            ct=ct_result,
            ball_james_solutions=bj_solutions,
            ct_exact=ct_result.analysis.exact_compatible,
            ball_james_lambda2_exact=lambda2_exact,
            exact_classification_agreement=(
                ct_result.analysis.exact_compatible == lambda2_exact
            ),
            ct_bj_habit_normal_angles_deg=angles,
        )

    def _compute_stretch_variants(
        self,
        parent: PhaseState,
        transformation_id: str,
    ) -> StretchVariantsResult:
        policy = self._project.numerical_policy
        metric = self._metric_result(transformation_id)
        frame = CartesianFrame(
            parent.lattice,
            CartesianConvention.SYMMETRIC_METRIC,
        )

        proper_rotations: list[np.ndarray] = []
        for index, operator in enumerate(parent.symmetry_matrices()):
            Q = frame.operator_to_cartesian(operator)
            determinant = float(np.linalg.det(Q))
            if determinant < 0.0:
                continue

            orthogonality = float(np.linalg.norm(Q.T @ Q - np.eye(3)))
            if orthogonality > policy.representation:
                raise ValueError(
                    f"Parent proper symmetry {index} is not orthogonal in the "
                    f"metric frame; residual={orthogonality:.3e}"
                )
            proper_rotations.append(Q)

        variants = generate_stretch_variants(
            metric.stretch,
            proper_rotations,
            tol=policy.algebraic,
        )
        return StretchVariantsResult(
            proper_parent_rotation_count=len(proper_rotations),
            variants=tuple(variants),
        )

    def _compute_bundle(self, transformation_id: str) -> TransformationBundle:
        topology = self.compute(CalculationKind.DISCRETE_TOPOLOGY, transformation_id)
        metric = self.compute(CalculationKind.METRIC_CORE, transformation_id)
        representation = self.compute(CalculationKind.REPRESENTATION, transformation_id)
        compatibility = self.compute(
            CalculationKind.AM_COMPATIBILITY,
            transformation_id,
        )
        contracts = self.compute(
            CalculationKind.SCIENTIFIC_CONTRACTS,
            transformation_id,
        )
        stretch_variants = self.compute(
            CalculationKind.STRETCH_VARIANTS,
            transformation_id,
        )

        if not isinstance(topology, DiscreteTopologyResult):
            raise TypeError("Internal topology result has wrong type")
        if not isinstance(metric, MetricCoreResult):
            raise TypeError("Internal metric result has wrong type")
        if not isinstance(representation, RepresentationResult):
            raise TypeError("Internal representation result has wrong type")
        if not isinstance(compatibility, AMCompatibilityResult):
            raise TypeError("Internal A/M compatibility result has wrong type")
        if not isinstance(contracts, TheoryConsistencyAudit):
            raise TypeError("Internal scientific-contract result has wrong type")
        if not isinstance(stretch_variants, StretchVariantsResult):
            raise TypeError("Internal stretch-variant result has wrong type")

        return TransformationBundle(
            transformation_id,
            topology,
            metric,
            representation,
            compatibility,
            contracts,
            stretch_variants,
        )
