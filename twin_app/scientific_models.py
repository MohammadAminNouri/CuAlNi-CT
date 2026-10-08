from __future__ import annotations

"""Immutable output contracts for the standalone twin-family engine.

The model deliberately keeps three notions separate:

* correspondence variants M_i (discrete crystallographic topology),
* distinct stretch variants U_j (metric-dependent nonlinear elasticity), and
* physical twin constructions between variants.

That separation prevents a correspondence variant number from being silently
relabelled as a stretch-variant number when metric degeneracy collapses several
correspondence variants onto the same U.
"""

from dataclasses import asdict, dataclass
from typing import Any, Literal

Vector3 = tuple[float, float, float]
IntVector3 = tuple[int, int, int]
Matrix3Text = tuple[
    tuple[str, str, str],
    tuple[str, str, str],
    tuple[str, str, str],
]


@dataclass(frozen=True)
class CorrespondenceVariantRecord:
    variant_index: int
    representative_parent_symmetry: Matrix3Text
    representative_determinant: int
    correspondence_M_from_A: Matrix3Text
    stretch_variant_index: int | None
    stretch_mapping_residual: float | None
    mapping_status: str


@dataclass(frozen=True)
class TwinElementRepresentation:
    route: Literal["I", "II"]
    plane_symbol: str
    direction_symbol: str
    plane_parent_crystal: Vector3
    direction_parent_crystal: Vector3
    plane_product_crystal: Vector3
    direction_product_crystal: Vector3
    shear_magnitude: float
    generator_parent_symmetry: Matrix3Text
    intercorrespondence_product: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class ClassicalTwinSystem:
    system_id: str
    classification: Literal["Type I", "Type II", "Compound"]
    representations: tuple[TwinElementRepresentation, ...]
    shear_magnitude: float
    provenance_count: int


@dataclass(frozen=True)
class HabitPlaneSolution:
    habit_branch: int
    other_variant_volume_fraction: float
    base_variant_volume_fraction: float
    habit_plane_parent_crystal: Vector3
    habit_normal_parent_cartesian: Vector3
    shape_vector_parent_cartesian: Vector3
    shape_vector_parent_crystal: Vector3
    habit_plane_product_crystal_base: Vector3
    frame_plane_residual: float
    frame_shape_vector_residual: float
    rank_one_residual: float
    rotation_residual: float
    middle_stretch_residual: float


@dataclass(frozen=True)
class PairTwinConstruction:
    construction_id: str
    branch: int
    classification: str
    representations: tuple[str, ...]
    classification_status: str
    classical_system_ids: tuple[str, ...]
    shear_magnitude: float
    a_parent_cartesian: Vector3
    n_parent_cartesian: Vector3
    twin_plane_product_crystal: Vector3
    shear_direction_product_crystal: Vector3
    rank_one_residual: float
    rotation_residual: float
    independent_outer_product_residual: float | None
    independent_shear_relative_residual: float | None
    discrete_plane_angle_deg: float | None
    discrete_direction_angle_deg: float | None
    discrete_shear_relative_residual: float | None
    habit_status: str
    habit_solutions: tuple[HabitPlaneSolution, ...]
    continuum_fraction: bool


@dataclass(frozen=True)
class VariantPairRecord:
    pair_id: str
    variant_i: int
    variant_j: int
    operator_forward: int
    operator_reverse: int
    stretch_i: int | None
    stretch_j: int | None
    status: str
    constructions: tuple[PairTwinConstruction, ...]


@dataclass(frozen=True)
class WeakPlaneCandidate:
    operator_index: int
    parent_rotation_order: int
    parent_axis: IntVector3
    product_axis: IntVector3
    plane1_primitive: IntVector3
    plane2_primitive: IntVector3
    generalized_twin_index: int
    generalized_strain: float
    generalized_shear: float
    intraplanar_distortion: float
    maximum_principal_intraplanar_strain: float
    plane_complexity: int


@dataclass(frozen=True)
class TwinFamilyRecord:
    family_id: str
    operator_indices: tuple[int, ...]
    route: str
    representative_pair: tuple[int, int]
    equivalent_pairs: tuple[tuple[int, int], ...]
    classical_systems: tuple[ClassicalTwinSystem, ...]
    weak_status: str
    weak_candidates: tuple[WeakPlaneCandidate, ...]
    pair_records: tuple[VariantPairRecord, ...]


@dataclass(frozen=True)
class ScientificAudit:
    topology_variant_count: int
    stretch_variant_count: int
    correspondence_subgroup_order: int
    operator_count: int
    mapped_correspondence_variant_count: int
    collapsed_correspondence_variant_count: int
    unmapped_correspondence_variant_count: int
    maximum_correspondence_to_stretch_residual: float
    ptmc_maximum_residual: float
    ball_james_maximum_residual: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class TwinFamilyReport:
    transformation_id: str
    parent_phase_id: str
    product_phase_id: str
    correspondence_variants: tuple[CorrespondenceVariantRecord, ...]
    families: tuple[TwinFamilyRecord, ...]
    audit: ScientificAudit

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
