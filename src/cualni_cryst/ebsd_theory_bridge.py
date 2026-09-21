from __future__ import annotations

"""Theory-to-EBSD bridge for arbitrary parent/product crystal symmetries.

The bridge starts from one *independently supplied* physical orientation
relationship

    x_A = R_A_from_M x_M

and constructs the complete set of crystallographically distinct product
variants.  No experimental orientation is used to create the theoretical
variant or operator library.

A parent symmetry S_A generates a raw child variant

    R_k = S_A R_A_from_M.

Two raw variants represent the same physical child orientation exactly when

    R_i S_M = R_j

for some proper product symmetry S_M.  The implementation therefore performs
the quotient by the complete product proper point group, then cross-checks the
variant count against the independent metric-native OrientationKernel topology.

The resulting VariantOR objects feed the vendor-neutral EBSD reconstruction
layer.  This module never identifies a correspondence C, a deformation F,
a stretch U, a polar rotation, and an OR with one another.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .ebsd_map import EBSDPhase
from .ebsd_reconstruction import (
    OperatorClass,
    VariantOR,
    build_operator_library,
)
from .orientation_kernel import OrientationKernel, require_so3, rotation_angle_deg
from .point_groups import point_group_operations


def right_product_quotient_distance_deg(
    first_R_parent_from_product: np.ndarray,
    second_R_parent_from_product: np.ndarray,
    product_symmetry: Sequence[np.ndarray],
) -> float:
    """Distance between OR matrices modulo proper product symmetry."""

    A = require_so3(
        first_R_parent_from_product,
        tolerance=2.0e-8,
        name="first OR",
    )
    B = require_so3(
        second_R_parent_from_product,
        tolerance=2.0e-8,
        name="second OR",
    )
    if not product_symmetry:
        raise ValueError("product_symmetry must not be empty")

    best = 180.0
    for symmetry in product_symmetry:
        S = require_so3(symmetry, tolerance=2.0e-8, name="product symmetry")
        # R and R S describe the same product crystal orientation.
        best = min(
            best,
            rotation_angle_deg(A @ S @ B.T, tolerance=2.0e-8),
        )
    return float(best)


def orientation_relationship_distance_deg(
    first_R_parent_from_product: np.ndarray,
    second_R_parent_from_product: np.ndarray,
    parent_symmetry: Sequence[np.ndarray],
    product_symmetry: Sequence[np.ndarray],
) -> float:
    """Full parent/product symmetry quotient distance between two ORs.

    Equivalent OR representatives satisfy

        R' = S_A R S_M^{-1}.

    Since a proper point group is closed under inversion, minimizing
    ``angle(S_A R1 S_M R2.T)`` covers the complete quotient.
    """

    R1 = require_so3(
        first_R_parent_from_product,
        tolerance=2.0e-8,
        name="first OR",
    )
    R2 = require_so3(
        second_R_parent_from_product,
        tolerance=2.0e-8,
        name="second OR",
    )
    if not parent_symmetry or not product_symmetry:
        raise ValueError("parent/product symmetry lists must not be empty")

    best = 180.0
    for parent in parent_symmetry:
        SA = require_so3(parent, tolerance=2.0e-8, name="parent symmetry")
        for product in product_symmetry:
            SM = require_so3(product, tolerance=2.0e-8, name="product symmetry")
            best = min(
                best,
                rotation_angle_deg(
                    SA @ R1 @ SM @ R2.T,
                    tolerance=2.0e-8,
                ),
            )
    return float(best)


@dataclass(frozen=True)
class ORVariantSet:
    base_R_parent_from_product: np.ndarray
    variants: tuple[VariantOR, ...]
    parent_proper_group_order: int
    product_proper_group_order: int
    proper_intersection_order: int | None
    topology_expected_variant_count: int | None
    quotient_deduplication_tolerance_deg: float

    @property
    def n_variants(self) -> int:
        return len(self.variants)


def build_or_variant_set(
    parent_phase: EBSDPhase,
    product_phase: EBSDPhase,
    base_R_parent_from_product: np.ndarray,
    *,
    quotient_tolerance_deg: float = 2.0e-7,
    crosscheck_topology: bool = True,
) -> ORVariantSet:
    """Build all physical OR variants without experimental fitting."""

    if quotient_tolerance_deg <= 0.0:
        raise ValueError("quotient_tolerance_deg must be positive")
    R0 = require_so3(
        base_R_parent_from_product,
        tolerance=2.0e-8,
        name="base OR",
    )

    parent_sym = parent_phase.proper_symmetry_cartesian
    product_sym = product_phase.proper_symmetry_cartesian

    matrices: list[np.ndarray] = []
    labels: list[str] = []
    for parent_index, SA in enumerate(parent_sym):
        candidate = np.asarray(SA, dtype=float) @ R0
        if any(
            right_product_quotient_distance_deg(
                candidate,
                existing,
                product_sym,
            )
            <= quotient_tolerance_deg
            for existing in matrices
        ):
            continue
        matrices.append(candidate)
        labels.append(f"parent_symmetry_{parent_index}")

    intersection_order: int | None = None
    expected: int | None = None
    if crosscheck_topology:
        kernel = OrientationKernel(
            parent_phase.lattice.metric(),
            product_phase.lattice.metric(),
            point_group_operations(parent_phase.point_group),
            point_group_operations(product_phase.point_group),
        )
        topology = kernel.topology(R0)
        intersection_order = len(topology.proper_intersection)
        expected = topology.proper_variant_count
        if len(matrices) != expected:
            raise AssertionError(
                "OR quotient enumeration disagrees with independent "
                "OrientationKernel topology: "
                f"enumerated={len(matrices)}, topology={expected}"
            )

    variants = tuple(
        VariantOR(index, matrix, labels[index])
        for index, matrix in enumerate(matrices)
    )
    return ORVariantSet(
        base_R_parent_from_product=R0,
        variants=variants,
        parent_proper_group_order=len(parent_sym),
        product_proper_group_order=len(product_sym),
        proper_intersection_order=intersection_order,
        topology_expected_variant_count=expected,
        quotient_deduplication_tolerance_deg=float(
            quotient_tolerance_deg
        ),
    )


@dataclass(frozen=True)
class TheoryLibrary:
    variant_set: ORVariantSet
    boundary_operators: tuple[OperatorClass, ...]

    @property
    def n_variants(self) -> int:
        return self.variant_set.n_variants

    @property
    def n_boundary_operators(self) -> int:
        return len(self.boundary_operators)


def build_theory_library(
    parent_phase: EBSDPhase,
    product_phase: EBSDPhase,
    base_R_parent_from_product: np.ndarray,
    *,
    quotient_tolerance_deg: float = 2.0e-7,
    operator_equivalence_tolerance_deg: float = 2.0e-6,
    crosscheck_topology: bool = True,
) -> TheoryLibrary:
    variant_set = build_or_variant_set(
        parent_phase,
        product_phase,
        base_R_parent_from_product,
        quotient_tolerance_deg=quotient_tolerance_deg,
        crosscheck_topology=crosscheck_topology,
    )
    operators = build_operator_library(
        variant_set.variants,
        product_phase.proper_symmetry_cartesian,
        equivalence_tolerance_deg=operator_equivalence_tolerance_deg,
    )
    if variant_set.n_variants > 1 and not operators:
        raise AssertionError(
            "multiple OR variants produced no non-identity boundary operators"
        )
    return TheoryLibrary(variant_set=variant_set, boundary_operators=operators)
