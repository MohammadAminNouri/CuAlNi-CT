from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.ct_orientation import CayronOrientationAdapter
from cualni_cryst.cualni_models import (
    do3_to_6m_branch,
    james_hane_6m_example_lattices,
)
from cualni_cryst.group_theory import correspondence_groupoid
from cualni_cryst.orientation import (
    OrientationService,
    matrix_from_axis_angle,
)
from cualni_cryst.orientation_kernel import (
    OrientationKernel,
    rotation_angle_deg,
    rotation_axis_angle,
)
from cualni_cryst.point_groups import (
    point_group_definitions,
    point_group_operations,
)
from cualni_cryst.project_state import james_hane_6m_reference_project
from cualni_cryst.representation import CartesianConvention
from cualni_cryst.symmetry import matrix_key
from cualni_cryst.twinning_ct import (
    classify_parent_order_two_isometry,
    twins_from_operator,
    type_i_from_parent_reflection,
    type_ii_from_parent_twofold,
)


BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks"


def _expr(value):
    if isinstance(value, (int, float)):
        return sp.Rational(str(value))
    return sp.sympify(str(value), locals={"sqrt": sp.sqrt})


def _matrix(rows):
    return sp.Matrix([[_expr(value) for value in row] for row in rows])


def _lattice_from_spec(spec):
    from cualni_cryst.lattice import Lattice

    kind = spec["kind"]
    if kind == "cubic":
        return Lattice.cubic(spec["a_angstrom"], length_unit="angstrom")
    if kind == "orthorhombic":
        return Lattice.orthorhombic(
            spec["a_angstrom"],
            spec["b_angstrom"],
            spec["c_angstrom"],
            length_unit="angstrom",
        )
    if kind == "monoclinic_unique_b":
        return Lattice.monoclinic_unique_b(
            spec["a_angstrom"],
            spec["b_angstrom"],
            spec["c_angstrom"],
            spec["beta_deg"],
            length_unit="angstrom",
        )
    raise AssertionError(kind)


def _benchmark_kernel(filename: str) -> OrientationKernel:
    manifest = json.loads((BENCHMARK_DIR / filename).read_text())
    inp = manifest["input"]
    A = _lattice_from_spec(inp["parent"]["lattice"])
    M = _lattice_from_spec(inp["product"]["lattice"])
    return OrientationKernel(
        A.metric(),
        M.metric(),
        point_group_operations(inp["parent"]["point_group"]),
        point_group_operations(inp["product"]["point_group"]),
        Correspondence(_matrix(inp["package_C_M_from_A"])),
    )


def _det_plus_unimodular(rng: random.Random, steps: int = 5) -> sp.Matrix:
    P = sp.eye(3)
    for _ in range(steps):
        choice = rng.randrange(3)
        if choice == 0:
            i, j = rng.sample(range(3), 2)
            E = sp.eye(3)
            E[j, i] = rng.choice((-1, 1))
        elif choice == 1:
            i, j = rng.sample(range(3), 2)
            E = sp.eye(3)
            E[i, i] = -1
            E[j, j] = -1
        else:
            # A 3-cycle is an even permutation.
            E = sp.Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        P = sp.simplify(P * E)
    assert P.det() == 1
    if np.linalg.cond(np.asarray(P, dtype=float)) >= 40.0:
        return _det_plus_unimodular(rng, steps=max(2, steps - 1))
    return P


def _all_twins(kernel: OrientationKernel):
    assert kernel.correspondence is not None
    groupoid = correspondence_groupoid(
        list(kernel.G_A),
        list(kernel.G_M),
        kernel.correspondence,
    )
    return [
        twin
        for operator in groupoid.operators
        for twin in twins_from_operator(
            operator,
            kernel.M_A,
            kernel.M_M,
            kernel.correspondence,
        )
    ]


def _candidate_family_distance(first, second) -> float:
    if len(first) != len(second):
        return float("inf")
    worst = 0.0
    for R in first:
        best = min(rotation_angle_deg(R @ S.T) for S in second)
        worst = max(worst, best)
    for S in second:
        best = min(rotation_angle_deg(S @ R.T) for R in first)
        worst = max(worst, best)
    return worst


@pytest.mark.parametrize(
    "angle",
    [0.0, 1.0e-10, 1.0e-8, 1.0e-6, 15.0, 90.0, 179.999999, 180.0],
)
def test_rotation_angle_kernel_is_stable_at_zero_and_pi(angle):
    R = matrix_from_axis_angle(np.array([0.371, -0.492, 0.787]), angle)
    assert rotation_angle_deg(R) == pytest.approx(angle, abs=3.0e-9)
    axis_angle = rotation_axis_angle(R)
    assert axis_angle.angle_deg == pytest.approx(angle, abs=3.0e-9)


@pytest.mark.parametrize(
    "filename",
    ["chen_2000_blind_2h_v1.json", "cayron_2022_niti_ct_v1.json"],
)
def test_c_f_u_r_are_separate_and_polar_decomposition_is_exact(filename):
    kernel = _benchmark_kernel(filename)
    result = kernel.kinematics_from_correspondence()

    assert result.maximum_residual < 2.0e-9
    assert np.allclose(
        result.deformation_M_from_A,
        result.polar_rotation_M_from_A @ result.stretch_A,
        atol=2.0e-10,
        rtol=2.0e-10,
    )
    assert np.all(np.linalg.eigvalsh(result.stretch_A) > 0.0)
    assert np.linalg.det(result.polar_rotation_M_from_A) > 0.0
    assert np.allclose(
        result.polar_orientation_A_from_M,
        result.polar_rotation_M_from_A.T,
        atol=1.0e-12,
    )


def test_generic_polar_kernel_crosslocks_existing_orientation_service():
    service = OrientationService(james_hane_6m_reference_project())
    A, M = james_hane_6m_example_lattices()
    branch = do3_to_6m_branch()
    kernel = OrientationKernel(
        A.metric(),
        M.metric(),
        branch.parent_point_group,
        branch.product_point_group,
        branch.correspondence,
    )
    generic = kernel.kinematics_from_correspondence().polar_orientation_A_from_M

    existing = service.reexpress(
        service.polar_orientation("do3_to_6m_reference"),
        CartesianConvention.SYMMETRIC_METRIC,
        CartesianConvention.SYMMETRIC_METRIC,
    )
    existing_R = np.asarray(existing.R_reference_from_moving, dtype=float)
    assert rotation_angle_deg(generic @ existing_R.T) < 2.0e-7

    kernel_topology = kernel.topology(generic)
    existing_topology = service.topology(existing).audit
    assert kernel_topology.full_variant_count == existing_topology.full_orientation_variant_count
    assert kernel_topology.proper_variant_count == existing_topology.proper_orientation_variant_count
    assert kernel_topology.full_operator_count == existing_topology.full_orientation_operator_count
    assert len(kernel_topology.full_intersection) == existing_topology.full_orientation_intersection_order
    assert kernel_topology.orientation_correspondence_intersections_equal == (
        existing_topology.orientation_correspondence_intersections_equal
    )


def test_symmetry_reduced_disorientation_crosslocks_existing_service():
    service = OrientationService(james_hane_6m_reference_project())
    A, M = james_hane_6m_example_lattices()
    branch = do3_to_6m_branch()
    kernel = OrientationKernel(
        A.metric(), M.metric(),
        branch.parent_point_group, branch.product_point_group,
        branch.correspondence,
    )

    pairs = [
        (
            matrix_from_axis_angle(np.array([1.0, 2.0, 3.0]), 17.0),
            matrix_from_axis_angle(np.array([-2.0, 1.0, 0.5]), 41.0),
        ),
        (
            matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]), 1.0e-6),
            matrix_from_axis_angle(np.array([0.0, 1.0, 0.0]), 179.999),
        ),
    ]
    for index, (R1, R2) in enumerate(pairs):
        state1 = service.state_from_matrix(
            "do3", "6m", R1,
            orientation_id=f"kernel_crosslock_a_{index}",
            reference_convention=CartesianConvention.SYMMETRIC_METRIC,
            moving_convention=CartesianConvention.SYMMETRIC_METRIC,
        )
        state2 = service.state_from_matrix(
            "do3", "6m", R2,
            orientation_id=f"kernel_crosslock_b_{index}",
            reference_convention=CartesianConvention.SYMMETRIC_METRIC,
            moving_convention=CartesianConvention.SYMMETRIC_METRIC,
        )
        expected = service.compare_orientations(
            state1, state2
        ).symmetry_reduced_disorientation_deg
        actual = kernel.disorientation(R1, R2).angle_deg
        assert actual == pytest.approx(expected, abs=2.0e-7)


def test_symmetry_equivalent_orientation_has_zero_generic_disorientation():
    kernel = _benchmark_kernel("cayron_2022_niti_ct_v1.json")
    R = matrix_from_axis_angle(np.array([1.0, -2.0, 0.7]), 33.0)
    SA = kernel.reference_proper_cartesian[-1]
    SM = kernel.moving_proper_cartesian[-1]
    equivalent = SA @ R @ SM.T
    assert kernel.disorientation(R, equivalent).angle_deg < 2.0e-7


def test_niti_published_natural_or_parallelisms_are_constructed_not_assumed():
    kernel = _benchmark_kernel("cayron_2022_niti_ct_v1.json")

    report = kernel.parallelism_orientations(
        np.array([-1.0, 1.0, 1.0]),  # parent B2 direction
        np.array([1.0, 0.0, 1.0]),   # product B19' direction
        np.array([1.0, 1.0, 0.0]),   # parent B2 plane
        np.array([0.0, 1.0, 0.0]),   # product B19' plane
        reference_first_kind="direction",
        moving_first_kind="direction",
        reference_second_kind="plane",
        moving_second_kind="plane",
        projective_first=True,
        projective_second=True,
    )

    assert report.candidates
    assert max(
        max(item.first_residual_deg, item.second_residual_deg)
        for item in report.candidates
    ) < 1.0e-7


def test_generic_closing_gap_kernel_crosslocks_existing_cayron_adapter():
    service = OrientationService(james_hane_6m_reference_project())
    adapter = CayronOrientationAdapter(
        service,
        "do3_to_6m_reference",
        reference_convention=CartesianConvention.SYMMETRIC_METRIC,
        moving_convention=CartesianConvention.SYMMETRIC_METRIC,
    )
    A, M = james_hane_6m_example_lattices()
    branch = do3_to_6m_branch()
    kernel = OrientationKernel(
        A.metric(), M.metric(),
        branch.parent_point_group, branch.product_point_group,
        branch.correspondence,
    )

    found = {"I": False, "II": False}
    for operation in branch.parent_point_group:
        kind = classify_parent_order_two_isometry(operation, A.metric())
        if kind not in {"reflection", "twofold"}:
            continue
        route = "I" if kind == "reflection" else "II"
        if found[route]:
            continue
        try:
            if route == "I":
                twin = type_i_from_parent_reflection(
                    operation, A.metric(), M.metric(), branch.correspondence
                )
                existing = adapter.type_i_from_parent_reflection(operation)
            else:
                twin = type_ii_from_parent_twofold(
                    operation, A.metric(), M.metric(), branch.correspondence
                )
                existing = adapter.type_ii_from_parent_twofold(operation)
        except ValueError as exc:
            if "Collapsed CT" in str(exc):
                continue
            raise

        generic = kernel.closing_gap_orientations(twin)
        generic_family = [item.R_A_from_M for item in generic.candidates]
        existing_family = [
            np.asarray(item.orientation.R_reference_from_moving, dtype=float)
            for item in existing.candidates
        ]
        assert _candidate_family_distance(generic_family, existing_family) < 2.0e-7
        found[route] = True

    assert found == {"I": True, "II": True}


@pytest.mark.parametrize(
    "filename",
    ["chen_2000_blind_2h_v1.json", "cayron_2022_niti_ct_v1.json"],
)
def test_complete_orientation_math_is_covariant_under_exact_basis_change(filename):
    kernel = _benchmark_kernel(filename)
    rng = random.Random(20260921 + sum(map(ord, filename)))
    base_polar = kernel.kinematics_from_correspondence()
    twins = _all_twins(kernel)
    assert twins

    for _ in range(4):
        P_A = _det_plus_unimodular(rng)
        P_M = _det_plus_unimodular(rng)
        rebased, gauge = kernel.rebase(P_A, P_M)

        transformed_polar = rebased.kinematics_from_correspondence()
        assert np.allclose(
            transformed_polar.deformation_M_from_A,
            gauge.transform_deformation(base_polar.deformation_M_from_A),
            atol=3.0e-9,
            rtol=3.0e-9,
        )
        assert np.allclose(
            transformed_polar.stretch_A,
            gauge.transform_reference_stretch(base_polar.stretch_A),
            atol=3.0e-9,
            rtol=3.0e-9,
        )
        expected_R = gauge.transform_orientation(
            base_polar.polar_orientation_A_from_M
        )
        assert rotation_angle_deg(
            transformed_polar.polar_orientation_A_from_M @ expected_R.T
        ) < 3.0e-7

        base_topology = kernel.topology(base_polar.polar_orientation_A_from_M)
        new_topology = rebased.topology(
            transformed_polar.polar_orientation_A_from_M
        )
        assert base_topology.full_variant_count == new_topology.full_variant_count
        assert base_topology.proper_variant_count == new_topology.proper_variant_count
        assert base_topology.full_operator_count == new_topology.full_operator_count
        assert base_topology.proper_operator_count == new_topology.proper_operator_count

        pulled_H = {
            matrix_key(sp.simplify(P_A * g * P_A.inv()))
            for g in new_topology.full_intersection
        }
        assert pulled_H == {
            matrix_key(g) for g in base_topology.full_intersection
        }

        # Cross-check one non-collapsed twin of each available route.
        checked = set()
        for twin in twins:
            if twin.kind in checked:
                continue
            g_prime = sp.simplify(
                P_A.inv() * twin.parent_symmetry * P_A
            )
            if twin.kind == "I":
                twin_prime = type_i_from_parent_reflection(
                    g_prime,
                    rebased.M_A,
                    rebased.M_M,
                    rebased.correspondence,
                )
            else:
                twin_prime = type_ii_from_parent_twofold(
                    g_prime,
                    rebased.M_A,
                    rebased.M_M,
                    rebased.correspondence,
                )

            base_family = [
                item.R_A_from_M
                for item in kernel.closing_gap_orientations(twin).candidates
            ]
            expected_family = [
                gauge.transform_orientation(R) for R in base_family
            ]
            new_family = [
                item.R_A_from_M
                for item in rebased.closing_gap_orientations(twin_prime).candidates
            ]
            assert _candidate_family_distance(expected_family, new_family) < 5.0e-7
            checked.add(twin.kind)

        assert checked


def _invariant_metric(group, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(3, 3))
    A = X.T @ X + np.diag([1.0, 1.7, 2.3])
    total = np.zeros((3, 3), dtype=float)
    for g in group:
        G = np.asarray(g, dtype=float)
        total += G.T @ A @ G
    total /= len(group)
    return 0.5 * (total + total.T)


def test_orientation_kernel_accepts_every_crystallographic_point_group():
    definitions = point_group_definitions()
    assert len(definitions) == 32

    for index, definition_A in enumerate(definitions):
        definition_M = definitions[(7 * index + 11) % len(definitions)]
        G_A = point_group_operations(definition_A.symbol)
        G_M = point_group_operations(definition_M.symbol)
        M_A = _invariant_metric(G_A, 1000 + index)
        M_M = _invariant_metric(G_M, 2000 + index)

        kernel = OrientationKernel(M_A, M_M, G_A, G_M)
        R = matrix_from_axis_angle(
            np.array([1.0 + 0.01 * index, 2.0, -0.7]),
            13.0 + 3.71 * index,
        )
        topology = kernel.topology(R)

        assert len(topology.full_intersection) >= 1
        assert len(topology.proper_intersection) >= 1
        assert topology.full_variant_count == len(G_A) // len(topology.full_intersection)
        assert topology.proper_variant_count == (
            len(kernel.G_A_proper) // len(topology.proper_intersection)
        )
        assert topology.full_burnside_count == topology.full_operator_count
        assert topology.proper_burnside_count == topology.proper_operator_count
        assert kernel.disorientation(R, R).angle_deg < 2.0e-7


def test_invalid_and_underdetermined_inputs_fail_loudly():
    G = point_group_operations("1")
    with pytest.raises(ValueError, match="positive definite"):
        OrientationKernel(np.diag([1.0, 1.0, -1.0]), np.eye(3), G, G)

    with pytest.raises(ValueError, match="does not preserve"):
        OrientationKernel(
            np.diag([1.0, 2.0, 3.0]),
            np.eye(3),
            point_group_operations("4/mmm"),
            G,
        )

    kernel = OrientationKernel(np.eye(3), np.eye(3), G, G)
    with pytest.raises(ValueError, match="proper rotation"):
        kernel.topology(np.diag([1.0, 1.0, -1.0]))

    with pytest.raises(ValueError, match="underdetermined"):
        kernel.parallelism_orientations(
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            reference_first_kind="direction",
            moving_first_kind="direction",
            reference_second_kind="direction",
            moving_second_kind="direction",
        )

    bad_P = sp.diag(2, 1, 1)
    with pytest.raises(ValueError, match="unimodular"):
        kernel.rebase(bad_P, sp.eye(3))


def test_opposite_basis_handedness_is_explicitly_not_silently_so3():
    kernel = OrientationKernel(
        np.eye(3), np.eye(3),
        point_group_operations("1"), point_group_operations("1"),
    )
    rebased, gauge = kernel.rebase(sp.diag(-1, 1, 1), sp.eye(3))
    assert gauge.relative_orientation_parity == -1
    with pytest.raises(ValueError, match="opposite handedness"):
        gauge.transform_orientation(np.eye(3))
    # The rebased metric/group state itself remains mathematically valid.
    assert rebased.topology(np.eye(3)).full_variant_count == 1
