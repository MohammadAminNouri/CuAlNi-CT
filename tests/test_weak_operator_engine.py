from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.lattice import Lattice
from cualni_cryst.point_groups import (
    point_group_definitions,
    point_group_operations,
)
from cualni_cryst.weak_operator_engine import (
    analyze_higher_order_element,
    analyze_non_twofold_groupoid,
    audit_parent_symmetry_element,
    audit_operator_route,
    exact_matrix_order,
    generalized_correspondence_geometry,
    generalized_distortion_geometry,
    generalized_twin_lattice_audit,
    enumerate_ranked_weak_planes,
)
from cualni_cryst.weak_twins import (
    BravaisNodeBasis,
    primitive_integer_plane,
)


BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks"

MG_INPUT_HASH = "cbfa9d3b3ee0286b7264aa0e8b1427442c7721e5fcf674c80a6ba03b360441e5"
MG_EXPECTED_HASH = "ce58a2fa5a0e476f133f59af76f058122e68af45e071cf601659723a9b0b4ec6"
NITI_INPUT_HASH = "24a2c2f3bd25b4fdf4538567be5874560da073302e5ee769d4c6957467552fe8"
NITI_EXPECTED_HASH = "32af32a620de1768a4f34c0a4eca114036fac589c95cc7c3b1c5df375ae50714"


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expr(value: object) -> sp.Expr:
    if isinstance(value, (int, float)):
        return sp.Rational(str(value))
    return sp.sympify(str(value))


def _matrix(rows) -> sp.Matrix:
    return sp.Matrix([[_expr(item) for item in row] for row in rows])


def _projective_tuple(values) -> tuple[int, int, int]:
    # Exact projective integer reduction for rational crystal indices.
    # Never cast components with int(): Rational(1,2) would become 0.
    return primitive_integer_plane(values)


def test_projective_test_normalizer_preserves_exact_rational_indices():
    assert _projective_tuple(
        sp.Matrix([0, sp.Rational(1, 2), 0])
    ) == (0, 1, 0)
    assert _projective_tuple(
        sp.Matrix([-sp.Rational(1, 2), sp.Rational(1, 4), 0])
    ) == (2, -1, 0)
    assert _projective_tuple(
        sp.Matrix([sp.Rational(3, 5), -sp.Rational(6, 5), sp.Rational(9, 5)])
    ) == (1, -2, 3)


def _hex_metric(a: float, c: float) -> np.ndarray:
    return np.array(
        [
            [a * a, -0.5 * a * a, 0.0],
            [-0.5 * a * a, a * a, 0.0],
            [0.0, 0.0, c * c],
        ],
        dtype=float,
    )


def _lattice(spec: dict) -> Lattice:
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
    raise AssertionError(f"unsupported benchmark lattice {kind!r}")


def _load(name: str) -> dict:
    return json.loads((BENCHMARK_DIR / name).read_text())


def _find_pair(candidates, p1, p2):
    target = (_projective_tuple(p1), _projective_tuple(p2))
    matches = [
        item
        for item in candidates
        if (
            _projective_tuple(item.result.plane1_primitive),
            _projective_tuple(item.result.plane2_primitive),
        )
        == target
    ]
    assert matches, f"blind search did not recover {target}"
    return matches[0]


def test_published_weak_benchmark_manifests_are_immutable():
    mg = _load("cayron_2022_mg_weak_twins_v1.json")
    niti = _load("cayron_2022_niti_weak_operator_v1.json")

    assert _canonical_hash(mg["input"]) == MG_INPUT_HASH
    assert _canonical_hash(mg["expected"]) == MG_EXPECTED_HASH
    assert mg["lock"]["input_sha256"] == MG_INPUT_HASH
    assert mg["lock"]["expected_sha256"] == MG_EXPECTED_HASH

    assert _canonical_hash(niti["input"]) == NITI_INPUT_HASH
    assert _canonical_hash(niti["expected"]) == NITI_EXPECTED_HASH
    assert niti["lock"]["input_sha256"] == NITI_INPUT_HASH
    assert niti["lock"]["expected_sha256"] == NITI_EXPECTED_HASH


def test_generalized_twin_index_audit_distinguishes_equal_and_unequal_volume():
    c1 = sp.Matrix(
        [[1, -sp.Rational(1, 2), 1],
         [0, 0, 2],
         [0, sp.Rational(1, 2), 0]]
    )
    c2 = sp.Matrix(
        [[1, -sp.Rational(1, 4), sp.Rational(3, 4)],
         [0, sp.Rational(1, 2), sp.Rational(3, 2)],
         [0, -sp.Rational(1, 2), sp.Rational(1, 2)]]
    )
    c3 = sp.Matrix(
        [[sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(3, 2)],
         [0, 1, 0],
         [-sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(1, 2)]]
    )

    for C, expected in ((c1, 2), (c2, 4), (c3, 4)):
        audit = generalized_twin_lattice_audit(C)
        assert audit.equal_volume
        assert audit.q_g == expected
        assert audit.domain.index == expected
        assert audit.codomain.index == expected

    unequal = generalized_twin_lattice_audit(sp.diag(2, 1, 1))
    assert not unequal.equal_volume
    assert unequal.q_g is None
    assert unequal.domain.index != unequal.codomain.index


def test_metric_geometry_separates_weak_distortion_from_rank_one_simple_shear():
    metric = _hex_metric(3.21, 5.21)
    C = sp.Matrix(
        [[1, -sp.Rational(1, 2), 1],
         [0, 0, 2],
         [0, sp.Rational(1, 2), 0]]
    )
    geometry = generalized_correspondence_geometry(metric, C)
    assert geometry.generalized_strain == pytest.approx(
        0.1300860007, abs=2.0e-9
    )
    assert geometry.metric_trace_residual < 2.0e-12
    assert geometry.equal_volume_residual < 2.0e-12

    # A conventional simple shear must be recognized as rank-one in an
    # orthonormal metric, independently of the weak-twin examples.
    F = np.eye(3)
    F[0, 1] = 0.2
    distortion = generalized_distortion_geometry(np.eye(3), F)
    assert distortion.generalized_shear == pytest.approx(0.2, abs=1.0e-12)
    assert distortion.numerical_rank == 1
    assert distortion.best_rank_one_relative_residual < 1.0e-12
    assert distortion.simple_shear_compatible


def test_cayron_2022_mg_cases_are_recovered_blind_from_C_axis_and_metric_only():
    manifest = _load("cayron_2022_mg_weak_twins_v1.json")
    lattice = manifest["input"]["lattice"]
    metric = _hex_metric(lattice["a_angstrom"], lattice["c_angstrom"])
    expected_by_id = {
        item["id"]: item for item in manifest["expected"]["cases"]
    }

    for case in manifest["input"]["cases"]:
        # IMPORTANT: no expected plane, q_g, s_g or epsilon is passed here.
        candidates = enumerate_ranked_weak_planes(
            metric,
            _matrix(case["C"]),
            case["axis"],
            node_basis=BravaisNodeBasis.primitive_conventional(),
            max_plane_index=case["max_plane_index"],
            maximum_generalized_shear=0.2,
        )
        expected = expected_by_id[case["id"]]
        match = _find_pair(
            candidates, expected["plane1"], expected["plane2"]
        )

        assert match.result.generalized_twin_index == expected["q_g"]
        assert match.result.selected.generalized_shear == pytest.approx(
            expected["s_g"], abs=expected["s_g_abs_tol"]
        )

        if "epsilon_g" in expected:
            assert match.result.generalized_strain == pytest.approx(
                expected["epsilon_g"],
                abs=expected["epsilon_abs_tol"],
            )
        else:
            # The paper text reports epsilon_g=0.137 for this sister while
            # printing the same C as the epsilon_g=0.130 extension family.
            # Eq. (6)/(7) makes epsilon a function of C and the metric, so the
            # code preserves the equation-consistent value and documents the
            # source inconsistency instead of forcing either number.
            assert expected["epsilon_status"] == (
                "source_internal_discrepancy_not_used_as_pass_fail_target"
            )
            assert match.result.generalized_strain == pytest.approx(
                expected["epsilon_g_equation_from_printed_C"],
                abs=2.0e-9,
            )
            assert abs(
                match.result.generalized_strain
                - expected["epsilon_g_source_reported"]
            ) > 0.005


def _niti_state():
    manifest = _load("cayron_2022_niti_ct_v1.json")
    inp = manifest["input"]
    A = _lattice(inp["parent"]["lattice"])
    M = _lattice(inp["product"]["lattice"])
    return (
        A.metric(),
        M.metric(),
        point_group_operations(inp["parent"]["point_group"]),
        point_group_operations(inp["product"]["point_group"]),
        Correspondence(_matrix(inp["package_C_M_from_A"])),
    )


def test_niti_higher_order_operator_is_discovered_without_operator_or_plane_hardcoding():
    weak_manifest = _load("cayron_2022_niti_weak_operator_v1.json")
    expected = weak_manifest["expected"]["weak_family"]
    M_A, M_M, G_A, G_M, C = _niti_state()

    report = analyze_non_twofold_groupoid(
        G_A,
        G_M,
        M_A,
        M_M,
        C,
        product_node_basis=BravaisNodeBasis.primitive_conventional(),
        max_plane_index=weak_manifest["input"]["protocol"]["weak_plane_search"][
            "max_plane_index"
        ],
        maximum_generalized_shear=0.35,
        maximum_intraplanar_distortion=0.20,
        eligible_orders=(3, 4, 6),
    )

    assert report.groupoid.n_variants == 12
    assert report.groupoid.n_operators == 7
    assert report.weak_operators

    # A weak operator is allowed only when the complete double coset contains
    # no exact Type-I/II route.
    for operator in report.weak_operators:
        assert not operator.operator_audit.classical_element_indices
        assert operator.operator_audit.weak_element_indices

    target_pair = (
        _projective_tuple(expected["plane1_B19prime"]),
        _projective_tuple(expected["plane2_B19prime"]),
    )
    matches = []
    for operator in report.weak_operators:
        for element in operator.weak_elements:
            for rank, candidate in enumerate(element.ranked_weak_planes):
                pair = (
                    _projective_tuple(candidate.result.plane1_primitive),
                    _projective_tuple(candidate.result.plane2_primitive),
                )
                if pair == target_pair:
                    matches.append((operator, element, rank, candidate))

    assert matches, "NiTi blind higher-order search missed Cayron's weak pair"

    # Choose the source-consistent branch by axis/order, not an operator index.
    matches = [
        item
        for item in matches
        if item[1].element_audit.order == expected["parent_rotation_order"]
        and _projective_tuple(item[1].parent_axis)
        == _projective_tuple(expected["parent_axis_B2"])
        and _projective_tuple(item[1].product_axis)
        == _projective_tuple(expected["product_axis_B19prime"])
    ]
    assert matches
    operator, element, rank, candidate = min(
        matches,
        key=lambda item: (
            item[3].geometry.intraplanar_distortion,
            item[2],
        ),
    )

    assert rank == 0, (
        "for the source-consistent fourfold element, Cayron's (1-33)||(31-1) "
        "pair should minimize the metric-native intrinsic in-plane distortion "
        "within the declared low-index search"
    )
    assert element.lattice_audit.q_g == expected["q_g"]
    assert candidate.result.generalized_twin_index == expected["q_g"]
    assert candidate.result.selected.generalized_shear == pytest.approx(
        expected["s_g"], abs=expected["s_g_abs_tol"]
    )
    assert candidate.result.generalized_strain == pytest.approx(
        expected["epsilon_g"], abs=expected["epsilon_abs_tol"]
    )
    assert candidate.geometry.proper_reticular_angle_deg == pytest.approx(
        expected["misorientation_deg"],
        abs=expected["misorientation_abs_tol_deg"],
    )
    assert candidate.geometry.intraplanar_distortion < 0.03

    all_pairs = {
        (
            _projective_tuple(item.result.plane1_primitive),
            _projective_tuple(item.result.plane2_primitive),
        )
        for item in element.ranked_weak_planes
    }
    for alternative in weak_manifest["expected"]["alternative_looser_pairs"]:
        assert (
            _projective_tuple(alternative["plane1"]),
            _projective_tuple(alternative["plane2"]),
        ) in all_pairs


def _invariant_metric(group, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(3, 3))
    base = X.T @ X + np.diag([1.0, 1.7, 2.3])
    total = np.zeros((3, 3), dtype=float)
    for operation in group:
        G = np.asarray(operation, dtype=float)
        total += G.T @ base @ G
    total /= len(group)
    return 0.5 * (total + total.T)


def test_every_element_of_all_32_point_groups_has_a_consistent_exact_route():
    definitions = point_group_definitions()
    assert len(definitions) == 32

    for index, definition in enumerate(definitions):
        group = point_group_operations(definition.symbol)
        metric = _invariant_metric(group, 7000 + index)
        for operation in group:
            audit = audit_parent_symmetry_element(operation, metric)
            assert audit.order in {1, 2, 3, 4, 6}
            if audit.order in {3, 4, 6} and sp.det(operation) == 1:
                assert audit.route == "axial_weak_rotation"
                assert audit.axis_parent is not None
            if audit.route == "improper_higher_order":
                assert sp.det(operation) == -1
                assert audit.order > 2


def test_operator_level_guard_prevents_false_weak_classification():
    M_A, M_M, G_A, G_M, C = _niti_state()
    report = analyze_non_twofold_groupoid(
        G_A,
        G_M,
        M_A,
        M_M,
        C,
        product_node_basis=BravaisNodeBasis.primitive_conventional(),
        max_plane_index=2,
        maximum_generalized_shear=0.35,
        maximum_intraplanar_distortion=0.25,
        eligible_orders=(4,),
        max_weak_planes_per_element=3,
    )

    for item in report.operators:
        audit = item.operator_audit
        if audit.classical_element_indices:
            assert audit.route in {"identity", "classical_exact"}
            assert not item.weak_elements
        if audit.route == "axial_weak":
            assert not audit.classical_element_indices


def test_weak_geometry_is_covariant_under_exact_unimodular_basis_change():
    metric = _hex_metric(3.21, 5.21)
    C = sp.Matrix(
        [[1, -sp.Rational(1, 2), 1],
         [0, 0, 2],
         [0, sp.Rational(1, 2), 0]]
    )
    u = sp.Matrix([1, 0, 0])
    p1 = sp.Matrix([0, 0, 1])
    p2 = sp.simplify(C.inv().T * p1)

    base = enumerate_ranked_weak_planes(
        metric,
        C,
        u,
        node_basis=BravaisNodeBasis.primitive_conventional(),
        max_plane_index=4,
        maximum_generalized_shear=0.2,
    )
    reference = _find_pair(base, [0, 0, 1], [0, 1, 0])

    # Exact SL(3,Z) coordinate change; the lattice itself is unchanged.
    P = sp.Matrix([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    Pf = np.asarray(P, dtype=float)
    metric_prime = Pf.T @ metric @ Pf
    C_prime = sp.simplify(P.inv() * C * P)
    u_prime = sp.simplify(P.inv() * u)
    p1_prime = sp.simplify(P.T * p1)
    p2_prime = sp.simplify(P.T * p2)

    transformed = enumerate_ranked_weak_planes(
        metric_prime,
        C_prime,
        u_prime,
        node_basis=BravaisNodeBasis.primitive_conventional(),
        max_plane_index=5,
        maximum_generalized_shear=0.2,
    )
    target1 = _projective_tuple(p1_prime)
    target2 = _projective_tuple(p2_prime)
    candidate = _find_pair(transformed, target1, target2)

    assert candidate.result.generalized_twin_index == (
        reference.result.generalized_twin_index
    )
    assert candidate.result.generalized_strain == pytest.approx(
        reference.result.generalized_strain, abs=3.0e-10
    )
    assert candidate.result.selected.generalized_shear == pytest.approx(
        reference.result.selected.generalized_shear, abs=3.0e-10
    )
    assert candidate.geometry.intraplanar_distortion == pytest.approx(
        reference.geometry.intraplanar_distortion, abs=3.0e-10
    )


def test_invalid_routes_and_inputs_fail_loudly():
    metric = np.eye(3)

    # A proper 90-degree element is a valid axial weak candidate.
    R4 = sp.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    audit = audit_parent_symmetry_element(R4, metric)
    assert audit.route == "axial_weak_rotation"
    assert audit.order == 4

    # Its improper partner is *not* silently promoted to the weak axial route.
    improper = -R4
    improper_audit = audit_parent_symmetry_element(improper, metric)
    assert improper_audit.route == "improper_higher_order"

    with pytest.raises(ValueError, match="axial weak route requires"):
        analyze_higher_order_element(
            improper,
            metric,
            metric,
            Correspondence(sp.eye(3)),
            product_node_basis=BravaisNodeBasis.primitive_conventional(),
            max_plane_index=2,
        )

    with pytest.raises(ValueError, match="positive definite"):
        generalized_correspondence_geometry(
            np.diag([1.0, 1.0, -1.0]), sp.eye(3)
        )

    with pytest.raises(ValueError, match="invertible"):
        generalized_twin_lattice_audit(sp.diag(1, 1, 0))

    # A non-crystallographic finite 5-fold SO(3) rotation is classified as
    # unsupported rather than being mislabeled as an axial weak twin.
    phi = 2 * sp.pi / 5
    R5 = sp.Matrix(
        [
            [sp.cos(phi), -sp.sin(phi), 0],
            [sp.sin(phi), sp.cos(phi), 0],
            [0, 0, 1],
        ]
    )
    fivefold = audit_parent_symmetry_element(R5, metric)
    assert exact_matrix_order(R5, maximum_order=10) == 5
    assert fivefold.route == "unsupported_finite_order"
