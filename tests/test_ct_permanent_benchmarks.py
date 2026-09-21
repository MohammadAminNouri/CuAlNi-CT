from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.ct import analyze_cmc, habit_planes_from_cmc
from cualni_cryst.group_theory import double_cosets, left_cosets, correspondence_groupoid
from cualni_cryst.lattice import Lattice, direction_cartesian, plane_normal_cartesian
from cualni_cryst.point_groups import point_group_operations
from cualni_cryst.twinning_ct import twins_from_operator


BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks"

# Hard-coded scientific locks. Editing both a benchmark and its embedded lock
# is not enough: changing one of these permanent v1 states requires a new ID.
KNOWN_LOCKS = {
    "chen_2000_blind_2h_v1": (
        "db67712dfefe24298cde96d67d19b3a3b87f88a33e8a231671354337bdcdacc0",
        "fe227369b426c10d4513498c59398e097f33d682accf949e9661c828d2fa18aa",
    ),
    "cayron_2006_burgers_groupoid_v1": (
        "5687e0b8b2312041de77684d6aaf51d542850b4affd0f71c71c879f41e70766d",
        "8197c65c59663eb1d18bc7418aed2e2bc88d129ce3811e2ad14af0f23983ef96",
    ),
    "cayron_2019_bain_correspondence_v1": (
        "7fa6ee0006b07ae90bfb58161d9075ee353cc5d94a7a679c6dbafc637d6055cc",
        "715d00516ef1fad7c5410980db6162eee99c23c7a17eda95b460879fdc362fb9",
    ),
    "cayron_2022_niti_ct_v1": (
        "13d7ec3079bab184e37aa693a7a1e08760f3eb28f60959042fc3f0e2e5f32033",
        "c00a17ab22a8ed84c3bb880ad393817d36bf34c7bf1add37ad1805c9512ad53e",
    ),
    "cayron_2026_niti_c1_compatibility_v1": (
        "999995c7caed8f5b64b71d754a0a6ba54f9af15842743da76787d75de63b2ca3",
        "82ded2c00703b76e20babb5db287f0727f78a3ad05f3e38d1828abc9d7a63acb",
    ),
}


def _load(filename: str) -> dict:
    return json.loads((BENCHMARK_DIR / filename).read_text())


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expr(value: object) -> sp.Expr:
    if isinstance(value, (int, float)):
        return sp.Rational(str(value))
    return sp.sympify(str(value), locals={"sqrt": sp.sqrt})


def _matrix(rows: list[list[object]]) -> sp.Matrix:
    return sp.Matrix([[_expr(value) for value in row] for row in rows])


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-15:
        raise ValueError("zero vector")
    return vector / norm


def _projective_residual(lhs: object, rhs: object) -> float:
    a = _unit(np.asarray(lhs, dtype=float))
    b = _unit(np.asarray(rhs, dtype=float))
    return float(np.linalg.norm(np.cross(a, b)))


def _projectively_parallel(lhs: object, rhs: object, tol: float = 1.0e-8) -> bool:
    return _projective_residual(lhs, rhs) <= tol


def _all_twins(
    parent_group: tuple[sp.Matrix, ...],
    product_group: tuple[sp.Matrix, ...],
    M_a: np.ndarray,
    M_m: np.ndarray,
    correspondence: Correspondence,
):
    groupoid = correspondence_groupoid(
        list(parent_group), list(product_group), correspondence
    )
    twins = [
        twin
        for operator in groupoid.operators
        for twin in twins_from_operator(operator, M_a, M_m, correspondence)
    ]
    return groupoid, twins


def test_permanent_benchmark_hash_locks():
    files = {
        item["benchmark_id"]: item
        for item in (
            _load("chen_2000_blind_2h_v1.json"),
            _load("cayron_2006_burgers_groupoid_v1.json"),
            _load("cayron_2019_bain_correspondence_v1.json"),
            _load("cayron_2022_niti_ct_v1.json"),
            _load("cayron_2026_niti_c1_compatibility_v1.json"),
        )
    }
    assert set(files) == set(KNOWN_LOCKS)

    for benchmark_id, manifest in files.items():
        expected_input_hash, expected_output_hash = KNOWN_LOCKS[benchmark_id]
        input_hash = _canonical_hash(manifest["input"])
        output_hash = _canonical_hash(manifest["expected"])
        assert input_hash == expected_input_hash
        assert output_hash == expected_output_hash
        assert manifest["lock"]["input_sha256"] == expected_input_hash
        assert manifest["lock"]["expected_sha256"] == expected_output_hash
        assert manifest["schema_version"] == 1


def test_chen_2000_original_blind_state_is_permanently_locked():
    manifest = _load("chen_2000_blind_2h_v1.json")
    inp = manifest["input"]  # calculation sees only this block
    expected = manifest["expected"]  # comparison happens after calculation

    A = Lattice.cubic(inp["parent"]["lattice"]["a_angstrom"], length_unit="angstrom")
    pm = inp["product"]["lattice"]
    M = Lattice.orthorhombic(
        pm["a_angstrom"], pm["b_angstrom"], pm["c_angstrom"], length_unit="angstrom"
    )
    correspondence = Correspondence(_matrix(inp["package_C_M_from_A"]))

    groupoid, twins = _all_twins(
        point_group_operations(inp["parent"]["point_group"]),
        point_group_operations(inp["product"]["point_group"]),
        A.metric(),
        M.metric(),
        correspondence,
    )

    assert len(groupoid.subgroup) == expected["correspondence_subgroup_order"]
    assert groupoid.n_variants == expected["variant_count"]
    assert groupoid.n_operators == expected["operator_count"]

    compound = expected["compound_family"]
    family_101 = [
        twin
        for twin in twins
        if _projectively_parallel(twin.plane_m, compound["plane_M"])
        and _projectively_parallel(twin.direction_m, compound["direction_M"])
    ]
    assert family_101
    assert {twin.kind for twin in family_101} == set(compound["representations"])
    assert all(twin.compound for twin in family_101)
    assert all(
        twin.representations == tuple(compound["representations"])
        for twin in family_101
    )

    type_i = expected["type_I_family"]
    family_121 = [
        twin
        for twin in twins
        if twin.kind == "I"
        and not twin.compound
        and _projectively_parallel(twin.plane_m, type_i["plane_M"])
    ]
    assert family_121
    assert any(
        math.isclose(
            twin.shear, type_i["shear"], rel_tol=0.0, abs_tol=5.0e-12
        )
        for twin in family_121
    )

    conjugate = expected["type_II_conjugate"]
    type_ii_candidates = [
        twin
        for twin in twins
        if twin.kind == "II"
        and not twin.compound
        and _projectively_parallel(
            twin.plane_m,
            conjugate["plane_M_ratio"],
            tol=conjugate["ratio_abs_tol"],
        )
        and _projectively_parallel(twin.direction_m, conjugate["direction_M"])
    ]
    assert type_ii_candidates
    assert any(
        math.isclose(
            twin.shear,
            conjugate["shear"],
            rel_tol=0.0,
            abs_tol=5.0e-12,
        )
        for twin in type_ii_candidates
    )


def _rotation_from_direction_and_plane(
    product_direction: np.ndarray,
    product_plane_normal: np.ndarray,
    parent_direction: np.ndarray,
    parent_plane_normal: np.ndarray,
) -> np.ndarray:
    """Return proper R_A_from_M from two orthogonal parallelisms."""

    dm = _unit(product_direction)
    nm = _unit(product_plane_normal)
    da = _unit(parent_direction)
    na = _unit(parent_plane_normal)

    assert abs(float(dm @ nm)) < 1.0e-10
    assert abs(float(da @ na)) < 1.0e-10

    em2 = _unit(np.cross(nm, dm))
    ea2 = _unit(np.cross(na, da))
    frame_m = np.column_stack([dm, em2, nm])
    frame_a = np.column_stack([da, ea2, na])
    rotation = frame_a @ frame_m.T

    assert np.linalg.norm(rotation.T @ rotation - np.eye(3)) < 1.0e-10
    assert abs(float(np.linalg.det(rotation)) - 1.0) < 1.0e-10
    return rotation


def test_cayron_2006_burgers_groupoid_blind_from_or_parallelisms():
    manifest = _load("cayron_2006_burgers_groupoid_v1.json")
    inp = manifest["input"]
    expected = manifest["expected"]

    parent_lattice = Lattice.cubic(1.0)
    product_lattice = Lattice(
        1.0, 1.0, 1.6, 90.0, 90.0, 120.0
    )
    parent_group = point_group_operations(inp["parent"]["point_group"])
    product_group = point_group_operations(inp["product"]["point_group"])

    burgers = inp["burgers_or"]
    d_m = direction_cartesian(
        np.asarray(burgers["direction_product"], dtype=float), product_lattice
    )
    n_m = plane_normal_cartesian(
        np.asarray(burgers["plane_product"], dtype=float), product_lattice
    )
    d_a = direction_cartesian(
        np.asarray(burgers["direction_parent"], dtype=float), parent_lattice
    )
    n_a = plane_normal_cartesian(
        np.asarray(burgers["plane_parent"], dtype=float), parent_lattice
    )
    R = _rotation_from_direction_and_plane(d_m, n_m, d_a, n_a)

    B_a = parent_lattice.structure_matrix()
    B_m = product_lattice.structure_matrix()
    parent_cart = [
        B_a @ np.asarray(g, dtype=float) @ np.linalg.inv(B_a)
        for g in parent_group
    ]
    product_cart = [
        B_m @ np.asarray(g, dtype=float) @ np.linalg.inv(B_m)
        for g in product_group
    ]

    H: list[sp.Matrix] = []
    for g_exact, g_cart in zip(parent_group, parent_cart, strict=True):
        if min(
            np.linalg.norm(g_cart - R @ h @ R.T, ord="fro")
            for h in product_cart
        ) < 1.0e-9:
            H.append(g_exact)

    variants = left_cosets(list(parent_group), H)
    operators = double_cosets(list(parent_group), H)

    assert len(H) == expected["intersection_order"]
    assert len(variants) == expected["variant_count"]
    assert len(operators) == expected["operator_count"]


def test_cayron_2019_bain_correspondence_variants_blind():
    manifest = _load("cayron_2019_bain_correspondence_v1.json")
    inp = manifest["input"]
    expected = manifest["expected"]

    correspondence = Correspondence(_matrix(inp["package_C_M_from_A"]))
    groupoid = correspondence_groupoid(
        list(point_group_operations(inp["parent"]["point_group"])),
        list(point_group_operations(inp["product"]["point_group"])),
        correspondence,
    )

    assert groupoid.n_variants == expected["correspondence_variant_count"]
    # Internal algebraic cross-checks, not literature-fit targets.
    assert len(groupoid.subgroup) * groupoid.n_variants == 48
    assert groupoid.burnside_count == groupoid.n_operators


def test_cayron_2022_niti_topology_and_twin_families_blind():
    manifest = _load("cayron_2022_niti_ct_v1.json")
    inp = manifest["input"]
    expected = manifest["expected"]

    A = Lattice.cubic(inp["parent"]["lattice"]["a_angstrom"], length_unit="angstrom")
    p = inp["product"]["lattice"]
    M = Lattice.monoclinic_unique_b(
        p["a_angstrom"],
        p["b_angstrom"],
        p["c_angstrom"],
        p["beta_deg"],
        length_unit="angstrom",
    )
    correspondence = Correspondence(_matrix(inp["package_C_M_from_A"]))

    groupoid, twins = _all_twins(
        point_group_operations(inp["parent"]["point_group"]),
        point_group_operations(inp["product"]["point_group"]),
        A.metric(),
        M.metric(),
        correspondence,
    )

    assert len(groupoid.subgroup) == expected["intersection_order"]
    assert groupoid.n_variants == expected["variant_count"]
    assert groupoid.n_operators == expected["operator_count"]

    # Match crystallographic geometry, never arbitrary operator numbering.
    for family in expected["twin_families"]:
        route = family.get("route")
        classification = family.get("kind")
        plane_tol = family.get("plane_projective_tol", 1.0e-8)

        candidates = [
            twin
            for twin in twins
            if (route is None or twin.kind == route)
            and (classification is None or twin.classification == classification)
            and _projectively_parallel(
                twin.plane_m, family["plane_M"], tol=plane_tol
            )
            and (
                "direction_M" not in family
                or _projectively_parallel(
                    twin.direction_m, family["direction_M"], tol=1.0e-8
                )
            )
        ]
        assert candidates, family["label"]
        assert any(
            math.isclose(
                twin.shear,
                family["shear"],
                rel_tol=0.0,
                abs_tol=family["shear_abs_tol"],
            )
            for twin in candidates
        ), family["label"]

    # The natural OR is a separate hypothesis in Cayron's paper. This benchmark
    # must never pretend that it was predicted blindly from C + metrics alone.
    assert expected["natural_or"]["status"] == "not_blind_from_correspondence_alone"


def test_cayron_2026_c1_cmc_and_compound_twin_blind():
    manifest = _load("cayron_2026_niti_c1_compatibility_v1.json")
    inp = manifest["input"]
    expected = manifest["expected"]

    A = Lattice.cubic(float(inp["parent"]["lattice"]["a"]))
    p = inp["product"]["lattice"]
    M = Lattice.monoclinic_unique_b(
        float(p["a_over_a0"]),
        float(sp.N(_expr(p["b_over_a0"]), 16)),
        float(p["c_over_a0"]),
        float(p["beta_deg"]),
    )
    correspondence = Correspondence(_matrix(inp["package_C_M_from_A"]))

    analysis = analyze_cmc(A.metric(), M.metric(), correspondence, tol=1.0e-8)
    cmc_expected = expected["cmc"]
    assert analysis.exact_compatible is cmc_expected["exact_compatible"]
    assert analysis.degeneracy_order == cmc_expected["degeneracy_order"]
    assert analysis.inertia == tuple(cmc_expected["inertia"])
    assert len(
        habit_planes_from_cmc(A.metric(), M.metric(), correspondence, tol=1.0e-8)
    ) == cmc_expected["habit_plane_count"]

    _, twins = _all_twins(
        point_group_operations(inp["parent"]["point_group"]),
        point_group_operations(inp["product"]["point_group"]),
        A.metric(),
        M.metric(),
        correspondence,
    )
    family = expected["compound_family"]
    candidates = [
        twin
        for twin in twins
        if twin.compound
        and _projectively_parallel(twin.plane_m, family["plane_M"])
        and _projectively_parallel(twin.direction_m, family["direction_M"])
    ]
    assert candidates
    assert any(
        math.isclose(
            twin.shear,
            family["shear"],
            rel_tol=0.0,
            abs_tol=family["shear_abs_tol"],
        )
        for twin in candidates
    )
