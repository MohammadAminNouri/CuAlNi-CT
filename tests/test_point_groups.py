import sympy as sp

from cualni_cryst.lattice import Lattice
from cualni_cryst.point_groups import (
    metric_preservation_residual,
    point_group_definitions,
    point_group_operations,
)

EXPECTED_ORDERS = {
    "1": 1,
    "-1": 2,
    "2": 2,
    "m": 2,
    "2/m": 4,
    "222": 4,
    "mm2": 4,
    "mmm": 8,
    "4": 4,
    "-4": 4,
    "4/m": 8,
    "422": 8,
    "4mm": 8,
    "-42m": 8,
    "4/mmm": 16,
    "3": 3,
    "-3": 6,
    "32": 6,
    "3m": 6,
    "-3m": 12,
    "6": 6,
    "-6": 6,
    "6/m": 12,
    "622": 12,
    "6mm": 12,
    "-6m2": 12,
    "6/mmm": 24,
    "23": 12,
    "m-3": 24,
    "432": 24,
    "-43m": 24,
    "m-3m": 48,
}


def _key(matrix):
    return tuple(sp.simplify(value) for value in list(sp.Matrix(matrix)))


def _representative_lattice(family: str) -> Lattice:
    if family == "triclinic":
        return Lattice(3.1, 4.2, 5.3, 75.0, 83.0, 67.0, length_unit="angstrom")
    if family == "monoclinic":
        return Lattice.monoclinic_unique_b(3.1, 4.2, 5.3, 104.0, length_unit="angstrom")
    if family == "orthorhombic":
        return Lattice.orthorhombic(3.1, 4.2, 5.3, length_unit="angstrom")
    if family == "tetragonal":
        return Lattice(3.1, 3.1, 5.3, 90.0, 90.0, 90.0, length_unit="angstrom")
    if family in {"trigonal", "hexagonal"}:
        return Lattice(3.1, 3.1, 5.3, 90.0, 90.0, 120.0, length_unit="angstrom")
    if family == "cubic":
        return Lattice.cubic(3.1, length_unit="angstrom")
    raise AssertionError(family)


def test_registry_contains_all_32_crystallographic_point_groups():
    definitions = point_group_definitions()
    assert len(definitions) == 32
    assert {item.symbol for item in definitions} == set(EXPECTED_ORDERS)


def test_every_group_has_expected_order_closure_and_det_plus_minus_one():
    for symbol, expected_order in EXPECTED_ORDERS.items():
        operations = point_group_operations(symbol)
        assert len(operations) == expected_order
        keys = {_key(operation) for operation in operations}
        assert len(keys) == expected_order
        for operation in operations:
            assert int(sp.det(operation)) in {-1, 1}
        for left in operations:
            for right in operations:
                assert _key(left * right) in keys


def test_every_builtin_group_preserves_its_conventional_family_metric():
    for definition in point_group_definitions():
        lattice = _representative_lattice(definition.crystal_family)
        residual = metric_preservation_residual(
            definition.operations(),
            lattice.metric(),
        )
        assert residual < 1e-12, (definition.symbol, residual)
