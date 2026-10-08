from __future__ import annotations

"""Readable, metric-validated point-group inventories for the standalone app.

The module does not decide whether a phase actually forms a twin.  It only
shows the exact operations contained in the selected crystallographic point
group and labels the mathematical route each operation could support if the
phase is used as the parent in the twin-family calculation.
"""

from collections import Counter
from dataclasses import dataclass
from math import gcd
from functools import reduce
from typing import Any

import numpy as np
import sympy as sp

from cualni_cryst.point_groups import point_group_operations, resolve_point_group
from cualni_cryst.weak_operator_engine import audit_parent_symmetry_element


Matrix3Text = tuple[
    tuple[str, str, str],
    tuple[str, str, str],
    tuple[str, str, str],
]


@dataclass(frozen=True)
class SymmetryOperationRecord:
    index: int
    order: int
    determinant: int
    kind: str
    twin_role: str
    axis_or_plane: tuple[int, int, int] | None
    matrix: Matrix3Text


@dataclass(frozen=True)
class SymmetryInventory:
    symbol: str
    crystal_family: str
    conventional_setting: str
    expected_order: int
    counts: tuple[tuple[str, int], ...]
    operations: tuple[SymmetryOperationRecord, ...]
    metric_preservation_maximum_residual: float


def _matrix_text(matrix: sp.Matrix) -> Matrix3Text:
    M = sp.Matrix(matrix)
    return tuple(
        tuple(str(sp.simplify(M[i, j])) for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


def _primitive_integer_vector(vector: Any) -> tuple[int, int, int] | None:
    v = sp.Matrix(vector)
    if v.shape not in {(3, 1), (1, 3)}:
        v = v.reshape(3, 1)
    values = [sp.Rational(sp.simplify(x)) for x in list(v)]
    if all(value == 0 for value in values):
        return None
    denominator = 1
    for value in values:
        denominator = int(sp.ilcm(denominator, int(value.q)))
    integers = [int(value * denominator) for value in values]
    common = reduce(gcd, (abs(value) for value in integers if value), 0) or 1
    integers = [value // common for value in integers]
    first = next((value for value in integers if value), 1)
    if first < 0:
        integers = [-value for value in integers]
    return tuple(integers)  # type: ignore[return-value]


def _axis_or_plane(operation: sp.Matrix, route: str) -> tuple[int, int, int] | None:
    G = sp.Matrix(operation)
    if route in {"type_II_twofold", "axial_weak_rotation"}:
        nullspace = (G - sp.eye(3)).nullspace()
        return _primitive_integer_vector(nullspace[0]) if len(nullspace) == 1 else None
    if route == "type_I_reflection":
        # Plane covectors transform with G^{-T}; a reflection-plane covector is
        # the -1 eigencovector of G^T.
        nullspace = (G.T + sp.eye(3)).nullspace()
        return _primitive_integer_vector(nullspace[0]) if len(nullspace) == 1 else None
    return None


def _display_kind(route: str, order: int) -> tuple[str, str]:
    if route == "identity":
        return "identity", "self relation only"
    if route == "inversion":
        return "inversion", "not a twin generator by itself"
    if route == "type_I_reflection":
        return "mirror reflection", "candidate Type-I generator when this phase is parent"
    if route == "type_II_twofold":
        return "proper 2-fold rotation", "candidate Type-II generator when this phase is parent"
    if route == "axial_weak_rotation":
        return (
            f"proper {order}-fold rotation",
            "higher-order candidate; weak status is decided only after complete operator-class audit",
        )
    if route == "improper_higher_order":
        return f"improper order-{order} operation", "not promoted to an axial weak route"
    return f"finite order-{order} operation", "unsupported by the present twin-route classifier"


def build_symmetry_inventory(
    point_group: str,
    metric: np.ndarray,
    *,
    tolerance: float = 1.0e-9,
) -> SymmetryInventory:
    definition = resolve_point_group(point_group)
    operations = point_group_operations(point_group)
    records: list[SymmetryOperationRecord] = []
    counter: Counter[str] = Counter()
    worst = 0.0

    for index, operation in enumerate(operations):
        audit = audit_parent_symmetry_element(
            operation,
            metric,
            tolerance=tolerance,
        )
        kind, role = _display_kind(audit.route, audit.order)
        counter[kind] += 1
        worst = max(worst, float(audit.metric_preservation_residual))
        records.append(
            SymmetryOperationRecord(
                index=index,
                order=int(audit.order),
                determinant=int(sp.simplify(operation.det())),
                kind=kind,
                twin_role=role,
                axis_or_plane=_axis_or_plane(operation, audit.route),
                matrix=_matrix_text(operation),
            )
        )

    if len(records) != int(definition.expected_order):
        raise AssertionError(
            f"{point_group}: expected {definition.expected_order} symmetry operations, "
            f"received {len(records)}"
        )

    return SymmetryInventory(
        symbol=definition.symbol,
        crystal_family=definition.crystal_family,
        conventional_setting=definition.conventional_setting,
        expected_order=int(definition.expected_order),
        counts=tuple(sorted(counter.items())),
        operations=tuple(records),
        metric_preservation_maximum_residual=float(worst),
    )
