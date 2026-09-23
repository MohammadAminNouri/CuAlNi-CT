from __future__ import annotations

"""Exact algebraic property tests for CT operator groupoids.

Existing tests lock important fixed cases.  This file attacks a different
question: does the set-valued double-coset operator algebra obey its defining
algebraic laws across the crystallographic point-group registry?

The oracle used here is an independent direct product of double-coset sets,
not the variant-adjacency implementation used by the production composition
routine.
"""

import sympy as sp

from cualni_cryst.group_theory import (
    double_cosets,
    inverse_operator_map,
    left_cosets,
    multivalued_operator_composition,
    operator_adjacency,
    validate_group,
    which_double_coset,
)
from cualni_cryst.point_groups import point_group_definitions
from cualni_cryst.symmetry import matrix_key


def _exact_identity(M: sp.Matrix) -> bool:
    return sp.simplify(sp.Matrix(M) - sp.eye(3)) == sp.zeros(3)


def _matrix_order(g: sp.Matrix, maximum: int = 24) -> int:
    power = sp.eye(3)
    for n in range(1, maximum + 1):
        power = sp.simplify(power * g)
        if _exact_identity(power):
            return n
    raise AssertionError(f"finite crystallographic order not found <= {maximum}")


def _cyclic_subgroup(g: sp.Matrix) -> list[sp.Matrix]:
    out = [sp.eye(3)]
    power = sp.eye(3)
    for _ in range(24):
        power = sp.simplify(power * g)
        if _exact_identity(power):
            break
        out.append(power)
    else:
        raise AssertionError("cyclic subgroup did not close")
    validate_group(out)
    return out


def _subgroup_key(H: list[sp.Matrix]) -> frozenset[tuple]:
    return frozenset(matrix_key(x) for x in H)


def _candidate_subgroups(G: list[sp.Matrix]) -> list[list[sp.Matrix]]:
    candidates: list[list[sp.Matrix]] = [[sp.eye(3)], list(G)]

    non_identity = [g for g in G if not _exact_identity(g)]
    if non_identity:
        by_order = sorted(
            ((_matrix_order(g), tuple(str(x) for x in matrix_key(g)), g) for g in non_identity),
            key=lambda item: (item[0], item[1]),
        )
        candidates.append(_cyclic_subgroup(by_order[-1][2]))
        order_two = [item for item in by_order if item[0] == 2]
        if order_two:
            candidates.append(_cyclic_subgroup(order_two[0][2]))

    unique: dict[frozenset[tuple], list[sp.Matrix]] = {}
    for H in candidates:
        validate_group(H)
        unique[_subgroup_key(H)] = H
    return list(unique.values())


def _independent_product_support(
    A: list[sp.Matrix],
    B: list[sp.Matrix],
    operators: list[list[sp.Matrix]],
) -> set[int]:
    support: set[int] = set()
    for a in A:
        for b in B:
            support.add(which_double_coset(sp.simplify(a * b), operators))
    return support


def _union_left(table, first: set[int], c: int) -> set[int]:
    out: set[int] = set()
    for x in first:
        out.update(table[x][c])
    return out


def _union_right(table, a: int, second: set[int]) -> set[int]:
    out: set[int] = set()
    for y in second:
        out.update(table[a][y])
    return out


def _audit_one_group(G: list[sp.Matrix]) -> None:
    validate_group(G)
    for H in _candidate_subgroups(G):
        variants = left_cosets(G, H)
        operators = double_cosets(G, H)
        adjacency = operator_adjacency(variants, operators)
        table = multivalued_operator_composition(adjacency)
        inverse = inverse_operator_map(operators)

        identity_operator = which_double_coset(sp.eye(3), operators)
        nops = len(operators)

        # Identity and inverse laws.
        for a in range(nops):
            assert table[identity_operator][a] == {a}
            assert table[a][identity_operator] == {a}
            assert identity_operator in table[a][inverse[a]]
            assert identity_operator in table[inverse[a]][a]
            assert inverse[inverse[a]] == a

        # Completely independent oracle: set multiplication of HgH classes.
        for a in range(nops):
            for b in range(nops):
                direct = _independent_product_support(
                    operators[a], operators[b], operators
                )
                assert table[a][b] == direct

        # Hypergroup associativity at support level.
        for a in range(nops):
            for b in range(nops):
                for c in range(nops):
                    left = _union_left(table, table[a][b], c)
                    right = _union_right(table, a, table[b][c])
                    assert left == right


def test_representative_crystal_families_obey_exact_double_coset_hypergroup_laws():
    wanted = {"2/m", "mmm", "4/mmm", "-3m", "6/mmm", "m-3m"}
    definitions = [d for d in point_group_definitions() if d.symbol in wanted]
    assert {d.symbol for d in definitions} == wanted
    for definition in definitions:
        _audit_one_group(list(definition.operations()))


def test_all_32_crystallographic_point_groups_obey_exact_double_coset_hypergroup_laws():
    definitions = point_group_definitions()
    assert len(definitions) == 32
    for definition in definitions:
        _audit_one_group(list(definition.operations()))
