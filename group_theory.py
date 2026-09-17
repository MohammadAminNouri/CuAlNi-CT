from __future__ import annotations

from dataclasses import dataclass
import sympy as sp
from .symmetry import matrix_key
from .correspondence import Correspondence


def _as_dict(group: list[sp.Matrix]) -> dict[tuple, sp.Matrix]:
    return {matrix_key(sp.Matrix(g)): sp.Matrix(g) for g in group}


def validate_group(group: list[sp.Matrix]) -> None:
    G = _as_dict(group)
    if matrix_key(sp.eye(3)) not in G:
        raise ValueError("Identity is missing.")
    for a in G.values():
        if matrix_key(a.inv()) not in G:
            raise ValueError("Inverse missing from group.")
        for b in G.values():
            if matrix_key(a * b) not in G:
                raise ValueError("Group is not closed.")


def correspondence_subgroup(
    parent_group: list[sp.Matrix],
    product_group: list[sp.Matrix],
    correspondence: Correspondence,
) -> list[sp.Matrix]:
    """H_C^A = G_A ∩ C_{A<-M} G_M C_{M<-A}.

    With our convention u_M = C_{M<-A} u_A, a product symmetry g_M becomes
    C^{-1} g_M C in parent coordinates.
    """
    Gp = _as_dict(parent_group)
    C = correspondence.C_m_from_a
    Ci = correspondence.C_a_from_m
    transformed = [sp.simplify(Ci * gm * C) for gm in product_group]
    H = [Gp[matrix_key(h)] for h in transformed if matrix_key(h) in Gp]
    return list(_as_dict(H).values())


def left_cosets(group: list[sp.Matrix], subgroup: list[sp.Matrix]) -> list[list[sp.Matrix]]:
    remaining = _as_dict(group)
    H = list(_as_dict(subgroup).values())
    cosets: list[list[sp.Matrix]] = []
    while remaining:
        g = next(iter(remaining.values()))
        coset = list(_as_dict([sp.simplify(g * h) for h in H]).values())
        cosets.append(coset)
        for x in coset:
            remaining.pop(matrix_key(x), None)
    return cosets


def double_cosets(group: list[sp.Matrix], subgroup: list[sp.Matrix]) -> list[list[sp.Matrix]]:
    remaining = _as_dict(group)
    H = list(_as_dict(subgroup).values())
    out: list[list[sp.Matrix]] = []
    while remaining:
        g = next(iter(remaining.values()))
        d = list(_as_dict([sp.simplify(h1 * g * h2) for h1 in H for h2 in H]).values())
        out.append(d)
        for x in d:
            remaining.pop(matrix_key(x), None)
    return out


def representatives(sets: list[list[sp.Matrix]]) -> list[sp.Matrix]:
    return [s[0] for s in sets]


def _which_double_coset(x: sp.Matrix, dcosets: list[list[sp.Matrix]]) -> int:
    k = matrix_key(x)
    for i, d in enumerate(dcosets):
        if any(matrix_key(y) == k for y in d):
            return i
    raise KeyError("Matrix not found in any double coset.")


def operator_adjacency(cosets: list[list[sp.Matrix]], dcosets: list[list[sp.Matrix]]) -> list[list[int]]:
    """Adjacency[i][j] = double-coset index for V_i -> V_j using representative g_i^-1 g_j."""
    reps = representatives(cosets)
    table: list[list[int]] = []
    for gi in reps:
        row = []
        for gj in reps:
            rel = sp.simplify(gi.inv() * gj)
            row.append(_which_double_coset(rel, dcosets))
        table.append(row)
    return table


@dataclass(frozen=True)
class GroupoidResult:
    subgroup: list[sp.Matrix]
    variants: list[list[sp.Matrix]]
    operators: list[list[sp.Matrix]]
    adjacency: list[list[int]]


def correspondence_groupoid(
    parent_group: list[sp.Matrix], product_group: list[sp.Matrix], correspondence: Correspondence
) -> GroupoidResult:
    validate_group(parent_group)
    validate_group(product_group)
    H = correspondence_subgroup(parent_group, product_group, correspondence)
    variants = left_cosets(parent_group, H)
    operators = double_cosets(parent_group, H)
    adjacency = operator_adjacency(variants, operators)
    if sum(len(x) for x in variants) != len(parent_group):
        raise AssertionError("Variant cosets do not partition parent group.")
    if sum(len(x) for x in operators) != len(parent_group):
        raise AssertionError("Double cosets do not partition parent group.")
    return GroupoidResult(H, variants, operators, adjacency)
