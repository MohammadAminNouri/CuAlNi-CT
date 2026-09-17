from __future__ import annotations

"""Exact finite-group machinery used by Cayron-style variant analysis.

The important rule in this module is that the *discrete* part of the problem is
performed with SymPy exact matrices.  No floating tolerance is used to decide
subgroup membership, coset membership or double-coset membership.
"""

from dataclasses import dataclass
from collections import Counter
from typing import Iterable
import sympy as sp

from .symmetry import matrix_key, classify_symmetry
from .correspondence import Correspondence


def _sort_key(M: sp.Matrix) -> tuple:
    return tuple(str(x) for x in matrix_key(M))


def _unique_sorted(mats: Iterable[sp.Matrix]) -> list[sp.Matrix]:
    d = {matrix_key(sp.Matrix(m)): sp.Matrix(m) for m in mats}
    return sorted(d.values(), key=_sort_key)


def _as_dict(group: Iterable[sp.Matrix]) -> dict[tuple, sp.Matrix]:
    return {matrix_key(sp.Matrix(g)): sp.Matrix(g) for g in group}


def validate_group(group: list[sp.Matrix]) -> None:
    """Raise if matrices do not form a finite group under multiplication."""
    G = _as_dict(group)
    if not G:
        raise ValueError("Empty set is not a group.")
    if matrix_key(sp.eye(3)) not in G:
        raise ValueError("Identity is missing.")
    for a in G.values():
        if matrix_key(sp.simplify(a.inv())) not in G:
            raise ValueError(f"Inverse missing for {a}")
        for b in G.values():
            if matrix_key(sp.simplify(a * b)) not in G:
                raise ValueError("Group is not closed under multiplication.")


def correspondence_subgroup(
    parent_group: list[sp.Matrix],
    product_group: list[sp.Matrix],
    correspondence: Correspondence,
) -> list[sp.Matrix]:
    r"""Return Cayron's correspondence intersection subgroup H_C^A.

    Package convention is ``u_M = C_m_from_a u_A``.  Therefore a daughter
    symmetry ``g_M`` written in daughter coordinates is represented in parent
    coordinates by ``C^{-1} g_M C`` and

        H_C^A = G_A ∩ C^{-1} G_M C.

    This is the same content as Cayron's notation
    ``G_A ∩ C^{A→M} G_M C^{M→A}``, but the Python names spell out the actual
    coordinate action to avoid the superscript-direction ambiguity.
    """
    Gp = _as_dict(parent_group)
    C = correspondence.C_m_from_a
    Ci = correspondence.C_a_from_m
    transformed = [sp.simplify(Ci * gm * C) for gm in product_group]
    return _unique_sorted(Gp[matrix_key(h)] for h in transformed if matrix_key(h) in Gp)


def left_cosets(group: list[sp.Matrix], subgroup: list[sp.Matrix]) -> list[list[sp.Matrix]]:
    """Deterministically partition G into left cosets gH."""
    validate_group(group)
    validate_group(subgroup)
    G = _unique_sorted(group)
    H = _unique_sorted(subgroup)
    remaining = {matrix_key(g): g for g in G}
    out: list[list[sp.Matrix]] = []
    while remaining:
        g = sorted(remaining.values(), key=_sort_key)[0]
        coset = _unique_sorted(sp.simplify(g * h) for h in H)
        out.append(coset)
        for x in coset:
            remaining.pop(matrix_key(x), None)
    return out


def double_cosets(group: list[sp.Matrix], subgroup: list[sp.Matrix]) -> list[list[sp.Matrix]]:
    r"""Deterministically partition G into double cosets HgH."""
    validate_group(group)
    validate_group(subgroup)
    G = _unique_sorted(group)
    H = _unique_sorted(subgroup)
    remaining = {matrix_key(g): g for g in G}
    out: list[list[sp.Matrix]] = []
    while remaining:
        g = sorted(remaining.values(), key=_sort_key)[0]
        d = _unique_sorted(sp.simplify(h1 * g * h2) for h1 in H for h2 in H)
        out.append(d)
        for x in d:
            remaining.pop(matrix_key(x), None)
    return out


def representatives(sets: list[list[sp.Matrix]]) -> list[sp.Matrix]:
    return [sorted(s, key=_sort_key)[0] for s in sets]


def which_double_coset(x: sp.Matrix, dcosets: list[list[sp.Matrix]]) -> int:
    k = matrix_key(sp.Matrix(x))
    for i, d in enumerate(dcosets):
        if k in {matrix_key(y) for y in d}:
            return i
    raise KeyError("Matrix not found in any double coset.")


def operator_adjacency(cosets: list[list[sp.Matrix]], dcosets: list[list[sp.Matrix]]) -> list[list[int]]:
    """Table[i][j] = operator class of the arrow variant i -> variant j."""
    reps = representatives(cosets)
    return [
        [which_double_coset(sp.simplify(gi.inv() * gj), dcosets) for gj in reps]
        for gi in reps
    ]


def burnside_double_coset_count(group: list[sp.Matrix], subgroup: list[sp.Matrix]) -> int:
    r"""Count H\G/H via Burnside's lemma, independently of enumeration.

    H×H acts on G by (h1,h2): g -> h1 g h2^{-1}.  A matrix g is fixed iff
    h1 g h2^{-1}=g.  The orbit count is the average number of fixed points.
    For the small crystallographic groups used here a direct exact count is
    clearer and safer than a character-table implementation.
    """
    G = _unique_sorted(group)
    H = _unique_sorted(subgroup)
    total_fixed = 0
    for h1 in H:
        for h2 in H:
            for g in G:
                if sp.simplify(h1 * g * h2.inv() - g) == sp.zeros(3):
                    total_fixed += 1
    den = len(H) ** 2
    if total_fixed % den:
        raise AssertionError("Burnside fixed-point total is not divisible by |H|^2")
    return total_fixed // den


def inverse_operator_map(dcosets: list[list[sp.Matrix]]) -> list[int]:
    """Return operator index containing inverses of each double coset."""
    out = []
    for d in dcosets:
        invset = {matrix_key(sp.simplify(g.inv())) for g in d}
        matches = [j for j, e in enumerate(dcosets) if invset == {matrix_key(x) for x in e}]
        if len(matches) != 1:
            raise AssertionError("Could not identify a unique inverse double coset")
        out.append(matches[0])
    return out


def multivalued_operator_composition(adjacency: list[list[int]]) -> list[list[set[int]]]:
    """Compute the groupoid's multivalued operator-composition signature.

    Entry [a][b] contains all operator classes c for which there are variants
    i,j,k with i->j of class a, j->k of class b, and i->k of class c.
    This is a transparent arrow-composition implementation of Cayron's groupoid
    idea.  It is deliberately a *set-valued* operation, not forced into a group.
    """
    A = adjacency
    nvar = len(A)
    nop = max(max(r) for r in A) + 1
    table = [[set() for _ in range(nop)] for _ in range(nop)]
    for i in range(nvar):
        for j in range(nvar):
            a = A[i][j]
            for k in range(nvar):
                b = A[j][k]
                c = A[i][k]
                table[a][b].add(c)
    return table


@dataclass(frozen=True)
class OperatorSummary:
    index: int
    size: int
    inverse_index: int
    ambivalent: bool
    symmetry_kinds: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class GroupoidResult:
    subgroup: list[sp.Matrix]
    variants: list[list[sp.Matrix]]
    operators: list[list[sp.Matrix]]
    adjacency: list[list[int]]
    inverse_operators: list[int]
    composition: list[list[set[int]]]
    summaries: list[OperatorSummary]
    burnside_count: int

    @property
    def n_variants(self) -> int:
        return len(self.variants)

    @property
    def n_operators(self) -> int:
        return len(self.operators)


def _operator_summaries(ops: list[list[sp.Matrix]], inv: list[int]) -> list[OperatorSummary]:
    out: list[OperatorSummary] = []
    for i, d in enumerate(ops):
        kinds = Counter(classify_symmetry(g).kind for g in d)
        out.append(
            OperatorSummary(
                index=i,
                size=len(d),
                inverse_index=inv[i],
                ambivalent=(inv[i] == i),
                symmetry_kinds=tuple(sorted(kinds.items())),
            )
        )
    return out


def correspondence_groupoid(
    parent_group: list[sp.Matrix], product_group: list[sp.Matrix], correspondence: Correspondence
) -> GroupoidResult:
    """Build and cross-check the complete discrete CT correspondence groupoid."""
    validate_group(parent_group)
    validate_group(product_group)
    H = correspondence_subgroup(parent_group, product_group, correspondence)
    validate_group(H)
    variants = left_cosets(parent_group, H)
    operators = double_cosets(parent_group, H)
    adjacency = operator_adjacency(variants, operators)
    inv = inverse_operator_map(operators)
    composition = multivalued_operator_composition(adjacency)
    burnside = burnside_double_coset_count(parent_group, H)
    if len(parent_group) % len(H):
        raise AssertionError("Lagrange theorem violated")
    if len(variants) != len(parent_group) // len(H):
        raise AssertionError("Incorrect number of left cosets")
    if sum(map(len, variants)) != len(parent_group):
        raise AssertionError("Variant cosets do not partition parent group")
    if sum(map(len, operators)) != len(parent_group):
        raise AssertionError("Double cosets do not partition parent group")
    if burnside != len(operators):
        raise AssertionError("Burnside count disagrees with explicit double-coset enumeration")
    return GroupoidResult(
        H, variants, operators, adjacency, inv, composition,
        _operator_summaries(operators, inv), burnside,
    )
