from __future__ import annotations

"""Auditable interpretation of exact point-group operations.

All operation numbers and axes are derived from SymmetryInventory. This is a
*declared point-group/metric* diagnostic, not an atomistic space-group search.
No reflection or rotation alone establishes a physical martensitic twin.
"""

from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable

from .symmetry_inventory import SymmetryInventory, SymmetryOperationRecord
from .point_group_explainer import explain_group


@dataclass(frozen=True)
class OperationFamily:
    name: str
    operations: int
    geometric_elements: tuple[tuple[int, int, int], ...]
    coordinate_type: str
    example: SymmetryOperationRecord


@dataclass(frozen=True)
class PointGroupGuide:
    symbol: str
    title: str
    family: str
    setting: str
    order: int
    proper: int
    improper: int
    inversion: bool
    signature: str
    families: tuple[OperationFamily, ...]
    role_explanation: str


def group_guide(inventory: SymmetryInventory, *, parent: bool) -> PointGroupGuide:
    d = explain_group(inventory.symbol, dict(inventory.counts), inventory.expected_order,
                      tuple(op.determinant for op in inventory.operations))
    if len(inventory.operations) != inventory.expected_order:
        raise ValueError("Point-group operations disagree with expected order")
    kinds: dict[str, list[SymmetryOperationRecord]] = defaultdict(list)
    for op in inventory.operations:
        kinds[op.kind].append(op)
    families = []
    for kind, ops in sorted(kinds.items(), key=lambda item: (
        0 if item[0] == 'identity' else 1 if 'rotation' in item[0] else 2,
        item[0],
    )):
        axes = tuple(sorted({op.axis_or_plane for op in ops if op.axis_or_plane is not None}))
        is_mirror = kind == 'mirror reflection'
        label = ({'identity': 'Identity', 'inversion': 'Inversion',
                  'mirror reflection': 'Mirror reflections'}).get(kind, kind.replace('proper ', 'Proper ').replace('improper ', 'Improper '))
        families.append(OperationFamily(
            name=label,
            operations=len(ops),
            geometric_elements=axes,
            coordinate_type='(hkl) plane covector' if is_mirror else '[uvw] rotation axis' if 'rotation' in kind else 'none',
            example=ops[0],
        ))
    if parent:
        role = ('Parent symmetry groups the generated martensite correspondence variants. '
                'Some parent mirrors and 180° rotations are candidate Type-I/Type-II generators. '
                'The actual twin type also requires the paired variants, their metric and a verified rank-one equation.')
    else:
        role = ('Product symmetry identifies equivalent correspondence descriptions and crystallographic '
                'planes/directions in the product basis. It does not supply a twin classification by itself.')
    return PointGroupGuide(
        symbol=inventory.symbol,
        title=str(d['plain_name']),
        family=inventory.crystal_family,
        setting=inventory.conventional_setting,
        order=inventory.expected_order,
        proper=int(d['proper']),
        improper=int(d['improper']),
        inversion=bool(d['inversion']),
        signature=str(d['signature']),
        families=tuple(families),
        role_explanation=role,
    )


def explanation_for_operation(op: SymmetryOperationRecord) -> tuple[str, str]:
    """Plain language and exact action category; never infer a physical twin."""
    if op.kind == 'identity':
        return ('The crystal is unchanged. This is the identity operation.',
                'It produces no distinct variant and cannot by itself produce a twin.')
    if op.kind == 'inversion':
        return ('Every fractional coordinate changes sign relative to the origin.',
                'Inversion is not a mirror plane or a 180° proper rotation.')
    if op.kind == 'mirror reflection':
        return ('The crystal is reflected across the crystallographic plane labelled (hkl).',
                'A parent mirror may generate a candidate Type-I description; metric compatibility must still be checked.')
    if op.kind == 'proper 2-fold rotation':
        return ('The crystal is rotated by half a turn (180°) around the listed direct-space [uvw] axis.',
                'A parent twofold rotation may generate a candidate Type-II description; the independent tensor and shear checks are required.')
    if op.kind.startswith('proper '):
        return ('The crystal is rotated around a direct-space axis and is unchanged after the symmetry operation.',
                'Higher-order rotation is symmetry information, not proof of a classical Type-I/II twin.')
    return ('This is an orientation-reversing crystallographic operation in the exact point-group inventory.',
            'It must not be silently interpreted as a mirror or classical twin generator.')
