from __future__ import annotations

"""32 point-group scientific micro-descriptions; no twin labels are inferred.

Descriptions characterize crystal-class point symmetry, not space-group
translations, Bravais centering or guaranteed physical transformation twins.
Every operation count and axis/plane displayed beside these descriptions is
computed from the validated exact symmetry inventory, not this table.
"""

POINT_GROUP_SIGNATURES: dict[str, str] = {
    "1": "Only the identity; no nontrivial point symmetry.",
    "-1": "Identity and inversion centre; no rotation or mirror plane.",
    "2": "One twofold rotation along the unique monoclinic b direction.",
    "m": "One mirror plane perpendicular to the unique monoclinic b direction.",
    "2/m": "Twofold axis, perpendicular mirror and inversion centre (unique b).",
    "222": "Three mutually perpendicular twofold axes; no mirrors or inversion.",
    "mm2": "Two intersecting mirror planes and a polar twofold direction.",
    "mmm": "Three orthogonal mirrors and twofold axes; centrosymmetric.",
    "4": "Principal fourfold rotation parallel to c, including its twofold square.",
    "-4": "Principal fourfold rotoinversion; not a proper fourfold rotation.",
    "4/m": "Fourfold rotation, perpendicular mirror and inversion centre.",
    "422": "Fourfold principal axis with perpendicular basal twofold axes.",
    "4mm": "Fourfold principal rotation with vertical mirror families; polar class.",
    "-42m": "Fourfold rotoinversion with basal twofold axes and mirror planes.",
    "4/mmm": "Centrosymmetric tetragonal holohedry: fourfold, twofold and mirror families.",
    "3": "One principal threefold rotation in the hexagonal-axis setting.",
    "-3": "Threefold axis together with inversion (trigonal centrosymmetric class).",
    "32": "Threefold principal axis plus basal twofold axes; proper rotations only.",
    "3m": "Threefold principal axis and vertical mirror planes; polar class.",
    "-3m": "Threefold axis with twofold/mirror families and inversion centre.",
    "6": "Principal sixfold rotation, including threefold and twofold powers.",
    "-6": "Sixfold rotoinversion, not a proper sixfold-axis class.",
    "6/m": "Sixfold axis, perpendicular mirror and inversion centre.",
    "622": "Sixfold principal axis and basal twofold axes; proper rotations only.",
    "6mm": "Sixfold principal axis with vertical mirror families; polar class.",
    "-6m2": "Hexagonal-bar-six improper symmetry, basal twofolds and mirrors.",
    "6/mmm": "Full hexagonal holohedry: sixfold, mirrors, twofolds and inversion.",
    "23": "Proper tetrahedral rotations: threefold body diagonals and twofold axes.",
    "m-3": "Centrosymmetric tetrahedral class with proper threefold/twofold rotations.",
    "432": "Proper cubic rotation group: fourfold, threefold and twofold axes.",
    "-43m": "Tetrahedral improper cubic class: threefolds, rotoinversions and mirrors.",
    "m-3m": "Full cubic holohedry: 48 operations, with mirrors, rotation axes and inversion.",
}


def explain_group(
    symbol: str, counts: dict[str, int], order: int, determinants: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Compute evidence-based symmetry meaning for the selected point group."""
    if symbol not in POINT_GROUP_SIGNATURES:
        raise ValueError(f"No reviewed crystallographic point-group description for {symbol!r}")
    if determinants is not None:
        proper = sum(d == 1 for d in determinants)
        improper = sum(d == -1 for d in determinants)
    else:
        proper = sum(count for kind, count in counts.items()
                     if kind == "identity" or kind.startswith("proper "))
        improper = order - proper
    if proper + improper != order or any(c < 0 for c in counts.values()):
        raise ValueError(f"Operation inventory for {symbol!r} does not sum to group order")
    mirror = counts.get("mirror reflection", 0)
    twofold = counts.get("proper 2-fold rotation", 0)
    higher = sum(c for name, c in counts.items() if name.startswith((
        "proper 3-fold", "proper 4-fold", "proper 6-fold"
    )))
    inv = counts.get("inversion", 0) == 1
    return {
        "signature": POINT_GROUP_SIGNATURES[symbol],
        "proper": proper,
        "improper": improper,
        "inversion": inv,
        "mirror": mirror,
        "twofold": twofold,
        "higher": higher,
        "route_I": "Mirror candidate" if mirror else "No parent-mirror candidate",
        "route_II": "180°-rotation candidate" if twofold else "No parent-twofold candidate",
        "route_weak": "Higher-order axis candidate" if higher else "No proper 3/4/6-fold candidate",
    }
