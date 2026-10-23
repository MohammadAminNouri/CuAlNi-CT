"""Non-destructive scientific-evidence classification for martensite twins.

This classifies *provenance claims*, NOT independently certifies their truth.
The discrete CT audit is a separate mandatory release gate.
"""
from __future__ import annotations

from typing import Any

CLASSICAL = frozenset({"Type I", "Type II", "Compound"})
NONCONVENTIONAL = "Nonconventional rank-one (parent mirror/twofold excluded)"
VALID_CATEGORIES = frozenset({
    "direct_classical_claim",
    "symmetry_transported_classical_claim",
    "parent_mirror_twofold_excluded_under_supplied_group",
    "unresolved_classical_provenance",
})


def evidence_category(classification: Any, classification_status: Any) -> str:
    """Do not merge 'unresolved' with symmetry-excluded constructions.

    This is a parser for evidence provenance written by the production engine.
    It never infers I, II, or Compound from geometry or an absent witness.
    """
    label, status = str(classification).strip(), str(classification_status).strip()
    low = status.lower()
    if label in CLASSICAL:
        if low.startswith("cross-locked:"):
            return "direct_classical_claim"
        if (
            low.startswith("symmetry-transported from directly cross-locked ")
            and "; g-index " in low
            and "; ordered stretch and physical rank-one tensor residuals " in low
            and low.endswith("this is not an independent discrete cross-lock for the target pair")
        ):
            return "symmetry_transported_classical_claim"
        raise ValueError("Classical subtype has no permitted direct/transport provenance")
    if label == NONCONVENTIONAL:
        if low.startswith("proved: neither a parent mirror nor a proper 180-degree rotation") and (
            "no classical parent" in low and "type-i mirror or type-ii twofold symmetry route exists" in low
        ):
            return "parent_mirror_twofold_excluded_under_supplied_group"
        raise ValueError("Parent-symmetry exclusion lacks its explicit P0 certificate")
    if label.startswith("Exact rank-one relation"):
        if not status:
            raise ValueError("Unresolved rank-one relation has no explanation")
        return "unresolved_classical_provenance"
    raise ValueError("Unexpected classification label; do not silently relabel a physical twin")


def category_for_construction(construction: Any) -> str:
    return evidence_category(construction.classification, construction.classification_status)
