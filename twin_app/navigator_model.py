from __future__ import annotations

"""Pure, testable, presentation-only adapter: scientific records -> graph nodes.

No material name, literature value or classification is inferred in this layer.
"""

from dataclasses import dataclass
from typing import Any

from .family_tree_graph import tree_layout
from .scientific_models import TwinFamilyReport


@dataclass(frozen=True)
class NavigatorSummary:
    family_count: int
    couple_count: int
    couples_with_habit: int
    twin_branches: int
    habit_branches: int
    unresolved_branches: int


def build_navigator_model(report: TwinFamilyReport) -> dict[str, Any]:
    layout = tree_layout(report)
    sections: list[dict[str, Any]] = []
    seen: set[str] = set()
    summary = NavigatorSummary(
        family_count=len(layout.families),
        couple_count=len(layout.couples),
        couples_with_habit=0,
        twin_branches=0,
        habit_branches=0,
        unresolved_branches=0,
    )
    habit_couples = 0
    twin_branches = habit_branches = unresolved = 0
    for family in layout.families:
        family_couples: list[dict[str, Any]] = []
        for node in layout.couples:
            if node.family.family_id != family.family.family_id:
                continue
            if node.key in seen:
                raise ValueError(f"Duplicate crystallographic couple key: {node.key}")
            seen.add(node.key)
            constructions = () if node.pair is None else node.pair.constructions
            twin_branches += len(constructions)
            exact = sum(len(branch.habit_solutions) for branch in constructions)
            habit_branches += exact
            is_unresolved = any("unresolved" in branch.classification_status.lower() for branch in constructions)
            unresolved += sum("unresolved" in branch.classification_status.lower() for branch in constructions)
            habit_couples += int(exact > 0)
            state = (
                "exact" if exact else
                "continuum" if any(branch.continuum_fraction for branch in constructions) else
                "unresolved" if is_unresolved else
                "weak-candidate" if family.family.route == "axial_weak" else
                "not-tested" if node.pair is None else
                "no-exact-habit"
            )
            family_couples.append({
                "id": node.key,
                "label": node.label,
                "state": state,
                "habitCount": exact,
                "twinCount": len(constructions),
                "twinTypeUnresolved": is_unresolved,
                "familyId": family.family.family_id,
                "description": (
                    f"{node.label}; {len(constructions)} exact martensite–martensite branch(es); "
                    f"{exact} exact austenite–martensite habit branch(es)"
                ),
            })
        sections.append({
            "id": family.family.family_id,
            "label": f"Family {family.family.family_id}",
            "route": family.family.route,
            "couples": family_couples,
            "count": len(family_couples),
            "hasExactHabit": any(node["state"] == "exact" for node in family_couples),
        })
    if len(seen) != len(layout.couples):
        raise AssertionError("Navigator must retain every correspondence couple")
    summary = NavigatorSummary(
        family_count=len(sections), couple_count=len(seen),
        couples_with_habit=habit_couples, twin_branches=twin_branches,
        habit_branches=habit_branches, unresolved_branches=unresolved,
    )
    return {
        "root": f"{report.parent_phase_id} → {report.product_phase_id}",
        "families": sections,
        "summary": vars(summary),
    }


def first_interpretable_couple(model: dict[str, Any]) -> str | None:
    all_couples = [c for fam in model["families"] for c in fam["couples"]]
    if not all_couples:
        return None
    return next((c["id"] for c in all_couples if c["state"] == "exact"), all_couples[0]["id"])
