from __future__ import annotations

"""Phase-2B interpretation gates for comparison availability."""

from dataclasses import replace
from typing import Any

import app.scientific_interpretation_v7 as v7
from app.phase2_contracts import ct_ptmc_or_availability, sanitize_finding


def or_comparison_finding(
    equivalence: Any,
    *,
    enabled: bool,
    tolerance_deg: float,
    unified: Any | None = None,
):
    finding = v7.or_comparison_finding(
        equivalence,
        enabled=enabled,
        tolerance_deg=tolerance_deg,
    )
    if not enabled or unified is None:
        return finding

    gate = ct_ptmc_or_availability(unified)
    if gate["evaluable"]:
        return finding

    ct_count = int(gate["ct_exact_or_rows"])
    ptmc_habits = int(gate["ptmc_exact_habit_rows"])
    ptmc_or = int(gate["ptmc_exact_or_rows"])
    reason = str(gate["reason"])

    if reason == "ptmc_or_unavailable":
        if ptmc_habits == 0:
            rationale = (
                "Cayron CT closing-gap ORs are available, but the enabled PTMC run produced no exact "
                "twinned-laminate habit branch, so there is no PTMC physical OR to compare."
            )
        else:
            rationale = (
                "Exact PTMC habit rows exist, but none carries an explicit physical OR observable in the "
                "active report; a CT↔PTMC OR residual therefore cannot be formed."
            )
    elif reason == "ct_or_unavailable":
        rationale = (
            "Exact PTMC OR-bearing habit rows are available, but no exact Cayron CT closing-gap OR is "
            "available in the active report."
        )
    else:
        rationale = (
            "Neither side supplies the pair of explicit physical OR observables required for a symmetry-reduced CT↔PTMC comparison."
        )

    return sanitize_finding(
        replace(
            finding,
            conclusion="CT↔PTMC OR comparison is not evaluable for this run.",
            rationale=rationale,
            physical_meaning=(
                "No cross-theory OR separation is defined until both an exact CT closing-gap OR and an exact PTMC OR-bearing branch are present."
            ),
            limitation=(
                "Unavailable OR observables are not disagreement and must not be converted into an infinite, failed or zero residual."
            ),
            verification=(
                "requested = yes",
                f"exact CT closing-gap OR rows = {ct_count}",
                f"exact PTMC habit rows = {ptmc_habits}",
                f"exact PTMC OR-bearing rows = {ptmc_or}",
                "pairwise CT↔PTMC OR residual evaluated = no",
                "status = not evaluable",
                f"requested OR tolerance = {float(tolerance_deg):.9g}°",
            ),
            tone="neutral",
        )
    )
