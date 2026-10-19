from __future__ import annotations

"""Read-only, source-complete UI adapter for computed physical twinning modes.

This module DOES NOT calculate physical modes, fit published shear values, or
assign a crystallographic twin type. Every type comes from the independently
cross-locked scientific record. The published mode letters are never assumed.
"""

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from typing import Any

from .mode_type_status import supported_classical_type

VERIFIED_TYPES = ("Compound", "Type I", "Type II")
DIAGNOSTIC_TYPE = "Unresolved"


@dataclass(frozen=True)
class TypeSummary:
    label: str
    entries: tuple[Any, ...]
    source_records: int
    stretch_couples: int
    recorded_habit_solutions: int
    continuous_families: int


@dataclass(frozen=True)
class ModeSummary:
    mode_id: str                 # Exact scientific report key, e.g. "Mode 1"
    display_name: str            # Presentation-only Mode A, B, ..., AA
    shear: float
    symmetry_pairs: int
    types: tuple[TypeSummary, ...]
    unresolved_entries: tuple[Any, ...]
    recorded_habit_solutions: int


@dataclass(frozen=True)
class ModeView:
    visible_modes: tuple[ModeSummary, ...]
    diagnostic_modes: tuple[ModeSummary, ...]  # no verified types, but rank-one data exist
    unresolved_record_count: int
    all_source_record_count: int


def alphabetic_index(zero_based: int) -> str:
    """Excel-like A..Z, AA..; ordinal only, not a literature mode assignment."""
    if not isinstance(zero_based, int) or zero_based < 0:
        raise ValueError("Mode ordinal must be a nonnegative integer")
    number = zero_based + 1
    label = ""
    while number:
        number, digit = divmod(number - 1, 26)
        label = chr(65 + digit) + label
    return label


def _source_id(entry: Any) -> tuple[str, int, int, int, int, str, int]:
    p, c = entry.pair, entry.construction
    return (
        str(entry.operator_family),
        int(p.variant_i), int(p.variant_j), int(p.stretch_i), int(p.stretch_j),
        str(c.construction_id), int(c.branch),
    )


def type_entries(mode: Any, twin_type: str) -> tuple[Any, ...]:
    """Return every source record, prioritising a demonstrable habit example.

    This is only a navigation preference. All alternative sources remain
    accessible; no signs/physical tensors are silently merged or deleted.
    """
    original = tuple(mode.by_type(twin_type))
    if not original:
        raise ValueError(f"Missing calculated {twin_type} branches in {mode.mode_id}")
    keys = [_source_id(e) for e in original]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate physical source ID in one mode/type")
    for e in original:
        c = e.construction
        if twin_type in VERIFIED_TYPES:
            if str(c.classification) != twin_type or not supported_classical_type(c.classification, c.classification_status):
                raise ValueError("Unverified type was promoted into verified mode navigation")
        elif twin_type == DIAGNOSTIC_TYPE:
            if str(c.classification) in VERIFIED_TYPES:
                raise ValueError("Verified classification was misfiled as unresolved")
    return tuple(sorted(original, key=lambda e: (
        -int(bool(e.construction.habit_solutions or e.construction.continuum_fraction)),
        -len(e.construction.habit_solutions),
        _source_id(e),
    )))


def _summarize_type(mode: Any, label: str) -> TypeSummary:
    entries = type_entries(mode, label)
    return TypeSummary(
        label=label,
        entries=entries,
        source_records=len(entries),
        stretch_couples=len({tuple(sorted((int(e.pair.stretch_i), int(e.pair.stretch_j)))) for e in entries}),
        recorded_habit_solutions=sum(len(e.construction.habit_solutions) for e in entries),
        continuous_families=sum(bool(e.construction.continuum_fraction) for e in entries),
    )


def build_mode_view(catalogue: Any) -> ModeView:
    """Expose only verified types; account for EVERY original source in audit.

    Distinct mode identities come from the scientific catalogue; this function
    uses no rounded-shear comparison, alloy name, benchmark or heuristic type.
    """
    visible: list[ModeSummary] = []
    diagnostics: list[ModeSummary] = []
    total = unresolved_count = 0
    mode_ids: set[str] = set()
    for mode in catalogue.modes:
        mode_id = str(mode.mode_id)
        if not mode_id or mode_id in mode_ids:
            raise ValueError("Empty or duplicate mode ID in scientific catalogue")
        mode_ids.add(mode_id)
        shear = float(mode.shear)
        if not isfinite(shear) or shear <= 0:
            raise ValueError("Mode shear must be finite and positive")
        declared = tuple(mode.types)
        if not declared or len(set(declared)) != len(declared):
            raise ValueError("Missing or repeated type in calculated mode")
        if set(declared) - set(VERIFIED_TYPES) - {DIAGNOSTIC_TYPE}:
            raise ValueError("Unknown twin-type classification must not enter the UI")
        grouped = {name: type_entries(mode, name) for name in declared}
        # No source can be hidden or multiply assigned when filtering unresolved.
        raw_ids = Counter(_source_id(e) for e in mode.entries)
        grouped_ids = Counter(_source_id(e) for group in grouped.values() for e in group)
        if raw_ids != grouped_ids or any(v != 1 for v in raw_ids.values()):
            raise ValueError("Mode/type groups do not preserve each calculated source exactly once")
        total += len(mode.entries)
        unknown = grouped.get(DIAGNOSTIC_TYPE, ())
        unresolved_count += len(unknown)
        verified = tuple(_summarize_type(mode, label) for label in VERIFIED_TYPES if label in grouped)
        # Alphabetic labels are assigned to visible modes in stable catalogue order.
        # This does NOT imply that Mode A corresponds to any paper's Mode A.
        summary = ModeSummary(
            mode_id=mode_id,
            display_name=f"Mode {alphabetic_index(len(visible))}" if verified else f"Unresolved group {len(diagnostics) + 1}",
            shear=shear,
            symmetry_pairs=len(mode.stretch_pairs),
            types=verified,
            unresolved_entries=tuple(unknown),
            recorded_habit_solutions=sum(t.recorded_habit_solutions for t in verified),
        )
        if verified:
            visible.append(summary)
        else:
            diagnostics.append(summary)
    return ModeView(tuple(visible), tuple(diagnostics), unresolved_count, total)


def summarize_modes(catalogue: Any) -> tuple[ModeSummary, ...]:
    """Compatibility helper for earlier UI tests and native callers."""
    return build_mode_view(catalogue).visible_modes


def entry_caption(entry: Any) -> str:
    p, c = entry.pair, entry.construction
    return (
        f"M{p.variant_i}–M{p.variant_j} (U{p.stretch_i}–U{p.stretch_j}) "
        f"· operator {entry.operator_family} · rank-one branch {c.branch:+d}"
    )
