from __future__ import annotations

"""Pure data types shared by scientific interpretation and Streamlit rendering."""

from dataclasses import dataclass
from typing import Any
import math


@dataclass(frozen=True)
class Evidence:
    quantity: str
    criterion: str
    value: Any
    interpretation: str


@dataclass(frozen=True)
class Finding:
    title: str
    conclusion: str
    rationale: str
    how: str
    physical_meaning: str
    limitation: str = ""
    evidence: tuple[Evidence, ...] = ()
    tone: str = "neutral"
    status: str = ""
    # Reconstructive "How" contract.  Defaults keep older callers compatible.
    theory: str = ""
    equations: tuple[str, ...] = ()
    symbols: tuple[tuple[str, str], ...] = ()
    assumptions: tuple[str, ...] = ()
    backend_mapping: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()
    verbal: str = ""
    # Logical/gating evidence when a numerical table is not the right proof.
    verification: tuple[str, ...] = ()


def sci(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    if number == 0.0:
        return "0"
    if abs(number) < 1.0e-4 or abs(number) >= 1.0e5:
        return f"{number:.{digits}e}"
    return f"{number:.{max(4, digits + 2)}g}"
