from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DataStatus(str, Enum):
    SOURCE_MEASURED = "SOURCE_MEASURED"
    SOURCE_DERIVED = "SOURCE_DERIVED"
    USER_MEASURED = "USER_MEASURED"
    SYMBOLIC = "SYMBOLIC"
    COMPUTATION_DERIVED = "COMPUTATION_DERIVED"
    HYPOTHETICAL_TEST = "HYPOTHETICAL_TEST"
    NOT_EVALUABLE = "NOT_EVALUABLE"


@dataclass(frozen=True)
class SourceRef:
    key: str
    citation: str
    doi: str = ""
    url: str = ""
    pages: str = ""
    equations: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class SourcedValue:
    value: Any
    status: DataStatus
    source_key: str
    units: str = ""
    temperature: str = ""
    stress_state: str = ""
    material_state: str = ""
    uncertainty: str = ""
    original_cell: str = ""
    conversion: str = ""
    notes: str = ""


@dataclass
class AuditTrail:
    entries: list[str] = field(default_factory=list)

    def add(self, text: str) -> None:
        self.entries.append(str(text))

    def markdown(self) -> str:
        if not self.entries:
            return "_No audit entries._"
        return "\n".join(f"- {x}" for x in self.entries)
