from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class DataStatus(str, Enum):
    SOURCE_MEASURED = "SOURCE_MEASURED"
    SOURCE_DERIVED = "SOURCE_DERIVED"
    USER_MEASURED = "USER_MEASURED"
    SYMBOLIC = "SYMBOLIC"
    COMPUTATION_DERIVED = "COMPUTATION_DERIVED"
    HYPOTHETICAL_TEST = "HYPOTHETICAL_TEST"


@dataclass(frozen=True)
class SourcedValue:
    value: Any
    status: DataStatus
    source: str
    units: str = ""
    temperature: str = ""
    state: str = ""
    uncertainty: str = ""
    notes: str = ""
