from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any
import numpy as np


def to_jsonable(value: Any) -> Any:
    """Recursively convert scientific Python objects to strict JSON values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, complex):
            return {"real": float(item.real), "imag": float(item.imag)}
        return to_jsonable(item)
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return [to_jsonable(x) for x in value.reshape(-1).tolist()] if value.ndim == 1 else [
                [to_jsonable(x) for x in row] for row in value.tolist()
            ]
        return [to_jsonable(x) for x in value.tolist()]
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_jsonable(value.to_dict())
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
