from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from .model import encoding_to_dict, CaseEvaluation
from .jsonutil import to_jsonable


def write_failure(
    case,
    evaluation: CaseEvaluation,
    *,
    minimized=None,
    shrink_history=None,
    root: str | Path = "/tmp/cualni_engineered_failures",
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    safe = case.case_id.replace("/", "__").replace(" ", "_")
    path = root / f"{safe}.json"
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "case": encoding_to_dict(case),
        "evaluation": {
            "passed": evaluation.passed,
            "diagnostics": evaluation.diagnostics,
            "failed_contracts": [
                {
                    "name": c.name,
                    "residual": c.residual,
                    "allowed": c.allowed,
                    "failure_class": c.failure_class.value,
                    "details": c.details,
                }
                for c in evaluation.failed_contracts
            ],
        },
        "shrink_history": list(shrink_history or []),
    }
    if minimized is not None:
        payload["minimized_case"] = encoding_to_dict(minimized)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
