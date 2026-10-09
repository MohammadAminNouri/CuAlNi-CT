from __future__ import annotations

"""Reproducibility export with explicit provenance and no benchmark contamination."""

import hashlib
import json
from typing import Any

from .scientific_models import TwinFamilyReport


METHOD_REFERENCES = (
    {"author": "K. Bhattacharya", "title": "Microstructure of Martensite", "use": "Conventional twinning and habit-plane notation; not a runtime answer source"},
    {"author": "J. M. Ball and R. D. James", "title": "Fine phase mixtures as minimizers of energy", "use": "Geometrically nonlinear rank-one compatibility"},
    {"title": "Correspondence Theory for NiTi shape-memory alloys", "doi": "10.3390/cryst12020130", "use": "Correspondence operators and variant genealogy"},
)


def make_export(report: TwinFamilyReport, *, input_payload: dict[str, Any], policy: dict[str, float] | str = "default") -> dict[str, Any]:
    canonical = json.dumps(input_payload, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "schema": "twin-family-research-record/v8",
        "input_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "input": input_payload,
        "coordinate_conventions": {
            "correspondence": "u_M = C_(M<-A) u_A",
            "twin": "R_t U_j - U_i = a tensor n",
            "habit": "R_h (U_i + lambda (a tensor n)) - I = b tensor m",
            "K1": "product crystal reciprocal plane covector",
            "eta1": "product crystal direct shear direction",
            "a_n_b_m": "parent Cartesian frame; m and displayed n are unit normals",
            "exact": "within numerical tolerance for entered metric; not a claim about experimental uncertainty",
        },
        "numerical_policy": policy,
        "scientific_report": report.to_dict(),
        "method_references": list(METHOD_REFERENCES),
    }
