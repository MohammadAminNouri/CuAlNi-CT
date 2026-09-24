from __future__ import annotations

import json
from pathlib import Path

from .physical import metric_to_cell


def project_payload(case) -> dict:
    C = []
    for row in case.C_m_from_a:
        out = []
        for x in row:
            if abs(float(x) - round(float(x))) < 1e-12:
                out.append(int(round(float(x))))
            else:
                out.append(str(float(x)))
        C.append(out)

    return {
        "schema_version": 1,
        "project_id": "engineered_generated_project",
        "title": "Engineered generated arbitrary two-phase input",
        "phases": [
            {
                "phase_id": "A",
                "label": "generated parent",
                "physical_phase": "generated parent",
                "cell_representation": "user_cell",
                "point_group": "1",
                "cell": {**metric_to_cell(case.M_parent), "length_unit": "angstrom"},
            },
            {
                "phase_id": "M",
                "label": "generated product",
                "physical_phase": "generated product",
                "cell_representation": "user_cell",
                "point_group": "1",
                "cell": {**metric_to_cell(case.M_product), "length_unit": "angstrom"},
            },
        ],
        "transformations": [
            {
                "transformation_id": "T",
                "parent_phase_id": "A",
                "product_phase_id": "M",
                "correspondence_M_from_A": C,
            }
        ],
    }


def run_project_pipeline(case, tmp_path: Path) -> dict:
    from cualni_cryst.project_io import load_project
    from cualni_cryst.calculation_service import CalculationService, CalculationKind
    from cualni_cryst.scientific_contracts import audit_core_theory_consistency

    path = tmp_path / "generated_project.json"
    path.write_text(json.dumps(project_payload(case), indent=2) + "\n", encoding="utf-8")
    loaded = load_project(path)
    project = loaded.project

    assert len(project.phases) == 2
    assert len(project.transformations) == 1
    parent = next(x for x in project.phases if x.phase_id == "A")
    product = next(x for x in project.phases if x.phase_id == "M")
    trans = project.transformations[0]

    audit = audit_core_theory_consistency(
        parent.lattice, product.lattice, trans.correspondence
    )

    service = CalculationService(project)
    bundle = service.compute(CalculationKind.TRANSFORMATION_BUNDLE, "T")
    bundle_dict = bundle.to_dict()

    return {
        "loaded_project_id": project.project_id,
        "audit_passed": bool(audit.passed),
        "audit_max_algebraic": float(audit.maximum_algebraic_residual),
        "bundle_called": True,
        "bundle_kind": CalculationKind.TRANSFORMATION_BUNDLE.value,
        "bundle_summary": bundle_dict,
        "cache_entries": service.cache_stats.entries,
    }
