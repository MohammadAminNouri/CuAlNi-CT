from __future__ import annotations

from validation.engineered.physical import first_order_case
from validation.engineered.project_pipeline import run_project_pipeline


def test_arbitrary_generated_input_survives_json_loader_and_application_calculation_boundary(tmp_path):
    for i in range(4):
        case = first_order_case("project-e2e", i, parent_condition=3.0)
        result = run_project_pipeline(case, tmp_path)
        assert result["loaded_project_id"] == "engineered_generated_project"
        assert result["audit_passed"], result
        assert result["audit_max_algebraic"] < 1e-7, result
        # CalculationService is part of the app boundary.  Require at least one
        # public bundle API and exercise it instead of stopping at low-level math.
        assert result["bundle_called"], result
