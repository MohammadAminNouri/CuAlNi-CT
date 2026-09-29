from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.cayron_martensite_assessment import build_cayron_martensite_assessment


@dataclass(frozen=True)
class _EnumLike:
    value: str


@dataclass(frozen=True)
class _Row:
    prediction_kind: _EnumLike
    branch_label: str
    exact: bool | None = True
    theory: _EnumLike = _EnumLike("cayron_ct")
    shear_magnitude: float | None = None
    shape_vector_parent_crystal: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class _Unified:
    rows: tuple[_Row, ...]


class _Response:
    def __init__(self, *, exact_am: bool = False):
        self.project_payload = {
            "transformations": [
                {
                    "transformation_id": "A_to_M",
                    "correspondence_M_from_A": [
                        ["1", "1/2", "0"],
                        ["-1/2", "1", "1/3"],
                        ["0", "-1/3", "1"],
                    ],
                }
            ]
        }
        self.result = {
            "summary": {
                "variant_count": 12,
                "operator_count": 8,
                "ct_exact_compatible": exact_am,
                "degeneracy_order": 1 if exact_am else 0,
            },
            "topology": {
                "parent_group_order": 48,
                "product_group_order": 4,
                "subgroup_order": 4,
                "n_variants": 12,
                "n_operators": 8,
            },
            "metric": {
                "parent_metric": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "product_metric": [[1.1, 0, 0], [0, 1.0, 0], [0, 0, 0.9]],
                "pulled_product_metric": [[1.1, 0, 0], [0, 1.0, 0], [0, 0, 0.9]],
                "cmc_dimensional": [[0.1, 0, 0], [0, 0.0, 0], [0, 0, -0.1]],
                "cmc_normalized": [[0.1, 0, 0], [0, 0.0, 0], [0, 0, -0.1]],
                "smc_dimensional": [[0.01, 0, 0], [0, 0.0, 0], [0, 0, -0.01]],
                "stretch": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
            },
            "ct_detail": {
                "exact_compatible": exact_am,
                "degeneracy_order": 1 if exact_am else 0,
                "reason": "first-order degeneracy: two compatible habit planes" if exact_am else "no exact CMC degeneracy",
                "eta_eigenvalues": [-0.1, 0.003876191302128462 if not exact_am else 0.0, 0.2],
                "generalized_mu": [0.9, 1.0038761913021285 if not exact_am else 1.0, 1.2],
                "nearest_zero_index": 1,
                "nearest_zero_residual": 0.003876191302128462 if not exact_am else 0.0,
                "inertia": [1, 1 if exact_am else 0, 1],
                "exact_habit_planes_parent_covectors": [[1.0, 0.0, 1.0], [1.0, 0.0, -1.0]] if exact_am else [],
                "approximate_diagnostic": {
                    "residual": 0.003876191302128462,
                    "admissible_signature": True,
                    "candidate_planes_parent_covectors": [[1.0, 0.0, 0.9], [1.0, 0.0, -0.9]],
                },
            },
        }


def _step(assessment, step_id: str):
    return next(item for item in assessment.steps if item.step_id == step_id)


def test_ct_only_assessment_separates_martensitic_construction_from_exact_am_compatibility():
    response = _Response(exact_am=False)
    unified = _Unified(
        rows=(
            _Row(_EnumLike("ct_mm_twin"), "CT twin I", shear_magnitude=0.12),
            _Row(_EnumLike("ct_mm_twin"), "CT twin II", shear_magnitude=0.12),
            _Row(_EnumLike("ct_am_habit"), "approximate diagnostic", exact=False, shape_vector_parent_crystal=(0.1, 0.0, 0.0)),
            _Row(_EnumLike("ct_closing_gap_or"), "closing gap A"),
            _Row(_EnumLike("ct_closing_gap_or"), "closing gap B"),
        )
    )
    assessment = build_cayron_martensite_assessment(
        response,
        unified,
        closing_gap_requested=True,
        supercompatibility_requested=True,
    )

    assert assessment.overall_status == "CT-consistent martensitic crystallography"
    assert "Exact parent/martensite compatibility is not reached" in assessment.overall_answer
    assert _step(assessment, "topology").status == "reached"
    assert _step(assessment, "mm_twins").status == "reached"
    assert _step(assessment, "am_exact").status == "not reached"
    assert _step(assessment, "nearest").status == "diagnostic only"
    assert _step(assessment, "habit").status == "not reached"
    assert _step(assessment, "closing_gap").status == "reached"
    assert _step(assessment, "supercompatibility").status == "not evaluable"


def test_ct_assessment_never_promotes_nearest_degeneracy_to_exact_habit_or_supercompatibility_seed():
    assessment = build_cayron_martensite_assessment(
        _Response(exact_am=False),
        _Unified(rows=()),
        closing_gap_requested=False,
        supercompatibility_requested=True,
    )
    habit = _step(assessment, "habit")
    super_step = _step(assessment, "supercompatibility")
    assert habit.status == "not reached"
    assert "Approximate diagnostic planes" in dict(habit.evidence)
    assert super_step.status == "not evaluable"
    assert "requires an exact CT A/M habit/shear seed" in super_step.answer


def test_ct_assessment_distinguishes_not_requested_from_not_reached():
    assessment = build_cayron_martensite_assessment(
        _Response(exact_am=True),
        _Unified(rows=(_Row(_EnumLike("ct_mm_twin"), "CT twin"),)),
        closing_gap_requested=False,
        supercompatibility_requested=False,
    )
    assert _step(assessment, "closing_gap").status == "not requested"
    assert _step(assessment, "supercompatibility").status == "not requested"
    assert _step(assessment, "habit").status == "reached"


def test_ct_assessment_reports_exact_supercompatibility_only_from_exact_ct_rows():
    assessment = build_cayron_martensite_assessment(
        _Response(exact_am=True),
        _Unified(
            rows=(
                _Row(_EnumLike("ct_mm_twin"), "CT twin"),
                _Row(_EnumLike("ct_am_habit"), "exact habit", exact=True, shape_vector_parent_crystal=(0.1, 0.0, 0.0)),
                _Row(_EnumLike("ct_supercompatibility"), "exact super", exact=True),
            )
        ),
        closing_gap_requested=False,
        supercompatibility_requested=True,
    )
    assert _step(assessment, "am_exact").status == "reached"
    assert _step(assessment, "smc").status == "reached"
    assert _step(assessment, "supercompatibility").status == "reached"


def test_ct_assessment_without_unified_inventory_is_explicitly_incomplete_not_negative():
    assessment = build_cayron_martensite_assessment(
        _Response(exact_am=False),
        None,
        closing_gap_requested=False,
        supercompatibility_requested=False,
    )
    assert assessment.overall_status == "incomplete CT inventory"
    assert _step(assessment, "mm_twins").status == "not evaluated"



def test_third_order_exact_cmc_is_not_forced_into_a_unique_habit_or_supercompatibility_seed():
    response = _Response(exact_am=True)
    response.result["ct_detail"]["degeneracy_order"] = 3
    response.result["summary"]["degeneracy_order"] = 3
    response.result["ct_detail"]["reason"] = "third-order degeneracy: pulled-back martensite metric equals parent metric"
    response.result["ct_detail"]["exact_habit_planes_parent_covectors"] = []
    assessment = build_cayron_martensite_assessment(
        response,
        _Unified(rows=(_Row(_EnumLike("ct_mm_twin"), "CT twin"),)),
        closing_gap_requested=False,
        supercompatibility_requested=True,
    )
    assert _step(assessment, "am_exact").status == "reached"
    assert _step(assessment, "habit").status == "not uniquely defined"
    assert _step(assessment, "supercompatibility").status == "not evaluable"
    overall = dict(_step(assessment, "overall").evidence)
    assert overall["Exact CT A/M habit plane"] == "not uniquely defined"


def test_v10_is_presentation_only_and_routes_research_page_through_v9():
    root = Path(__file__).resolve().parents[2]
    contract = (root / "app" / "cayron_martensite_assessment.py").read_text(encoding="utf-8")
    renderer = (root / "app" / "research_workspaces_v10.py").read_text(encoding="utf-8")
    page = (root / "app" / "pages" / "2_CT_Equivalence_Lab.py").read_text(encoding="utf-8")

    forbidden_solver_calls = (
        "analyze_austenite_martensite(",
        "TheoryComparisonAdapter(",
        "twins_from_operator(",
        "CayronOrientationAdapter(",
        "single_variant_austenite_habit_solutions(",
    )
    for token in forbidden_solver_calls:
        assert token not in contract
        assert token not in renderer

    assert "v9.render_research_extension()" in renderer
    assert "v4._render_conclusions = _render_conclusions_v10" in renderer
    assert "research_workspaces_v10" in page
    assert "CMC=C^T M_M C-M_A" in contract
    assert "SMC=M_A^{-1}-C^{-1}M_M^{-1}C^{-T}" in contract
    assert "2(m_A^T n)d_A=a" in contract
    assert "Cayron 2026 Eq. (32)" in contract
    assert "Eq. (41)" in contract
    assert "Eqs. (17)–(20)" in contract
