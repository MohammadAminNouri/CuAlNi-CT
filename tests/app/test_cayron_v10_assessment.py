from __future__ import annotations

import ast
from dataclasses import dataclass, field
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
    residuals: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class _Unified:
    rows: tuple[_Row, ...]


class _Response:
    def __init__(self, *, exact_am: bool = False):
        self.project_payload = {
            "numerical_policy": {"algebraic": 1.0e-10},
            "transformations": [
                {
                    "transformation_id": "A_to_M",
                    "correspondence_M_from_A": [
                        ["1", "1/2", "0"],
                        ["-1/2", "1", "1/3"],
                        ["0", "-1/3", "1"],
                    ],
                }
            ],
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
                "reason": (
                    "first-order degeneracy: two compatible habit planes"
                    if exact_am
                    else "no exact CMC degeneracy"
                ),
                "eta_eigenvalues": [
                    -0.1,
                    0.0 if exact_am else 0.003876191302128462,
                    0.2,
                ],
                "generalized_mu": [
                    0.9,
                    1.0 if exact_am else 1.0038761913021285,
                    1.2,
                ],
                "nearest_zero_index": 1,
                "nearest_zero_residual": (
                    0.0 if exact_am else 0.003876191302128462
                ),
                "inertia": [1, 1 if exact_am else 0, 1],
                "exact_habit_planes_parent_covectors": (
                    [[1.0, 0.0, 1.0], [1.0, 0.0, -1.0]]
                    if exact_am
                    else []
                ),
                "approximate_diagnostic": {
                    "residual": 0.003876191302128462,
                    "admissible_signature": True,
                    "candidate_planes_parent_covectors": [
                        [1.0, 0.0, 0.9],
                        [1.0, 0.0, -0.9],
                    ],
                },
            },
        }


def _step(assessment, step_id: str):
    return next(item for item in assessment.steps if item.step_id == step_id)


def test_ct_only_assessment_separates_topology_twins_and_exact_am():
    assessment = build_cayron_martensite_assessment(
        _Response(exact_am=False),
        _Unified(
            rows=(
                _Row(_EnumLike("ct_mm_twin"), "CT twin I", shear_magnitude=0.12),
                _Row(_EnumLike("ct_mm_twin"), "CT twin II", shear_magnitude=0.12),
                _Row(
                    _EnumLike("ct_am_habit"),
                    "approximate diagnostic",
                    exact=False,
                    shape_vector_parent_crystal=(0.1, 0.0, 0.0),
                ),
                _Row(_EnumLike("ct_closing_gap_or"), "closing gap A"),
            )
        ),
        closing_gap_requested=True,
        supercompatibility_requested=True,
    )

    assert assessment.overall_status == "CT-consistent martensitic crystallography"
    assert _step(assessment, "topology").status == "reached"
    assert _step(assessment, "mm_twins").status == "reached"
    assert _step(assessment, "am_exact").status == "not reached"
    assert _step(assessment, "nearest").status == "diagnostic only"
    assert _step(assessment, "habit").status == "not reached"
    assert _step(assessment, "closing_gap").status == "reached"
    assert _step(assessment, "supercompatibility").status == "not evaluable"


def test_operator_count_is_reported_from_current_state_not_hardcoded_to_seven():
    assessment = build_cayron_martensite_assessment(
        _Response(exact_am=False),
        _Unified(rows=()),
        closing_gap_requested=False,
        supercompatibility_requested=False,
    )
    topology = _step(assessment, "topology")
    assert "8 double-coset operator class" in topology.answer
    assert "not a universal constant" in topology.reasoning
    assert "7 operator classes" in topology.reasoning


def test_nearest_degeneracy_is_explicitly_app_diagnostic_not_cayron_table2_distance():
    assessment = build_cayron_martensite_assessment(
        _Response(exact_am=False),
        _Unified(rows=()),
        closing_gap_requested=False,
        supercompatibility_requested=False,
    )
    step = _step(assessment, "nearest")
    assert step.status == "diagnostic only"
    assert "app-defined" in step.reasoning
    assert "not Cayron's lattice-parameter distance" in step.reasoning
    assert r"r_{\mathrm{app}}=\min_i|\eta_i|" in step.formulae


def test_habit_uses_m_not_p_for_am_interface():
    assessment = build_cayron_martensite_assessment(
        _Response(exact_am=True),
        _Unified(rows=()),
        closing_gap_requested=False,
        supercompatibility_requested=False,
    )
    habit = _step(assessment, "habit")
    joined = "\n".join(habit.formulae)
    assert "m_A" in joined
    assert "p_A=M_A" not in joined
    assert "p_A is reserved for an M/M twin plane" in habit.reasoning


def test_second_order_habit_uses_unique_plane_formula_not_first_order_two_plane_formula():
    response = _Response(exact_am=True)
    response.result["ct_detail"]["degeneracy_order"] = 2
    response.result["summary"]["degeneracy_order"] = 2
    response.result["ct_detail"]["reason"] = (
        "second-order degeneracy: unique compatible habit plane"
    )
    response.result["ct_detail"]["eta_eigenvalues"] = [-0.262461, 0.0, 0.0]
    response.result["ct_detail"]["generalized_mu"] = [0.737539, 1.0, 1.0]
    response.result["ct_detail"]["inertia"] = [1, 2, 0]
    response.result["ct_detail"]["exact_habit_planes_parent_covectors"] = [
        [1.0, -1.0, -2.290016]
    ]

    assessment = build_cayron_martensite_assessment(
        response,
        _Unified(rows=()),
        closing_gap_requested=False,
        supercompatibility_requested=False,
    )
    habit = _step(assessment, "habit")
    joined = "\n".join(habit.formulae)

    assert habit.status == "reached"
    assert r"\eta_i=\eta_j=0" in joined
    assert r"m_A\propto M_Av_k" in joined
    assert r"m_A^\pm" not in joined
    assert r"q_-X_-^2+q_+X_+^2" not in joined
    assert "double plane collapses to one unique projective plane" in habit.reasoning


def test_exact_cmc_state_does_not_display_approximate_distance_formula():
    assessment = build_cayron_martensite_assessment(
        _Response(exact_am=True),
        _Unified(rows=()),
        closing_gap_requested=False,
        supercompatibility_requested=False,
    )
    step = _step(assessment, "nearest")
    assert step.status == "exact reached"
    assert step.formulae == ()
    assert "exactly CMC-degenerate" in step.question
    assert "could incorrectly make an exact state look approximate" in step.reasoning


def test_mm_type_ii_display_contains_complete_plane_direction_and_shear_construction():
    assessment = build_cayron_martensite_assessment(
        _Response(exact_am=False),
        _Unified(rows=(_Row(_EnumLike("ct_mm_twin"), "CT twin II"),)),
        closing_gap_requested=False,
        supercompatibility_requested=False,
    )
    step = _step(assessment, "mm_twins")
    joined = "\n".join(step.formulae)
    assert r"a_M=C\,a_A,\qquad p_M=M_Ma_M" in joined
    assert r"C_{\mathrm{int}}^{\ast}=C_{\mathrm{int}}^{-T}" in joined
    assert r"jp_M=-(C_{\mathrm{int}}^{\ast}-I)p_M" in joined
    assert r"\mathrm{Type~II~twin~system}:\quad(jp_M,a_M)" in joined


def test_exact_supercompatibility_is_decided_from_native_residual_not_row_exact():
    assessment = build_cayron_martensite_assessment(
        _Response(exact_am=True),
        _Unified(
            rows=(
                _Row(_EnumLike("ct_mm_twin"), "CT twin"),
                _Row(
                    _EnumLike("ct_am_habit"),
                    "exact habit",
                    exact=True,
                    shape_vector_parent_crystal=(0.1, 0.0, 0.0),
                ),
                # Deliberately exact=False: the final verdict must come from
                # ct_supercompatibility_dimensionless, not row.exact.
                _Row(
                    _EnumLike("ct_supercompatibility"),
                    "super branch",
                    exact=False,
                    residuals={"ct_supercompatibility_dimensionless": 5.0e-12},
                ),
            )
        ),
        closing_gap_requested=False,
        supercompatibility_requested=True,
    )
    step = _step(assessment, "supercompatibility")
    assert step.status == "reached"
    assert "1/1" in step.answer


def test_nonzero_supercompatibility_residual_is_not_misclassified_as_exact():
    assessment = build_cayron_martensite_assessment(
        _Response(exact_am=True),
        _Unified(
            rows=(
                _Row(_EnumLike("ct_mm_twin"), "CT twin"),
                _Row(
                    _EnumLike("ct_am_habit"),
                    "exact habit",
                    exact=True,
                    shape_vector_parent_crystal=(0.1, 0.0, 0.0),
                ),
                # Deliberately exact=True to reproduce the old presentation bug.
                _Row(
                    _EnumLike("ct_supercompatibility"),
                    "C1-like incompatible branch",
                    exact=True,
                    residuals={"ct_supercompatibility_dimensionless": 0.21515},
                ),
            )
        ),
        closing_gap_requested=False,
        supercompatibility_requested=True,
    )
    step = _step(assessment, "supercompatibility")
    assert step.status == "not reached"
    assert "0/1" in step.answer
    assert "0.21515" in step.answer


def test_requested_but_uncalculated_is_not_reported_as_calculated_failure():
    assessment = build_cayron_martensite_assessment(
        _Response(exact_am=False),
        None,
        closing_gap_requested=True,
        supercompatibility_requested=True,
    )
    assert assessment.overall_status == "incomplete CT inventory"
    assert _step(assessment, "mm_twins").status == "not evaluated"
    assert _step(assessment, "closing_gap").status == "not evaluated"
    assert _step(assessment, "supercompatibility").status == "not evaluable"


def test_requested_and_calculated_zero_closing_gap_is_not_relabelled_not_evaluated():
    assessment = build_cayron_martensite_assessment(
        _Response(exact_am=False),
        _Unified(rows=()),
        closing_gap_requested=True,
        supercompatibility_requested=False,
    )
    step = _step(assessment, "closing_gap")
    assert step.status == "not reached"
    assert "requested and evaluated" in step.answer


def test_third_order_exact_cmc_is_not_forced_to_unique_habit():
    response = _Response(exact_am=True)
    response.result["ct_detail"]["degeneracy_order"] = 3
    response.result["summary"]["degeneracy_order"] = 3
    response.result["ct_detail"]["reason"] = (
        "third-order degeneracy: pulled-back martensite metric equals parent metric"
    )
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


def test_assessment_contains_explicit_equations_and_no_paper_equation_number_dependency():
    root = Path(__file__).resolve().parents[2]
    contract = (root / "app" / "cayron_martensite_assessment.py").read_text(
        encoding="utf-8"
    )
    renderer = (root / "app" / "research_workspaces_v10.py").read_text(
        encoding="utf-8"
    )
    page = (root / "app" / "pages" / "2_CT_Equivalence_Lab.py").read_text(
        encoding="utf-8"
    )

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

    assert "CMC=C^T M_M C-M_A" in contract
    assert "SMC=M_A^{-1}-C^{-1}M_M^{-1}C^{-T}" in contract
    assert "2(m_A^Tn)d_A=a" in contract
    assert r"C_{\mathrm{int}}=C\,G_A\,C^{-1}" in contract
    assert r"p_M=C^{-T}p_A" in contract
    assert r"a_M=C\,a_A" in contract
    assert r"p_M=M_Ma_M" in contract
    assert r"jp_M=-(C_{\mathrm{int}}^{\ast}-I)p_M" in contract

    # No prose dependency on paper equation numbers.
    for token in (
        "Cayron 2026 Eq.",
        "Eq. (",
        "Eqs. (",
        "equation (",
        "equations (",
    ):
        assert token not in contract

    assert "research_workspaces_v12" in page

    renderer_tree = ast.parse(renderer)
    assert any(
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and isinstance(node.targets[0].value, ast.Name)
        and node.targets[0].value.id == "v4"
        and node.targets[0].attr == "_render_conclusions"
        and isinstance(node.value, ast.Name)
        and node.value.id == "_render_conclusions_v10"
        for node in ast.walk(renderer_tree)
    )


def test_v10_notation_guide_distinguishes_T_from_polar_R():
    root = Path(__file__).resolve().parents[2]
    renderer = (root / "app" / "research_workspaces_v10.py").read_text(
        encoding="utf-8"
    )
    assert "T_(A→M) [Cayron]" in renderer
    assert "R or R_F" in renderer
    assert "C_(M←A) [app]" in renderer
    assert "Same numerical matrix as Cayron C^(M→A)" in renderer
    assert "T/R(M←A)" not in renderer


def test_v6_compatibility_audit_suppresses_approximate_planes_for_exact_ct_states():
    root = Path(__file__).resolve().parents[2]
    renderer = (root / "app" / "streamlit_workstation_v6.py").read_text(
        encoding="utf-8"
    )
    assert "def _show_approximate_ct_planes" in renderer
    assert 'bool(ct.get("exact_compatible", False))' in renderer
    assert 'if _show_approximate_ct_planes(ct):' in renderer
    assert 'columns=["m₁", "m₂", "m₃"]' in renderer
    assert "m_A and −m_A represent the same physical plane" in renderer


def test_third_order_smc_is_exact_zero_limit_not_missing_shear():
    response = _Response(exact_am=True)
    response.result["ct_detail"]["degeneracy_order"] = 3
    response.result["summary"]["degeneracy_order"] = 3
    response.result["ct_detail"]["reason"] = (
        "third-order degeneracy: pulled-back martensite metric equals parent metric"
    )
    response.result["ct_detail"]["exact_habit_planes_parent_covectors"] = []
    response.result["metric"]["smc_dimensional"] = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]

    assessment = build_cayron_martensite_assessment(
        response,
        _Unified(rows=()),
        closing_gap_requested=False,
        supercompatibility_requested=False,
    )
    smc = _step(assessment, "smc")
    assert smc.status == "exact trivial limit"
    assert "SMC = 0" in smc.answer
    assert "d_A = 0" in smc.answer
    formulas = "\n".join(smc.formulae)
    assert r"SMC=M_A^{-1}-C^{-1}M_M^{-1}C^{-T}=0" in formulas
    assert r"d_A=SMC\,m_A=0" in formulas


def test_third_order_zero_mm_rows_are_evaluated_not_reported_missing():
    response = _Response(exact_am=True)
    response.result["ct_detail"]["degeneracy_order"] = 3
    response.result["summary"]["degeneracy_order"] = 3
    response.result["ct_detail"]["exact_habit_planes_parent_covectors"] = []

    assessment = build_cayron_martensite_assessment(
        response,
        _Unified(rows=()),
        closing_gap_requested=False,
        supercompatibility_requested=False,
    )
    twins = _step(assessment, "mm_twins")
    assert twins.status == "evaluated zero branches"
    assert "evaluated" in twins.answer
    assert "zero nontrivial" in twins.answer
    assert "not a missing calculation" in twins.answer


def test_degenerate_limit_presentation_layers_do_not_conflate_branch_counts_with_existence():
    root = Path(__file__).resolve().parents[2]
    workstation = (root / "app" / "streamlit_workstation_v6.py").read_text(encoding="utf-8")
    v12 = (root / "app" / "research_workspaces_v12.py").read_text(encoding="utf-8")
    page = (root / "app" / "pages" / "2_CT_Equivalence_Lab.py").read_text(encoding="utf-8")

    assert "_topology_finding_phase3" in workstation
    assert "correspondence/topological variant" in workstation
    assert "distinct metric stretch variant" in workstation

    assert "Branch count is therefore not used as an existence flag" in v12
    assert "U = I" in v12
    assert "evaluated zero-result" in v12
    assert "Rerunning unchanged inputs" in v12
    assert "missing prerequisite" in v12
    assert "research_workspaces_v12" in page
