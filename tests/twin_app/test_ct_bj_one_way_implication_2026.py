"""Check only the physically assigned, one-way Cayron CT => cofactor claim."""

from types import SimpleNamespace as NS

from cualni_cryst.theory_unified import PredictionKind
from twin_app.theory_evidence_ui import ct_bj_same_twin_implication_evidence


def check(eps=0.0, cc=True, assigned=True):
    row_ct = NS(row_id="ct_twin_0_0", prediction_kind=PredictionKind.CT_MM_TWIN,
                metadata={"operator_index": 0, "twin_index": 0,
                          "twin_classification": "Compound"}, residuals={})
    row_super = NS(row_id="ct_super_0_0_0",
                   prediction_kind=PredictionKind.CT_SUPERCOMPATIBILITY,
                   metadata={"operator_index": 0, "twin_index": 0},
                   residuals={"ct_supercompatibility_dimensionless": eps})
    row_bj = NS(row_id="bj_mm_0", prediction_kind=PredictionKind.BALL_JAMES_MM,
                metadata={"cofactor_all_satisfied": cc},
                residuals={"cc1_lambda2_minus_one": 0.0, "cc2": 0.0,
                           "cc3_margin": 0.01})
    def kind(name):
        return NS(value=name)
    match = NS(family=kind("martensite_martensite_twin"),
               left_theory=kind("cayron_ct"), right_theory=kind("ball_james"),
               left_row_id="ct_twin_0_0", right_row_id="bj_mm_0",
               all_required_components_within_tolerance=assigned)
    rep = NS(rows=(row_ct, row_super, row_bj))
    eq = NS(matches=(match,))
    return ct_bj_same_twin_implication_evidence(rep, eq)[0]


def test_exact_ct_hypotheses_and_valid_bj_cofactor_are_consistent():
    result = check()
    assert result["One-way implication status"] == "CONSISTENT_WITH_CT_IMPLIES_COFACTOR"


def test_if_ct_is_nonzero_no_converse_theorem_is_inferred():
    result = check(eps=0.215, cc=True)
    assert result["One-way implication status"] == "CT_HYPOTHESIS_FALSE_NO_THEOREM_TEST"


def test_partial_twin_assignment_never_proves_theorem():
    result = check(assigned=None)
    assert result["One-way implication status"] == "INCOMPLETE_PHYSICAL_ASSIGNMENT"


def test_contradiction_is_reported_not_hidden():
    result = check(eps=0.0, cc=False)
    assert result["One-way implication status"] == "NUMERICAL_CONTRADICTION_REVIEW_FRAMES_AND_TOLERANCES"
