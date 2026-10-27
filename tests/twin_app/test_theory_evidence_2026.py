"""Independent evidence-layer regressions; physics engines remain untouched."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from cualni_cryst.theory_unified import PredictionKind
from twin_app.theory_evidence_ui import (
    ball_james_branch_evidence,
    ct_supercompatibility_evidence,
    assigned_interface_evidence,
    evidence_report,
)


def fixture(tampered=False):
    base = np.eye(3)
    other = np.diag([1.1, 1.0, 1.0])
    a, n = np.array([0.1, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    # CC3 = -0.0025: the rank-one equation is exact but the all-f
    # supercompatibility inequality is NOT satisfied.
    c = NS(cc1_lambda2_minus_one=0.0, cc2_value=0.0,
           cc3_margin=-0.0025, all_satisfied=False)
    r = np.eye(3)
    if tampered:
        r[0, 0] = 1.02
    t = NS(base_variant_index=0, other_variant_index=1, branch=0,
           rotation=r, a=a, n_reference=n, cofactor=c)
    bj = NS(variants=(NS(index=0, U=base), NS(index=1, U=other)),
            martensite_twin_branches=(t,))
    return NS(ball_james_report=bj, rows=())


def test_correct_rank_one_does_not_imply_supercompatibility():
    result = ball_james_branch_evidence(fixture())
    assert len(result) == 1
    row = result[0]
    assert row["Rank-one residual"] < 1e-14
    assert row["Geometric jump verified"] is True
    assert row["CC quantities reproduced"] is True
    assert row["Independent CC3 margin"] == pytest.approx(-0.0025)
    assert row["Stored CC1–CC3 pass"] is False
    assert row["Audit status"] == "REPRODUCED"


def test_corrupted_rotation_is_not_silently_certified():
    row = ball_james_branch_evidence(fixture(tampered=True))[0]
    assert row["Audit status"] == "REVIEW_REQUIRED"
    assert row["Geometric jump verified"] is False


def test_tampered_cofactor_is_detected_independently():
    report = fixture()
    report.ball_james_report.martensite_twin_branches[0].cofactor.cc3_margin = 0.0
    row = ball_james_branch_evidence(report)[0]
    assert row["CC quantities reproduced"] is False
    assert row["Audit status"] == "REVIEW_REQUIRED"


def test_missing_variant_reference_fails_closed():
    report = fixture()
    report.ball_james_report.martensite_twin_branches[0].other_variant_index = 11
    with pytest.raises(ValueError, match="does not exist"):
        ball_james_branch_evidence(report)


def test_ct_epsilon_is_preserved_without_inventing_a_percentage():
    def row(index, epsilon):
        return NS(prediction_kind=PredictionKind.CT_SUPERCOMPATIBILITY,
                  branch_label=f"pair-{index}",
                  residuals={"ct_supercompatibility_dimensionless": epsilon},
                  metadata={"operator_index": index, "twin_index": 0,
                            "habit_index": 0, "twin_classification": "Type I"},
                  habit_plane_parent_crystal=(1, 0, 0),
                  twin_plane_parent_crystal=(0, 0, 1),
                  twin_direction_parent_crystal=(0, 1, 0),
                  shear_magnitude=0.27)
    report = NS(rows=(row(0, 0.215), row(1, 0.003)))
    out = ct_supercompatibility_evidence(report)
    assert [r["Dimensionless epsilon"] for r in out] == [0.003, 0.215]
    assert all("mismatch" in r["Interpretation"] for r in out)


def test_partial_match_remains_unproven_and_provenance_kept():
    def theory(value):
        return NS(value=value)
    left = NS(row_id="ct1", provenance="Cayron CMC/SMC",
              prediction_kind=theory("ct_am_habit"))
    right = NS(row_id="bj1", provenance="Ball-James M/M",
               prediction_kind=theory("ball_james_mm"))
    unified = NS(rows=(left, right))
    residuals = NS(to_dict=lambda: {"rank_one_tensor_relative": None,
                                    "twin_plane_angle_deg": 0.02})
    pair = NS(left_row_id="ct1", right_row_id="bj1",
              family=theory("martensite_martensite_twin"),
              left_theory=theory("cayron_ct"), right_theory=theory("ball_james"),
              residuals=residuals, required_components=("rank_one_tensor_relative",),
              missing_required_components=("rank_one_tensor_relative",),
              within_tolerance={}, all_required_components_within_tolerance=None)
    result = assigned_interface_evidence(unified, NS(matches=(pair,)))
    assert result[0]["Agreement"] == "INCOMPLETE_UNPROVEN"
    assert result[0]["Observable residuals"]["rank_one_tensor_relative"] is None
    assert result[0]["A provenance"] == "Cayron CMC/SMC"


def test_full_report_retains_all_scope_warnings():
    result = evidence_report(fixture(), NS(matches=()))
    assert result["BJ branches requiring review"] == 0
    assert len(result["BJ independent branch tensor checks"]) == 1
    assert any("Cu-Al-Ni" in item for item in result["audit_contract"]["limits"])
    assert any("not an all-f proof" in item for item in result["audit_contract"]["limits"])
