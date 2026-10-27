"""Cayron 2026 NiTi-only published analytical checks with source ambiguities."""

import math

import pytest

from twin_app.cayron_2026_reference import (
    appendix_c_o4_at_beta_98,
    kudoh_1996_niti_source_case,
    niti_b19prime_cmc_domains,
    o2_printed_vs_equation_derived,
)


def test_kudoh_table_2_c1_eq_ineq_and_others():
    result = {r["Domain"]: r for r in kudoh_1996_niti_source_case()}
    assert result["C1"]["Equality residual"] == pytest.approx(0.137364, abs=1e-6)
    assert result["C1"]["Inequality distance"] == pytest.approx(0.068402, abs=1e-6)
    assert result["C1"]["Inequality satisfied"] is True
    assert result["C2a"]["Inequality distance"] == pytest.approx(0.227385, abs=1e-6)
    assert result["C2b"]["Equality residual"] == pytest.approx(0.091359, abs=1e-6)
    assert result["C3"]["Inequality distance"] == pytest.approx(0.210399, abs=1e-6)
    assert result["C3"]["Alternative |c−target|"] == pytest.approx(0.269706, abs=1e-6)


def test_c2b_source_conflict_is_preserved_explicitly():
    row = kudoh_1996_niti_source_case()[2]
    assert row["Inequality satisfied"] is None
    assert row["Eq47 inequality satisfied"] is True
    assert row["Table2 inequality satisfied"] is False
    assert "SOURCE_CONFLICT" in row["Source consistency"]


def test_ideal_c1_metric_has_zero_equality_gap():
    rows = niti_b19prime_cmc_domains(.96279069767, math.sqrt(2), 4.646/3.01, 97.78)
    assert rows[0]["Equality residual"] < 1e-13
    assert rows[0]["Inequality satisfied"]


def test_printed_o2_cci_discrepancy_is_not_called_validated():
    rows = o2_printed_vs_equation_derived(98.0)
    assert rows[0]["SMC d_z"] == .5
    assert rows[0]["Type I ||U^-1 e_001||²"] == .5
    assert rows[1]["SMC d_z"] == 0
    assert rows[1]["Type I ||U^-1 e_001||²"] == 1
    assert "NOT AUTHOR-CONFIRMED" in rows[1]["Provenance"]


def test_appendix_c_benchmark_reproduced_from_analytic_functions():
    r = appendix_c_o4_at_beta_98()
    assert r["a"] == pytest.approx(.8825, abs=2e-5)
    assert r["c"] == pytest.approx(1.6182, abs=4e-5)
    assert r["b"] == pytest.approx(math.sqrt(2))


@pytest.mark.parametrize("args", [(0,1,1,98), (1,-1,1,98),
                                  (1,1,1,180), (1,1,1,float("nan"))])
def test_invalid_metric_input_is_not_accepted(args):
    with pytest.raises(ValueError):
        niti_b19prime_cmc_domains(*args)
