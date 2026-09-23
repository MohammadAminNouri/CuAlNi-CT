from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "validation/independent/audit_cayron2026_independent_v1.py"
BENCHMARK = ROOT / "data/benchmarks/cayron_2026_o2_type_i_source_audit_v1.json"

EXPECTED_AUDIT_LINES = (
    "1) Core paper correspondence/SMC transcription reproduces Table 3: PASS",
    "2) Printed O2 Type-I c=sqrt(2), a=c/sin(beta): FAILS Eq.43 exactly",
    "3) The same printed O2 Type-I family: FAILS independent CCI exactly (1/2 vs 1)",
    "4) O2 Type-II analytical family from the paper: PASS",
    "5) O4 Appendix-C analytical family from the paper: PASS",
    "6) O2 Type-I algebra closes for b=c=sqrt(2), a=1/sin(beta)",
    "7) Diagnosis is invariant to common length-unit scaling: PASS",
)


# Paper's correspondence matrices, transcribed independently of cualni_cryst.
C_MA_NUM = np.array([
    [0.0, 0.0, 1.0],
    [0.5, 0.5, 0.0],
    [-0.5, 0.5, 0.0],
])
C_AM_NUM = np.linalg.inv(C_MA_NUM)


def _metric(a: float, b: float, c: float, beta_deg: float) -> np.ndarray:
    beta = math.radians(beta_deg)
    return np.array([
        [a*a, 0.0, a*c*math.cos(beta)],
        [0.0, b*b, 0.0],
        [a*c*math.cos(beta), 0.0, c*c],
    ])


def _independent_dz_and_cci(beta_deg: float, a: float) -> tuple[float, float]:
    """Direct matrix evaluation, no CuAlNi-CT imports and no closed-form shortcut."""
    MM = _metric(a, math.sqrt(2.0), math.sqrt(2.0), beta_deg)
    SMC = np.eye(3) - C_AM_NUM @ np.linalg.inv(MM) @ C_AM_NUM.T
    d = SMC @ np.array([0.0, 0.0, 1.0])
    U2 = C_MA_NUM.T @ MM @ C_MA_NUM
    cci_sq = float(np.linalg.inv(U2)[2, 2])
    return float(d[2]), cci_sq


def test_independent_source_audit_and_positive_controls_execute_cleanly():
    """The external audit is a required independent oracle, not an optional note."""
    assert AUDIT.is_file(), f"missing independent audit: {AUDIT}"
    proc = subprocess.run(
        [sys.executable, str(AUDIT)],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout
    for line in EXPECTED_AUDIT_LINES:
        assert line in proc.stdout, proc.stdout


def test_o2_type_i_source_discrepancy_is_an_exact_symbolic_identity():
    """Prove the discrepancy algebraically rather than by numerical tolerance."""
    beta = sp.symbols("beta", positive=True, real=True)
    a, b, c = sp.symbols("a b c", positive=True, real=True)

    C_MA = sp.Matrix([
        [0, 0, 1],
        [sp.Rational(1, 2), sp.Rational(1, 2), 0],
        [-sp.Rational(1, 2), sp.Rational(1, 2), 0],
    ])
    C_AM = C_MA.inv()
    MM = sp.Matrix([
        [a**2, 0, a*c*sp.cos(beta)],
        [0, b**2, 0],
        [a*c*sp.cos(beta), 0, c**2],
    ])

    SMC = sp.eye(3) - C_AM * MM.inv() * C_AM.T
    d = sp.simplify(sp.trigsimp(SMC * sp.Matrix([0, 0, 1])))
    dz = sp.simplify(sp.trigsimp(d[2]))
    assert sp.simplify(
        sp.trigsimp(dz - (1 - 1/(a**2 * sp.sin(beta)**2)))
    ) == 0

    U2 = sp.simplify(C_MA.T * MM * C_MA)
    cci_sq = sp.simplify(sp.trigsimp(U2.inv()[2, 2]))
    assert sp.simplify(
        sp.trigsimp(cci_sq - 1/(a**2 * sp.sin(beta)**2))
    ) == 0

    printed = sp.sqrt(2) / sp.sin(beta)
    derived = 1 / sp.sin(beta)

    assert sp.simplify(sp.trigsimp(dz.subs(a, printed))) == sp.Rational(1, 2)
    assert sp.simplify(sp.trigsimp(cci_sq.subs(a, printed))) == sp.Rational(1, 2)
    assert sp.simplify(sp.trigsimp(dz.subs(a, derived))) == 0
    assert sp.simplify(sp.trigsimp(cci_sq.subs(a, derived))) == 1


def test_o2_type_i_dense_beta_family_regression_is_not_a_single_point_accident():
    """Numerically cross-check the symbolic identity across the full useful family."""
    for beta_deg in np.linspace(90.1, 119.9, 599):
        s = math.sin(math.radians(float(beta_deg)))

        printed_a = math.sqrt(2.0) / s
        printed_dz, printed_cci = _independent_dz_and_cci(beta_deg, printed_a)
        np.testing.assert_allclose(printed_dz, 0.5, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(printed_cci, 0.5, atol=2e-12, rtol=2e-12)

        derived_a = 1.0 / s
        derived_dz, derived_cci = _independent_dz_and_cci(beta_deg, derived_a)
        np.testing.assert_allclose(derived_dz, 0.0, atol=3e-12, rtol=0.0)
        np.testing.assert_allclose(derived_cci, 1.0, atol=3e-12, rtol=0.0)


def test_source_and_project_derived_provenance_are_never_conflated():
    data = json.loads(BENCHMARK.read_text(encoding="utf-8"))

    assert data["printed_o2_type_i"]["status"] == "SOURCE_INTERNAL_INCONSISTENCY"
    assert data["printed_o2_type_i"]["relation"] == "c=sqrt(2); a=c/sin(beta)"
    assert data["printed_o2_type_i"]["eq43_dz"] == "1/2"
    assert data["printed_o2_type_i"]["type_i_cci_squared"] == "1/2"

    assert data["equation_derived_closure"]["status"] == "PROJECT_DERIVED_RESULT"
    assert data["equation_derived_closure"]["relation"] == "c=sqrt(2); a=1/sin(beta)=c/(sqrt(2)*sin(beta))"
    assert data["equation_derived_closure"]["eq43_dz"] == "0"
    assert data["equation_derived_closure"]["type_i_cci_squared"] == "1"

    controls = data["positive_controls"]
    assert controls["table_3"]["status"] == "VERIFIED_PUBLISHED_RESULT"
    assert controls["o2_type_ii"]["status"] == "VERIFIED_PUBLISHED_RESULT"
    assert controls["o4_appendix_c"]["status"] == "VERIFIED_PUBLISHED_RESULT"

    assert data["interpretation_policy"] == (
        "Do not silently alter the printed paper relation and do not attribute "
        "the project-derived closure to the author without author confirmation."
    )
