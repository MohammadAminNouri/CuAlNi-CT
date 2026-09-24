#!/usr/bin/env python3
from __future__ import annotations

"""
Final non-mutating scientific/application readiness gate for CuAlNi-CT.

Purpose
-------
Run only AFTER the V3.1 numerical core has been committed to
science/universal-validation-recovered.

This gate does not repair, rewrite, normalize, or loosen any scientific
equation/tolerance.  It attacks the committed code through independent
identities, exact-binary64 certification, unit/basis laws, application-service
smoke tests, CLI/package checks, and known-regression source guards.

Any failure is evidence to inspect.  A failure is never made green by widening
a scientific tolerance.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import mpmath as mp
import numpy as np
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
REPORT_DEFAULT = Path("/tmp/cualni_final_scientific_app_readiness.json")


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def _run(cmd: list[str], *, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def _git(*args: str) -> str:
    cp = _run(["git", *args], timeout=60)
    if cp.returncode:
        raise RuntimeError(cp.stdout)
    return cp.stdout.strip()


def _finite_tree(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(v) for v in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _rational_binary64(x: float) -> sp.Rational:
    n, d = float(x).as_integer_ratio()
    return sp.Rational(n, d)


def _sp_matrix_binary64(A: np.ndarray) -> sp.Matrix:
    A = np.asarray(A, dtype=float)
    return sp.Matrix([[_rational_binary64(x) for x in row] for row in A])


def _mpf_binary64(x: float) -> mp.mpf:
    n, d = float(x).as_integer_ratio()
    return mp.mpf(n) / mp.mpf(d)


def _mp_matrix_binary64(A: np.ndarray) -> mp.matrix:
    A = np.asarray(A, dtype=float)
    return mp.matrix([[_mpf_binary64(x) for x in row] for row in A])


def _canonical_metric(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float).reshape(3, 3)
    return 0.5 * (A + A.T)


def _certified_mu(Ma: np.ndarray, Mm: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Certified roots of det(C.T Mm C - mu Ma)=0 for the exact binary64 inputs."""
    Ma = _canonical_metric(Ma)
    Mm = _canonical_metric(Mm)
    C = np.asarray(C, dtype=float).reshape(3, 3)
    x = sp.symbols("mu")
    A = _sp_matrix_binary64(Ma)
    M = _sp_matrix_binary64(Mm)
    X = _sp_matrix_binary64(C)
    poly = sp.Poly(sp.expand((X.T * M * X - x * A).det()), x)
    intervals = sp.polys.polytools.intervals(
        poly, eps=sp.Rational(1, 10**32)
    )
    roots: list[float] = []
    for (lo, hi), multiplicity in intervals:
        roots.extend([float((lo + hi) / 2)] * int(multiplicity))
    roots.sort()
    if len(roots) != 3:
        raise AssertionError(f"expected 3 certified real roots, got {intervals}")
    result = np.asarray(roots, dtype=float)
    if np.min(result) <= 0.0:
        raise AssertionError(f"generalized metric spectrum is not positive: {result}")
    return result


def _classify_mu(mu: np.ndarray, tol: float) -> tuple[bool, int]:
    eta = np.sort(np.asarray(mu, dtype=float)) - 1.0
    zero = np.abs(eta) <= tol
    nzero = int(np.sum(zero))
    if nzero == 3:
        return True, 3
    if nzero == 2:
        return True, 2
    if nzero == 1:
        others = eta[~zero]
        ok = bool(others[0] * others[1] < 0.0)
        return ok, 1 if ok else 0
    return False, 0


def _mp_smc_direct(Ma: np.ndarray, Mm: np.ndarray, C: np.ndarray, digits: int = 100) -> np.ndarray:
    """Independent direct Cayron expression Ma^-1 - C^-1 Mm^-1 C^-T."""
    Ma = _canonical_metric(Ma)
    Mm = _canonical_metric(Mm)
    C = np.asarray(C, dtype=float).reshape(3, 3)
    with mp.workdps(digits):
        A = _mp_matrix_binary64(Ma)
        M = _mp_matrix_binary64(Mm)
        X = _mp_matrix_binary64(C)
        S = (A ** -1) - (X ** -1) * (M ** -1) * ((X.T) ** -1)
        arr = np.array(
            [[float(S[i, j]) for j in range(3)] for i in range(3)],
            dtype=float,
        )
    return 0.5 * (arr + arr.T)


def _rel(A: np.ndarray, B: np.ndarray) -> float:
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    den = max(float(np.linalg.norm(B, ord="fro")), np.finfo(float).tiny)
    return float(np.linalg.norm(A - B, ord="fro") / den)


def _projective_covector_distance(p: np.ndarray, q: np.ndarray, M: np.ndarray) -> float:
    """Sign-insensitive metric distance between reciprocal covectors."""
    p = np.asarray(p, float).reshape(3)
    q = np.asarray(q, float).reshape(3)
    M = np.asarray(M, float).reshape(3, 3)
    Mi_p = np.linalg.solve(M, p)
    Mi_q = np.linalg.solve(M, q)
    p = p / math.sqrt(float(p @ Mi_p))
    q = q / math.sqrt(float(q @ Mi_q))
    if float(p @ np.linalg.solve(M, q)) < 0.0:
        q = -q
    d = p - q
    return math.sqrt(max(0.0, float(d @ np.linalg.solve(M, d))))


def _integer_unimodular_bank() -> tuple[np.ndarray, ...]:
    mats = [
        np.eye(3, dtype=float),
        np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], float),
        np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]], float),
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float),
        np.array([[1, 1, 0], [1, 2, 0], [0, 0, 1]], float),
        np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]], float),
    ]
    for M in mats:
        d = round(float(np.linalg.det(M)))
        if abs(d) != 1:
            raise AssertionError(f"internal basis bank contains non-unimodular matrix: det={d}")
    return tuple(mats)


def _controlled_pencil(index: int, delta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rational-ish SPD pencil with known generalized eigenvalues."""
    bank = _integer_unimodular_bank()
    P = bank[index % len(bank)]
    k = (index % 7) - 3
    d = np.array([2.0 ** (-2 * k), 1.0, 2.0 ** (2 * k)], float)
    mu = np.array([0.58 + 0.01 * (index % 3), 1.0 + delta, 1.72 + 0.02 * (index % 5)])
    Ma = P.T @ np.diag(d) @ P
    G = P.T @ np.diag(d * mu) @ P

    # Nontrivial exact unimodular correspondence in a subset of cases.
    C = bank[(index + 2) % len(bank)]
    Ci_exact = np.asarray(sp.Matrix(C.astype(int)).inv(), dtype=float)
    Mm = Ci_exact.T @ G @ Ci_exact
    return _canonical_metric(Ma), _canonical_metric(Mm), C


def check_v31_surface() -> str:
    required = [
        ROOT / "src/cualni_cryst/adaptive_metric.py",
        ROOT / "validation/engineered/certified_binary64.py",
        ROOT / "docs/NUMERICAL_CORE_V3.md",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise AssertionError(
            "V3.1 committed core is not present yet; missing " + ", ".join(missing)
        )
    return "required V3.1 production/oracle/documentation files are present"


def check_known_regression_patterns() -> str:
    paths = {
        "lattice": ROOT / "src/cualni_cryst/lattice.py",
        "ct": ROOT / "src/cualni_cryst/ct.py",
        "ptmc": ROOT / "src/cualni_cryst/ptmc.py",
        "ball_james": ROOT / "src/cualni_cryst/ball_james.py",
        "hp": ROOT / "validation/engineered/high_precision.py",
        "production": ROOT / "validation/engineered/production.py",
    }
    text = {k: p.read_text(encoding="utf-8") for k, p in paths.items()}

    forbidden: list[tuple[str, str, str]] = [
        ("lattice", "from scipy.linalg import sqrtm", "generic sqrtm restored in SPD metric kernel"),
        ("lattice", "sqrtm(", "generic sqrtm restored in SPD metric kernel"),
        ("hp", "Rational(str(float", "decimal-string high-precision oracle restored"),
        ("production", "Rational(str(float", "decimal-string production adapter restored"),
        ("ct", "qi*qj > tol", "eigenvalue-product/tolerance dimensional bug restored"),
        ("ct", "qi * qj > tol", "eigenvalue-product/tolerance dimensional bug restored"),
        ("ptmc", "<=1e-6 and not any(abs(f-r)<1e-7", "old hidden PTMC acceptance widening restored"),
        ("ptmc", "if abs(c2)>1e-14", "old absolute PTMC degree cutoff restored"),
        ("ptmc", "if abs(z.imag)>1e-8", "old absolute PTMC complex-root cutoff restored"),
    ]
    hits = []
    for key, needle, reason in forbidden:
        if needle in text[key]:
            hits.append(f"{paths[key].relative_to(ROOT)}: {reason}: {needle!r}")
    if hits:
        raise AssertionError("\n".join(hits))

    return "previously identified numerical anti-patterns are absent"


def check_compileall() -> str:
    cp = _run(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "src",
            "validation",
            "tests",
            "tools",
        ],
        timeout=180,
    )
    if cp.returncode:
        raise AssertionError(cp.stdout)
    return "Python source tree compiles"


def check_literal_cmc() -> str:
    from cualni_cryst.correspondence import Correspondence
    from cualni_cryst.ct import cmc

    bank = _integer_unimodular_bank()
    worst = 0.0
    for i in range(64):
        Ma, Mm, C = _controlled_pencil(i, delta=((i % 9) - 4) * 2.0e-9)
        corr = Correspondence(sp.Matrix(C.astype(int)))
        expected = C.T @ Mm @ C - Ma
        got = cmc(Ma, Mm, corr)
        if not np.array_equal(got, expected):
            worst = max(worst, float(np.max(np.abs(got - expected))))
            raise AssertionError(
                "cmc() is not the literal binary64 evaluation of "
                "C.T @ M_m @ C - M_a; max abs discrepancy="
                f"{np.max(np.abs(got-expected)):.17e}"
            )
    return f"64 literal CMC identities passed; worst discrepancy={worst:.1e}"


def check_certified_adaptive_classification() -> str:
    from cualni_cryst.adaptive_metric import adaptive_metric_eigensystem
    from cualni_cryst.numerics import DEFAULT_NUMERICAL_POLICY

    tol = float(DEFAULT_NUMERICAL_POLICY.exact_eigenvalue)
    deltas = [
        -4.0 * tol,
        -1.2 * tol,
        -0.8 * tol,
        -0.2 * tol,
        0.0,
        0.2 * tol,
        0.8 * tol,
        1.2 * tol,
        4.0 * tol,
    ]
    n = 0
    escalated = 0
    worst_mu = 0.0
    for i in range(36):
        Ma, Mm, C = _controlled_pencil(i, deltas[i % len(deltas)])
        certified = _certified_mu(Ma, Mm, C)
        result = adaptive_metric_eigensystem(Ma, Mm, C, decision_tol=tol)
        got_class = _classify_mu(result.mu, tol)
        ref_class = _classify_mu(certified, tol)
        if got_class != ref_class:
            raise AssertionError(
                f"classification mismatch i={i}: production={got_class}, "
                f"certified={ref_class}, production_mu={result.mu}, certified_mu={certified}"
            )
        worst_mu = max(worst_mu, float(np.max(np.abs(result.mu - certified))))
        escalated += int(bool(result.escalated))
        n += 1
    return (
        f"{n} fresh exact-binary64 classification cases agree; "
        f"escalations={escalated}; max |mu-prod-mu-cert|={worst_mu:.3e}"
    )


def check_generalized_vectors_and_habit_planes() -> str:
    from cualni_cryst.adaptive_metric import high_precision_metric_eigensystem
    from cualni_cryst.correspondence import Correspondence
    from cualni_cryst.ct import analyze_cmc, habit_planes_from_cmc

    rng = np.random.default_rng(492771)
    worst_eq = 0.0
    worst_orth = 0.0
    worst_cone = 0.0

    for i in range(24):
        A = rng.integers(-2, 3, size=(3, 3)).astype(float)
        Ma = A.T @ A + np.diag([1.0, 2.0, 4.0])
        Ma = _canonical_metric(Ma)

        # Metric-orthonormal generalized eigenbasis V=S^-1 Q.
        w, Qm = np.linalg.eigh(Ma)
        S = (Qm * np.sqrt(w)) @ Qm.T
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        V = np.linalg.solve(S, Q)

        eta = np.array(
            [
                -(0.08 + 0.01 * (i % 4)),
                0.0,
                0.12 + 0.015 * (i % 5),
            ],
            float,
        )
        mu = 1.0 + eta
        G = Ma @ V @ np.diag(mu) @ V.T @ Ma
        Mm = _canonical_metric(G)
        C = np.eye(3)
        corr = Correspondence(sp.eye(3))

        hp_mu, hp_V = high_precision_metric_eigensystem(Ma, Mm, C, digits=110)
        G_literal = C.T @ Mm @ C
        scale = max(np.linalg.norm(G_literal, "fro"), np.linalg.norm(Ma, "fro"), 1.0)
        eq = float(
            np.linalg.norm(
                G_literal @ hp_V - Ma @ hp_V @ np.diag(hp_mu), "fro"
            )
            / scale
        )
        orth = float(np.linalg.norm(hp_V.T @ Ma @ hp_V - np.eye(3), "fro"))
        worst_eq = max(worst_eq, eq)
        worst_orth = max(worst_orth, orth)
        if eq > 5e-10 or orth > 5e-10:
            raise AssertionError(
                f"high-precision generalized vector certificate failed: eq={eq:.3e}, orth={orth:.3e}"
            )

        ana = analyze_cmc(Ma, Mm, corr, tol=1e-8)
        if not ana.exact_compatible or ana.degeneracy_order != 1:
            raise AssertionError(
                f"synthetic first-order compatible pencil misclassified: {ana}"
            )
        planes = habit_planes_from_cmc(Ma, Mm, corr, tol=1e-8)
        if len(planes) != 2:
            raise AssertionError(f"expected 2 habit planes, got {len(planes)}")

        A_cmc = Mm - Ma
        for p in planes:
            p = np.asarray(p, float).reshape(3)
            null_basis = np.linalg.svd(p.reshape(1, 3))[2][1:, :].T
            # Test basis vectors and nontrivial combinations in the plane.
            candidates = [
                null_basis[:, 0],
                null_basis[:, 1],
                null_basis[:, 0] + 0.37 * null_basis[:, 1],
                -0.23 * null_basis[:, 0] + null_basis[:, 1],
            ]
            for u in candidates:
                num = abs(float(u @ A_cmc @ u))
                den = max(
                    float(np.linalg.norm(A_cmc, "fro")) * float(u @ u),
                    np.finfo(float).tiny,
                )
                cone = num / den
                worst_cone = max(worst_cone, cone)
                if cone > 5e-10:
                    raise AssertionError(
                        f"habit plane does not factor CMC cone: residual={cone:.3e}"
                    )

    return (
        f"24 generalized-vector + habit-plane cases passed; "
        f"max eigen residual={worst_eq:.3e}, metric-orth={worst_orth:.3e}, "
        f"cone residual={worst_cone:.3e}"
    )


def check_smc_definition_and_covariance() -> str:
    from cualni_cryst.correspondence import Correspondence
    from cualni_cryst.ct import smc

    bank = _integer_unimodular_bank()
    worst_oracle = 0.0
    worst_cov = 0.0

    for i in range(24):
        Ma, Mm, C = _controlled_pencil(i, delta=0.15 + 0.01 * (i % 3))
        Cint = C.astype(int)
        corr = Correspondence(sp.Matrix(Cint))
        got = smc(Ma, Mm, corr)
        ref = _mp_smc_direct(Ma, Mm, C, digits=110)
        r = _rel(got, ref)
        worst_oracle = max(worst_oracle, r)
        if r > 2e-11:
            raise AssertionError(
                f"SMC disagrees with independent direct high-precision Cayron expression: {r:.3e}"
            )

        # Exact binary-friendly basis transformations: signed permutation/powers of two.
        Pa = np.diag([2.0 ** ((i % 3) - 1), 2.0, 0.5])
        Pm = bank[(i + 1) % len(bank)]
        Ma2 = Pa.T @ Ma @ Pa
        Mm2 = Pm.T @ Mm @ Pm
        Pmi = np.asarray(sp.Matrix(Pm.astype(int)).inv(), dtype=float)
        C2 = Pmi @ C @ Pa

        # C2 may be dyadic rather than integer: encode exact binary64 rationals for Correspondence.
        C2sp = sp.Matrix(
            [[_rational_binary64(C2[a, b]) for b in range(3)] for a in range(3)]
        )
        got2 = smc(Ma2, Mm2, Correspondence(C2sp))
        expected2 = np.linalg.solve(Pa, got) @ np.linalg.inv(Pa).T
        c = _rel(got2, expected2)
        worst_cov = max(worst_cov, c)
        if c > 2e-10:
            raise AssertionError(
                f"SMC parent-basis covariance failed: residual={c:.3e}"
            )

    return (
        f"24 SMC definition/covariance cases passed; "
        f"max oracle residual={worst_oracle:.3e}, max covariance residual={worst_cov:.3e}"
    )


def check_binary_exact_unit_scaling_laws() -> str:
    from cualni_cryst.correspondence import Correspondence
    from cualni_cryst.ct import cmc, normalized_cmc, smc
    from cualni_cryst.adaptive_metric import adaptive_metric_eigensystem
    from cualni_cryst.numerics import DEFAULT_NUMERICAL_POLICY

    corr = Correspondence(sp.eye(3))
    Ma = np.array(
        [[4.0, 0.5, -0.25], [0.5, 2.0, 0.125], [-0.25, 0.125, 1.5]],
        float,
    )
    Mm = np.array(
        [[4.4, 0.45, -0.22], [0.45, 2.0, 0.14], [-0.22, 0.14, 1.32]],
        float,
    )
    Ma = _canonical_metric(Ma)
    Mm = _canonical_metric(Mm)

    base_cmc = cmc(Ma, Mm, corr)
    base_n = normalized_cmc(Ma, Mm, corr)
    base_s = smc(Ma, Mm, corr)
    base_mu = adaptive_metric_eigensystem(
        Ma, Mm, np.eye(3),
        decision_tol=DEFAULT_NUMERICAL_POLICY.exact_eigenvalue,
    ).mu

    worst = 0.0
    for exponent in (-30, -12, 12, 30):
        length_scale = 2.0 ** exponent
        metric_scale = length_scale * length_scale
        Ma2 = metric_scale * Ma
        Mm2 = metric_scale * Mm

        r1 = _rel(cmc(Ma2, Mm2, corr), metric_scale * base_cmc)
        r2 = _rel(normalized_cmc(Ma2, Mm2, corr), base_n)
        r3 = _rel(smc(Ma2, Mm2, corr), base_s / metric_scale)
        mu2 = adaptive_metric_eigensystem(
            Ma2, Mm2, np.eye(3),
            decision_tol=DEFAULT_NUMERICAL_POLICY.exact_eigenvalue,
        ).mu
        r4 = float(np.max(np.abs(mu2 - base_mu)))
        worst = max(worst, r1, r2, r3, r4)
        if r1 > 5e-13 or r2 > 2e-11 or r3 > 2e-11 or r4 > 2e-11:
            raise AssertionError(
                "binary-exact power-of-two unit scaling law failed: "
                f"k={exponent}, CMC={r1:.3e}, normalized={r2:.3e}, "
                f"SMC={r3:.3e}, mu={r4:.3e}"
            )

    return f"power-of-two length scaling laws passed; worst residual={worst:.3e}"


def check_angle_endpoint_geometry() -> str:
    from cualni_cryst.crystal_objects import (
        CrystalBasisRef,
        Direction,
        Plane,
        axis_angle_deg,
        direction_angle_deg,
        direction_plane_angle_deg,
        interplanar_angle_deg,
        plane_normal_angle_deg,
    )
    from cualni_cryst.lattice import Lattice

    lattice = Lattice(
        3.17, 4.21, 5.08,
        alpha_deg=77.0, beta_deg=103.0, gamma_deg=68.0,
        label="endpoint_triclinic",
    )
    basis = CrystalBasisRef("phase", "basis", "triclinic")
    u = Direction((1.0, -2.0, 3.0), basis)
    um = Direction((-1.0, 2.0, -3.0), basis)
    p = Plane((1.0, 0.0, 1.0), basis)
    pm = Plane((-1.0, 0.0, -1.0), basis)

    values = {
        "dir_self": direction_angle_deg(u, u, lattice),
        "dir_opposite": direction_angle_deg(u, um, lattice),
        "axis_opposite": axis_angle_deg(u, um, lattice),
        "plane_self": plane_normal_angle_deg(p, p, lattice),
        "plane_opposite": plane_normal_angle_deg(p, pm, lattice),
        "interplanar_opposite": interplanar_angle_deg(p, pm, lattice),
    }
    expected = {
        "dir_self": 0.0,
        "dir_opposite": 180.0,
        "axis_opposite": 0.0,
        "plane_self": 0.0,
        "plane_opposite": 180.0,
        "interplanar_opposite": 0.0,
    }
    for key, target in expected.items():
        if values[key] != target:
            raise AssertionError(
                f"angle endpoint {key} is not exact: got={values[key]!r}, expected={target!r}"
            )

    in_plane = Direction((0.0, 1.0, 0.0), basis)
    coordinate_plane = Plane((1.0, 0.0, 0.0), basis)
    if direction_plane_angle_deg(in_plane, coordinate_plane, lattice) != 0.0:
        raise AssertionError("exact incidence endpoint did not return 0 degrees")

    return "direct/projective/reciprocal angle endpoints are exact"


def check_spd_kernel() -> str:
    from cualni_cryst.lattice import metric_inv_sqrt, metric_sqrt

    bank = _integer_unimodular_bank()
    worst_sqrt = 0.0
    worst_white = 0.0
    for i in range(36):
        P = bank[i % len(bank)]
        k = i % 10
        d = np.array([2.0 ** (-k), 1.0, 2.0 ** k])
        M = _canonical_metric(P.T @ np.diag(d) @ P)
        S = metric_sqrt(M)
        W = metric_inv_sqrt(M)
        a = _rel(S @ S, M)
        b = float(np.linalg.norm(W @ M @ W - np.eye(3), "fro"))
        worst_sqrt = max(worst_sqrt, a)
        worst_white = max(worst_white, b)
        # Bound scales with conditioning; these cases top out around 2^18 times basis effects.
        if a > 2e-10 or b > 2e-9:
            raise AssertionError(
                f"SPD sqrt/invsqrt instability: sqrt={a:.3e}, whiten={b:.3e}, cond={np.linalg.cond(M):.3e}"
            )

    # Genuine indefiniteness must fail rather than being clipped/projected.
    bad = np.diag([1.0, 1.0, -2.0 ** -40])
    for fn in (metric_sqrt, metric_inv_sqrt):
        try:
            fn(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{fn.__name__} silently accepted an indefinite metric")

    return (
        f"SPD kernel passed; max sqrt residual={worst_sqrt:.3e}, "
        f"max whitening residual={worst_white:.3e}"
    )


def check_application_service_and_generic_input() -> str:
    from cualni_cryst.calculation_service import CalculationKind, CalculationService
    from cualni_cryst.project_io import generic_template, project_from_dict
    from cualni_cryst.project_state import james_hane_6m_reference_project

    ref = james_hane_6m_reference_project()
    service = CalculationService(ref)
    bundle = service.compute(
        CalculationKind.TRANSFORMATION_BUNDLE,
        "do3_to_6m_reference",
    )
    payload = bundle.to_dict()
    if not _finite_tree(payload):
        raise AssertionError("reference application bundle contains NaN/Inf")
    if payload["topology"]["n_variants"] != 12 or payload["topology"]["n_operators"] != 8:
        raise AssertionError("reference application bundle topology changed unexpectedly")
    json.dumps(payload, allow_nan=False)

    # Real user-facing arbitrary-input path: generic JSON project, triclinic cells,
    # point group 1, and an exact fractional correspondence.
    user = generic_template()
    user["project_id"] = "final_readiness_generic"
    user["phases"][0]["point_group"] = "1"
    user["phases"][0]["cell"] = {
        "a": 3.125,
        "b": 4.375,
        "c": 5.625,
        "alpha": 79.0,
        "beta": 101.0,
        "gamma": 73.0,
        "length_unit": "angstrom",
    }
    user["phases"][1]["point_group"] = "1"
    user["phases"][1]["cell"] = {
        "a": 3.25,
        "b": 4.125,
        "c": 5.875,
        "alpha": 82.0,
        "beta": 97.0,
        "gamma": 76.0,
        "length_unit": "angstrom",
    }
    user["transformations"] = [
        {
            "transformation_id": "generic_A_to_B",
            "parent_phase_id": "phase_A",
            "product_phase_id": "phase_B",
            "correspondence_M_from_A": [
                [1, "1/2", 0],
                [0, 1, "1/4"],
                [0, 0, 1],
            ],
        }
    ]
    loaded = project_from_dict(user)
    generic_service = CalculationService(loaded.project)
    metric = generic_service.compute(
        CalculationKind.METRIC_CORE,
        "generic_A_to_B",
    )
    representation = generic_service.compute(
        CalculationKind.REPRESENTATION,
        "generic_A_to_B",
    )
    if not _finite_tree(metric.to_dict()) or not _finite_tree(representation.to_dict()):
        raise AssertionError("generic arbitrary-input calculation produced NaN/Inf")
    json.dumps(metric.to_dict(), allow_nan=False)
    json.dumps(representation.to_dict(), allow_nan=False)

    return "reference bundle and generic triclinic/fractional-correspondence app path are finite + JSON-ready"


def check_invalid_input_fails_loudly() -> str:
    from cualni_cryst.adaptive_metric import adaptive_metric_eigensystem
    from cualni_cryst.correspondence import Correspondence
    from cualni_cryst.ct import cmc

    good = np.eye(3)
    bad_metric = np.diag([1.0, 1.0, -1.0])
    singular = np.diag([1.0, 1.0, 0.0])

    try:
        adaptive_metric_eigensystem(good, bad_metric, good)
    except ValueError:
        pass
    else:
        raise AssertionError("indefinite product metric was not rejected")

    try:
        adaptive_metric_eigensystem(good, good, singular)
    except ValueError:
        pass
    else:
        raise AssertionError("singular correspondence was not rejected")

    try:
        cmc(good, np.array([[1.0, 0, np.nan], [0, 1, 0], [0, 0, 1]]), Correspondence(sp.eye(3)))
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite metric was not rejected")

    return "indefinite/singular/non-finite invalid inputs fail loudly"


def check_git_immutability(before: str) -> str:
    after = _git("status", "--porcelain=v1")
    if after != before:
        raise AssertionError(
            "final readiness gate mutated repository state\n"
            f"before={before!r}\nafter={after!r}"
        )
    return "gate is repository-non-mutating"


CHECKS: tuple[tuple[str, Callable[[], str]], ...] = (
    ("v31_surface", check_v31_surface),
    ("known_regression_patterns", check_known_regression_patterns),
    ("compileall", check_compileall),
    ("literal_cmc", check_literal_cmc),
    ("certified_adaptive_classification", check_certified_adaptive_classification),
    ("generalized_vectors_habit_planes", check_generalized_vectors_and_habit_planes),
    ("smc_definition_covariance", check_smc_definition_and_covariance),
    ("binary_exact_unit_scaling", check_binary_exact_unit_scaling_laws),
    ("angle_endpoint_geometry", check_angle_endpoint_geometry),
    ("spd_kernel", check_spd_kernel),
    ("application_service_generic_input", check_application_service_and_generic_input),
    ("invalid_input_contracts", check_invalid_input_fails_loudly),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    args = parser.parse_args()

    start_status = _git("status", "--porcelain=v1")
    start_sha = _git("rev-parse", "HEAD")
    results: list[Check] = []

    for name, fn in CHECKS:
        try:
            detail = fn()
        except Exception:
            results.append(Check(name, False, traceback.format_exc()))
            print(f"[FAIL] {name}")
            print(results[-1].detail)
        else:
            results.append(Check(name, True, detail))
            print(f"[PASS] {name}: {detail}")

    try:
        detail = check_git_immutability(start_status)
    except Exception:
        results.append(Check("git_immutability", False, traceback.format_exc()))
        print("[FAIL] git_immutability")
        print(results[-1].detail)
    else:
        results.append(Check("git_immutability", True, detail))
        print(f"[PASS] git_immutability: {detail}")

    sensitive = [
        "src/cualni_cryst/adaptive_metric.py",
        "src/cualni_cryst/ct.py",
        "src/cualni_cryst/lattice.py",
        "src/cualni_cryst/stretch.py",
        "src/cualni_cryst/ptmc.py",
        "src/cualni_cryst/ball_james.py",
        "src/cualni_cryst/crystal_objects.py",
        "validation/engineered/certified_binary64.py",
        "validation/engineered/high_precision.py",
        "validation/engineered/metamorphic.py",
    ]
    hashes = {}
    for rel in sensitive:
        path = ROOT / rel
        if path.exists():
            hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()

    report = {
        "campaign": "FINAL_SCIENTIFIC_APP_READINESS_V1",
        "head_sha": start_sha,
        "python": sys.version,
        "numpy": np.__version__,
        "sympy": sp.__version__,
        "mpmath": mp.__version__,
        "checks": [asdict(x) for x in results],
        "sensitive_sha256": hashes,
        "status": "PASS" if all(x.passed for x in results) else "FAIL",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.report)}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
