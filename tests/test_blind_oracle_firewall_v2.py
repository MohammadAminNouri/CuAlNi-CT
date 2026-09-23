from __future__ import annotations

"""Blindness firewall for theory solvers.

Permanent benchmark tests already separate input and expected blocks logically.
This file adds a harder boundary:
  1. core solver modules may not contain benchmark file names or benchmark IDs;
  2. representative CT calculations must execute while all runtime access to
     data/benchmarks is denied.

This does not prove mathematical correctness; it proves that benchmark/oracle
files are not a hidden runtime dependency of the solver path.
"""

import inspect
import json
from pathlib import Path

import numpy as np
import sympy as sp

import cualni_cryst.ct as ct_module
import cualni_cryst.group_theory as group_module
import cualni_cryst.twinning_ct as twin_module
import cualni_cryst.weak_operator_engine as weak_operator_module
import cualni_cryst.weak_twins as weak_twins_module
import cualni_cryst.stretch as stretch_module
from cualni_cryst.correspondence import Correspondence
from cualni_cryst.ct import analyze_cmc, habit_planes_from_cmc
from cualni_cryst.group_theory import correspondence_groupoid
from cualni_cryst.lattice import Lattice
from cualni_cryst.point_groups import point_group_operations
from cualni_cryst.twinning_ct import twins_from_operator


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data" / "benchmarks"


def _benchmark_tokens() -> set[str]:
    tokens: set[str] = set()
    for path in BENCHMARK_DIR.glob("*.json"):
        tokens.add(path.name)
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        benchmark_id = payload.get("benchmark_id")
        if benchmark_id:
            tokens.add(str(benchmark_id))
    return tokens


def test_core_solver_sources_do_not_embed_permanent_benchmark_identifiers():
    modules = (
        ct_module,
        group_module,
        twin_module,
        weak_operator_module,
        weak_twins_module,
        stretch_module,
    )
    tokens = _benchmark_tokens()
    assert tokens
    for module in modules:
        source = inspect.getsource(module)
        leaked = sorted(token for token in tokens if token in source)
        assert not leaked, (module.__name__, leaked)


def test_representative_cayron_solver_path_runs_with_benchmark_directory_runtime_denied(monkeypatch):
    # Input is loaded BEFORE the firewall. Expected results are intentionally
    # never read at all.
    manifest = json.loads(
        (BENCHMARK_DIR / "cayron_2026_niti_c1_compatibility_v1.json").read_text()
    )
    inp = manifest["input"]
    del manifest

    original_open = Path.open

    def guarded_open(self: Path, *args, **kwargs):
        path = self.resolve(strict=False)
        try:
            path.relative_to(BENCHMARK_DIR.resolve())
        except ValueError:
            return original_open(self, *args, **kwargs)
        raise AssertionError(f"oracle/benchmark runtime access denied: {path}")

    monkeypatch.setattr(Path, "open", guarded_open)

    A = Lattice.cubic(float(inp["parent"]["lattice"]["a"]))
    p = inp["product"]["lattice"]
    M = Lattice.monoclinic_unique_b(
        float(p["a_over_a0"]),
        float(np.sqrt(2.0)),
        float(p["c_over_a0"]),
        float(p["beta_deg"]),
    )
    C = Correspondence(
        sp.Matrix(
            [
                [0, 0, 1],
                [sp.Rational(1, 2), sp.Rational(1, 2), 0],
                [-sp.Rational(1, 2), sp.Rational(1, 2), 0],
            ]
        )
    )

    analysis = analyze_cmc(A.metric(), M.metric(), C, tol=1e-8)
    planes = habit_planes_from_cmc(A.metric(), M.metric(), C, tol=1e-8)
    groupoid = correspondence_groupoid(
        list(point_group_operations("m-3m")),
        list(point_group_operations("2/m")),
        C,
    )
    twins = [
        twin
        for operator in groupoid.operators
        for twin in twins_from_operator(operator, A.metric(), M.metric(), C)
    ]

    # These are internal sanity statements, not literature answers.
    assert analysis.degeneracy_order >= 0
    assert planes
    assert groupoid.n_variants > 0
    assert groupoid.n_operators > 0
    assert twins
