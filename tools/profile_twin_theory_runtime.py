"""Measure which exact theory stage stalls the deployed NiTi example.

Non-mutating profiler. It does not change scientific equations or write user data.
"""
from __future__ import annotations

from time import perf_counter

from app.application import CalculationRequest, LatticeInput, PhaseInput, TransformationInput
from cualni_cryst.project_io import project_from_dict
from cualni_cryst.theory_unified import PTMCMode
from cualni_cryst.theory_unified_components import StagedTheoryComparisonAdapter
from cualni_cryst.theory_equivalence_optimized import (
    build_equivalence_report_optimized,
    independent_mm_audit,
)
from twin_app.streamlit_app import NITI_PARENT, NITI_PRODUCT, NITI_CORRESPONDENCE


def stage(label, callback):
    print(f"START {label}", flush=True)
    start = perf_counter()
    result = callback()
    print(f"END   {label}: {perf_counter() - start:.3f}s", flush=True)
    return result


def main():
    pa = PhaseInput("A", "B2 parent", LatticeInput(*NITI_PARENT), "m-3m")
    pm = PhaseInput("M", "B19' product", LatticeInput(*NITI_PRODUCT), "2/m")
    tr = TransformationInput.from_rows("A_to_M", "A", "M", NITI_CORRESPONDENCE)
    request = CalculationRequest(
        project_id="runtime_diagnosis", title="Input NiTi",
        parent=pa, product=pm, transformation=tr,
    )
    payload = request.to_project_payload()
    project = stage("validate project", lambda: project_from_dict(
        dict(payload), source="theory_runtime_diagnosis"
    ).project)
    a = StagedTheoryComparisonAdapter(project, "A_to_M")
    ct_lite = stage("CT without closing gap", lambda: a.ct_component(
        include_closing_gap=False, include_supercompatibility=False,
    ))
    bj = stage("Ball–James fractions=11", lambda: a.ball_james_component(
        fraction_samples=11,
    ))
    ptmc = stage("PTMC all twinning", lambda: a.ptmc_component(
        mode=PTMCMode.ALL_TWINNING,
    ))
    lite = stage("assemble basic", lambda: a.assemble(ct_lite, bj, ptmc))
    stage("match basic optimized", lambda: build_equivalence_report_optimized(
        project, "A_to_M", lite,
    ))
    ct_full = stage("CT closing gap + shear/shear", lambda: a.ct_component(
        include_closing_gap=True, include_supercompatibility=True,
    ))
    full = stage("assemble full", lambda: a.assemble(ct_full, bj, ptmc))
    stage("match full optimized", lambda: build_equivalence_report_optimized(
        project, "A_to_M", full,
    ))
    stage("independent MM audit", lambda: independent_mm_audit(
        project, "A_to_M",
    ))
    print("PROFILE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
