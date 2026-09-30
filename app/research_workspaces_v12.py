from __future__ import annotations

"""Degenerate-limit interpretation hardening layered over V11.

Presentation/comparison semantics only. This module does not alter CT, Ball--James,
PTMC, twinning, group-theory, metric, orientation, EBSD, or numerical solvers.
It corrects how already-computed exact-degenerate states are described.
"""

from typing import Any, Mapping
import math

import streamlit as st

import app.research_workspaces as rw
import app.research_workspaces_v4 as v4
import app.research_workspaces_v10 as v10
import app.research_workspaces_v11 as v11
import app.scientific_interpretation_v6 as interp
from app.scientific_types import Evidence, Finding, sci


_original_native_ct_mm_table = v11._render_native_ct_mm_twin_table
_original_next_route = v11._render_next_route
_original_status_box = v10._status_box



def _status_box_v12(status: str, answer: str) -> None:
    normalized = str(status).strip().lower()
    if normalized == "exact trivial limit":
        st.success(f"**Answer — {status}.** {answer}")
        return
    if normalized == "evaluated zero branches":
        st.info(f"**Answer — {status}.** {answer}")
        return
    _original_status_box(status, answer)

def _project_policy_value(name: str, default: float) -> float:
    response = st.session_state.get("current_response")
    if response is None:
        return default
    project = (
        response.get("project_payload", response.get("project", {}))
        if isinstance(response, Mapping)
        else getattr(response, "project_payload", {})
    )
    if not isinstance(project, Mapping):
        return default
    policy = project.get("numerical_policy", {})
    if not isinstance(policy, Mapping):
        return default
    aliases = {
        "exact_eigenvalue": ("exact_eigenvalue", "exact_eigenvalue_tolerance"),
        "algebraic": ("algebraic", "algebraic_tolerance"),
    }
    for key in aliases.get(name, (name,)):
        try:
            value = float(policy[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    return default


def _mapping_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _ball_james_lambdas(unified: Any) -> tuple[float, ...]:
    if unified is None:
        return ()
    report = _mapping_get(unified, "ball_james_report")
    if report is None and isinstance(unified, Mapping):
        raw = unified.get("raw_reports", {})
        if isinstance(raw, Mapping):
            report = raw.get("ball_james")
    spectrum = _mapping_get(report, "metric_spectrum")
    values = _mapping_get(spectrum, "lambdas", ())
    try:
        numbers = tuple(float(value) for value in values)
    except Exception:
        return ()
    return numbers if len(numbers) >= 3 and all(math.isfinite(v) for v in numbers) else ()


def _ball_james_exact_and_identity(unified: Any) -> tuple[bool | None, bool, tuple[float, ...]]:
    lambdas = _ball_james_lambdas(unified)
    if len(lambdas) < 3:
        return None, False, lambdas
    tolerance = _project_policy_value("exact_eigenvalue", 1.0e-8)
    exact = abs(lambdas[1] - 1.0) <= tolerance
    identity_limit = max(abs(value - 1.0) for value in lambdas[:3]) <= tolerance
    return exact, identity_limit, lambdas


def _am_existence_finding_v12(
    checks: Mapping[str, Any],
    *,
    exact_ct_habits: int,
    bj_am: int,
    approx_ct_habits: int,
) -> Finding:
    """Classify A/M existence from native criteria, not enumerated branch count.

    In the fully degenerate U=I limit the Ball--James lambda_2 criterion is exact
    while the ordinary nontrivial branch parameterization may enumerate zero
    branches. Zero enumerated branches must not be reinterpreted as failed
    existence.
    """

    unified = rw._current_unified_report()
    bj_exact, identity_limit, lambdas = _ball_james_exact_and_identity(unified)
    if bj_exact is None:
        return interp.am_existence_finding(
            checks,
            exact_ct_habits=exact_ct_habits,
            bj_am=bj_am,
            approx_ct_habits=approx_ct_habits,
        )

    ct_exact = bool(checks.get("ct_am_exact_compatible"))
    agree = ct_exact == bj_exact
    tolerance = _project_policy_value("exact_eigenvalue", 1.0e-8)

    if agree and ct_exact and identity_limit:
        conclusion = (
            "CT and Ball–James agree that exact single-variant metric compatibility is "
            "satisfied. This is the fully degenerate U = I limit; zero enumerated "
            "nontrivial A/M branches does not mean incompatibility."
        )
        rationale = (
            "CT reports exact CMC compatibility, while the independent Ball–James stretch "
            "spectrum is lambda = (1, 1, 1) within the project's exact-eigenvalue tolerance. "
            "At U = I the rank-one equation is satisfied trivially with zero shape strain, "
            "so the usual nontrivial branch parameterization has no unique interface branch "
            "to enumerate. Branch count is therefore not used as an existence flag."
        )
        physical_meaning = (
            "The parent metric and pulled-back product metric coincide. No nonzero shape "
            "strain or uniquely selected habit plane is required for exact metric matching."
        )
        tone = "good"
    elif agree and ct_exact:
        conclusion = "CT and Ball–James independently agree that an exact single-variant A/M interface exists."
        rationale = (
            "CT exact compatibility and the independent Ball–James lambda_2 = 1 criterion "
            "are both satisfied. Enumerated branch counts are reported separately from the "
            "native yes/no existence criteria."
        )
        physical_meaning = "A homogeneous martensite variant satisfies the exact A/M compatibility condition in both formulations."
        tone = "good"
    elif agree:
        conclusion = "CT and Ball–James independently agree that no exact single-variant A/M interface exists."
        rationale = (
            "CT does not reach exact CMC compatibility and the independent Ball–James "
            "middle stretch does not satisfy lambda_2 = 1 within the exact tolerance."
        )
        physical_meaning = "A homogeneous single variant does not satisfy the exact A/M compatibility condition for this lattice state."
        tone = "warn"
    else:
        conclusion = "CT and Ball–James give different native A/M existence classifications; inspect the native criteria before drawing a physical conclusion."
        rationale = (
            f"CT exact-compatible = {ct_exact}; Ball–James lambda_2 exact = {bj_exact}. "
            "The comparison uses the native CT CMC classification and the native Ball–James "
            "stretch criterion, not whether a nontrivial branch happened to be enumerated."
        )
        physical_meaning = "The two native formulations are not presently giving the same exact A/M existence classification."
        tone = "bad"

    return Finding(
        title="Single-variant A/M interface",
        conclusion=conclusion,
        rationale=rationale,
        how=(
            "CT evaluates CMC degeneracy independently. Ball–James evaluates the ordered "
            "principal stretches independently and tests lambda_2 = 1. Only after those "
            "native existence criteria are classified are their branch inventories compared."
        ),
        physical_meaning=physical_meaning,
        limitation=(
            "An existence classification is not the same as a count of unique nontrivial "
            "habit/shape-strain branches. In degenerate limits, branch parameterizations can "
            "collapse even though exact compatibility remains satisfied. PTMC is a separate "
            "macroscopic mixture question."
        ),
        evidence=(
            Evidence("CT exact A/M", "native CMC criterion", "yes" if ct_exact else "no", ""),
            Evidence("Ball–James principal stretches", "ordered lambda spectrum", list(lambdas[:3]), ""),
            Evidence("Ball–James lambda_2", f"|lambda_2-1| <= {tolerance:.3e}", sci(abs(lambdas[1]-1.0), 6), "exact" if bj_exact else "not exact"),
            Evidence("Exact CT habit branches", "nontrivial branch inventory", exact_ct_habits, "not an existence flag"),
            Evidence("Ball–James A/M branches", "nontrivial branch inventory", bj_am, "not an existence flag"),
            Evidence("Fully degenerate U=I limit", "all three stretches equal one", "yes" if identity_limit else "no", ""),
        ),
        tone=tone,
        theory="Independent Cayron-CT CMC and Ball–James metric/rank-one existence tests",
        equations=(
            r"CMC=C^TM_MC-M_A",
            r"\lambda_2(U)=1",
            r"RU-I=b\otimes m",
            r"U=I\Rightarrow b=0\text{ is the trivial exact limit}",
        ),
        backend_mapping=(
            "Uses the already-computed CT exact-compatibility flag from the unified report.",
            "Uses the already-computed Ball–James metric stretch spectrum; no solver is rerun here.",
            "Nontrivial branch counts remain evidence only and are never substituted for the native existence criteria.",
        ),
        verification=(
            f"CT exact status = {'satisfied' if ct_exact else 'not satisfied'}",
            f"Ball–James lambda_2 status = {'satisfied' if bj_exact else 'not satisfied'}",
            f"U=I degenerate limit = {'yes' if identity_limit else 'no'}",
        ),
    )


def _mm_audit_finding_v12(audit: Mapping[str, Any] | None) -> Finding | None:
    if not audit:
        return None
    count = int(audit.get("relation_count", 0) or 0)
    if count != 0:
        return interp.mm_audit_finding(audit)

    _, identity_limit, lambdas = _ball_james_exact_and_identity(rw._current_unified_report())
    extra = (
        " The current stretch spectrum is the U = I degenerate limit, so collapse of "
        "nontrivial M/M stretch/twin geometry is consistent with the metric state."
        if identity_limit
        else ""
    )
    return Finding(
        title="Transformation twins — CT ↔ Mallard ↔ Ball–James",
        conclusion=(
            "The independent M/M audit was evaluated and found zero nontrivial physical "
            "M/M relations. With no relation to compare, this is an evaluated zero-result, "
            "not a failed cross-theory agreement test."
        ),
        rationale=(
            "A pass/fail agreement statement requires at least one physical relation. "
            "Here the authoritative audit returned relation_count = 0." + extra
        ),
        how=(
            "CT, Mallard and Ball–James are still run through their independent routes. "
            "The audit reports whether there are physical M/M relations available for a "
            "cross-theory comparison before attempting an agreement verdict."
        ),
        physical_meaning=(
            "No nontrivial M/M twin relation is established by this evaluated audit for the current state."
        ),
        limitation=(
            "Zero comparable relations is not evidence that the theories disagree, and it "
            "must not be relabelled as an unrun calculation."
        ),
        evidence=(
            Evidence("Authoritative physical relation count", "independent audit", count, "evaluated zero-result"),
            Evidence("Ball–James stretch spectrum", "context only", list(lambdas[:3]) if lambdas else "not available", ""),
            Evidence("U=I degenerate limit", "all stretches equal one", "yes" if identity_limit else "no", ""),
        ),
        tone="neutral",
        verification=(
            "calculation executed = yes",
            "physical M/M relation count = 0",
            "cross-theory disagreement verdict = not applicable",
        ),
    )


def _render_native_ct_mm_twin_table_v12() -> None:
    unified = rw._current_unified_report()
    if unified is None:
        _original_native_ct_mm_table()
        return
    rows = v11._ct_mm_rows(unified)
    if rows:
        _original_native_ct_mm_table()
        return

    _, identity_limit, lambdas = _ball_james_exact_and_identity(unified)
    st.info(
        "The unified CT inventory has already been evaluated and contains zero native "
        "nontrivial M/M twin rows for this state. Rerunning unchanged inputs is not a "
        "missing prerequisite."
    )
    if identity_limit:
        st.caption(
            "For the current third-order identity-stretch limit, the distinct metric stretch "
            "orbit collapses to U = I (lambda = 1,1,1), so absence of nontrivial M/M rows is "
            "consistent with the degenerate metric geometry."
        )


def _render_next_route_v12(step: Any) -> None:
    step_id = str(getattr(step, "step_id", ""))
    status = str(getattr(step, "status", "")).strip().lower()
    if step_id == "mm_twins" and status == "not reached" and rw._current_unified_report() is not None:
        with st.expander("How to interpret this evaluated zero-branch result", expanded=False):
            st.write(
                "The unified CT calculation has already run. 'Not reached' here means the "
                "evaluated inventory contains zero nontrivial native M/M twin branches; it "
                "does not mean that another Calculate / update click is required."
            )
            st.markdown(
                "**Inspect:** Theory comparison → Conclusions → Native theory inventory and provenance, "
                "or Workbench → Numerical audit & export for the frozen state."
            )
        return
    _original_next_route(step)


# Patch only already-computed interpretation/presentation hooks.
v10._status_box = _status_box_v12
v4.am_existence_finding = _am_existence_finding_v12
v4.mm_audit_finding = _mm_audit_finding_v12
v11._render_native_ct_mm_twin_table = _render_native_ct_mm_twin_table_v12
v11._render_next_route = _render_next_route_v12


def render_research_extension() -> None:
    v11.render_research_extension()
