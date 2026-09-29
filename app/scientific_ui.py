from __future__ import annotations

"""Shared professor-facing presentation primitives.

Rendering never changes scientific data.  Each Finding is displayed in the
stable order Answer -> Why -> How -> Verify -> Physical meaning -> Limitation.
"""

from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st

from app.scientific_types import Evidence, Finding, sci


def install_styles() -> None:
    st.markdown(
        """
<style>
.block-container {padding-top:.75rem; padding-bottom:3.5rem; max-width:1380px;}
[data-testid="stSidebarNav"] {display:none;}
[data-testid="stMetricValue"] {font-size:1.18rem;}
[data-testid="stMetricLabel"] {font-size:.78rem; opacity:.82;}
.sci-kicker {font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; opacity:.58;}
.sci-title {font-size:1.92rem; font-weight:760; line-height:1.08; margin:.08rem 0 .12rem 0;}
.sci-subtitle {font-size:.94rem; opacity:.74; max-width:1000px; margin-bottom:.65rem;}
.sci-question {font-size:.86rem; opacity:.68; margin:.15rem 0 .45rem 0;}
.sci-card {border:1px solid rgba(128,128,128,.24); border-radius:.65rem; padding:.72rem .84rem; margin:.45rem 0;}
.sci-card.good {border-left:4px solid #38a169;}
.sci-card.warn {border-left:4px solid #d69e2e;}
.sci-card.bad {border-left:4px solid #d9534f;}
.sci-card.neutral {border-left:4px solid #5b8def;}
.sci-card-title {font-size:.77rem; text-transform:uppercase; letter-spacing:.06em; opacity:.67; margin-bottom:.18rem;}
.sci-card-conclusion {font-size:1.05rem; font-weight:690; line-height:1.35;}
.sci-card-rationale {font-size:.88rem; opacity:.82; line-height:1.45; margin-top:.28rem;}
.sci-status {font-size:.72rem; opacity:.7; margin-top:.3rem;}
.sci-state {border:1px solid rgba(128,128,128,.22); border-radius:.55rem; padding:.46rem .65rem; margin:.25rem 0 .65rem 0; font-size:.86rem;}
div[data-testid="stExpander"] details summary p {font-weight:600;}
</style>
""",
        unsafe_allow_html=True,
    )


def custom_sidebar_navigation(*, current: str) -> None:
    st.markdown("#### CuAlNi-CT")
    try:
        st.page_link("streamlit_app.py", label="Workbench", icon="🧭")
        st.page_link("pages/2_CT_Equivalence_Lab.py", label="Theory comparison", icon="🔬")
    except Exception:
        pass
    st.divider()


def page_header(title: str, subtitle: str, *, kicker: str = "Phase-transformation crystallography") -> None:
    st.markdown(f'<div class="sci-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sci-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sci-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def section_header(title: str, question: str, *, answer_hint: str | None = None) -> None:
    st.markdown(f"## {title}")
    st.markdown(f'<div class="sci-question"><b>Question:</b> {question}</div>', unsafe_allow_html=True)
    if answer_hint:
        st.caption(answer_hint)


def state_strip(text: str) -> None:
    st.markdown(f'<div class="sci-state">{text}</div>', unsafe_allow_html=True)


def _escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _display_value(value: Any) -> Any:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if value is None:
        return "not evaluable"
    return value



def validate_finding_contract(finding: Finding) -> tuple[str, ...]:
    """Return professor-facing contract defects without mutating the finding."""
    defects: list[str] = []
    if not str(finding.conclusion).strip():
        defects.append("missing Answer/conclusion")
    if not str(finding.rationale).strip():
        defects.append("missing Why/rationale")
    if not (finding.evidence or finding.verification):
        defects.append("missing Verify evidence")
    if not (str(finding.theory).strip() or str(finding.how).strip()):
        defects.append("missing How theory/construction")
    if not (finding.backend_mapping or str(finding.how).strip()):
        defects.append("missing backend mapping/implementation route")
    if not str(finding.physical_meaning).strip():
        defects.append("missing Physical meaning")
    return tuple(defects)

def render_finding(finding: Finding, *, details_label: str = "Why / how / verify") -> None:
    defects = validate_finding_contract(finding)
    if defects:
        st.error("Scientific result suppressed because its presentation contract is incomplete: " + "; ".join(defects))
        return
    tone = finding.tone if finding.tone in {"good", "warn", "bad", "neutral"} else "neutral"
    status_html = f'<div class="sci-status">{_escape(finding.status)}</div>' if finding.status else ""
    st.markdown(
        f'<div class="sci-card {tone}">'
        f'<div class="sci-card-title">{_escape(finding.title)}</div>'
        f'<div class="sci-card-conclusion">{_escape(finding.conclusion)}</div>'
        f'<div class="sci-card-rationale">{_escape(finding.rationale)}</div>'
        f'{status_html}</div>',
        unsafe_allow_html=True,
    )
    with st.expander(details_label, expanded=False):
        why_tab, how_tab, verify_tab, physical_tab, limit_tab = st.tabs(
            ["Why", "How", "Verify", "Physical meaning", "Limitation"]
        )
        with why_tab:
            st.write(finding.rationale)
        with how_tab:
            if finding.theory:
                st.markdown("**Theory / construction**")
                st.write(finding.theory)
            if finding.equations:
                st.markdown("**Governing equation / criterion**")
                for equation in finding.equations:
                    st.latex(equation)
            if finding.symbols:
                st.markdown("**Symbols**")
                st.dataframe(
                    pd.DataFrame(finding.symbols, columns=["symbol", "meaning"]),
                    hide_index=True,
                    use_container_width=True,
                )
            if finding.assumptions:
                st.markdown("**Assumptions / frame**")
                for item in finding.assumptions:
                    st.write(f"• {item}")
            if finding.backend_mapping:
                st.markdown("**What the code evaluates**")
                for item in finding.backend_mapping:
                    st.write(f"• {item}")
            if finding.provenance:
                st.markdown("**Reference / provenance**")
                for item in finding.provenance:
                    st.write(f"• {item}")
            if finding.verbal:
                st.markdown("**20-second verbal version**")
                st.write(finding.verbal)
            if finding.how:
                st.markdown("**Implementation note**")
                st.write(finding.how)
        with verify_tab:
            if finding.evidence:
                render_evidence(finding.evidence)
            if finding.verification:
                st.markdown("**Logical / gating audit**")
                for item in finding.verification:
                    st.write(f"• {item}")
            if not finding.evidence and not finding.verification:
                st.error("Verification contract missing; this result is not presented as a scientific conclusion.")
        with physical_tab:
            st.write(finding.physical_meaning or "No additional physical interpretation is claimed.")
        with limit_tab:
            st.write(finding.limitation or "No additional limitation was supplied by this result.")


def render_evidence(rows: Sequence[Evidence] | Iterable[Evidence]) -> None:
    data = [
        {
            "quantity": row.quantity,
            "criterion": row.criterion,
            "computed": _display_value(row.value),
            "interpretation": row.interpretation,
        }
        for row in rows
    ]
    st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)


def compact_metrics(items: Sequence[tuple[str, Any]], *, columns: int | None = None) -> None:
    if not items:
        return
    n = columns or min(len(items), 5)
    for start in range(0, len(items), n):
        batch = items[start : start + n]
        cols = st.columns(len(batch))
        for col, (label, value) in zip(cols, batch, strict=True):
            col.metric(label, _display_value(value))


def format_dataframe_scientific(frame: pd.DataFrame, *, digits: int = 3) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_bool_dtype(output[column]):
            output[column] = output[column].map(_display_value)
        elif pd.api.types.is_numeric_dtype(output[column]):
            output[column] = output[column].map(lambda x: sci(x, digits) if pd.notna(x) else "—")
    return output


def audit_expander(title: str = "Full numerical audit", *, expanded: bool = False):
    return st.expander(title, expanded=expanded)


def status_word(value: bool | None, *, true: str = "satisfied", false: str = "not satisfied") -> str:
    if value is True:
        return true
    if value is False:
        return false
    return "not evaluable"
