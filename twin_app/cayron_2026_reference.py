from __future__ import annotations

"""Paper-scoped NiTi B2/B19' analytical checks (Cayron, Acta Mater. 316, 2026).

NOT a general Cu-Al-Ni metric optimizer. Nothing here selects an orientation
relationship, modifies a martensitic solver or predicts experimental variants.
Always report the paper's contradictory printed formulas as source issues.
"""

import math


DOI = "10.1016/j.actamat.2026.122399"


def niti_b19prime_cmc_domains(a: float, b: float, c: float, beta_deg: float) -> list[dict]:
    """Eq. (47), dimensional ratios a,b,c relative to cubic a_B2.

    Explicitly keep Table 2's C2b inequality distinct from Eq. (47).
    Neither an equality distance nor a satisfied inequality by itself
    establishes an actual habit plane without the source correspondence.
    """
    if not all(math.isfinite(float(x)) for x in (a, b, c, beta_deg)):
        raise ValueError("All NiTi dimensionless ratios/angles must be finite")
    if min(a, b, c) <= 0 or not 0 < beta_deg < 180:
        raise ValueError("Ratios must be positive and monoclinic angle inside (0,180)")
    rad = math.radians(beta_deg)
    sin2 = math.sin(rad) ** 2
    k = 2 * a * a + c * c - a * a * c * c * sin2
    base = (1 - a * a * sin2)
    target_c_sq = (2 * (1 - a * a) / base) if abs(base) > 1e-12 else None
    c2_gap = (abs(c * c - target_c_sq)
              if target_c_sq is not None else None)
    if target_c_sq is not None and target_c_sq >= 0:
        c_gap = abs(c - math.sqrt(target_c_sq))
    else:
        c_gap = None
    sqrt2 = math.sqrt(2)
    return [
        {
            "Domain": "C1",
            "Equality residual": abs(b * b - 2),
            "Equality equation": "b² = 2",
            "Inequality equation": "2a²+c²−a²c²sin²β ≥ 2",
            "Inequality satisfied": k >= 2,
            "Inequality distance": abs(k - 2),
            "Source consistency": "Eq. 47 / Table 2 agree",
        },
        {
            "Domain": "C2a",
            "Equality residual": c2_gap,
            "Equality equation": "c² = 2(1−a²)/(1−a²sin²β)",
            "Inequality equation": "a sinβ ≥ 1 and b ≤ √2",
            "Inequality satisfied": a * math.sin(rad) >= 1 and b <= sqrt2,
            "Inequality distance": abs(a * a * sin2 - 1) + abs(b * b - 2),
            "Source consistency": "Eq. 47 / Table 2 agree",
        },
        {
            "Domain": "C2b",
            "Equality residual": abs(a * a - 1) + abs(sin2 - 1),
            "Equality equation": "a=1 and β=90°",
            "Inequality equation": "b ≤ √2 in Eq. 47; b ≥ √2 in Table 2",
            "Inequality satisfied": None,
            "Inequality distance": abs(b * b - 2),
            "Source consistency": "SOURCE_CONFLICT: inequality direction differs",
            "Eq47 inequality satisfied": b <= sqrt2,
            "Table2 inequality satisfied": b >= sqrt2,
        },
        {
            "Domain": "C3",
            "Equality residual": c2_gap,
            "Equality equation": "c² = 2(1−a²)/(1−a²sin²β)",
            "Inequality equation": "a ≤ 1 and b ≥ √2",
            "Inequality satisfied": a <= 1 and b >= sqrt2,
            "Inequality distance": abs(a * a - 1) + abs(b * b - 2),
            "Source consistency": (
                "Table 2's displayed 0.269706 is a c-ratio gap, "
                "not this squared-ratio equality gap"
            ),
            "Alternative |c−target|": c_gap,
        },
    ]


def kudoh_1996_niti_source_case() -> list[dict]:
    """Table 2 evaluated from Figure 7's dimensional source precision."""
    return niti_b19prime_cmc_domains(
        2.898 / 3.01, 4.108 / 3.01, 4.646 / 3.01, 97.78,
    )


def o2_printed_vs_equation_derived(beta_deg: float) -> list[dict]:
    """Independent exact identity from Cayron Eq.41/43 with m_A=(001)."""
    if not math.isfinite(beta_deg) or not 0 < beta_deg < 180:
        raise ValueError("Monoclinic angle must lie inside (0,180)")
    s = math.sin(math.radians(beta_deg))
    if abs(s) <= 1e-12:
        raise ValueError("Degenerate sin(beta)")
    return [
        {"Family": "O2 Type I as printed", "a": math.sqrt(2)/s,
         "b": math.sqrt(2), "c": math.sqrt(2),
         "SMC d_z": 0.5, "Type I ||U^-1 e_001||²": 0.5,
         "Provenance": "PUBLISHED, INTERNALLY INCONSISTENT"},
        {"Family": "O2 Type I from Eq. 41/43", "a": 1/s,
         "b": math.sqrt(2), "c": math.sqrt(2),
         "SMC d_z": 0.0, "Type I ||U^-1 e_001||²": 1.0,
         "Provenance": "PROJECT-DERIVED, NOT AUTHOR-CONFIRMED"},
    ]


def appendix_c_o4_at_beta_98() -> dict:
    """Independent transcription of Appendix C's displayed O4 family."""
    beta = math.radians(98.0)
    r = math.sqrt(1 - 12 * math.sqrt(2) / math.tan(beta))
    a = 0.5 * math.sqrt(6 - 4 * math.sqrt(2) / math.tan(beta) - 2 * r)
    u = math.sqrt(6 - 4 * math.sqrt(2) * math.cos(beta) / math.sin(beta) - 2 * r)
    v = math.sqrt(9 + 4 * math.sqrt(2) * math.cos(beta) * math.sin(beta)
                  + r + math.cos(2 * beta) * (3 - r))
    c = 0.5 * (-math.cos(beta) * u + v)
    return {"beta_deg": 98.0, "a": a, "b": math.sqrt(2), "c": c,
            "paper_displayed_a": 0.8825, "paper_displayed_c": 1.6182}


def render_cayron_2026_reference():
    """A reproducible, expressly literature-only design-domain vignette."""
    import pandas as pd
    import streamlit as st

    with st.expander("NiTi literature-only: Cayron 2026 C1/C2a/C2b/C3 phase-design equations", expanded=False):
        st.caption(
            "These numbers are computed from Kudoh NiTi B2/B19′ (Figure 7, "
            "aB2=3.01 Å; a=2.898 Å, b=4.108 Å, c=4.646 Å, β=97.78°). "
            "They are NOT calculations for the active Cu–Al–Ni input. "
            "The equality and inequality distances are dimensionless and distinct."
        )
        st.dataframe(pd.DataFrame(kudoh_1996_niti_source_case()),
                     hide_index=True, width="stretch")
        st.caption(
            "Table 2 C1 comparison: 0.137364 equality gap and 0.068402 inequality "
            "margin (rounded). The analytical C2b inequality is b≤√2 in Eq. 47 "
            "but b≥√2 in Table 2. Neither formulation is silently substituted."
        )
        st.markdown("**O2 Type-I published-family consistency check (β=98°)**")
        st.dataframe(pd.DataFrame(o2_printed_vs_equation_derived(98.0)),
                     hide_index=True, width="stretch")
        st.markdown("**O4 Appendix-C analytical source check (β=98°)**")
        st.dataframe(pd.DataFrame([appendix_c_o4_at_beta_98()]),
                     hide_index=True, width="stretch")
        st.warning(
            "These calculations are NiTi B2/B19′ reference oracles only. "
            "The general CMC backend remains the applicable approach for an "
            "arbitrary phase pair, provided its correspondence and metrics are valid. "
            "The paper's O2 Type-I published equation remains unconfirmed by the author."
        )
