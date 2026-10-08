from __future__ import annotations

"""Deterministic, presentation-only graph for ALL calculated variant couples.

Scientific identity follows the report: root -> operator/inverse-operator
family -> all correspondence-variant couples. Every couple occupies y=0.
The twin constructions and A/M habit branches belong in ONE details panel,
never in a second row of visual cards.
"""

from dataclasses import dataclass
from typing import Any

from .scientific_models import TwinFamilyRecord, TwinFamilyReport, VariantPairRecord


@dataclass(frozen=True)
class FamilyNode:
    family: TwinFamilyRecord
    x: float
    count: int


@dataclass(frozen=True)
class CoupleNode:
    key: str
    label: str
    family: TwinFamilyRecord
    pair: VariantPairRecord | None
    variant_i: int
    variant_j: int
    x: float
    y: int = 0


@dataclass(frozen=True)
class TreeLayout:
    families: tuple[FamilyNode, ...]
    couples: tuple[CoupleNode, ...]
    root_x: float


def normalized_rank_one(a: tuple[float, float, float], n: tuple[float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Book convention: n-hat unit, a rescaled so a tensor n is unchanged.

    Rank-one products are gauge-invariant; printing an unnormalized n as
    'n-hat' would misrepresent the physical twinning elements.
    """
    from math import sqrt, isfinite
    length = sqrt(sum(float(x) ** 2 for x in n))
    if not isfinite(length) or length <= 1.0e-15:
        raise ValueError("Cannot present a zero/nonfinite rank-one normal as n-hat")
    return (
        tuple(float(x) * length for x in a),
        tuple(float(x) / length for x in n),
    )


def tree_layout(report: TwinFamilyReport) -> TreeLayout:
    """Include every reported pair, not merely the family representative.

    A pair without an available physical rank-one solution remains in the tree;
    omission would misleadingly imply it is not a crystallographic couple.
    """
    couples: list[CoupleNode] = []
    families: list[FamilyNode] = []
    x = 0.0
    for family in report.families:
        pairs_by_index = {
            (min(p.variant_i, p.variant_j), max(p.variant_i, p.variant_j)): p
            for p in family.pair_records
        }
        pairs = sorted(set(family.equivalent_pairs) | set(pairs_by_index))
        family_couples = []
        for i, j in pairs:
            i, j = min(i, j), max(i, j)
            item = CoupleNode(
                key=f"{family.family_id}:M{i}-M{j}",
                label=f"M{i} ↔ M{j}",
                family=family,
                pair=pairs_by_index.get((i, j)),
                variant_i=i,
                variant_j=j,
                x=x,
            )
            family_couples.append(item)
            couples.append(item)
            x += 1.0
        if family_couples:
            midpoint = (family_couples[0].x + family_couples[-1].x) / 2.0
        else:
            midpoint = x
            x += 1.0
        families.append(FamilyNode(family, midpoint, len(family_couples)))
        x += 0.65  # only separates families, never puts couples on a lower row

    if families:
        root_x = (families[0].x + families[-1].x) / 2.0
    else:
        root_x = 0.0
    return TreeLayout(tuple(families), tuple(couples), root_x)


def _pair_status(node: CoupleNode) -> str:
    if node.pair is None:
        return "Pair geometry not evaluated"
    constructions = node.pair.constructions
    if not constructions:
        return node.pair.status
    count = sum(len(c.habit_solutions) for c in constructions)
    if count:
        return f"{len(constructions)} twin solution(s), {count} exact habit branch(es)"
    if any(c.continuum_fraction for c in constructions):
        return "Continuous A/M compatibility family"
    return f"{len(constructions)} twin solution(s), no exact A/M habit"


def _mode_short(family: TwinFamilyRecord) -> str:
    return {"classical_exact":"Classical", "axial_weak":"Weak", "unsupported":"Unresolved"}.get(
        family.route, family.route.replace("_", " ").capitalize()
    )


def plot_family_tree(report: TwinFamilyReport, selected_key: str | None = None) -> Any:
    """Plotly selection lives only in the renderer; this function is pure.

    Trace 2 contains all the same-level couple nodes, each with a stable key
    as its customdata. A selected Plotly point maps unambiguously to one pair.
    """
    import plotly.graph_objects as go

    layout = tree_layout(report)
    fig = go.Figure()
    # Trace 0: physical connectors from root to family and family to EVERY couple.
    xs: list[float | None] = []
    ys: list[float | None] = []
    for family in layout.families:
        xs.extend([layout.root_x, family.x, None])
        ys.extend([2.0, 1.0, None])
    for item in layout.couples:
        parent_x = next(x.x for x in layout.families if x.family.family_id == item.family.family_id)
        xs.extend([parent_x, item.x, None])
        ys.extend([1.0, 0.0, None])
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", line=dict(color="#A5ABB2", width=1.35),
        hoverinfo="skip", showlegend=False, name="Connectors",
    ))
    # Trace 1: operator-family nodes.
    fig.add_trace(go.Scatter(
        x=[f.x for f in layout.families], y=[1.0] * len(layout.families),
        mode="markers+text",
        marker=dict(size=24, color="#6E7B86", symbol="diamond"),
        text=[f.family.family_id for f in layout.families], textposition="top center",
        customdata=[[f.family.family_id] for f in layout.families],
        hovertext=[f"{f.family.family_id} · {_mode_short(f.family)} · {f.count} couples" for f in layout.families],
        hovertemplate="%{hovertext}<extra></extra>", name="Operator families", showlegend=False,
    ))
    # Trace 2: ALL variant couples are siblings at the SAME y position.
    fig.add_trace(go.Scatter(
        x=[item.x for item in layout.couples], y=[0.0] * len(layout.couples),
        mode="markers+text",
        marker=dict(
            size=[31 if item.key == selected_key else 24 for item in layout.couples],
            color=["#275C95" if item.key == selected_key else "#566C7C" for item in layout.couples],
            line=dict(color="#FFFFFF", width=1.5),
            symbol="circle",
        ),
        text=[item.label.replace(" ↔ ", "<br>↔ ") for item in layout.couples],
        textposition="bottom center",
        customdata=[[item.key] for item in layout.couples],
        hovertext=[f"{item.label} · {_pair_status(item)}" for item in layout.couples],
        hovertemplate="%{hovertext}<extra></extra>",
        name="Twin couples", showlegend=False,
    ))
    # Trace 3: unique root.
    fig.add_trace(go.Scatter(
        x=[layout.root_x], y=[2.0], mode="markers+text",
        marker=dict(size=30, color="#34434F", symbol="diamond"),
        text=[f"A → M<br>{report.audit.topology_variant_count} M · {report.audit.stretch_variant_count} U"],
        textposition="top center", name="Transformation root", hoverinfo="skip", showlegend=False,
    ))
    # Keep labels readable even for 12-variant cases with many couples.
    # The *whole* single-level tree remains navigable by panning and zooming;
    # only the viewport is cropped, never the scientific node inventory.
    selected = next((node for node in layout.couples if node.key == selected_key), None)
    center = selected.x if selected is not None else layout.root_x
    half_window = 5.75 if len(layout.couples) > 11 else max(2.0, len(layout.couples) / 2.0 + 0.75)
    visible_min, visible_max = center - half_window, center + half_window
    fig.update_layout(
        height=375,
        margin=dict(l=22, r=22, b=80, t=62),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        dragmode="pan", clickmode="event+select",
        xaxis=dict(visible=False, range=[visible_min, visible_max], fixedrange=False),
        yaxis=dict(visible=False, range=[-0.58, 2.52], fixedrange=True),
        showlegend=False,
        uirevision=f"twin-layout-{selected_key}",
    )
    return fig
