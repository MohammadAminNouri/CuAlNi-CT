from __future__ import annotations

"""Read-only, deterministic three-level crystallographic family tree.

Root (y=2) -> operator/inverse family (y=1) -> every M_i/M_j couple
(y=0). Focusing one family changes the viewport, never the scientific
report or the identity of its couples. No literature outputs enter the graph.
"""

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any

from .scientific_models import TwinFamilyRecord, TwinFamilyReport, VariantPairRecord
from .ux_language import COLOR_FAMILY, COLOR_SELECTED, COLOR_HABIT, COLOR_EDGE


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


def normalized_rank_one(a, n):
    """Choose unit n, keeping (a ⊗ n) *exactly* unchanged."""
    length = sqrt(sum(float(x)**2 for x in n))
    if not isfinite(length) or length <= 1.e-15:
        raise ValueError("Cannot display a zero or nonfinite twin-plane normal")
    return tuple(float(x)*length for x in a), tuple(float(x)/length for x in n)


def tree_layout(report: TwinFamilyReport) -> TreeLayout:
    couples: list[CoupleNode] = []
    families: list[FamilyNode] = []
    x = 0.
    for family in report.families:
        pairs_by_index = {
            tuple(sorted((p.variant_i, p.variant_j))): p
            for p in family.pair_records
        }
        all_pairs = sorted({tuple(sorted(p)) for p in family.equivalent_pairs} | set(pairs_by_index))
        first = x
        for i, j in all_pairs:
            couples.append(CoupleNode(
                key=f"{family.family_id}:M{i}-M{j}",
                label=f"M{i} ↔ M{j}", family=family,
                pair=pairs_by_index.get((i, j)),
                variant_i=i, variant_j=j, x=x,
            ))
            x += 1.
        families.append(FamilyNode(family, (first + x - 1.) / 2. if all_pairs else x, len(all_pairs)))
        x += 1.0  # visual separation only; all couples remain at y=0
    root_x = (families[0].x + families[-1].x) / 2. if families else 0.
    return TreeLayout(tuple(families), tuple(couples), root_x)


def couple_habit_count(node: CoupleNode) -> int:
    if node.pair is None:
        return 0
    return sum(len(c.habit_solutions) for c in node.pair.constructions)


def _pair_status(node: CoupleNode) -> str:
    if node.pair is None:
        return "Correspondence couple; physical twin geometry not evaluated"
    if not node.pair.constructions:
        return node.pair.status
    count = couple_habit_count(node)
    if count:
        return f"{len(node.pair.constructions)} twin branch(es), {count} exact A/M habit solution(s)"
    if any(c.continuum_fraction for c in node.pair.constructions):
        return "Continuous compatible-fraction family; no discrete habit selected"
    return f"{len(node.pair.constructions)} twin branch(es), no exact A/M habit plane"


def plot_family_tree(
    report: TwinFamilyReport,
    selected_key: str | None = None,
    focus_family: str | None = None,
) -> Any:
    """Four ordered Plotly traces; trace #2 is the clickable couple level.

    All-view and family-focus share the same node keys. The root and family
    ancestry stay visible in *focus view*, preventing cropped orphan branches.
    The scientific tree_layout always holds every pair, even those not in the
    focused viewport; the family picker gives access to every reported couple.
    """
    import plotly.graph_objects as go

    complete = tree_layout(report)
    shown_families = tuple(
        f for f in complete.families if focus_family is None or f.family.family_id == focus_family
    )
    shown_couples = tuple(
        c for c in complete.couples if focus_family is None or c.family.family_id == focus_family
    )
    # A focused family gets a local, stable coordinate system. No data changed.
    if focus_family is not None:
        positions = {c.key: i * 1.25 for i, c in enumerate(shown_couples)}
        family_positions = {
            f.family.family_id: ((len(shown_couples) - 1) * 1.25 / 2. if shown_couples else 0.)
            for f in shown_families
        }
        root_x = next(iter(family_positions.values()), 0.)
    else:
        positions = {c.key: c.x * 1.25 for c in shown_couples}
        family_positions = {f.family.family_id: f.x * 1.25 for f in shown_families}
        root_x = complete.root_x * 1.25

    fig = go.Figure()
    xs: list[float | None] = []
    ys: list[float | None] = []
    for f in shown_families:
        x = family_positions[f.family.family_id]
        xs.extend([root_x, root_x, x, x, None])
        ys.extend([2., 1.50, 1.50, 1., None])
    for c in shown_couples:
        x = positions[c.key]
        fx = family_positions[c.family.family_id]
        xs.extend([fx, fx, x, x, None])
        ys.extend([1., .57, .57, .0, None])
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", line=dict(color=COLOR_EDGE, width=1.8),
        hoverinfo="skip", showlegend=False, name="Family connections",
    ))
    fig.add_trace(go.Scatter(
        x=[family_positions[f.family.family_id] for f in shown_families],
        y=[1.] * len(shown_families), mode="markers+text",
        text=[f"{f.family.family_id}" for f in shown_families],
        textposition="middle center", textfont=dict(color="#FFFFFF", size=12),
        marker=dict(symbol="square", size=42, color=COLOR_FAMILY, line=dict(width=1, color="#3C566B")),
        hovertext=[f"{f.family.family_id} · {f.family.route.replace('_', ' ')} · {f.count} couples"
                   for f in shown_families],
        hovertemplate="%{hovertext}<extra></extra>", showlegend=False,
        name="Operator families",
    ))
    fig.add_trace(go.Scatter(
        x=[positions[c.key] for c in shown_couples],
        y=[0.] * len(shown_couples), mode="markers+text",
        text=[c.label for c in shown_couples], textposition="bottom center",
        textfont=dict(color="#DDE5ED", size=12),
        marker=dict(
            size=[29 if c.key == selected_key else 22 for c in shown_couples],
            color=[COLOR_SELECTED if c.key == selected_key else
                   COLOR_HABIT if couple_habit_count(c) else COLOR_EDGE for c in shown_couples],
            symbol=["circle" if c.key == selected_key else
                    "diamond" if couple_habit_count(c) else "circle-open" for c in shown_couples],
            line=dict(width=1.5, color="#D4DEE5"),
        ),
        customdata=[[c.key] for c in shown_couples],
        hovertext=[f"{c.label} · {_pair_status(c)}" for c in shown_couples],
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False, name="Twin couples",
    ))
    fig.add_trace(go.Scatter(
        x=[root_x], y=[2.], mode="markers+text",
        marker=dict(symbol="diamond", size=28, color=COLOR_FAMILY),
        text=["A → M"], textposition="top center", textfont=dict(color="#DDE5ED", size=14),
        hoverinfo="skip", showlegend=False, name="Transformation root",
    ))

    if focus_family is not None:
        total = len(shown_couples)
        center = root_x
    else:
        total = len(shown_couples)
        center = root_x
    # A focused group is fully visible up to 10 nodes; larger groups are
    # keyboard-selectable and pannable without making tiny unreadable labels.
    half = max(3.5, min(9.0, ((total-1)*1.25)/2.+1.25))
    fig.update_layout(
        height=325,
        margin=dict(l=26, r=26, b=85, t=50),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        dragmode="pan", clickmode="event+select",
        xaxis=dict(visible=False, range=[center-half, center+half], fixedrange=False),
        yaxis=dict(visible=False, range=[-.73, 2.48], fixedrange=True),
        showlegend=False,
        uirevision=f"family-{focus_family or 'all'}",
    )
    return fig
