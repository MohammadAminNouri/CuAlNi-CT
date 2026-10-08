from __future__ import annotations

import ast
from pathlib import Path


def _source() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "twin_app" / "scientific_engine.py").read_text(encoding="utf-8")


def test_engine_keeps_correspondence_and_stretch_variants_separate():
    source = _source()
    assert "CorrespondenceVariantRecord" in source
    assert "stretch_variant_index" in source
    assert "metric collapse" in source
    assert "correspondence_to_stretch" in source


def test_engine_groups_unoriented_pairs_by_operator_and_inverse_operator():
    source = _source()
    assert "groupoid.adjacency[i][j]" in source
    assert "groupoid.adjacency[j][i]" in source
    assert "tuple(sorted({forward, reverse}))" in source


def test_engine_uses_existing_independent_science_routes_not_a_lookup_table():
    source = _source()
    tree = ast.parse(source)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "correspondence_groupoid" in calls
    assert "twins_from_operator" in calls
    assert "stretch_from_metrics" in calls
    assert "audit_operator_route" in calls
    assert "analyze_higher_order_element" in calls
    assert "PTMCAdapter" in source
    assert "BallJamesAdapter" in source


def test_weak_planes_require_explicit_node_basis_and_are_not_guessed():
    source = _source()
    assert "product_node_basis: BravaisNodeBasis | None = None" in source
    assert "requires explicit product primitive-node basis" in source
    assert "centering" in source
    assert "not inferred" in source


def test_habit_planes_are_attached_to_the_specific_twin_relation():
    source = _source()
    assert "habit_by_relation" in source
    assert "(ui, uj, int(relation.branch))" in source
    assert "no exact A/M habit-plane solution for this twin construction" in source
