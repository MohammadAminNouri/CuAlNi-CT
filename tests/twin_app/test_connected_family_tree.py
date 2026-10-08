from __future__ import annotations

from types import SimpleNamespace as Obj

from twin_app.family_tree_graph import plot_family_tree, tree_layout


def _case():
    def pair(i, j, habits=()):
        twins = () if habits is None else (Obj(
            branch=1, classification="Type II", shear_magnitude=.28,
            habit_solutions=tuple(Obj(habit_branch=k, other_variant_volume_fraction=.271)
                                  for k in habits),
            continuum_fraction=False,
        ),)
        return Obj(variant_i=i, variant_j=j, pair_id=f"M{i}-M{j}",
                   status="evaluated", constructions=twins,
                   stretch_i=i, stretch_j=j, operator_forward=1, operator_reverse=1)
    f1 = Obj(family_id="F1", route="classical_exact",
             equivalent_pairs=((1,2),(1,3),(2,4)),
             representative_pair=(1,2),
             pair_records=(pair(1,2,(1,-1)),pair(1,3,()),pair(2,4,None)),
             classical_systems=(),weak_candidates=())
    f2 = Obj(family_id="F2", route="axial_weak",
             equivalent_pairs=((1,5),(2,6)), representative_pair=(1,5),
             pair_records=(), classical_systems=(), weak_candidates=())
    return Obj(parent_phase_id="A",product_phase_id="M",families=(f1,f2),
               audit=Obj(topology_variant_count=12,stretch_variant_count=12))


def test_all_couples_are_siblings_at_same_depth_and_none_are_folded_away():
    layout = tree_layout(_case())
    assert len(layout.families)==2
    assert len(layout.couples)==5
    assert set(c.label for c in layout.couples)=={
        "M1 ↔ M2", "M1 ↔ M3", "M2 ↔ M4", "M1 ↔ M5", "M2 ↔ M6"
    }
    assert {c.y for c in layout.couples} == {0}
    assert [c.x for c in layout.couples] == sorted(c.x for c in layout.couples)
    assert layout.couples[0].pair is not None
    assert layout.couples[3].pair is None  # weak family still visible, not faked


def test_graph_has_one_root_two_families_and_all_couples_selection_keys():
    report = _case()
    layout = tree_layout(report)
    fig = plot_family_tree(report, layout.couples[0].key)
    assert len(fig.data)==4
    assert len(fig.data[2].x)==5
    assert set(fig.data[2].y)=={0.0}
    assert len(fig.data[1].x)==2
    assert len(fig.data[3].x)==1
    assert len({c.key for c in layout.couples}) == len(layout.couples)
    assert fig.data[2].customdata[0][0] == layout.couples[0].key
    assert '↔' in fig.data[2].hovertext[0]


def test_habit_branches_not_precomputed_from_book():
    layout = tree_layout(_case())
    assert len(layout.couples[0].pair.constructions[0].habit_solutions)==2
    assert len(layout.couples[1].pair.constructions[0].habit_solutions)==0
    # Diagram contains topology only: physics is displayed for selected pair.
    figure = plot_family_tree(_case())
    assert not any('0.271' in str(trace.text) for trace in figure.data)


def test_book_rank_one_normalization_preserves_full_physical_tensor():
    from twin_app.family_tree_graph import normalized_rank_one
    a=(0.2,-0.3,0.4)
    n=(0.0,2.0,0.0)
    aout, nhat = normalized_rank_one(a,n)
    assert abs(sum(x*x for x in nhat)-1.0)<1e-12
    for i in range(3):
        for j in range(3):
            assert abs(a[i]*n[j]-aout[i]*nhat[j])<1e-12
