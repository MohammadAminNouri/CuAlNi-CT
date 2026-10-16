"""Scientific classification equivariance (new regression, never a NiTi look-up)."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pytest

from twin_app.symmetry_completion import (
    complete_classifications_by_symmetry, SymmetryClassificationConflict,
)

@dataclass(frozen=True)
class Branch:
    construction_id: str
    classification: str
    classification_status: str
    representations: tuple[str, ...]
    a_parent_cartesian: tuple[float,float,float]
    n_parent_cartesian: tuple[float,float,float]

@dataclass(frozen=True)
class Pair:
    pair_id: str
    stretch_i: int
    stretch_j: int
    constructions: tuple[Branch,...]

@dataclass(frozen=True)
class Family:
    family_id: str
    pair_records: tuple[Pair,...]


def _basic():
    U = {1: np.diag([1.1, 0.9, 1.0]), 2: np.diag([1.0, 1.1, 0.9])}
    one=Branch('A','Type I','cross-locked: independently verified',('I',),(0.1,0.,0.),(0.,1.,0.))
    pending=Branch('B','Exact rank-one relation — classification cross-lock unresolved','unresolved',(),(0.1,0.,0.),(0.,1.,0.))
    families=(Family('F1',(Pair('M1–M2',1,2,(one,)),Pair('M3–M4',1,2,(pending,)))),)
    return families,U


def test_exact_equivariance_transports_one_label_without_touching_geometry():
    fs,U=_basic()
    result,audit=complete_classifications_by_symmetry(fs,parent_metric=np.eye(3),parent_group=(np.eye(3),),stretch_variants=U)
    assert (audit.direct_branches,audit.transported_branches,audit.unresolved_branches)==(1,1,0)
    b=result[0].pair_records[1].constructions[0]
    assert b.classification=='Type I'
    assert 'symmetry-transported' in b.classification_status
    assert b.a_parent_cartesian==fs[0].pair_records[1].constructions[0].a_parent_cartesian
    assert fs[0].pair_records[1].constructions[0].classification.startswith('Exact rank')


def test_same_shear_but_different_tensor_must_not_be_labeled():
    fs,U=_basic()
    b=fs[0].pair_records[1].constructions[0]
    fake=Branch(b.construction_id,b.classification,b.classification_status,b.representations,(0.,0.1,0.),(1.,0.,0.))
    replaced=(Family('F1',(fs[0].pair_records[0],Pair('M3–M4',1,2,(fake,)))),)
    _,audit=complete_classifications_by_symmetry(replaced,parent_metric=np.eye(3),parent_group=(np.eye(3),),stretch_variants=U)
    assert (audit.transported_branches,audit.unresolved_branches)==(0,1)


def test_contradictory_direct_crosslocked_types_stop_release():
    fs,U=_basic()
    first=fs[0].pair_records[0].constructions[0]
    other=Branch('B','Type II','cross-locked: independently verified',('II',),first.a_parent_cartesian,first.n_parent_cartesian)
    bad=(Family('F1',(fs[0].pair_records[0],Pair('M3–M4',1,2,(other,)))),)
    with pytest.raises(SymmetryClassificationConflict):
        complete_classifications_by_symmetry(bad,parent_metric=np.eye(3),parent_group=(np.eye(3),),stretch_variants=U)


def test_nonmetric_point_group_is_rejected():
    fs,U=_basic()
    with pytest.raises(ValueError,match='violates lattice metric'):
        complete_classifications_by_symmetry(fs,parent_metric=np.diag([2.,3.,4.]),parent_group=(np.array([[0,1,0],[1,0,0],[0,0,1]]),),stretch_variants=U)


def test_invalid_or_badly_conditioned_metric_is_rejected():
    fs,U=_basic()
    with pytest.raises(ValueError):
        complete_classifications_by_symmetry(fs,parent_metric=np.diag([1.,1.,1e-16]),parent_group=(np.eye(3),),stretch_variants=U)


def test_real_niti_from_published_inputs_has_full_classification_coverage():
    ROOT=Path(__file__).resolve().parents[2]
    if not (ROOT/'src/cualni_cryst').exists():
        pytest.skip('scientific engine unavailable in standalone package')
    from app.application import CalculationRequest, LatticeInput, PhaseInput, TransformationInput
    from twin_app.scientific_engine import build_twin_family_report
    data=json.loads((ROOT/'data/benchmarks/bhattacharya_niti_twin_habit_v1.json').read_text())
    inp=data['input']
    def phase(label,p):
        return PhaseInput(phase_id=label,label=label,point_group=p['point_group'],lattice=LatticeInput(p['a'],p['b'],p['c'],p['alpha'],p['beta'],p['gamma'],'angstrom'))
    request=CalculationRequest(project_id='niti_covariant_validation',title='NiTi symmetry completion',parent=phase('A',inp['parent']),product=phase('M',inp['product']),transformation=TransformationInput.from_rows('A_to_M','A','M',inp['C_M_from_A']))
    report=build_twin_family_report(request.to_project_payload(),'A_to_M')
    constructions=[c for f in report.families for p in f.pair_records for c in p.constructions]
    from collections import Counter
    assert Counter(c.classification for c in constructions)=={'Type I':36,'Type II':36,'Compound':12}
    assert sum('symmetry-transported' in c.classification_status for c in constructions)==29
    assert sum('cross-locked' in c.classification_status and 'symmetry-transported' not in c.classification_status for c in constructions)==55
    for c in constructions:
        assert c.rank_one_residual<1e-8
        assert c.rotation_residual<1e-8


def test_reversing_variant_order_does_not_fake_identical_interface():
    fs,U=_basic()
    first=fs[0].pair_records[0]
    pending=fs[0].pair_records[1]
    reversed_pair=Pair('M4–M3',2,1,pending.constructions)
    result,audit=complete_classifications_by_symmetry(
        (Family('F1',(first,reversed_pair)),),parent_metric=np.eye(3),
        parent_group=(np.eye(3),),stretch_variants=U)
    assert audit.transported_branches==0
    assert audit.unresolved_branches==1


def test_non_cubic_metric_symmetry_propagation():
    U1=np.diag([1.12,0.96,1.01]); U2=np.diag([0.98,1.08,1.05])
    fs,_=_basic()
    op=np.diag([-1.,1.,-1.])
    a=(0.2,0.3,0.4);n=(0.3,-0.4,0.5)
    first=Branch('A','Type I','cross-locked: direct CT',('I',),a,n)
    other=Branch('B','Exact rank-one relation — classification cross-lock unresolved','missing',(),tuple(op@a),tuple(op@n))
    fam=(Family('X',(Pair('M1–M2',1,2,(first,)),Pair('M3–M4',1,2,(other,)))),)
    result,audit=complete_classifications_by_symmetry(fam,parent_metric=np.diag([4.,9.,16.]),parent_group=(np.eye(3),op),stretch_variants={1:U1,2:U2})
    assert audit.transported_branches==1
    assert result[0].pair_records[1].constructions[0].classification=='Type I'
