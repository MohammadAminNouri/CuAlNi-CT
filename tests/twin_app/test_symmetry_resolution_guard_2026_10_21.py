"""Targeted and end-to-end regression of near-degenerate variant transport.

Published Cu-Al-Ni inputs serve only as geometry inputs; no expected twin
answers are injected into production. The 1e-9 perturbation is SYNTHETIC.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pytest
from twin_app.symmetry_completion import (
    SymmetryClassificationConflict, complete_classifications_by_symmetry,
)


@dataclass(frozen=True)
class Branch:
    construction_id: str
    classification: str
    classification_status: str
    representations: tuple[str, ...]
    a_parent_cartesian: tuple[float, float, float]
    n_parent_cartesian: tuple[float, float, float]


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


def test_different_branches_nearly_coincident_are_not_falsely_equated():
    e=1e-9
    U1=np.diag([.92-e,.92+e,1.17])
    U2=np.diag([.92+e,.92-e,1.17])
    # Dyads differ by less than a previous fixed 1e-7 tolerance,
    # but significantly more than binary64 roundoff.
    a1=(.217,0.,0.)
    a2=(.217,2.06e-9,0.)
    n=(0.,1.,0.)
    b1=Branch('B1','Type I','cross-locked: direct CT',('I',),a1,n)
    b2=Branch('B2','Type II','cross-locked: direct CT',('II',),a2,n)
    fam=(Family('F',(Pair('M1-M2',1,2,(b1,b2)),)),)
    result,audit=complete_classifications_by_symmetry(
        fam,parent_metric=np.eye(3),parent_group=(np.eye(3),),
        stretch_variants={1:U1,2:U2},tolerance=1e-7)
    assert [b.classification for b in result[0].pair_records[0].constructions]==['Type I','Type II']
    assert audit.direct_branches==2 and audit.transported_branches==0


def test_true_geometrical_contradiction_still_fails_closed():
    U1=np.diag([.9,1.,1.1]);U2=np.diag([1.,.9,1.1])
    a=(.2,0.,0.);n=(0.,1.,0.)
    b1=Branch('B1','Type I','cross-locked: direct',('I',),a,n)
    b2=Branch('B2','Type II','cross-locked: direct',('II',),a,n)
    fam=(Family('F',(Pair('M1-M2',1,2,(b1,b2)),)),)
    with pytest.raises(SymmetryClassificationConflict):
        complete_classifications_by_symmetry(
            fam,parent_metric=np.eye(3),parent_group=(np.eye(3),),
            stretch_variants={1:U1,2:U2},tolerance=1e-7)


def test_true_symmetry_transports_still_succeed():
    U1=np.diag([.9,1.,1.1]); U2=np.diag([1.,.9,1.1])
    Q=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,1.]])
    a=np.array([.2,.05,0.]);n=np.array([.03,.99,0.])
    source=Branch('B1','Type I','cross-locked: direct',('I',),tuple(a),tuple(n))
    target=Branch('B2','Unresolved','awaiting crosslock',(),tuple(Q@a),tuple(Q@n))
    fam=(Family('F',(Pair('M1-M2',1,2,(source,)), Pair('M2-M1',2,1,(target,)))),)
    result,audit=complete_classifications_by_symmetry(
        fam,parent_metric=np.eye(3),parent_group=(np.eye(3),Q),
        stretch_variants={1:U1,2:U2},tolerance=1e-7)
    assert audit.direct_branches==1 and audit.transported_branches==1
    assert result[0].pair_records[1].constructions[0].classification=='Type I'


def test_near_roundoff_variant_transport_refuses_to_guess():
    U1=np.diag([.9,1.,1.1])
    U2=U1.copy();U2[0,0]+=1e-15
    b=Branch('B','Unresolved','unresolved',(),(.2,0.,0.),(0.,1.,0.))
    fam=(Family('F',(Pair('M1-M2',1,2,(b,)),)),)
    with pytest.raises(ValueError,match='UNRESOLVED_SYMMETRY_TRANSPORT'):
        complete_classifications_by_symmetry(
            fam,parent_metric=np.eye(3),parent_group=(np.eye(3),),
            stretch_variants={1:U1,2:U2},tolerance=1e-7)


def test_actual_synthetic_cualni_near_symmetry_end_to_end():
    # Exercises the FULL application; old code raises SymmetryClassificationConflict.
    from app.application import CalculationRequest, LatticeInput, PhaseInput, TransformationInput
    from twin_app.scientific_engine import build_twin_family_report
    from twin_app.mode_catalogue import build_mode_catalogue
    f=Path(__file__).resolve().parents[2]/'data'/'benchmarks'/'bhattacharya_cualni_orthorhombic_habit_v1.json'
    if not f.is_file():
        pytest.skip('Published benchmark fixture unavailable; not a mathematical pass')
    source=json.loads(f.read_text(encoding='utf-8'))['input']
    def phase(k,ident):
        x=source[k].copy()
        if k=='product':x['a']=float(x['c']*(1.+1.e-9))
        return PhaseInput(phase_id=ident,label=k,point_group=x['point_group'],
            lattice=LatticeInput(x['a'],x['b'],x['c'],x['alpha'],x['beta'],x['gamma'],'angstrom'))
    req=CalculationRequest(project_id='independent_near_metric_symmetry',title='Synthetic near symmetry',
        parent=phase('parent','A'),product=phase('product','M'),
        transformation=TransformationInput.from_rows('A_to_M','A','M',source['C_M_from_A']))
    report=build_twin_family_report(req.to_project_payload(),'A_to_M')
    assert report.audit.stretch_variant_count==6
    constructions=[c for f in report.families for p in f.pair_records for c in p.constructions]
    assert len(constructions)>=20
    assert all(c.rank_one_residual<1e-6 for c in constructions)
    modes=build_mode_catalogue(report,req.to_project_payload())
    assert modes.mapped_stretch_count==6
