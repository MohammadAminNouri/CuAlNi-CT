"""Mutation-focused local checks; actual cases run via CLI --case both.
No material-specific subtype answers are inserted into production.
"""
import numpy as np
from dataclasses import dataclass
import classification_retention_audit_2026_10_22 as A


def test_dyad_is_gauge_invariant_but_sign_reversal_is_not():
    d1=A.normalized_dyad([2,0,0],[0,3,0])
    d2=A.normalized_dyad([-6,0,0],[0,-1,0])
    d3=A.normalized_dyad([2,0,0],[0,-3,0])
    assert A.dyad_error(d1,d2)==0
    assert A.dyad_error(d1,d3)>1.9


def test_swapped_plane_and_direction_do_not_match():
    a=A.normalized_dyad([1,0,0],[0,1,0])
    b=A.normalized_dyad([0,1,0],[1,0,0])
    assert A.dyad_error(a,b)>1


def test_nonzero_finite_rankone_tensor():
    assert np.linalg.matrix_rank(A.normalized_dyad([1,2,3],[2,1,0]))==1


@dataclass(frozen=True)
class FakeBranch:
    construction_id:str='P1.B1'
    branch:int=1
    classification:str='Exact rank-one relation -- unsupported subtype'
    classification_status:str='unresolved'
    representations:tuple=()
    continuum_fraction:bool=False
    habit_solutions:tuple=()


@dataclass(frozen=True)
class FakePair:
    pair_id:str='M1-M2'
    stretch_i:int=1
    stretch_j:int=2


@dataclass(frozen=True)
class FakeFamily:
    family_id:str='F1'


@dataclass(frozen=True)
class FakeEntry:
    operator_family:str
    pair:FakePair
    construction:FakeBranch


@dataclass(frozen=True)
class FakeMode:
    mode_id:str='Mode 1'
    types:tuple=('Unresolved',)
    entries:tuple=()
    def by_type(self,t):
        return tuple(e for e in self.entries if A.label_and_status(e.construction)[0]==t)


@dataclass(frozen=True)
class FakeCatalogue:
    modes:tuple


def state():
    b=FakeBranch();p=FakePair();f=FakeFamily()
    k=('F1','M1-M2','P1.B1',1)
    m=FakeMode(entries=(FakeEntry('F1',p,b),))
    return k,(f,p,b),FakeCatalogue((m,))


def test_exact_retention_passes_for_unresolved_branch():
    k,source,cat=state()
    mode_records=A.build_items_from_catalogue(cat)
    rows,err=A.check_retention({k:source},mode_records,cat)
    assert not err
    assert len(rows)==1 and rows[0]['type']=='Unresolved'


def test_missing_branch_fails_closed():
    k,source,cat=state()
    _,err=A.check_retention({k:source},{},cat)
    assert any('missing or duplicated' in x['problem'] for x in err)


def test_duplicate_branch_fails_closed():
    k,source,cat=state()
    row=A.build_items_from_catalogue(cat)[k][0]
    _,err=A.check_retention({k:source},{k:[row,row]},cat)
    assert any('missing or duplicated' in x['problem'] for x in err)


def test_invented_entry_fails_closed():
    k,source,cat=state()
    other=('F1','M1-M3','P1.B1',1)
    _,err=A.check_retention({}, {other:A.build_items_from_catalogue(cat)[k]},cat)
    assert any('invented' in x['problem'] for x in err)


def test_changed_habit_inventory_fails_closed():
    k,source,cat=state()
    from dataclasses import replace
    new_b=replace(source[2],habit_solutions=('fake altered habit',))
    new_cat=FakeCatalogue((FakeMode(entries=(FakeEntry('F1',source[1],new_b),)),))
    _,err=A.check_retention({k:source},A.build_items_from_catalogue(new_cat),new_cat)
    assert any('changed' in x['problem'] for x in err)


def test_unresolved_does_not_get_classical_label():
    b=FakeBranch()
    assert A.label_and_status(b)==('Unresolved',False)


def test_classical_label_with_invalid_provenance_fails():
    k,source,cat=state()
    from dataclasses import replace
    claim=replace(source[2],classification='Type II',classification_status='warning: cross-locked appears elsewhere',representations=('II',))
    mode=FakeMode(entries=(FakeEntry('F1',source[1],claim),),types=('Type II',))
    _,err=A.check_retention({k:(source[0],source[1],claim)}, A.build_items_from_catalogue(FakeCatalogue((mode,))), FakeCatalogue((mode,)))
    assert any('lacks allowed direct/transported evidence' in x['problem'] for x in err)
