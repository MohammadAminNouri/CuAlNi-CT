from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace as Record

import numpy as np
import sympy as sp

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'twin_app'/'scientific_engine.py'


def _load_independent_crosslock_routines():
    """Exercise the REAL production classification code without loading repo backend.

    Numerical integration is separately mandatory in the full repo. AST
    isolation permits these focused logic tests also in the github.dev ZIP.
    """
    original=ast.parse(ENGINE.read_text(encoding='utf-8'))
    wanted={'_system_has_type_i_mirror_provenance_match','_pair_classification'}
    fns=[node for node in original.body if isinstance(node,ast.FunctionDef) and node.name in wanted]
    assert {f.name for f in fns} == wanted
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*fns],type_ignores=[])
    ast.fix_missing_locations(module)
    env={
        'sp':sp,
        'np':np,
        'matrix_key':lambda value:tuple(sp.Matrix(value)),
        '_system_type_i_geometry_residual':lambda *a:(25.0,25.0),
        '_system_has_type_ii_generator_match':lambda *a:False,
    }
    exec(compile(module,str(ENGINE),'exec'),env)
    return env['_pair_classification']


def _case(*,mirror_match=True,shear=.2804,other_kind=False):
    # Q is a proper pi rotation about z. -Q is the plane mirror normal to z.
    Q=sp.diag(-1,-1,1)
    reflection = -Q if mirror_match else sp.diag(-1,1,1)
    system=Record(system_id='ctI',shear_magnitude=shear,
                  representations=(Record(route='I',generator_parent_symmetry=tuple(
                      tuple(str(reflection[i,j]) for j in range(3)) for i in range(3))),))
    bj=Record(mallard_matches=(Record(kind='II' if other_kind else 'I',parent_symmetry_index=0),),
              compound_by_multiple_twofolds=False)
    return Q,system,bj


def _run(*,mirror_match=True,shear=.2804,other_kind=False):
    classify=_load_independent_crosslock_routines()
    Q,sys,bj=_case(mirror_match=mirror_match,shear=shear,other_kind=other_kind)
    return classify(bj,(sys,),family_has_classical_route=True,shear_magnitude=.2804,
        shear_tolerance=1e-6,pair_plane_product=(1.,0.,0.),
        pair_direction_product=(0.,1.,0.),product_metric=np.eye(3),
        geometry_tolerance_deg=1e-5,parent_group=(Q,))


def test_exact_minus_parent_twofold_licenses_type_i_in_other_frame():
    result=_run()
    assert result[0]=='Type I'
    assert result[3]==('ctI',)
    # The direct physical product-frame test is deliberately nonmatching;
    # no fictitious plane/direction angle is claimed by the provenance lock.
    assert result[4] is None and result[5] is None


def test_wrong_parent_mirror_does_not_pass():
    assert 'unresolved' in _run(mirror_match=False)[0]


def test_shear_agreement_is_still_mandatory():
    assert 'unresolved' in _run(shear=.38)[0]


def test_wrong_mallard_kind_cannot_create_type_i_provenance():
    assert 'unresolved' in _run(other_kind=True)[0]
