#!/usr/bin/env python3
"""Read-only near-symmetry classification investigation for CuAlNi-CT.

Reproduce the epsilon=1e-9 synthetic Cu-Al-Ni 2H perturbation already used in
numerical_boundary_falsification.py. Capture the exact family/branch data at
symmetry classification *before* the existing conflict is thrown. No monkey-
patch persists beyond this Python process. Do not change any solver tolerances
or classify an unresolved branch as Compound simply to obtain green results.
"""
from __future__ import annotations
import argparse
import json
import math
import traceback
from pathlib import Path

import numpy as np
import mpmath as mp

ROOT=Path(__file__).resolve().parent
# Support placement in repository root or a local audit subdirectory.
if not (ROOT/'data/benchmarks/bhattacharya_cualni_orthorhombic_habit_v1.json').exists():
    ROOT=ROOT.parent
if not (ROOT/'data/benchmarks/bhattacharya_cualni_orthorhombic_habit_v1.json').exists():
    raise SystemExit('Place this script in the CuAlNi-CT repository root or one directory underneath it.')
import sys
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
import numerical_boundary_falsification as audit
import twin_app.symmetry_completion as completion
from twin_app.scientific_engine import build_twin_family_report


def collect_entries(families):
    entries=[]
    for family in families:
        for pair in family.pair_records:
            if not pair.constructions:
                continue
            for branch in pair.constructions:
                entries.append({
                    'family': str(family.family_id), 'pair':str(pair.pair_id),
                    'construction':str(branch.construction_id),
                    'stretch_i':int(pair.stretch_i),'stretch_j':int(pair.stretch_j),
                    'class':str(branch.classification),
                    'status':str(branch.classification_status),
                    'dyad':np.outer(np.asarray(branch.a_parent_cartesian,float),
                                    np.asarray(branch.n_parent_cartesian,float)),
                    'shear':float(branch.shear_magnitude),
                    'rank_one_residual':float(branch.rank_one_residual),
                    'rotation_residual':float(branch.rotation_residual),
                })
    return entries


def rel(A,B):
    na=float(np.linalg.norm(A,'fro'));nb=float(np.linalg.norm(B,'fro'))
    return float(np.linalg.norm(A-B,'fro')/max(na,nb,1e-14))


def diagnose(kws,exc,original):
    families=kws['families'];tol=float(kws['tolerance']);
    stretches={int(k):np.asarray(v,float) for k,v in kws['stretch_variants'].items()}
    operations=completion._operations(kws['parent_metric'],kws['parent_group'],tol)
    entries=collect_entries(families)
    direct=[e for e in entries if e['class'] in completion.LABELS and 'cross-locked' in e['status']]
    gaps=[(rel(stretches[i],stretches[j]),i,j) for i in stretches for j in stretches if i<j]
    gaps.sort()
    min_gap=gaps[0][0] if gaps else None
    # Show whether classification matching is capable of uniquely identifying a
    # transformed stretch. A single Q with >1 matches is intrinsically ambiguous.
    collisions=[]
    for i,U in stretches.items():
        for oi,Q in enumerate(operations):
            V=Q@U@Q.T
            choices=sorted((rel(V,T),j) for j,T in stretches.items())
            if len(choices)>1 and choices[1][0]<=tol:
                collisions.append({'source_stretch':i,'operation_index':oi,
                                   'first_two_matches':[{'target':j,'relative_error':d} for d,j in choices[:2]]})
    # Capture direct contradictory witnesses with ALL three independent residuals.
    conflicts=[]
    for target in entries:
        if target['class'] not in completion.LABELS:continue
        witnesses=[]
        for source in direct:
            if source is target:continue
            best=None
            for oi,Q in enumerate(operations):
                ei=rel(Q@stretches[source['stretch_i']]@Q.T,stretches[target['stretch_i']])
                ej=rel(Q@stretches[source['stretch_j']]@Q.T,stretches[target['stretch_j']])
                ed=rel(Q@source['dyad']@Q.T,target['dyad'])
                if max(ei,ej,ed)>tol:continue
                t=(max(ei,ej,ed),oi,ei,ej,ed)
                if best is None or t<best:best=t
            if best is not None:
                _,oi,ei,ej,ed=best
                witnesses.append({'source':f"{source['family']}:{source['pair']}:{source['construction']}",
                                  'source_class':source['class'],
                                  'target':f"{target['family']}:{target['pair']}:{target['construction']}",
                                  'target_class':target['class'],
                                  'source_stretch_pair':[source['stretch_i'],source['stretch_j']],
                                  'target_stretch_pair':[target['stretch_i'],target['stretch_j']],
                                  'operation_index':oi,'stretch_i_relative_error':ei,
                                  'stretch_j_relative_error':ej,'dyad_relative_error':ed})
        contrary=[w for w in witnesses if w['source_class']!=target['class']]
        if contrary:
            conflicts.append({'target':f"{target['family']}:{target['pair']}:{target['construction']}",
                              'target_class':target['class'],
                              'target_status':target['status'],
                              'target_shear':target['shear'],
                              'contradictory_witnesses':contrary})
    candidate_tol=(min(tol, min_gap/8) if min_gap is not None else tol)
    retry={}
    # This is ONLY a diagnostic counterfactual. No changes to the physical solver
    # and no claim that an arbitrary narrow tolerance is the final solution.
    for t in [candidate_tol,1e-10]:
        if not 0<t<=1e-5:continue
        try:
            _,a=original(
                families,parent_metric=kws['parent_metric'],
                parent_group=kws['parent_group'],stretch_variants=stretches,tolerance=t)
            outcome={'result':'NO_CLASSIFICATION_CONFLICT',
                     'direct':a.direct_branches,'transported':a.transported_branches,
                     'unresolved':a.unresolved_branches}
        except Exception as ex:
            outcome={'result':'EXCEPTION','exception':repr(ex)}
        retry[format(t,'.15g')]=outcome
    report={'original_exception':repr(exc),
            'original_classification_tolerance':tol,
            'number_of_stretches':len(stretches),
            'minimum_distinct_relative_stretch_gap':min_gap,
            'closest_stretch_pair':list(gaps[0][1:]) if gaps else None,
            'gap_to_tolerance_ratio':min_gap/tol if min_gap is not None else None,
            'construction_count':len(entries),'direct_crosslocked_count':len(direct),
            'classification_counts':{kind:sum(e['class']==kind for e in entries)
                                     for kind in ['Type I','Type II','Compound']},
            'multiple_match_cases':len(collisions),
            'sample_multiple_matches':collisions[:12],
            'opposite_direct_classification_witnesses':conflicts,
            'alternative_tolerance_diagnostics':retry,
            'caution':'This is a synthetic metric perturbation and a diagnostic only; tighter tolerance is not a certified repair.'}
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epsilon',default='1e-9')
    args=parser.parse_args()
    mp.mp.dps=85
    _,gap,state=audit.cubic_mp_orbit(audit.bench_input(),args.epsilon)
    orig=completion.complete_classifications_by_symmetry
    captured={}
    def intercept(families,*,parent_metric,parent_group,stretch_variants,tolerance=1e-7):
        kw=dict(families=families,parent_metric=parent_metric,parent_group=parent_group,
                stretch_variants=stretch_variants,tolerance=tolerance)
        try:return orig(**kw)
        except completion.SymmetryClassificationConflict as e:
            captured['data']=diagnose(kw,e,orig)
            raise
    completion.complete_classifications_by_symmetry=intercept
    outcome='UNKNOWN'; error=None
    try:
        report=build_twin_family_report(audit.independent.payload(state),'A_to_M')
        outcome='APPLICATION_COMPLETED'
        captured['report']={'stretch_variant_count':report.audit.stretch_variant_count,
                            'warnings':list(report.audit.warnings)}
    except Exception as exc:
        outcome='APPLICATION_EXCEPTION'
        error=repr(exc)
        if 'data' not in captured:captured['traceback']=traceback.format_exc()[-7000:]
    finally:
        completion.complete_classifications_by_symmetry=orig
    result={'epsilon':args.epsilon,'independent_orbit_min_frobenius_gap':gap,
            'status':outcome,'error':error,**captured}
    output=ROOT/'audit_findings'/'boundary_classification_witness_2026_10_21.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True,default=str)+'\n')
    d=result.get('data',{})
    print('STATUS:',outcome,'EXCEPTION:',error)
    print('MIN RELATIVE GAP:',d.get('minimum_distinct_relative_stretch_gap'))
    print('ORIGINAL CLASSIFICATION TOL:',d.get('original_classification_tolerance'))
    print('AMBIGUOUS SYMMETRY MATCHES:',d.get('multiple_match_cases'))
    print('DIRECT CONTRADICTORY WITNESS TARGETS:',len(d.get('opposite_direct_classification_witnesses',[])))
    print('RECHECK AT SMALLER TOLERANCES:',d.get('alternative_tolerance_diagnostics'))
    print('EVIDENCE FILE:',output)
    # Fail audibly when the original exception has been reproduced; diagnostic
    # success is creation of the evidence, not falsely claiming science passes.
    if not d and outcome=='APPLICATION_EXCEPTION':return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
