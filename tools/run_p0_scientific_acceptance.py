#!/usr/bin/env python3
"""P0 complete metamorphic validation, no published answer constants.

Runs real CuAlNi-CT scientific engine twice under a GL(3,Z) passive product
basis change. Asserts SAME physical U and full twin/habit tensors, correct
classical exclusions, and full report inventory. Writes evidence under /tmp.

The cases are SYNTHETIC, not experimentally verified Cu-Al-Ni.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import sympy as sp
from app.application import CalculationRequest,LatticeInput,PhaseInput,TransformationInput
from twin_app.scientific_engine import build_twin_family_report
from twin_app.p0_correspondence_rebasis import rebase_product_project_payload
from twin_app.p0_classification import NONCONVENTIONAL,screen_parent_twofold
from cualni_cryst.project_io import project_from_dict

I=(("1","0","0"),("0","1","0"),("0","0","1"))
PARENT=(3.,3.,3.,90.,90.,90.)
CASES={
 'orthorhombic':((2.86,3.075,3.195,90.,90.,90.),'mmm'),
 'monoclinic':((2.91,3.08,3.18,90.,103.,90.),'2/m'),
}


def payload_for(product,group):
 def phase(pid,cell,pg):return PhaseInput(pid,pid,LatticeInput(*cell,"angstrom"),pg)
 return CalculationRequest("p0_metamorphic_case","Synthetic coordinate-invariance test",
   phase('A',PARENT,'m-3m'),phase('M',product,group),
   TransformationInput.from_rows('A_to_M','A','M',I)).to_project_payload()


def _symmetric_sqrt(M,inverse=False):
 e,Q=np.linalg.eigh(M)
 if e[0]<=0: raise AssertionError('Non-SPD physical metric')
 return (Q*(e**(-.5 if inverse else .5)))@Q.T


def _stretch(Ga,Gm,C):
 B=_symmetric_sqrt(Ga,True)
 X=B@C.T@Gm@C@B
 return _symmetric_sqrt(.5*(X+X.T))


def _relative(a,b):
 return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b),1e-14))


def physical_inventory(report,payload):
 state=project_from_dict(payload).project
 tr=state.transformation('A_to_M')
 parent=state.phase('A'); product=state.phase('M')
 Ga,Gm=parent.lattice.metric(),product.lattice.metric()
 stretches={}
 for rec in report.correspondence_variants:
  if rec.stretch_variant_index is None:continue
  C=np.array([[float(sp.Rational(x)) for x in row] for row in rec.correspondence_M_from_A])
  U=_stretch(Ga,Gm,C)
  idx=int(rec.stretch_variant_index)
  if idx in stretches and _relative(stretches[idx],U)>1e-7:
   raise AssertionError('Correspondence-to-stretch mapping disagreement')
  stretches[idx]=U
 if len(stretches)!=report.audit.stretch_variant_count:raise AssertionError('Missing mapped U tensors')
 classes=Counter(); observations=[]; hints=[]; components=[]
 for family in report.families:
  for pair in family.pair_records:
   if not pair.constructions:continue
   ui,uj=stretches[pair.stretch_i],stretches[pair.stretch_j]
   for c in pair.constructions:
    classes[c.classification]+=1
    if c.classification==NONCONVENTIONAL:
     screen=screen_parent_twofold(ui,uj,parent_metric=Ga,
        parent_group=parent.symmetry_matrices())
     if not screen.conventional_twofold_excluded:
      raise AssertionError('Nonconventional label has a parent twofold')
    if c.classification not in ('Type I','Type II','Compound',NONCONVENTIONAL):
     hints.append(f'{pair.pair_id}: {c.classification}')
    D=np.outer(c.a_parent_cartesian,c.n_parent_cartesian)
    R=(ui+D)@np.linalg.inv(uj)
    if _relative(R.T@R,np.eye(3))>1e-6 or abs(np.linalg.det(R)-1)>1e-6:
     raise AssertionError('Invalid rank-one proper rotation')
    # Determine tensor ordering solely by physical eigenframe matrices.
    akey=tuple(np.round(ui.flatten(),8)); bkey=tuple(np.round(uj.flatten(),8))
    reversed_order=akey>bkey
    if reversed_order:
     D=-R.T@D
     akey,bkey=bkey,akey
    data=(akey,bkey,D)
    observations.append(data)
    for h in c.habit_solutions:
     lam=float(h.other_variant_volume_fraction)
     H=np.outer(h.shape_vector_parent_cartesian,h.habit_normal_parent_cartesian)
     components.append((akey,bkey,D,1-lam if reversed_order else lam,H))
 return {
    'stretch_variants':list(stretches.values()),
    'twin_tensors':observations,'habit_tensors':components,
    'classifications':dict(classes),'uncertified':hints,
    'warnings':list(report.audit.warnings),
 }


def _as_array(row):
 return np.concatenate([np.array(row[0]),np.array(row[1]),np.array(row[2]).flatten()])


def _matched(a,b,*,habit=False,tol=5e-6):
 if len(a)!=len(b):raise AssertionError(f'Solution-set count differs: {len(a)} vs {len(b)}')
 def vector(row):
  if not habit:return _as_array(row)
  return np.concatenate([_as_array(row[:3]),[row[3]],np.array(row[4]).flatten()])
 pool=[vector(x) for x in b]
 used=set();worst=0.
 for entry in a:
  v=vector(entry)
  ranked=sorted(((float(np.linalg.norm(v-w)/max(np.linalg.norm(v),np.linalg.norm(w),1.)),i) for i,w in enumerate(pool) if i not in used))
  if not ranked or ranked[0][0]>tol:
   raise AssertionError(f'Physical {"habit" if habit else "twin"} tensor changed under rebasing: {ranked[:1]}')
  worst=max(worst,ranked[0][0]);used.add(ranked[0][1])
 return worst


def execute_case(name,product,group,P):
 base=payload_for(product,group)
 amended,rebase_audit=rebase_product_project_payload(base,'A_to_M',P)
 a=build_twin_family_report(base,'A_to_M')
 b=build_twin_family_report(amended,'A_to_M')
 A=physical_inventory(a,base);B=physical_inventory(b,amended)
 if A['classifications']!=B['classifications']:
  raise AssertionError(f'Physical classification histogram changed: {A["classifications"]} vs {B["classifications"]}')
 if A['uncertified'] or B['uncertified']:
  raise AssertionError(f'Unresolved classical evidence: {A["uncertified"][:4]} / {B["uncertified"][:4]}')
 U_diff=_matched([((),(),u) for u in A['stretch_variants']],
                 [((),(),u) for u in B['stretch_variants']],tol=5e-7)
 twin_diff=_matched(A['twin_tensors'],B['twin_tensors'])
 habit_diff=_matched(A['habit_tensors'],B['habit_tensors'],habit=True)
 return dict(case=name,rebase_P=np.asarray(P,dtype=int).tolist(),
             stretches=len(A['stretch_variants']),
             classification_counts=A['classifications'],
             full_twin_branches=len(A['twin_tensors']),
             full_habit_solutions=len(A['habit_tensors']),
             max_stretch_rebase_difference=U_diff,
             max_twin_tensor_rebase_difference=twin_diff,
             max_habit_tensor_rebase_difference=habit_diff,
             physical_metric_rebase_residual=rebase_audit.metric_residual,
             audit_warnings=A['warnings'],
             evidence_scope='SYNTHETIC_MATHEMATICAL_TEST_NOT_EXPERIMENTAL')


def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--profile',choices=('quick','deep'),default='quick')
 args=parser.parse_args()
 choices=['orthorhombic'] if args.profile=='quick' else ['orthorhombic','monoclinic']
 # Non-identity matrices with det=1, including shear and right-handed axis swap.
 matrices={
  'orthorhombic':[[1,1,0],[0,1,0],[0,0,1]],
  'monoclinic':[[0,1,0],[1,0,0],[0,0,-1]],
 }
 output={'audit':'CuAlNi-CT P0 GL3Z full-pipeline',
   'timestamp_utc':datetime.now(timezone.utc).isoformat(),'profile':args.profile,'results':[],
   'scientific_claim':'synthetic only, NOT experimentally verified'}
 for name in choices:
  print('RUN',name,'full correspondence and symmetry rebasing',flush=True)
  t0=time.monotonic()
  try:
   result=execute_case(name,*CASES[name],matrices[name]);result['runtime_seconds']=round(time.monotonic()-t0,2)
   output['results'].append({'status':'pass',**result})
   print('PASS',name,'branches',result['full_twin_branches'],'habits',result['full_habit_solutions'],
      'maximum physical twin mismatch',f'{result["max_twin_tensor_rebase_difference"]:.3e}',flush=True)
  except Exception as exc:
   output['results'].append({'status':'fail','case':name,'error':str(exc)})
   print('FAIL',name,type(exc).__name__,str(exc),flush=True)
 path=Path('/tmp/cualni_p0_rebase_acceptance_20261014.json')
 path.write_text(json.dumps(output,indent=2,default=float))
 print('EVIDENCE',path,flush=True)
 if any(r['status']!='pass' for r in output['results']):
  raise SystemExit('P0 ACCEPTANCE FAILED — preserve evidence, do not push')
 print('P0 SYNTHETIC INVARIANCE PASSED; EXPERIMENTAL VALIDATION NOT ESTABLISHED')

if __name__=='__main__':main()
