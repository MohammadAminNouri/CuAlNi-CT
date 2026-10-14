#!/usr/bin/env python3
"""P1 scientific acceptance, numeric residual/orbit/uncertainty evidence.

No material benchmark answers are embedded. A reference cell WITHOUT a
correspondence is never sent to the solver. Results are conditional on entered
lattice, symmetry and C. For a specimen, use --specimen-json and supply C.
"""
from __future__ import annotations
import argparse,json,os,sys,hashlib
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
import numpy as np
import sympy as sp
ROOT=Path(__file__).resolve().parents[1]
for item in (str(ROOT),str(ROOT/'src')):
 if item not in sys.path:sys.path.insert(0,item)
from twin_app.p1_cualni import CELLS,gram,check_cualni_specimen,CrystalInputError
from twin_app.p1_numerics import stretch_from_metrics_independent,root_oracle,symmetry_preserving_sensitivity
from twin_app.p1_evidence import report_orbits,compare_observed_planes,EvidenceError,branch_evidence_status
from twin_app.p1_ensemble import propagate_metric_uncertainty
from app.application import LatticeInput,PhaseInput,TransformationInput,CalculationRequest
from twin_app.scientific_engine import build_twin_family_report
from cualni_cryst.project_io import project_from_dict


def generate_payload(spec):
    def phase(ident,cell,group):return PhaseInput(ident,ident,LatticeInput(*tuple(map(float,cell)),length_unit='angstrom'),group)
    return CalculationRequest('p1_case','Measured/entered transformation; C is user-provided',
       phase('A',spec['parent_cell'],spec['parent_point_group']),
       phase('M',spec['daughter_cell'],spec['daughter_point_group']),
       TransformationInput.from_rows('A_to_M','A','M',spec['correspondence_M_from_A'])).to_project_payload()


def analyze_report(payload,*,max_branches=None):
    state=project_from_dict(payload).project
    tr=state.transformation('A_to_M');a=state.phase(tr.parent_phase_id);m=state.phase(tr.product_phase_id)
    Ga=np.asarray(a.lattice.metric(),float);Gm=np.asarray(m.lattice.metric(),float)
    report=build_twin_family_report(payload,'A_to_M')
    stretches={}
    for item in report.correspondence_variants:
       if item.stretch_variant_index is None:continue
       C=np.array([[float(sp.Rational(x)) for x in row] for row in item.correspondence_M_from_A])
       U=stretch_from_metrics_independent(Ga,Gm,C)
       n=int(item.stretch_variant_index)
       if n in stretches and np.linalg.norm(stretches[n]-U)>2e-7:
          raise AssertionError('Stretch reconstruction inconsistent among correspondence variants')
       stretches[n]=U
    checks=[];failures=[];root_indeterminate=[]
    for f in report.families:
     for p in f.pair_records:
      for c in p.constructions:
       if p.stretch_i not in stretches or p.stretch_j not in stretches:
           failures.append({'pair':p.pair_id,'error':'unmapped stretch used'});continue
       ui,uj=stretches[p.stretch_i],stretches[p.stretch_j]
       D=np.outer(c.a_parent_cartesian,c.n_parent_cartesian)
       rot=(ui+D)@np.linalg.inv(uj)
       orth=float(np.linalg.norm(rot.T@rot-np.eye(3),'fro'))
       det=float(abs(np.linalg.det(rot)-1))
       if orth>1e-6 or det>1e-6:
           failures.append({'pair':p.pair_id,'branch':c.branch,'rotation_orth':orth,'det':det})
       fractions=sorted({float(h.other_variant_volume_fraction) for h in c.habit_solutions})
       oracle=root_oracle(ui,c.a_parent_cartesian,c.n_parent_cartesian,fractions)
       if oracle['status']=='MISMATCH' or oracle['status'].startswith('FAIL'):
           failures.append({'pair':p.pair_id,'branch':c.branch,'root_oracle':oracle})
       if oracle['status'] in ('NEEDS_BOUNDARY_INVESTIGATION','INCONCLUSIVE_CONTINUUM_OR_NEAR_ZERO_POLYNOMIAL'):
           root_indeterminate.append({'pair':p.pair_id,'branch':c.branch,'oracle':oracle['status']})
       checks.append({'pair':p.pair_id,'branch':c.branch,'type':c.classification,
                      'orthogonality_residual':orth,'determinant_residual':det,
                      'rank_one_residual':float(c.rank_one_residual),'root_oracle':oracle,
                      'scientific_evidence':branch_evidence_status(c)})
       if max_branches and len(checks)>=max_branches:break
      if max_branches and len(checks)>=max_branches:break
     if max_branches and len(checks)>=max_branches:break
    orbits=report_orbits(report,Ga,a.symmetry_matrices())
    result={'mode':'source_free_numerical_audit','input_transformation_id':'A_to_M',
     'stretch_variants':len(stretches),'twin_checks':len(checks),
     'worst_twin_orthogonality_residual':max((x['orthogonality_residual'] for x in checks),default=0),
     'classification_counts':dict(Counter(x['type'] for x in checks)),
     'root_boundary_or_continuum_inconclusive':root_indeterminate,'failed_checks':failures,
     'macroscopic_orbits':orbits,'branches':checks,
     'note':'Mathematical compatibility and frame invariance are separate from experimental validation.'}
    return result,report,Ga,Gm


def run_specimen(path,*,uncertainty_samples=0):
    spec=json.loads(Path(path).read_text(encoding='utf-8'))
    validity=check_cualni_specimen(spec)
    if spec.get('correspondence_M_from_A') is None:
       raise SystemExit('STOP: Cu-Al-Ni lattice/reference ONLY. No verified C; predictions cannot be generated. Submit an explicit rational correspondence, with its source.')
    payload=generate_payload(spec)
    result,report,Ga,Gm=analyze_report(payload)
    result['cualni_input_audit']=validity
    evidence=spec.get('observed_interface_planes',[])
    if evidence:
        R=spec.get('measured_OR_M_cart_to_A_cart')
        candidates=[h.habit_normal_parent_cartesian for f in report.families
                     for p in f.pair_records for c in p.constructions for h in c.habit_solutions]
        results=[]
        for observation in evidence:
            try:
                results.append(compare_observed_planes(candidates,observation,
                    parent_metric=Ga,product_metric=Gm,OR_M_cart_to_A_cart=R))
            except EvidenceError as e:results.append({'status':'OBSERVATION_NOT_COMPARABLE','reason':str(e)})
        result['observed_plane_geometry']=results
    else:result['observed_plane_geometry']='NO_INDEPENDENT_OBSERVATIONS'
    if uncertainty_samples:
        uq=spec.get('metric_uncertainty_model')
        if not isinstance(uq,dict):
            raise SystemExit('STOP: metric_uncertainty_model with measured covariance and declared distribution is required; no default measurement uncertainty is invented')
        uc=dict(uq)
        uc['parent_cell']=spec['parent_cell']
        uc['daughter_cell']=spec['daughter_cell']
        uc['parent_family']=spec.get('parent_metric_family')
        uc['daughter_family']=spec.get('daughter_metric_family')
        def solver(p,d):
            amended={**spec,'parent_cell':p,'daughter_cell':d}
            return build_twin_family_report(generate_payload(amended),'A_to_M')
        result['full_engine_metric_uncertainty']=propagate_metric_uncertainty(
            uc,solver,n_samples=uncertainty_samples,seed=int(uq.get('seed',20261014)))
    result['material_status']='EXPLORATORY_PREDICTIONS_CONDITIONAL_ON_INPUT_CORRESPONDENCE_NOT_CERTIFIED'
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--reference-only',action='store_true');g.add_argument('--synthetic',choices=('orthorhombic','monoclinic'))
    g.add_argument('--specimen-json')
    ap.add_argument('--uncertainty-samples',type=int,default=0,help='For specimen input only, user-supplied Gaussian covariance, 2..200 reruns of complete solver')
    args=ap.parse_args()
    if args.reference_only:
        output={'references':{k:{'phase':v.phase,'cell':v.lattice,'point_group':v.point_group,
             'setting':v.cell_setting,'C':None,'OR':None,'source':v.source_url} for k,v in CELLS.items()},
             'status':'REFERENCE_LATTICE_ONLY_NO_PREDICTION'}
    elif args.synthetic:
        product,group=({'orthorhombic':((2.86,3.075,3.195,90.,90.,90.),'mmm'),
                         'monoclinic':((2.91,3.08,3.18,90.,103.,90.),'2/m')})[args.synthetic]
        payload=generate_payload({'parent_cell':(3.,3.,3.,90.,90.,90.),'parent_point_group':'m-3m',
                  'daughter_cell':product,'daughter_point_group':group,
                  'correspondence_M_from_A':[['1','0','0'],['0','1','0'],['0','0','1']]})
        output,_,_,_=analyze_report(payload)
        output['status']='SYNTHETIC_ONLY_NO_EXPERIMENTAL_REFERENCE'
    else:output=run_specimen(args.specimen_json,uncertainty_samples=args.uncertainty_samples)
    final={'p1_version':'2026-10-14','source_commit_expected':'74ce501ba0d53746064d608eb034decd9b02796d',
           'generated_utc':datetime.now(timezone.utc).isoformat(),'results':output}
    target=Path('/tmp/cualni_p1_scientific_acceptance_20261014.json')
    data=json.dumps(final,indent=2,default=str,sort_keys=True)
    target.write_text(data,encoding='utf-8')
    print('P1 RESULTS:',target)
    print('P1 JSON SHA256:',hashlib.sha256(data.encode()).hexdigest())
    print('BRANCHES AUDITED:',output.get('twin_checks','N/A'))
    print('MATHEMATICAL FAILURES:',len(output.get('failed_checks',[])))
    print('DEGENERATE/BOUNDARY CASES:',len(output.get('root_boundary_or_continuum_inconclusive',[])))
    print('EVIDENCE STATUS:',output.get('material_status',output.get('status')))
    if output.get('failed_checks'):
        for x in output['failed_checks'][:8]:print('FAIL:',str(x)[:240])
        raise SystemExit(1)

if __name__=='__main__':main()
