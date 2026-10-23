#!/usr/bin/env python3
"""Adversarial scientific boundary audit; NO alteration to production source.

Input provenance: published Cu-Al-Ni DO3->2H orthorhombic input from the
repository's Bhattacharya benchmark, with explicitly SYNTHETIC continuous
metric perturbations (not additional measured specimens). Only `input` read.

Independent oracle: mpmath arbitrary precision for exact symmetry orbit of
metric-derived stretches and the characteristic determinant root polynomial.
Checks PTMC and Ball-James variant deduplication disagreement and near-tangent
PTMC laminate fraction completeness with high-precision certification of the
actual binary64-encoded numerical problem.  Flags ambiguous answers instead
of calling near-roundoff discrepancies physical defects.

Pairs and dyads used for tangent construction are determined through the
independent geometric invariant-plane oracle bundled in the earlier audit.
No output of the production solvers is used to generate oracle roots.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
import math
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.optimize import minimize_scalar
import sympy as sp

import upstream_twin_falsification as independent

ROOT = Path(__file__).resolve().parent
MP_DPS = 85
EPSILONS = ['1e-4','1e-6','1e-7','1e-8','3e-9','1e-9','3e-10','1e-10','0']
TANGENCY_EPS = ['-1e-6','-1e-8','-1e-10','0','1e-10','1e-8','1e-6']


def bench_input():
    path = ROOT/'data/benchmarks/bhattacharya_cualni_orthorhombic_habit_v1.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    assert 'input' in data
    return data['input']  # NEVER LOOK AT expected


def mp_from_float(x):
    # Certified arithmetic for the ACTUALLY ENCODED binary64 input.
    p,q = float(x).as_integer_ratio()
    return mp.mpf(p) / mp.mpf(q)


def mp_matrix_from_float(x):
    arr = np.asarray(x,float)
    return mp.matrix([[mp_from_float(v) for v in row] for row in arr])


def mp_frob(A,B):
    return mp.sqrt(mp.fsum((A[i,j]-B[i,j])**2 for i in range(3) for j in range(3)))


def cubic_mp_orbit(inp, eps):
    """The reference 2H C is exact; metric entries are high-precision decimals.

    Alter ONLY product a/c ratio to probe 2H nearly degenerate metric. Note:
    this is a synthetic perturbation of a literature-derived cell.
    """
    p=inp['parent']; pr=inp['product']
    a0=mp.mpf(str(p['a']))
    c=mp.mpf(str(pr['c']))
    a=c*(1+mp.mpf(eps))
    b=mp.mpf(str(pr['b']))
    C=mp.matrix([[mp.mpf(str(sp.Rational(t))) for t in row]
                 for row in inp['C_M_from_A']])
    M=mp.diag([a*a,b*b,c*c])
    G=C.T*M*C/(a0*a0)
    lam,X=mp.eigsy(G)
    U=X*mp.diag([mp.sqrt(lam[k]) for k in range(3)])*X.T
    variants=[]
    for q in independent.PARENT_SO3:
        Q=mp.matrix([[mp.mpf(int(z)) for z in row] for row in q])
        V=Q*U*Q.T
        if all(mp_frob(V,u)>mp.mpf('1e-45') for u in variants):
            variants.append(V)
    mind=min((mp_frob(x,y) for i,x in enumerate(variants) for y in variants[i+1:]),default=mp.inf)
    # Independent double-precision experimental state (no expected output).
    new=copy.deepcopy(inp)
    new['product']['a']=float(a)
    new['product']['b']=float(b)
    new['product']['c']=float(c)
    return variants,float(mind),new


def find_floats_variants(state):
    from cualni_cryst.ptmc_adapter import _build_variants
    from cualni_cryst.ball_james_adapter import generate_stretch_variants
    from cualni_cryst.stretch import stretch_from_metrics
    from cualni_cryst.correspondence import Correspondence
    # Production functions are used as TEST SUBJECTS, never to build the oracle.
    M_a=independent.metric_from_lengths(state['parent'])
    M_m=independent.metric_from_lengths(state['product'])
    C=np.asarray(sp.Matrix(state['C_M_from_A']),float)
    ops=tuple(independent.PARENT_SO3)
    ptmc=_build_variants(M_a,M_m,C,ops,algebraic_tolerance=1e-10)
    bj=generate_stretch_variants(
        stretch_from_metrics(M_a,M_m,Correspondence(sp.Matrix(state['C_M_from_A']))),
        tuple(enumerate(ops)),tolerance=1e-9)
    return ptmc,bj


def app_report(state):
    from twin_app.scientific_engine import build_twin_family_report
    return build_twin_family_report(independent.payload(state),'A_to_M')


def issue(report,priority,kind,where,observed,reference,interpretation,files):
    finding=dict(priority=priority,kind=kind,where=where,observed=observed,
                 reference=reference,interpretation=interpretation,likely_files=files,
                 state='OPEN_UNREVIEWED',not_certified_production_defect=True)
    finding['id']=hashlib.sha256(json.dumps(finding,sort_keys=True,default=str).encode()).hexdigest()[:13]
    report['findings'].append(finding)
    print(f'  [{priority}] {kind}: {where}',flush=True)


def variant_stage(report,app_levels):
    print('\n=== STAGE 1: NEAR-ENHANCED METRIC SYMMETRY (2H, SYNTHETIC) ===',flush=True)
    base=bench_input()
    for eps in EPSILONS:
        v,sep,state=cubic_mp_orbit(base,eps)
        reference_count=len(v)
        row=dict(epsilon=eps,certified_distinct_stretches=reference_count,
                 min_certified_frobenius_gap=sep,point_group='mmm',
                 original_product_a=base['product']['a'],
                 synthetic_product_a=state['product']['a'])
        try:
            ptmc,bj=find_floats_variants(state)
            row['ptmc_variant_count']=len(ptmc)
            row['ball_james_variant_count']=len(bj)
            if len(ptmc)!=len(bj):
                issue(report,'P1','variant_generators_disagree',f'epsilon={eps}',
                      {'PTMC':len(ptmc),'BallJames':len(bj)},
                      {'MP_unique':reference_count,'smallest_gap':sep},
                      'Potential cross-theory deduplication mismatch. Check current floating precision and default tolerance policy BEFORE patching.',
                      ['src/cualni_cryst/ptmc_adapter.py','src/cualni_cryst/ball_james_adapter.py'])
            if len(ptmc)!=reference_count or len(bj)!=reference_count:
                priority='P2' if sep<3e-8 else 'P1'
                issue(report,priority,'distinct_variant_numerically_collapsed',f'epsilon={eps}',
                      {'PTMC':len(ptmc),'BallJames':len(bj)},
                      {'MP_unique':reference_count,'smallest_gap':sep},
                      'Exact distinct stretches may be deliberately merged by numerical tolerance; determine whether app discloses resolution limit.',
                      ['src/cualni_cryst/ptmc_adapter.py','src/cualni_cryst/ball_james_adapter.py'])
        except Exception as exc:
            row['backend_error']=repr(exc)
            issue(report,'P1','variant_generator_exception',f'epsilon={eps}',repr(exc),
                  {'MP_unique':reference_count,'smallest_gap':sep},
                  'Investigate input-domain failure versus numerical instability.',
                  ['src/cualni_cryst/ptmc_adapter.py','src/cualni_cryst/ball_james_adapter.py'])
        print('eps=',eps,'MP=',reference_count,'min_gap=%.3e'%sep,
              'PTMC=',row.get('ptmc_variant_count','ERR'),
              'BJ=',row.get('ball_james_variant_count','ERR'),flush=True)
        # The costly UI/full theory pipeline is intentionally limited.
        if eps in app_levels:
            print('  Checking complete user-facing application...',flush=True)
            try:
                app=app_report(state)
                row['app_variant_count']=app.audit.stretch_variant_count
                row['app_pair_constructions']=sum(len(pair.constructions)
                         for f in app.families for pair in f.pair_records)
                row['app_warnings']=list(app.audit.warnings)
                if app.audit.stretch_variant_count!=reference_count:
                    issue(report,'P2' if sep<3e-8 else 'P1',
                          'app_loses_distinct_variants',f'epsilon={eps}',
                          row['app_variant_count'],reference_count,
                          'Check whether app explicitly warns that sub-tolerance physical variants were numerically merged.',
                          ['twin_app/scientific_engine.py','twin_app/scientific_models.py'])
            except Exception as exc:
                row['app_exception']=repr(exc)
                issue(report,'P1','full_app_exception_near_metric_symmetry',f'epsilon={eps}',
                      dict(exception=repr(exc),trace=traceback.format_exc()[-3500:]),
                      dict(MP_unique=reference_count,min_gap=sep),
                      'Determine whether the full app fails closed with actionable diagnostics or crashes on valid nearly symmetric lattice metrics.',
                      ['twin_app/scientific_engine.py','src/cualni_cryst/ptmc_adapter.py'])
        report['variant_rows'].append(row)
    return report


def determinant_mp(F0,D,f):
    F=F0 + f*D
    return mp.det(F.T*F - mp.eye(3))


def middle_mu_mp(F0,D,f):
    F=F0+f*D
    eigen,_=mp.eigsy(F.T*F)
    return eigen[1]


def isolate_mp_roots(F0,D):
    """High-precision quadratic polynomial from ACTUAL binary64 encoded tensors.

    No import of production PTMC polynomial, no planted root or benchmark
    expected value. Validate quadratic assumption at 1/7 and 5/7.
    """
    F=mp_matrix_from_float(F0)
    B=mp_matrix_from_float(D)
    z=mp.mpf(0);one=mp.mpf(1)
    y0=determinant_mp(F,B,z)
    yp=determinant_mp(F,B,one)
    ym=determinant_mp(F,B,-one)
    aa=(yp+ym)/2-y0
    bb=(yp-ym)/2
    cc=y0
    def poly(f):return aa*f*f+bb*f+cc
    pver=max(abs(determinant_mp(F,B,t)-poly(t)) for t in (mp.mpf(1)/7,mp.mpf(5)/7))
    if pver>mp.mpf('1e-24'):
        return dict(valid=False,reason='rank_one_polynomial_identity_failed',residual=str(pver))
    scale=max(abs(aa),abs(bb),abs(cc),mp.mpf('1e-90'))
    if abs(aa)<=scale*mp.mpf('1e-35'):
        if abs(bb)<=scale*mp.mpf('1e-35'):
            return dict(valid=True,continuum=abs(cc)<=scale*mp.mpf('1e-32'),
                unresolved=True,reason='linear_or_nearly_constant_determinant',coefficients=[str(x) for x in (cc,bb,aa)])
        roots=[-cc/bb]
        discr=None
    else:
        discr=bb*bb-4*aa*cc
        if discr<0:roots=[]
        elif discr==0:roots=[-bb/(2*aa)]
        else:
            sr=mp.sqrt(discr)
            t=-mp.sign(bb)*(abs(bb)+sr)/2
            roots=[t/aa,cc/t] if t else [(-bb+sr)/(2*aa),(-bb-sr)/(2*aa)]
    good=[]
    for rt in roots:
        if rt>=-mp.mpf('1e-20') and rt<=1+mp.mpf('1e-20'):
            f=max(mp.mpf(0),min(mp.mpf(1),rt))
            mu=middle_mu_mp(F,B,f)
            resid=abs(mu-1)
            if resid<=mp.mpf('1e-21'):
                if all(abs(f-zz['frac_mp'])>mp.mpf('1e-25') for zz in good):
                    good.append(dict(frac_mp=f, mu_resid_mp=resid))
            else:
                # determinant zero but not middle stretch == 1, forbidden as a habit root
                continue
    good.sort(key=lambda x:x['frac_mp'])
    return dict(valid=True,continuum=False,discriminant=str(discr) if discr is not None else None,
                coefficients=[str(x) for x in (cc,bb,aa)],
                roots=[float(x['frac_mp']) for x in good],
                middle_mu_residuals=[float(x['mu_resid_mp']) for x in good],
                root_separation=float(good[1]['frac_mp']-good[0]['frac_mp']) if len(good)==2 else None,
                verification_residual=str(pver))


def find_tangent_from_reference(inp):
    """Select by measured-input curve, not by planted expected habit roots."""
    variants,_=independent.metric_variants(inp)
    candidates=[]
    for i in range(len(variants)):
        for j in range(i+1,len(variants)):
            try:
                branches,_=independent.independent_interfaces(variants[i],variants[j])
            except ArithmeticError:continue
            for branch_index,branch in enumerate(branches):
                U=variants[i]
                D=branch['dyad']
                def m(f):return float(np.sort(np.linalg.svd(U+float(f)*D,compute_uv=False))[1])
                opt=minimize_scalar(lambda f:-m(f),bounds=(.2,.8),method='bounded',options={'xatol':1e-12})
                f=float(opt.x);peak=m(f);edges=(m(0),m(1))
                if opt.success and .2<f<.8 and peak>max(edges)+.002:
                    candidates.append((peak-max(edges),i,j,branch_index,f,peak,U,D))
    if not candidates:
        raise RuntimeError('No interior middle-stretch tangent candidate found')
    candidates.sort(key=lambda z:(-z[0],z[1],z[2],z[3]))
    return candidates[0]


def tangency_stage(report):
    print('\n=== STAGE 2: NEAR-MERGING HABIT FRACTIONS (2H, SYNTHETIC) ===',flush=True)
    inp=bench_input()
    delta,i,j,k,fmax,tcrit,U,D=find_tangent_from_reference(inp)
    print('Published starting parent a0=',inp['parent']['a'],
          'chosen independent pair=',[i,j], 'branch=',k,
          'tangent_fraction=',fmax,'tcrit=',tcrit,flush=True)
    print('WARNING: re-scaled parent a0 is a SYNTHETIC numerical boundary probe, NOT a measured lattice.',flush=True)
    from cualni_cryst.ptmc_adapter import solve_affine_ptmc_parameters
    for eps in TANGENCY_EPS:
        t=tcrit*(1+float(eps))
        F=U/t
        inc=D/t
        truth=isolate_mp_roots(F,inc)
        row=dict(epsilon=eps,parent_a_synthetic=inp['parent']['a']*t,
                 original_parent_a=inp['parent']['a'],
                 original_pair=[i,j],branch=k,
                 center_fraction=fmax,tangent_scale=tcrit,
                 oracle=truth)
        try:
            actual=solve_affine_ptmc_parameters(F,inc,
                  dilatational_factor=1.0,parameter_name='fraction',domain=(0.,1.),
                  polynomial_tolerance=1e-10,middle_stretch_tolerance=1e-7)
            predicted=list(actual.discrete_roots)
            row['app_roots']=predicted
            row['app_continuum']=bool(actual.continuum)
            row['app_quadratic_coeffs']=list(actual.polynomial_coefficients_ascending)
            if truth.get('valid') and not truth.get('unresolved'):
                roots=truth.get('roots',[])
                sep=truth.get('root_separation')
                if truth.get('continuum') is False and actual.continuum:
                    issue(report,'P2','false_continuum_near_tangency',f'tangent-epsilon={eps}',
                           'production continuum',truth,
                           'Investigate absolute/relative coefficients and near-zero poly floor; may be indistinguishable from continuum at binary64.',
                           ['src/cualni_cryst/ptmc_adapter.py'])
                missing=[x for x in roots if not any(abs(x-y)<1e-5 for y in predicted)]
                extra=[x for x in predicted if not any(abs(x-y)<1e-5 for y in roots)]
                if missing or extra:
                    # Very close roots and thresholds are not unambiguous proof of solver bug.
                    distance=min([abs(x-y) for x in roots for y in predicted],default=math.inf)
                    strong=(sep is None or sep>1e-3) and abs(float(eps))>=1e-8
                    priority='P1' if strong else 'P2'
                    issue(report,priority,'fraction_root_set_disagrees_with_high_precision',
                          f'tangent-epsilon={eps}',
                          {'app':predicted,'missing':missing,'extra':extra,'continuum':actual.continuum},
                          truth,'The MP oracle certifies roots for the encoded F,D, but finite numerical accuracy and tolerance need review.',
                          ['src/cualni_cryst/ptmc_adapter.py'])
            for x in predicted:
                M=mp_matrix_from_float(F)
                B=mp_matrix_from_float(inc)
                mu2=middle_mu_mp(M,B,mp_from_float(x))
                if abs(mu2-1)>mp.mpf('4e-7'):
                    issue(report,'P1','reported_root_not_middle_stretch',f'tangent-epsilon={eps}',
                          {'fraction':x,'certified_middle_mu_error':float(abs(mu2-1))},
                          {'accepted_tol':1e-7,'oracle':truth},
                          'Reject false compatibility; check physical middle stretch and polynomial root screening.',
                          ['src/cualni_cryst/ptmc_adapter.py'])
        except Exception as exc:
            row['app_error']=repr(exc)
            issue(report,'P1','production_fraction_solver_exception',f'tangent-epsilon={eps}',repr(exc),
                  truth,'Investigate local polynomial conditioning; do not change tolerance without theorem-level evidence.',
                  ['src/cualni_cryst/ptmc_adapter.py'])
        print('eps=',eps,'oracle=',truth.get('roots',truth.get('reason','uncertain')),
              'app=',row.get('app_roots','ERR'),
              'root_sep=',truth.get('root_separation'),flush=True)
        report['tangency_rows'].append(row)
    return report


def independent_classification_stage(report):
    """Use generator geometry (NOT app's label code) for near-enhanced 2H state."""
    print('\n=== STAGE 3: NEAR-DEGENERATE TYPE I/II/COMPOUND LABELS ===',flush=True)
    inp=bench_input()
    _,_,state=cubic_mp_orbit(inp,'1e-7')
    try:
        app=app_report(state)
        from cualni_cryst.ptmc_adapter import PTMCAdapter
        from cualni_cryst.project_io import project_from_dict
        variants=independent.metric_variants(state)[0]
        # PTMC output used ONLY to translate app indices into oracle geometry;
        # NEVER use PTMC twin normal, dyad or classification as oracle input.
        project=project_from_dict(independent.payload(state),source='index_translation_only').project
        observed_variants=PTMCAdapter(project,'A_to_M').analyze_all_twinning().variants
        translation={}
        for item in observed_variants:
            err=[np.linalg.norm(item.U-ref) for ref in variants]
            translation[item.index]=int(np.argmin(err))
            if min(err)>1e-7:
                issue(report,'P1','classification_variant_mapping_unstable',str(item.index),min(err),'within 1e-7',
                      'Oracle and production frame mapping should be reviewed before labels.',
                      ['src/cualni_cryst/ptmc_adapter.py'])
        class_counts=Counter()
        checked=0
        for family in app.families:
            for pair in family.pair_records:
                if pair.stretch_i is None or pair.stretch_j is None:continue
                i=translation.get(pair.stretch_i-1);j=translation.get(pair.stretch_j-1)
                if i is None or j is None or i==j:continue
                for c in pair.constructions:
                    dyad=np.outer(c.a_parent_cartesian,c.n_parent_cartesian)
                    branches,_=independent.independent_interfaces(variants[i],variants[j])
                    if not branches:continue
                    errs=[np.linalg.norm(dyad-x['dyad']) for x in branches]
                    idx=int(np.argmin(errs))
                    if errs[idx]>2e-6:
                        issue(report,'P1','classification_geom_oracle_disagrees',pair.pair_id,errs[idx],'<2e-6',
                              'Reconstruct variant mapping and branch dyad before interpreting labels.',
                              ['twin_app/scientific_engine.py'])
                        continue
                    label,nax=independent.geometrical_classification(variants[i],variants[j],branches[idx])
                    checked+=1
                    class_counts[c.classification]+=1
                    if label!='unresolved' and label!=c.classification:
                        issue(report,'P1','near_degenerate_type_label_mismatch',
                              str([pair.pair_id,c.construction_id]),c.classification,
                              {'generator_geometry':label,'twofold_axes':nax},
                              'Potential wrong classification after nearby metric symmetry; confirm generators at higher precision.',
                              ['twin_app/scientific_engine.py','src/cualni_cryst/twinning_ct.py'])
                    if label=='unresolved' and c.classification!='Unresolved':
                        issue(report,'P2','label_not_independently_certified',
                              str([pair.pair_id,c.construction_id]),c.classification,
                              {'generator_geometry':label,'twofold_axes':nax},
                              'Do not interpret an unresolved independent classification as proof of production failure.',
                              ['twin_app/scientific_engine.py'])
        report['classification_summary']=dict(checked=checked,app_labels=dict(class_counts),
                                              status='independent_generator_crosscheck')
        print('Checked labels=',checked,'app distribution=',dict(class_counts),flush=True)
    except Exception as exc:
        issue(report,'P1','classification_stage_exception','near_2h_eps_1e-7',
              {'exception':repr(exc),'trace':traceback.format_exc()[-3500:]},
              'Valid nearly-degenerate cell, no predicted twin-type labels supplied',
              'Separate validity of input and oracle from potential backend crash.',
              ['twin_app/scientific_engine.py'])


def selftest():
    mp.mp.dps=MP_DPS
    # Test the arithmetic and root filtering; not a production benchmark.
    M=mp.matrix([[mp.mpf('0.8'),0,0],[0,1,0],[0,0,mp.mpf('1.2')]])
    assert abs(mp.det(M.T*M-mp.eye(3)))<mp.mpf('1e-60')
    assert len(independent.PARENT_SO3)==24
    q=bench_input() if (ROOT/'data/benchmarks/bhattacharya_cualni_orthorhombic_habit_v1.json').exists() else None
    if q:
        v,_,_=cubic_mp_orbit(q,'1e-8')
        assert len(v)==6, len(v)
        v,_,_=cubic_mp_orbit(q,'0')
        assert len(v)==3,len(v)
    print('SELFTEST PASS: high-precision symmetry orbit and polynomial domain sanity',flush=True)


def persist(report,out):
    directory=Path(out);directory.mkdir(parents=True,exist_ok=True)
    file=directory/'audit.json'
    file.write_text(json.dumps(report,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')
    prior=directory/'repair_ledger.jsonl'
    known=set()
    if prior.exists():
        for ln in prior.read_text().splitlines():
            if ln.strip():
                old=json.loads(ln);known.add((old.get('commit'),old.get('id')))
    with prior.open('a',encoding='utf-8') as h:
        for item in report['findings']:
            if (report['commit'],item['id']) not in known:
                h.write(json.dumps({'commit':report['commit'],**item},sort_keys=True,default=str)+'\n')
    md=['# Numerical-boundary falsification repair ledger','',
        f'Branch: {report["branch"]}',f'Commit: {report["commit"]}',
        'Every finding remains OPEN_UNREVIEWED until independently reproduced.',
        'Synthetic parameter changes are not experimental measurements.',
        'No production implementation or reference fixture was changed.','']
    for k,x in enumerate(report['findings'],1):
        md.extend([f'## {k}. {x["priority"]} — {x["kind"]}',
                   f'ID: `{x["id"]}` — {x["where"]}',
                   f'Observed: `{str(x["observed"])[:1500]}`',
                   f'Oracle/reference: `{str(x["reference"])[:1500]}`',
                   f'Interpretation: {x["interpretation"]}',
                   f'Files: {", ".join(x["likely_files"])}',''])
    if not report['findings']:
        md.append('No discrepancy detected at the tested boundaries, not a proof of completeness.')
    (directory/'repair_ledger.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    # Append-only shared repair history; preserve older upstream audit findings.
    master_dir=ROOT/'audit_findings'
    master_dir.mkdir(exist_ok=True)
    master=master_dir/'SCIENTIFIC_REPAIR_MASTER.jsonl'
    master_prior=set()
    if master.exists():
        for line in master.read_text(encoding='utf-8').splitlines():
            if line.strip():
                old=json.loads(line)
                master_prior.add((old.get('commit'),old.get('id')))
    with master.open('a',encoding='utf-8') as stream:
        for finding in report['findings']:
            key=(report['commit'],finding['id'])
            if key not in master_prior:
                stream.write(json.dumps({'commit':report['commit'],
                         'campaign':'near_degeneracy_2026_10_21',**finding},
                         sort_keys=True,default=str)+'\n')
    index=master_dir/'REPAIR_WORKLIST.md'
    if not index.exists():
        index.write_text('# Scientific repair worklist\n\n'
             'These are TEST FINDINGS until independently reproduced. '
             'Never modify solver tolerances or physics just to silence a test.\n\n',encoding='utf-8')
    marker=f'## Numerical boundary audit — commit {report["commit"][:12]} ({report["phase"]})'
    existing=index.read_text(encoding='utf-8')
    if marker not in existing:
        with index.open('a',encoding='utf-8') as stream:
            stream.write(f'{marker}\n\n'
              f'- Report: `{file}`\n- Repair ledger: `{directory/"repair_ledger.md"}`\n'
              f'- New findings: {len(report["findings"])}. '
              'Status OPEN_UNREVIEWED pending independent reproduction.\n\n')
    print('\nAUDIT RESULT:',report['status'],'finding counts',dict(Counter(f['priority'] for f in report['findings'])),flush=True)
    print('JSON:',file,'\nLEDGER:',directory/'repair_ledger.md',flush=True)
    print('NO application source files edited by audit.',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',choices=['all','variants','tangency','classification'],default='variants')
    ap.add_argument('--selftest',action='store_true')
    ap.add_argument('--out',default='audit_findings/numerical_boundary_2026_10_21')
    args=ap.parse_args()
    mp.mp.dps=MP_DPS
    if args.selftest:selftest();return 0
    if not (ROOT/'data/benchmarks/bhattacharya_cualni_orthorhombic_habit_v1.json').exists():
        raise FileNotFoundError('Run this script from the CuAlNi-CT repository root')
    try:
        sha=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
        branch=subprocess.check_output(['git','branch','--show-current'],stderr=subprocess.DEVNULL,text=True).strip()
    except Exception:sha=branch='unknown'
    report=dict(schema='numerical_boundary_2026_10_21_v1',created=datetime.now(timezone.utc).isoformat(),
                commit=sha,branch=branch,phase=args.phase,MP_decimal_digits=MP_DPS,
                input_basis='Bhattacharya 2H worked input, orthorhombic metrics',
                warning='All enhanced-symmetry/critical parent scale perturbations are SYNTHETIC and not measured specimens',
                oracle_independence='mpmath eigen-orbits; high-precision encoded determinant roots; independent invariant-plane dyads',
                findings=[],variant_rows=[],tangency_rows=[],classification_summary={})
    if args.phase in ['all','variants']:
        variant_stage(report,app_levels={'1e-7','1e-9'})
    if args.phase in ['all','tangency']:
        tangency_stage(report)
    if args.phase in ['all','classification']:
        independent_classification_stage(report)
    report['status']='REVIEW_REQUIRED' if report['findings'] else 'PASS_CONDITIONAL'
    persist(report,args.out)
    return 2 if report['findings'] else 0

if __name__=='__main__':
    sys.exit(main())
