#!/usr/bin/env python3
"""Read-only independent tensor geometry audit of CuAlNi PTMC 2H and 6M.

Independent high-precision implementation of the nonlinear-elasticity
rank-one formula, without calling PTMC/Ball-James production solvers for oracle
construction. Production is used to obtain crystallographic input/stretch
states and to enumerate its proposed results. The comparison is limited to
encoded floating-point inputs and mathematical rank-one/PTMC geometry.

No audit can prove crystallographic Type I/II/Compound just from stretch and
habit dyads: that requires separate discrete CT cross-lock verification.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import traceback
from datetime import datetime, timezone
import mpmath as mp
import numpy as np

DPS=85
MU_BOUND=mp.mpf('5e-10')
MU_UNCERTAIN=mp.mpf('5e-7')
SPECTRAL_GAP=mp.mpf('2e-5')
DYAD_TOL=2e-7
GEOMETRY_TOL=2e-7
ROOT_EQ_TOL=3e-7


def enc(x):
    n,d=float(x).as_integer_ratio()
    return mp.mpf(n)/mp.mpf(d)


def mx(a):
    x=np.asarray(a,float)
    assert x.shape==(3,3) and np.isfinite(x).all(), 'matrix not finite 3x3'
    return mp.matrix([[enc(x[i,j]) for j in range(3)] for i in range(3)])


def nm(A):
    return mp.sqrt(sum(v*v for v in A))


def fl(A):
    return np.asarray([[float(A[i,j]) for j in range(A.cols)] for i in range(A.rows)],float)


def norm_res(A,B):
    return float(nm(A-B)/max(nm(A),nm(B),mp.mpf('1e-40')))


def branches(base,other):
    """Independently construct the (up to two) rank-one dyads D=R other-base.

    Calculations with eigenpairs of (other base^{-1})^T(other base^{-1}).
    Every candidate is checked against its matrix identity, SO(3), and rank.
    No production rank-one factors enter the construction.
    """
    A=other*base**-1
    C=(A.T*A + (A.T*A).T)/2
    eig,V=mp.eigsy(C)
    lo,mid,hi=(eig[i] for i in range(3))
    mid_err=abs(mid-1)
    if mid_err > MU_UNCERTAIN: return 'INCOMPATIBLE',[],str(mid_err)
    if mid_err > MU_BOUND or min(1-lo,hi-1)<SPECTRAL_GAP:
        return 'INCONCLUSIVE',[],str(mid_err)
    if lo<=0 or lo>=1 or hi<=1 or hi-lo<mp.mpf('1e-9'):
        return 'INCONCLUSIVE',[],str(mid_err)
    u=V[:,0]; w=V[:,2]
    a=mp.sqrt(hi*(1-lo)); b=mp.sqrt(lo*(hi-1)); den=mp.sqrt(hi-lo)
    result=[]
    for s in (-1,1):
        v=(a*u+s*b*w)/den
        n=(mp.sqrt(hi)-mp.sqrt(lo))/den*(-mp.sqrt(1-lo)*u+s*mp.sqrt(hi-1)*w)
        D=v*n.T*base
        R=(mp.eye(3)+v*n.T)*A**-1
        residual_rank=norm_res(R*other-base,D)
        residual_so=float(nm(R.T*R-mp.eye(3)))
        determinant=float(abs(mp.det(R)-1))
        if max(residual_rank,residual_so,determinant)>5e-7:
            return 'INCONCLUSIVE',[],str(mid_err)
        result.append({'sign':s,'D':D,'R':R,'n_raw':base.T*n, 'a_raw':v,
                       'identity_error':residual_rank,'rotation_error':residual_so})
    # Distinctness of the two physical dyads, not merely different labels.
    if norm_res(result[0]['D'],result[1]['D']) < 1e-8:
        return 'INCONCLUSIVE',[],str(mid_err)
    return 'TWO_BRANCHES',result,str(mid_err)


def coefficient_roots(base,D):
    I=mp.eye(3)
    def v(f):
        F=base+f*D
        return mp.det(F.T*F-I)
    a,b,c=v(-1),v(0),v(1)
    c0=b;c1=(c-a)/2;c2=(c+a)/2-b
    scale=max(abs(c0),abs(c1),abs(c2),mp.mpf('1e-100'))
    err=max(abs(v(t)-c0-c1*t-c2*t*t) for t in (mp.mpf('0.17'),mp.mpf('0.63'),mp.mpf('1.7')))/scale
    if err>mp.mpf('1e-48'):return 'INCONCLUSIVE',[],str(err)
    if scale<mp.mpf('1e-45') or abs(c2)<scale*mp.mpf('1e-38'):
        return 'INCONCLUSIVE',[],str(err)
    disc=c1*c1-4*c0*c2
    dnorm=disc/max(c1*c1,abs(4*c0*c2),mp.mpf('1e-100'))
    if abs(dnorm)<mp.mpf('1e-30'):return 'INCONCLUSIVE',[],str(dnorm)
    if disc<0:return 'NO_ROOTS',[],str(dnorm)
    sd=mp.sqrt(disc)
    rr=sorted(((-c1-sd)/(2*c2),(-c1+sd)/(2*c2)))
    rr=[r for r in rr if 0<=r<=1]
    if not rr:return 'NO_ROOTS',[],str(dnorm)
    if len(rr)==2 and abs(rr[1]-rr[0])<mp.mpf('1e-8'):
        return 'INCONCLUSIVE',[],str(dnorm)
    return 'ROOTS',rr,str(dnorm)


def match_set(reference,production,tol):
    """Explicit optimal one-to-one matching, no nearest-neighbour reuse."""
    if len(reference)!=len(production):
        return {'classification':'MISMATCH','reference_count':len(reference),'production_count':len(production)}
    if not reference:return {'classification':'MATCH','max_relative_error':0.0}
    if len(reference)>2:raise ValueError('expected at most two branches')
    choices=[]
    for perm in itertools.permutations(range(len(production))):
        errors=[norm_res(reference[i],production[j]) for i,j in enumerate(perm)]
        choices.append((max(errors),perm,errors))
    worst,perm,errors=min(choices)
    return {'classification':'MATCH' if worst<=tol else 'MISMATCH',
            'max_relative_error':worst,'matching':list(perm),'individual_errors':errors}


def relative_vector(x,y):
    a=np.asarray(x,float);b=np.asarray(y,float)
    d=max(np.linalg.norm(a),np.linalg.norm(b),1e-20)
    return float(np.linalg.norm(a-b)/d)


def model_for(name):
    # Repo's input construction only; not its compatibility algorithms.
    from cualni_cryst.ptmc_adapter import PTMCCrystalInput
    from cualni_cryst.lattice import Lattice
    import sympy as sp
    if name=='2h':
        inp=json.loads(Path('data/benchmarks/bhattacharya_cualni_orthorhombic_habit_v1.json').read_text())['input']
        def lat(x):return Lattice(x['a'],x['b'],x['c'],x['alpha'],x['beta'],x['gamma'],length_unit='angstrom')
        return PTMCCrystalInput(lat(inp['parent']),lat(inp['product']),
            np.array(sp.Matrix(inp['C_M_from_A']),float),inp['parent']['point_group'],inp['product']['point_group'])
    if name=='6m':
        from cualni_cryst.cualni_models import james_hane_6m_example_lattices,do3_to_6m_branch
        parent,product=james_hane_6m_example_lattices()
        return PTMCCrystalInput(parent,product,do3_to_6m_branch().correspondence,'m-3m','2/m')
    raise ValueError(name)


def symmetry_screen(model,report,relation):
    """Independent *necessary geometric* witness, NOT complete CT classification."""
    from cualni_cryst.point_groups import point_group_operations
    U=np.asarray(report.variants[relation.base_variant_index].U,float)
    V=np.asarray(report.variants[relation.other_variant_index].U,float)
    M=np.asarray(model.parent_lattice.metric(),float)
    eig,axes=np.linalg.eigh(M)
    B=(axes*np.sqrt(eig))@axes.T
    Bi=np.linalg.inv(B)
    dyad=np.outer(relation.a,relation.n_reference)
    a=np.asarray(relation.a,float);n=np.asarray(relation.n_reference,float)
    matches=[]
    for idx,g in enumerate(point_group_operations(model.parent_point_group)):
        Q=B@np.asarray(g,float)@Bi
        if (abs(np.linalg.det(Q)-1)>1e-8 or abs(np.trace(Q)+1)>1e-8 or
            np.linalg.norm(Q@Q-np.eye(3))>1e-8):continue
        transform_err=relative_vector(Q@U@Q.T,V)
        if transform_err>1e-8:continue
        val,vec=np.linalg.eigh((Q+Q.T)/2)
        axis=vec[:,int(np.argmin(abs(val-1)))]
        ni=1-abs(float(np.dot(n,axis)))/max(np.linalg.norm(n),1e-30)
        ii=1-abs(float(np.dot(a,U@axis)))/max(np.linalg.norm(a)*np.linalg.norm(U@axis),1e-30)
        matches.append({'operation':idx,'mapping_error':transform_err,
                        'type_i_normal_alignment_error':ni,
                        'type_ii_shear_alignment_error':ii})
    i=any(x['type_i_normal_alignment_error']<2e-7 for x in matches)
    ii=any(x['type_ii_shear_alignment_error']<2e-7 for x in matches)
    return {'proper_twofold_generators':len(matches),
            'geometric_type_i_witness':i,'geometric_type_ii_witness':ii,
            'candidate':'Compound' if i and ii else 'Type I' if i else 'Type II' if ii else 'Unresolved',
            'witnesses':matches}


def audit_case(name):
    model=model_for(name)
    production=model.analyze_all_twinning()
    variants={v.index:mx(v.U) for v in production.variants}
    prod_pairs={}
    for r in production.twin_relations:
        prod_pairs.setdefault((r.base_variant_index,r.other_variant_index),[]).append(r)
    checks=[];fatal=[];inconclusive=[];sol_count=0
    # Application counts and classification provenance are a separate layer.
    for i,j in itertools.permutations(range(len(variants)),2):
        state,oracle,mu_error=branches(variants[i],variants[j])
        reported=prod_pairs.get((i,j),[])
        entry={'pair':[i,j],'oracle_status':state,'middle_mu_error':mu_error,
               'production_branch_count':len(reported),'rank_one':None,
               'habits':[],'symmetry_screen':[]}
        if state=='INCONCLUSIVE':
            inconclusive.append({'pair':[i,j],'reason':'rank-one spectrum near numerical boundary'})
            checks.append(entry);continue
        if state=='INCOMPATIBLE':
            if reported:fatal.append({'pair':[i,j],'problem':'production has relations for incompatible pair'})
            checks.append(entry);continue
        dyads=[mx(np.outer(r.a,r.n_reference)) for r in reported]
        outcome=match_set([x['D'] for x in oracle],dyads,DYAD_TOL)
        entry['rank_one']=outcome
        if outcome['classification']=='MISMATCH':
            fatal.append({'pair':[i,j],'problem':'twin dyad set differs from independent oracle','result':outcome})
            checks.append(entry);continue
        # Preserve correspondence by tensor, not by +/- branch label.
        for oi,pidx in enumerate(outcome['matching']):
            rankone=oracle[oi]
            r=reported[pidx]
            D=rankone['D']; U=variants[i]
            rootstate,rf,discriminant=coefficient_roots(U,D)
            # Dataclass contains ndarray; compare by object identity only.
            param_idx=next(k for k,x in enumerate(production.twin_relations) if x is r)
            prm=production.parameter_roots[param_idx]
            rootmatch={'classification':'INCONCLUSIVE','oracle_status':rootstate,
                       'production_roots':list(prm.discrete_roots),'oracle_roots':[float(x) for x in rf],
                       'normalized_discriminant':discriminant}
            if rootstate in ('ROOTS','NO_ROOTS'):
                found=sorted(float(x) for x in prm.discrete_roots)
                expect=sorted(float(x) for x in rf)
                ok=len(found)==len(expect) and all(abs(a-b)<ROOT_EQ_TOL for a,b in zip(found,expect))
                rootmatch['classification']='MATCH' if ok else 'MISMATCH'
                if not ok:fatal.append({'pair':[i,j],'branch':r.branch,'problem':'habit fractions mismatch independent dyad'})
            else:inconclusive.append({'pair':[i,j],'branch':r.branch,'reason':'root discriminant/degree near boundary'})
            actual_sols=[s for s in production.solutions if
                s.base_variant_index==i and s.other_variant_index==j and s.twin_branch==r.branch]
            fraction_rows=[]
            if rootmatch['classification']=='MATCH':
                for fraction in rf:
                    F=U+fraction*D
                    habstatus,habits,habmu=branches(mp.eye(3),F)
                    here=[s for s in actual_sols if abs(s.parameter_value-float(fraction))<ROOT_EQ_TOL]
                    if habstatus=='INCONCLUSIVE':
                        inconclusive.append({'pair':[i,j],'branch':r.branch,'fraction':float(fraction),
                            'reason':'habit-plane degeneracy'})
                        fraction_rows.append({'fraction':float(fraction),'status':'INCONCLUSIVE','mu_error':habmu})
                        continue
                    if habstatus=='INCOMPATIBLE':
                        fatal.append({'pair':[i,j],'branch':r.branch,'fraction':float(fraction),
                            'problem':'root not habit-compatible in independent oracle'})
                        fraction_rows.append({'fraction':float(fraction),'status':'MISMATCH'})
                        continue
                    prod_h=[mx(np.outer(s.rank_one_vector,s.habit_normal_parent_cartesian)) for s in here]
                    compare=match_set([x['D'] for x in habits],prod_h,GEOMETRY_TOL)
                    frame_max=0.0
                    for s in here:
                        Rh=mx(s.habit_rotation)
                        macro=mx(s.macroscopic_deformation)
                        dh=mx(np.outer(s.rank_one_vector,s.habit_normal_parent_cartesian))
                        errors=[norm_res(Rh*F,macro),norm_res(macro-mp.eye(3),dh),
                                float(nm(Rh.T*Rh-mp.eye(3))),float(abs(mp.det(Rh)-1))]
                        frame_max=max(frame_max,*errors)
                    compare['max_internal_frame_residual']=frame_max
                    compare['classification']=('MATCH' if compare['classification']=='MATCH' and frame_max<=GEOMETRY_TOL else 'MISMATCH')
                    if compare['classification']=='MISMATCH':
                        fatal.append({'pair':[i,j],'branch':r.branch,'fraction':float(fraction),
                            'problem':'habit-plane dyad or physical frame mismatch','result':compare})
                    sol_count+=len(here)
                    fraction_rows.append({'fraction':float(fraction),'status':compare['classification'],
                                          'matching':compare})
            # A fraction missing from the reference must never become an
            # unexamined extra habit result, even when no reference roots exist.
            if rootmatch['classification']=='MATCH':
                extras=[float(sol.parameter_value) for sol in actual_sols
                        if all(abs(sol.parameter_value-float(f))>=ROOT_EQ_TOL for f in rf)]
                if extras:
                    fatal.append({'pair':[i,j],'branch':int(r.branch),
                        'problem':'extra habit solutions without an independently valid fraction',
                        'fractions':extras})
            # Independently verify the PRODUCTION twinning SO(3) and dyad,
            # not only its geometry's agreement with the independent dyad.
            Rp=mx(r.rotation_hat)
            estimated_D=mx(np.outer(r.a,r.n_reference))
            production_geometry_error=max(norm_res(Rp*variants[j]-variants[i],estimated_D),
                float(nm(Rp.T*Rp-mp.eye(3))),float(abs(mp.det(Rp)-1)))
            if production_geometry_error>GEOMETRY_TOL:
                fatal.append({'pair':[i,j],'branch':int(r.branch),
                    'problem':'production twin rotation/dyad violates SO3 or rank-one equation',
                    'error':production_geometry_error})
            entry['habits'].append({'production_branch':int(r.branch),
                'production_rotation_max_residual':production_geometry_error,
                'oracle_branch':rankone['sign'],'roots':rootmatch,'fractions':fraction_rows})
            entry['symmetry_screen'].append({'production_branch':int(r.branch),
                **symmetry_screen(model,production,r)})
        checks.append(entry)
    return {'name':name,'variants':len(variants),'ordered_pairs':len(variants)*(len(variants)-1),
            'production_twin_branches':len(production.twin_relations),
            'production_habit_solutions':len(production.solutions),
            'independently_matched_habit_solutions':sol_count,
            'mismatches':fatal,'inconclusive':inconclusive,
            'pair_checks':checks,'status':'MISMATCH' if fatal else 'INCONCLUSIVE' if inconclusive else 'PASS_CONDITIONAL'}


def selftest():
    with mp.workdps(DPS):
        i=mp.eye(3)
        # Construct a known compatible pair by a proper 180-degree rotation.
        U=mx(np.diag([.8,1.1,1.2]))
        Q=mx(np.array([[0,1,0],[1,0,0],[0,0,-1]],float))
        S=Q*U*Q.T
        status,cases,error=branches(U,S)
        assert status=='TWO_BRANCHES' and len(cases)==2,(status,error)
        assert all(x['rotation_error']<1e-10 for x in cases)
        # The matched set is invariant to output ordering and dyad sign-gauge.
        assert match_set([x['D'] for x in cases],[cases[1]['D'],cases[0]['D']],1e-8)['classification']=='MATCH'
        assert match_set([x['D'] for x in cases],[cases[0]['D']],1e-8)['classification']=='MISMATCH'
        # Genuine habit branch: known F has singular values below / at / above 1.
        F=mx(np.diag([.8,1.,1.2]))
        h, hs, error=branches(i,F)
        assert h=='TWO_BRANCHES' and len(hs)==2,(h,error)
        # Incompatible full-stretch state.
        bad=mx(np.diag([.8,.9,1.2]))
        h,hs,error=branches(i,bad)
        assert h=='INCOMPATIBLE'
        print('SELFTEST PASS: independent tensor construction, SO(3), both branches, identity, mismatch detection')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--case',choices=['2h','6m','both'],default='both')
    a=parser.parse_args()
    with mp.workdps(DPS):
        if a.selftest:
            selftest();return
        cases=[a.case] if a.case!='both' else ['2h','6m']
        results=[]
        for name in cases:
            try:
                result=audit_case(name)
            except Exception as e:
                result={'name':name,'status':'EXECUTION_ERROR','error':repr(e),'traceback':traceback.format_exc()}
            results.append(result)
            print(name, 'status:',result['status'],'twins:',result.get('production_twin_branches'),
                  'habit solutions:',result.get('production_habit_solutions'),
                  'mismatches:',len(result.get('mismatches',[])),
                  'inconclusive:',len(result.get('inconclusive',[])),flush=True)
        path=Path('audit_findings/geometry_independent_2026_10_22.json')
        path.parent.mkdir(exist_ok=True,parents=True)
        obj={'timestamp_utc':datetime.now(timezone.utc).isoformat(),
             'scope':'independent rank-one and habit tensor geometry; type witness screen NOT discrete CT cross-lock',
             'precision_digits':DPS,'cases':results,
             'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
        print('EVIDENCE:',path)
        if any(x['status'] in ('MISMATCH','EXECUTION_ERROR') for x in results):sys.exit(2)
        if any(x['status']=='INCONCLUSIVE' for x in results):sys.exit(3)

if __name__=='__main__':main()
