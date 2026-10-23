#!/usr/bin/env python3
"""Independent upstream twin-enumeration falsification, 2026-10-21.

- Uses ONLY experimental/literature parent-product metrics and assumed/verified
  correspondence from the repo's source input blocks. NEVER reads `expected`.
- Constructs all stretch variants without production's variant generator.
- Checks every unordered variant pair, including those never reported by app.
- Derives independent rank-one dyads by an invariant-plane NULL-CONE geometry
  and reconstruction of rank-one vector from metric, unlike production formulas.
- Independent crystallographic twin classification via parent SO(3) twofold
  symmetry generators and geometric constraints of Mallard's law.
- Checks downstream fractions geometrically without accepting preloaded answers.
- Accumulates P0/P1/P2 actionable defect ledger; NEVER mutates production code.

CAUTION: This is a mathematical/model-conditional audit. Parent/product point
  group, phase and correspondence must be correct for physical interpretation.
  Landa's 2H state uses the canonical 2H correspondence as an ASSUMPTION.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

from scipy.optimize import brentq, minimize_scalar
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
EPS_SPECTRAL = 2e-10
NEAR_SPECTRAL = 1e-7
BRANCH_TOL = 1e-6
CLASS_ANGLE_TOL = 1e-5  # projective sine, ~0.00057 degrees
SHEAR_TOL = 2e-6
ROOT_TOL = 3e-6
HABIT_TOL = 2e-6


def ordered_unique(tensors, tol=3e-10):
    out=[]
    for U in tensors:
        if not any(np.linalg.norm(U - X) < tol for X in out):
            out.append(U)
    return out


def metric_from_lengths(p):
    """General triclinic direct-space metric; no production Lattice import."""
    a,b,c=(float(p[x]) for x in ('a','b','c'))
    alpha,beta,gamma=(math.radians(float(p[x])) for x in ('alpha','beta','gamma'))
    return np.array([
        [a*a, a*b*math.cos(gamma), a*c*math.cos(beta)],
        [a*b*math.cos(gamma), b*b, b*c*math.cos(alpha)],
        [a*c*math.cos(beta), b*c*math.cos(alpha), c*c]
    ], dtype=float)


def pure_rotations_cubic():
    """Generate 24 proper parent cubic rotations without production symmetry."""
    result=[]
    for perm in itertools.permutations(range(3)):
        E=np.eye(3)[:,perm]
        for signs in itertools.product((-1.,1.), repeat=3):
            Q=E@np.diag(signs)
            if round(float(np.linalg.det(Q)))==1:
                result.append(Q)
    assert len(result)==24
    return result


PARENT_SO3 = pure_rotations_cubic()


def metric_variants(inp):
    M_a=metric_from_lengths(inp['parent'])
    M_m=metric_from_lengths(inp['product'])
    C=np.array(sp.Matrix(inp['C_M_from_A']),float)
    eig,V=np.linalg.eigh(M_a)
    if min(eig)<=0: raise ValueError('Nonpositive parent metric')
    W=(V*(eig**-0.5))@V.T
    pulled=W@(C.T@M_m@C)@W
    lam,X=np.linalg.eigh(0.5*(pulled+pulled.T))
    if min(lam)<=0: raise ValueError('Nonpositive product pullback metric')
    U=(X*np.sqrt(lam))@X.T
    variants=ordered_unique([Q@U@Q.T for Q in PARENT_SO3])
    return variants,dict(metric_parent=M_a.tolist(),metric_product=M_m.tolist(),C=C.tolist(),lambdas=np.sqrt(lam).tolist())


def rel_spectrum(U,V):
    A=np.linalg.solve(U.T,V.T).T
    B=0.5*(A.T@A+(A.T@A).T)
    mu,X=np.linalg.eigh(B)
    return A,B,mu,X


def independent_interfaces(U,V):
    """Build interface from the two null-cone tangent planes of A^T A-I.

    For a compatible A = V U^-1 and a unit habit normal m:
       R A = I + b⊗m,
       G = A^T A-I = b⊗m + m⊗b + |b|^2 m⊗m.
    Given m, p=(I-mm^T) G m, and t=m·b solves
       (1+t)^2 = 1+m·Gm-|p|^2.
    Positive determinant fixes 1+t>0. This is *not* the formula from
    production's eigenvector b/m expressions.
    """
    A,B,mu,X=rel_spectrum(U,V)
    scale=max(1.,np.linalg.norm(B))
    gap=abs(float(mu[1])-1)
    near= bool(gap > EPS_SPECTRAL and gap < NEAR_SPECTRAL)
    if gap>EPS_SPECTRAL:
        return [],dict(mu=mu.tolist(),gap=gap,condition='near_boundary' if near else 'incompatible')
    if mu[0]>=1-2e-9 or mu[2]<=1+2e-9:
        return [],dict(mu=mu.tolist(),gap=gap,condition='degenerate_or_one_sided')
    G=B-np.eye(3)
    e_lo,e_mid,e_hi=X.T
    out=[]
    for sign in [-1,1]:
        tangent=math.sqrt(max(0.,mu[2]-1))*e_lo + sign*math.sqrt(max(0.,1-mu[0]))*e_hi
        m=np.cross(e_mid,tangent)
        m=m/np.linalg.norm(m)
        p=G@m-float(m@G@m)*m
        discr=1+float(m@G@m)-float(p@p)
        if discr<=0:
            raise ArithmeticError(f'Invalid independent shear reconstruction: {discr}')
        b=p+(-1+math.sqrt(discr))*m
        R=(np.eye(3)+np.outer(b,m))@np.linalg.inv(A)
        n_ref=U.T@m
        dyad=np.outer(b,n_ref)
        n_norm=np.linalg.norm(n_ref)
        normal=n_ref/n_norm
        a=b*n_norm
        eq_err=np.linalg.norm(R@V-U-dyad)/max(1.,np.linalg.norm(U))
        rot_err=np.linalg.norm(R.T@R-np.eye(3))
        det_err=abs(np.linalg.det(R)-1.)
        if max(eq_err,rot_err,det_err)>2e-7:
            raise ArithmeticError(f'Independent oracle residual: {eq_err}, {rot_err}, {det_err}')
        if not any(np.linalg.norm(dyad-q['dyad'])<1e-9 for q in out):
            out.append(dict(dyad=dyad,normal=normal,a=a,rotation=R,
                            shear=float(np.linalg.norm(a)*np.linalg.norm(np.linalg.solve(U,normal))),
                            residual=max(eq_err,rot_err,det_err)))
    return out,dict(mu=mu.tolist(),gap=gap,condition='compatible')


def projective_sin(u,v):
    u=np.asarray(u,float);v=np.asarray(v,float)
    return float(np.linalg.norm(np.cross(u,v))/(np.linalg.norm(u)*np.linalg.norm(v)))


def twofold_axes(U,V):
    axes=[]
    for Q in PARENT_SO3:
        if np.linalg.norm(Q@Q-np.eye(3))>1e-10 or abs(np.trace(Q)+1)>1e-9:
            continue
        if np.linalg.norm(Q@U@Q.T-V)>2e-8:
            continue
        axis= np.linalg.svd(Q-np.eye(3))[2][-1]
        axis/=np.linalg.norm(axis)
        if not any(projective_sin(axis,a)<1e-7 for a in axes):
            axes.append(axis)
    return axes


def geometrical_classification(U, V, solution):
    axes=twofold_axes(U,V)
    if not axes:
        return 'unresolved',0
    type_I=any(projective_sin(solution['normal'],axis)<CLASS_ANGLE_TOL for axis in axes)
    type_II=any(projective_sin(solution['a'],U@axis)<CLASS_ANGLE_TOL for axis in axes)
    if type_I and type_II: return 'Compound',len(axes)
    if type_I: return 'Type I',len(axes)
    if type_II: return 'Type II',len(axes)
    return 'unresolved',len(axes)



def independent_fraction_scan(U, dyad, ngrid=257):
    """Locate potential habit fractions via *middle SVD* bracketing/minimization.

    Does not call production's polynomial determinant/roots or read known
    fractions. A grid can miss very close root pairs; suspicious near-minima
    and near-continua are UNRESOLVED, not successes.
    """
    def g(f):
        vals=np.linalg.svd(U+float(f)*dyad,compute_uv=False)
        return float(np.sort(vals)[1]-1)
    grid=np.linspace(0.,1.,ngrid)
    y=np.array([g(x) for x in grid])
    roots=[];questionable=[]
    if np.sum(np.abs(y)<1e-9)>ngrid//10:
        return [],['near-continuum/flat middle-stretch residual']
    for k in range(ngrid-1):
        if abs(y[k])<3e-9: roots.append(float(grid[k]))
        if y[k]*y[k+1]<0:
            root=float(brentq(g,float(grid[k]),float(grid[k+1]),xtol=5e-14))
            if abs(g(root))<=3e-9: roots.append(root)
            else: questionable.append(dict(fraction=root,residual=abs(g(root)),reason='poor_bracket'))
    if abs(y[-1])<3e-9: roots.append(1.)
    for k in range(1,ngrid-1):
        if abs(y[k])<=abs(y[k-1]) and abs(y[k])<=abs(y[k+1]):
            q=minimize_scalar(lambda z: abs(g(z)),
                bounds=(float(grid[k-1]),float(grid[k+1])),
                method='bounded',options={'xatol':5e-13})
            if q.fun<3e-9: roots.append(float(q.x))
            elif q.fun<5e-7:
                questionable.append(dict(fraction=float(q.x),residual=float(q.fun),reason='near_tangency'))
    clusters=[]
    for root in sorted(roots):
        if not clusters or abs(root-clusters[-1][-1])>2e-5:
            clusters.append([root])
        else:
            clusters[-1].append(root)
    # Prefer a bracket-certified root to a nearby minimum whose fraction
    # was only accurate to ~1e-8: evaluate and keep the best residual.
    out=[min(cluster,key=lambda f: abs(g(f))) for cluster in clusters]
    return out,questionable


def independent_habit_dyads(pre_deformation):
    """Apply the geometric null-cone oracle to F(f) against undeformed I."""
    return independent_interfaces(np.eye(3),pre_deformation)


def record(report, *, priority, kind, case, context, detail, location, proposed_fix):
    item=dict(priority=priority,kind=kind,case=case,context=context,
              detail=detail,likely_files=location,proposed_fix_to_investigate=proposed_fix,
              state='OPEN_UNREVIEWED',not_a_confirmed_code_fix=True)
    item['id']=hashlib.sha256(json.dumps({k:item[k] for k in ('kind','case','context','detail')},sort_keys=True,default=str).encode()).hexdigest()[:12]
    report['findings'].append(item)
    return item


def input_states():
    root=ROOT/'data/benchmarks'
    base=json.loads((root/'bhattacharya_cualni_orthorhombic_habit_v1.json').read_text())['input']
    niti=json.loads((root/'bhattacharya_niti_twin_habit_v1.json').read_text())['input']
    states=json.loads((ROOT/'data/reference/cualni_literature_states.json').read_text())['states']
    landa=next(d for d in states if d['key']=='landa_2007_2h')
    import copy
    second=copy.deepcopy(base)
    for key in ('a','b','c'):
        second['parent'][key]=float(landa['parent']['a_angstrom'])
    for key in ('a','b','c'):
        second['product'][key]=float(landa['martensite'][key+'_angstrom'])
    second.pop('reported_stretch_ratios',None)
    return {
       '2h_reference':(base,'Bhattacharya published 2H worked input, correspondence from reference input'),
       '2h_landa':(second,'Landa 2007 2H published lattice lengths (as stored by repository); canonical 2H correspondence *assumed*, not independently specimen verified'),
       'niti':(niti,'Bhattacharya NiTi B2/B19prime published worked input'),
    }


def payload(inp):
    from app.application import CalculationRequest,LatticeInput,PhaseInput,TransformationInput
    def phase(tag,p):
        return PhaseInput(phase_id=tag,label=tag,point_group=p['point_group'],
                          lattice=LatticeInput(p['a'],p['b'],p['c'],p['alpha'],p['beta'],p['gamma'],'angstrom'))
    req=CalculationRequest(project_id='blind_upstream_falsification',title='Blind independent twin pair enumeration',
                           parent=phase('A',inp['parent']),product=phase('M',inp['product']),
                           transformation=TransformationInput.from_rows('A_to_M','A','M',inp['C_M_from_A']))
    return req.to_project_payload()


def compare_case(case_name, inp, provenance, report):
    from twin_app.scientific_engine import build_twin_family_report
    from cualni_cryst.project_io import project_from_dict
    from cualni_cryst.ptmc_adapter import PTMCAdapter

    print(f'\n=== CASE {case_name}: published lattice metrics ===',flush=True)
    variants, input_audit=metric_variants(inp)
    n=len(variants)
    print(f'Independent variants={n}; all unordered pairs={n*(n-1)//2}',flush=True)
    independent={}
    for i in range(n):
        for j in range(i+1,n):
            branches,meta=independent_interfaces(variants[i],variants[j])
            independent[i,j]=dict(branches=branches,meta=meta)
    print('Independent compatible pairs=',sum(bool(x['branches']) for x in independent.values()),
          'independent branches=',sum(len(x['branches']) for x in independent.values()),flush=True)
    print('Computing full actual user-facing app...',flush=True)
    pay=payload(inp)
    app=build_twin_family_report(pay,'A_to_M')
    # PTMC is used ONLY to translate opaque app variant indices into independent
    # reference-frame stretch matrices. No PTMC twin, normal, root, or result
    # is used as an oracle input.
    state=project_from_dict(dict(pay),source='index_translation_only').project
    ptmc=PTMCAdapter(state,'A_to_M').analyze_all_twinning(dilatational_factor=1.)
    if len(ptmc.variants)!=n:
        record(report,priority='P0',kind='variant_count_mismatch',case=case_name,context='all_variants',
               detail=dict(oracle=n,app=len(ptmc.variants)),location=['src/cualni_cryst/ptmc_adapter.py'],
               proposed_fix='Audit symmetry-orbit generation and variant deduplication; do not relax tolerance')
    # Map production variant index to independent variant by physical matrix,
    # without using any production twin outputs for predictions.
    translation={}
    for v in ptmc.variants:
        errors=np.array([np.linalg.norm(v.U-u) for u in variants])
        best=int(np.argmin(errors))
        if float(errors[best])>2e-7:
            record(report,priority='P0',kind='variant_tensor_mismatch',case=case_name,context=int(v.index),
                detail=dict(best_oracle=best,error=float(errors[best])),location=['src/cualni_cryst/stretch.py','src/cualni_cryst/ptmc_adapter.py'],
                proposed_fix='Trace the product pullback and cubic variant SO(3) action')
        translation[int(v.index)]=best
    if len(set(translation.values()))!=len(translation):
        record(report,priority='P0',kind='variant_collision',case=case_name,context='index_translation',
               detail=translation,location=['src/cualni_cryst/ptmc_adapter.py'],
               proposed_fix='Inspect variant uniqueness and symmetry tolerance')

    seen=defaultdict(list)
    app_total=0
    for family in app.families:
        for pair in family.pair_records:
            if pair.stretch_i is None or pair.stretch_j is None:
                if pair.constructions:
                    record(report,priority='P0',kind='unmapped_app_twin',case=case_name,context=pair.pair_id,
                           detail='App emitted a twin with missing physical stretch index',
                           location=['twin_app/scientific_engine.py'],proposed_fix='Reject or resolve unmapped physical variants before classification')
                continue
            i=translation.get(pair.stretch_i-1)
            j=translation.get(pair.stretch_j-1)
            if i is None or j is None or i==j:
                continue
            key=tuple(sorted((i,j)))
            for c in pair.constructions:
                seen[key].append((i,j,c,pair.pair_id))
                app_total+=1

    if app.audit.stretch_variant_count!=n:
        record(report,priority='P0',kind='app_variant_count_mismatch',case=case_name,context='app_audit',
               detail=dict(oracle=n,app=app.audit.stretch_variant_count),
               location=['twin_app/scientific_engine.py'],proposed_fix='Check physical correspondence-to-stretch mapping')
    tested=0;compatible=0;verified=0;labels=Counter();flags=Counter()
    for (i,j),oracle in independent.items():
        raw=seen.get((i,j),[])
        branches=oracle['branches']
        condition=oracle['meta']['condition']
        if condition=='compatible': compatible+=1
        if condition.startswith('near') or condition.startswith('degenerate'):
            flags[condition]+=1
            record(report,priority='P2',kind='spectral_near_degeneracy',case=case_name,context=[i,j],
                   detail=oracle['meta'],location=['src/cualni_cryst/ptmc_adapter.py'],
                   proposed_fix='Repeat with certified high-precision roots and deterministic degeneracy classification')
            continue
        tested+=1
        if not branches and raw:
            record(report,priority='P0',kind='false_positive_twin_pair',case=case_name,context=[i,j],
                  detail=dict(oracle=oracle['meta'],reported=len(raw)),
                  location=['src/cualni_cryst/ptmc_adapter.py','twin_app/scientific_engine.py'],
                  proposed_fix='Examine relative middle eigenvalue and false rank-one acceptance')
            continue
        if branches and not raw:
            record(report,priority='P0',kind='missing_twin_pair',case=case_name,context=[i,j],
                  detail=dict(spectrum=oracle['meta'],oracle_branches=len(branches)),
                  location=['src/cualni_cryst/ptmc_adapter.py','twin_app/scientific_engine.py'],
                  proposed_fix='Trace independent stretch-orbit pair to groupoid/app mapping')
            continue
        for base_i,other_j,construction,pair_id in raw:
            # Compare the app's dyad against the oracle for its OWN direction.
            directed,_=independent_interfaces(variants[base_i],variants[other_j])
            observed=np.outer(construction.a_parent_cartesian,construction.n_parent_cartesian)
            candidates=[float(np.linalg.norm(observed-o['dyad']) / max(1.,np.linalg.norm(o['dyad']))) for o in directed]
            best=int(np.argmin(candidates)) if candidates else -1
            best_err=candidates[best] if candidates else math.inf
            ctx=dict(pair=[i,j],app_pair=pair_id,construction=construction.construction_id,branch=construction.branch)
            if best_err>BRANCH_TOL:
                record(report,priority='P0',kind='false_or_incorrect_twin_branch',case=case_name,context=ctx,
                       detail=dict(best_dyad_residual=best_err,oracle_spectrum=oracle['meta']),
                       location=['src/cualni_cryst/ptmc_adapter.py','src/cualni_cryst/ball_james_adapter.py'],
                       proposed_fix='Examine both rank-one branches and branch orientation, then independently verify SO(3) rotation')
                continue
            o=directed[best]
            verified+=1
            labels[construction.classification]+=1
            if abs(float(construction.shear_magnitude)-o['shear'])>SHEAR_TOL:
                record(report,priority='P0',kind='twin_shear_magnitude_wrong',case=case_name,context=ctx,
                       detail=dict(oracle_shear=o['shear'],reported_shear=construction.shear_magnitude),
                       location=['src/cualni_cryst/ptmc_adapter.py'],
                       proposed_fix='Correct reference/current-frame normal metric for s=norm(a)*norm(U^-T n)')
            expected,n_axes=geometrical_classification(variants[base_i],variants[other_j],o)
            if expected!='unresolved' and construction.classification!=expected:
                record(report,priority='P1',kind='twin_classification_mismatch',case=case_name,context=ctx,
                       detail=dict(oracle=expected,reported=construction.classification,twofold_generator_axes=n_axes),
                       location=['twin_app/scientific_engine.py','src/cualni_cryst/twinning_ct.py'],
                       proposed_fix='Revisit CT generator provenance / Type-I plane vs Type-II shear direction crosslocks')
            elif expected=='unresolved' and construction.classification in ('Type I','Type II','Compound'):
                record(report,priority='P2',kind='unexplained_classification',case=case_name,context=ctx,
                       detail=dict(oracle_status=expected,reported=construction.classification),
                       location=['twin_app/scientific_engine.py'],proposed_fix='Inspect generator correspondence and crystallographic setting; do not automatically relabel')
            # Search independently for ALL fraction roots on this actual
            # geometric branch (including branches with zero reported habits).
            # Production's polynomial solver and fractions are NEVER oracle inputs.
            roots, uncertain=independent_fraction_scan(variants[base_i],o['dyad'])
            observed_fracs=sorted(set(float(h.other_variant_volume_fraction)
                    for h in construction.habit_solutions))
            distinct_fracs=[]
            for f0 in observed_fracs:
                if not distinct_fracs or abs(f0-distinct_fracs[-1])>1e-5:
                    distinct_fracs.append(f0)
            for v in uncertain:
                record(report,priority='P2',kind='unresolved_fraction_near_tangency',case=case_name,
                    context=ctx,detail=v,location=['src/cualni_cryst/ptmc_adapter.py'],
                    proposed_fix='Use high-precision certified isolation, with explicit tangency and continuum handling')
            for fr in roots:
                if not any(abs(fr-z)<ROOT_TOL for z in distinct_fracs):
                    record(report,priority='P0',kind='missing_habit_fraction',case=case_name,
                        context=ctx,detail=dict(oracle_fraction=fr,reported=distinct_fracs),
                        location=['src/cualni_cryst/ptmc_adapter.py','twin_app/scientific_engine.py'],
                        proposed_fix='Check determinant polynomial conditioning, quadratic discriminant and root filtering')
                # Independently reconstruct BOTH habit shape dyads at this root.
                F=variants[base_i]+float(fr)*o['dyad']
                try:
                    shapes,state=independent_habit_dyads(F)
                except ArithmeticError as exc:
                    shapes=[];state=dict(oracle_error=repr(exc))
                if len(shapes)!=2:
                    record(report,priority='P2',kind='unresolved_habit_shape_branches',case=case_name,
                        context={**ctx,'fraction':fr},detail=state,
                        location=['src/cualni_cryst/ptmc_adapter.py'],
                        proposed_fix='Use higher-precision null-cone geometry near repeated eigenvalues')
                else:
                    actual=[np.outer(h.shape_vector_parent_cartesian,h.habit_normal_parent_cartesian)
                        for h in construction.habit_solutions
                        if abs(float(h.other_variant_volume_fraction)-fr)<ROOT_TOL]
                    for idx, shape in enumerate(shapes):
                        if not any(np.linalg.norm(A-shape['dyad'])<1e-5 for A in actual):
                            record(report,priority='P0',kind='missing_habit_shape_branch',case=case_name,
                                context={**ctx,'fraction':fr,'shape_branch':idx},
                                detail=dict(oracle_shape_dyad=shape['dyad'].tolist(),reported=len(actual)),
                                location=['src/cualni_cryst/ptmc_adapter.py','twin_app/scientific_engine.py'],
                                proposed_fix='Check missing ± habit-plane branch and shape dyadic conversion')
            for fr in distinct_fracs:
                if not any(abs(fr-z)<ROOT_TOL for z in roots):
                    kind='unresolved_reported_fraction' if uncertain else 'false_habit_fraction'
                    record(report,priority='P2' if uncertain else 'P0',kind=kind,case=case_name,
                        context=ctx,detail=dict(reported_fraction=fr,oracle_roots=roots),
                        location=['src/cualni_cryst/ptmc_adapter.py'],
                        proposed_fix='Reevaluate with dense bracket and high-precision polynomial root certification')
            # For every reported habit fraction check physical metric identity,
            # using the INDEPENDENT twin dyad. No known fraction is fed in.
            for h in construction.habit_solutions:
                f=float(h.other_variant_volume_fraction)
                if not 0<=f<=1:
                    record(report,priority='P0',kind='habit_fraction_outside_domain',case=case_name,context=ctx,
                          detail=f,location=['src/cualni_cryst/ptmc_adapter.py'],
                          proposed_fix='Review fraction-domain enforcement')
                    continue
                F=variants[base_i] + f*o['dyad']
                sig=np.linalg.svd(F,compute_uv=False)
                m=np.asarray(h.habit_normal_parent_cartesian,float)
                b=np.asarray(h.shape_vector_parent_cartesian,float)
                H=np.eye(3)+np.outer(b,m)
                e_mid=abs(np.sort(sig)[1]-1)
                gram=float(np.linalg.norm(F.T@F-H.T@H))/max(1.,float(np.linalg.norm(F.T@F)))
                if max(e_mid,gram)>HABIT_TOL:
                    record(report,priority='P0',kind='false_habit_solution',case=case_name,context={**ctx,'fraction':f},
                           detail=dict(middle_stretch_error=e_mid,habit_gram_error=gram),
                           location=['src/cualni_cryst/ptmc_adapter.py','twin_app/scientific_engine.py'],
                           proposed_fix='Inspect laminate fraction, habit rotation and rank-one dyadic reconstruction')
        if branches:
            # Both independently constructed solutions are required, not just
            # one that happens to agree. Compare every APP branch under the
            # same directed pair, with a strict one-to-one matching.
            for bi,oj in sorted({(bi,oj) for bi,oj,_,_ in raw}):
                target,_=independent_interfaces(variants[bi],variants[oj])
                actual=[np.outer(c.a_parent_cartesian,c.n_parent_cartesian) for b1,o1,c,_ in raw if (b1,o1)==(bi,oj)]
                if len(actual)!=len(target):
                    record(report,priority='P0',kind='twin_branch_count_mismatch',case=case_name,context=dict(pair=[i,j],directed=[bi,oj]),
                          detail=dict(oracle=len(target),app=len(actual)),
                          location=['src/cualni_cryst/ptmc_adapter.py','twin_app/scientific_engine.py'],
                          proposed_fix='Audit ± rank-one branch enumeration and UI deduplication')
                for k,o in enumerate(target):
                    if not any(np.linalg.norm(A-o['dyad'])<BRANCH_TOL for A in actual):
                        record(report,priority='P0',kind='missing_twin_branch',case=case_name,context=dict(pair=[i,j],directed=[bi,oj],oracle_branch=k),
                              detail=dict(oracle_normal=o['normal'].tolist(),oracle_shear=o['shear']),
                              location=['src/cualni_cryst/ptmc_adapter.py','twin_app/scientific_engine.py'],
                              proposed_fix='Inspect rank-one branch sign enumeration and correspondence-to-stretch mapping')
    summary=dict(case=case_name,provenance=provenance,model_status='CONDITIONAL_ON_SUPPLIED_CORRESPONDENCE',
                 independent_variant_count=n,all_pairs=n*(n-1)//2,compatible_pairs=compatible,
                 evaluated_nonambiguous_pairs=tested,independent_branch_count=sum(len(x['branches']) for x in independent.values()),
                 app_branch_constructions=app_total,verified_app_branches=verified,
                 app_classifications=dict(labels),ambiguous=dict(flags),
                 findings=len([x for x in report['findings'] if x['case']==case_name]),
                 input=dict(parent=inp['parent'],product=inp['product'],C_M_from_A=inp['C_M_from_A']))
    print('APP branches=',app_total,'verified matching independent geometric interfaces=',verified,flush=True)
    print('App classification labels=',dict(labels),flush=True)
    print('Findings for case=',summary['findings'],'ambiguous cases=',dict(flags),flush=True)
    return summary


def selftest():
    rng=np.random.default_rng(31)
    for i in range(60):
        Q,_=np.linalg.qr(rng.normal(size=(3,3)))
        if np.linalg.det(Q)<0:Q[:,0]*=-1
        stretch=Q@np.diag(rng.uniform(.75,1.25,3))@Q.T
        axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
        twofold=2*np.outer(axis,axis)-np.eye(3)
        other=twofold@stretch@twofold.T
        branches,meta=independent_interfaces(stretch,other)
        assert len(branches)==2,(i,meta)
        assert max(x['residual'] for x in branches)<1e-9
    print('SELFTEST PASS: 60 independent geometric SO(3)/rank-one constructions',flush=True)


def main():
    ap=argparse.ArgumentParser(description='Independent twin completeness and repair ledger')
    ap.add_argument('--case',choices=['2h_reference','2h_landa','niti','all'],default='2h_reference')
    ap.add_argument('--out',default='audit_findings/upstream_2026_10_21')
    ap.add_argument('--selftest',action='store_true')
    args=ap.parse_args()
    if args.selftest:
        selftest();return 0
    states=input_states()
    names=list(states) if args.case=='all' else [args.case]
    try:
        sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
        branch=subprocess.check_output(['git','branch','--show-current'],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:
        sha=branch='unknown'
    report=dict(schema='upstream_independent_falsification_v1',timestamp=datetime.now(timezone.utc).isoformat(),
                branch=branch,head=sha,method='Independent metric stretch orbit; null-cone rank-one dyad; generator geometry',
                independence_caveats=['No published expected fractions, shears, normals read',
                                      'Published lattice values are model inputs, not experimentally indexed variant truth',
                                      'PTMC variant list used strictly to translate app output indices',
                                      'Null-cone method shares mathematical theorem, not production algebraic implementation',
                                      'Near-degenerate cases need higher precision; no finite scan proves global completeness'],
                cases=[],findings=[])
    for name in names:
        inp,provenance=states[name]
        try:
            report['cases'].append(compare_case(name,inp,provenance,report))
        except Exception as exc:
            import traceback
            record(report,priority='P0',kind='audit_or_production_exception',case=name,context='complete_case',
                   detail=dict(type=type(exc).__name__,message=str(exc),trace=traceback.format_exc()),
                   location=['upstream_twin_falsification.py','src/cualni_cryst/ptmc_adapter.py','twin_app/scientific_engine.py'],
                   proposed_fix='Determine whether this is an oracle bug or production domain error; do not patch until reproduced')
            print('ERROR',name,type(exc).__name__,str(exc),flush=True)
    counts=Counter(x['priority'] for x in report['findings'])
    report['counts']=dict(counts)
    report['status']='REVIEW_REQUIRED' if report['findings'] else 'PASS_CONDITIONAL'
    base_path=Path(args.out)
    # Never overwrite a different case's report; preserve each run.
    path=base_path/args.case
    path.mkdir(parents=True,exist_ok=True)
    json_path=path/'audit.json'
    json_path.write_text(json.dumps(report,indent=2,default=str,sort_keys=True)+'\n')
    md=[f'# Independent upstream crystallographic audit — {report["status"]}',
        '',f'Branch: `{branch}`  Commit: `{sha}`',
        '',f'Cases: {", ".join(names)}',
        '',f'Findings: {len(report["findings"])} (P0={counts.get("P0",0)}, P1={counts.get("P1",0)}, P2={counts.get("P2",0)})',
        '', 'Every OPEN finding is an investigative hypothesis, not a certified production-code bug.',
        'Fix only after isolating a minimal numerical reproduction and verifying the independent oracle.',
        'Do not change reference inputs/expected values/tolerances to force success.', '',
        '## Case totals', '']
    for c in report['cases']:
        md.append(f'- **{c["case"]}**: {c["independent_variant_count"]} variants, {c["all_pairs"]} pairs, {c["compatible_pairs"]} compatible pairs, {c["independent_branch_count"]} oracle branches, {c["app_branch_constructions"]} app branches; {c["findings"]} findings')
    md+=['','## Unresolved defects / proposed repair investigation','']
    for x in report['findings']:
        md += [f'### {x["priority"]} {x["kind"]} — {x["id"]}',
               f'- Case: `{x["case"]}`; context: `{json.dumps(x["context"],default=str)}`',
               f'- Numerical evidence: `{json.dumps(x["detail"],default=str)[:900]}`',
               f'- Investigate: {", ".join(x["likely_files"])}',
               f'- Suggested next action: {x["proposed_fix_to_investigate"]}',
               '- Status: **OPEN_UNREVIEWED**', '']
    if not report['findings']:
        md+=['No mismatch found under the tested input state and the oracle tolerances.',
             'This does not certify complete experimental or mathematical truth.']
    (path/'repair_ledger.md').write_text('\n'.join(md)+'\n')
    # Append-only master ledger across this and future case runs; dedup IDs
    # to avoid spamming reruns. Do not erase unresolved earlier findings.
    master=base_path/'findings_history.jsonl'
    prior=set()
    if master.exists():
        for line in master.read_text().splitlines():
            if line.strip():
                try:
                    old=json.loads(line)
                    prior.add((old.get('head'),old.get('id')))
                except ValueError:
                    raise ValueError('Master findings ledger corrupted; preserving it')
    with master.open('a',encoding='utf-8') as stream:
        for item in report['findings']:
            if (sha,item['id']) not in prior:
                stream.write(json.dumps({'head':sha,'timestamp':report['timestamp'],**item},sort_keys=True,default=str)+'\n')
    run_history=base_path/'runs_history.jsonl'
    with run_history.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps({'head':sha,'timestamp':report['timestamp'],
            'cases':names,'status':report['status'],'counts':dict(counts),
            'report_path':str(json_path)},sort_keys=True)+'\n')
    print('\nAUDIT STATUS:',report['status'],flush=True)
    print('FINDINGS:',len(report['findings']),dict(counts),flush=True)
    print('RETAINED JSON:',json_path,flush=True)
    print('REPAIR LEDGER:',path/'repair_ledger.md',flush=True)
    print('MASTER FINDINGS HISTORY:',master,flush=True)
    print('RUN HISTORY:',run_history,flush=True)
    print('Production source files modified: NONE (this script only writes audit records)',flush=True)
    return 1 if any(x['priority']=='P0' for x in report['findings']) else 2 if report['findings'] else 0


if __name__=='__main__':
    sys.exit(main())
