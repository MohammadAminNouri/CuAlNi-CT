#!/usr/bin/env python3
"""Read-only independent diagnostic of PTMC variant-pair and habit inventories.

No imports of PTMC root-solving internals; reads ONLY the public report.
The independent algebra uses high-precision determinants/eigenvalues of the
encoded, binary64 transformation stretches and rank-one *factors*.

Limit: this is a finite-input audit, NOT a proof of completeness across all
possible material data, exact physical measurements, or classification types.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback
import hashlib
import subprocess

import mpmath as mp
import numpy as np

MP_DPS = 85
# Stable numerical regions only. Tighter/degenerate inputs are INCONCLUSIVE.
PAIR_COMPATIBLE = mp.mpf('3e-10')
PAIR_INCOMPATIBLE = mp.mpf('1e-6')
BRANCH_SPECTRAL_MARGIN = mp.mpf('1e-5')
ROOT_SPECTRAL_MARGIN = mp.mpf('5e-7')
ROOT_MIDDLE_ACCEPT = mp.mpf('2e-9')
ROOT_MATCH = 2e-7


def encoded(x):
    """Exact rational representation of one original IEEE-754 binary64 number."""
    n, d = float(x).as_integer_ratio()
    return mp.mpf(n) / mp.mpf(d)


def mat(A):
    array = np.asarray(A, dtype=float)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError('Expected finite 3x3 matrix')
    return mp.matrix([[encoded(array[i,j]) for j in range(3)] for i in range(3)])


def vec(v):
    array=np.asarray(v,dtype=float).reshape(3)
    if not np.isfinite(array).all():
        raise ValueError('Expected finite vector')
    return mp.matrix([encoded(x) for x in array])


def rank_one(a,n):
    return vec(a) * vec(n).T


def mid_eigenspectrum(F, G=None):
    # Generalized Cauchy--Green eigenvalues of F_other F_base^{-1}, or F itself.
    A = F if G is None else F * (G**-1)
    C = A.T * A
    vals, _ = mp.eigsy((C+C.T)/2)
    return tuple(vals[i] for i in range(3))


def coeff_oracle(U, a, n):
    A=rank_one(a,n)
    ident=mp.eye(3)
    def poly(f):
        F=U+f*A
        return mp.det(F.T*F-ident)
    yneg, y0, ypos=poly(mp.mpf(-1)),poly(mp.mpf(0)),poly(mp.mpf(1))
    c0=y0; c1=(ypos-yneg)/2; c2=(ypos+yneg)/2-y0
    errors=[abs(poly(f) - (c0+c1*f+c2*f*f)) for f in (mp.mpf('0.23'),mp.mpf('0.67'),mp.mpf('1.51'))]
    magnitude=max(abs(yneg),abs(y0),abs(ypos), mp.mpf('1e-100'))
    identity_err=max(errors)/magnitude
    return (c0,c1,c2),identity_err,A


def isolated_roots(c):
    """Discriminant analysis; nonzero near-zero coefficients stay indeterminate."""
    a,b,d=c[2],c[1],c[0]
    scale=max(abs(x) for x in c)
    if scale == 0:
        return 'exact_zero_polynomial', [], None
    if scale < mp.mpf('1e-45'):
        return 'near_zero_polynomial', [], None
    if abs(a) < scale*mp.mpf('1e-40'):
        if abs(b) < scale*mp.mpf('1e-40'):
            return 'uncertain_degree', [], None
        return 'linear',[-d/b],None
    disc=b*b-4*a*d
    norm=disc/max(b*b,abs(4*a*d),mp.mpf('1e-100'))
    if disc == 0:
        return 'double_root',[-b/(2*a)],norm
    if abs(norm) < mp.mpf('1e-35'):
        return 'near_double_precision_boundary', [], norm
    if disc < 0:
        return 'no_real_roots',[],norm
    s=mp.sqrt(disc)
    q= -(b+(s if b>=0 else -s))/2
    if q==0:
        roots=[(-b-s)/(2*a),(-b+s)/(2*a)]
    else:
        roots=[q/a,d/q]
    return 'two_real_roots', sorted(roots),norm


def root_diagnostics(U, relation, reported):
    c, error, D=coeff_oracle(U,relation.a,relation.n_reference)
    status, raw, norm=isolated_roots(c)
    entry={'polynomial_status':status,'normalized_discriminant':None if norm is None else str(norm),
           'polynomial_identity_residual':str(error), 'reference_roots':[],
           'production_roots':[float(x) for x in reported], 'classification':'INCONCLUSIVE',
           'comments':[], 'middle_spectral_checks':[]}
    if error > mp.mpf('1e-52'):
        entry['comments'].append('Rank-one determinant degree-two identity not resolved at 85 digits')
        return entry
    if status in {'exact_zero_polynomial','near_zero_polynomial','uncertain_degree','near_double_precision_boundary'}:
        entry['comments'].append('No categorical discrete-root conclusion near an algebraic or numerical degeneracy')
        return entry
    admissible=[]
    for f in raw:
        if not (-mp.mpf('1e-30') <= f <= 1+mp.mpf('1e-30')):
            continue
        F=U+f*D
        ev=mid_eigenspectrum(F)
        middle=abs(mp.sqrt(ev[1])-1)
        gap=min(abs(mp.sqrt(ev[0])-1),abs(mp.sqrt(ev[2])-1))
        entry['middle_spectral_checks'].append({'f':str(f),'middle_error':str(middle),'outer_gap':str(gap)})
        if middle > ROOT_MIDDLE_ACCEPT or gap < ROOT_SPECTRAL_MARGIN:
            entry['comments'].append('Middle-singular-value status unresolved for polynomial root')
            return entry
        admissible.append(f)
    entry['reference_roots']=[float(f) for f in admissible]
    got=[float(x) for x in reported]
    # Compare multisets one-to-one, not just counts or nearest-neighbour matches.
    remaining=list(got); missing=[]
    for f in admissible:
        if not remaining:
            missing.append(float(f)); continue
        j=min(range(len(remaining)),key=lambda k:abs(remaining[k]-float(f)))
        if abs(remaining[j]-float(f)) <= ROOT_MATCH:
            remaining.pop(j)
        else:
            missing.append(float(f))
    # Avoid claiming globally exact absence/presence for cases with very small
    # but nonzero normalized discriminants or nearly merging roots.
    if norm is not None and abs(norm)<mp.mpf('1e-10'):
        entry['classification']='INCONCLUSIVE'
        entry['comments'].append('Discriminant approaches a numerical topology transition')
    elif missing or remaining:
        entry['classification']='MISMATCH'
        entry['missing']=missing;entry['unexpected']=remaining
    else:
        entry['classification']='MATCH'
    return entry


def pair_oracle(variants, relations):
    relation_map=defaultdict(list)
    for r in relations: relation_map[(int(r.base_variant_index),int(r.other_variant_index))].append(r)
    out=[]
    for i,v in enumerate(variants):
        for j,w in enumerate(variants):
            if i==j: continue
            ev=mid_eigenspectrum(mat(w.U),mat(v.U))
            error=abs(ev[1]-1)
            exterior=min(abs(ev[0]-1),abs(ev[2]-1))
            present=relation_map.get((i,j),[])
            bs=sorted(int(r.branch) for r in present)
            state='INCONCLUSIVE'
            if error<PAIR_COMPATIBLE and ev[0]<1-BRANCH_SPECTRAL_MARGIN and ev[2]>1+BRANCH_SPECTRAL_MARGIN:
                state='MATCH' if bs==[-1,1] else 'MISMATCH'
            elif error>PAIR_INCOMPATIBLE:
                state='MATCH' if not present else 'MISMATCH'
            out.append({'pair':[i,j],'middle_mu_error':str(error),
                        'outer_eigen_distance':str(exterior),'branches_found':bs,
                        'classification':state})
    return out


def case_input(name):
    from cualni_cryst.ptmc_adapter import PTMCCrystalInput
    from cualni_cryst.lattice import Lattice
    import sympy as sp
    if name=='2h':
        bench=json.loads(Path('data/benchmarks/bhattacharya_cualni_orthorhombic_habit_v1.json').read_text())
        inp=bench['input']
        def lattice(d):return Lattice(d['a'],d['b'],d['c'],d['alpha'],d['beta'],d['gamma'],length_unit='angstrom')
        model=PTMCCrystalInput(lattice(inp['parent']),lattice(inp['product']),
                               np.array(sp.Matrix(inp['C_M_from_A']),dtype=float),
                               inp['parent']['point_group'],inp['product']['point_group'],
                               transformation_id='audit_bhattacharya_2h')
    elif name=='6m':
        from cualni_cryst.cualni_models import james_hane_6m_example_lattices, do3_to_6m_branch
        parent,product=james_hane_6m_example_lattices()
        model=PTMCCrystalInput(parent,product,do3_to_6m_branch().correspondence,
                               'm-3m','2/m',transformation_id='audit_james_hane_6m')
    else:raise ValueError(name)
    return model


def audit_case(name):
    model=case_input(name)
    report=model.analyze_all_twinning()
    rows=pair_oracle(report.variants,report.twin_relations)
    roots=[]; mismatches=[]; inconclusive=[]
    if len(report.twin_relations)!=len(report.parameter_roots):
        mismatches.append('Report lost one-to-one relation-to-root association')
    for k,(relation, params) in enumerate(zip(report.twin_relations,report.parameter_roots)):
        U=mat(report.variants[relation.base_variant_index].U)
        check=root_diagnostics(U,relation,params.discrete_roots)
        check['pair']=[int(relation.base_variant_index),int(relation.other_variant_index)]
        check['branch']=int(relation.branch)
        check['index']=k
        check['continuum']=bool(params.continuum)
        if params.continuum:
            check['classification']='INCONCLUSIVE'
            check['comments'].append('Continuum proof not covered by discrete-root oracle')
        roots.append(check)
        if check['classification']=='MISMATCH': mismatches.append({'root_index':k,'pair':check['pair'],'branch':check['branch']})
        if check['classification']=='INCONCLUSIVE': inconclusive.append({'root_index':k,'pair':check['pair']})
    for p in rows:
        if p['classification']=='MISMATCH':mismatches.append({'pair_inventory':p['pair']})
        if p['classification']=='INCONCLUSIVE':inconclusive.append({'pair_inventory':p['pair']})

    # Internal output cardinality check, not an independent classification proof.
    relation_keys=[(r.base_variant_index,r.other_variant_index,r.branch) for r in report.twin_relations]
    if len(relation_keys)!=len(set(relation_keys)):
        mismatches.append('Duplicate ordered twin branches')
    for k, relation in enumerate(report.twin_relations):
        sols=[s for s in report.solutions if (s.base_variant_index,s.other_variant_index,s.twin_branch)==relation_keys[k]]
        vals=list(report.parameter_roots[k].discrete_roots)
        for f in vals:
            corresponding=[s for s in sols if abs(s.parameter_value-f)<ROOT_MATCH]
            # We check only nondegenerate fractions already independently matched.
            if roots[k]['classification']=='MATCH' and len(corresponding)!=2:
                mismatches.append({'missing_habit_branches_for_relation':relation_keys[k],
                                   'fraction':float(f),'got':[s.habit_branch for s in corresponding]})
            if len(corresponding)==2 and sorted(s.habit_branch for s in corresponding)!=[-1,1]:
                mismatches.append({'unexpected_habit_branch_labels':relation_keys[k],
                                   'fraction':float(f)})
    result={'name':name,'variant_count':len(report.variants),'ordered_pair_count':len(rows),
            'rank_one_branch_count':len(report.twin_relations),'habit_solution_count':len(report.solutions),
            'root_parameter_records':len(report.parameter_roots),
            'pair_inventory':rows,'root_inventory':roots,'mismatches':mismatches,'inconclusive':inconclusive}
    return result


def selftest():
    with mp.workdps(MP_DPS):
        U=mat(np.diag([0.8,0.9,1.2]))
        class R:pass
        r=R();r.a=np.array([1.,0.,0.]);r.n_reference=np.array([0.,1.,0.])
        c,err,_=coeff_oracle(U,r.a,r.n_reference)
        typ,roots,_=isolated_roots(c)
        assert typ=='two_real_roots' and len(roots)==2 and err<mp.mpf('1e-55'),(typ,roots,err)
        # both signs retained, no false duplicated zeros
        good=root_diagnostics(U,r,[0.26153393661244])
        assert good['classification']=='MATCH',good
        # No invented roots for strictly positive polynomial
        s,rr,_=isolated_roots((mp.mpf('1'),mp.mpf('0'),mp.mpf('1')))
        assert s=='no_real_roots' and not rr
        # Exact tangent is one root; linear case correctly preserves boundary
        s,rr,_=isolated_roots((mp.mpf('0.25'),mp.mpf('-1'),mp.mpf('1')))
        assert s=='double_root' and rr==[mp.mpf('.5')],(s,rr)
        s,rr,_=isolated_roots((mp.mpf('0'),mp.mpf('1'),mp.mpf('0')))
        assert s=='linear' and rr==[0]
    print('SELFTEST: PASS (high-precision determinant, roots, tangency, sign, linear)')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['2h','6m','both'],default='2h')
    parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--output',default='audit_findings/independent_inventory_2026_10_22.json')
    args=parser.parse_args()
    if args.selftest:
        selftest();return 0
    try:
        with mp.workdps(MP_DPS):
            cases=[audit_case(i) for i in (['2h','6m'] if args.case=='both' else [args.case])]
    except Exception:
        traceback.print_exc()
        return 3
    mismatch=sum(len(c['mismatches']) for c in cases)
    uncertain=sum(len(c['inconclusive']) for c in cases)
    status='MISMATCH' if mismatch else ('INCONCLUSIVE' if uncertain else 'PASS_CONDITIONAL')
    def git_read(*args):
        try:
            x=subprocess.run(['git',*args],capture_output=True,text=True,timeout=5,check=False)
            return x.stdout.strip() if x.returncode==0 else None
        except (OSError,subprocess.TimeoutExpired):
            return None
    provenance={'branch':git_read('branch','--show-current'),
                'head_commit':git_read('rev-parse','HEAD'),
                'audit_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    payload={'schema':'independent_inventory_2026_10_22_v1','created_utc':datetime.now(timezone.utc).isoformat(),
             'provenance':provenance,
             'method':'85-digit independent determinant polynomial + eigenvalue checks of encoded binary64 state',
             'status':status,'cases':cases,'mismatch_count':mismatch,'inconclusive_count':uncertain,
             'limitations':['Finite benchmark inputs only, not global scientific proof',
               'Direct CT Type I/II/Compound classification independently not covered',
               'Near-multiple roots, symmetry degeneracy and continua remain inconclusive',
               'No experimental lattice-parameter uncertainty quantified',
               'Original input U,a,n are binary64 approximations; their reconstructed outer product is exactly rank one in mpmath']}
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    for case in cases:
        rp=Counter(x['classification'] for x in case['root_inventory'])
        pp=Counter(x['classification'] for x in case['pair_inventory'])
        print(f"{case['name']}: variants={case['variant_count']} ordered_pairs={case['ordered_pair_count']} "
              f"twin_branches={case['rank_one_branch_count']} habit_solutions={case['habit_solution_count']}")
        print('  independent pair checks:',dict(pp),'fraction checks:',dict(rp))
        print('  mismatches:',len(case['mismatches']),'inconclusive:',len(case['inconclusive']))
    print('AUDIT STATUS:',status,'mismatches=',mismatch,'inconclusive=',uncertain)
    print('EVIDENCE:',path.resolve())
    return 2 if mismatch else (4 if uncertain else 0)


if __name__=='__main__':
    sys.exit(main())
