#!/usr/bin/env python3
"""Blind, bounded audit: all reported two-variant habit fractions and normals.

Scientific sources:
* Novak et al., MSEA 273-275 (1999), 280-285: cubic gamma1/beta1 and
  orthorhombic gamma1' 2H Cu-Al-Ni lattice lengths in Angstrom.
* Bhattacharya, Microstructure of Martensite (2003), martensitic laminates.
* Ball and James (1987): rank-one compatible interfaces.
* Petersmann, Montanuniversitaet Leoben thesis, chapter 2: C_h=F_f^T F_f.

The standard 2H C_M_from_A comes from the separate, explicit repository
literature-reference branch: it is NOT specimen-verified for Novak et al.

No 'expected' value from any benchmark is read.  No production polynomial-root
routine, reported fraction, or habit normal is used in the oracle.

LIMITATION: the twin pair/variant tensors and twin rank-one increments are taken
from production PTMC. Thus this tests completeness CONDITIONALLY on those inputs;
it does not independently establish a specimen's correspondence or prove its
martensite/martensite twin enumeration complete.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from app.application import CalculationRequest, LatticeInput, PhaseInput, TransformationInput
from cualni_cryst.project_io import project_from_dict
from cualni_cryst.ptmc_adapter import PTMCAdapter
from twin_app.scientific_engine import build_twin_family_report

N_GRID = 513
ROOT_RESIDUAL = 2e-9
NEAR_THRESHOLD = 2e-7
FRACTION_MATCH = 5e-7
TENSOR_TOL = 2e-6
NORMAL_TOL_RAD = 4e-5
SOURCE = (
    'Novak et al., On the anisotropy of martensitic transformations in Cu-based '
    'alloys, Mater. Sci. Eng. A 273-275 (1999) 280-285, page 283. '
    'Cubic a0=5.836 A; 2H orthorhombic a=4.389, b=5.342, c=4.224 A.'
)


def request_payload(inp):
    def phase(phase_id, d):
        return PhaseInput(
            phase_id=phase_id, label=phase_id, point_group=d['point_group'],
            lattice=LatticeInput(
                d['a'], d['b'], d['c'], d['alpha'], d['beta'],
                d['gamma'], 'angstrom'))
    return CalculationRequest(
        project_id='blind_habit_fraction_audit', title='Independent root audit',
        parent=phase('A', inp['parent']), product=phase('M', inp['product']),
        transformation=TransformationInput.from_rows(
            'A_to_M', 'A', 'M', inp['C_M_from_A'])
    ).to_project_payload()


def input_states():
    d = json.loads(Path('data/benchmarks/bhattacharya_cualni_orthorhombic_habit_v1.json').read_text())
    standard = copy.deepcopy(d['input'])  # Do NOT access 'expected'.
    novel = copy.deepcopy(standard)
    # Directly reported lengths from Novak et al. (1999), not inferred from the
    # 1999 publication's predicted habit-plane or twin-fraction outputs.
    for key in ('a','b','c'):
        novel['parent'][key] = 5.836
    for key, value in [('a',4.389),('b',5.342),('c',4.224)]:
        novel['product'][key] = value
    novel.pop('reported_stretch_ratios', None)
    return [('bhattacharya_reference_input', standard),
            ('novak_1999_2H_lengths_with_assumed_2H_correspondence', novel)]


def unique(xs, eps=1e-6):
    result = []
    for x in sorted(map(float, xs)):
        if not result or abs(x-result[-1]) > eps:
            result.append(x)
    return result


def middle_gap(U, increment, fraction):
    F = U + fraction * increment
    sigma = np.linalg.svd(F, compute_uv=False)
    return float(np.sort(sigma)[1] - 1.0)


def bracket_oracle(U, increment):
    """Independent dense sign-bracketing plus tangent/near-root minimization.

    This never uses PTMC's determinant-quadratic reconstruction. The method
    can miss roots closer than the scan resolution, so suspicious tangencies
    or near-flat regions explicitly make the result UNRESOLVED.
    """
    grid = np.linspace(0.0,1.0,N_GRID)
    gaps = np.array([middle_gap(U,increment,float(f)) for f in grid])
    f_gap = lambda f: middle_gap(U,increment,float(f))
    roots = []
    suspicious = []
    if np.count_nonzero(np.abs(gaps)<ROOT_RESIDUAL) > 4:
        return [], ['near-continuum/flat middle-singular-value function']
    for j in range(len(grid)-1):
        l,r = float(grid[j]),float(grid[j+1])
        yl,yr = float(gaps[j]),float(gaps[j+1])
        if abs(yl) <= ROOT_RESIDUAL:
            roots.append(l)
        if yl*yr < 0:
            root = brentq(f_gap,l,r,xtol=5e-14,rtol=9e-16)
            if abs(f_gap(root)) <= ROOT_RESIDUAL:
                roots.append(root)
            else:
                suspicious.append(f'bracket residual too high near {root:.9g}')
    if abs(gaps[-1])<=ROOT_RESIDUAL:
        roots.append(1.0)
    # Non-sign-changing roots may be tangencies: search local minima of |g|.
    # A residual merely "close" to zero is NOT automatically declared a root.
    for j in range(1,len(grid)-1):
        if abs(gaps[j]) <= abs(gaps[j-1]) and abs(gaps[j]) <= abs(gaps[j+1]):
            opt = minimize_scalar(
                lambda f: abs(f_gap(f)),
                bounds=(float(grid[j-1]),float(grid[j+1])),
                method='bounded',options={'xatol':4e-14})
            f, err = float(opt.x),float(opt.fun)
            if err <= ROOT_RESIDUAL:
                roots.append(f)
            elif err < NEAR_THRESHOLD:
                suspicious.append(f'near-tangency at f={f:.10f}, residual={err:.3e}')
    roots=unique(roots, eps=5e-6)
    # Avoid declaring completeness for an unresolved two-root merger.
    for f in roots:
        if abs(f_gap(f)) > ROOT_RESIDUAL:
            suspicious.append(f'oracle root not sufficiently accurate at {f}')
    return roots,suspicious


def projective_angle(u,v):
    a=np.asarray(u,float);b=np.asarray(v,float)
    a=a/np.linalg.norm(a);b=b/np.linalg.norm(b)
    return float(np.arccos(np.clip(abs(a@b),-1.0,1.0)))


def independent_habit_normals(F):
    """Null-cone tangent-plane construction, not production's rank-one solver.

    If Q=F^T F-I has inertia (-,0,+), the two admissible interface
    tangent planes span (e2, sqrt(mu3-1)e1 +/- sqrt(1-mu1)e3).
    Their independent Cartesian normals are given by cross products.
    """
    mu,E = np.linalg.eigh(F.T@F)
    if not(mu[0] < 1-1e-6 and mu[2] > 1+1e-6):
        return None
    low,mid,high = (E[:,j] for j in range(3))
    normals=[]
    for sign in (-1,+1):
        tangent=np.sqrt(mu[2]-1)*low + sign*np.sqrt(1-mu[0])*high
        normal=np.cross(mid,tangent)
        normals.append(normal/np.linalg.norm(normal))
    return normals


def analyse_case(name, inp):
    payload=request_payload(inp)
    print(f'\n=== {name} ===',flush=True)
    print('Run 1/2: complete user-facing twin/habit application...',flush=True)
    app=build_twin_family_report(payload,'A_to_M')
    print('Run 2/2: retrieve all independent-scan INPUT twin gradients...',flush=True)
    project=project_from_dict(dict(payload),source='blind_roots').project
    ptmc=PTMCAdapter(project,'A_to_M').analyze_all_twinning(dilatational_factor=1.0)
    app_by_key=defaultdict(list)
    all_constructions=0
    for fam in app.families:
        for pair in fam.pair_records:
            for c in pair.constructions:
                all_constructions+=1
                if pair.stretch_i is None or pair.stretch_j is None:
                    raise AssertionError('construction without stretch-pair identity')
                key=(pair.stretch_i-1,pair.stretch_j-1,c.branch)
                app_by_key[key].extend(c.habit_solutions)
    ptmc_by_key={}
    for t in ptmc.twin_relations:
        key=(t.base_variant_index,t.other_variant_index,t.branch)
        if key in ptmc_by_key:
            raise AssertionError(f'duplicate PTMC twin-relation identity: {key}')
        ptmc_by_key[key]=t
    errors=[];undecidable=[];verified=0;scanned=0;independent_roots=0
    # PTMC enumerates BOTH ordered representations (i,j) and (j,i).
    # The user-facing tree intentionally keeps only one UNORDERED physical
    # variant pair, so reverse-directed keys must NOT be called missing twins.
    ptmc_unordered={tuple(sorted((i,j))) for i,j,_ in ptmc_by_key}
    app_unordered={tuple(sorted((i,j))) for i,j,_ in app_by_key}
    for pair in sorted(ptmc_unordered-app_unordered):
        errors.append(f'{pair}: unordered physical PTMC twin pair absent from full app')
    for pair in sorted(ptmc_unordered & app_unordered):
        # PTMC includes both directions: count branches in ONE direction only.
        possible_directed = [(i,j,branch) for i,j,branch in ptmc_by_key
                             if (i,j)==pair]
        actual_directed = [(i,j,branch) for i,j,branch in app_by_key
                           if tuple(sorted((i,j)))==pair]
        if len(actual_directed) != len(possible_directed):
            errors.append(f'{pair}: expected {len(possible_directed)} M/M branches '
                          f'for an unoriented pair; tree contains {len(actual_directed)}')
    for key in sorted(app_by_key):
        rel=ptmc_by_key.get(key)
        if rel is None:
            errors.append(f'{key}: app construction not present in PTMC relation list')
            continue
        U=ptmc.variants[rel.base_variant_index].U
        inc=np.outer(rel.a,rel.n_reference)
        # Before habit auditing, independently re-evaluate the *input* twin
        # equation; a failed twin means the downstream fraction is meaningless.
        twin_err=np.linalg.norm(rel.rotation_hat @ ptmc.variants[rel.other_variant_index].U
                                - U - inc)
        rot_err=np.linalg.norm(rel.rotation_hat.T @ rel.rotation_hat - np.eye(3))
        if twin_err>1e-7 or rot_err>1e-7 or np.linalg.det(rel.rotation_hat)<0:
            errors.append(f'{key}: wrong input M/M rank-one gradient '
                          f'(rank_one={twin_err:.3e}, rotation={rot_err:.3e})')
        # Critical: root oracle only sees U, a and n, NEVER solver fractions.
        roots,uncertain=bracket_oracle(U,inc)
        undecidable.extend(f'{key}: {x}' for x in uncertain)
        scanned+=1; independent_roots+=len(roots)
        reported=app_by_key.get(key)
        if reported is None:
            errors.append(f'{key}: PTMC twin relation missing from full app')
            continue
        reported_roots=unique([h.other_variant_volume_fraction for h in reported])
        for f in roots:
            if min([abs(f-x) for x in reported_roots],default=1)>FRACTION_MATCH:
                errors.append(f'{key}: OMITTED independent fraction {f:.12f}')
        for f in reported_roots:
            if min([abs(f-x) for x in roots],default=1)>FRACTION_MATCH:
                if not uncertain:
                    errors.append(f'{key}: APP fraction lacks independent root {f:.12f}')
                else:
                    undecidable.append(f'{key}: app fraction {f:.12f} lacks resolved oracle root')
        for f in roots:
            hs=[h for h in reported if abs(h.other_variant_volume_fraction-f)<FRACTION_MATCH]
            if not hs:
                continue
            F=U+f*inc
            g=middle_gap(U,inc,f)
            if abs(g)>ROOT_RESIDUAL:
                errors.append(f'{key}: alleged independent root residual {g:.4e}')
            geom=independent_habit_normals(F)
            if geom is None:
                undecidable.append(f'{key}: degenerate null-cone normals at f={f:.9f}')
                continue
            for n in geom:
                if min(projective_angle(n,h.habit_normal_parent_cartesian) for h in hs)>NORMAL_TOL_RAD:
                    errors.append(f'{key}: MISSING one independent habit-normal branch at f={f:.10f}')
            for h in hs:
                m=np.asarray(h.habit_normal_parent_cartesian,float)
                b=np.asarray(h.shape_vector_parent_cartesian,float)
                H=np.eye(3)+np.outer(b,m)
                relative=np.linalg.norm(F.T@F-H.T@H)/max(1.0,np.linalg.norm(F.T@F))
                if np.linalg.det(F)<=0 or abs(np.linalg.norm(m)-1)>1e-6 or relative>TENSOR_TOL:
                    errors.append(f'{key}: FALSE habit rank-one geometry at f={f:.10f}, gram_res={relative:.3e}')
                if min(projective_angle(m,n) for n in geom)>NORMAL_TOL_RAD:
                    errors.append(f'{key}: habit normal violates independently constructed null cone')
                verified+=1
    app_count=sum(len(v) for v in app_by_key.values())
    print('Full app: twin constructions=',all_constructions,
          'habit solutions=',app_count,
          'topology variants=',app.audit.topology_variant_count,flush=True)
    print('Oracle: twin branches=',scanned,'fraction roots=',independent_roots,
          'habit solutions independently checked=',verified,flush=True)
    print('Contradictions=',len(errors),'unresolved=',len(undecidable),flush=True)
    for e in errors[:15]: print('  FAIL:',e)
    for u in undecidable[:12]: print('  UNRESOLVED:',u)
    return dict(case=name, source=SOURCE if name.startswith('novak') else
        'Bhattacharya published worked Cu-Al-Ni 2H example',
        correspondence_status='assumed standard DO3-to-2H for Novak specimen',
        twin_constructions=all_constructions,app_habit_solutions=app_count,
        independent_scan_roots=independent_roots,independently_checked_habits=verified,
        failures=errors,unresolved=undecidable,
        verdict='FAIL' if errors else ('UNRESOLVED' if undecidable else 'PASS_CONDITIONAL'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=['baseline','novak','both'],default='both')
    parser.add_argument('--report',default='/tmp/habit_completeness_audit.json')
    args=parser.parse_args()
    chosen=input_states()
    if args.case=='baseline': chosen=chosen[:1]
    if args.case=='novak': chosen=chosen[1:]
    rows=[analyse_case(name,inp) for name,inp in chosen]
    Path(args.report).write_text(json.dumps(rows,indent=2))
    print('\nREPORT:',args.report,flush=True)
    print('OVERALL:', 'FAIL' if any(x['verdict']=='FAIL' for x in rows) else
          'UNRESOLVED' if any(x['verdict']=='UNRESOLVED' for x in rows) else
          'PASS_CONDITIONAL',flush=True)
    return 1 if any(x['verdict']=='FAIL' for x in rows) else 2 if any(
        x['verdict']=='UNRESOLVED' for x in rows) else 0

if __name__=='__main__':
    raise SystemExit(main())
