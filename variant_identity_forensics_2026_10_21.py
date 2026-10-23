#!/usr/bin/env python3
"""Independent variant-identity audit for CuAlNi-CT (2026-10-21).

Does not alter any scientific application source files. Demonstrates the
well-defined exact variant orbit of a cubic-parent/orthorhombic-product
synthetic state and compares two public-library production generators.
Nothing in the oracle is a hard-coded published Cu-Al-Ni answer.

Usage from Codespace repository root:
  python -u variant_identity_forensics_2026_10_21.py --selftest
  python -u variant_identity_forensics_2026_10_21.py --phase baseline
  python -u variant_identity_forensics_2026_10_21.py --phase hostile
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import mpmath as mp
import numpy as np

ROOT = Path.cwd().resolve()
if not (ROOT / 'src/cualni_cryst').is_dir():
    raise SystemExit('Run this script from the CuAlNi-CT repository root (src/cualni_cryst not found).')
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

EPS_GRID = ('1e-6', '1e-8', '3e-9', '1e-9', '3e-10', '1e-10', '1e-11', '1e-12', '0')
# Conserve CPU: no full Streamlit/build_twin_family_report inside hostile sweeps.
ALG_SWEEP = (1e-8, 1e-10, 1e-12)
BJ_SWEEP = (1e-8, 1e-9, 1e-11, 1e-12)
OUT_DIR = ROOT / 'audit_findings' / 'variant_identity_2026_10_21'
REPAIR_PATH = ROOT / 'audit_findings' / 'REPAIR_WORKLIST.md'
MASTER_PATH = ROOT / 'audit_findings' / 'SCIENTIFIC_REPAIR_MASTER.jsonl'


def proper_signed_permutations():
    """Generate cubic proper point group *independently* of application API."""
    seen = set()
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            Q = np.zeros((3, 3), dtype=float)
            for i, j in enumerate(perm):
                Q[i, j] = signs[i]
            if round(np.linalg.det(Q)) != 1:
                continue
            key = tuple(int(x) for x in Q.reshape(-1))
            if key not in seen:
                seen.add(key)
                out.append(Q)
    if len(out) != 24:
        raise AssertionError(f'Cubic proper group should contain 24 operations; got {len(out)}')
    return tuple(out)


def binary_exact(x):
    a, b = float(x).as_integer_ratio()
    return mp.mpf(a) / b


def case(eps_text, length_scale):
    e = float(eps_text)
    # In exact arithmetic, a cubic parent, distinct orthorhombic stretches
    # and the identity correspondence yield a proper cubic orbit of six.
    # At epsilon=0 the first two principal stretches coincide: orbit of three.
    stretches = np.array((0.92 - e, 0.92 + e, 1.17), float)
    S = float(length_scale)
    ma = np.eye(3) * (S*S)
    mm = np.diag((S*stretches)**2)
    return ma, mm, stretches


def exact_binary64_orbit_size(ma, mm, rotations):
    # All case matrices are diagonal; compute U diagonal using 90-digit math.
    # Convert IEEE 754 values *exactly* so the reference is independent of
    # Python's display str precision and of production matrix square roots.
    with mp.workdps(90):
        diag = [mp.sqrt(binary_exact(mm[i, i]) / binary_exact(ma[i, i])) for i in range(3)]
        signatures = set()
        for Q in rotations:
            # Exact sign-permutation conjugation of diagonal U.
            indices = [int(np.argmax(np.abs(Q[i]))) for i in range(3)]
            signatures.add(tuple(diag[j] for j in indices))
        return len(signatures), sorted(float(abs(diag[i]-diag[j])) for i in range(3) for j in range(i+1,3))[0]


def minimum_pair_gap(variants):
    matrices = [np.asarray(x.U, dtype=float) for x in variants]
    return min((float(np.linalg.norm(a-b, 'fro')) for a,b in itertools.combinations(matrices,2)),default=None)


def reference_orbit_matrices(ma, mm, rotations):
    """Exact-binary64 90-digit diagonal reference, rounded only for comparison."""
    with mp.workdps(90):
        diag = [mp.sqrt(binary_exact(mm[i, i]) / binary_exact(ma[i, i])) for i in range(3)]
        seen = {}
        for Q in rotations:
            indices = tuple(int(np.argmax(np.abs(Q[i]))) for i in range(3))
            key = tuple(diag[j] for j in indices)
            seen[key] = np.diag([float(v) for v in key])
        return list(seen.values())


def geometric_orbit_residual(variants, expected_matrices):
    """Best one-to-one coverage of the independent orbit, not only its count."""
    from scipy.optimize import linear_sum_assignment
    have = [np.asarray(v.U,dtype=float) for v in variants]
    if not have or not expected_matrices:
        return float('inf'), 0
    matrix = np.array([[np.linalg.norm(a-b,'fro') for b in expected_matrices] for a in have])
    row,col = linear_sum_assignment(matrix)
    return float(np.max(matrix[row,col])), len(row)


def provenance_misassignments(variants, ma, mm, rotations, tol):
    """Independent check of every claimed generating cubic symmetry index."""
    base = np.diag(np.sqrt(np.diag(mm) / np.diag(ma)))
    failures=[]
    for v in variants:
        for idx in v.generating_parent_symmetry_indices:
            Q=rotations[int(idx)]
            candidate=Q@base@Q.T
            d=float(np.linalg.norm(candidate-np.asarray(v.U,float),ord='fro'))
            if d>tol:
                failures.append(dict(variant=int(v.index),generator=int(idx),distance=d))
    return failures


def run_once(eps_text, length_scale, algebraic, bj_tol, ordering):
    from cualni_cryst.ptmc_adapter import _build_variants
    from cualni_cryst.ball_james_adapter import (
        generate_stretch_variants, parent_symmetry_rotations,
    )
    parent, product, stretches = case(eps_text, length_scale)
    rotations = proper_signed_permutations()
    if ordering == 'reverse':
        rotations = tuple(reversed(rotations))
    elif ordering == 'sorted':
        rotations = tuple(sorted(rotations, key=lambda q: tuple(-int(z) for z in q.reshape(-1))))
    elif ordering != 'forward':
        raise ValueError(ordering)
    actual_parent = tuple(rotations)
    proper = parent_symmetry_rotations(parent, actual_parent, tolerance=max(algebraic, 1e-9))
    U0 = np.diag(stretches)
    bj = generate_stretch_variants(U0, proper, tolerance=bj_tol)
    pt = _build_variants(parent, product, np.eye(3), actual_parent, algebraic_tolerance=algebraic)
    exact, exact_gap = exact_binary64_orbit_size(parent, product, rotations)
    # Independent roundoff floor: scale with stretch tensor norm, not with an
    # unrelated user-controlled algebraic admissibility tolerance.
    resolution = float(128.0*np.finfo(float).eps*max(1.0, float(np.linalg.norm(U0, 'fro'))))
    expected = exact if exact_gap > 8*resolution or exact == 3 else None
    reference=reference_orbit_matrices(parent, product, rotations)
    pt_cover,pt_pairs=geometric_orbit_residual(pt,reference)
    bj_cover,bj_pairs=geometric_orbit_residual(bj,reference)
    # Candidate representations can differ by tiny sqrt/linear algebra errors;
    # generators may only claim the same physical stretch within this bound.
    provenance_tol = resolution * 4
    pt_false=provenance_misassignments(pt,parent,product,rotations,provenance_tol)
    bj_false=provenance_misassignments(bj,parent,product,rotations,provenance_tol)
    return dict(epsilon=eps_text, lattice_scale=length_scale,
                algebraic_tolerance=algebraic,bj_variant_tolerance=bj_tol,operation_order=ordering,
                exact_binary64_orbit=exact,exact_min_principal_gap=exact_gap,
                roundoff_resolution=resolution,expected_resolved_count=expected,
                ptmc_count=len(pt),ball_james_count=len(bj),
                ptmc_min_returned_gap=minimum_pair_gap(pt),bj_min_returned_gap=minimum_pair_gap(bj),
                ptmc_orbit_max_one_to_one_error=pt_cover,bj_orbit_max_one_to_one_error=bj_cover,
                ptmc_orbit_matched_count=pt_pairs,bj_orbit_matched_count=bj_pairs,
                ptmc_false_generator_membership=pt_false,bj_false_generator_membership=bj_false,
                ptmc_generators=[list(v.generating_parent_symmetry_indices) for v in pt],
                bj_generators=[list(v.generating_parent_symmetry_indices) for v in bj],
                ptmc_variant_loss=bool(expected is not None and (len(pt)!=expected or pt_cover>provenance_tol)),
                ball_james_variant_loss=bool(expected is not None and (len(bj)!=expected or bj_cover>provenance_tol)))


def write_rows(rows, phase):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outfile = OUT_DIR / ('baseline.json' if phase=='baseline' else 'hostile.json')
    with outfile.open('w',encoding='utf8') as f:
        json.dump({'audit':'2026-10-21-variant-identity-v1', 'phase':phase,
                   'timestamp_utc':datetime.now(timezone.utc).isoformat(),
                   'rows':rows},f,indent=2,sort_keys=True)
    return outfile


def append_findings(rows, phase):
    findings = []
    for row in rows:
        if row.get('expected_resolved_count') is None:
            continue
        for field,key,path,fn in (
            ('ptmc_count','P1_VARIANT_LOSS_PTMC','src/cualni_cryst/ptmc_adapter.py','_build_variants'),
            ('ball_james_count','P1_VARIANT_LOSS_BJ','src/cualni_cryst/ball_james_adapter.py','generate_stretch_variants')):
            if row[field] == row['expected_resolved_count']:
                continue
            finding = {'audit':'variant-identity-2026-10-21', 'phase':phase,
                       'issue':key, 'status':'REPRODUCED_NUMERIC_MERGE_DEFECT',
                       'scientific_scope':'synthetic orthorhombic stretch, cubic parent, exact binary64 input',
                       'input':{k:row[k] for k in ['epsilon','lattice_scale','algebraic_tolerance','bj_variant_tolerance','operation_order']},
                       'reference_method':'90-digit exact-binary64 diagonal sqrt and proper signed-permutation orbit',
                       'reference_count':row['expected_resolved_count'], 'observed_count':row[field],
                       'independent_principal_gap':row['exact_min_principal_gap'],
                       'roundoff_resolution':row['roundoff_resolution'],
                       'suspected_source':path,'suspected_function':fn,
                       'priority':'P1 - distinct resolvable variants merged',
                       'note':'Does not establish specimen-specific validity of a predicted martensite habit.'}
            findings.append(finding)
    if not findings:
        return 0
    MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_keys=set()
    if MASTER_PATH.exists():
        for l in MASTER_PATH.read_text(encoding='utf8').splitlines():
            try:
                d=json.loads(l)
                if d.get('audit')=='variant-identity-2026-10-21':
                    existing_keys.add(json.dumps((d.get('phase'),d.get('issue'),d.get('input')),sort_keys=True))
            except (ValueError,TypeError):
                pass
    appended=[]
    with MASTER_PATH.open('a',encoding='utf8') as f:
        for item in findings:
            key=json.dumps((item['phase'],item['issue'],item['input']),sort_keys=True)
            if key not in existing_keys:
                f.write(json.dumps(item,sort_keys=True)+'\n')
                existing_keys.add(key)
                appended.append(item)
    if appended:
        with REPAIR_PATH.open('a',encoding='utf8') as f:
            f.write('\n## Reproduced variant-orbit identity faults: '+datetime.now(timezone.utc).strftime('%Y-%m-%d')+'\n')
            for item in appended:
                f.write(f"- [{item['issue']}] epsilon={item['input']['epsilon']}, scale={item['input']['lattice_scale']}, "
                        f"order={item['input']['operation_order']}, reference={item['reference_count']}, "
                        f"production={item['observed_count']}. Investigate {item['suspected_source']} "
                        f"::{item['suspected_function']}; DO NOT fix with another arbitrary identity threshold.\n")
    return len(appended)


def selftest():
    rotations=proper_signed_permutations()
    assert len(rotations)==24
    assert all(abs(np.linalg.det(Q)-1)<1e-12 for Q in rotations)
    assert all(np.allclose(Q.T@Q,np.eye(3),atol=0,rtol=0) for Q in rotations)
    for eps,expected in [('0',3),('1e-6',6),('1e-10',6)]:
        a,b,_ = case(eps,1)
        exact,_ = exact_binary64_orbit_size(a,b,rotations)
        assert exact==expected,(eps,exact,expected)
    print('SELFTEST PASS: 24 proper cubic operations, 3/6 exact orbit switch, IEEE-754-exact high-precision reference.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--phase', choices=['baseline','hostile','all'],default=None)
    args=parser.parse_args()
    if args.selftest:
        selftest()
        if args.phase is None:return 0
    if args.phase is None:
        parser.error('Choose --selftest or --phase baseline|hostile|all')
    phases=['baseline','hostile'] if args.phase=='all' else [args.phase]
    for phase in phases:
        rows=[]
        if phase=='baseline':
            configurations=[(e,1.0,1e-10,1e-9,'forward') for e in EPS_GRID]
        else:
            configurations=[]
            for e in ('1e-8','1e-9','1e-10','1e-11','1e-12','0'):
                for ordering in ('forward','reverse','sorted'):
                    configurations.append((e,1.0,1e-10,1e-9,ordering))
            for e in ('1e-9','1e-11'):
                for alg in ALG_SWEEP:
                    configurations.append((e,1.0,alg,1e-9,'forward'))
                for bt in BJ_SWEEP:
                    configurations.append((e,1.0,1e-10,bt,'forward'))
                for s in (0.01,100.0):
                    configurations.append((e,s,1e-10,1e-9,'forward'))
        for j, conf in enumerate(configurations,1):
            try:
                row=run_once(*conf)
                rows.append(row)
                label='FAIL' if (row['ptmc_variant_loss'] or row['ball_james_variant_loss'] or row['ptmc_false_generator_membership'] or row['bj_false_generator_membership']) else 'PASS'
                print(f"{phase} [{j:02d}/{len(configurations):02d}] eps={row['epsilon']:>6} "
                      f"scale={row['lattice_scale']:>5} order={row['operation_order']:<7} "
                      f"reference={row['exact_binary64_orbit']} PTMC={row['ptmc_count']} "
                      f"BJ={row['ball_james_count']} {label}",flush=True)
            except Exception as e:
                # A failure is not erased or silently reinterpreted as PASS.
                row={'input':conf,'error_type':type(e).__name__,'error':str(e),
                     'status':'EXECUTION_ERROR'}
                rows.append(row)
                print(f'{phase} [{j:02d}/{len(configurations):02d}] ERROR {type(e).__name__}: {e}',flush=True)
        outfile=write_rows(rows,phase)
        new=append_findings(rows,phase)
        failures=sum(r.get('ptmc_variant_loss',False) or r.get('ball_james_variant_loss',False)
                     or bool(r.get('ptmc_false_generator_membership')) or bool(r.get('bj_false_generator_membership'))
                     or 'error' in r for r in rows)
        print(f'=== {phase.upper()} SUMMARY: {failures}/{len(rows)} divergent cases; {new} new ledger entries. ===')
        print('JSON:',outfile)
        print('MASTER:',MASTER_PATH)
        print('WORKLIST:',REPAIR_PATH)
    return 0


if __name__=='__main__':
    sys.exit(main())
