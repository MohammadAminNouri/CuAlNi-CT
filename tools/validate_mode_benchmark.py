#!/usr/bin/env python3
"""ONE-SHOT independent benchmark audit. This tool is NOT used by the app.

Numerical solver always runs before published expected values are consulted.
Never import reference answers into the mode catalogue or scientific engine.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from app.application import CalculationRequest, LatticeInput, PhaseInput, TransformationInput
from twin_app.scientific_engine import build_twin_family_report
from twin_app.mode_catalogue import build_mode_catalogue

ROOT = Path(__file__).resolve().parents[1]


def make_phase(pid, label, data):
    return PhaseInput(pid,label,LatticeInput(data['a'],data['b'],data['c'],
                 data['alpha'],data['beta'],data['gamma'],'angstrom'),data['point_group'])


def main() -> int:
    fixture = json.loads((ROOT/'data/benchmarks/bhattacharya_niti_twin_habit_v1.json').read_text())
    inp = fixture['input']
    payload = CalculationRequest(
        'blind_mode_probe','Mode-to-table validation',
        make_phase('A','Parent',inp['parent']),
        make_phase('M','Product',inp['product']),
        TransformationInput.from_rows('A_to_M','A','M',inp['C_M_from_A']),
    ).to_project_payload()
    print('Calculating from input lattice metrics, point groups and exact C only...')
    report = build_twin_family_report(payload,'A_to_M')
    catalogue = build_mode_catalogue(report,payload)
    print('\nCALCULATED MODES (before consulting literature expected values)')
    for mode in catalogue.modes:
        print(f'{mode.mode_id}: {len(mode.stretch_pairs)} symmetry-related stretch couples; '
              f's={mode.shear:.9f}; types={", ".join(mode.types)}; '
              f'source_branches={len(mode.entries)}')
        for kind in mode.types:
            entries=mode.by_type(kind)
            habs=[h for e in entries for h in e.construction.habit_solutions]
            fractions=sorted({round(h.other_variant_volume_fraction,6) for h in habs})
            print(f'  {kind}: {len(entries)} source records; lambda={fractions or "NO EXACT HABIT"}')
    print(f'Non-twinning symmetry orbits: {[len(o) for o in catalogue.incompatible_pair_orbits]}')
    print('\nPUBLISHED REFERENCE COMPARISON (reference read only after calculation)')
    failures=[]
    expected=fixture['expected']
    for reference in expected['twin_shear_families']:
        found=[m for m in catalogue.modes if abs(m.shear-reference['shear'])<=reference['abs_tol']]
        matched=len(found)==1 and reference['classification'] in found[0].types
        print(('PASS' if matched else 'FAIL'),reference['classification'],
              f"s_published={reference['shear']:.6f}",
              f"s_computed={found[0].shear:.9f}" if found else 'MISSING')
        if not matched: failures.append(('shear',reference))
    for reference in expected['habit_fraction_targets']:
        found=[m for m in catalogue.modes if abs(m.shear-reference['shear'])<0.001]
        solutions=[h for m in found for e in m.by_type(reference['classification'])
                   for h in e.construction.habit_solutions]
        matched=any(min(abs(h.other_variant_volume_fraction-reference['lambda']),
                        abs(1-h.other_variant_volume_fraction-reference['lambda']))
                    <=reference['lambda_abs_tol'] for h in solutions)
        print(('PASS' if matched else 'FAIL'),reference['classification'],
              f"lambda_published={reference['lambda']:.5f}")
        if not matched:failures.append(('lambda',reference))
    for reference in expected['no_exact_interface_shears']:
        found=[m for m in catalogue.modes if abs(m.shear-reference['shear'])<=reference['abs_tol']]
        matched=len(found)==1 and all(not e.construction.habit_solutions and not e.construction.continuum_fraction
                                       for e in found[0].entries)
        print(('PASS' if matched else 'FAIL'),'NO EXACT A/M HABIT',reference['shear'])
        if not matched:failures.append(('no-habit',reference))
    if failures:
        print('\nFAILED: benchmark output not certified; do not present these modes as validated.')
        return 1
    print('\nPASSED: classification/shear/fraction/no-habit targets matched, up to published rounding.')
    print('Numerical twin plane, shear direction and full physical habit b,m still require the existing immutable oracle tests.')
    return 0


if __name__=='__main__':
    sys.exit(main())
