"""Input-derived twinning-mode acceptance tests; literature is oracle ONLY here."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.application import CalculationRequest, LatticeInput, PhaseInput, TransformationInput
from twin_app.mode_catalogue import build_mode_catalogue, ModeClassificationError
from twin_app.scientific_engine import build_twin_family_report

BENCHMARK = Path(__file__).resolve().parents[2] / "data/benchmarks/bhattacharya_niti_twin_habit_v1.json"


def payload_for(input_data):
    inp = input_data
    def phase(i, label):
        values = inp[label]
        return PhaseInput(
            phase_id=i, label=label, point_group=values['point_group'],
            lattice=LatticeInput(values['a'], values['b'], values['c'],
                                 values['alpha'], values['beta'], values['gamma'], 'angstrom'),
        )
    return CalculationRequest(
        project_id='blind_mode_validation', title='NiTi mode validation',
        parent=phase('A', 'parent'), product=phase('M', 'product'),
        transformation=TransformationInput.from_rows('A_to_M', 'A', 'M', inp['C_M_from_A']),
    ).to_project_payload()


@pytest.fixture(scope='module')
def computed():
    data = json.loads(BENCHMARK.read_text())
    payload = payload_for(data['input'])
    report = build_twin_family_report(payload, 'A_to_M')
    return data, report, build_mode_catalogue(report, payload)


def _mode_for_shear(catalogue, expected, tol):
    matches = [mode for mode in catalogue.modes if abs(mode.shear - expected) <= tol]
    assert len(matches) == 1, (expected, [(m.mode_id, m.shear) for m in catalogue.modes])
    return matches[0]


def test_niti_modes_derived_without_reference_values(computed):
    _, report, modes = computed
    assert report.audit.topology_variant_count == 12
    assert modes.mapped_stretch_count == 12
    assert modes.stretch_pair_count == 66
    assert modes.correspondence_pair_count == 66
    assert len(modes.modes) == 4
    assert len(modes.incompatible_pair_orbits) == 1
    assert sorted(len(mode.stretch_pairs) for mode in modes.modes) == [6, 12, 12, 12]
    assert len(modes.incompatible_pair_orbits[0]) == 24


def test_every_published_niti_twinning_shear_and_type_is_visible(computed):
    reference, _, modes = computed
    for entry in reference['expected']['twin_shear_families']:
        mode = _mode_for_shear(modes, entry['shear'], entry['abs_tol'])
        assert entry['classification'] in mode.types, (
            mode.mode_id, entry['classification'], mode.types)
        assert mode.by_type(entry['classification'])
        assert any(abs(e.construction.shear_magnitude - entry['shear']) <= entry['abs_tol']
                   for e in mode.by_type(entry['classification']))


def test_published_niti_habit_fractions_remain_associated_with_correct_type(computed):
    reference, _, modes = computed
    for expected in reference['expected']['habit_fraction_targets']:
        mode = _mode_for_shear(modes, expected['shear'], 0.001)
        alternatives = [h for item in mode.by_type(expected['classification'])
                        for h in item.construction.habit_solutions]
        assert any(min(abs(h.other_variant_volume_fraction - expected['lambda']),
                       abs(1 - h.other_variant_volume_fraction - expected['lambda']))
                   <= expected['lambda_abs_tol'] for h in alternatives)


def test_niti_no_habit_modes_are_preserved_and_not_hidden(computed):
    reference, _, modes = computed
    for expected in reference['expected']['no_exact_interface_shears']:
        mode = _mode_for_shear(modes, expected['shear'], expected['abs_tol'])
        assert mode.entries
        assert all(not e.construction.habit_solutions and not e.construction.continuum_fraction
                   for e in mode.entries)


def test_mode_catalogue_exposes_every_calculated_twin_branch(computed):
    _, report, modes = computed
    from_report = {
        (family.family_id, pair.pair_id, b.construction_id, b.branch)
        for family in report.families for pair in family.pair_records
        for b in pair.constructions
    }
    from_modes = {
        (e.operator_family, e.pair.pair_id, e.construction.construction_id, e.construction.branch)
        for mode in modes.modes for e in mode.entries
    }
    assert from_modes == from_report, (len(from_modes), len(from_report), from_report-from_modes)


def test_no_literature_values_in_production_mode_modules():
    root = Path(__file__).resolve().parents[2] / 'twin_app'
    for fname in ('mode_catalogue.py', 'mode_renderer.py'):
        text = (root/fname).read_text()
        for prohibited in ('Bhattacharya', '0.2385', '0.2804', '0.3096', '0.1423', 'NiTi', 'CuAlNi'):
            assert prohibited not in text, (fname, prohibited)
