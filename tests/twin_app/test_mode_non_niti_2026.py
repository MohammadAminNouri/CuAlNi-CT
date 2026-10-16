"""Non-NiTi mode classification test: data-driven orthorhombic transformation."""
import json
from pathlib import Path
import pytest

from app.application import CalculationRequest, LatticeInput, PhaseInput, TransformationInput
from twin_app.mode_catalogue import build_mode_catalogue
from twin_app.scientific_engine import build_twin_family_report

BENCHMARK = Path(__file__).resolve().parents[2] / 'data/benchmarks/bhattacharya_cualni_orthorhombic_habit_v1.json'

@pytest.fixture(scope='module')
def calculated():
    data=json.loads(BENCHMARK.read_text())
    inp=data['input']
    def phase(key, pid):
        x=inp[key]
        return PhaseInput(pid,key,LatticeInput(x['a'],x['b'],x['c'],x['alpha'],x['beta'],x['gamma'],'angstrom'),x['point_group'])
    payload=CalculationRequest('non_niti_mode_validation','Non NiTi modes', phase('parent','A'),phase('product','M'), TransformationInput.from_rows('A_to_M','A','M',inp['C_M_from_A'])).to_project_payload()
    report=build_twin_family_report(payload,'A_to_M')
    return data,build_mode_catalogue(report,payload)


def test_orthorhombic_types_and_habits_come_from_calculation(calculated):
    reference,catalogue=calculated
    assert catalogue.modes
    for target in reference['expected']['habit_modes']:
        branches=[h for mode in catalogue.modes for e in mode.by_type(target['classification']) for h in e.construction.habit_solutions]
        assert any(min(abs(h.other_variant_volume_fraction-target['lambda']),
                       abs(1-h.other_variant_volume_fraction-target['lambda']))
                   <=target['lambda_abs_tol'] for h in branches),target
