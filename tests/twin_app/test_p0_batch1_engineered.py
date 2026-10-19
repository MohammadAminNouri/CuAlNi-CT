"""Behavioural P0 regression tests for scientific input and report traceability.

Unlike a literal-word assertion, CIF/metric tests exercise the real functions.
Real browser interaction is a separate manual acceptance gate.
"""
from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pytest

from twin_app.structure_validation import parse_cif, independent_spglib_check
from twin_app.input_explainer import (
    cell_compatible_with_family, convert_cell_lengths, validate_metric_parameters,
)
from twin_app.input_logic import canonical_correspondence_rows
from twin_app.point_group_visualizer import cell_basis, element_geometry

ROOT = Path(__file__).resolve().parents[2]

SIC_CIF = b"""data_SiC
_cell_length_a 4.358
_cell_length_b 4.358
_cell_length_c 4.358
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'F -4 3 m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0 0 0
C1 C .25 .25 .25
"""


def test_asymmetric_unit_expands_using_cif_declared_operations():
    model = parse_cif(SIC_CIF, filename="SiC.cif")
    assert model.site_count == 8  # 2 asymmetric atoms, 4 formula units
    assert model.atomic_numbers.count(14) == 4
    assert model.atomic_numbers.count(6) == 4
    assert model.symmetry_model.replace(' ', '') == 'F-43m'
    assert not model.has_partial_occupancy


def test_cif_atom_expansion_does_not_infer_a_transformation_correspondence():
    model = parse_cif(SIC_CIF, filename="SiC.cif")
    assert not hasattr(model, 'correspondence')
    assert not hasattr(model, 'orientation_relationship')
    # A symmetry-check result based on CIF-declared operations is not
    # independent diffraction evidence of those operations.
    pytest.importorskip('spglib')
    evidence = independent_spglib_check(model)
    assert evidence['number'] == 216
    assert 'not independent experimental confirmation' in evidence['verification_scope']
    assert 'no lattice correspondence inferred' in evidence['verification_scope']


def test_partial_occupancy_cif_imports_metric_but_blocks_atomic_symmetry():
    cif = SIC_CIF.replace(b'C1 C .25 .25 .25', b'C1 C .25 .25 .25')
    cif = cif.replace(b'_atom_site_fract_z\n', b'_atom_site_fract_z\n_atom_site_occupancy\n')
    cif = cif.replace(b'Si1 Si 0 0 0\n', b'Si1 Si 0 0 0 1\n').replace(
        b'C1 C .25 .25 .25\n', b'C1 C .25 .25 .25 0.5\n'
    )
    model = parse_cif(cif, filename="partial.cif")
    assert model.has_partial_occupancy
    with pytest.raises(ValueError, match='Partial occupanc'):
        independent_spglib_check(model)


@pytest.mark.parametrize('family,cell,expected', [
    ('triclinic', (3.1, 4.2, 5.0, 78, 82, 105), True),
    ('monoclinic', (3.1, 4.2, 5.0, 90, 103, 90), True),
    ('monoclinic', (3.1, 4.2, 5.0, 89, 103, 90), False),
    ('orthorhombic', (3.1, 4.2, 5.0, 90, 90, 90), True),
    ('orthorhombic', (3.1, 4.2, 5.0, 90, 103, 90), False),
    ('tetragonal', (3.1, 3.1, 5.0, 90, 90, 90), True),
    ('tetragonal', (3.1, 3.5, 5.0, 90, 90, 90), False),
    ('trigonal', (3.1, 3.1, 5.0, 90, 90, 120), True),
    ('trigonal', (3.1, 3.1, 3.1, 75, 75, 75), False),
    ('hexagonal', (3.1, 3.1, 5.0, 90, 90, 120), True),
    ('hexagonal', (3.1, 3.1, 5.0, 90, 90, 90), False),
    ('cubic', (3.1, 3.1, 3.1, 90, 90, 90), True),
    ('cubic', (3.1, 3.1, 5.0, 90, 90, 90), False),
])
def test_all_seven_crystal_systems_respect_selected_cell_constraints(family, cell, expected):
    assert cell_compatible_with_family(cell[:3], cell[3:], family) is expected


@pytest.mark.parametrize('unit,dimensions', [
    ('angstrom', (3.015, 4.120, 4.622)),
    ('nanometer', (0.3015, 0.4120, 0.4622)),
    ('picometer', (301.5, 412, 462.2)),
    ('micrometer', (0.0003015, 0.0004120, 0.0004622)),
    ('meter', (3.015e-10, 4.120e-10, 4.622e-10)),
])
def test_five_units_preserve_geometry_and_plane_duality(unit, dimensions):
    a, b, c = dimensions
    ang = (90, 102, 90)
    validate_metric_parameters(dimensions, ang)
    B = cell_basis(a, b, c, *ang)
    assert np.isfinite(B).all() and np.linalg.det(B) > 0
    _, axis = element_geometry(B, 'proper 2-fold rotation', (0, 0, 1))
    _, normal = element_geometry(B, 'mirror reflection', (0, 0, 1))
    assert abs(float(axis @ normal)) < 0.999
    in_angstrom = convert_cell_lengths(dimensions, from_unit=unit, to_unit='angstrom')
    assert np.allclose(in_angstrom, (3.015, 4.120, 4.622), rtol=1e-12, atol=1e-13)


def test_correspondence_inverse_is_exact_even_for_thirds():
    import sympy as sp
    C = sp.Matrix([[1, sp.Rational(1, 3), 0], [0, 1, 0], [0, 0, 2]])
    reverse = C.inv()
    a = canonical_correspondence_rows(C.tolist(), direction='A_TO_M')
    b = canonical_correspondence_rows(reverse.tolist(), direction='M_TO_A')
    assert a == b


def test_invalid_correspondence_fails_closed():
    with pytest.raises(ValueError, match='invertible'):
        canonical_correspondence_rows((('1', '0', '0'), ('1', '0', '0'), ('0', '0', '1')), direction='A_TO_M')


def test_report_exposes_all_distinct_grouped_habit_records():
    # A UI boundary source contract: the rank-one branch can be represented by
    # multiple source records, so the expanded details must not show only one.
    text = (ROOT / 'twin_app/tree_renderer.py').read_text(encoding='utf-8')
    assert 'for sol in selected_interface.all_habit_solutions:' in text
    assert 'scope_key=f"{node.key}:{selected_interface.label}"' in text


def test_ui_does_not_claim_unverified_preset_is_user_measured():
    ui = (ROOT / 'twin_app/streamlit_app.py').read_text(encoding='utf-8')
    assert '"status": "UNVERIFIED"' in ui
    assert 'cell_compatible_with_family(' in ui
    assert 'Nothing was imported.' in ui


def test_point_group_3d_unrendered_cases_disclose_limitations():
    ui = (ROOT / 'twin_app/input_ui.py').read_text(encoding='utf-8')
    assert '3D symmetry overlay is not implemented' in ui


def test_meter_scale_length_fields_preserve_visible_nonzero_significance():
    ui = (ROOT / 'twin_app/input_ui.py').read_text(encoding='utf-8')
    assert '"%.8e" if length_unit == "meter"' in ui
