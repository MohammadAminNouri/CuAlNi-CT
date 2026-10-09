from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from twin_app.branch_presentation import distinct_twin_branches
from twin_app.input_explainer import preview_correspondence, validate_metric_parameters, convert_cell_lengths
from twin_app.structure_validation import parse_cif, independent_spglib_check


@dataclass(frozen=True)
class FakeBranch:
    branch: int
    a_parent_cartesian: tuple[float, float, float]
    n_parent_cartesian: tuple[float, float, float]
    classification: str = "unresolved"
    habit_solutions: tuple = ()


def b(branch, a, n):
    return FakeBranch(branch, tuple(a), tuple(n))


def test_niti_correspondence_explains_exact_columns_and_convention():
    value = preview_correspondence(
        (("0", "0", "1"), ("1/2", "1/2", "0"), ("-1/2", "1/2", "0")),
        direction="A_TO_M",
    )
    assert value.basis_mappings == (
        ("Parent [1,0,0]", "Product [0, 1/2, -1/2]"),
        ("Parent [0,1,0]", "Product [0, 1/2, 1/2]"),
        ("Parent [0,0,1]", "Product [1, 0, 0]"),
    )
    assert "orientation relationship" in value.notes[1]
    assert value.determinant != "0"


def test_reverse_exact_correspondence_preserves_parent_basis_mapping():
    original = (("0", "0", "1"), ("1/2", "1/2", "0"), ("-1/2", "1/2", "0"))
    import sympy as sp
    reverse = sp.Matrix(original).inv()
    recovered = preview_correspondence(
        [[str(reverse[i,j]) for j in range(3)] for i in range(3)], direction="M_TO_A"
    )
    expected = preview_correspondence(original, direction="A_TO_M")
    assert recovered.canonical_matrix == expected.canonical_matrix
    assert recovered.basis_mappings == expected.basis_mappings


@pytest.mark.parametrize("angles", [(0, 90, 90), (180, 90, 90), (10, 10, 175), (np.nan, 90, 90)])
def test_metric_validator_rejects_unphysical_cells(angles):
    with pytest.raises(ValueError):
        validate_metric_parameters((3, 4, 5), angles)


def test_metric_validator_accepts_general_monoclinic_cell():
    vol, condition = validate_metric_parameters((2.889, 4.120, 4.622), (90, 96.8, 90))
    assert vol > 0
    assert 1 <= condition < 10


def test_distinct_twins_do_not_depend_on_algebraic_branch_sign():
    a = b(-1, (0.1, 0.2, 0), (1, 0, 0))
    c = b(+1, (0, 0.2, 0.1), (0, 1, 0))
    first = distinct_twin_branches((a, c))
    second = distinct_twin_branches((c, a))
    assert len(first) == len(second) == 2
    assert [np.outer(x.representative.a_parent_cartesian, x.representative.n_parent_cartesian).tolist() for x in first] == [np.outer(x.representative.a_parent_cartesian, x.representative.n_parent_cartesian).tolist() for x in second]
    assert [x.label for x in first] == ["Interface A", "Interface B"]


def test_duplicate_physical_tensor_collapses_but_all_source_indices_remain():
    a = b(-1, (0.2, 0, 0), (1, 0, 0))
    same_tensor = b(+1, (-0.2, 0, 0), (-1, 0, 0))
    found = distinct_twin_branches((a, same_tensor))
    assert len(found) == 1
    assert set(found[0].source_branch_signs) == {-1, 1}
    assert set(found[0].equivalent_source_indices) == {0, 1}


def test_cif_import_reads_atomic_basis_without_inferring_correspondence():
    cif = b'''data_B2
_cell_length_a 3.015
_cell_length_b 3.015
_cell_length_c 3.015
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ni1 Ni 0 0 0
Ti1 Ti 0.5 0.5 0.5
'''
    struct = parse_cif(cif, filename="B2.cif")
    assert struct.site_count == 2
    assert struct.lengths == (3.015, 3.015, 3.015)
    assert struct.atomic_numbers == (28, 22)
    assert not hasattr(struct, "correspondence")


def test_independent_spglib_is_explicit_optional_and_never_generates_correspondence(monkeypatch):
    cif = b'''data_crystal
_cell_length_a 3
_cell_length_b 3
_cell_length_c 3
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ni1 Ni 0 0 0
'''
    data = parse_cif(cif, filename="test.cif")
    observed = {}
    def detector(cell, symprec):
        observed["cell"] = cell
        return SimpleNamespace(international="Pm-3m", number=221, pointgroup="m-3m", hall="-P 4 2 3")
    monkeypatch.setitem(sys.modules, "spglib", SimpleNamespace(get_symmetry_dataset=detector))
    result = independent_spglib_check(data)
    assert result["pointgroup"] == "m-3m"
    assert observed["cell"][0].shape == (3, 3)
    assert "no lattice correspondence inferred" in result["verification_scope"]


def test_zero_shear_vector_cannot_masquerade_as_real_twin():
    with pytest.raises(ValueError, match="nonzero"):
        distinct_twin_branches((b(-1, (0, 0, 0), (1, 0, 0)),))


def test_actual_spglib_b2_when_scientific_extra_is_available():
    pytest.importorskip("spglib")
    cif = b'''data_B2
_cell_length_a 3.015
_cell_length_b 3.015
_cell_length_c 3.015
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ni1 Ni 0 0 0
Ti1 Ti 0.5 0.5 0.5
'''
    result = independent_spglib_check(parse_cif(cif, filename="B2.cif"))
    assert result["pointgroup"] == "m-3m"
    assert result["number"] == 221


def test_switching_lattice_unit_preserves_physical_lengths():
    nm = convert_cell_lengths((3.015, 4.12, 4.622), from_unit="angstrom", to_unit="nanometer")
    assert np.allclose(nm, (0.3015, 0.412, 0.4622))
    assert np.allclose(convert_cell_lengths(nm, from_unit="nanometer", to_unit="angstrom"), (3.015, 4.12, 4.622))
    assert convert_cell_lengths((100,), from_unit="picometer", to_unit="angstrom") == (1,)
