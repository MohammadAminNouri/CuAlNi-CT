from __future__ import annotations

import math

import pytest
import sympy as sp

from cualni_cryst.ar_gen_cry_examples import (
    ARCHIVE, EXAMPLES, NI_TI_REFERENCE_PAIRS, SOURCE_FILES,
    provenance, parse_cif_phase, correspondence_from_pairs, invert_correspondence,
)
from cualni_cryst.legacy_suite_bridge import exact_correspondence
from cualni_cryst.ar_gen_cry_pipeline import cry_stage, gen_stage


@pytest.mark.parametrize("key", list(EXAMPLES))
def test_all_archived_example_cells_validate_with_exact_symmetry(key):
    example = EXAMPLES[key]
    probe = cry_stage(example.parent, example.product, (1, 0, 0), (1, 0, 0))
    assert probe["parent"]["point_group_order"] >= 2
    assert probe["product"]["point_group_order"] >= 2
    assert probe["parent"]["d_spacing_angstrom"] > 0
    assert probe["product"]["d_spacing_angstrom"] > 0
    audit = provenance(key)
    assert audit["kind"] == "archived_example"
    assert audit["archive"] == ARCHIVE
    assert len(audit["files"]) == 2
    assert all(item["sha256"] == SOURCE_FILES[item["path"]] for item in audit["files"])


def test_archived_niti_parameters_and_monoclinic_angle():
    e = EXAMPLES["NiTi · B2 → B19′"]
    assert e.parent.a == 3.015
    assert e.parent.point_group == "m-3m"
    assert e.product.a == 2.889
    assert e.product.b == 4.12
    assert e.product.c == 4.622
    assert e.product.point_group == "2/m"
    assert e.product.beta == pytest.approx(96.8)
    assert len(NI_TI_REFERENCE_PAIRS) == 2
    assert "OR" in e.status
    assert provenance(e.label, edited=True)["kind"] == "edited_archived_example"


def test_guided_correspondence_identity_and_rotation():
    identity = "1 0 0\n0 1 0\n0 0 1"
    assert correspondence_from_pairs(identity, identity) == identity
    second = "0 1 0\n-1 0 0\n0 0 1"
    mapped = correspondence_from_pairs(identity, second)
    C = exact_correspondence(mapped).C_M_from_A
    assert C == sp.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert exact_correspondence(invert_correspondence(mapped)).C_M_from_A == C.inv()
    assert invert_correspondence(invert_correspondence(mapped)) == mapped


@pytest.mark.parametrize("a,b", [
    ("1 0 0\n1 0 0\n0 0 1", "1 0 0\n0 1 0\n0 0 1"),
    ("1 0 0\n0 1 0\n0 0 1", "1 0 0\n0 0 0\n0 0 1"),
    ("1 0 0\n0 1 0", "1 0 0\n0 1 0\n0 0 1"),
    ("1 0 0\n0 1 0\n0 0 1", "1 0 0\n0 __import__ 0\n0 0 1"),
])
def test_guided_correspondence_fails_closed(a, b):
    with pytest.raises((ValueError, TypeError, ZeroDivisionError)):
        correspondence_from_pairs(a, b)


MINIMAL_CIF = b"""data_NiTi_B2
_cell_length_a       3.015(2)
_cell_length_b       3.015
_cell_length_c       3.015
_cell_angle_alpha    90
_cell_angle_beta     90
_cell_angle_gamma    90
"""


def test_cif_import_is_explicit_and_reads_uncertainty_notation():
    phase = parse_cif_phase(MINIMAL_CIF, "B2", "m-3m")
    assert phase.name == "B2"
    assert phase.a == pytest.approx(3.015)
    assert phase.beta == pytest.approx(90)
    assert phase.point_group == "m-3m"
    assert len(phase.verified_symmetry()) == 48


@pytest.mark.parametrize("source", [
    b"", b"\x00bad",
    b"data_bad\n_cell_length_a 1\n",
    b"data_test\n_cell_length_a foo\n",
])
def test_cif_rejects_incomplete_or_malformed_input(source):
    with pytest.raises((ValueError, TypeError)):
        parse_cif_phase(source, "Bad", "m-3m")


def test_archived_niti_groupoid_demo_is_formal_not_experimental():
    e = EXAMPLES["NiTi · B2 → B19′"]
    cry = cry_stage(e.parent, e.product, (1, 0, 0), (1, 0, 0))
    result = gen_stage(cry, "1 0 0\n0 1 0\n0 0 1")
    assert result["burnside_count"] == result["operator_classes"]
    assert result["subgroup_order"] * result["variants"] == 48
    assert "experimental OR" in result["provenance"]
