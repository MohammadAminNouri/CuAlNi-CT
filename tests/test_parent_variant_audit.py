from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cualni_cryst.lattice import Lattice
from cualni_cryst.ebsd_map import EBSDPhase
from cualni_cryst.parent_variant_audit import (
    parent_generated_variants, daughter_variant_assignment,
)


def symmetries():
    pa = EBSDPhase.from_point_group(
        1, "beta", Lattice(5.8, 5.8, 5.8, 90, 90, 90), "m-3m",
    ).proper_symmetry_cartesian
    ma = EBSDPhase.from_point_group(
        2, "18R", Lattice(10.0, 5.0, 4.0, 90, 108, 90), "2/m",
    ).proper_symmetry_cartesian
    return pa, ma


def test_cubic_to_monoclinic_variant_count_and_C2_quotient():
    p, m = symmetries()
    family = parent_generated_variants(np.eye(3), p, m)
    # For R=identity the C2 about monoclinic b coincides with parent C2,
    # hence 24/2 = 12 inequivalent parent-generated variants.
    assert len(family) == 12
    assert len(parent_generated_variants(
        Rotation.from_euler("xyz", [12, 27, 6], degrees=True).as_matrix(), p, m
    )) == 24


def test_complete_variant_assignment_invariant_to_indexer_C2_copy():
    p, m = symmetries()
    gA = Rotation.from_euler("xyz", [25, 10, -12], degrees=True).as_matrix()
    R = Rotation.from_euler("xyz", [10, 17, 8], degrees=True).as_matrix()
    family = parent_generated_variants(R, p, m)
    gM = gA @ family[6]
    observed = np.stack((gM, gM @ m[1]))
    result = daughter_variant_assignment(observed, gA, R, p, m)
    assert result["parent_generated_variant_count"] == 24
    assert result["accepted_fraction"] == 1.0
    assert result["median_residual_deg"] < 1e-5
    assert result["variant_index_1based"][0] == result["variant_index_1based"][1]


def test_wrong_parent_has_high_residual_for_generic_child():
    p, m = symmetries()
    R = Rotation.from_euler("xyz", [5, 11, 13], degrees=True).as_matrix()
    gA = Rotation.from_euler("xyz", [21, 8, 43], degrees=True).as_matrix()
    measured = (gA @ parent_generated_variants(R, p, m)[3])[None,:,:]
    bad = Rotation.from_euler("xyz", [58, 77, -90], degrees=True).as_matrix()
    good = daughter_variant_assignment(measured, gA, R, p, m)
    wrong = daughter_variant_assignment(measured, bad, R, p, m)
    assert good["accepted_fraction"] == 1.0
    assert wrong["median_residual_deg"] > good["median_residual_deg"] + 1


def test_non_so3_reference_or_rejected():
    p, m = symmetries()
    with pytest.raises(ValueError):
        parent_generated_variants(np.diag([1,1,-1]), p, m)
