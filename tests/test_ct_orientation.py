from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import sympy as sp

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.ct_orientation import CayronOrientationAdapter
from cualni_cryst.cualni_models import do3_to_6m_branch
from cualni_cryst.group_theory import correspondence_groupoid
from cualni_cryst.lattice import Lattice
from cualni_cryst.orientation import (
    OrientationService,
    matrix_from_axis_angle,
    rotation_audit,
)
from cualni_cryst.project_state import (
    OrientationTheoryOrigin,
    james_hane_6m_reference_project,
)
from cualni_cryst.representation import CartesianConvention
from cualni_cryst.symmetry import classify_symmetry, cubic_full_m3m
from cualni_cryst.twinning_ct import (
    type_i_from_parent_reflection,
    type_ii_from_parent_twofold,
)


@pytest.fixture
def service() -> OrientationService:
    return OrientationService(james_hane_6m_reference_project())


@pytest.fixture
def adapter(service: OrientationService) -> CayronOrientationAdapter:
    return CayronOrientationAdapter(service, "do3_to_6m_reference")


def _first_noncollapsed_reflection(adapter: CayronOrientationAdapter):
    for operation in cubic_full_m3m():
        if classify_symmetry(operation).kind != "reflection":
            continue
        try:
            return operation, adapter.type_i_from_parent_reflection(operation)
        except ValueError as exc:
            if "Collapsed CT" not in str(exc):
                raise
    raise AssertionError("No noncollapsed Type-I relation found")


def _first_noncollapsed_twofold(adapter: CayronOrientationAdapter):
    for operation in cubic_full_m3m():
        info = classify_symmetry(operation)
        if not (info.kind == "rotation" and info.order == 2):
            continue
        try:
            return operation, adapter.type_ii_from_parent_twofold(operation)
        except ValueError as exc:
            if "Collapsed CT" not in str(exc):
                raise
    raise AssertionError("No noncollapsed Type-II relation found")


@pytest.mark.parametrize("kind", ["I", "II"])
def test_closing_gap_candidates_are_proper_and_satisfy_exact_parallelisms(
    adapter: CayronOrientationAdapter,
    kind: str,
):
    if kind == "I":
        _, report = _first_noncollapsed_reflection(adapter)
    else:
        _, report = _first_noncollapsed_twofold(adapter)

    assert report.twin_kind == kind
    assert report.selected is None
    assert report.natural_orientation_id == ""
    assert 1 <= len(report.candidates) <= 4

    for candidate in report.candidates:
        matrix = np.asarray(candidate.orientation.R_reference_from_moving)
        assert rotation_audit(matrix).maximum_residual < 1.0e-10
        assert candidate.rotation_residual < 1.0e-10
        assert candidate.direction_parallelism_residual_deg < 1.0e-6
        assert candidate.plane_parallelism_residual_deg < 1.0e-6
        assert candidate.parent_incidence_residual_deg < 1.0e-6
        assert candidate.martensite_incidence_residual_deg < 1.0e-6

    assert report.correspondence_plane_residual_deg < 1.0e-6
    assert report.correspondence_direction_residual_deg < 1.0e-6
    assert report.intercorrespondence_residual < 1.0e-10


@pytest.mark.parametrize(
    "angle_deg",
    [
        0.0,
        1.0e-9,
        1.0e-8,
        1.0e-7,
        1.0e-6,
        1.0e-5,
        1.0e-3,
        90.0,
        179.999999,
        180.0,
    ],
)
def test_natural_or_rotation_angle_retains_small_rotations(
    angle_deg: float,
):
    rotation = matrix_from_axis_angle(
        np.array([1.0, 2.0, -3.0]),
        angle_deg,
    )

    measured = CayronOrientationAdapter._rotation_angle_deg(rotation)

    assert measured == pytest.approx(
        angle_deg,
        abs=2.0e-10,
    )


def test_natural_or_rotation_angle_does_not_collapse_1e_minus_7_degree():
    angle_deg = 1.0e-7

    rotation = matrix_from_axis_angle(
        np.array([0.371, -0.492, 0.787]),
        angle_deg,
    )

    measured = CayronOrientationAdapter._rotation_angle_deg(rotation)

    assert measured > 0.0

    assert measured == pytest.approx(
        angle_deg,
        rel=2.0e-6,
        abs=2.0e-12,
    )


def test_explicit_natural_or_selects_minimum_symmetry_reduced_branch(
    adapter: CayronOrientationAdapter,
):
    operation, first = _first_noncollapsed_reflection(adapter)
    natural = first.candidates[0].orientation

    ranked = adapter.type_i_from_parent_reflection(
        operation,
        natural_orientation=natural,
    )

    assert ranked.selected is not None
    assert ranked.minimum_natural_deviation_deg is not None
    assert ranked.minimum_natural_deviation_deg < 1.0e-7
    assert ranked.selected_candidate_index in ranked.selection_tie_indices


def test_polar_orientation_remains_a_comparator_and_is_not_relabelled(
    adapter: CayronOrientationAdapter,
):
    polar = adapter.polar_comparator()
    assert polar.theory_origin is OrientationTheoryOrigin.POLAR_CORRESPONDENCE

    natural_hypothesis = adapter.natural_orientation(polar)
    assert (
        natural_hypothesis.state.theory_origin
        is OrientationTheoryOrigin.POLAR_CORRESPONDENCE
    )
    assert any(
        "does not establish" in warning for warning in natural_hypothesis.warnings
    )


def test_polar_hypothesis_warning_is_propagated_into_closing_gap_report(
    adapter: CayronOrientationAdapter,
):
    operation, _ = _first_noncollapsed_reflection(adapter)
    report = adapter.type_i_from_parent_reflection(
        operation,
        natural_orientation=adapter.polar_comparator(),
    )
    assert any("R_polar" in warning for warning in report.warnings)


def test_natural_or_phase_mismatch_is_rejected(
    service: OrientationService,
    adapter: CayronOrientationAdapter,
):
    wrong_direction = service.state_from_matrix(
        "6m",
        "do3",
        np.eye(3),
    )
    with pytest.raises(ValueError, match="reference phase"):
        adapter.natural_orientation(wrong_direction)


def test_closing_gap_orientation_is_representation_invariant(
    service: OrientationService,
    adapter: CayronOrientationAdapter,
):
    _, report = _first_noncollapsed_twofold(adapter)
    state = report.candidates[0].orientation

    symmetric = service.reexpress(
        state,
        CartesianConvention.SYMMETRIC_METRIC,
        CartesianConvention.SYMMETRIC_METRIC,
    )
    comparison = service.compare_orientations(state, symmetric)
    assert comparison.symmetry_reduced_disorientation_deg < 1.0e-7


def test_full_double_coset_path_keeps_exact_parent_symmetry_provenance(
    service: OrientationService,
    adapter: CayronOrientationAdapter,
):
    branch = do3_to_6m_branch()
    groupoid = correspondence_groupoid(
        list(branch.parent_point_group),
        list(branch.product_point_group),
        branch.correspondence,
    )

    operator = next(
        operator
        for operator in groupoid.operators
        if adapter.from_operator(operator).twin_reports
    )

    report = adapter.from_operator(operator)
    assert report.twin_reports
    assert report.best_report_index is None

    natural = report.twin_reports[0].candidates[0].orientation
    ranked = adapter.from_operator(operator, natural_orientation=natural)
    assert ranked.best_report_index is not None
    assert ranked.minimum_natural_deviation_deg is not None
    assert ranked.minimum_natural_deviation_deg < 1.0e-7


def test_generic_noncubic_parent_order_two_isometries_are_supported():
    parent = Lattice.monoclinic_unique_b(
        3.0,
        4.0,
        5.0,
        104.0,
        length_unit="angstrom",
    )
    product = Lattice(
        3.2,
        4.1,
        5.3,
        82.0,
        101.0,
        76.0,
        length_unit="angstrom",
    )
    correspondence = Correspondence(
        sp.eye(3),
        label="identity synthetic noncubic regression",
    )

    reflection = sp.diag(1, -1, 1)
    twofold = sp.diag(-1, 1, -1)

    type_i = type_i_from_parent_reflection(
        reflection,
        parent.metric(),
        product.metric(),
        correspondence,
    )
    type_ii = type_ii_from_parent_twofold(
        twofold,
        parent.metric(),
        product.metric(),
        correspondence,
    )

    assert type_i.kind == "I"
    assert type_ii.kind == "II"
    assert np.isfinite(type_i.shear) and type_i.shear > 0.0
    assert np.isfinite(type_ii.shear) and type_ii.shear > 0.0
    assert abs(type_i.plane_m @ type_i.direction_m) < 1.0e-10
    assert abs(type_ii.plane_m @ type_ii.direction_m) < 1.0e-10


def test_parent_operation_that_does_not_preserve_metric_is_rejected():
    parent = Lattice.monoclinic_unique_b(3.0, 4.0, 5.0, 104.0)
    product = Lattice(3.2, 4.1, 5.3, 82.0, 101.0, 76.0)
    correspondence = Correspondence(sp.eye(3))

    not_a_symmetry = sp.Matrix([[1, 1, 0], [0, -1, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="does not preserve"):
        type_i_from_parent_reflection(
            not_a_symmetry,
            parent.metric(),
            product.metric(),
            correspondence,
        )


def test_weak_twin_route_refuses_fake_exact_orientation(
    adapter: CayronOrientationAdapter,
):
    with pytest.raises(NotImplementedError, match="intrinsic distortion"):
        adapter.weak_orientation_prediction()


def test_natural_or_bound_to_different_transformation_is_rejected(
    adapter: CayronOrientationAdapter,
):
    _, report = _first_noncollapsed_reflection(adapter)
    state = replace(
        report.candidates[0].orientation,
        transformation_id="some_other_transformation",
    )
    with pytest.raises(ValueError, match="different transformation"):
        adapter.natural_orientation(state)
