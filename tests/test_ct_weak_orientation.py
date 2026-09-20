from __future__ import annotations

import numpy as np
import sympy as sp

from cualni_cryst.ct_weak_orientation import CTWeakOrientationAdapter
from cualni_cryst.orientation import OrientationService, rotation_audit
from cualni_cryst.project_state import james_hane_6m_reference_project
from cualni_cryst.symmetry import cubic_full_m3m
from cualni_cryst.weak_twins import BravaisNodeBasis


def _higher_order_proper_parent_rotation() -> sp.Matrix:
    for operation in cubic_full_m3m():
        matrix = sp.Matrix(operation)
        if matrix.det() != 1:
            continue
        if matrix == sp.eye(3) or matrix**2 == sp.eye(3):
            continue
        if len((matrix - sp.eye(3)).nullspace()) == 1:
            return matrix
    raise AssertionError("cubic m-3m unexpectedly lacks a higher-order proper rotation")


def test_ct_weak_route_is_additive_and_returns_audited_proper_ors():
    project = james_hane_6m_reference_project()
    service = OrientationService(project)
    adapter = CTWeakOrientationAdapter(service, "do3_to_6m_reference")

    report = adapter.from_parent_operation(
        _higher_order_proper_parent_rotation(),
        product_node_basis=BravaisNodeBasis.primitive_conventional(),
        max_plane_index=2,
        maximum_generalized_shear=None,
        max_weak_twins=8,
    )

    assert report.weak_twins
    assert report.orientation_candidates
    assert report.selected is None
    assert report.natural_orientation_id == ""
    assert len(report.parent_axis) == 3
    assert len(report.product_axis) == 3

    for candidate in report.orientation_candidates:
        matrix = np.asarray(candidate.orientation.R_reference_from_moving)
        assert rotation_audit(matrix).maximum_residual < 1.0e-10
        assert candidate.direction_parallelism_residual_deg < 1.0e-6
        assert candidate.plane_parallelism_residual_deg < 1.0e-6


def test_exact_type_ii_operation_is_rejected_by_weak_route():
    project = james_hane_6m_reference_project()
    service = OrientationService(project)
    adapter = CTWeakOrientationAdapter(service, "do3_to_6m_reference")

    twofold = next(
        sp.Matrix(operation)
        for operation in cubic_full_m3m()
        if sp.Matrix(operation).det() == 1
        and sp.Matrix(operation) != sp.eye(3)
        and sp.Matrix(operation) ** 2 == sp.eye(3)
    )

    try:
        adapter.from_parent_operation(
            twofold,
            product_node_basis=BravaisNodeBasis.primitive_conventional(),
            max_plane_index=1,
        )
    except ValueError as exc:
        assert "Type-II" in str(exc)
    else:
        raise AssertionError(
            "exact Type-II operator was incorrectly sent to weak route"
        )
