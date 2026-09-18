import numpy as np

from cualni_cryst.orientation import OrientationService
from cualni_cryst.project_state import james_hane_6m_reference_project
from cualni_cryst.representation import CartesianConvention


def _service():
    return OrientationService(james_hane_6m_reference_project())


def test_polar_reference_has_12_cayron_orientation_variants_not_24_raw_products():
    service = _service()
    state = service.polar_orientation("do3_to_6m_reference")
    topology = service.topology(state)

    assert topology.audit.full_reference_group_order == 48
    assert topology.audit.full_moving_group_order == 4
    assert topology.audit.proper_reference_group_order == 24
    assert topology.audit.proper_moving_group_order == 2

    assert topology.audit.full_orientation_intersection_order == 4
    assert topology.audit.proper_orientation_intersection_order == 2
    assert topology.audit.full_orientation_variant_count == 12
    assert topology.audit.proper_orientation_variant_count == 12
    assert len(service.variants(state)) == 12


def test_polar_reference_has_eight_orientation_double_coset_operators():
    service = _service()
    state = service.polar_orientation("do3_to_6m_reference")
    topology = service.topology(state)

    assert topology.audit.full_orientation_operator_count == 8
    assert topology.audit.proper_orientation_operator_count == 8
    assert len(service.operators(state)) == 8
    assert sum(operator.size for operator in service.operators(state)) == 48


def test_polar_reference_h_t_equals_h_c_but_only_as_computed_result():
    service = _service()
    state = service.polar_orientation("do3_to_6m_reference")
    audit = service.topology(state).audit

    assert audit.correspondence_intersection_order == 4
    assert audit.correspondence_variant_count == 12
    assert audit.correspondence_operator_count == 8
    assert audit.orientation_correspondence_intersections_equal is True
    assert audit.one_to_one_correspondence_orientation_topology is True
    assert audit.maximum_orientation_intersection_residual < 1.0e-10


def test_generic_or_does_not_assume_h_t_equals_h_c():
    service = _service()
    state = service.state_from_axis_angle(
        "do3",
        "6m",
        np.array([1.0, 2.0, 3.0]),
        17.0,
    )
    audit = service.topology(state).audit

    # For a generic relative orientation of two centrosymmetric crystals,
    # inversion remains common in the full groups, while no non-trivial proper
    # symmetry need remain coincident.
    assert audit.full_orientation_intersection_order == 2
    assert audit.proper_orientation_intersection_order == 1
    assert audit.full_orientation_variant_count == 24
    assert audit.proper_orientation_variant_count == 24

    # H_C belongs to the transformation/correspondence and stays unchanged.
    assert audit.correspondence_intersection_order == 4
    assert audit.correspondence_variant_count == 12
    assert audit.orientation_correspondence_intersections_equal is False
    assert audit.one_to_one_correspondence_orientation_topology is False


def test_variant_cosets_partition_all_24_proper_parent_symmetries_once():
    service = _service()
    state = service.polar_orientation("do3_to_6m_reference")
    variants = service.variants(state)

    flattened = [
        symmetry_index
        for variant in variants
        for symmetry_index in variant.reference_coset_symmetry_indices
    ]
    assert len(flattened) == 24
    assert len(set(flattened)) == 24
    assert all(variant.equivalent_proper_matrix_count == 2 for variant in variants)


def test_operator_classification_exposes_cayron_ambivalent_and_polar_classes():
    service = _service()
    state = service.polar_orientation("do3_to_6m_reference")
    operators = service.operators(state)

    classes = {operator.cayron_class for operator in operators}
    assert {"identity", "ambivalent", "polar"}.issubset(classes)
    assert any(operator.contains_parent_reflection for operator in operators)
    assert any(operator.contains_parent_180_rotation for operator in operators)


def test_topology_is_invariant_under_cartesian_reexpression():
    service = _service()
    state = service.polar_orientation("do3_to_6m_reference")
    base = service.topology(state).audit

    for reference_convention in CartesianConvention:
        for moving_convention in CartesianConvention:
            reexpressed = service.reexpress(
                state,
                reference_convention,
                moving_convention,
            )
            current = service.topology(reexpressed).audit
            assert current.full_orientation_intersection_order == (
                base.full_orientation_intersection_order
            )
            assert current.proper_orientation_intersection_order == (
                base.proper_orientation_intersection_order
            )
            assert current.full_orientation_variant_count == (
                base.full_orientation_variant_count
            )
            assert current.full_orientation_operator_count == (
                base.full_orientation_operator_count
            )
            assert current.orientation_correspondence_intersections_equal == (
                base.orientation_correspondence_intersections_equal
            )


def test_coordinate_axis_angle_can_change_while_topology_does_not():
    service = _service()
    state = service.polar_orientation("do3_to_6m_reference")
    symmetric_angle = service.report(state).axis_angle.angle_deg
    ptclab = service.reexpress(
        state,
        CartesianConvention.PTCLAB_A_X_C_XZ,
        CartesianConvention.PTCLAB_A_X_C_XZ,
    )
    ptclab_angle = service.report(ptclab).axis_angle.angle_deg

    assert abs(symmetric_angle - ptclab_angle) > 1.0
    assert service.topology(state).audit.full_orientation_variant_count == 12
    assert service.topology(ptclab).audit.full_orientation_variant_count == 12


def test_polar_report_labels_topology_as_cayron_style_without_promoting_to_cayron_t():
    service = _service()
    report = service.report(service.polar_orientation("do3_to_6m_reference"))
    warning_text = " ".join(report.warnings)

    assert report.orientation_variant_count == 12
    assert report.orientation_operator_count == 8
    assert "left cosets G_A/H_T" in warning_text
    assert "not automatically Cayron" in warning_text
