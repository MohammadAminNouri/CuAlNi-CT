
from cualni_cryst.project_state import james_hane_6m_reference_project
from cualni_cryst.supercompatibility_atlas import (
    evaluate_supercompatibility_atlas,
    independent_product_lattice_parameters,
    symmetry_preserving_product_grid,
    uniform_product_scale_sweep,
)


def test_uniform_scale_sweep_is_explicit_and_preserves_every_state():
    project = james_hane_6m_reference_project()
    states = uniform_product_scale_sweep(
        project,
        "do3_to_6m_reference",
        [0.995, 1.0, 1.005],
    )
    assert len(states) == 3
    assert [item[2]["scale_factor"] for item in states] == [0.995, 1.0, 1.005]

    report = evaluate_supercompatibility_atlas(
        states,
        "do3_to_6m_reference",
    )
    assert len(report.states) == 3
    assert sum(report.counts.values()) == 3
    assert all(state.state_id for state in report.states)
    assert all(state.matched_relation_count >= 0 for state in report.states)
    assert all(
        state.pair_agreement_count + state.pair_disagreement_count
        + state.not_comparable_pair_count == state.matched_relation_count
        for state in report.states
    )


def test_symmetry_preserving_grid_varies_only_independent_product_cell_parameters():
    project = james_hane_6m_reference_project()
    transformation_id = "do3_to_6m_reference"
    names = independent_product_lattice_parameters(project, transformation_id)
    assert names
    # A small one-axis grid is enough to audit that the helper returns validated
    # ProjectStates instead of arbitrary metric perturbations.
    axis = names[0]
    states = symmetry_preserving_product_grid(
        project,
        transformation_id,
        {axis: [0.999, 1.0, 1.001]},
    )
    assert len(states) == 3
    for state_id, candidate, parameters in states:
        assert state_id
        assert axis in parameters
        candidate.validate().assert_passed()


def test_no_state_level_false_supercompatibility_from_different_twin_branches():
    """Never aggregate Eq.42 on one twin with BJ CC2 on another twin."""
    from cualni_cryst.supercompatibility_atlas import (
        AtlasClass, evaluate_supercompatibility_state,
    )
    p=james_hane_6m_reference_project()
    state=evaluate_supercompatibility_state(p,"do3_to_6m_reference")
    if state.classification is AtlasClass.BOTH:
        assert any(
            pair.classification is AtlasClass.BOTH
            and pair.match_completeness=="full"
            and pair.match_geometry_within_tolerance is True
            and not pair.missing_required_components
            and pair.ct_supercompatible
            and pair.cofactor_compatible
            for pair in state.pair_results
        )
    assert (state.pair_agreement_count+state.pair_disagreement_count+
            state.not_comparable_pair_count==state.matched_relation_count)
