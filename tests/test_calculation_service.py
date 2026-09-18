import json
from dataclasses import replace

import numpy as np

from cualni_cryst.calculation_service import (
    CalculationKind,
    CalculationService,
    DiscreteTopologyResult,
    MetricCoreResult,
    TransformationBundle,
    dependency_order,
)
from cualni_cryst.lattice import Lattice
from cualni_cryst.project_state import james_hane_6m_reference_project
from cualni_cryst.representation import CartesianConvention
from cualni_cryst.stretch import principal_stretches, stretch_from_metrics

TRANSFORMATION_ID = "do3_to_6m_reference"


def _project_with_product_beta(beta_deg: float):
    project = james_hane_6m_reference_project()
    parent = project.phase("austenite_do3")
    product = project.phase("martensite_long_period")
    lattice = Lattice.monoclinic_unique_b(
        product.lattice.a,
        product.lattice.b,
        product.lattice.c,
        beta_deg,
        label=product.lattice.label,
    )
    changed_product = replace(product, lattice=lattice)
    return replace(project, phases=(parent, changed_product))


def test_dependency_graph_for_bundle_is_acyclic_and_complete():
    order = dependency_order(CalculationKind.TRANSFORMATION_BUNDLE)

    assert order[-1] is CalculationKind.TRANSFORMATION_BUNDLE
    assert len(order) == len(set(order))
    assert CalculationKind.DISCRETE_TOPOLOGY in order
    assert CalculationKind.METRIC_CORE in order
    assert CalculationKind.AM_COMPATIBILITY in order
    assert CalculationKind.SCIENTIFIC_CONTRACTS in order


def test_reference_bundle_reproduces_locked_discrete_topology():
    service = CalculationService(james_hane_6m_reference_project())

    bundle = service.compute(
        CalculationKind.TRANSFORMATION_BUNDLE,
        TRANSFORMATION_ID,
    )

    assert isinstance(bundle, TransformationBundle)
    assert bundle.topology.parent_group_order == 48
    assert bundle.topology.product_group_order == 4
    assert bundle.topology.subgroup_order == 4
    assert bundle.topology.n_variants == 12
    assert bundle.topology.n_operators == 8
    assert bundle.topology.groupoid.burnside_count == 8


def test_metric_core_matches_independent_existing_stretch_api():
    project = james_hane_6m_reference_project()
    service = CalculationService(project)

    result = service.compute(CalculationKind.METRIC_CORE, TRANSFORMATION_ID)

    assert isinstance(result, MetricCoreResult)

    transformation = project.transformation(TRANSFORMATION_ID)
    parent = project.phase(transformation.parent_phase_id)
    product = project.phase(transformation.product_phase_id)
    U = stretch_from_metrics(
        parent.lattice.metric(),
        product.lattice.metric(),
        transformation.correspondence,
    )
    lambdas, _ = principal_stretches(U)

    assert np.allclose(result.stretch, U, atol=1e-12, rtol=1e-12)
    assert np.allclose(result.principal_stretches, lambdas, atol=1e-12, rtol=1e-12)


def test_repeated_calculation_hits_fingerprint_cache():
    service = CalculationService(james_hane_6m_reference_project())

    first = service.compute(CalculationKind.METRIC_CORE, TRANSFORMATION_ID)
    after_first = service.cache_stats
    second = service.compute(CalculationKind.METRIC_CORE, TRANSFORMATION_ID)
    after_second = service.cache_stats

    assert first is second
    assert after_first.misses >= 1
    assert after_second.hits == after_first.hits + 1
    assert after_second.entries == after_first.entries


def test_beta_change_invalidates_metric_domains_but_not_discrete_topology():
    original = james_hane_6m_reference_project()
    changed = _project_with_product_beta(96.0)
    service = CalculationService(original)

    impact = service.impact(changed, TRANSFORMATION_ID)

    assert CalculationKind.DISCRETE_TOPOLOGY in impact.unchanged
    assert CalculationKind.METRIC_CORE in impact.changed
    assert CalculationKind.REPRESENTATION in impact.changed
    assert CalculationKind.AM_COMPATIBILITY in impact.changed
    assert CalculationKind.SCIENTIFIC_CONTRACTS in impact.changed
    assert CalculationKind.STRETCH_VARIANTS in impact.changed
    assert CalculationKind.TRANSFORMATION_BUNDLE in impact.changed


def test_cartesian_convention_change_only_invalidates_representation_and_bundle():
    project = james_hane_6m_reference_project()
    transformation = project.transformation(TRANSFORMATION_ID)
    changed_transformation = replace(
        transformation,
        parent_cartesian_convention=CartesianConvention.PTCLAB_A_X_C_XZ,
    )
    changed = replace(project, transformations=(changed_transformation,))
    service = CalculationService(project)

    impact = service.impact(changed, TRANSFORMATION_ID)

    assert CalculationKind.DISCRETE_TOPOLOGY in impact.unchanged
    assert CalculationKind.METRIC_CORE in impact.unchanged
    assert CalculationKind.AM_COMPATIBILITY in impact.unchanged
    assert CalculationKind.SCIENTIFIC_CONTRACTS in impact.unchanged
    assert CalculationKind.STRETCH_VARIANTS in impact.unchanged
    assert CalculationKind.REPRESENTATION in impact.changed
    assert CalculationKind.TRANSFORMATION_BUNDLE in impact.changed


def test_service_reuses_topology_cache_after_metric_only_state_change():
    original = james_hane_6m_reference_project()
    changed = _project_with_product_beta(96.0)
    service = CalculationService(original)

    topology_before = service.compute(
        CalculationKind.DISCRETE_TOPOLOGY,
        TRANSFORMATION_ID,
    )
    stats_before = service.cache_stats

    service.set_project(changed)
    topology_after = service.compute(
        CalculationKind.DISCRETE_TOPOLOGY,
        TRANSFORMATION_ID,
    )
    stats_after = service.cache_stats

    assert topology_before is topology_after
    assert stats_after.hits == stats_before.hits + 1


def test_metric_cache_is_not_reused_after_beta_change():
    original = james_hane_6m_reference_project()
    changed = _project_with_product_beta(96.0)
    service = CalculationService(original)

    before = service.compute(CalculationKind.METRIC_CORE, TRANSFORMATION_ID)
    service.set_project(changed)
    after = service.compute(CalculationKind.METRIC_CORE, TRANSFORMATION_ID)

    assert isinstance(before, MetricCoreResult)
    assert isinstance(after, MetricCoreResult)
    assert before is not after
    assert not np.allclose(before.product_metric, after.product_metric)


def test_stretch_variants_use_metric_orthonormal_parent_rotations():
    service = CalculationService(james_hane_6m_reference_project())

    result = service.compute(CalculationKind.STRETCH_VARIANTS, TRANSFORMATION_ID)

    assert result.proper_parent_rotation_count == 24
    assert result.n_variants == 12
    for variant in result.variants:
        assert np.allclose(variant, variant.T, atol=1e-12)


def test_source_rounded_reference_is_not_silently_promoted_to_exact_am_compatibility():
    service = CalculationService(james_hane_6m_reference_project())

    result = service.compute(CalculationKind.AM_COMPATIBILITY, TRANSFORMATION_ID)

    assert not result.ct_exact
    assert not result.ball_james_lambda2_exact
    assert result.exact_classification_agreement
    assert result.ct.approximate.residual > 0.0


def test_bundle_and_impact_are_json_ready_for_future_ui():
    service = CalculationService(james_hane_6m_reference_project())
    bundle = service.compute(CalculationKind.TRANSFORMATION_BUNDLE, TRANSFORMATION_ID)
    changed = _project_with_product_beta(96.0)

    bundle_json = json.dumps(bundle.to_dict())
    impact_json = json.dumps(service.impact(changed, TRANSFORMATION_ID).to_dict())

    assert '"n_variants": 12' in bundle_json
    assert '"metric_core"' in impact_json


def test_topology_result_type_is_explicit():
    service = CalculationService(james_hane_6m_reference_project())

    result = service.compute(CalculationKind.DISCRETE_TOPOLOGY, TRANSFORMATION_ID)

    assert isinstance(result, DiscreteTopologyResult)
