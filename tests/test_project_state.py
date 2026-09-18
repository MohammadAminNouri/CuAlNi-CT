import json

import numpy as np

from cualni_cryst.crystal_objects import CrystalBasisRef, Direction, Plane
from cualni_cryst.lattice import Lattice
from cualni_cryst.project_state import (
    Composition,
    CompositionBasis,
    CompositionComponent,
    PhaseState,
    ProjectState,
    StateProvenance,
    ThermomechanicalState,
    ValidationSeverity,
    james_hane_6m_reference_project,
)
from cualni_cryst.provenance import DataStatus


def test_reference_project_is_structurally_and_geometrically_valid():
    project = james_hane_6m_reference_project()
    report = project.validate()

    report.assert_passed()
    assert report.passed
    assert not report.errors


def test_long_period_physical_phase_is_separate_from_6m_cell_representation():
    project = james_hane_6m_reference_project()
    product = project.phase("martensite_long_period")

    assert product.physical_phase == "long-period Cu-Al-Ni martensite"
    assert product.cell_representation == "6M"
    assert product.basis.cell_representation == "6M"


def test_project_resolves_typed_object_to_the_correct_lattice():
    project = james_hane_6m_reference_project()
    product = project.phase("martensite_long_period")
    direction = Direction((1, 0, 1), product.basis)

    lattice = project.lattice_for(direction)

    assert lattice is product.lattice
    assert direction.length(lattice) > 0.0


def test_correspondence_mapping_round_trip_preserves_direction_and_plane():
    project = james_hane_6m_reference_project()
    parent = project.phase("austenite_do3")

    direction = Direction((1, 2, -1), parent.basis)
    plane = Plane((2, -1, 0), parent.basis)

    direction_m = project.map_direction("do3_to_6m_reference", direction)
    plane_m = project.map_plane("do3_to_6m_reference", plane)

    assert direction_m.basis == project.phase("martensite_long_period").basis
    assert plane_m.basis == project.phase("martensite_long_period").basis
    assert direction_m.provenance.status is DataStatus.COMPUTATION_DERIVED
    assert plane_m.provenance.status is DataStatus.COMPUTATION_DERIVED

    direction_back = project.map_direction("do3_to_6m_reference", direction_m)
    plane_back = project.map_plane("do3_to_6m_reference", plane_m)

    assert np.allclose(direction_back.array, direction.array, atol=1e-12)
    assert np.allclose(plane_back.array, plane.array, atol=1e-12)


def test_correspondence_mapping_preserves_incidence_pairing():
    project = james_hane_6m_reference_project()
    parent = project.phase("austenite_do3")

    direction = Direction((1, 0, 0), parent.basis)
    plane = Plane((0, 1, 0), parent.basis)
    assert np.isclose(plane.array @ direction.array, 0.0)

    mapped_direction = project.map_direction("do3_to_6m_reference", direction)
    mapped_plane = project.map_plane("do3_to_6m_reference", plane)

    assert np.isclose(
        mapped_plane.array @ mapped_direction.array,
        plane.array @ direction.array,
        atol=1e-12,
    )


def test_object_from_unregistered_basis_cannot_be_resolved_or_mapped():
    project = james_hane_6m_reference_project()
    foreign = CrystalBasisRef("foreign_phase", "foreign_basis")
    direction = Direction((1, 0, 0), foreign)

    try:
        project.lattice_for(direction)
    except KeyError:
        pass
    else:
        raise AssertionError("Unregistered crystallographic basis must be rejected")

    try:
        project.map_direction("do3_to_6m_reference", direction)
    except ValueError:
        pass
    else:
        raise AssertionError("Foreign basis must not be silently mapped")


def test_composition_balance_is_explicit_and_resolves_to_100_weight_percent():
    project = james_hane_6m_reference_project()
    composition = project.composition
    assert composition is not None

    assert composition.basis is CompositionBasis.WEIGHT_PERCENT
    assert np.isclose(composition.value_for("Al"), 14.0)
    assert np.isclose(composition.value_for("Ni"), 4.0)
    assert np.isclose(composition.value_for("Cu"), 82.0)

    total = sum(component.value for component in composition.resolved_components())
    assert np.isclose(total, 100.0)


def test_invalid_composition_and_nonsymmetric_cauchy_stress_are_rejected():
    try:
        Composition(
            (
                CompositionComponent("Al", 70.0),
                CompositionComponent("Ni", 40.0),
            ),
            CompositionBasis.WEIGHT_PERCENT,
            balance_element="Cu",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Composition above 100 wt% must be rejected")

    try:
        ThermomechanicalState(
            cauchy_stress_mpa=(
                (0.0, 10.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            )
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Nonsymmetric Cauchy stress must be rejected")


def test_reference_bridge_uses_registered_phase_lattices_and_passes_parity():
    project = james_hane_6m_reference_project()
    bridge = project.bridge("do3_to_6m_reference")

    audit = bridge.audit()

    audit.assert_within(project.numerical_policy.representation)


def test_validation_detects_symmetry_that_does_not_preserve_metric():
    basis = CrystalBasisRef("monoclinic_bad", "unique_b", "6M")
    lattice = Lattice.monoclinic_unique_b(4.43, 5.33, 12.79, 95.68)

    bad_swap = (
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    )

    phase = PhaseState(
        phase_id="monoclinic_bad",
        label="intentional negative test",
        physical_phase="test",
        cell_representation="6M",
        lattice=lattice,
        basis=basis,
        point_group_symbol="invalid-test",
        symmetry_operators=(bad_swap,),
    )
    project = ProjectState(
        "bad_symmetry_project",
        "negative symmetry test",
        phases=(phase,),
    )

    report = project.validate()

    assert not report.passed
    assert any(
        issue.code == "SYMMETRY_METRIC_MISMATCH"
        and issue.severity is ValidationSeverity.ERROR
        for issue in report.issues
    )


def test_validation_detects_unknown_provenance_source_key():
    basis = CrystalBasisRef("phase_x", "basis_x")
    phase = PhaseState(
        phase_id="phase_x",
        label="phase",
        physical_phase="test",
        cell_representation="",
        lattice=Lattice.cubic(3.0),
        basis=basis,
        point_group_symbol="1",
        symmetry_operators=(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),),
        provenance=StateProvenance(
            DataStatus.SOURCE_MEASURED,
            "missing_source",
        ),
    )

    report = ProjectState("p", "unknown-source test", phases=(phase,)).validate()

    assert any(issue.code == "UNKNOWN_SOURCE_KEY" for issue in report.errors)


def test_project_payload_is_json_serializable_and_preserves_exact_correspondence():
    project = james_hane_6m_reference_project()

    payload = project.to_dict()
    encoded = json.dumps(payload)

    assert payload["project_id"] == "james_hane_cualni_6m_reference"
    assert '"correspondence_M_from_A_exact"' in encoded
    assert '"weight_percent"' in encoded
    assert '"6M"' in encoded
