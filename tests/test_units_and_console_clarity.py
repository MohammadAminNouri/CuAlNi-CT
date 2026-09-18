import numpy as np
import pytest

from cualni_cryst.crystallography_console import (
    ConsoleRenderer,
    CrystallographyConsole,
)
from cualni_cryst.cualni_models import james_hane_6m_example_lattices
from cualni_cryst.lattice import Lattice
from cualni_cryst.project_state import james_hane_6m_reference_project
from cualni_cryst.units import (
    length_unit_symbol,
    normalize_length_unit,
    reciprocal_length_unit_symbol,
)


def test_unit_aliases_are_explicit_and_do_not_rescale_values():
    assert normalize_length_unit("Angstrom") == "angstrom"
    assert normalize_length_unit("Å") == "angstrom"
    assert length_unit_symbol("angstrom") == "Å"
    assert reciprocal_length_unit_symbol("angstrom") == "Å⁻¹"

    lattice = Lattice.cubic(5.836, length_unit="angstrom")
    assert lattice.a == 5.836
    assert lattice.length_unit == "angstrom"


def test_unknown_units_are_rejected_instead_of_silently_labeled():
    with pytest.raises(ValueError, match="Unsupported lattice-length unit"):
        Lattice.cubic(3.0, length_unit="furlong")


def test_james_hane_reference_lattices_are_explicitly_angstrom():
    parent, product = james_hane_6m_example_lattices()
    assert parent.length_unit == "angstrom"
    assert product.length_unit == "angstrom"


def test_project_state_serializes_lattice_unit():
    project = james_hane_6m_reference_project()
    payload = project.to_dict()
    units = {
        phase["phase_id"]: phase["lattice"]["length_unit"]
        for phase in payload["phases"]
    }
    assert units["austenite_do3"] == "angstrom"
    assert units["martensite_long_period"] == "angstrom"


def test_console_reports_angstrom_and_inverse_angstrom():
    console = CrystallographyConsole(james_hane_6m_reference_project())

    direction = console.inspect("6m", "[1 0 1]")
    plane = console.inspect("6m", "(0 1 1)")

    assert direction.dimensional_unit_label == "Å"
    assert plane.dimensional_unit_label == "Å⁻¹"
    assert plane.secondary_unit_label == "Å"
    assert not direction.warnings
    assert not plane.warnings


def test_renderer_uses_precise_metric_and_ptclab_labels():
    console = CrystallographyConsole(james_hane_6m_reference_project())
    report = console.inspect("6m", "[1 0 1]")
    text = ConsoleRenderer().object_report(report, show_derivation=True)

    assert "metric-unit direct coeffs" in text
    assert "PTCLab Cartesian (x||a, c in xz)" in text
    assert "Å" in text
    assert "lattice-length units" not in text


def test_plane_renderer_says_reciprocal_coeffs():
    console = CrystallographyConsole(james_hane_6m_reference_project())
    report = console.inspect("6m", "(0 1 1)")
    text = ConsoleRenderer().object_report(report)

    assert "metric-unit reciprocal coeffs" in text
    assert "Å⁻¹" in text


def test_reference_numeric_result_is_unchanged_by_unit_metadata():
    console = CrystallographyConsole(james_hane_6m_reference_project())
    report = console.inspect("6m", "[1 0 1]")

    assert np.isclose(report.dimensional_quantity, 13.11462948721663, atol=1e-12)
    assert report.maximum_representation_residual < 1e-12
