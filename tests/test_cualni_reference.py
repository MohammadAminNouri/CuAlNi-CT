import pytest

from cualni_cryst.cualni_reference import (
    CHEN_2000_EBSD_2H,
    CUALNI_LITERATURE_STATES,
    IBARRA_2006_BETA3,
    IBARRA_2006_GAMMA3,
    JAMES_HANE_2000_6M,
    LANDA_2007_2H,
    OTSUKA_1974_STRESS_18R,
    available_cualni_literature_states,
)


def test_cualni_registry_is_multiple_source_specific_states_not_one_default():
    keys = set(available_cualni_literature_states())
    assert keys == set(CUALNI_LITERATURE_STATES)
    assert len(keys) >= 6
    assert "james_hane_2000_6m" in keys
    assert "landa_2007_2h" in keys
    assert "ibarra_2006_beta3" in keys
    assert "ibarra_2006_gamma3" in keys


def test_complete_cells_construct_metrics_without_inference():
    assert JAMES_HANE_2000_6M.parent_cell.to_lattice().metric().shape == (3, 3)
    assert LANDA_2007_2H.martensite_cell.to_lattice().metric().shape == (3, 3)
    assert IBARRA_2006_BETA3.martensite_cell.to_lattice().metric().shape == (3, 3)
    assert IBARRA_2006_GAMMA3.martensite_cell.to_lattice().metric().shape == (3, 3)


def test_incomplete_1974_18r_cell_fails_loudly_instead_of_inventing_beta():
    assert not OTSUKA_1974_STRESS_18R.martensite_cell.complete_metric
    with pytest.raises(ValueError):
        OTSUKA_1974_STRESS_18R.martensite_cell.to_lattice()


def test_ebsd_only_state_does_not_invent_lattice_parameters():
    assert CHEN_2000_EBSD_2H.parent_cell is None
    assert CHEN_2000_EBSD_2H.martensite_cell is None
    assert CHEN_2000_EBSD_2H.orientation_anchors
    assert CHEN_2000_EBSD_2H.twin_anchors


def test_every_reference_state_has_registered_source_key():
    from cualni_cryst.sources import SOURCES

    for state in CUALNI_LITERATURE_STATES.values():
        assert state.source_key in SOURCES
