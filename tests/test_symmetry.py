from cualni_cryst.group_theory import validate_group
from cualni_cryst.symmetry import (
    cubic_full_m3m,
    cubic_proper_rotations,
    monoclinic_2_over_m_unique_b,
    orthorhombic_mmm,
)


def test_group_orders():
    assert len(cubic_full_m3m()) == 48
    assert len(cubic_proper_rotations()) == 24
    assert len(monoclinic_2_over_m_unique_b()) == 4
    assert len(orthorhombic_mmm()) == 8


def test_group_closure():
    validate_group(cubic_full_m3m())
    validate_group(monoclinic_2_over_m_unique_b())
