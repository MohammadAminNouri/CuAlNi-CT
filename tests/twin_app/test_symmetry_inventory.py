from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "src" / "cualni_cryst").exists():
    pytest.skip("repository backend is not present in this extracted package", allow_module_level=True)

from cualni_cryst.lattice import Lattice
from cualni_cryst.point_groups import point_group_definitions

from twin_app.symmetry_inventory import build_symmetry_inventory


# Independent crystallographic contract from the canonical 32 crystal classes.
# Do not derive these orders or system memberships from the production registry;
# otherwise a wrong registry entry could validate itself.
EXPECTED_32 = {
    "1": ("triclinic", 1),
    "-1": ("triclinic", 2),
    "2": ("monoclinic", 2),
    "m": ("monoclinic", 2),
    "2/m": ("monoclinic", 4),
    "222": ("orthorhombic", 4),
    "mm2": ("orthorhombic", 4),
    "mmm": ("orthorhombic", 8),
    "4": ("tetragonal", 4),
    "-4": ("tetragonal", 4),
    "4/m": ("tetragonal", 8),
    "422": ("tetragonal", 8),
    "4mm": ("tetragonal", 8),
    "-42m": ("tetragonal", 8),
    "4/mmm": ("tetragonal", 16),
    "3": ("trigonal", 3),
    "-3": ("trigonal", 6),
    "32": ("trigonal", 6),
    "3m": ("trigonal", 6),
    "-3m": ("trigonal", 12),
    "6": ("hexagonal", 6),
    "-6": ("hexagonal", 6),
    "6/m": ("hexagonal", 12),
    "622": ("hexagonal", 12),
    "6mm": ("hexagonal", 12),
    "-6m2": ("hexagonal", 12),
    "6/mmm": ("hexagonal", 24),
    "23": ("cubic", 12),
    "m-3": ("cubic", 24),
    "432": ("cubic", 24),
    "-43m": ("cubic", 24),
    "m-3m": ("cubic", 48),
}


def _lattice(family: str) -> Lattice:
    if family == "cubic":
        return Lattice(3.1, 3.1, 3.1, 90, 90, 90)
    if family == "tetragonal":
        return Lattice(3.1, 3.1, 4.2, 90, 90, 90)
    if family == "orthorhombic":
        return Lattice(3.1, 3.4, 4.2, 90, 90, 90)
    if family == "hexagonal":
        return Lattice(3.1, 3.1, 5.0, 90, 90, 120)
    if family == "trigonal":
        return Lattice(3.1, 3.1, 5.0, 90, 90, 120)
    if family == "monoclinic":
        return Lattice(3.1, 3.4, 4.2, 90, 101, 90)
    return Lattice(3.1, 3.4, 4.2, 79, 83, 74)


def test_builtin_registry_matches_independent_32_point_group_contract():
    definitions = point_group_definitions()
    observed = {
        definition.symbol: (definition.crystal_family, definition.expected_order)
        for definition in definitions
    }
    assert observed == EXPECTED_32


def test_all_32_builtin_point_groups_have_readable_metric_validated_inventories():
    definitions = point_group_definitions()
    assert len(definitions) == 32
    for definition in definitions:
        expected_family, expected_order = EXPECTED_32[definition.symbol]
        inventory = build_symmetry_inventory(
            definition.symbol,
            _lattice(expected_family).metric(),
        )
        assert inventory.crystal_family == expected_family
        assert inventory.expected_order == expected_order
        assert len(inventory.operations) == expected_order
        assert sum(count for _, count in inventory.counts) == expected_order
        assert inventory.metric_preservation_maximum_residual < 1.0e-9


def test_symmetry_display_keeps_rotation_axes_and_mirror_covectors_distinct():
    source = (ROOT / "twin_app" / "input_ui.py").read_text(encoding="utf-8")
    assert "rotation axis [uvw]" in source
    assert "mirror-plane covector (hkl)" in source
    assert "These are not the same kind of object" in source
