from pathlib import Path
import numpy as np
import pytest

from cualni_cryst.legacy_suite_bridge import (
    Phase, crystal_geometry, exact_correspondence, variant_groupoid, inspect_ebsd
)


def test_geometry_cubic_and_reciprocal():
    phase = Phase("cubic", "m-3m", 2, 2, 2)
    result = crystal_geometry(phase, (1, 0, 0), (1, 0, 0))
    assert result["direction_length_angstrom"] == pytest.approx(2)
    assert result["d_spacing_angstrom"] == pytest.approx(2)
    assert result["point_group_order"] == 48


def test_incompatible_cell_symmetry_is_rejected():
    with pytest.raises(ValueError, match="not compatible"):
        crystal_geometry(Phase("invalid", "m-3m", 2, 3, 4), (1, 0, 0), (1, 0, 0))


def test_correspondence_rejects_code_and_singularity():
    for text in ("1 0 0\n0 eval(1) 0\n0 0 1", "1 0 0\n0 0 0\n0 0 1"):
        with pytest.raises((ValueError, TypeError)):
            exact_correspondence(text)


def test_exact_symmetry_groupoid_invariants():
    parent = Phase("parent", "m-3m", 2, 2, 2)
    product = Phase("product", "mmm", 2, 3, 4)
    output = variant_groupoid(parent, product, "1 0 0\n0 1 0\n0 0 1")
    assert output["variants"] == 6
    assert output["variants"] * output["subgroup_order"] == 48
    assert output["operator_classes"] == output["burnside_count"]
    assert sum(output["class_sizes"]) == 48
    assert len(output["adjacency"]) == 6


def test_no_unverified_ebsd_formats(tmp_path: Path):
    path = tmp_path / "sample.pickle"
    path.write_bytes(b"unsafe")
    with pytest.raises(ValueError, match="Expected"):
        inspect_ebsd(path)
