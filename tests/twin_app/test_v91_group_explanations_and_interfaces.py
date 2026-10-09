from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import math

import numpy as np
import pytest

from twin_app.interface_interpretation import (
    compare_interface_geometries,
    interface_choice_description,
)

ROOT = Path(__file__).resolve().parents[2]


def _branch(a, n, *, classification="Exact rank-one relation — classification cross-lock unresolved"):
    return SimpleNamespace(a_parent_cartesian=a, n_parent_cartesian=n,
                           classification=classification,
                           classification_status="classification cross-lock unresolved")


def test_interface_normal_angle_is_unoriented_and_unit_invariant():
    a = _branch([1.0, 0.0, 0.0], [0, 1, 0])
    b = _branch([1.0, 0.0, 0.0], [0, 0, -2])
    c = compare_interface_geometries(a, b)
    assert math.isclose(c.normal_separation_deg, 90.0, abs_tol=1e-8)
    assert "parent Cartesian" in c.narrative
    flip = _branch([-1.0, 0.0, 0.0], [0, -1, 0])
    # Joint reversal of a and n leaves the *physical* rank-one tensor unchanged.
    same = compare_interface_geometries(a, flip)
    assert same.normal_separation_deg == pytest.approx(0.0)
    assert same.relative_tensor_distance == pytest.approx(0.0)


def test_parallel_planes_cannot_be_misdescribed_as_rotated_planes():
    a = _branch([1, 0, 0], [0, 1, 0])
    b = _branch([0.5, 0, 1], [0, -1, 0])
    result = compare_interface_geometries(a, b)
    assert result.normal_separation_deg == pytest.approx(0)
    assert result.relative_tensor_distance > 0
    assert "parallel" in result.narrative


def test_interface_choices_do_not_forge_twin_type():
    branch = _branch([1, 0, 0], [0, 1, 0])
    x = SimpleNamespace(classifications=(branch.classification,), representative=branch)
    assert "unresolved" in interface_choice_description(x)
    assert "Type I" not in interface_choice_description(x)
    y = SimpleNamespace(classifications=("Type I", "Type II"), representative=branch)
    assert "disagree" in interface_choice_description(y)


def test_nonfinite_and_zero_normals_rejected():
    ok = _branch([1, 0, 0], [1, 0, 0])
    for bad in (_branch([1, 0, 0], [0, 0, 0]), _branch([1, 0, float("nan")], [1, 0, 0])):
        with pytest.raises(ValueError):
            compare_interface_geometries(ok, bad)


def test_ui_contains_explicit_two_level_distinction_and_no_solver_changes():
    src = (ROOT / "twin_app" / "tree_renderer.py").read_text()
    assert "rank-one equation" in src
    assert "not changing the material" in src
    assert "Habit-plane alternatives belong" in src
    assert "Scientific verification and full coordinates" in src
    assert "No exact A/M habit plane for this twin branch" in src
    guide = (ROOT / "twin_app" / "point_group_guide.py").read_text()
    assert "Lattice parameters alone" not in guide or "not" in guide
    assert "parent" in guide and "product" in guide


@pytest.mark.skipif(not (ROOT / "src" / "cualni_cryst").exists(), reason="requires full repository backend")
def test_cubic_and_monoclinic_group_content_is_computed_not_invented():
    from twin_app.point_group_guide import group_guide
    from twin_app.symmetry_inventory import build_symmetry_inventory
    cubic = build_symmetry_inventory("432", np.eye(3))
    result = group_guide(cubic, parent=True)
    assert result.order == 24 and result.proper == 24 and result.improper == 0
    fourfold = next(f for f in result.families if f.name == "Proper 4-fold rotation")
    assert fourfold.operations == 6 and len(fourfold.geometric_elements) == 3
    assert fourfold.coordinate_type == "[uvw] rotation axis"
    monoclinic = build_symmetry_inventory("2/m", np.diag([1., 2., 3.]))
    result2 = group_guide(monoclinic, parent=False)
    assert result2.order == 4 and result2.inversion
    mirror = next(f for f in result2.families if f.name == "Mirror reflections")
    assert mirror.coordinate_type == "(hkl) plane covector"
    assert mirror.operations == 1 and len(mirror.geometric_elements) == 1
