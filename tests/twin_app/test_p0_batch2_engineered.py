"""New P0 Batch 2 behavioral contract tests. No benchmark values in production."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as R

import numpy as np
import pytest

from cualni_cryst.lattice import Lattice
from cualni_cryst.point_groups import point_group_definitions
from twin_app.point_group_visualizer import cell_basis, make_operation_scene
from twin_app.operation_witness import operation_witness
from twin_app.report_integrity import ReportIntegrityError, validate_report_integrity
from twin_app.symmetry_inventory import build_symmetry_inventory

ROOT = Path(__file__).resolve().parents[2]


def _lattice(family):
    if family == "cubic": return (3.1, 3.1, 3.1, 90., 90., 90.)
    if family == "tetragonal": return (3.1, 3.1, 5.3, 90., 90., 90.)
    if family == "orthorhombic": return (3.1, 4.2, 5.3, 90., 90., 90.)
    if family in ("trigonal", "hexagonal"): return (3.1, 3.1, 5.3, 90., 90., 120.)
    if family == "monoclinic": return (3.1, 4.2, 5.3, 90., 104., 90.)
    return (3.1, 4.2, 5.3, 75., 83., 67.)


def test_all_32_point_group_ops_have_a_metric_certified_actual_action():
    definitions = point_group_definitions()
    assert len(definitions) == 32
    seen = set()
    for definition in definitions:
        symbol = definition.symbol
        a, b, c, alpha, beta, gamma = _lattice(definition.crystal_family)
        lattice = Lattice(a, b, c, alpha, beta, gamma)
        inv = build_symmetry_inventory(symbol, lattice.metric())
        B = cell_basis(a, b, c, alpha, beta, gamma)
        assert len(inv.operations) == definition.expected_order
        for operation in inv.operations:
            w = operation_witness(B, operation.matrix)
            assert w.determinant in (-1, 1)
            assert w.relative_metric_residual < 1e-10, (symbol, operation)
            source = np.asarray(w.before_cartesian)
            mapped = np.asarray(w.after_cartesian)
            assert np.isclose(np.linalg.norm(source), np.linalg.norm(mapped), rtol=1e-10)
        seen.add(symbol)
    assert len(seen) == 32


def test_witness_uses_reciprocal_safe_nonorthogonal_embedding():
    B = cell_basis(3.1, 4.2, 5.3, 90., 104., 90.)
    mirror = (("1", "0", "0"), ("0", "-1", "0"), ("0", "0", "1"))
    w = operation_witness(B, mirror)
    assert w.after_fractional[1] == pytest.approx(-w.before_fractional[1])
    assert w.relative_metric_residual < 1e-12


def test_improper_and_inversion_and_identity_have_action_traces():
    B = cell_basis(3.1, 3.1, 3.1, 90, 90, 90)
    for G, kind in (
        ((("1","0","0"),("0","1","0"),("0","0","1")), "identity"),
        ((("-1","0","0"),("0","-1","0"),("0","0","-1")), "inversion"),
        ((("0","-1","0"),("1","0","0"),("0","0","-1")), "improper order-4 operation"),
    ):
        fig = make_operation_scene(B, kind, None, operation_matrix=G)
        names = [getattr(t, "name", "") for t in fig.data]
        assert "Before symmetry operation" in names
        assert "After symmetry operation" in names


def test_zero_and_nonisometric_symmetry_actions_are_rejected():
    B = np.eye(3)
    with pytest.raises(ValueError, match="determinant"):
        operation_witness(B, (("0","0","0"),("0","1","0"),("0","0","1")))
    with pytest.raises(ValueError, match="preserve"):
        operation_witness(B, (("1","1","0"),("0","1","0"),("0","0","1")))


def _report():
    hab = R(
        habit_branch=1, other_variant_volume_fraction=.3, base_variant_volume_fraction=.7,
        habit_normal_parent_cartesian=(1.,0.,0.),
        habit_plane_parent_crystal=(1.,0.,0.), shape_vector_parent_cartesian=(0.,.2,0.),
        shape_vector_parent_crystal=(0.,.2,0.),
        frame_plane_residual=0., frame_shape_vector_residual=0.,
        rank_one_residual=1e-10, rotation_residual=1e-10, middle_stretch_residual=1e-10,
    )
    branch = R(
        construction_id="P1.B1", shear_magnitude=.25,
        a_parent_cartesian=(.25,0.,0.), n_parent_cartesian=(0.,1.,0.),
        twin_plane_product_crystal=(0.,1.,0.), shear_direction_product_crystal=(1.,0.,0.),
        rank_one_residual=1e-10, rotation_residual=1e-10,
        independent_outer_product_residual=1e-9, independent_shear_relative_residual=1e-8,
        discrete_plane_angle_deg=None, discrete_direction_angle_deg=None,
        discrete_shear_relative_residual=None, classification="Type II",
        classification_status="cross-locked: exact parent twofold provenance + shear",
        continuum_fraction=False, habit_solutions=(hab,),
    )
    pair = R(variant_i=1, variant_j=2, constructions=(branch,))
    fam = R(family_id="F1", equivalent_pairs=((1,2),),pair_records=(pair,))
    audit = R(topology_variant_count=2, stretch_variant_count=2,
              maximum_correspondence_to_stretch_residual=1e-12,
              ptmc_maximum_residual=1e-12,ball_james_maximum_residual=1e-12)
    return R(audit=audit,correspondence_variants=(
        R(variant_index=1, stretch_variant_index=1, stretch_mapping_residual=0.),
        R(variant_index=2, stretch_variant_index=2, stretch_mapping_residual=0.),
    ), families=(fam,))


def test_valid_synthetic_report_structure_is_admissible():
    validate_report_integrity(_report())


@pytest.mark.parametrize("corruption", [
    "duplicate_pair", "missing_pair", "invalid_variant_index", "fake_compound",
    "bad_lambda", "zero_K1", "wrong_unit_habit_m", "nan_shear", "nonfinite_residual",
])
def test_corrupt_or_overclaimed_report_fails_closed(corruption):
    report=deepcopy(_report())
    family=report.families[0]
    branch=family.pair_records[0].constructions[0]
    if corruption == "duplicate_pair":
        report.families=(family, deepcopy(family))
        report.families[1].family_id="F2"
    elif corruption == "missing_pair":
        family.pair_records=()
    elif corruption == "invalid_variant_index":
        family.equivalent_pairs=((1,3),)
    elif corruption == "fake_compound":
        branch.classification="Compound"
        branch.classification_status="operator family exists; unverified"
    elif corruption == "bad_lambda":
        branch.habit_solutions[0].other_variant_volume_fraction=.9
    elif corruption == "zero_K1":
        branch.twin_plane_product_crystal=(0.,0.,0.)
    elif corruption == "wrong_unit_habit_m":
        branch.habit_solutions[0].habit_normal_parent_cartesian=(2.,0.,0.)
    elif corruption == "nan_shear":
        branch.shear_magnitude=float("nan")
    elif corruption == "nonfinite_residual":
        report.audit.ptmc_maximum_residual=float("inf")
    with pytest.raises(ReportIntegrityError):
        validate_report_integrity(report)


def test_integrity_gate_is_before_any_report_display_or_export():
    app=(ROOT / "twin_app/streamlit_app.py").read_text(encoding="utf-8")
    assert "validate_report_integrity(report)" in app
    assert app.index("validate_report_integrity(report)") < app.index('st.session_state["twin_family_report"] = report')


def test_programmatic_selection_resets_only_custom_navigator_instance():
    code=(ROOT / "twin_app/tree_renderer.py").read_text(encoding="utf-8")
    assert 'tf_nav_epoch_p0b2' in code
    assert 'epoch_key = "tf_nav_epoch_p0b2"' in code
    assert code.count("st.session_state[epoch_key] += 1") == 2
    assert 'key=f"tf_browser_family_tree_v8_epoch_' in code
    html=(ROOT / "twin_app/navigator_frontend/index.html").read_text(encoding="utf-8")
    assert "streamlit:setComponentValue" in html
