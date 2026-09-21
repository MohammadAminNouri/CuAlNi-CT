from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cualni_cryst.ebsd import bunge_euler_to_g
from cualni_cryst.ebsd_analysis import (
    build_neighbor_graph,
    grain_statistics,
    kernel_average_misorientation,
    minimum_disorientation,
    segment_grains,
    symmetry_aligned_mean,
)
from cualni_cryst.ebsd_io import (
    DelimitedSchema,
    load_ang,
    load_ctf,
    load_delimited,
)
from cualni_cryst.ebsd_map import (
    AngleUnit,
    EBSDMap,
    EBSDPhase,
    MatrixDirection,
    OrientationConvention,
    assert_bunge_adapter_parity,
    audit_map,
    convert_raw_matrices,
    eulers_to_matrices,
    quaternions_to_matrices,
)
from cualni_cryst.ebsd_reconstruction import (
    VariantOR,
    assign_variant,
    build_operator_library,
    classify_boundary_operator,
    predict_product_orientation,
    reconstruct_parent,
)
from cualni_cryst.lattice import Lattice
from cualni_cryst.orientation_kernel import rotation_angle_deg
from cualni_cryst.point_groups import point_group_definitions


def _rx(angle):
    return Rotation.from_euler("X", angle, degrees=True).as_matrix()


def _rz(angle):
    return Rotation.from_euler("Z", angle, degrees=True).as_matrix()


def test_bunge_vectorized_converter_crosslocks_legacy_adapter():
    assert_bunge_adapter_parity()
    angles = np.array([[13.0, 27.0, 41.0], [250.0, 89.5, 17.0]])
    conv = OrientationConvention.bunge_crystal_to_sample_degrees()
    matrices = eulers_to_matrices(angles, conv)
    for index, row in enumerate(angles):
        expected = bunge_euler_to_g(*row)
        assert rotation_angle_deg(matrices[index] @ expected.T) < 1.0e-9


def test_matrix_direction_and_quaternion_order_are_explicit():
    R = _rz(37.0) @ _rx(11.0)
    passive = OrientationConvention(
        raw_matrix_direction=MatrixDirection.SAMPLE_TO_CRYSTAL,
        label="explicit passive test",
    )
    converted = convert_raw_matrices(R.T, passive)[0]
    assert rotation_angle_deg(converted @ R.T) < 1.0e-10

    quat_xyzw = Rotation.from_matrix(R).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    active = OrientationConvention.bunge_crystal_to_sample_degrees()
    rebuilt = quaternions_to_matrices(
        quat_wxyz, active, order="wxyz"
    )[0]
    assert rotation_angle_deg(rebuilt @ R.T) < 1.0e-10


def test_ang_parser_preserves_standard_columns_and_radians(tmp_path):
    text = """# GRID: SquareGrid
# XSTEP: 1.0
# YSTEP: 1.0
# NCOLS_ODD: 2
# NCOLS_EVEN: 2
# NROWS: 1
0 0 0 0 0 100 0.9 1 55 0.1
1.5707963267948966 0 0 1 0 90 0.8 1 50 0.2
"""
    path = tmp_path / "map.ang"
    path.write_text(text)
    data = load_ang(path)
    assert data.n_points == 2
    assert np.all(data.phase_id == 1)
    assert data.quality["IQ"].tolist() == [100.0, 90.0]
    assert rotation_angle_deg(data.orientations[1]) == pytest.approx(
        90.0, abs=1.0e-9
    )
    assert data.metadata["reference_frame_normalization_applied"] is False


def test_ctf_parser_reads_2d_degrees_and_refuses_ambiguous_3d(tmp_path):
    text2d = """Channel Text File
Prj test
JobMode Grid
XCells 2
YCells 1
XStep 1
YStep 1
Phases 1
Phase X Y Bands Error Euler1 Euler2 Euler3 MAD BC BS
1 0 0 8 0 0 0 0 0.2 100 200
1 1 0 8 0 90 0 0 0.3 90 180
"""
    path = tmp_path / "map.ctf"
    path.write_text(text2d)
    data = load_ctf(path)
    assert rotation_angle_deg(data.orientations[1]) == pytest.approx(
        90.0, abs=1.0e-9
    )
    assert "MAD" in data.quality

    text3d = text2d.replace("YCells 1", "YCells 1\nZCells 2")
    path3d = tmp_path / "map3d.ctf"
    path3d.write_text(text3d)
    with pytest.raises(ValueError, match="angle unit is ambiguous"):
        load_ctf(path3d)
    loaded = load_ctf(
        path3d,
        three_dimensional_angle_unit=AngleUnit.DEGREE,
    )
    assert loaded.n_points == 2


def test_generic_delimited_parser_handles_euler_and_explicit_schema(tmp_path):
    path = tmp_path / "generic.csv"
    path.write_text(
        "phase,x,y,p1,P,p2,quality\n"
        "1,0,0,0,0,0,0.9\n"
        "1,1,0,30,0,0,0.8\n"
    )
    schema = DelimitedSchema(
        phase="phase",
        x="x",
        y="y",
        euler=("p1", "P", "p2"),
        quality={"score": "quality"},
    )
    data = load_delimited(
        path,
        schema,
        convention=OrientationConvention.bunge_crystal_to_sample_degrees(),
    )
    assert data.n_points == 2
    assert data.quality["score"].tolist() == [0.9, 0.8]


def test_all_32_point_groups_can_define_ebsd_phases():
    definitions = point_group_definitions()
    assert len(definitions) == 32
    family_lattice = {
        "triclinic": Lattice(3.0, 4.0, 5.0, 70, 80, 75),
        "monoclinic": Lattice.monoclinic_unique_b(3.0, 4.0, 5.0, 105),
        "orthorhombic": Lattice.orthorhombic(3.0, 4.0, 5.0),
        "tetragonal": Lattice(3.0, 3.0, 5.0),
        "trigonal": Lattice(3.0, 3.0, 5.0, 90, 90, 120),
        "hexagonal": Lattice(3.0, 3.0, 5.0, 90, 90, 120),
        "cubic": Lattice.cubic(3.0),
    }
    for index, definition in enumerate(definitions, start=1):
        phase = EBSDPhase.from_point_group(
            index,
            definition.symbol,
            family_lattice[definition.crystal_family],
            definition.symbol,
        )
        assert phase.proper_symmetry_cartesian


def test_map_audit_and_phase_aware_grain_segmentation():
    phase = EBSDPhase.from_point_group(
        1, "cubic", Lattice.cubic(3.0), "m-3m"
    )
    orientations = np.array(
        [
            np.eye(3),
            _rz(1.0),
            _rz(20.0),
            _rz(21.0),
        ]
    )
    data = EBSDMap(
        orientations=orientations,
        phase_id=np.ones(4, dtype=int),
        indexed=np.ones(4, dtype=bool),
        x=np.array([0.0, 1.0, 2.0, 3.0]),
        y=np.zeros(4),
        z=np.zeros(4),
    )
    audit = audit_map(data)
    assert audit.n_indexed == 4
    graph = build_neighbor_graph(data, radius=1.01)
    segmentation = segment_grains(
        data,
        {1: phase},
        threshold_deg=5.0,
        neighbor_graph=graph,
    )
    assert segmentation.n_grains == 2
    assert segmentation.grain_id.tolist() == [0, 0, 1, 1]

    grains = grain_statistics(data, segmentation, {1: phase})
    assert [grain.size for grain in grains] == [2, 2]
    assert max(grain.gos_deg for grain in grains) < 1.0

    kam = kernel_average_misorientation(
        data,
        {1: phase},
        neighbor_graph=graph,
        maximum_neighbor_misorientation_deg=5.0,
    )
    assert np.isfinite(kam[[0, 1, 2, 3]]).all()


def test_cubic_symmetry_equivalent_orientations_have_zero_disorientation():
    phase = EBSDPhase.from_point_group(
        1, "cubic", Lattice.cubic(3.0), "m-3m"
    )
    g = _rz(17.0) @ _rx(23.0)
    equivalent = g @ phase.proper_symmetry_cartesian[-1]
    result = minimum_disorientation(
        g,
        equivalent,
        phase.proper_symmetry_cartesian,
    )
    assert result.angle_deg < 1.0e-7


def test_symmetry_aligned_mean_does_not_average_different_symmetry_copies():
    phase = EBSDPhase.from_point_group(
        1, "cubic", Lattice.cubic(3.0), "m-3m"
    )
    base = _rz(11.0) @ _rx(7.0)
    values = np.array(
        [
            base,
            base @ phase.proper_symmetry_cartesian[3],
            _rz(0.2) @ base,
        ]
    )
    mean = symmetry_aligned_mean(
        values,
        phase.proper_symmetry_cartesian,
    )
    assert minimum_disorientation(
        mean,
        base,
        phase.proper_symmetry_cartesian,
    ).angle_deg < 0.2


def test_parent_reconstruction_recovers_parent_and_variants_with_noise():
    parent_phase = EBSDPhase.from_point_group(
        1, "parent", Lattice.cubic(3.0), "m-3m"
    )
    product_phase = EBSDPhase.from_point_group(
        2, "product", Lattice.orthorhombic(3.0, 4.0, 5.0), "mmm"
    )

    variants = (
        VariantOR(0, np.eye(3), "v0"),
        VariantOR(1, _rz(90.0), "v1"),
        VariantOR(2, _rx(180.0), "v2"),
    )
    true_parent = _rz(17.0) @ _rx(8.0)
    sequence = [0, 1, 2, 1, 0, 2]
    measured = []
    for index, variant_index in enumerate(sequence):
        exact = predict_product_orientation(
            true_parent, variants[variant_index]
        )
        noise = Rotation.from_rotvec(
            np.deg2rad(0.15 + 0.02 * index)
            * np.array([1.0, 2.0, -1.0]) / np.sqrt(6.0)
        ).as_matrix()
        measured.append(noise @ exact)
    measured = np.asarray(measured)

    reconstruction = reconstruct_parent(
        measured,
        variants,
        product_phase.proper_symmetry_cartesian,
        parent_phase.proper_symmetry_cartesian,
        inlier_tolerance_deg=1.0,
        variant_acceptance_deg=1.0,
    )
    parent_error = minimum_disorientation(
        reconstruction.parent_orientation,
        true_parent,
        parent_phase.proper_symmetry_cartesian,
    ).angle_deg
    assert parent_error < 0.3
    assert reconstruction.weighted_inlier_fraction == pytest.approx(1.0)
    assert all(item.accepted for item in reconstruction.variant_assignments)


def test_variant_operator_library_and_boundary_classification_are_matrix_based():
    product_phase = EBSDPhase.from_point_group(
        2, "product", Lattice.orthorhombic(3.0, 4.0, 5.0), "mmm"
    )
    variants = (
        VariantOR(0, np.eye(3)),
        VariantOR(1, _rz(45.0)),
        VariantOR(2, _rz(90.0)),
    )
    library = build_operator_library(
        variants,
        product_phase.proper_symmetry_cartesian,
    )
    assert library

    parent = _rx(13.0)
    g0 = predict_product_orientation(parent, variants[0])
    g1 = predict_product_orientation(parent, variants[1])
    assignment = classify_boundary_operator(
        g0,
        g1,
        library,
        product_phase.proper_symmetry_cartesian,
        maximum_residual_deg=0.1,
    )
    assert assignment.accepted
    assert assignment.best_residual_deg < 1.0e-7


def test_invalid_orientation_is_not_silently_repaired():
    bad = np.diag([1.0, 1.0, 1.1])
    convention = OrientationConvention.bunge_crystal_to_sample_degrees()
    with pytest.raises(ValueError, match="not in SO"):
        convert_raw_matrices(bad, convention)


def test_unindexed_points_may_store_nan_but_indexed_points_may_not():
    matrices = np.full((2, 3, 3), np.nan)
    matrices[0] = np.eye(3)
    data = EBSDMap(
        orientations=matrices,
        phase_id=np.array([1, 0]),
        indexed=np.array([True, False]),
        x=np.array([0.0, 1.0]),
        y=np.zeros(2),
        z=np.zeros(2),
    )
    assert data.n_points == 2

    with pytest.raises(ValueError):
        EBSDMap(
            orientations=matrices,
            phase_id=np.array([1, 1]),
            indexed=np.array([True, True]),
            x=np.array([0.0, 1.0]),
            y=np.zeros(2),
            z=np.zeros(2),
        )
