from __future__ import annotations

"""Cross-tool export and parent-daughter roundtrip science tests.

The examples are wholly synthetic; private CuAlNi EBSD and fitted Q must not
be copied into this public repository's test fixtures.
"""

from io import BytesIO
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from zipfile import ZipFile

from cualni_cryst.lattice import Lattice
from cualni_cryst.ebsd_map import EBSDPhase
from cualni_cryst.parent_replay_exchange import (
    forward_predict_daughter, forward_inverse_roundtrip, export_replay_zip
)
from cualni_cryst.parent_or_interop import proper_cubic_parent_operations


def example():
    beta = proper_cubic_parent_operations()
    mart = EBSDPhase.from_point_group(
        1, "reduced 18R", Lattice(10, 5, 4, 90, 108, 90), "2/m"
    ).proper_symmetry_cartesian
    parent = Rotation.from_euler("xyz", [20, -18, 40], degrees=True).as_matrix()
    Q = Rotation.from_euler("xyz", [13, 27, -9], degrees=True).as_matrix()
    return parent, Q, tuple(beta), tuple(mart)


def test_complete_parent_daughter_parent_roundtrip_with_monoclinic_indexer_copies():
    gA, Q, beta, mart = example()
    predictions = forward_predict_daughter(gA, Q, beta, mart)
    assert predictions["n_variants"] == 24
    assert predictions["n_daughter_symmetry_representatives"] == 2
    obs = predictions["daughter_symmetry_representatives"].reshape(-1, 3, 3)
    report = forward_inverse_roundtrip(gA, Q, beta, mart, observed_daughter_g=obs)
    assert report["n_observations"] == 48
    assert report["weighted_fraction_within_threshold"] == pytest.approx(1.0)
    assert report["median_forward_residual_deg"] <= 1e-4
    assert report["max_forward_inverse_disagreement_deg"] <= 1e-4
    assert all(x>0 for x in report["variant_id_1based"])
    assert report["scientific_status"].startswith("conditional")


def test_forward_match_deteriorates_with_independent_15degree_wrong_or():
    gA, Q, beta, mart = example()
    daughters=forward_predict_daughter(gA, Q, beta, mart)[
        "daughter_symmetry_representatives"
    ].reshape(-1, 3, 3)
    wrong = Q @ Rotation.from_rotvec(np.deg2rad(15)*np.array([.31,.47,.826438])).as_matrix()
    correct = forward_inverse_roundtrip(gA,Q,beta,mart,observed_daughter_g=daughters)
    incorrect = forward_inverse_roundtrip(gA,wrong,beta,mart,observed_daughter_g=daughters)
    assert incorrect["median_forward_residual_deg"] > correct["median_forward_residual_deg"] + 3
    assert incorrect["weighted_fraction_within_threshold"] < 0.5


def test_roundtrip_weighted_area_fraction_and_no_generated_parent_certification():
    gA, Q, beta, mart = example()
    daughters=forward_predict_daughter(gA,Q,beta,mart)["daughter_crystal_to_sample"]
    report = forward_inverse_roundtrip(gA,Q,beta,mart,
                                       observed_daughter_g=daughters[:3],
                                       weights=[1,50,100])
    assert report["weighted_fraction_within_threshold"] == pytest.approx(1)
    assert report["physical_or_certified"] is False
    assert report["absolute_parent_frame_certified"] is False


def test_portable_zip_is_self_describing_and_mtex_direction_is_transpose():
    gA,Q,beta,mart=example()
    blob=export_replay_zip(
        gA,Q,beta,mart,
        h5oina_filename="single-crystal-site-6.h5oina",
        h5oina_sha256="a"*64,
        parent_cell={"symmetry":"m-3m","a_angstrom":None},
        daughter_cell={"symmetry":"2/m","cell_a_angstrom":10},
    )
    with ZipFile(BytesIO(blob)) as z:
        assert {"or_contract.json","forward_inverse_results.json",
                "predicted_18r_variants_xyzw_bunge.csv",
                "conditional_beta_orientation_xyzw_bunge.csv",
                "run_external_mtex_replay.m","README.txt"} <= set(z.namelist())
        manifest=json.loads(z.read("or_contract.json"))
        assert manifest["raw_ebsd_embedded"] is False
        assert np.allclose(
            np.asarray(manifest["or"]["mtex_p2c_daughter_from_parent"]),Q.T
        )
        assert manifest["daughter_variant_count"]==24
        mtex=z.read("run_external_mtex_replay.m").decode()
        for operation in ("calcGrains","parentGrainReconstructor",
                          "calcVariantGraph","clusterVariantGraph",
                          "calcParentFromVote","mergeSimilar","calcParentEBSD",
                          "calcParent2Child"):
            assert operation in mtex
        assert "NOT a verified AZtec importer" in z.read("README.txt").decode()


def test_incorrect_frame_and_untrusted_export_paths_are_rejected():
    gA,Q,beta,mart=example()
    with pytest.raises(ValueError):
        forward_inverse_roundtrip(gA,np.diag([1,1,-1]),beta,mart)
    with pytest.raises(ValueError):
        export_replay_zip(gA,Q,beta,mart,h5oina_filename="scan.exe")
    with pytest.raises(ValueError):
        export_replay_zip(gA,Q,beta,mart,h5oina_filename="scan.h5oina",
                          h5oina_sha256="garbage")
    with pytest.raises(ValueError):
        forward_inverse_roundtrip(gA,Q,beta,mart,weights=[1,2])


def test_symmetry_equivalent_parent_relabeling_keeps_daughter_fit():
    gA,Q,beta,mart=example()
    target=forward_predict_daughter(gA,Q,beta,mart)["daughter_crystal_to_sample"]
    rotated_parent=gA@beta[7]
    report=forward_inverse_roundtrip(rotated_parent,Q,beta,mart,
                                     observed_daughter_g=target)
    assert report["median_forward_residual_deg"] <= 1e-4


def test_original_cif_content_passed_through_without_structural_fabrication():
    gA,Q,beta,mart=example()
    cif=b"data_original_18r\\n_cell_length_a 10.0\\n_cell_length_b 5.0\\n"
    blob=export_replay_zip(
        gA,Q,beta,mart,h5oina_filename="scan.h5oina",
        indexed_daughter_cif=cif,
    )
    with ZipFile(BytesIO(blob)) as z:
        assert z.read("USER_SUPPLIED_indexed_18R_structure.cif") == cif
        contract=json.loads(z.read("or_contract.json"))
        assert contract["indexed_daughter_cif_supplied"] is True
        assert contract["parent_cif_supplied"] is False
    with pytest.raises(ValueError,match="CIF"):
        export_replay_zip(gA,Q,beta,mart,h5oina_filename="scan.h5oina",
                          indexed_daughter_cif=b"not cif")
