from __future__ import annotations

"""Full NiTi B2 -> B19' coherence suite across shared CuAlNi-CT pipelines.

Independent source quantities: Cayron 2020 OR geometry and Cayron 2026 CT
metrics are separate calibrations. ALL EBSD maps, grain boundaries, parent
directions and specimen orientations constructed below are SYNTHETIC.
Tests exercise the scientific interfaces, not experimental NiTi phase proof.
"""

from io import BytesIO
import json

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_h5oina_cualni import fixture_h5oina
from cualni_cryst.ar_gen_cry_pipeline import ar_stage, ar_parent_fit, cry_stage, gen_stage
from cualni_cryst.legacy_suite_bridge import Phase
from cualni_cryst.ebsd_map import EBSDPhase
from cualni_cryst.ebsd_h5oina import load_h5oina_sample
from cualni_cryst.lattice import Lattice
from cualni_cryst.niti_literature_or import niti_published_or, direction_cartesian
from cualni_cryst.parent_fullmap_h5oina import reconstruct_conditional_parent_h5oina
from cualni_cryst.parent_or_interop import (
    frozen_or_boundary_audit, mtex_or_contract, proper_cubic_parent_operations,
)
from cualni_cryst.parent_or_evidence import (
    audit_direct_basis, rebase_orientation_relationship,
    parent_orientation_from_two_directions, compare_parent_to_anchor,
)
from cualni_cryst.parent_replay_exchange import (
    forward_predict_daughter, forward_inverse_roundtrip, export_replay_zip,
)
from cualni_cryst.parent_variant_audit import daughter_variant_assignment


def setup_niti():
    parent=Lattice.cubic(3.015)
    daughter=Lattice.monoclinic_unique_b(2.889,4.120,4.622,96.8)
    ops_p=tuple(proper_cubic_parent_operations())
    ops_m=tuple(EBSDPhase.from_point_group(
        1,"NiTi B19-prime",daughter,"2/m",
    ).proper_symmetry_cartesian)
    Q=np.asarray(niti_published_or("AQ",parent=parent,daughter=daughter)["Q_parent_from_daughter"])
    gA=Rotation.from_euler("xyz",[18.,-24.,35.],degrees=True).as_matrix()
    return parent,daughter,ops_p,ops_m,Q,gA


def make_synthetic_niti_h5oina() -> tuple[bytes,np.ndarray]:
    p,m,pops,mops,Q,gA=setup_niti()
    predictions=forward_predict_daughter(gA,Q,pops,mops)
    ix=np.asarray([0,1,3,4,6,7],dtype=int)
    # Six distinct martensite variant orientations, including daughter
    # C2 indexer representatives, written in vendor radians/ZXZ convention.
    daughters=np.asarray([
        predictions["daughter_symmetry_representatives"][j,j%len(mops)]
        for j in range(len(ix))
    ])
    eulers=Rotation.from_matrix(daughters).as_euler("ZXZ")
    buf=BytesIO(fixture_h5oina())
    with h5py.File(buf,"r+") as h:
        h["1/EBSD/Header/Phases/1/Lattice Dimensions"][...] = [[m.a,m.b,m.c]]
        h["1/EBSD/Header/Phases/1/Lattice Angles"][...] = [
            np.deg2rad([m.alpha_deg,m.beta_deg,m.gamma_deg])
        ]
        h["1/EBSD/Data/Euler"][ix]=eulers.astype("f4")
    return buf.getvalue(),daughters


def test_niti_2026_cry_gen_correspondence_not_orientation():
    p=Phase("NiTi B2","m-3m",3.010,3.010,3.010)
    m=Phase("NiTi B19-prime","2/m",2.898,4.108,4.646,90,97.78,90)
    stage=cry_stage(p,m,(1,0,0),(0,1,0))
    Ctext="0 0 1\n1/2 1/2 0\n-1/2 1/2 0"
    result=gen_stage(stage,Ctext)
    assert result["variants"]==12
    assert result["operator_classes"]==7
    C=np.asarray([[0,0,1],[0.5,0.5,0],[-0.5,0.5,0]])
    F=m.lattice().structure_matrix() @ C/p.a
    stretches=np.linalg.svd(F)[1]
    assert np.sort(stretches)==pytest.approx(
        [0.930026279,0.965048059,1.119484928],rel=2e-8,
    )
    assert abs(stretches[1]-1)>0.03
    assert np.linalg.det(F)>0
    # A matrix in direct-lattice correspondence units is not an SO(3) OR.
    assert not np.allclose(C.T@C,np.eye(3),atol=1e-3)


def test_niti_2020_full_grid_h5oina_reconstructs_parent_and_fails_wrong_cell():
    p,m,pops,mops,Q,gA=setup_niti()
    raw,daughters=make_synthetic_niti_h5oina()
    source=ar_stage("synthetic_niti.h5oina",raw)
    assert source["indexed_points"]==6
    assert source["phase_geometry"][0]["abc_angstrom"]==pytest.approx(
        [2.889,4.120,4.622],abs=1e-5,
    )
    scan,audit=load_h5oina_sample("synthetic_niti.h5oina",raw,maximum_sample=6)
    assert len(scan.orientations)==6
    assert np.allclose(scan.orientations,daughters,atol=4e-6)

    assign=daughter_variant_assignment(
        scan.orientations,gA,Q,pops,mops,acceptance_deg=0.1,
    )
    assert assign["parent_generated_variant_count"]==12
    assert assign["accepted_fraction"]==pytest.approx(1.0)
    assert len(set(assign["variant_index_1based"]))==6

    blob,report=reconstruct_conditional_parent_h5oina(
        "synthetic_niti.h5oina",raw,gA,Q,pops,mops,
        expected_daughter_lattice=m,max_fit_deg=0.1,
        min_variant_patch_pixels=1,chunk_pixels=64,
    )
    assert report["n_fitted_within_deg"]==6
    assert report["n_quality"]==6
    assert report["parent_variant_count"]==12
    assert report["physical_parent_certified"] is False
    with h5py.File(BytesIO(blob),"r") as h:
        mask=h["map/assigned_variant_id_1based"][:]>0
        assert np.count_nonzero(mask)==6
        parents=Rotation.from_quat(h["map/parent_quat_xyzw"][:][mask]).as_matrix()
        assert max(compare_parent_to_anchor(v,gA,pops) for v in parents)<0.002
        assert h.attrs["source_h5oina_sha256"]==report["source_h5oina_sha256"]
    wrong_cell=Lattice.monoclinic_unique_b(2.898,4.108,4.646,97.78)
    with pytest.raises(ValueError,match="does not match the indexed H5OINA phase"):
        reconstruct_conditional_parent_h5oina(
            "synthetic_niti.h5oina",raw,gA,Q,pops,mops,
            expected_daughter_lattice=wrong_cell,
        )


def test_niti_2020_orientation_replay_boundary_probabilities_and_wrong_or():
    p,m,pops,mops,Q,gA=setup_niti()
    model=forward_predict_daughter(gA,Q,pops,mops)
    daughters=model["daughter_crystal_to_sample"]
    assert len(daughters)==12
    replay=forward_inverse_roundtrip(
        gA,Q,pops,mops,observed_daughter_g=daughters,accepted_deg=0.1,
    )
    assert replay["n_predicted_variants"]==12
    assert replay["weighted_fraction_within_threshold"]==pytest.approx(1)
    assert replay["max_forward_inverse_disagreement_deg"]<1e-4
    pairs=np.asarray([[daughters[i],daughters[i+1]] for i in range(5)])
    edges=frozen_or_boundary_audit(pairs,Q,pops,mops)
    assert edges["median_operator_misfit_deg"]<1e-4
    assert np.min(edges["mtex_style_probability"])>0.9999
    Qbad=np.asarray(niti_published_or("C",parent=p,daughter=m)["Q_parent_from_daughter"])
    wrong=forward_inverse_roundtrip(
        gA,Qbad,pops,mops,observed_daughter_g=daughters,accepted_deg=1,
    )
    assert wrong["weighted_fraction_within_threshold"]<0.4
    assert wrong["median_forward_residual_deg"]>3
    contract=mtex_or_contract(Q)
    assert np.allclose(contract["mtex_p2c_daughter_from_parent"],Q.T,atol=1e-12)
    zipped=export_replay_zip(
        gA,Q,pops,mops,h5oina_filename="synthetic_niti.h5oina",
        observed_daughter_g=daughters,
    )
    assert len(zipped)>1000


def test_niti_basis_conversion_and_two_independent_parent_axes():
    p,m,pops,mops,Q,gA=setup_niti()
    # Right-handed a/c interchange with one sign flip. The new beta angle
    # is 180 - 96.8 = 83.2 degrees; this is the SAME physical NiTi lattice.
    P=np.asarray([[0,0,-1],[0,1,0],[1,0,0]],dtype=int)
    m_new=Lattice(4.622,4.120,2.889,90,83.2,90)
    old_to_new=audit_direct_basis(m,m_new,P)
    assert old_to_new.metric_relative_residual<1e-12
    identity=audit_direct_basis(p,p,np.eye(3,dtype=int))
    Qnew=rebase_orientation_relationship(Q,identity,old_to_new)
    for uvw in ((1,0,0),(0,0,1),(1,1,1)):
        v_new=m_new.structure_matrix()@uvw
        v_old=m.structure_matrix()@P@uvw
        assert np.allclose(Qnew@v_new/np.linalg.norm(v_new),
                           Q@v_old/np.linalg.norm(v_old),atol=1e-12)
    d1=direction_cartesian(p,(1,0,0))
    d2=direction_cartesian(p,(0,1,1))
    anchored=parent_orientation_from_two_directions(
        p,(1,0,0),(0,1,1),gA@d1,gA@d2,
    )
    assert compare_parent_to_anchor(anchored,gA,pops)<1e-5


def test_niti_actual_ar_entrypoint_blind_parent_fit_and_independent_anchors():
    p,m,pops,mops,Q,gA=setup_niti()
    raw,_=make_synthetic_niti_h5oina()
    cry=cry_stage(
        Phase("NiTi B2","m-3m",3.015,3.015,3.015),
        Phase("NiTi B19-prime","2/m",2.889,4.120,4.622,90,96.8,90),
        (1,0,0),(0,1,0),
    )
    observed_axes={
        "parent_uvw":[[1,0,0],[0,1,1]],
        "sample_xyz":[
            (gA@direction_cartesian(p,(1,0,0))).tolist(),
            (gA@direction_cartesian(p,(0,1,1))).tolist(),
        ],
    }
    fitted=ar_parent_fit(
        "synthetic_niti.h5oina",raw,cry,1,json.dumps([Q.tolist()]),
        max_points=6,
        parent_anchor_orientation_json=json.dumps(gA.tolist()),
        parent_direction_pairs_json=json.dumps(observed_axes),
    )
    assert fitted["experimental_certification"] is False
    assert fitted["evidence_gate"]["physical_parent_or_certified"] is False
    assert fitted["full_parent_generated_variant_assignment"]["accepted_fraction"]==pytest.approx(1)
    assert fitted["parent_anchor_residual_deg"]<0.01
    assert fitted["parent_direction_pair_anchor"]["difference_from_parent_orientation_deg"]<1e-4
    assert compare_parent_to_anchor(
        np.asarray(fitted["parent_crystal_to_sample_rotation"]),gA,pops,
    )<0.01


def test_niti_specimen_frame_covariance_and_one_sided_frame_error():
    p,m,pops,mops,Q,gA=setup_niti()
    daughter=forward_predict_daughter(gA,Q,pops,mops)["daughter_crystal_to_sample"]
    H=Rotation.from_euler("xyz",[33,-14,20],degrees=True).as_matrix()
    rotated_daughter=H@daughter
    calibrated=forward_inverse_roundtrip(
        H@gA,Q,pops,mops,observed_daughter_g=rotated_daughter,
        accepted_deg=0.1,
    )
    assert calibrated["weighted_fraction_within_threshold"]==pytest.approx(1)
    assert calibrated["median_forward_residual_deg"]<1e-4
    mismatched=forward_inverse_roundtrip(
        gA,Q,pops,mops,observed_daughter_g=rotated_daughter,
        accepted_deg=1,
    )
    assert mismatched["weighted_fraction_within_threshold"]<0.5
    pairs=np.asarray([[daughter[i],daughter[i+1]] for i in range(5)])
    original=frozen_or_boundary_audit(pairs,Q,pops,mops)
    rotated=frozen_or_boundary_audit(H@pairs,Q,pops,mops)
    assert rotated["operator_misfit_deg"]==pytest.approx(
        original["operator_misfit_deg"],abs=1e-4,
    )
    # Global specimen rotations preserve daughter/daughter disorientation.
    assert rotated["mtex_style_probability"]==pytest.approx(
        original["mtex_style_probability"],abs=1e-6,
    )
