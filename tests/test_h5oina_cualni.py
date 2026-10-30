from __future__ import annotations

"""Synthetic Oxford H5OINA schema tests; no personal EBSD measurements in Git."""

from io import BytesIO

import h5py
import numpy as np
import pytest

from cualni_cryst.ebsd_h5oina import load_h5oina_sample
from cualni_cryst.ar_gen_cry_pipeline import ar_stage, ar_parent_fit, cry_stage
from cualni_cryst.legacy_suite_bridge import Phase


def fixture_h5oina(*, wrong_unit: bool = False, wrong_shape: bool = False) -> bytes:
    output = BytesIO()
    with h5py.File(output, "w") as h:
        h.create_dataset("Manufacturer", data=[b"Oxford Instruments"])
        h.create_dataset("Software Version", data=[b"6.2"])
        root = h.create_group("1/EBSD")
        data = root.create_group("Data")
        head = root.create_group("Header")
        head.create_dataset("Scanning Rotation Angle", data=[np.pi])
        head.create_dataset("Specimen Orientation Euler", data=[[0.0, 0.0, 0.0]])
        head.create_dataset("Tilt Angle", data=[np.deg2rad(70)])
        head.create_dataset("Tilt Axis", data=[0.0])
        head.create_dataset("X Cells", data=[4])
        head.create_dataset("Y Cells", data=[2])
        pixels = np.array([1, 1, 0, 1, 1, 0, 1, 1], np.uint8)
        data.create_dataset("Phase", data=pixels)
        eulers = np.zeros((8, 4 if wrong_shape else 3), np.float32)
        ds = data.create_dataset("Euler", data=eulers)
        ds.attrs["Unit"] = "deg" if wrong_unit else "rad"
        data.create_dataset("X", data=np.tile(np.arange(4), 2).astype("f4"))
        data.create_dataset("Y", data=np.repeat([0, 1], 4).astype("f4"))
        data.create_dataset("Mean Angular Deviation", data=np.full(8, 0.01, "f4"))
        ph = head.create_group("Phases/1")
        ph.create_dataset("Phase Name", data=[b"Synthetic monoclinic test phase"])
        ld = ph.create_dataset("Lattice Dimensions", data=[[10.0, 5.0, 4.0]])
        ld.attrs["Unit"] = "angstrom"
        la = ph.create_dataset("Lattice Angles", data=[np.deg2rad([90, 108.0, 90])])
        la.attrs["Unit"] = "rad"
        g = ph.create_dataset("Laue Group", data=[2])
        g.attrs["Symbol"] = "2/m"
        g = ph.create_dataset("Space Group", data=[12])
        g.attrs["Symbol"] = "C 2/m"
    return output.getvalue()


def test_h5oina_strict_phase_metrics_and_sampled_orientations():
    raw = fixture_h5oina()
    sampled, report = load_h5oina_sample("test.h5oina", raw)
    assert report["points"] == 8 and report["indexed_points"] == 6
    assert report["scan_rotation_angle_deg"] == pytest.approx(180, abs=1e-5)
    assert report["specimen_surface_euler_deg"] == pytest.approx([0, 0, 0])
    assert report["frame_corrected"] is False
    assert report["phase_counts"] == {"0": 2, "1": 6}
    assert report["phase_geometry"][0]["abc_angstrom"] == pytest.approx([10.0, 5.0, 4.0])
    assert report["phase_geometry"][0]["angles_deg"] == pytest.approx([90, 108.0, 90])
    assert report["selected_orientations"] == 6
    assert sampled.n_points == 6
    assert not sampled.metadata["reference_frame_normalization_applied"]
    assert np.allclose(sampled.orientations, np.eye(3)[None, :, :])


def test_ar_pipeline_native_ebsd_file():
    report = ar_stage("experiment.h5oina", fixture_h5oina())
    assert report["source_format"] == "h5oina"
    assert report["indexed_pct"] == pytest.approx(75)
    assert report["phase_geometry"][0]["space_group"] == "C 2/m"
    assert report["mean_angular_deviation_median_deg"] == pytest.approx(
        float(np.rad2deg(0.01)), rel=1e-5
    )


@pytest.mark.parametrize("payload", [b"not HDF5", b"\x89HDF"])
def test_reject_invalid_h5_data(payload):
    with pytest.raises(ValueError):
        load_h5oina_sample("test.h5oina", payload)


@pytest.mark.parametrize("bad", [{"wrong_unit": True}, {"wrong_shape": True}])
def test_reject_wrong_h5_schema_or_units(bad):
    with pytest.raises(ValueError):
        load_h5oina_sample("test.h5oina", fixture_h5oina(**bad))


def test_h5oina_parent_fit_refuses_incompatible_cry_product_cell():
    parent = Phase("Parent", "m-3m", 5.82, 5.82, 5.82)
    wrong = Phase("Unrelated NiTi", "2/m", 2.889, 4.12, 4.622, 90, 96.8, 90)
    cry = cry_stage(parent, wrong, (1, 0, 0), (1, 0, 0))
    import json
    with pytest.raises(ValueError, match="does not match"):
        ar_parent_fit("test.h5oina", fixture_h5oina(), cry, 1,
                      json.dumps([np.eye(3).tolist()]), max_points=3)


def test_h5oina_no_unique_parent_or_from_one_daughter_phase():
    _, audit = load_h5oina_sample("test.h5oina", fixture_h5oina())
    assert "No experimental parent OR" in audit["warning"]
    assert len(audit["phase_geometry"]) == 1


def test_independently_calibrated_frame_rotation_is_explicit_not_inferred():
    from scipy.spatial.transform import Rotation
    raw = fixture_h5oina()
    ebsd_raw, report_raw = load_h5oina_sample("test.h5oina", raw)
    frame = Rotation.from_euler("y", 180, degrees=True).as_matrix()
    ebsd_corrected, report_corrected = load_h5oina_sample(
        "test.h5oina", raw, sample_frame_correction=frame
    )
    assert report_raw["frame_corrected"] is False
    assert report_corrected["frame_corrected"] is True
    assert np.allclose(ebsd_corrected.orientations,
                       np.einsum("ij,njk->nik", frame, ebsd_raw.orientations))
    assert "EXPLICIT" in ebsd_corrected.metadata["orientation_convention"] or (
        "EXPLICIT" in ebsd_corrected.metadata.get("orientation_convention", "")
    )
    assert np.allclose(np.einsum(
        "nij,nkj->nik", ebsd_raw.orientations, ebsd_raw.orientations
    ), np.einsum(
        "nij,nkj->nik", ebsd_corrected.orientations, ebsd_corrected.orientations
    ))


def test_improper_frame_rotation_fails_closed():
    with pytest.raises(ValueError):
        load_h5oina_sample(
            "test.h5oina", fixture_h5oina(),
            sample_frame_correction=np.diag([1.0, 1.0, -1.0])
        )


def test_parent_fit_records_both_explicit_frame_and_anchor_evidence():
    import json
    from scipy.spatial.transform import Rotation
    pa = Phase("Parent", "m-3m", 5.8, 5.8, 5.8)
    ma = Phase("Synthetic reduced", "2/m", 10.0, 5.0, 4.0, 90, 108, 90)
    cry = cry_stage(pa, ma, (1, 0, 0), (1, 0, 0))
    explicit_frame = Rotation.from_euler("y", 180, degrees=True).as_matrix()
    fit = ar_parent_fit(
        "test.h5oina", fixture_h5oina(), cry, 1,
        json.dumps([np.eye(3).tolist()]), max_points=3,
        parent_anchor_orientation_json=json.dumps(explicit_frame.tolist()),
        indexed_18r_basis_verified=True, specimen_frame_verified=True,
        specimen_frame_correction_json=json.dumps(explicit_frame.tolist()),
    )
    assert np.allclose(fit["explicit_frame_correction_matrix"], explicit_frame)
    assert fit["parent_anchor_residual_deg"] < 1e-4
    assert fit["evidence_gate"]["declared_prerequisites_present"] is True
    assert fit["experimental_certification"] is False
    assert fit["full_parent_generated_variant_assignment"]["accepted_fraction"] == 1


def test_parent_fit_invalid_explicit_frame_fails():
    import json
    pa = Phase("Parent", "m-3m", 5.8, 5.8, 5.8)
    ma = Phase("Synthetic reduced", "2/m", 10.0, 5.0, 4.0, 90, 108, 90)
    cry = cry_stage(pa, ma, (1, 0, 0), (1, 0, 0))
    with pytest.raises(ValueError, match="proper|SO3|correction"):
        ar_parent_fit(
            "test.h5oina", fixture_h5oina(), cry, 1,
            json.dumps([np.eye(3).tolist()]), max_points=3,
            specimen_frame_correction_json=json.dumps([[1,0,0],[0,1,0],[0,0,-1]]),
        )
