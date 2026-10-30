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
        ph.create_dataset("Phase Name", data=[b"18R synthetic monoclinic"])
        ld = ph.create_dataset("Lattice Dimensions", data=[[13.8017, 5.2856, 4.3987]])
        ld.attrs["Unit"] = "angstrom"
        la = ph.create_dataset("Lattice Angles", data=[np.deg2rad([90, 113.6, 90])])
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
    assert report["phase_counts"] == {"0": 2, "1": 6}
    assert report["phase_geometry"][0]["abc_angstrom"] == pytest.approx([13.8017, 5.2856, 4.3987])
    assert report["phase_geometry"][0]["angles_deg"] == pytest.approx([90, 113.6, 90])
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
