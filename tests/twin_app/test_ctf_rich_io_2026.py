"""All native CTF columns must be surfaced without guessed phase/OR/twin labels."""
import numpy as np
import pytest

from twin_app.ctf_rich_io import (
    inspect_ctf_measurements, plot_ctf_numeric_map,
    ctf_field_histogram, raw_catalog_npz, raw_catalog_json,
)
def synthetic_2h_ctf():
    lines=[
        "Channel Text File","Prj test","JobMode Grid","XCells 4",
        "YCells 3","XStep 3","YStep 3","Phases 3",
        "3.544;3.544;3.544\t90;90;90\tparent\t11\t225\t\t",
        "5.342;4.224;4.390\t90;90;90\t2H\t3\t59\t\t",
        "13.802;5.286;4.399\t90;113.6;90\t18R\t2\t12\t\t",
        "Phase X Y Bands Error Euler1 Euler2 Euler3 MAD BC BS",
    ]
    for y in range(3):
        for x in range(4):
            lines.append(
                f"2 {3*x} {3*y} 8 0 {13 if x<2 else 71} "
                "37 12 0.9 100 80"
            )
    return ("\n".join(lines)+"\n").encode()


def test_complete_raw_catalog_including_quality_eulers_and_metadata():
    data = synthetic_2h_ctf()
    report = inspect_ctf_measurements(data, "aztec.ctf")
    assert report["n_points"] == 12
    assert report["indexed_count"] == 12
    assert report["grid"]["complete_unique_cartesian_grid"] is True
    assert (report["grid"]["n_x"],report["grid"]["n_y"]) == (4,3)
    assert {"PHASE","BANDS","ERROR","MAD","BC","BS","EULER1",
            "EULER2","EULER3"}.issubset({
                x.upper() for x in report["numeric_columns"]
            })
    assert np.array_equal(report["maps"]["EULER1"][:4], [13,13,71,71])
    assert "Band" not in report["interpretation"]
    assert report["phase_rows"][0]["phase_id"] == 2
    assert len(report["quality_summary"]) == 5
    assert len(report["file_sha256"]) == 64


def test_unindexed_quality_preserved_but_unindexed_eulers_masked():
    data=synthetic_2h_ctf().replace(
        b"2 0 0 8 0 13 37 12 0.9 100 80",
        b"0 0 0 0 3 0 0 0 0.0 35 43",
    )
    catalog=inspect_ctf_measurements(data,"u.ctf")
    assert catalog["indexed_count"] == 11
    assert np.isnan(catalog["maps"]["EULER1"][0])
    assert np.isnan(catalog["maps"]["MAD"][0])
    assert catalog["maps"]["BC"][0] == 35
    assert catalog["maps"]["BS"][0] == 43
    assert catalog["maps"]["ERROR"][0] == 3
    assert catalog["grid"]["complete_unique_cartesian_grid"]


def test_missing_phase_header_does_not_hide_raw_maps():
    minimal=(
        "Channel Text File\nXCells 2\nYCells 1\nPhases 1\n"
        "Phase X Y Bands Error Euler1 Euler2 Euler3 MAD BC BS\n"
        "2 0 0 8 0 10 20 30 0.8 100 70\n"
        "2 1 0 9 0 15 21 31 0.9 90 80\n"
    ).encode()
    catalog=inspect_ctf_measurements(minimal,"minimal.ctf")
    assert catalog["indexed_count"] == 2
    assert catalog["maps"]["BANDS"].tolist() == [8,9]


def test_irregular_2d_coordinates_not_rearranged_into_fake_pixels():
    text=synthetic_2h_ctf().replace(b"2 9 6 8",b"2 40 6 8")
    catalog=inspect_ctf_measurements(text,"irr.ctf")
    assert catalog["grid"]["complete_unique_cartesian_grid"] is False
    fig=plot_ctf_numeric_map(catalog,"BC")
    assert len(fig.data)==1
    assert fig.data[0].type=="scattergl"


def test_3d_z_slice_makes_map_correct_without_averaging_layers():
    text=(
        "Channel Text File\nXCells 2\nYCells 1\nZCells 2\n"
        "Phase X Y Z Bands Euler1 Euler2 Euler3 MAD BC BS\n"
        "2 0 0 0 8 10 20 30 .9 80 60\n"
        "2 1 0 0 8 10 20 30 .9 81 61\n"
        "2 0 0 1 8 10 20 30 .9 82 62\n"
        "2 1 0 1 8 10 20 30 .9 83 63\n"
    ).encode()
    catalog=inspect_ctf_measurements(text,"3d.ctf")
    assert catalog["grid"]["n_z"] == 2
    assert catalog["grid"]["complete_unique_cartesian_grid"] is True
    fig=plot_ctf_numeric_map(catalog,"BC",z_value=1.)
    assert list(fig.data[0].z[0]) == [82,83]


def test_hist_and_npz_and_json_preserve_provenance():
    data=inspect_ctf_measurements(synthetic_2h_ctf(),"aztec.ctf")
    hist=ctf_field_histogram(data,"MAD")
    assert len(hist.data[0].x)==12
    import io,json
    with np.load(io.BytesIO(raw_catalog_npz(data))) as archive:
        assert np.array_equal(archive["phase"],data["maps"]["phase"])
    payload=json.loads(raw_catalog_json(data))
    assert payload["n_points"]==12
    assert "maps" not in payload
    assert "raw_header" in payload


def test_malformed_data_rejected_not_guess_filled():
    data=synthetic_2h_ctf()
    with pytest.raises(ValueError,match="nonnumeric"):
        inspect_ctf_measurements(data.replace(b"0.9 100 80",b"oops 100 80"),"a.ctf")
    with pytest.raises(ValueError,match="filename"):
        inspect_ctf_measurements(data,"../a.ctf")
