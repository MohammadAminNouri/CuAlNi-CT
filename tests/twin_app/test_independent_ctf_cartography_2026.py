"""Falsification: independent AZtec CTF maps versus theory-dependent results.

All expectations concern measured map diagnostics or declared CTF headers.
No oracle OR, twin variant IDs or physically certified 2H/18R indexing.
"""
import io
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from app.application import CalculationRequest, LatticeInput, PhaseInput, TransformationInput
from twin_app.ctf_independent import (
    compare_ctf_model_settings, declared_ctf_phases,
    independent_ctf_cartography, plot_ctf_field,
)


def synthetic_2h_ctf():
    lines = [
        "Channel Text File", "Prj test",
        "JobMode Grid", "XCells 4", "YCells 3",
        "XStep 3", "YStep 3", "Phases 3",
        "3.544;3.544;3.544\t90;90;90\tparent\t11\t225\t\t",
        "5.342;4.224;4.390\t90;90;90\t2H\t3\t59\t\t",
        "13.802;5.286;4.399\t90;113.6;90\t18R\t2\t12\t\t",
        "Phase X Y Bands Error Euler1 Euler2 Euler3 MAD BC BS",
    ]
    for y in range(3):
        for x in range(4):
            # Two genuinely distinct measured orientation domains of 6 pixels
            # each. The AZtec orientation is deliberately not a known OR.
            euler1 = 13 if x < 2 else 71
            lines.append(
                f"2 {3*x} {3*y} 8 0 {euler1} 37 12 0.9 100 80"
            )
    return ("\n".join(lines) + "\n").encode("utf-8")


def matching_project():
    return CalculationRequest(
        project_id="ctf_exact", title="CTF phase compatibility",
        parent=PhaseInput(
            phase_id="A", label="Austenite", point_group="m-3m",
            lattice=LatticeInput(3.544,3.544,3.544,90,90,90,"angstrom"),
        ),
        product=PhaseInput(
            phase_id="M", label="2H", point_group="mmm",
            lattice=LatticeInput(5.342,4.224,4.390,90,90,90,"angstrom"),
        ),
        transformation=TransformationInput.from_rows(
            "A_to_M","A","M",
            [["1","0","0"],["0","1","0"],["0","0","1"]],
        ),
    ).to_project_payload()


def test_ctf_header_captures_unindexed_parent_and_18r():
    phases = declared_ctf_phases(synthetic_2h_ctf())
    assert [p["phase_id"] for p in phases] == [1,2,3]
    assert [p["space_group_number"] for p in phases] == [225,59,12]
    assert [p["declared_point_group"] for p in phases] == ["m-3m","mmm","2/m"]
    assert "NOT independently verified" in phases[1]["provenance"]


def test_or_free_grain_maps_use_real_orientations_and_declared_symmetry():
    results = independent_ctf_cartography(
        synthetic_2h_ctf(), "synthetic.ctf",
        misorientation_threshold_deg=3., minimum_grain_pixels=3,
    )
    assert results["n_pixels"] == 12
    assert results["n_indexed"] == 12
    assert results["n_grains"] == 2, results["grains"]
    assert results["n_distinct_grain_adjacencies"] == 1
    assert {g["phase_id"] for g in results["grains"]} == {2}
    assert sorted(g["pixels"] for g in results["grains"]) == [6,6]
    assert len(results["maps"]["grain_id"]) == 12
    assert np.isfinite(results["maps"]["kam_deg"]).all()
    assert results["source_boundary"].startswith("No parent orientation")
    fig = plot_ctf_field(results["maps"], "phase")
    assert len(fig.data) == 1
    assert results["status"] == "MEASURED_GRAIN_DIAGNOSTICS_NOT_TWIN_CERTIFICATION"


def test_independent_grain_threshold_is_real_physical_parameter():
    # Cannot "pass" a cutoff by substituting expected variant labels.
    results = independent_ctf_cartography(
        synthetic_2h_ctf(), "synthetic.ctf",
        misorientation_threshold_deg=90.,
        minimum_grain_pixels=3,
    )
    assert results["n_grains"] == 1


def test_model_matches_only_same_declared_cell_axes_and_group():
    phases=declared_ctf_phases(synthetic_2h_ctf())
    ok=compare_ctf_model_settings(
        phases, matching_project(), "A_to_M", 1,2,
    )
    assert ok["pass"],ok
    assert len(ok["comparisons"]) == 2
    assert all(p["maximum_length_relative_difference"] == pytest.approx(0)
               for p in ok["comparisons"])
    # The CTF header includes 18R but NO phase-3 indexed pixels:
    # choosing it as the product is NOT a valid 2H comparison.
    wrong=compare_ctf_model_settings(
        phases, matching_project(), "A_to_M", 1,3,
    )
    assert not wrong["pass"]


def test_wrong_point_group_or_wrong_2h_metric_blocks_comparison():
    payload = matching_project()
    phases = declared_ctf_phases(synthetic_2h_ctf())
    payload["phases"][1]["lattice"]["a"] = 4.9
    mismatch=compare_ctf_model_settings(phases,payload,"A_to_M",1,2)
    assert not mismatch["pass"]


def test_no_header_no_invented_atomic_symmetry():
    generic = b"Channel Text File\nPhases 1\nPhase X Y Bands Error Euler1 Euler2 Euler3 MAD BC BS\n2 0 0 7 0 0 0 0 0.9 80 60\n"
    assert declared_ctf_phases(generic) == ()
    with pytest.raises(ValueError, match="phase declarations"):
        independent_ctf_cartography(generic,"generic.ctf")


def test_broken_phase_header_fails_closed():
    text=synthetic_2h_ctf().replace(b"\t59\t",b"\t999\t")
    with pytest.raises(ValueError,match="space-group"):
        declared_ctf_phases(text)
