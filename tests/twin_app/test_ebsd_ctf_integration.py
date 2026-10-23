from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from twin_app.ebsd_integration import parse_explicit_or, build_ctf_config, run_ctf_observation, summarize_twin_candidates


def test_or_rejects_empty_reflection_and_nonrotation():
    for x in ("", "1 0 0\n0 1 0\n0 0 -1", "1 0 0\n0 1 0\n0 0 2", "0 0"):
        with pytest.raises(ValueError):
            parse_explicit_or(x)
    assert np.array_equal(parse_explicit_or("1 0 0\n0 1 0\n0 0 1"), np.eye(3))


def test_ctf_rejects_cif_and_missing_metadata():
    with pytest.raises(ValueError,match="Channel Text File"):
        run_ctf_observation(b"data_cif", {}, "A_to_M", parent_phase_id=1, product_phase_id=2, verified_or=np.eye(3), filename="input.ctf")


def test_no_physical_twin_certification_from_operator_fit():
    family=SimpleNamespace(pair_records=(SimpleNamespace(variant_i=1,variant_j=2,constructions=(object(),)),))
    report=SimpleNamespace(families=(family,))
    obs=SimpleNamespace(boundaries=({"accepted":True,"ambiguous":False},),physical_twin_evidence="NOT_CERTIFIED_FROM_CTF_ALONE")
    summary=summarize_twin_candidates(report,obs)
    assert summary["theoretical_twin_pairs"]==1
    assert summary["accepted_operator_fits"]==1
    assert summary["physical_twin_certification"]=="NOT_CERTIFIED_FROM_CTF_ALONE"
    assert summary["three_dimensional_twin_plane_measured"] is False


def test_config_uses_supplied_phase_metrics_without_manufacturing_a_transformation():
    from app.application import CalculationRequest, LatticeInput, PhaseInput, TransformationInput
    payload = CalculationRequest(
        project_id="test_ctf", title="Synthetic CTF input contract",
        parent=PhaseInput(phase_id="A", label="Parent", point_group="4/mmm", lattice=LatticeInput(3.1,3.1,5,90,90,90,"angstrom")),
        product=PhaseInput(phase_id="M", label="Product", point_group="mmm", lattice=LatticeInput(3,4,5,90,90,90,"angstrom")),
        transformation=TransformationInput.from_rows("A_to_M", "A", "M", [["1","0","0"],["0","1","0"],["0","0","1"]]),
    ).to_project_payload()
    cfg=build_ctf_config(payload,"A_to_M",parent_phase_id=1,product_phase_id=2,verified_or=np.eye(3))
    assert cfg["input"]["format"]=="ctf"
    assert cfg["phases"][0]["lattice"]["a"]==3.1
    assert cfg["phases"][1]["point_group"]=="mmm"
    assert cfg["or_refinement"]["enabled"] is False
    assert cfg["trace_validation"]["enabled"] is False
    with pytest.raises(ValueError, match="Distinct"):
        build_ctf_config(payload,"A_to_M",parent_phase_id=2,product_phase_id=2,verified_or=np.eye(3))


def test_real_ctf_parser_pipeline_with_independently_generated_synthetic_grains():
    """Whole CTF -> grain reconstruction path; not experimental 2H/18R validation."""
    from scipy.spatial.transform import Rotation
    from app.application import CalculationRequest, LatticeInput, PhaseInput, TransformationInput
    from cualni_cryst.ebsd_map import EBSDPhase
    from cualni_cryst.lattice import Lattice
    from cualni_cryst.ebsd_theory_bridge import build_theory_library
    parent_input=PhaseInput(phase_id='A',label='Parent',point_group='4/mmm',lattice=LatticeInput(3.1,3.1,5,90,90,90,'angstrom'))
    product_input=PhaseInput(phase_id='M',label='Product',point_group='mmm',lattice=LatticeInput(3,4,5,90,90,90,'angstrom'))
    project=CalculationRequest(project_id='synthetic_ctf',title='Independent CTF exercise',parent=parent_input,product=product_input,transformation=TransformationInput.from_rows('A_to_M','A','M',[['1','0','0'],['0','1','0'],['0','0','1']])).to_project_payload()
    R=Rotation.from_euler('ZXZ',[23,37,11],degrees=True).as_matrix()
    parent=EBSDPhase.from_point_group(1,'Parent',Lattice(3.1,3.1,5),'4/mmm')
    product=EBSDPhase.from_point_group(2,'Product',Lattice.orthorhombic(3,4,5),'mmm')
    theory=build_theory_library(parent,product,R)
    assert theory.n_variants >= 3
    g_parent=Rotation.from_euler('xyz',[11,-7,23],degrees=True).as_matrix()
    rows=[]
    for grain in range(3):
        g=g_parent @ theory.variant_set.variants[grain].R_parent_from_product
        e=Rotation.from_matrix(g).as_euler('ZXZ',degrees=True)
        for x in (grain*2,grain*2+1):
            for y in range(3):
                rows.append(f'2 {x} {y} 8 0 {e[0]:.12g} {e[1]:.12g} {e[2]:.12g} 0.2 100 200')
    header='Channel Text File\nPrj synthetic\nJobMode Grid\nXCells 6\nYCells 3\nXStep 1\nYStep 1\nPhases 2\nPhase X Y Bands Error Euler1 Euler2 Euler3 MAD BC BS\n'
    observation=run_ctf_observation((header+'\n'.join(rows)+'\n').encode(),project,'A_to_M',parent_phase_id=1,product_phase_id=2,verified_or=R,filename='synthetic.ctf',minimum_grain_points=2)
    assert observation.summary['segmentation']['n_grains']==3
    assert len(observation.boundaries)==2
    assert len(observation.grain_assignments)>=3
    assert observation.physical_twin_evidence=='NOT_CERTIFIED_FROM_CTF_ALONE'
