"""No inferred correspondence, and symmetry-consistent IPF on indexed CTF."""
import numpy as np
from pathlib import Path

from twin_app.ctf_phase_handoff import (
    apply_ctf_crystal_handoff,ctf_crystal_handoff,
)


def sample_ctf():
    return (
        "Channel Text File\nXCells 2\nYCells 2\nPhases 3\n"
        "3.544;3.544;3.544\t90;90;90\tBeta3\t11\t225\t\t\n"
        "5.342;4.224;4.390\t90;90;90\t2H\t3\t59\t\t\n"
        "13.802;5.286;4.399\t90;113.6;90\t18R\t2\t12\t\t\n"
        "Phase X Y Bands Error Euler1 Euler2 Euler3 MAD BC BS\n"
        "3 0 0 8 0 12 35 44 .6 100 90\n"
        "3 1 0 8 0 27 32 12 .6 100 90\n"
        "3 0 1 8 0 55 25 21 .6 100 90\n"
        "0 1 1 0 3 0 0 0 0 55 45\n"
    ).encode()


def test_explicit_ctf_phase_handoff_preserves_C_and_OR_as_unknown():
    raw=sample_ctf()
    result=ctf_crystal_handoff(raw,parent_id=1,product_id=3)
    assert result["parent"]["point_group"]=="m-3m"
    assert result["parent"]["crystal_family"]=="cubic"
    assert result["product"]["point_group"]=="2/m"
    assert result["product"]["crystal_family"]=="monoclinic"
    assert result["product"]["angles_deg"][1]==113.6
    assert result["correspondence_status"]=="NOT_INFERRED"
    assert result["orientation_relationship_status"]=="NOT_IDENTIFIED"
    state={"correspondence_A_TO_M_0_0":"3/7"}
    apply_ctf_crystal_handoff(result,state,length_unit="angstrom")
    assert state["product_point_group_monoclinic"]=="2/m"
    assert state["product_a"]==13.802
    assert state["parent_a"]==3.544
    assert state["correspondence_A_TO_M_0_0"]=="3/7"
    assert state["tf_ctf_applied_ids"]==(1,3)


def test_missing_phases_never_create_fictional_parent():
    import pytest
    with pytest.raises(ValueError,match="phase IDs"):
        ctf_crystal_handoff(sample_ctf(),parent_id=9,product_id=3)
    with pytest.raises(ValueError,match="differ"):
        ctf_crystal_handoff(sample_ctf(),parent_id=3,product_id=3)


def test_ipf_uses_orix_standard_laue_key_and_valid_phase():
    import pytest
    pytest.importorskip("orix")
    from twin_app.ctf_ipf import symmetry_ipf_colors,plot_symmetry_ipf
    raw=sample_ctf()
    ipf=symmetry_ipf_colors(raw,"sample.ctf",phase_id=3,sample_direction="Z")
    assert ipf["rgb"].shape==(4,3)
    assert np.all((ipf["rgb"]>=0)&(ipf["rgb"]<=1))
    assert ipf["active_mask"].sum()==3
    np.testing.assert_allclose(ipf["rgb"][3],[.94,.94,.94])
    assert ipf["point_group"].replace(" ","")=="2/m"
    figure=plot_symmetry_ipf(ipf)
    assert len(figure.data)==1
    with pytest.raises(ValueError,match="No indexed pixels"):
        symmetry_ipf_colors(raw,"sample.ctf",phase_id=2)
    with pytest.raises(ValueError,match="reference direction"):
        symmetry_ipf_colors(raw,"sample.ctf",phase_id=3,sample_direction="N")


def test_blind_or_requires_observable_boundaries_and_never_returns_unique_C():
    from twin_app.ctf_blind_or import explore_blind_or
    from twin_app.ctf_independent import declared_ctf_phases
    result=explore_blind_or(
        {"grains":[],"adjacent_grain_pairs":[]},
        declared_ctf_phases(sample_ctf()),
        parent_phase_id=1,product_phase_id=3,
    )
    assert result["status"]=="insufficient_evidence"
    assert result["unique_physical_OR"] is False
    assert result["correspondence_C"]=="NOT_INFERRED"
