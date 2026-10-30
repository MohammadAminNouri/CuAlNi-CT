from __future__ import annotations

"""No untrusted pickle is ever executed in these tests."""

import io
from pathlib import Path
import zipfile
import pytest
from cualni_cryst.arpge_archive import (
    decode_transition,decode_cryst_metrics_without_unpickling,
    inspect_arpge_archive, audit_legacy_orientation_presets,
)

# Tiny self-contained source-format specimen, not executable source code.
NI_TI_AQ = (
    b"(lp0\nS'NiTi_AQ'\np1\naS'NiTi_B2'\np2\na"
    b'S"NiTi_B19\\\'"\np3\na(lp4\nI0\naI1\naI0\naa'
    b'(lp5\nI1\naI1\naI0\naa(lp6\na(lp7\na'
    b'(lp8\nI1\naI0\naI1\naa(lp9\nI-1\naI1\naI1\naa'
    b'(lp10\na(lp11\na.'
)


def test_safe_transition_decodes_2021_niti_aq_without_pickle_execution():
    record=decode_transition(NI_TI_AQ)
    assert record["name"]=="NiTi_AQ"
    assert record["parent"]=="NiTi_B2"
    assert record["primary_product_plane_hkl"]==[0,1,0]
    assert record["primary_parent_plane_hkl"]==[1,1,0]
    assert record["primary_product_direction_uvw"]==[1,0,1]
    assert record["primary_parent_direction_uvw"]==[-1,1,1]
    assert record["source_index_order"]=="daughter_then_parent"


def test_safe_archive_refuses_execution_opcodes():
    with pytest.raises(ValueError,match="executable|Unsupported"):
        decode_transition(b"cos\nsystem\n(S'echo dangerous'\ntR.")
    with pytest.raises(ValueError):
        decode_transition(b"\x80\x04cos\nsystem\n.")


def test_static_crystal_metric_prefix_parses_without_executing_numpy_payload():
    prefix=(b"(S'NiTi_B2'\np0\n"
            b"F3.015\nF3.015\nF3.015\n"
            b"F1.5707963267948966\nF1.5707963267948966\n"
            b"F1.5707963267948966\nS'Cubic'\np1\n"
            b"S'P'\np2\nS'm3m'\np3\n")
    record=decode_cryst_metrics_without_unpickling(prefix+b"cos\nsystem\n.")
    assert record["a_angstrom"]==pytest.approx(3.015)
    assert record["beta_deg"]==pytest.approx(90.0)
    assert record["atom_positions_validated"] is False


def test_zip_reader_rejects_zip_slip_and_extracts_only_verified_fields(tmp_path:Path):
    good=tmp_path/"safe.zip"
    with zipfile.ZipFile(good,"w") as z:
        z.writestr("original/Transitions/NiTi_AQ.trans",NI_TI_AQ)
    summary=inspect_arpge_archive(str(good))
    assert summary["transition_count"]==1
    assert summary["rejected_records"]==[]
    assert summary["legacy_source_executed"] is False
    assert summary["pickle_unserialized"] is False
    bad=tmp_path/"unsafe.zip"
    with zipfile.ZipFile(bad,"w") as z:
        z.writestr("../escape.trans",NI_TI_AQ)
    with pytest.raises(ValueError,match="Unsafe archive"):
        inspect_arpge_archive(str(bad))

def test_2021_legacy_burgers_source_index_order_is_hcp_then_bcc():
    text=(b"(lp0\nS'Burgers'\np1\naS'Ti_beta'\np2\naS'Ti_alpha'\np3\na"
          b"(lp4\nI0\naI0\naI1\naa(lp5\nI1\naI1\naI0\naa"
          b"(lp6\na(lp7\na"
          b"(lp8\nI1\naI0\naI0\naa(lp9\nI1\naI-1\naI1\naa"
          b"(lp10\na(lp11\na.")
    record=decode_transition(text)
    assert record["product"]=="Ti_alpha"
    assert record["primary_product_plane_hkl"]==[0,0,1]
    assert record["primary_parent_plane_hkl"]==[1,1,0]
    assert record["primary_product_direction_uvw"]==[1,0,0]
    assert record["primary_parent_direction_uvw"]==[1,-1,1]

def test_archived_niti_aq_or_matches_independent_2020_cartesian_geometry():
    import numpy as np
    from cualni_cryst.niti_literature_or import niti_published_or
    archived=decode_transition(NI_TI_AQ)
    phases=[
        {"name":"NiTi_B2","a_angstrom":3.015,"b_angstrom":3.015,
         "c_angstrom":3.015,"alpha_deg":90,"beta_deg":90,"gamma_deg":90},
        {"name":"NiTi_B19'","a_angstrom":2.889,"b_angstrom":4.12,
         "c_angstrom":4.622,"alpha_deg":90,"beta_deg":96.8,"gamma_deg":90},
    ]
    evaluated=audit_legacy_orientation_presets({
        "transitions":[archived],"crystal_metrics":phases,
    })
    result=evaluated["results"][0]
    expected=niti_published_or("AQ")["Q_parent_from_daughter"]
    assert result["status"]=="geometric candidate"
    assert result["max_direction_plane_residual_deg"]<1e-5
    assert np.allclose(result["Q_parent_from_daughter"],expected,atol=1e-10)
    assert evaluated["tested_transitions"]==1
