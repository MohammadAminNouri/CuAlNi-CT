from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cualni_cryst.arpge_genova_or import or_groupoid,audit_archived_or_groupoids
from cualni_cryst.lattice import Lattice
from cualni_cryst.ebsd_map import EBSDPhase
from cualni_cryst.parent_or_interop import proper_cubic_parent_operations
from cualni_cryst.niti_literature_or import niti_published_or


def test_original_2021_niti_aq_groupoid_has_twelve_variants_and_seven_operators():
    parent=tuple(proper_cubic_parent_operations())
    daughter=tuple(EBSDPhase.from_point_group(
        1,"NiTi B19prime",Lattice.monoclinic_unique_b(2.889,4.120,4.622,96.8),
        "2/m",
    ).proper_symmetry_cartesian)
    Q=np.asarray(niti_published_or("AQ")["Q_parent_from_daughter"])
    result=or_groupoid(Q,parent,daughter)
    assert result["parent_symmetry_order"]==24
    assert result["daughter_symmetry_order"]==2
    assert result["intersection_order"]==2
    assert result["n_variants"]==12
    assert result["n_operator_classes"]==7
    assert result["burnside_count"]==7
    assert sum(map(len,result["operator_class_member_parent_op_ids"]))==24
    assert len(result["operator_composition"])==7
    assert all(len(row)==7 for row in result["operator_composition"])
    assert all(any(len(c)>0 for c in row) for row in result["operator_composition"])
    assert result["status"].startswith("numerically verified")


def test_generic_proper_or_has_twenty_four_variants_and_twenty_four_operators():
    parent=tuple(proper_cubic_parent_operations())
    daughter=tuple(EBSDPhase.from_point_group(
        1,"NiTi B19prime",Lattice.monoclinic_unique_b(2.889,4.12,4.622,96.8),
        "2/m",
    ).proper_symmetry_cartesian)
    arbitrary=Rotation.from_euler("xyz",[9,13,29],degrees=True).as_matrix()
    result=or_groupoid(arbitrary,parent,daughter)
    assert result["intersection_order"]==1
    assert result["n_variants"]==24
    assert result["n_operator_classes"]==24
    assert result["burnside_count"]==24
    with pytest.raises(ValueError,match="proper"):
        or_groupoid(np.diag([1,1,-1]),parent,daughter)
    with pytest.raises(ValueError,match="group"):
        or_groupoid(arbitrary,parent[:10],daughter)


def test_all_source_presets_have_fail_closed_groupoid_report():
    from cualni_cryst.arpge_archive import decode_transition,audit_legacy_orientation_presets
    raw=(b"(lp0\nS'NiTi_AQ'\np1\naS'NiTi_B2'\np2\na"
         b'S"NiTi_B19\\\'"\np3\na(lp4\nI0\naI1\naI0\naa'
         b'(lp5\nI1\naI1\naI0\naa(lp6\na(lp7\na'
         b'(lp8\nI1\naI0\naI1\naa(lp9\nI-1\naI1\naI1\naa'
         b'(lp10\na(lp11\na.')
    inventory={
        "transitions":[decode_transition(raw)],
        "crystal_metrics":[
            dict(name="NiTi_B2",a_angstrom=3.015,b_angstrom=3.015,
                 c_angstrom=3.015,alpha_deg=90,beta_deg=90,
                 gamma_deg=90,point_group_legacy="m-3m"),
            dict(name="NiTi_B19'",a_angstrom=2.889,b_angstrom=4.120,
                 c_angstrom=4.622,alpha_deg=90,beta_deg=96.8,
                 gamma_deg=90,point_group_legacy="2/m"),
        ],
    }
    geo=audit_legacy_orientation_presets(inventory)
    output=audit_archived_or_groupoids(inventory,geo)
    assert output["n_presets"]==1
    assert output["results"][0]["variants"]==12
    assert output["results"][0]["operator_classes"]==7
    assert output["original_saved_operator_table_parity_verified"] is False
