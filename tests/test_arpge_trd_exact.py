from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cualni_cryst.arpge_trd_exact import (
    legacy_llc_from_twin_chains, exact_sigma3_chain,
    identify_sigma3_edges,sigma3_tree_csl_and_llc,
)


@pytest.mark.parametrize("chains,expected",[
    ([],0),([["A"]],1),([["A","B"],["A","C"]],2),
    ([["A","B"],["C","D"]],4),
    ([["A","B","C"],["A","B"]],3),
    ([["A"],["A"],[]],1),
])
def test_original_2021_prefix_twin_word_llc_semantics(chains,expected):
    assert legacy_llc_from_twin_chains(chains)==expected


def test_composed_sigma3_is_not_automatically_three_to_n():
    first=exact_sigma3_chain([[1,1,1]])
    assert first["sigma_index"]==3
    twice=exact_sigma3_chain([[1,1,1],[1,1,1]])
    assert twice["sigma_index"]==1
    assert twice["chain_hops"]==2
    different=exact_sigma3_chain([[1,1,1],[1,-1,1]])
    assert different["sigma_index"]==9
    assert np.array_equal(
        np.asarray(different["numerator"]).T@np.asarray(different["numerator"]),
        81*np.eye(3,dtype=int),
    )
    with pytest.raises(ValueError,match="signed"):
        exact_sigma3_chain([[1,0,0]])


def test_exact_sigma3_spatial_domain_and_closure():
    twin=Rotation.from_rotvec(np.asarray([1.,1.,1.])/np.sqrt(3)*np.pi/3).as_matrix()
    orientations=np.asarray([np.eye(3),twin,twin@twin])
    result=identify_sigma3_edges(orientations,np.asarray([[0,1],[1,2]]),
                                 tolerance_deg=.001)
    assert result["n_accepted_sigma3_edges"]==2
    graph=sigma3_tree_csl_and_llc(3,result)
    assert graph["n_domains"]==1
    assert graph["domain_sizes"]==[3]
    # The product of two ideal twins is a cubic symmetry rotation => Sigma1.
    assert graph["composed_sigma_index_from_root"].tolist()==[1,3,1]
    assert graph["inconsistent_closed_loop_edges"]==0


def test_no_false_sigma3_edge_under_strict_tolerance():
    orientations=np.asarray([
        np.eye(3),Rotation.from_euler("xyz",[10,20,30],degrees=True).as_matrix(),
    ])
    result=identify_sigma3_edges(orientations,np.asarray([[0,1]]),
                                 tolerance_deg=.05)
    assert result["n_accepted_sigma3_edges"]==0
    graph=sigma3_tree_csl_and_llc(2,result)
    assert graph["n_domains"]==2
    with pytest.raises(ValueError,match="signed"):
        exact_sigma3_chain([[2,2,2]])
