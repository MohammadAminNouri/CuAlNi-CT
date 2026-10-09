"""Property-based metamorphic tests of input conventions and interface identity.

The scientific tensor identities are invariants. This is not a proof of physical
completeness; published-case integrations remain separate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis", reason="install the dev extra to run property-based science tests")
from hypothesis import given, settings, strategies as st

from twin_app.branch_presentation import distinct_twin_branches
from twin_app.input_explainer import preview_correspondence, validate_metric_parameters


@dataclass(frozen=True)
class Branch:
    branch: int
    a_parent_cartesian: tuple[float, float, float]
    n_parent_cartesian: tuple[float, float, float]


@settings(max_examples=64, deadline=None)
@given(st.integers(min_value=1, max_value=40), st.integers(min_value=1, max_value=40), st.integers(min_value=1, max_value=40))
def test_cubic_metric_volume_and_condition_are_scale_safe(a, b, c):
    # For an orthogonal metric: volume is product of lengths and normalized
    # condition of the angular Gram matrix is exactly 1.
    volume, condition = validate_metric_parameters((a, b, c), (90, 90, 90))
    assert np.isclose(volume, a*b*c, rtol=1e-13)
    assert np.isclose(condition, 1., rtol=1e-13)


@settings(max_examples=64, deadline=None)
@given(st.integers(min_value=1, max_value=10), st.integers(min_value=1, max_value=10), st.integers(min_value=1, max_value=10))
def test_correspondence_exact_inverse_and_columns(d1, d2, d3):
    matrix = ((str(d1), "0", "0"), ("0", str(d2), "0"), ("0", "0", str(d3)))
    forward = preview_correspondence(matrix, direction="A_TO_M")
    reverse = preview_correspondence(
        ((f"1/{d1}", "0", "0"), ("0", f"1/{d2}", "0"), ("0", "0", f"1/{d3}")),
        direction="M_TO_A",
    )
    assert forward.canonical_matrix == reverse.canonical_matrix
    assert forward.basis_mappings == reverse.basis_mappings


@settings(max_examples=64, deadline=None)
@given(st.integers(min_value=1, max_value=40), st.integers(min_value=1, max_value=40))
def test_identical_rank_one_tensors_deduplicate_even_when_vectors_reversed(a, b):
    x = Branch(-1, (a/10, b/10, 0.), (1., 0., 0.))
    y = Branch(+1, (-a/10, -b/10, 0.), (-1., 0., 0.))
    assert len(distinct_twin_branches([x, y])) == 1
