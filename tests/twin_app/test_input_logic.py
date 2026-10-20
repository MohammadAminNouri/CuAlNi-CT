import pytest
import sympy as sp

from twin_app.input_logic import canonical_correspondence_rows, displayed_relation, exact_matrix


def test_correspondence_direction_is_explicit_and_reverse_is_exact_inverse():
    reverse = (
        ("0", "1", "-1"),
        ("0", "1", "1"),
        ("1", "0", "0"),
    )
    canonical = canonical_correspondence_rows(reverse, direction="M_TO_A")
    assert canonical == (
        ("0", "0", "1"),
        ("1/2", "1/2", "0"),
        ("-1/2", "1/2", "0"),
    )
    # The generic UI names are intentional: the solver's A/M convention is
    # parent/product and must not silently change the underlying mapping.
    assert displayed_relation("A_TO_M") == (
        "Parent crystal → Product crystal",
        "u_product = C_(parent→product) u_parent",
    )
    assert displayed_relation("M_TO_A") == (
        "Product crystal → Parent crystal",
        "u_parent = C_(product→parent) u_product",
    )
    assert exact_matrix(reverse) * exact_matrix(canonical) == sp.eye(3)
    assert canonical_correspondence_rows(canonical, direction="A_TO_M") == canonical


def test_correspondence_direction_rejects_unknown_direction():
    with pytest.raises(ValueError, match="direction must be"):
        displayed_relation("SIDEWAYS")
    with pytest.raises(ValueError, match="direction must be"):
        canonical_correspondence_rows(sp.eye(3).tolist(), direction="SIDEWAYS")
