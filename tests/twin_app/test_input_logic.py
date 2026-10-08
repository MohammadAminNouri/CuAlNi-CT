from twin_app.input_logic import canonical_correspondence_rows, displayed_relation


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
    assert displayed_relation("A_TO_M") == ("A → M", "u_M = C_(M←A) u_A")
    assert displayed_relation("M_TO_A") == ("M → A", "u_A = C_(A←M) u_M")
