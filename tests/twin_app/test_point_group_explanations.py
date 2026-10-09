from __future__ import annotations

import pytest

from twin_app.point_group_explainer import POINT_GROUP_SIGNATURES, explain_group


EXPECTED_SYMBOLS = {
    "1", "-1", "2", "m", "2/m", "222", "mm2", "mmm", "4", "-4",
    "4/m", "422", "4mm", "-42m", "4/mmm", "3", "-3", "32", "3m",
    "-3m", "6", "-6", "6/m", "622", "6mm", "-6m2", "6/mmm",
    "23", "m-3", "432", "-43m", "m-3m",
}


def test_each_of_the_32_classes_has_reviewed_non_generic_scientific_description():
    assert len(POINT_GROUP_SIGNATURES) == 32
    assert set(POINT_GROUP_SIGNATURES) == EXPECTED_SYMBOLS
    assert len(set(POINT_GROUP_SIGNATURES.values())) == 32
    assert all(len(v) >= 35 and len(v) <= 130 for v in POINT_GROUP_SIGNATURES.values())


def test_operation_facts_come_from_actual_determinants_and_route_inventory():
    data = explain_group(
        "m-3m",
        {"identity": 1, "inversion": 1, "mirror reflection": 9,
         "proper 2-fold rotation": 9, "proper 3-fold rotation": 8,
         "proper 4-fold rotation": 6, "improper order-4 operation": 6,
         "improper order-6 operation": 8},
        48,
        (1,) * 24 + (-1,) * 24,
    )
    assert data["proper"] == data["improper"] == 24
    assert data["mirror"] == 9
    assert data["twofold"] == 9
    assert data["higher"] == 14
    assert data["inversion"] is True
    assert "candidate" in str(data["route_I"]).lower()


def test_invalid_inventory_cannot_be_presented_as_sound_symmetry():
    with pytest.raises(ValueError):
        explain_group("m-3m", {"identity": 1}, 48, (1,))
    with pytest.raises(ValueError):
        explain_group("not-a-group", {"identity": 1}, 1, (1,))
