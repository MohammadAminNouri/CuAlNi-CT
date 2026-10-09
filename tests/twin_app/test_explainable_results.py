from __future__ import annotations

from types import SimpleNamespace as Obj

from twin_app.point_group_explainer import POINT_GROUP_SHORT_NAMES, POINT_GROUP_SIGNATURES, simple_group_label
from twin_app.ux_language import (
    COLOR_FAMILY, COLOR_HABIT, COLOR_SELECTED, COLOR_UNRESOLVED,
    QUANTITY_HELP, representative_habit_solutions, twin_name,
)


def test_all_32_crystallographic_classes_have_readable_names():
    assert len(POINT_GROUP_SIGNATURES) == len(POINT_GROUP_SHORT_NAMES) == 32
    assert set(POINT_GROUP_SIGNATURES) == set(POINT_GROUP_SHORT_NAMES)
    assert simple_group_label("432") == "Cubic rotations only"
    assert simple_group_label("m-3m") == "Full cubic symmetry"
    assert simple_group_label("2/m") == "Half-turn, mirror and inversion"


def test_book_glossary_distinguishes_plane_vector_fraction_and_normal():
    for label in ("K₁", "η₁", "s", "a", "n̂", "λ", "b", "m"):
        assert label in QUANTITY_HELP
        assert len(QUANTITY_HELP[label]) > 20
    assert "plane" in QUANTITY_HELP["K₁"].lower()
    assert "direction" in QUANTITY_HELP["η₁"].lower()
    assert "magnitude" in QUANTITY_HELP["b"].lower() or "length" in QUANTITY_HELP["b"].lower()
    assert "fraction" in QUANTITY_HELP["λ"].lower()


def test_representatives_keep_both_signs_and_every_complementary_solution():
    plus = Obj(habit_branch=1, other_variant_volume_fraction=.29202)
    minus = Obj(habit_branch=-1, other_variant_volume_fraction=.29202)
    plus_other = Obj(habit_branch=1, other_variant_volume_fraction=.70798)
    minus_other = Obj(habit_branch=-1, other_variant_volume_fraction=.70798)
    original = (plus, minus, plus_other, minus_other)
    shown, extra = representative_habit_solutions(original)
    assert shown == (plus, minus)
    assert extra == (plus_other, minus_other)
    assert len(shown) + len(extra) == len(original)
    assert all(any(entry is original_item for entry in shown + extra)
               for original_item in original)


def test_uncertainty_label_is_not_assigned_a_false_twin_type():
    assert "not verified" in twin_name("classification cross-lock unresolved")
    assert twin_name("Type I").startswith("Type I")
    assert twin_name("Type II").startswith("Type II")
    assert twin_name("Compound").startswith("Compound")


def test_palette_is_three_low_saturation_accents_plus_neutral_structure():
    assert len({COLOR_SELECTED, COLOR_HABIT, COLOR_UNRESOLVED}) == 3
    assert COLOR_FAMILY not in {COLOR_SELECTED, COLOR_HABIT, COLOR_UNRESOLVED}
