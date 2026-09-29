from __future__ import annotations

from types import SimpleNamespace

from app.phase2_contracts import (
    deduplicate_pole_entries,
    physical_pole_label,
    projective_representative_notation,
    sanitize_scientific_text,
)


class _Point:
    def __init__(self, vector):
        self.vector = vector

    def to_dict(self):
        return {"cartesian_reference": list(self.vector)}


class _Family:
    def __init__(self, points):
        self.points = tuple(_Point(item) for item in points)


def test_raw_professor_booleans_are_humanized():
    text = "CT exact-compatible = False; CC1=False, CC2=True, CC3=False."
    cleaned = sanitize_scientific_text(text)
    assert "False" not in cleaned
    assert "True" not in cleaned
    assert "CT exact A/M compatibility: not satisfied" in cleaned
    assert "CC1: not satisfied" in cleaned
    assert "CC2: satisfied" in cleaned


def test_projective_plane_is_not_rendered_as_family_braces():
    assert projective_representative_notation("{3 0 10}", kind="plane") == "±(3 0 10)"
    assert projective_representative_notation("<11 -7 9>", kind="direction") == "±[11 -7 9]"


def test_duplicate_projective_pole_families_are_grouped_but_provenance_is_kept():
    a = _Family([(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)])
    b = _Family([(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    grouped = deduplicate_pole_entries(
        [
            {"family": a, "native_label": "sign branch A", "display_label": "CT twin"},
            {"family": b, "native_label": "sign branch D", "display_label": "CT twin"},
        ]
    )
    assert len(grouped) == 1
    assert grouped[0]["native_labels"] == ["sign branch A", "sign branch D"]


def test_ct_twin_label_uses_physical_metadata_not_sign_branch_text():
    row = SimpleNamespace(
        theory=SimpleNamespace(value="cayron_ct"),
        prediction_kind=SimpleNamespace(value="ct_mm_twin"),
        exact=True,
        branch_label="Type-I twin · operator 2 · twin 1 · sign branch A · type_i",
        metadata={"operator_index": 2, "twin_index": 1, "twin_kind": "I"},
    )
    label = physical_pole_label(row)
    assert label == "Cayron CT · operator 2 · twin 1 · Type-I"
    assert "sign branch" not in label
