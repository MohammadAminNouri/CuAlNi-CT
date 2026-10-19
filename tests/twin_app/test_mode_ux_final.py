"""Source-preserving mode/type UI tests; no published numbers used as answers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from twin_app.mode_view_model import (
    VERIFIED_TYPES, alphabetic_index, build_mode_view, entry_caption, type_entries,
)


@dataclass(frozen=True)
class Pair:
    variant_i: int
    variant_j: int
    stretch_i: int
    stretch_j: int


@dataclass(frozen=True)
class Construction:
    construction_id: str
    branch: int
    classification: str
    classification_status: str
    habit_solutions: tuple
    continuum_fraction: bool = False


@dataclass(frozen=True)
class Entry:
    pair: Pair
    construction: Construction
    operator_family: str


class Mode:
    def __init__(self, mode_id: str, mappings, *, shear=0.321, pairs=((1, 3),)):
        self.mode_id = mode_id
        self.shear = shear
        self.stretch_pairs = pairs
        self.types = tuple(mappings)
        self._mapping = mappings
        self.entries = tuple(entry for t in self.types for entry in mappings[t])

    def by_type(self, label):
        return self._mapping[label]


class Catalogue:
    def __init__(self, modes):
        self.modes = tuple(modes)


def entry(label, i, j, branch=1, habits=(), family="F1"):
    classification = label if label in VERIFIED_TYPES else "Exact rank-one relation — classification unresolved"
    status = ("cross-locked: independent geometry and provenance" if label in VERIFIED_TYPES else
              "independent cross-lock unresolved")
    return Entry(Pair(i, j, i, j), Construction(f"P{i}.B{branch}", branch, classification, status, tuple(habits)), family)


def test_only_computed_verified_types_appear_and_no_empty_categories():
    mode = Mode("Mode 1", {"Type I": (entry("Type I", 1, 3),),
                           "Type II": (entry("Type II", 1, 3, -1),)})
    view = build_mode_view(Catalogue([mode]))
    assert len(view.visible_modes) == 1
    result, = view.visible_modes
    assert result.display_name == "Mode A"
    assert tuple(item.label for item in result.types) == ("Type I", "Type II")
    assert "Compound" not in [item.label for item in result.types]
    assert result.unresolved_entries == ()
    assert view.all_source_record_count == 2


def test_compound_only_shows_compound():
    mode = Mode("Mode 1", {"Compound": (entry("Compound", 1, 2),)})
    visible, = build_mode_view(Catalogue([mode])).visible_modes
    assert tuple(t.label for t in visible.types) == ("Compound",)


def test_unresolved_never_promoted_and_all_sources_preserved():
    good = entry("Type I", 1, 3, habits=(object(),))
    unknown = entry("Unresolved", 2, 4)
    unverified_only = entry("Unresolved", 5, 6)
    mode_a = Mode("Mode 1", {"Type I": (good,), "Unresolved": (unknown,)})
    mode_b = Mode("Mode 2", {"Unresolved": (unverified_only,)})
    view = build_mode_view(Catalogue((mode_a, mode_b)))
    assert len(view.visible_modes) == 1
    assert view.visible_modes[0].display_name == "Mode A"
    assert [t.label for t in view.visible_modes[0].types] == ["Type I"]
    assert len(view.visible_modes[0].unresolved_entries) == 1
    assert len(view.diagnostic_modes) == 1
    assert view.unresolved_record_count == 2
    assert view.all_source_record_count == 3
    assert "M1–M3" in entry_caption(good)


def test_mode_letters_do_not_require_literature_or_shear_mapping():
    assert [alphabetic_index(i) for i in (0, 1, 25, 26, 27, 51, 52)] == ["A", "B", "Z", "AA", "AB", "AZ", "BA"]
    with pytest.raises(ValueError):
        alphabetic_index(-1)
    modes = [Mode(f"Mode {i}", {"Type I": (entry("Type I", i, i + 1),)}, shear=i * 0.017)
             for i in range(1, 29)]
    out = build_mode_view(Catalogue(modes)).visible_modes
    assert out[0].display_name == "Mode A"
    assert out[25].display_name == "Mode Z"
    assert out[26].display_name == "Mode AA"
    assert out[27].display_name == "Mode AB"


def test_habit_preferred_for_browsing_without_modifying_records():
    a, b = entry("Type II", 1, 3), entry("Type II", 2, 4, habits=(object(), object()))
    mode = Mode("Mode 1", {"Type II": (a, b)})
    selected = type_entries(mode, "Type II")
    assert selected == (b, a)
    assert len(selected) == len(mode.entries)
    view = build_mode_view(Catalogue((mode,)))
    assert view.visible_modes[0].recorded_habit_solutions == 2
    assert view.visible_modes[0].types[0].source_records == 2


def test_missing_or_duplicate_source_fails_closed():
    e = entry("Type I", 1, 3)
    with pytest.raises(ValueError, match="Duplicate"):
        build_mode_view(Catalogue((Mode("Mode 1", {"Type I": (e, e)}),)))
    with pytest.raises(ValueError, match="Missing"):
        build_mode_view(Catalogue((Mode("Mode 1", {"Type I": ()}),)))


def test_wrong_classification_and_status_cannot_claim_verified_type():
    bad = Entry(Pair(1, 3, 1, 3), Construction("P1.B1", 1, "Type I", "not independently validated", ()), "F1")
    with pytest.raises(ValueError, match="Unverified"):
        build_mode_view(Catalogue((Mode("Mode 1", {"Type I": (bad,)}),)))
    with pytest.raises(ValueError, match="Unknown twin-type"):
        build_mode_view(Catalogue((Mode("Mode 1", {"Other": (entry("Unresolved", 1, 3),)}),)))


def test_duplicate_mode_ids_or_bad_shear_rejected():
    e = entry("Type I", 1, 3)
    m = Mode("Mode 1", {"Type I": (e,)})
    with pytest.raises(ValueError, match="duplicate mode ID"):
        build_mode_view(Catalogue((m, m)))
    with pytest.raises(ValueError, match="positive"):
        build_mode_view(Catalogue((Mode("Mode 1", {"Type I": (e,)}, shear=float("nan")),)))


def test_ui_is_computed_only_and_never_reintroduces_correspondence_family_tree():
    root = Path(__file__).resolve().parents[2] / "twin_app"
    if not root.is_dir():  # standalone package smoke tests before installing
        root = Path(__file__).resolve().parents[1] / "src"
    view = (root / "mode_view_model.py").read_text(encoding="utf-8")
    renderer = (root / "mode_renderer.py").read_text(encoding="utf-8")
    for code in (view, renderer):
        for forbidden in ("0.2385", "0.2804", "0.3096", "0.1423", "NiTi", "CuAlNi", "Bhattacharya"):
            assert forbidden not in code, forbidden
    assert "render_navigator(" not in renderer
    assert "build_mode_view(catalogue)" in renderer
    assert "_show_audit(view, catalogue, report)" in renderer
    assert "st.radio(" in renderer  # habits remain selectable in the results pane
    assert "render_mode_tree(graph_model" in renderer
    assert "ct_ux_mode_selection" in renderer


def test_proved_symmetry_transport_can_be_displayed_without_claiming_direct_crosslock():
    status = ("symmetry-transported from directly cross-locked F1:P1:C1; "
              "G-index 2; ordered stretch and physical rank-one tensor residuals "
              "1.0e-09, 2.0e-09, 3.0e-09; "
              "this is NOT an independent discrete cross-lock for the target pair")
    transported = Entry(Pair(1, 3, 1, 3),
                        Construction("P1.B1", 1, "Type II", status, ()), "F1")
    view = build_mode_view(Catalogue((Mode("Mode 1", {"Type II": (transported,)}),)))
    assert len(view.visible_modes) == 1
    assert view.visible_modes[0].types[0].label == "Type II"
    assert view.all_source_record_count == 1


@pytest.mark.parametrize("bad_status", [
    "unresolved: previously cross-locked but no valid current witness",
    "not cross-locked",
    "symmetry-transported from directly cross-locked P1 (missing operator proof)",
])
def test_unverified_or_incomplete_symmetry_transport_stays_rejected(bad_status):
    bad = Entry(Pair(1, 3, 1, 3),
                Construction("P1.B1", 1, "Type I", bad_status, ()), "F1")
    with pytest.raises(ValueError, match="Unverified"):
        build_mode_view(Catalogue((Mode("Mode 1", {"Type I": (bad,)}),)))


def test_universal_workbench_does_not_show_phase_specific_warnings_by_default():
    root = Path(__file__).resolve().parents[2] / "twin_app"
    ui = (root / "streamlit_app.py").read_text(encoding="utf-8")
    evidence = (root / "p1_ui.py").read_text(encoding="utf-8")
    assert "2H / M18R evidence audit" not in ui
    assert "experimental validation of any 2H or M18R specimen" not in ui
    assert "if show_material_references:" in ui
    assert "render_cualni_reference_guide()" in ui
    assert "**4. Cu–Al–Ni and experimental status**" not in evidence
    assert "cualni_p1_selected_source_evidence.json" not in evidence
