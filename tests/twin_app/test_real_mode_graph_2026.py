"""Behavioral, fail-closed graph contracts. No benchmark values are hardcoded."""
from pathlib import Path
from dataclasses import dataclass
import ast

from twin_app.mode_tree_component import build_tree_model, validate_selection


@dataclass(frozen=True)
class Type:
    label: str
    source_records: int = 1
    stretch_couples: int = 2
    recorded_habit_solutions: int = 0
    continuous_families: int = 0
    entries: tuple = (object(),)


@dataclass(frozen=True)
class Mode:
    mode_id: str
    display_name: str
    shear: float
    symmetry_pairs: int
    types: tuple


def mode(i, kinds, *, shear=0.3):
    return Mode(f"Mode {i}", f"Mode {i}", shear, 12,
                tuple(Type(k) for k in kinds))


def test_unlimited_mode_graph_all_modes_and_only_verified_types():
    modes = tuple(mode(i, ("Type I", "Type II") if i % 2 else ("Compound",))
                  for i in range(1, 45))
    result = build_tree_model(modes)
    assert len(result["modes"]) == 44
    assert sum(len(m["types"]) for m in result["modes"]) == 66
    assert result["modes"][-1]["id"] == "Mode 44"
    assert result["modes"][1]["types"][0]["label"] == "Compound"


def test_reject_bogus_or_unverified_or_invalid_selections():
    model = build_tree_model((mode(1, ("Compound",)),))
    assert validate_selection({"mode": "Mode 1", "type": "Compound"}, model) == ("Mode 1", "Compound")
    for bad in ({"mode": "Mode 1", "type": "Type I"},
                {"mode": "Mode 100", "type": "Compound"},
                {"mode": "Mode 1", "type": "Compound", "shear": 99},
                None, {"mode": 1, "type": "Compound"}):
        if isinstance(bad, dict) and "shear" in bad:
            # Extra client fields are ignored: scientific data is never read from them.
            assert validate_selection(bad, model) == ("Mode 1", "Compound")
        else:
            assert validate_selection(bad, model) is None


def test_partial_symmetry_coverage_is_visible_in_graph():
    model = build_tree_model((mode(1, ("Type I", "Type II")),))
    assert all(t["verifiedPairs"] == 2 and t["totalPairs"] == 12
               for t in model["modes"][0]["types"])


def test_unverified_type_or_missing_sources_rejected():
    try:
        build_tree_model((mode(1, ("Unresolved",)),))
    except ValueError as e:
        assert "unresolved" in str(e).lower() or "unsupported" in str(e).lower()
    else:
        raise AssertionError("Unverified type entered graph")
    invalid = Mode("M1", "Mode A", 0.3, 12, (Type("Type I", entries=()),))
    try:
        build_tree_model((invalid,))
    except ValueError:
        pass
    else:
        raise AssertionError("Missing sources entered graph")


def test_interactive_frontend_and_calculation_are_separate():
    root = Path(__file__).resolve().parents[2]
    component = (root / "twin_app/mode_tree_component.py").read_text()
    renderer = (root / "twin_app/mode_renderer.py").read_text()
    html = (root / "twin_app/mode_tree_frontend/index.html").read_text()
    ast.parse(component)
    ast.parse(renderer)
    assert 'render_mode_tree(graph_model' in renderer
    assert 'twin, pair = entry.construction, entry.pair' in renderer
    assert 'build_mode_view(catalogue)' in renderer
    assert 'validate_selection(value, model)' in component
    assert 'streamlit:setComponentValue' in html
    assert 'streamlit:componentReady' in html
    assert 'mode-col' in html and 'type-fork' in html and 'root-stem' in html
    assert 'aria-pressed' in html and 'prefers-reduced-motion' in html
    for name in ('Type I', 'Type II', 'Compound'):
        assert name in html
    for forbidden in ('0.2385', '0.2804', '0.3096', '0.1423'):
        assert forbidden not in component + renderer + html
