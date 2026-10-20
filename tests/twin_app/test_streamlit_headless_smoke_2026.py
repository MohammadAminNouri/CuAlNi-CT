"""Headless deployment-compatible Streamlit 3.14 startup and navigation smoke test."""
from __future__ import annotations
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]

def test_deployed_entrypoint_renders_accessible_linear_workflow():
    app = AppTest.from_file(str(ROOT / "twin_app" / "streamlit_app.py"), default_timeout=90)
    app.run()
    assert not app.exception, [str(e.message) for e in app.exception]
    assert len(app.tabs) == 0
    headings = [h.value for h in app.subheader]
    assert "01  /  Crystal setup" in headings
    assert "02  /  Calculated twinning modes" in headings
    # Default NiTi/general workflow must not display Cu-Al-Ni reference guides.
    assert not any("Cu–Al–Ni · phase/setting reference" in e.label for e in app.get("expander"))
    guide_toggle = next(c for c in app.checkbox if c.label == "Show optional material-specific references")
    guide_toggle.set_value(True)
    app.run()
    assert not app.exception, [str(e.message) for e in app.exception]
    assert any("Cu–Al–Ni · phase/setting reference" in e.label for e in app.get("expander"))

def test_switch_exact_correspondence_methods_and_reverse_direction_without_loss():
    """Real Streamlit widget callbacks must preserve the matrix, not just pure helpers."""
    from sympy import Matrix
    from twin_app.input_logic import exact_matrix
    from twin_app.streamlit_app import NITI_CORRESPONDENCE

    app = AppTest.from_file(str(ROOT / "twin_app" / "streamlit_app.py"), default_timeout=90)
    app.run()
    assert not app.exception, [str(e.message) for e in app.exception]

    def switch(key, option):
        controls = [r for r in app.radio if r.key == key]
        assert len(controls) == 1, key
        controls[0].set_value(option)
        app.run()
        assert not app.exception, [str(e.message) for e in app.exception]

    switch("correspondence_entry_method", "Exact rational 3 × 3 matrix")
    forward = tuple(tuple(app.session_state[f"correspondence_A_TO_M_{i}_{j}"]
                          for j in range(3)) for i in range(3))
    assert exact_matrix(forward) == exact_matrix(NITI_CORRESPONDENCE)

    switch("correspondence_direction_choice", "Product crystal → Parent crystal")
    reverse = tuple(tuple(app.session_state[f"correspondence_M_TO_A_{i}_{j}"]
                          for j in range(3)) for i in range(3))
    assert exact_matrix(reverse) == exact_matrix(NITI_CORRESPONDENCE).inv()

    switch("correspondence_entry_method", "Three paired lattice vectors")
    from twin_app.cayron_correspondence import calculate_from_paired_vectors, parse_triplet
    sources = [parse_triplet(app.session_state[f"correspondence_supercell_M_TO_A_src_{i}"])
               for i in range(3)]
    destinations = [parse_triplet(app.session_state[f"correspondence_supercell_M_TO_A_dst_{i}"])
                    for i in range(3)]
    paired = calculate_from_paired_vectors(sources, destinations, direction="M_TO_A")
    assert exact_matrix(paired.canonical_M_from_A) == exact_matrix(NITI_CORRESPONDENCE)
