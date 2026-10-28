"""Input method must be prominent; CIF import must not silently clamp its cell."""
from pathlib import Path

from twin_app.input_explainer import cif_compatible_families


ROOT = Path(__file__).resolve().parents[2]
APP = (ROOT / "twin_app" / "streamlit_app.py").read_text(encoding="utf-8")
INPUT = (ROOT / "twin_app" / "input_ui.py").read_text(encoding="utf-8")
THEORY = (ROOT / "twin_app" / "theory_comparison_ui.py").read_text(encoding="utf-8")


def test_cif_choice_is_top_level_and_uploads_both_structures():
    radio = APP.index('source = st.radio(')
    editor = APP.index('with st.expander("Crystal parameters and correspondence"')
    assert radio < editor
    assert '"Import CIF structures"' in APP[radio:editor]
    assert 'expanded=existing is None or source == "Import CIF structures"' in APP
    assert 'st.file_uploader(f"{name.capitalize()} CIF"' in APP
    assert 'cif_pair_signature = _cif_pair_signature(raw_cif)' in APP
    assert 'st.session_state[f"{name}_family"] = cif_families[name]' in APP


def test_metric_compatible_cif_families_are_explicit_not_a_fake_space_group():
    assert cif_compatible_families((3, 3, 3), (90, 90, 90))[0] == "cubic"
    assert cif_compatible_families((4, 5, 6), (90, 90, 90))[0] == "orthorhombic"
    assert cif_compatible_families((4, 5, 6), (90, 98, 90))[0] == "monoclinic"
    assert cif_compatible_families((4, 4, 9), (90, 90, 120))[0] == "hexagonal"
    assert cif_compatible_families((3, 5, 7), (83, 91, 108)) == ("triclinic",)
    assert 'Confirm the actual point group and basis below' in APP


def test_long_optional_symmetry_prose_hidden_by_default():
    assert 'st.toggle(' in INPUT
    assert '"Show point-group details", value=False' in INPUT
    assert 'if not st.toggle(' in INPUT
    # Original scientific details remain available when requested.
    assert 'operation_witness' in INPUT
    assert 'metric_preservation_maximum_residual' in INPUT


def test_redundant_old_findings_and_marketing_removed_from_default_ui():
    assert "Four scientifically defensible findings" not in THEORY
    assert "How to turn these results into thesis evidence" not in THEORY
    assert "Crystallographic research workbench" not in APP
    assert "Experimental validation requires measurements</span>" not in APP
    assert "Cayron CT · Ball–James · PTMC · Cofactor" in THEORY
    assert "render_cayron_paper_comparison(unified, equivalence)" in THEORY


def test_original_cif_guard_and_solvers_not_bypassed():
    assert 'disabled=cif_pair_signature is None' in APP
    assert '"cif_pair": cif_pair_signature, "project": payload' in APP
    assert 'source == "Import CIF structures" and cif_admission_reason is not None' in APP
    assert "build_twin_family_report(" in APP
    assert "validate_report_integrity(report)" in APP
