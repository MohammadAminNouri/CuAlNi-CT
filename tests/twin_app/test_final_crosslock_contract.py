from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENGINE = (ROOT / "twin_app" / "scientific_engine.py").read_text(encoding="utf-8")
MODELS = (ROOT / "twin_app" / "scientific_models.py").read_text(encoding="utf-8")


def test_classification_is_not_shear_only():
    assert "_system_type_i_geometry_residual" in ENGINE
    assert "_system_has_type_ii_generator_match" in ENGINE
    assert "K1/eta1 geometry + shear" in ENGINE
    assert "exact parent twofold provenance + shear" in ENGINE


def test_type_ii_does_not_compare_conjugate_k2_eta2_to_physical_k1_eta1():
    assert "Type-II's K2/eta2" in ENGINE
    assert "deliberately *not* compared to K1/eta1" in ENGINE


def test_crosslock_residuals_are_preserved_in_output_contract():
    for token in (
        "discrete_plane_angle_deg",
        "discrete_direction_angle_deg",
        "discrete_shear_relative_residual",
    ):
        assert token in MODELS
        assert token in ENGINE


def test_standalone_entrypoint_bootstraps_repo_and_src_paths():
    source = (ROOT / "twin_app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert 'ROOT = Path(__file__).resolve().parents[1]' in source
    assert 'ROOT / "src"' in source


def test_compound_never_deduplicates_type_i_k1_eta1_against_type_ii_k2_eta2():
    assert 'if str(lhs.kind) != str(rhs.kind):' in ENGINE
    assert 'if proposed == "Compound" and type_i_candidates and type_ii_candidates:' in ENGINE
    assert 'same rank-one branch has Type-I K1/eta1 geometry + shear' in ENGINE


def test_invalid_edit_state_is_caught_before_calculation_boundary():
    source = (ROOT / "twin_app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert 'One input is incomplete or incompatible.' in source
    # Only the actual calculation button is the safety boundary. CIF import
    # contains an earlier button and must not be mistaken for calculation.
    error_pos = source.index('One input is incomplete or incompatible.')
    calculate_pos = source.index('if st.button("Calculate physical twinning modes and habit planes"')
    solver_pos = source.index('report = build_twin_family_report(')
    assert error_pos < calculate_pos < solver_pos
    # Invalid crystal inputs return before the calculation button.
    assert "\n            return\n" in source[error_pos:calculate_pos]
    # Unconfirmed CIF pairs are blocked again immediately at calculation.
    pre_solver = source[calculate_pos:solver_pos]
    assert 'cif_admission_reason is not None' in pre_solver
    assert 'st.error("Calculation blocked: " + cif_admission_reason)' in pre_solver
    assert "\n            return\n" in pre_solver
