from __future__ import annotations

"""General, predictable entry point; no NiTi-vs-custom operating modes."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

import streamlit as st

from app.application import CalculationRequest, TransformationInput
from app.errors import ApplicationError
from cualni_cryst.numerics import DEFAULT_NUMERICAL_POLICY
from twin_app.input_ui import (
    render_correspondence_input, render_phase_input, render_weak_basis_input,
)
from twin_app.scientific_engine import build_twin_family_report
from twin_app.mode_catalogue import build_mode_catalogue
from twin_app.mode_renderer import render_mode_report
from twin_app.report_integrity import validate_report_integrity
from twin_app.report_export import make_export
from twin_app.structure_validation import parse_cif, independent_spglib_check
from twin_app.input_explainer import convert_cell_lengths, cell_compatible_with_family, cif_compatible_families


APP_TITLE = "Martensitic Crystallography · Research Workbench"

# Starting example INPUT only. The same inputs are fully editable; the solver
# never reads expected outputs from Bhattacharya's tables.
NITI_PARENT = (3.015, 3.015, 3.015, 90., 90., 90.)
NITI_PRODUCT = (2.889, 4.120, 4.622, 90., 96.8, 90.)
NITI_CORRESPONDENCE = (
    ("0", "0", "1"),
    ("1/2", "1/2", "0"),
    ("-1/2", "1/2", "0"),
)


def _style(*, appearance: str = "dark", font_scale: str = "standard") -> None:
    base_background = {"dark": "#101922", "light": "#fff", "contrast": "#050505"}[appearance]
    text_color = {"dark": "#f2f5f8", "light": "#18313f", "contrast": "#fff"}[appearance]
    muted_color = {"dark": "#c2cdd6", "light": "#344d5c", "contrast": "#fff"}[appearance]
    surface_color = {"dark": "#182631", "light": "#f3f7fa", "contrast": "#111111"}[appearance]
    field_color = {"dark": "#213342", "light": "#fff", "contrast": "#050505"}[appearance]
    border_color = {"dark": "#60788c", "light": "#758d9b", "contrast": "#fff"}[appearance]
    focus_color = {"dark": "#b8dfff", "light": "#175e9f", "contrast": "#fff"}[appearance]
    fontsize = "1.12rem" if font_scale == "large" else "1rem"
    st.markdown("""<style>
    :root { --tf-blue:#5C85AD; --tf-teal:#458C81; --tf-amber:#B68C4F;
            --tf-grey:#788491; }
    html { scroll-behavior: auto !important; }
    *, *::before, *::after {
      animation-duration: 0s !important; animation-delay: 0s !important;
      transition-duration: 0s !important;
    }
    .block-container { max-width: 1320px; padding-top: 1.25rem; padding-bottom: 3rem; }
    p, li { line-height: 1.65; font-size: 1.06rem; }
    label { font-size: 1.04rem !important; }
    [data-testid="stCaptionContainer"] p { font-size: 0.97rem; line-height:1.55; }
    div[data-baseweb="select"] { min-height: 48px; }
    [data-testid="stExpander"] { margin-top: 0.7rem; }
    .tf-alert-blue,.tf-alert-teal,.tf-alert-amber { padding: .85rem 1rem; border-left: 5px solid; border-radius: 5px; margin: .85rem 0 1rem; line-height: 1.6; }
    .tf-alert-blue { border-color: var(--tf-blue); background: rgba(92,133,173,.11); }
    .tf-alert-teal { border-color: var(--tf-teal); background: rgba(69,140,129,.11); }
    .tf-alert-amber { border-color: var(--tf-amber); background: rgba(182,140,79,.11); }
    h1 { font-size: 2.1rem !important; line-height:1.25; font-weight:750 !important; }
    h2 { font-size: 1.35rem !important; line-height:1.3; }
    h3,h4 { line-height:1.3; }
    button { font-weight: 600 !important; min-height: 44px; }
    button:focus-visible, [role="radio"]:focus-visible, [role="option"]:focus-visible {
      outline: 3px solid #8dbdff !important; outline-offset: 3px !important;
    }
    .tf-inline { padding: .72rem .95rem; border: 1px solid rgba(128,140,154,.26);
       border-left-width: 5px; border-radius: .45rem; margin: .7rem 0 .9rem 0;
       font-weight: 600; line-height: 1.4; }
    .tf-selection { border-left-color:var(--tf-blue); }
    .tf-habit { border-left-color:var(--tf-teal); }
    .tf-uncertain { border-left-color:var(--tf-amber); }
    .tf-legend { display:flex; flex-wrap:wrap; gap:.45rem 1.05rem;
      margin:.2rem 0 1rem; font-size:.9rem; }
    .tf-dot { display:inline-block; vertical-align:middle; width:.68rem; height:.68rem;
      border-radius:100%; margin-right:.3rem; }
    .tf-blue { background:var(--tf-blue); }
    .tf-teal { background:var(--tf-teal); }
    .tf-grey { background:var(--tf-grey); }
    [data-testid="stCaptionContainer"] p { line-height: 1.55; font-size: 1rem; }
    @media (prefers-reduced-motion: reduce) {
       *, *::before, *::after { animation:none !important; transition:none !important; }
    }
    .tf-pane-heading { color:var(--tf-accent); font-size:.81rem; font-weight:800; letter-spacing:.075em; margin-top:1rem; }
    .tf-quantity { border:1px solid rgba(124,146,163,.38); border-radius:11px; padding:15px 15px 14px; margin:8px 0 13px; background:rgba(115,136,152,.045); min-height:124px; }
    .tf-quantity-label {font-weight:700; font-size:1.07rem; margin-bottom:10px;}
    .tf-quantity-label span {font-weight:500; opacity:.8;}
    .tf-quantity-value {font-family:ui-monospace,Consolas,monospace; font-size:1.17rem; word-break:break-word; line-height:1.55; font-weight:600;}
    .tf-quantity-help {font-size:1rem; opacity:1; margin-top:9px; line-height:1.48;}
    .tf-banner {border:1px solid rgba(124,146,163,.4); border-left:5px solid #8592a0; border-radius:10px; padding:13px 16px; margin:16px 0; line-height:1.65;}
    .tf-banner-blue {border-left-color:#85b4e3;background:rgba(111,156,201,.11);}
    .tf-banner-amber {border-left-color:#e3bd81;background:rgba(227,189,129,.12);}
    .tf-banner-exact {border-left-color:#75bfb5;background:rgba(117,191,181,.11);}
    .tf-banner-neutral {border-left-color:#8995a2;background:rgba(125,145,160,.08);}
    [data-testid="stWidgetLabel"] {font-size:1rem;font-weight:650;}
    [data-testid="stExpander"] summary { min-height:48px; align-items:center; }
    </style>""" + f"""<style>
    :root {{
      --tf-fsize: {fontsize};
      --tf-bg: {base_background}; --tf-fg: {text_color};
      --tf-muted: {muted_color}; --tf-surface: {surface_color};
      --tf-field: {field_color}; --tf-border: {border_color};
      --tf-accent: {focus_color};
      /* Streamlit/BaseWeb tokens: the runtime appearance control must recolor
         real widget surfaces, not only the outer app background. */
      --background-color: {base_background};
      --secondary-background-color: {surface_color};
      --text-color: {text_color};
    }}
    .stApp, [data-testid="stAppViewContainer"] {{
      background: var(--tf-bg) !important; color:var(--tf-fg) !important;
      color-scheme: {"light" if appearance == "light" else "dark"};
    }}
    [data-testid="stSidebar"],[data-testid="stSidebarContent"] {{
      background:var(--tf-surface) !important;color:var(--tf-fg) !important;
    }}
    .stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp p,
    .stApp li,.stApp label,.stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stExpander"] summary {{
      color:var(--tf-fg) !important;
    }}
    .stApp [data-testid="stCaptionContainer"] p,
    .stApp [data-testid="stCaptionContainer"] span,
    .stApp [data-testid="stSidebar"] small {{
      color:var(--tf-muted) !important;
    }}
    .stApp [data-baseweb="input"],.stApp [data-baseweb="textarea"],
    .stApp [data-baseweb="select"],.stApp [data-testid="stNumberInput"] input,
    .stApp [data-testid="stTextInput"] input {{
      background:var(--tf-field) !important;
      color:var(--tf-fg) !important;
      border-color:var(--tf-border) !important;
    }}
    .stApp [data-baseweb="input"] input,.stApp [data-baseweb="textarea"] textarea,
    .stApp [data-baseweb="select"] *,
    .stApp [role="radiogroup"] label,.stApp [role="radiogroup"] p {{
      color:var(--tf-fg) !important;
    }}
    .stApp input::placeholder,.stApp textarea::placeholder {{
      color:var(--tf-muted) !important; opacity:1;
    }}
    .stApp [data-testid="stExpander"],.stApp [data-testid="stForm"],
    .stApp [data-testid="stAlert"] {{
      border-color:var(--tf-border);
    }}
    /* The built-in dark Streamlit theme does not automatically switch with our
       in-app light control. Keep formulas and ordinary widget controls legible. */
    [data-testid="stHeader"] {{background:var(--tf-bg) !important;}}
    .stApp .katex,.stApp .katex * {{color:var(--tf-fg) !important;}}
    .stApp [data-testid="stButton"] button,
    .stApp [data-testid="stDownloadButton"] button {{
      color:var(--tf-fg) !important;
      background:var(--tf-surface) !important;
      border-color:var(--tf-border) !important;
    }}
    .stApp [data-testid="stButton"] button:hover,
    .stApp [data-testid="stDownloadButton"] button:hover {{
      border-color:var(--tf-accent) !important;
    }}
    .stApp [data-testid="stRadio"] label span,
    .stApp [data-testid="stCheckbox"] label span,
    .stApp [data-testid="stSelectbox"] label span {{
      color:var(--tf-fg) !important;
    }}
    [data-baseweb="popover"],[role="listbox"] {{
      background:var(--tf-field) !important; color:var(--tf-fg) !important;
    }}
    [role="listbox"] [role="option"],[role="listbox"] span {{
      color:var(--tf-fg) !important;
    }}
    .stApp button:focus-visible,.stApp input:focus-visible,
    .stApp textarea:focus-visible,.stApp [role="radio"]:focus-visible {{
      outline:3px solid var(--tf-accent) !important;outline-offset:2px;
    }}
    .stApp p,.stApp label {{font-size:{fontsize}}}
    </style>""", unsafe_allow_html=True)


def _fingerprint(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cif_pair_signature(raw: dict[str, bytes]) -> str | None:
    """Identify one complete pair of CIF uploads; never infer missing phases."""
    if set(raw) != {"parent", "product"}:
        return None
    digest = hashlib.sha256()
    for role in ("parent", "product"):
        content = raw[role]
        digest.update(role.encode("ascii"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _cif_admission_reason(
    *, pair_signature: str | None, applied_signature: str | None,
    confirmed: bool,
) -> str | None:
    """Fail closed when structures, applied cells or acknowledgement are missing."""
    if pair_signature is None:
        return "Upload two valid CIF structures, one for each phase."
    if pair_signature != applied_signature:
        return "Apply both imported cell parameters before calculating."
    if not confirmed:
        return "Verify the crystallographic settings and correspondence explicitly."
    return None


def _basis_payload(basis_result: Any) -> object:
    if basis_result.basis is None:
        return None
    matrix = basis_result.basis.P_conventional_from_primitive
    return [[str(matrix[i, j]) for j in range(3)] for i in range(3)]


def _load_niti_example() -> None:
    """Resets input fields only; never supplies published twin/habit results."""
    for role, family, group, lattice in (
        ("parent", "cubic", "m-3m", NITI_PARENT),
        ("product", "monoclinic", "2/m", NITI_PRODUCT),
    ):
        st.session_state[f"{role}_family"] = family
        st.session_state[f"{role}_point_group_{family}"] = group
        for label, value in zip(("a", "b", "c", "alpha", "beta", "gamma"), lattice):
            st.session_state[f"{role}_{label}"] = value
    st.session_state["correspondence_direction_choice"] = "Parent crystal → Product crystal"
    # Keep both exact correspondence entry methods in sync with the restored NiTi INPUT.
    from twin_app.cayron_correspondence import seed_paired_vectors
    source_vectors, destination_vectors = seed_paired_vectors(NITI_CORRESPONDENCE, direction="A_TO_M")
    for j in range(3):
        st.session_state[f"correspondence_supercell_A_TO_M_src_{j}"] = source_vectors[j]
        st.session_state[f"correspondence_supercell_A_TO_M_dst_{j}"] = destination_vectors[j]
    st.session_state["correspondence_last_direction"] = "A_TO_M"
    for i, row in enumerate(NITI_CORRESPONDENCE):
        for j, value in enumerate(row):
            st.session_state[f"correspondence_A_TO_M_{i}_{j}"] = value
    st.session_state["twin_length_unit"] = "angstrom"
    st.session_state["tf_v9_last_unit"] = "angstrom"
    for key in (
        "twin_family_report", "twin_family_result_signature",
        "twin_tree_family_view", "twin_selected_couple",
        "tf_selected_couple_v8", "tf_text_finder_v8",
    ):
        st.session_state.pop(key, None)



def _render_research_workspace_header() -> None:
    st.title("Twin Family Explorer")
    st.caption("Crystal structures · correspondence · twins · habit planes")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    with st.sidebar:
        st.markdown("**Twin Family Explorer**")
        st.divider()
        st.subheader("Display preferences")
        appearance = st.radio("Appearance", ("Dark", "Light", "High contrast"), horizontal=False, key="tf_appearance_v8", help="Choose what is easiest to read. This never changes the scientific result.")
        font_scale = st.radio("Text size", ("Standard", "Large"), key="tf_font_scale_v8", horizontal=True)
        density = st.radio("Tree spacing", ("Comfortable", "Compact"), key="tf_density_v8", horizontal=True)
        st.divider()
        show_material_references = st.checkbox(
            "Show optional material-specific references", value=False,
            key="tf_show_material_references",
            help="Separate reference examples; never alters your active crystal inputs or calculated results.",
        )
    theme = {"Dark": "dark", "Light": "light", "High contrast": "contrast"}[appearance]
    _style(appearance=theme, font_scale=font_scale.lower())
    _render_research_workspace_header()
    # Keep material-specific research guides entirely opt-in: an NiTi or
    # arbitrary user-input calculation must never inherit Cu-Al-Ni labels.
    if show_material_references:
        st.caption("Optional material references (not active inputs or experimental validation)")
        from twin_app.p1_ui import render_cualni_reference_guide
        render_cualni_reference_guide()

    # One linear scientific workflow: enter parameters, press Calculate once,
    # then inspect the full result genealogy directly below.
    existing = st.session_state.get("twin_family_report")
    st.subheader("01  /  Crystal setup")
    # Input method is outside the editor: CIF upload is visible immediately
    # when selected, even when a previous calculation already exists.
    source = st.radio(
        "Input method",
        ("NiTi input example", "Manual lattice data", "Import CIF structures"),
        horizontal=True, key="tf_v9_input_method",
    )
    with st.expander("Crystal parameters and correspondence",
                     expanded=existing is None or source == "Import CIF structures"):
        imported_spglib = {}
        cif_pair_signature = None
        if source == "Import CIF structures":
            left, right = st.columns(2)
            imported = {}
            raw_cif = {}
            cif_families = {}
            st.caption("Upload the parent and product CIFs (2 MB each). The correspondence is entered below.")
            for name, col in (("parent", left), ("product", right)):
                with col:
                    uploaded = st.file_uploader(f"{name.capitalize()} CIF", type=("cif",), key=f"tf_v9_cif_{name}")
                    if uploaded is not None:
                        try:
                            bytes_cif = uploaded.getvalue()
                            parsed = parse_cif(bytes_cif, filename=uploaded.name)
                            imported[name] = parsed
                            raw_cif[name] = bytes_cif
                            allowed_families = cif_compatible_families(
                                parsed.lengths, parsed.angles,
                            )
                            cif_families[name] = st.selectbox(
                                f"{name.capitalize()} crystal system",
                                allowed_families,
                                key=f"tf_v9_cif_family_{name}",
                                help="Cell geometry only. Confirm the actual point group and basis below.",
                            )
                            st.caption(f"Imported · {parsed.site_count} sites")
                            if not parsed.symmetry_model:
                                st.warning("No declared space group. Verify the cell and symmetry.")
                            if parsed.has_partial_occupancy:
                                st.warning("Partial occupancy: atomic symmetry needs independent verification.")
                            try:
                                evidence = independent_spglib_check(parsed)
                            except (RuntimeError, ValueError) as exc:
                                st.caption(f"Independent symmetry check unavailable: {exc}")
                            else:
                                imported_spglib[name] = evidence
                                st.caption(f"spglib: {evidence['international']} (#{evidence['number']}), point group {evidence['pointgroup']}. Confirm the setting below.")
                        except (ValueError, RuntimeError) as exc:
                            st.error(f"Could not read {name} CIF: {exc}")
            cif_pair_signature = _cif_pair_signature(raw_cif)
            if st.button("Use imported cell parameters", key="tf_v9_use_cif_cells", disabled=cif_pair_signature is None):
                # Validate all pending imports before mutating any widget state.
                # Family-setting constraints may lock b/c/angles, so never
                # silently overwrite an independent CIF cell parameter.
                incompatible = []
                for name, parsed in imported.items():
                    chosen_family = cif_families.get(name)
                    if not cell_compatible_with_family(
                        parsed.lengths, parsed.angles, chosen_family
                    ):
                        incompatible.append(f"{name}: {chosen_family} conventional setting")
                if incompatible:
                    st.error("CIF cell conflicts with selected setting: " + "; ".join(incompatible))
                else:
                    display_unit = st.session_state.get("twin_length_unit", "angstrom")
                    for name, parsed in imported.items():
                        # Setting this phase's family before its widgets render
                        # prevents silent clamping of the imported CIF metric.
                        st.session_state[f"{name}_family"] = cif_families[name]
                        converted_lengths = convert_cell_lengths(
                            parsed.lengths, from_unit="angstrom", to_unit=display_unit
                        )
                        for label, value in zip(
                            ("a", "b", "c", "alpha", "beta", "gamma"),
                            converted_lengths + parsed.angles,
                        ):
                            st.session_state[f"{name}_{label}"] = value
                    st.session_state["tf_v9_applied_cif_pair_signature"] = cif_pair_signature
                    st.rerun()
        if source == "NiTi input example":
            st.button("Restore published NiTi crystal inputs", on_click=_load_niti_example)

        try:
            length_unit = st.selectbox(
                "Lattice length unit",
                ("angstrom", "nanometer", "picometer", "micrometer", "meter"),
                key="twin_length_unit",
            )
            previous_unit = st.session_state.get("tf_v9_last_unit", length_unit)
            if previous_unit != length_unit:
                for role in ("parent", "product"):
                    labels = ("a", "b", "c")
                    keys = tuple(f"{role}_{label}" for label in labels)
                    if all(key in st.session_state for key in keys):
                        new_lengths = convert_cell_lengths(
                            tuple(st.session_state[key] for key in keys),
                            from_unit=previous_unit, to_unit=length_unit,
                        )
                        for key, value in zip(keys, new_lengths):
                            st.session_state[key] = value
            st.session_state["tf_v9_last_unit"] = length_unit
            parent_ui = render_phase_input(
                prefix="parent", heading="Parent crystal (initial phase)", role="parent", length_unit=length_unit,
                default_family="cubic", default_point_group="m-3m", defaults=NITI_PARENT,
            )
            st.divider()
            product_ui = render_phase_input(
                prefix="product", heading="Product crystal (transformed phase)", role="product", length_unit=length_unit,
                default_family="monoclinic", default_point_group="2/m", defaults=NITI_PRODUCT,
            )
            st.divider()
            # Independent structure symmetry is advisory, never an override of
            # explicitly entered point-group and correspondence conventions.
            for role, selected_phase in (("parent", parent_ui.phase), ("product", product_ui.phase)):
                evidence = imported_spglib.get(role)
                if evidence and evidence["pointgroup"].replace(" ", "") != selected_phase.point_group.replace(" ", ""):
                    st.warning(
                        f"{role.capitalize()} point-group mismatch: CIF atoms suggest "
                        f"{evidence['pointgroup']}, but the calculation is set to "
                        f"{selected_phase.point_group}. Resolve the conventional setting before trusting the result."
                    )
                elif evidence:
                    st.caption(
                        f"{role.capitalize()} declared class agrees with the independent CIF-derived class. "
                        "Confirm the crystal-axis setting separately."
                    )
            correspondence_ui = render_correspondence_input(
                prefix="correspondence", default_rows=NITI_CORRESPONDENCE,
            )
            weak_basis_ui = render_weak_basis_input()

            transformation = TransformationInput.from_rows(
                "A_to_M", parent_ui.phase.phase_id, product_ui.phase.phase_id,
                correspondence_ui.canonical_rows, label="Entered lattice correspondence",
            )
            request = CalculationRequest(
                project_id="twin_family_workbench", title="Crystal twin transformation",
                parent=parent_ui.phase, product=product_ui.phase, transformation=transformation,
                notes="Crystallography derived only from entered parent/product metrics and correspondence.",
            )
            payload = request.to_project_payload()
            # Entered values, literature presets and CIF uploads are not
            # automatically *measured by this user*. Keep the backend adapter
            # untouched; adjust provenance for this standalone UI only.
            for phase_record in payload["phases"]:
                phase_record["provenance"] = {
                    "status": "UNVERIFIED",
                    "notes": (
                        f"Standalone UI input route: {source}. Physical source, "
                        "measurement conditions and uncertainties are not independently verified."
                    ),
                }
            for transformation_record in payload["transformations"]:
                transformation_record["provenance"] = {
                    "status": "UNVERIFIED",
                    "notes": (
                        "Entered crystallographic correspondence. Not independently "
                        "validated against a measured orientation relationship."
                    ),
                }
            cif_admission_reason = None
            if source == "Import CIF structures":
                applied = st.session_state.get("tf_v9_applied_cif_pair_signature")
                pair_applied = cif_pair_signature is not None and applied == cif_pair_signature
                if pair_applied:
                    confirm_key = "tf_v9_confirm_cif_" + _fingerprint({
                        "cif_pair": cif_pair_signature, "project": payload,
                    })[:20]
                    st.info("Check the two point groups and the correspondence below. CIFs do not supply the transformation matrix.")
                    confirmed = st.checkbox(
                        "I verified the crystal settings and correspondence.",
                        value=False, key=confirm_key,
                    )
                else:
                    confirmed = False
                cif_admission_reason = _cif_admission_reason(
                    pair_signature=cif_pair_signature,
                    applied_signature=applied,
                    confirmed=confirmed,
                )
                if cif_admission_reason:
                    st.caption("CIF: " + cif_admission_reason)
        except (ApplicationError, ValueError, ArithmeticError, AssertionError) as exc:
            st.error("Invalid crystal input.")
            with st.expander("Which input failed?", expanded=False):
                st.code(f"{type(exc).__name__}: {exc}")
            return

    signature = _fingerprint({"project": payload, "weak_basis": _basis_payload(weak_basis_ui)})
    previous = st.session_state.get("twin_family_result_signature")
    if previous is not None and previous != signature:
        st.caption("Inputs changed. Recalculate to update results.")

    # Experimental data belong with CIF/manual inputs, not below predictions.
    from twin_app.ebsd_ctf_ui import render_ctf_panel
    valid_report = (
        st.session_state.get("twin_family_report")
        if st.session_state.get("twin_family_result_signature") == signature
        else None
    )
    render_ctf_panel(valid_report, payload, transformation.transformation_id, signature)

    if st.button("Calculate physical twinning modes and habit planes", type="primary", width="stretch"):
        if source == "Import CIF structures" and cif_admission_reason is not None:
            st.error("Calculation blocked: " + cif_admission_reason)
            return
        st.session_state.pop("twin_family_report", None)
        st.session_state.pop("twin_mode_catalogue", None)
        st.session_state.pop("twin_family_result_signature", None)
        try:
            with st.spinner("Calculating exact symmetry and compatible twin interfaces…"):
                report = build_twin_family_report(
                    payload, transformation.transformation_id,
                    product_node_basis=weak_basis_ui.basis,
                )
                # Presentation-only structural/provenance audit. The scientific
                # engine and its independent matrix certifier remain untouched.
                validate_report_integrity(report)
                catalogue = build_mode_catalogue(report, payload)
        except (ApplicationError, ValueError, ArithmeticError, AssertionError) as exc:
            st.error(f"Calculation could not finish: {exc}")
        except Exception as exc:
            st.error("Calculation failed. No partial or invented result will be shown.")
            with st.expander("Technical error", expanded=False):
                st.code("".join(traceback.format_exception(exc)))
        else:
            st.session_state.pop("twin_tree_family_view", None)
            st.session_state.pop("twin_selected_couple", None)
            st.session_state.pop("tf_selected_couple_v6", None)
            st.session_state.pop("tf_selected_couple_v8", None)
            st.session_state.pop("tf_text_finder_v8", None)
            st.session_state.pop("tf_nav_epoch_p0b2", None)
            for saved_key in list(st.session_state):
                if saved_key == "ct_physical_mode" or saved_key.startswith(("ct_type_", "ct_entry_", "ct_ux_")):
                    st.session_state.pop(saved_key, None)
            st.session_state["twin_mode_catalogue"] = catalogue
            st.session_state["twin_family_report"] = report
            st.session_state["twin_family_result_signature"] = signature
            st.success("Calculation complete.")

    st.subheader("02  /  Calculated twinning modes")
    report = st.session_state.get("twin_family_report")
    if report is not None and st.session_state.get("twin_family_result_signature") == signature:
        catalogue = st.session_state.get("twin_mode_catalogue")
        if catalogue is None:
            st.warning("Stored report predates the physical-mode catalogue; recalculate to validate mode mapping.")
            return
        st.markdown("### Summary")
        summary_a, summary_b = st.columns(2)
        summary_a.metric("Calculated twinning modes", len(catalogue.modes))
        summary_b.metric("Distinct stretch variants", catalogue.mapped_stretch_count)
        st.caption("Computed variants and compatible pairs, not observed twins.")
        st.warning("Numerical compatibility is not experimental validation of the entered transformation. Independent specimen-specific phase, correspondence, orientation and interface measurements are required.")
        render_mode_report(catalogue, report)
        from twin_app.research_tools_ui import render_research_tools
        render_research_tools(catalogue, report, payload, signature)

        # Separate, opt-in comparison of the independent theory backends.
        # Existing modes, twin classifications and numerical engines are unchanged.
        from twin_app.theory_comparison_ui import render_theory_comparison
        render_theory_comparison(payload, transformation.transformation_id, signature)
    else:
        st.caption("Enter both phases and their correspondence, then calculate.")

if __name__ == "__main__":
    main()
