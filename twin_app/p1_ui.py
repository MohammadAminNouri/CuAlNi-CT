"""P1 Streamlit research presentation of evidence, not a new scientific solver."""
from __future__ import annotations
import json
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any
import streamlit as st
import numpy as np
from .p1_cualni import CELLS,metric_to_cell,gram,reduced_gamma_to_unique_b
from .p1_evidence import branch_evidence_status


def render_cualni_reference_guide():
    """Explicitly reference-only; never loads an unverified correspondence."""
    with st.expander('Cu–Al–Ni · phase/setting reference (not prevalidated predictions)',expanded=False):
        st.markdown('**Identify the phase before calculating.** The reported 2H and monoclinic cells come from different sources and may have different specimen composition and temperature. No entry supplies a verified parent→martensite correspondence.')
        for key in ('CuAlNi_2H_diffraction_reported','CuAlNi_monoclinic_C2m_EBSD_reported','CuAlNi_6M_James_Hane_2000'):
            r=CELLS[key]
            with st.expander(r.phase+' · '+r.cell_setting,expanded=False):
                a,b,c,al,be,ga=r.lattice
                st.code(f'a={a:.6g} Å  b={b:.6g} Å  c={c:.6g} Å\nα={al:.6g}°  β={be:.6g}°  γ={ga:.6g}°',language='text')
                st.caption(r.sample_warning)
                st.caption('Point group: '+r.point_group+' · Citation: '+r.source_url)
                if '6M' in key:
                    v=reduced_gamma_to_unique_b(r.lattice)
                    st.caption('For conventional unique-b metric *display only*: '+str(tuple(round(x,6) for x in v['new_cell'])))
                    st.caption('A real correspondence and symmetry operators must be transformed with the same basis matrix; this is NOT the 18R structure by implication.')
        st.warning('A CIF or lattice metric does not establish the correspondence C, the transformation OR, exact atom-site structure, or measured habit plane. Do not claim experimental validation from these values alone.')

    render_blind_evidence_comparison_panel()


def render_p1_source_evidence(entry,report, *, inline=False):
    """Retain complete equation and experimental evidence inside the first tab."""
    b=entry.construction
    evidence=branch_evidence_status(b)
    context=nullcontext() if inline else st.expander(
        'Numerical validity, twins, habits and evidence', expanded=False
    )
    with context:
        st.markdown('**1. Numerical equations**')
        x,y=st.columns(2)
        x.metric('Twin rank-one residual',f'{b.rank_one_residual:.2e}')
        y.metric('Proper-rotation residual',f'{b.rotation_residual:.2e}')
        if evidence['equation_status']!='RESIDUAL_PASS_DOUBLE_PRECISION':
            st.error('Numerical compatibility is NOT verified at the default residual threshold.')
        else:
            st.caption('Rank-one tensor and proper-rotation residuals pass the numerical threshold; this does not certify a physical microstructure or measurement precision.')
        st.markdown('**2. Twinning type evidence**')
        st.write(evidence['type_status'].replace('_',' ').title())
        st.caption(b.classification_status)
        st.markdown('**3. Habit solution inventory**')
        hs=tuple(b.habit_solutions)
        st.write(f'{len(hs)} reported habit record(s); '+('continuous compatible fraction family reported' if b.continuum_fraction else 'discrete calculations'))
        if hs:
            for h in hs:
                st.caption(f'λ={h.other_variant_volume_fraction:.9g} · branch {h.habit_branch:+d} · rank-one {h.rank_one_residual:.2e} · middle stretch {h.middle_stretch_residual:.2e}')
            fractions=[float(h.other_variant_volume_fraction) for h in hs]
            if any(min(v,1-v)<1e-6 for v in fractions):
                st.warning('Habit fraction near λ=0 or λ=1. Distinct-limit/endpoint interpretation requires additional mathematical care.')
        st.caption('Recorded habit branches are NOT automatically unique macroscopic or microscopic mechanisms. Equivalence requires proper parent-group orbit analysis with the correct metric and source provenance.')
        st.markdown('**4. Experimental evidence and limitations**')
        st.warning('Correspondence, orientation relationship, interface indexing and measurement uncertainty have NOT been independently verified against the active specimen by this calculation.')
        st.caption('This calculation uses the entered crystallographic correspondence and metrics. Source authenticity and experimental observation require independent evidence.')
        st.download_button('Export selected source numerical evidence (JSON)',data=json.dumps({
            'transformation_id':report.transformation_id,
            'source_pair':entry.pair.pair_id,
            'branch':b.branch,
            'classification':b.classification,
            'evidence':evidence,
            'habit_fractions':[float(h.other_variant_volume_fraction) for h in hs],
            'twin_shear':float(b.shear_magnitude),
            'a_parent_cartesian':list(b.a_parent_cartesian),
            'n_parent_cartesian':list(b.n_parent_cartesian),
            'habit_normals_parent_cartesian':[list(h.habit_normal_parent_cartesian) for h in hs],
            'warning':'Equation checks only; NOT experimentally validated.'},indent=2),
            file_name='twin_selected_source_numerical_evidence.json',mime='application/json',
            key='p1_evidence_'+str(report.transformation_id)+'_'+str(entry.pair.pair_id)+'_'+str(b.branch))


def render_p1_symmetry_orbit_panel(report,project_payload,signature, *, inline=False):
    """Keep the on-demand orbit analysis unchanged inside the research-tools tab."""
    context=nullcontext() if inline else st.expander(
        'Macroscopic symmetry orbits and solution counts', expanded=False
    )
    with context:
        st.caption('Raw records count interfaces; a symmetry orbit groups equivalent macroscopic tensors. Neither count by itself enumerates distinct physical laminate microstructures.')
        key='p1_orbit_result_'+signature[:20]
        if st.button('Calculate complete physical-tensor symmetry orbit inventory',key='p1_compute_orbits_'+signature[:20]):
            try:
                from cualni_cryst.project_io import project_from_dict
                from .p1_evidence import report_orbits
                with st.spinner('Grouping reported tensors using the true parent metric and proper symmetry operations…'):
                    project=project_from_dict(dict(project_payload),source='p1_orbits').project
                    transform=project.transformation(report.transformation_id)
                    parent=project.phase(transform.parent_phase_id)
                    result=report_orbits(report,parent.lattice.metric(),parent.symmetry_matrices())
                    st.session_state[key]=result
            except (ValueError,ArithmeticError,RuntimeError,AssertionError) as exc:
                st.error('The orbit calculation was not verified. No group count is shown.')
                st.code(f'{type(exc).__name__}: {exc}')
        result=st.session_state.get(key)
        if result:
            a,b=st.columns(2)
            a.metric('Raw twin branches',result['raw_twin_branches'])
            b.metric('Twin tensor symmetry orbits',result['macroscopic_twin_tensor_orbits'])
            x,y=st.columns(2)
            x.metric('Raw A/M habit records',result['raw_habit_records'])
            y.metric('Habit tensor symmetry orbits',result['macroscopic_habit_tensor_orbits'])
            st.warning('These orbits identify macroscopic b⊗m or a⊗n tensors only; physically distinct internal laminate structures cannot be inferred from this summary.')
            st.download_button('Export symmetry orbits, full provenance (JSON)',json.dumps(result,indent=2,default=str),
                file_name='cualni_p1_macroscopic_orbits.json',mime='application/json',
                key='p1_export_orbits_'+signature[:20])



def render_blind_evidence_comparison_panel():
    """Explicit, opt-in source-external comparison; no inferred reference data.

    All uploaded JSON files are session-scoped. Computed comparisons are tied
    to the hash of their exact current input bytes; no stale success display.
    """
    with st.expander('Research audit · compare independently measured 2H/M18R twin/habit geometry', expanded=False):
        st.warning('This is NOT a physical-validation certificate. Freeze complete predicted candidates before accessing specimen measurements. Two-dimensional EBSD surface traces do not determine 3D habit planes.')
        st.caption('Provide three JSON files: frozen prediction set (with solver SHA and crystal-input SHA), independent specimen evidence, and frame geometry. They must refer to the same specimen and explicitly indexed phase. Read docs/validation/BLIND_2H_18R_PROTOCOL.md.')
        files = [
            st.file_uploader('Frozen solver predictions (.json)', type=['json'], key='blind_predictions'),
            st.file_uploader('Independent measured interfaces (.json)', type=['json'], key='blind_observations'),
            st.file_uploader('Specimen metrics and frame transforms (.json)', type=['json'], key='blind_frames'),
        ]
        if not all(f is not None for f in files):
            st.info('No comparison performed. Measurements are never substituted with the published reference-cell table.')
            return
        from hashlib import sha256
        signature = sha256(b'\x00BLIND\x00'.join(f.getvalue() for f in files)).hexdigest()
        key = 'p1_blind_external_comparison'
        if st.button('Calculate all candidate residuals (no experimental certification)', key='blind_compare_action'):
            try:
                from .p1_blind_validation import compare_blind_interfaces
                prediction, evidence, geometry = [
                    json.loads(f.getvalue().decode('utf-8')) for f in files
                ]
                result = compare_blind_interfaces(
                    prediction, evidence,
                    parent_metric=geometry['parent_metric'],
                    product_metric=geometry.get('product_metric'),
                    product_to_parent_rotation=geometry.get('product_to_parent_rotation'),
                    parent_to_sample_rotation=geometry.get('parent_to_sample_rotation'),
                    null_trials=4096,
                )
                st.session_state[key] = {'signature': signature, 'result': result}
            except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
                st.session_state.pop(key, None)
                st.error('Evidence comparison rejected: ' + str(exc))
        cached = st.session_state.get(key)
        if cached is None or cached['signature'] != signature:
            st.caption('No current, valid comparison to display.')
            return
        result = cached['result']
        st.warning('GEOMETRIC ASSESSMENT ONLY. Independent experimental authentication, 3D observability, phase assignment and blind preregistration are not established here.')
        st.caption('Matching predictions cannot identify a physical variant or certify lattice correspondence by themselves. Interpret null values only under the declared isotropic hypothetical model.')
        st.dataframe([{
            'observation': r['observation_id'],
            'kind': r['interface_kind'],
            'dimensionality': r['geometry_dimension'],
            'nearest branch': r['nearest_candidate'],
            'angle (deg)': r['angle_deg'],
            'uncertainty (deg)': r['angular_uncertainty_deg'],
            'matching branches': r['matching_candidate_count'],
            'all searched': r['compared_candidates'],
            'isotropic-null fraction': r['isotropic_null_chance_at_observed_best_angle'],
        } for r in result['observations']], width="stretch")
        st.download_button('Download complete residuals, alternatives and limitations',
                           json.dumps(result, indent=2),
                           file_name='cualni_blind_geometry_assessment_NOT_VALIDATED.json',
                           mime='application/json', key='blind_export_current')
