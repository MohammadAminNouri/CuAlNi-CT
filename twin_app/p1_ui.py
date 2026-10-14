"""P1 Streamlit research presentation of evidence, not a new scientific solver."""
from __future__ import annotations
import json
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


def render_p1_source_evidence(entry,report):
    """One predictable science panel adjacent to numerical source results."""
    b=entry.construction
    evidence=branch_evidence_status(b)
    with st.expander('Scientific validity · numerical checks, twins, habits and evidence',expanded=False):
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
        st.markdown('**4. Cu–Al–Ni and experimental status**')
        st.warning('Experimental correspondence, OR, interface indexing and measurement uncertainty have NOT been verified by this calculation. Phase-specific Cu–Al–Ni claims require independently checked evidence.')
        st.caption('The application never assumes a Cu–Al–Ni correspondence from composition, 2H/18R phase labels or a lattice-parameter table.')
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
            file_name='cualni_p1_selected_source_evidence.json',mime='application/json',
            key='p1_evidence_'+str(report.transformation_id)+'_'+str(entry.pair.pair_id)+'_'+str(b.branch))


def render_p1_symmetry_orbit_panel(report,project_payload,signature):
    """On-demand audited orbit grouping; never confuses microstructure with H-orbit."""
    with st.expander('P1 · macroscopic symmetry orbits and solution-count interpretation',expanded=False):
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
