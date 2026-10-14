"""Strict provenance gate for any claim of experimental Cu-Al-Ni prediction.

Lattice parameters alone NEVER validate correspondence, OR or habit plane.
No source-specific C, orientation relationship or EBSD observation is invented.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any,Mapping
import numpy as np
import sympy as sp

class ExperimentalEvidenceIncomplete(ValueError):
    pass

@dataclass(frozen=True)
class ExperimentalEvidenceAudit:
    experimental_status: str
    source_identifier: str
    phase_setting: str
    measured_plane_count: int
    correspondence_determinant: str


def validate_experimental_evidence(provenance:Mapping[str,Any]) -> ExperimentalEvidenceAudit:
    required=(
        'source_identifier','alloy_composition','measurement_temperature_K',
        'parent_phase','martensite_phase','parent_lattice_setting',
        'martensite_lattice_setting','correspondence_source_statement',
        'C_M_from_A','parent_and_daughter_OR_matrix',
        'orientation_reference_frames','measured_interface_planes',
    )
    missing=[key for key in required if key not in provenance or provenance[key] is None or provenance[key]=='' or provenance[key]==[]]
    if missing:
        raise ExperimentalEvidenceIncomplete(
            'Experimental comparison UNVERIFIED. Missing: '+', '.join(missing)
        )
    if not np.isfinite(float(provenance['measurement_temperature_K'])):
        raise ValueError('Invalid measurement temperature')
    C=sp.Matrix(provenance['C_M_from_A'])
    if C.shape!=(3,3) or C.det()==0:
        raise ValueError('Correspondence matrix must be invertible and 3x3')
    OR=np.asarray(provenance['parent_and_daughter_OR_matrix'],dtype=float)
    if OR.shape!=(3,3) or not np.isfinite(OR).all():
        raise ValueError('Orientation relationship must be a finite 3x3 matrix')
    if np.linalg.norm(OR.T@OR-np.eye(3),'fro')>1e-5 or abs(np.linalg.det(OR)-1)>1e-5:
        raise ValueError('Orientation relationship must be a proper rotation')
    observations=provenance['measured_interface_planes']
    if not isinstance(observations,list) or not observations:
        raise ExperimentalEvidenceIncomplete('At least one independent measured interface plane is required')
    for index,entry in enumerate(observations):
        if not isinstance(entry,dict) or any(k not in entry for k in ('normal','reference_frame','uncertainty_deg')):
            raise ExperimentalEvidenceIncomplete(f'Interface #{index} lacks normal/frame/uncertainty')
        normal=np.asarray(entry['normal'],dtype=float)
        uncertainty=float(entry['uncertainty_deg'])
        if normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<=0:
            raise ValueError(f'Invalid measured interface #{index}')
        if not (0<uncertainty<=90):
            raise ValueError(f'Unphysical uncertainty for interface #{index}')
    # This verifies EVIDENCE STRUCTURE ONLY, not truth of an external reference!
    # Dataset corroboration must be conducted independently before claiming
    # experimental agreement. There is deliberately no 'certified' status here.
    return ExperimentalEvidenceAudit(
        experimental_status='EVIDENCE STRUCTURE COMPLETE — EXTERNAL SOURCE REVIEW REQUIRED',
        source_identifier=str(provenance['source_identifier']),
        phase_setting=str(provenance['martensite_lattice_setting']),
        measured_plane_count=len(observations),
        correspondence_determinant=str(C.det()),
    )
