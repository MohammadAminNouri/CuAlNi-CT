"""Never silently certify real CuAlNi from metrics or synthetic correspondence."""
import pytest
from twin_app.p0_experimental_evidence import (
    ExperimentalEvidenceIncomplete,validate_experimental_evidence,
)


def test_lattice_metrics_alone_cannot_certify_real_alloy():
    with pytest.raises(ExperimentalEvidenceIncomplete):
        validate_experimental_evidence({
            'source_identifier':'James & Hane (2000), Acta Mater 48, 197-222',
            'parent_phase':'cubic','martensite_phase':'6M',
            'product_cell':[4.430,5.330,12.79,90,90,95.68],
        })


def test_allegedly_complete_input_with_improper_or_rejected():
    d={key:'user-provided' for key in (
        'source_identifier','alloy_composition','parent_phase','martensite_phase',
        'parent_lattice_setting','martensite_lattice_setting',
        'correspondence_source_statement','orientation_reference_frames',
    )}
    d.update(measurement_temperature_K=300,
         C_M_from_A=[[1,0,0],[0,1,0],[0,0,1]],
         parent_and_daughter_OR_matrix=[[-1,0,0],[0,1,0],[0,0,1]],
         measured_interface_planes=[{'normal':[0,0,1],'reference_frame':'sample','uncertainty_deg':1}])
    with pytest.raises(ValueError,match='proper rotation'):
        validate_experimental_evidence(d)
