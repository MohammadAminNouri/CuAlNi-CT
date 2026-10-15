"""P1 closure negative controls: fail closed on plausible scientific false positives."""
import numpy as np
import pytest
from twin_app.p1_cualni import CELLS,check_cualni_specimen,CrystalInputError
from twin_app.p1_ensemble import propagate_metric_uncertainty, EnsembleInputError


def _two_h():
    return {
        "phase_label": "gamma1_prime_2H",
        "daughter_cell": CELLS["CuAlNi_2H_diffraction_reported"].lattice,
        "parent_cell": (3.,3.,3.,90.,90.,90.),
        "daughter_point_group": "mmm",
        "parent_point_group": "m-3m",
        "cell_setting": "orthorhombic conventional",
    }


def test_2h_cannot_claim_a_monoclinic_setting():
    spec=_two_h()
    spec["cell_setting"]="monoclinic unique-b"
    with pytest.raises(CrystalInputError,match="orthorhombic"):
        check_cualni_specimen(spec)


def test_monoclinic_cannot_claim_an_orthorhombic_setting():
    spec=_two_h()
    spec.update(
        phase_label="beta_prime_monoclinic",
        daughter_cell=CELLS["CuAlNi_monoclinic_C2m_EBSD_reported"].lattice,
        daughter_point_group="2/m",
        cell_setting="orthorhombic conventional",
    )
    with pytest.raises(CrystalInputError,match="unique-b"):
        check_cualni_specimen(spec)


def test_valid_2h_remains_lattice_only():
    audit=check_cualni_specimen(_two_h())
    assert audit["evidence_status"]=="LATTICE_ONLY"
    assert audit["accepted_as_physical_prediction"] is False


def _uncertainty_model(cov):
    return {
        "parent_cell":(3.,3.,3.,90.,90.,90.),
        "daughter_cell":(3.,3.1,3.2,90.,90.,90.),
        "parent_family":"cubic",
        "daughter_family":"orthorhombic",
        "measurement_covariance":cov,
        "measurement_distribution":"multivariate_gaussian",
    }


def test_scaled_asymmetric_covariance_is_rejected_before_solver():
    covariance=np.diag((2e6,2e6,2e6,2e6)).astype(float)
    covariance[0,1]=100000.0
    covariance[1,0]=100000.5
    def must_not_call(*args):
        raise AssertionError("Invalid covariance reached scientific solver")
    with pytest.raises(EnsembleInputError,match="symmetric"):
        propagate_metric_uncertainty(_uncertainty_model(covariance),
                                     must_not_call,n_samples=2)


def test_valid_covariance_is_not_rejected_by_symmetry_gate():
    covariance=np.diag((1e-4,1e-4,1e-4,1e-4))
    from types import SimpleNamespace
    def solver(*args):
        return SimpleNamespace(families=[])
    out=propagate_metric_uncertainty(_uncertainty_model(covariance),
                                     solver,n_samples=2,seed=5)
    assert out["covariance_symmetry_relative_residual"]==0
    assert out["status"]=="FULL_SOURCE_INVENTORY_STABLE_CONDITIONAL_ON_C"
