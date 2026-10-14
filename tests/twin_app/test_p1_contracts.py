import numpy as np
import pytest
import sympy as sp
from twin_app.p1_cualni import (CELLS,gram,metric_to_cell,reduced_gamma_to_unique_b,check_cualni_specimen,CrystalInputError,GAMMA_TO_UNIQUE_B)
from twin_app.p1_numerics import (stretch_from_metrics_independent,spectrum_diagnostics,root_oracle,symmetry_preserving_sensitivity,NumericalUncertainty)
from twin_app.p1_evidence import (proper_cartesian_group,projective_angle_deg,compare_observed_planes,branch_evidence_status,report_orbits,EvidenceError,reciprocal_normal_to_cartesian)
from dataclasses import dataclass


def _cell0():return (3.,3.,3.,90.,90.,90.)

def test_cualni_phase_registry_is_not_preloaded_solution():
    assert len(CELLS)>=3
    for ref in CELLS.values():
        assert ref.correspondence is None and ref.measured_or is None
        assert ref.interface_normals is None and np.linalg.eigvalsh(gram(ref.lattice))[0]>0


def test_monoclinic_gamma_to_beta_rebasis_preserves_metric_and_volume():
    cell=CELLS['CuAlNi_6M_James_Hane_2000'].lattice
    x=reduced_gamma_to_unique_b(cell)
    assert abs(x['new_cell'][4]-95.68)<1e-10
    assert abs(x['new_cell'][3]-90)<1e-10
    P=np.asarray(GAMMA_TO_UNIQUE_B,float)
    assert np.allclose(gram(x['new_cell']),P.T@gram(cell)@P)
    assert abs(x['volume_ratio']-1)<1e-13


def test_reject_mixed_6M_18R_and_bad_point_group():
    specimen={'phase_label':'faulted_monoclinic_reduced_6M','daughter_cell':CELLS['CuAlNi_6M_James_Hane_2000'].lattice,
              'parent_cell':_cell0(),'daughter_point_group':'2/m','parent_point_group':'m-3m','cell_setting':'gamma-angle reduced'}
    with pytest.raises(CrystalInputError,match='rebased'):check_cualni_specimen(specimen)
    specimen['phase_label']='gamma1_prime_2H'
    specimen['cell_setting']='orthorhombic conventional'
    with pytest.raises(CrystalInputError):check_cualni_specimen(specimen)


def test_c_matrix_exact_and_experiment_not_implicitly_certified():
    specimen={'phase_label':'gamma1_prime_2H','daughter_cell':CELLS['CuAlNi_2H_diffraction_reported'].lattice,
              'parent_cell':_cell0(),'daughter_point_group':'mmm','parent_point_group':'m-3m','cell_setting':'orthorhombic conventional'}
    assert check_cualni_specimen(specimen)['evidence_status']=='LATTICE_ONLY'
    specimen['correspondence_M_from_A']=[['1','0','0'],['0','1/2','0'],['0','0','1']]
    assert check_cualni_specimen(specimen)['evidence_status']=='C_UNVERIFIED'
    specimen['correspondence_source']='DOI example not externally checked'
    assert check_cualni_specimen(specimen)['accepted_as_physical_prediction'] is False
    specimen['experimental_status']='validated'
    with pytest.raises(CrystalInputError):check_cualni_specimen(specimen)


def test_independent_stretch_identity_and_scaling():
    p=gram(_cell0());m=gram((3.2,2.8,3.1,90,90,90))
    u=stretch_from_metrics_independent(p,m,np.eye(3))
    assert np.allclose(np.linalg.eigvalsh(u),sorted([3.2/3,2.8/3,3.1/3]))
    assert np.allclose(u,stretch_from_metrics_independent(p*7,m*7,np.eye(3)))


def test_degenerate_sensitivity_flagged_and_no_parent_daughter_mixup():
    u=np.diag((.97,.97,1.04))
    assert spectrum_diagnostics(u)['eigenvector_sensitivity_status']=='NOT_UNIQUE_AT_DEGENERACY'
    d=symmetry_preserving_sensitivity(_cell0(),(3.04,3.04,3.04,90,90,90),np.eye(3),
       parent_family='cubic',product_family='cubic',gram_fn=gram)
    assert d['status']=='EIGENCLUSTER_NONSMOOTH'
    assert set(d['derivatives'])=={'parent_a_shared','daughter_a_shared'}
    assert not any('a_A' in x for x in d['derivatives'])


def test_orthorhombic_sensitivities_keep_fixed_right_angles():
    s=symmetry_preserving_sensitivity(_cell0(),(2.8,3.1,3.2,90,90,90),np.eye(3),
      parent_family='cubic',product_family='orthorhombic',gram_fn=gram)
    assert s['status']=='LOCAL_SYMMETRY_PRESERVING'
    assert len(s['derivatives'])==4
    assert 'daughter_a' in s['derivatives'] and 'parent_a_shared' in s['derivatives']


def test_root_oracle_reports_missing_fraction_and_endpoint():
    u=np.diag((.92,1,1.08));a=np.array((.2,.1,.01));n=np.array((0.,1.,0.))
    r=root_oracle(u,a,n)
    assert any(abs(v)<1e-8 for v in r['roots']) and 'ENDPOINT_ROOT' in r['flags']
    assert r['unmatched_oracle']
    r2=root_oracle(u,a,n,(0.0,))
    assert r2['unmatched_oracle']==[]
    assert r2['status']=='NEEDS_BOUNDARY_INVESTIGATION'


def test_orbit_group_proper_symmetry_and_crystal_normals():
    G=np.eye(3)
    q=np.diag((-1.,-1.,1.))
    group=proper_cartesian_group(G,(np.eye(3),q))
    assert len(group)==2
    n=reciprocal_normal_to_cartesian((1,0,0),np.diag((4,9,16)))
    assert projective_angle_deg(n,(1,0,0))<1e-10
    with pytest.raises(EvidenceError):proper_cartesian_group(G,(np.eye(3),np.diag((2,1,1))))


def test_measured_product_plane_requires_valid_or():
    observed={'frame':'product_reciprocal_miller','coordinates':[1,0,0],'uncertainty_deg':2.}
    with pytest.raises(EvidenceError):compare_observed_planes([(1,0,0)],observed,parent_metric=np.eye(3))
    out=compare_observed_planes([(1,0,0)],observed,parent_metric=np.eye(3),product_metric=np.eye(3),OR_M_cart_to_A_cart=np.eye(3))
    assert out['angle_deg']<1e-8 and out['status'].startswith('GEOMETRIC_MATCH')
    assert 'NOT_EXPERIMENT' not in out['status']  # not falsely marked experimentally certified


def test_branch_status_not_mistaken_for_experimental_validation():
    @dataclass
    class Branch:classification:str='Type I';classification_status:str='direct cross-locked';rank_one_residual:float=1e-12;rotation_residual:float=1e-12
    b=Branch();assert branch_evidence_status(b)['experimental_status']=='NOT_EXPERIMENTALLY_VALIDATED'
    b.classification='Nonconventional rank-one (parent twofold excluded)'
    assert branch_evidence_status(b)['type_status'].startswith('NONCONVENTIONAL')


def test_ensemble_requires_actual_covariance_and_symmetry():
    from twin_app.p1_ensemble import (free_parameter_names,perturb_cell,propagate_metric_uncertainty,EnsembleInputError)
    assert free_parameter_names('monoclinic_unique_b')==('a','b','c','beta_deg')
    with pytest.raises(EnsembleInputError):perturb_cell(_cell0(),'cubic',{'a_shared':-1})
    spec={'parent_cell':_cell0(),'daughter_cell':(2.8,3.1,3.2,90.,90.,90.),
          'parent_family':'cubic','daughter_family':'orthorhombic',
          'measurement_covariance':np.eye(4)*1e-4}
    with pytest.raises(EnsembleInputError,match='distribution'):
        propagate_metric_uncertainty(spec,lambda p,d:None,n_samples=3)


def test_full_ensemble_branch_switching_not_averaged():
    from twin_app.p1_ensemble import propagate_metric_uncertainty
    from types import SimpleNamespace
    def mock_report(p,d):
        # Controlled full-pipeline stand-in: supports branch switching.
        b=SimpleNamespace(branch=1,classification='Type I',shear_magnitude=1.,habit_solutions=[],a_parent_cartesian=(.1,0,0),n_parent_cartesian=(1,0,0))
        if d[0]>3.:b.branch=2
        pair=SimpleNamespace(variant_i=1,variant_j=2,constructions=[b])
        f=SimpleNamespace(pair_records=[pair])
        return SimpleNamespace(families=[f])
    spec={'parent_cell':_cell0(),'daughter_cell':(3.,3.1,3.2,90.,90.,90.),
          'parent_family':'cubic','daughter_family':'orthorhombic',
          'measurement_covariance':np.diag([1e-4,.01,1e-4,1e-4]),
          'measurement_distribution':'multivariate_gaussian'}
    out=propagate_metric_uncertainty(spec,mock_report,n_samples=15,seed=12)
    assert out['status']=='INCONCLUSIVE_BRANCH_SWITCHING_OR_FAILED_SAMPLES'
    assert not out['shear_intervals_conditional']


def test_nonidentity_rational_correspondence_basis_invariance():
    G0=gram((3.2,3.2,3.2,90,90,90))
    G1=gram((4.5,5.1,3.8,90,107,90))
    C=np.array([[1.,.5,0.],[0.,1.,.3333333333333333],[0.,0.,1.]])
    P=np.array([[1,1,0],[0,1,1],[0,0,1]],float)
    U=stretch_from_metrics_independent(G0,G1,C)
    U2=stretch_from_metrics_independent(G0,P.T@G1@P,np.linalg.solve(P,C))
    assert np.linalg.norm(U-U2,'fro')<1e-12
    # Transform a reciprocal plane h'=P^T h. Cartesian plane cannot be
    # naively compared in different symmetric-root embedding gauges.
    old_h=np.array([1.,3.,7.]); new_h=P.T@old_h
    assert np.allclose(np.linalg.solve(P.T,new_h),old_h)


def test_root_completeness_positive_interior_and_deliberate_wrong_answer():
    Ui=np.diag([.8,.9,1.2]);a=np.array([.3,0,0]);n=np.array([1.,0.,0.])
    out=root_oracle(Ui,a,n)
    assert len(out['roots'])==1 and abs(out['roots'][0]-2/3)<1e-9
    assert out['status']=='MISMATCH'
    wrong=root_oracle(Ui,a,n,[.5])
    assert wrong['unmatched_oracle'] and wrong['unmatched_reported']
    correct=root_oracle(Ui,a,n,[2/3])
    assert correct['status']=='PASS_INTERIOR_ROOT_COVERAGE'


def test_root_continuum_must_not_be_passed_as_two_discrete_habits():
    out=root_oracle(np.eye(3),[.2,0.,0.],[0.,1.,0.])
    assert out['status']=='INCONCLUSIVE_CONTINUUM_OR_NEAR_ZERO_POLYNOMIAL'


def test_ensemble_detects_same_key_physically_different_tensor():
    from twin_app.p1_ensemble import propagate_metric_uncertainty
    from types import SimpleNamespace
    def mock_solver(p,d):
        shear_tensor=(1.,0,0) if d[0] < 3. else (0.,1.,0.)
        b=SimpleNamespace(branch=1,classification='Type I',shear_magnitude=.3,habit_solutions=[],
                          a_parent_cartesian=shear_tensor,n_parent_cartesian=(1.,0.,0.))
        return SimpleNamespace(families=[SimpleNamespace(pair_records=[SimpleNamespace(
           variant_i=1,variant_j=2,constructions=[b])])])
    spec={'parent_cell':_cell0(),'daughter_cell':(3.,3.1,3.2,90.,90.,90.),
          'parent_family':'cubic','daughter_family':'orthorhombic',
          'measurement_covariance':np.diag([1e-4,.01,1e-4,1e-4]),
          'measurement_distribution':'multivariate_gaussian'}
    out=propagate_metric_uncertainty(spec,mock_solver,n_samples=10,seed=18)
    assert not out['physical_tensor_tracking_under_5pct_heuristic']
    assert out['shear_intervals_conditional']=={}
