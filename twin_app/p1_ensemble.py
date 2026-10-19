"""P1 opt-in input-uncertainty propagation through the FULL scientific engine.

The user MUST provide a measurement covariance in physical parameter units.
Source branch identity may change across perturbed metrics: those cases are
reported as TOPOLOGY UNSTABLE, not forcibly aligned or averaged.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable,Mapping
import numpy as np
from .p1_cualni import gram,CrystalInputError


class EnsembleInputError(ValueError):pass


def free_parameter_names(family):
    values={'cubic':('a_shared',), 'orthorhombic':('a','b','c'),
           'monoclinic_unique_b':('a','b','c','beta_deg'),
           'triclinic':('a','b','c','alpha_deg','beta_deg','gamma_deg')}
    if family not in values:raise EnsembleInputError('Unsupported crystallographic metric family')
    return values[family]


def perturb_cell(cell,family,changes:Mapping):
    x=np.array(cell,dtype=float,copy=True)
    keys=free_parameter_names(family)
    if x.shape!=(6,):raise EnsembleInputError('Six lattice parameters required')
    if set(changes)!=set(keys):raise EnsembleInputError('Exact symmetry-constrained independent metric parameter names required')
    ii={'a':0,'b':1,'c':2,'alpha_deg':3,'beta_deg':4,'gamma_deg':5}
    for name,value in changes.items():
        if name=='a_shared':x[:3]=float(value)
        else:x[ii[name]]=float(value)
    try:gram(x)
    except CrystalInputError as e:raise EnsembleInputError(str(e)) from e
    if family=='orthorhombic' and np.max(np.abs(x[3:]-90))>1e-8:raise EnsembleInputError('Orthorhombic angles invalid')
    if family=='monoclinic_unique_b' and max(abs(x[3]-90),abs(x[5]-90))>1e-8:raise EnsembleInputError('Monoclinic unique-b axes invalid')
    if family=='cubic' and (max(x[:3])-min(x[:3])>1e-9 or max(abs(x[3:]-90))>1e-8):raise EnsembleInputError('Cubic symmetry violation')
    return tuple(float(v) for v in x)


def measured_free_values(cell,family):
    x=np.asarray(cell,float)
    idx={'a':0,'b':1,'c':2,'alpha_deg':3,'beta_deg':4,'gamma_deg':5,'a_shared':0}
    return np.array([x[idx[n]] for n in free_parameter_names(family)],float)


def source_scalar_inventory(report):
    """Source labels can reorder; they are keys ONLY IF stable across samples."""
    result={}
    for fam in report.families:
        for pair in fam.pair_records:
            for b in pair.constructions:
                key=(int(pair.variant_i),int(pair.variant_j),int(b.branch))
                if key in result:raise EnsembleInputError('Source identity duplicated across families')
                D=np.outer(b.a_parent_cartesian,b.n_parent_cartesian)
                if not np.isfinite(D).all():raise EnsembleInputError('Invalid physical branch tensor')
                result[key]={'twin_tensor':D.tolist(),'shear':float(b.shear_magnitude),
                             'classification':str(b.classification),
                             'lambda':tuple(sorted(set(round(float(h.other_variant_volume_fraction),12) for h in b.habit_solutions)))}
    return result


def propagate_metric_uncertainty(spec:Mapping, solver:Callable,*,n_samples:int,seed:int=20261014):
    """Run full solver with user-supplied covariance, NEVER assumed uncertainties.

    `solver(parent_cell,daughter_cell)` returns the immutable scientific
    report; exact C and crystallographic settings must be held fixed. Numerical
    breakdown, invalid metrics and branch switching are retained as failures.
    """
    if not isinstance(n_samples,int) or not 2<=n_samples<=200:
        raise EnsembleInputError('2..200 samples per audit; research expensive')
    pa=spec['parent_cell'];da=spec['daughter_cell'];pf=spec['parent_family'];df=spec['daughter_family']
    pn=free_parameter_names(pf);dn=free_parameter_names(df)
    names=['parent_'+n for n in pn]+['daughter_'+n for n in dn]
    cov=np.asarray(spec.get('measurement_covariance'),float)
    if cov.shape!=(len(names),len(names)) or not np.isfinite(cov).all():
        raise EnsembleInputError('A finite covariance for all allowed metric parameters is REQUIRED')
    # np.allclose's default rtol=1e-5 permits substantial asymmetric
    # off-diagonals when covariances have large physical-unit magnitudes.
    # Use a strict, explicitly scaled Frobenius residual instead.
    covariance_scale=max(1.,float(np.linalg.norm(cov,ord='fro')))
    symmetry_residual=float(np.linalg.norm(cov-cov.T,ord='fro')/covariance_scale)
    if symmetry_residual>1e-12:
        raise EnsembleInputError('Measurement covariance must be symmetric (relative Frobenius residual <= 1e-12)')
    cov=(cov+cov.T)/2.
    eig=np.linalg.eigvalsh(cov)
    if eig[0]<-1e-12*max(1.,float(np.max(np.abs(eig)))):
        raise EnsembleInputError('Measurement covariance must be positive semidefinite')
    base=np.r_[measured_free_values(pa,pf),measured_free_values(da,df)]
    rng=np.random.default_rng(seed)
    # User explicitly declares Gaussian metric measurement errors; never inferred.
    if spec.get('measurement_distribution')!='multivariate_gaussian':
        raise EnsembleInputError('Specify measurement_distribution=multivariate_gaussian; no implicit probability model')
    baseline=source_scalar_inventory(solver(pa,da))
    outputs=[];failures=[]
    for i in range(n_samples):
        values=rng.multivariate_normal(base,cov,method='svd')
        try:
            p=perturb_cell(pa,pf,dict(zip(pn,values[:len(pn)])))
            d=perturb_cell(da,df,dict(zip(dn,values[len(pn):])))
            output=source_scalar_inventory(solver(p,d))
            outputs.append(output)
        except Exception as e:
            failures.append({'sample':i,'error_type':type(e).__name__,'message':str(e)[:250]})
    stable=all(set(x)==set(baseline) for x in outputs)
    # IDs and branch signs are not stable physical identifiers by themselves.
    # This numerical continuity gate is deliberately conservative: a tensor
    # change >5% aborts confidence intervals, without assuming why it moved.
    tracked=stable and all(all(
        np.linalg.norm(np.asarray(x[k]['twin_tensor'])-np.asarray(baseline[k]['twin_tensor']))/
        max(np.linalg.norm(np.asarray(x[k]['twin_tensor'])),
            np.linalg.norm(np.asarray(baseline[k]['twin_tensor'])),1e-14)<=0.05
        for k in baseline) for x in outputs)

    counts={'baseline':len(baseline),'sample_counts':[len(x) for x in outputs],
            'invalid_metric_or_solver_failures':len(failures)}
    classification_stable=all(all(v['classification']==baseline[k]['classification'] for k,v in x.items() if k in baseline) for x in outputs)
    shear_ranges={}
    if stable and tracked and classification_stable and len(outputs)==n_samples:
        for key,origin in baseline.items():
            values=np.array([x[key]['shear'] for x in outputs],float)
            shear_ranges[str(key)]={'baseline':origin['shear'],'p02_5':float(np.percentile(values,2.5)),
                                    'p97_5':float(np.percentile(values,97.5)),
                                    'range_max_minus_min':float(np.ptp(values))}
    return {'status':'NO_BASELINE_TWIN_BRANCHES_UNCERTAINTY_NOT_ESTIMABLE' if not baseline else
                    'FULL_SOURCE_INVENTORY_STABLE_CONDITIONAL_ON_C' if stable and tracked and classification_stable and not failures else
                    'INCONCLUSIVE_BRANCH_SWITCHING_OR_FAILED_SAMPLES',
            'parameter_names':names,'distribution_assumption':'USER_DECLARED_GAUSSIAN_COVARIANCE',
            'seed':seed,'requested_samples':n_samples,'successful_samples':len(outputs),
            'classification_stable':classification_stable,'source_key_inventory_stable':stable,
            'covariance_symmetry_relative_residual':symmetry_residual,
            'physical_tensor_tracking_under_5pct_heuristic':tracked,
            'counts':counts,'failures':failures,'shear_intervals_conditional':shear_ranges,
            'warning':'The 5% tensor-drift gate is a computational tracking heuristic, not a physical uncertainty bound. Source IDs may fail to track the same physical variant after branch rearrangement. No automatic correspondence between swapped modes. Uncertainty in C, phase identity, OR, stacking faults and model inadequacy NOT included.'}
