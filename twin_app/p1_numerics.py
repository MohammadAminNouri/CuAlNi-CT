"""Material-independent P1 numerical diagnostics. No published alloy outputs used.

Checks explicitly separate mathematical residual, solver resolution, and experimental
uncertainty. Never turn a tiny algebraic residual into a precision claim about EBSD.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import acos, degrees, sqrt
from typing import Mapping, Sequence
import numpy as np


class NumericalUncertainty(ValueError): pass


def _spd(M,what):
    X=np.asarray(M,dtype=float)
    if X.shape!=(3,3) or not np.isfinite(X).all():raise NumericalUncertainty(f"{what}: finite 3x3 required")
    if np.linalg.norm(X-X.T,'fro')>1e-10*max(1.,np.linalg.norm(X,'fro')):raise NumericalUncertainty(f"{what}: not symmetric")
    eig=np.linalg.eigvalsh(X)
    if eig[0]<=0:raise NumericalUncertainty(f"{what}: must be positive definite")
    return (X+X.T)/2


def stretch_from_metrics_independent(G_parent,G_product,C):
    """U² = G_A^{-1/2} Cᵀ G_M C G_A^{-1/2} in the parent metric-root gauge.

    This is a geometry identity and independent of the active PTMC solver. Its
    Cartesian frame is the symmetric metric-root frame; do not compare its
    oriented vectors across incompatible gauge conventions without alignment.
    """
    A=_spd(G_parent,"parent metric");M=_spd(G_product,"product metric")
    C=np.asarray(C,float)
    if C.shape!=(3,3) or not np.isfinite(C).all() or abs(float(np.linalg.det(C)))<1e-15:
        raise NumericalUncertainty("Correspondence must be finite and nonsingular")
    w,V=np.linalg.eigh(A); Bm12=(V*(1/np.sqrt(w)))@V.T
    V2=Bm12@C.T@M@C@Bm12
    V2=(V2+V2.T)/2
    ev,E=np.linalg.eigh(_spd(V2,"U squared"))
    return (E*np.sqrt(ev))@E.T


def spectrum_diagnostics(U, *, numerical_tolerance=1e-8)->dict:
    V=_spd(U,"stretch U"); ev=np.linalg.eigvalsh(V)
    if not (0<numerical_tolerance<=1e-3):raise ValueError("Bad tolerance")
    scale=max(1.,max(map(abs,ev)))
    gaps=np.diff(ev)/scale
    near=bool(np.any(gaps<=numerical_tolerance))
    return {"principal_stretches":ev.tolist(),"detU":float(np.prod(ev)),
            "stretch_condition":float(ev[-1]/ev[0]),
            "minimum_relative_gap":float(np.min(gaps)),
            "middle_distance_to_one":float(abs(ev[1]-1)),
            "near_degenerate":near,
            "middle_stretch_near_one":bool(abs(ev[1]-1)<=numerical_tolerance*scale),
            "eigenvector_sensitivity_status":"NOT_UNIQUE_AT_DEGENERACY" if near else "SIMPLE_SPECTRUM_LOCAL_ONLY"}


def root_oracle(U,a,n,reported_fractions:Sequence[float]=(),*,precision=65,root_tolerance=1e-7)->dict:
    """Independent degree-6 determinant polynomial for det(F(λ)ᵀF(λ)-I).

    For an EXACT rank-one D=a⊗n the degree is ≤2 algebraically; the general
    3x3 polynomial is assembled independently instead of assuming this.
    Delicate endpoint, repeated, continuum or higher-degree outputs are flagged
    for investigation. Roots are filtered by the middle singular value condition.
    """
    from mpmath import mp
    U=_spd(U,"reference U")
    a=np.asarray(a,float);n=np.asarray(n,float)
    if a.shape!=(3,) or n.shape!=(3,) or not np.isfinite(a).all() or not np.isfinite(n).all() or np.linalg.norm(n)==0:
        raise NumericalUncertainty("Finite nonzero rank-one vectors required")
    if precision<35 or precision>150:raise ValueError("Precision must be 35..150 digits")
    if not 0<root_tolerance<1e-3:raise ValueError("Bad root tolerance")
    with mp.workdps(precision):
        V=[[mp.mpf(str(x)) for x in row] for row in U]
        aa=[mp.mpf(str(x)) for x in a];nn=[mp.mpf(str(x)) for x in n]
        D=[[aa[i]*nn[j] for j in range(3)] for i in range(3)]
        # Coefficients of (U + λD)^T(U + λD) - I.
        p=[]
        for i in range(3):
            row=[]
            for j in range(3):
                row.append([sum(V[k][i]*V[k][j] for k in range(3))-(1 if i==j else 0),
                            sum(V[k][i]*D[k][j]+D[k][i]*V[k][j] for k in range(3)),
                            sum(D[k][i]*D[k][j] for k in range(3))])
            p.append(row)
        def mul(x,y):
            z=[mp.mpf(0)]*(len(x)+len(y)-1)
            for i,v in enumerate(x):
                for j,w in enumerate(y):z[i+j]+=v*w
            return z
        def add(z,x,sign=1):
            for k,v in enumerate(x):z[k]+=sign*v
        coef=[mp.mpf(0)]*7
        for perm,sign in (((0,1,2),1),((1,2,0),1),((2,0,1),1),((2,1,0),-1),((1,0,2),-1),((0,2,1),-1)):
            add(coef,mul(mul(p[0][perm[0]],p[1][perm[1]]),p[2][perm[2]]),sign)
        magn=max(abs(v) for v in coef)
        # Algebraic rank-one law: degree ≤2 (higher coefficients cancel in mp arithmetic).
        normalized_high=float(max(abs(v) for v in coef[3:])/max(magn,mp.mpf('1e-60')))
        if normalized_high>1e-26:
            return {"status":"FAIL_NOT_DEGREE_TWO_RANK_ONE", "higher_degree_relative":normalized_high,
                    "roots":[],"unmatched_reported":list(reported_fractions)}
        low=list(coef[:3]);sc=max(map(abs,low))
        if sc<mp.mpf('1e-27'):
            return {"status":"INCONCLUSIVE_CONTINUUM_OR_NEAR_ZERO_POLYNOMIAL","roots":[],
                    "fraction_count":None,"max_reported_distance":None}
        # Root coefficients are NOT literature answers. Use mp quadratic solver.
        threshold=sc*mp.mpf('1e-27');roots=[]
        if abs(low[2])<threshold:
            if abs(low[1])<threshold:
                candidates=[]
            else:candidates=[-low[0]/low[1]]
        else:
            disc=low[1]*low[1]-4*low[2]*low[0]
            if disc<(-sc*sc*mp.mpf('1e-24')):candidates=[]
            else:
                if disc<0:disc=mp.mpf(0)
                sq=mp.sqrt(disc)
                candidates=[(-low[1]+sq)/(2*low[2]),(-low[1]-sq)/(2*low[2])]
        flags=[]
        for x in candidates:
            val=float(x)
            if not -root_tolerance<=val<=1+root_tolerance:continue
            X=U+val*np.outer(a,n)
            middle=float(np.linalg.svd(X,compute_uv=False)[1])
            if abs(middle-1)>max(root_tolerance*2,1e-9):continue
            if any(abs(val-y)<root_tolerance for y in roots):
                flags.append("DOUBLE_OR_CLOSE_ROOT");continue
            roots.append(val)
            if abs(val)<root_tolerance or abs(1-val)<root_tolerance:flags.append("ENDPOINT_ROOT")
        roots.sort()
        reported=[float(x) for x in reported_fractions]
        if any(not np.isfinite(x) or x<0 or x>1 for x in reported):raise NumericalUncertainty("Invalid reported lambda")
        unmatched_oracle=[x for x in roots if not any(abs(x-r)<=root_tolerance for r in reported)]
        unmatched_reported=[x for x in reported if not any(abs(x-r)<=root_tolerance for r in roots)]
        return {"status":"NEEDS_BOUNDARY_INVESTIGATION" if flags else
                "MISMATCH" if unmatched_oracle or unmatched_reported else "PASS_INTERIOR_ROOT_COVERAGE",
                "roots":roots,"reported":reported,"unmatched_oracle":unmatched_oracle,
                "unmatched_reported":unmatched_reported,"flags":sorted(set(flags)),
                "higher_degree_relative":normalized_high,
                "polynomial_coefficient_relative":[float(x/sc) for x in low]}


def phase_sensitivity(parent_cell,product_cell,C,*,gram_fn,allowed_parent,allowed_product,
                      sigma_parent=None,sigma_product=None, step_relative=2e-5)->dict:
    """Symmetry-constrained six-parameter phase sensitivity, never labels M as A.

    `allowed_*` specifies actual free lattice parameters (indices 0..5), e.g.
    cubic common length requires a SPECIAL linked perturbation performed by the
    caller, not independent a,b,c changes. This routine therefore raises for
    the cubic case unless the user provides a physically valid perturbation map.
    """
    pa=np.asarray(parent_cell,float);pr=np.asarray(product_cell,float)
    if pa.shape!=(6,) or pr.shape!=(6,):raise ValueError("Both phase cells need 6 parameters")
    names=('a','b','c','alpha_deg','beta_deg','gamma_deg')
    U0=stretch_from_metrics_independent(gram_fn(pa),gram_fn(pr),C)
    spectrum=spectrum_diagnostics(U0)
    if spectrum['near_degenerate']:
        return {"status":"DEGENERATE_EIGENVALUE_SENSITIVITY_NOT_UNIQUE",
                "baseline":spectrum,"derivatives":{},"uncertainty":None}
    derivatives={};J=[];valid=[]
    for prefix,base,indices in (("parent",pa,allowed_parent),("daughter",pr,allowed_product)):
        for idx in indices:
            idx=int(idx)
            if not 0<=idx<6:raise ValueError("Bad parameter index")
            h=step_relative*max(abs(base[idx]),1.)
            p=base.copy();q=base.copy();p[idx]+=h;q[idx]-=h
            try:
                values=[]
                for arr in (p,q):
                    gp=gram_fn(arr) if prefix=="parent" else gram_fn(pa)
                    gm=gram_fn(arr) if prefix=="daughter" else gram_fn(pr)
                    values.append(np.linalg.eigvalsh(stretch_from_metrics_independent(gp,gm,C)))
            except (ValueError,NumericalUncertainty):
                return {"status":"UNSTABLE_PERTURBATION_OR_INVALID_METRIC","baseline":spectrum,"derivatives":{},"uncertainty":None}
            d=(values[0]-values[1])/(2*h)
            derivatives[prefix+'_'+names[idx]]={"per_unit":d.tolist(),"units":"per angstrom" if idx<3 else "per degree"}
            J.append(d);valid.append((prefix,idx))
    # User-supplied sigma is a bound/standard deviation only if user declares statistical model.
    sig=[]
    for prefix,idx in valid:
        group=sigma_parent if prefix=="parent" else sigma_product
        if group is None:sig=[];break
        s=float(group[idx]);sig.append(s)
    unc=None
    if sig:
        if any(not np.isfinite(s) or s<0 for s in sig):raise ValueError("Bad uncertainties")
        unc=(np.sqrt(np.sum((np.asarray(J)*np.asarray(sig)[:,None])**2,axis=0))).tolist()
    return {"status":"LOCAL_METRIC_SENSITIVITY_ONLY_NOT_PHYSICAL_UNLESS_PERTURBATIONS_PRESERVE_SYMMETRY",
            "baseline":spectrum,"derivatives":derivatives,
            "independent_sigma_delta_method":unc,"parameters":valid,
            "warning":"Covariance and physical constraints are not inferred. Near boundaries rerun full nonlinear solver for samples."}


def symmetry_preserving_sensitivity(parent_cell,product_cell,C,*,parent_family,product_family,
                                   gram_fn,relative_step=1e-5,sigmas=None)->dict:
    """Local numerical derivatives only along crystallographically allowed metrics.

    'cubic': a=b=c; 'orthorhombic': a,b,c free, right angles fixed;
    'monoclinic_unique_b': a,b,c,beta free; 'triclinic': six free.
    Repeated eigenvalues are reported as unordered CLUSTERS, not false smooth
    eigenvalue derivatives. Uncertainties are only propagated if all sigma data
    are explicitly supplied; no statistical model is fabricated.
    """
    maps={'cubic':('a_shared',),'orthorhombic':('a','b','c'),
          'monoclinic_unique_b':('a','b','c','beta_deg'),
          'triclinic':('a','b','c','alpha_deg','beta_deg','gamma_deg')}
    if parent_family not in maps or product_family not in maps:
        raise NumericalUncertainty("Unknown phase family; cannot preserve metric symmetry")
    a=np.asarray(parent_cell,float);m=np.asarray(product_cell,float)
    if a.shape!=(6,) or m.shape!=(6,):raise NumericalUncertainty("Six phase parameters required")
    def check(cell,fam):
        tol=1e-6
        if fam=='cubic' and (np.max(np.abs(cell[:3]-cell[0]))>tol or np.max(np.abs(cell[3:]-90))>tol):
            raise NumericalUncertainty('Cubic metric parameter constraints violated')
        if fam=='orthorhombic' and np.max(np.abs(cell[3:]-90))>tol:
            raise NumericalUncertainty('Orthorhombic right-angle constraints violated')
        if fam=='monoclinic_unique_b' and max(abs(cell[3]-90),abs(cell[5]-90))>tol:
            raise NumericalUncertainty('Monoclinic unique-b constraints violated')
    check(a,parent_family);check(m,product_family)
    U=stretch_from_metrics_independent(gram_fn(a),gram_fn(m),C)
    baseline=spectrum_diagnostics(U)
    entries={}; columns=[];labels=[]
    for phase,orig,fam in [('parent',a,parent_family),('daughter',m,product_family)]:
        for pname in maps[fam]:
            idx={'a':0,'b':1,'c':2,'alpha_deg':3,'beta_deg':4,'gamma_deg':5,'a_shared':0}[pname]
            h=relative_step*max(abs(orig[idx]),1.)
            vals=[]
            for sign in (+1,-1):
                change=orig.copy()
                if pname=='a_shared':change[:3]+=sign*h
                else:change[idx]+=sign*h
                check(change,fam)
                pa=change if phase=='parent' else a
                pr=change if phase=='daughter' else m
                vals.append(np.linalg.eigvalsh(stretch_from_metrics_independent(gram_fn(pa),gram_fn(pr),C)))
            central=(vals[0]-vals[1])/(2*h)
            up=(vals[0]-np.linalg.eigvalsh(U))/h
            down=(np.linalg.eigvalsh(U)-vals[1])/h
            entry={'left_derivative':down.tolist(),'right_derivative':up.tolist(),
                   'central_sorted_derivative':central.tolist(),
                   'physically_allowed_parameter':phase+'_'+pname,
                   'units':'per angstrom' if idx<3 else 'per degree'}
            if baseline['near_degenerate']:
                entry['status']='ORDERED_EIGENVALUE_KINK_POSSIBLE; ONE-SIDED_DERIVATIVES_ONLY'
            elif np.linalg.norm(up-down)>1e-2*max(1.,np.linalg.norm(central)):
                entry['status']='NONLINEAR_OR_CROSSING; LINEAR_UNCERTAINTY_BLOCKED'
            else:
                entry['status']='LOCAL_DERIVATIVE_CONVERGED'
            key=phase+'_'+pname;entries[key]=entry;labels.append(key);columns.append(central)
    delta=None
    if sigmas is not None:
        if set(sigmas)!=set(labels):raise NumericalUncertainty('Uncertainty sigmas must exactly match symmetry-allowed parameters')
        if baseline['near_degenerate'] or any(e['status']!='LOCAL_DERIVATIVE_CONVERGED' for e in entries.values()):
            delta={'status':'BLOCKED_NONSmooth_OR_DEGENERATE'}
        else:
            sigma=np.array([float(sigmas[k]) for k in labels])
            if not np.isfinite(sigma).all() or np.min(sigma)<0:
                raise NumericalUncertainty('Invalid independent measurement sigmas')
            J=np.array(columns).T
            delta={'status':'LOCAL_INDEPENDENT_SIGMA_ONLY','sigma_principal_stretches':
                   np.sqrt((J*J)@(sigma*sigma)).tolist(),
                   'assumption':'Input parameter standard deviations independent, linear regime; NOT EBSD OR/phase uncertainty'}
    return {'status':'EIGENCLUSTER_NONSMOOTH' if baseline['near_degenerate'] else 'LOCAL_SYMMETRY_PRESERVING',
            'baseline':baseline,'derivatives':entries,'uncertainty':delta}
