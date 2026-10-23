"""Focused check for accidental binary64 representation splits.

Legacy published benchmarks must also be tested separately; these stand-alone
checks do not prove their full PTMC/habit-plane predictions.
"""
import numpy as np
import pytest
from cualni_cryst.variant_identity import match_stretch_orbit, pullback_orbit_keys


def _metric(a,b,c,ang=(90,90,90)):
    alpha,beta,gamma=np.deg2rad(ang)
    return np.array([[a*a,a*b*np.cos(gamma),a*c*np.cos(beta)],
                     [a*b*np.cos(gamma),b*b,b*c*np.cos(alpha)],
                     [a*c*np.cos(beta),b*c*np.cos(alpha),c*c]],float)


def _sqrt_spd(m):
    d,v=np.linalg.eigh(0.5*(m+m.T))
    return (v*np.sqrt(d))@v.T


def _cubic_ops():
    from itertools import permutations, product
    ops=[]
    for p in permutations(range(3)):
        for s in product([-1,1],repeat=3):
            q=np.zeros((3,3))
            for i,j in enumerate(p):q[i,j]=s[i]
            if round(np.linalg.det(q))==1:ops.append(q)
    return ops


def _check_orbit(parent, product, correspondence, expected, *, order='forward'):
    gs=_cubic_ops()
    if order=='reverse':
        gs=list(reversed(gs))
    elif order=='sorted':
        gs=sorted(gs,key=lambda g:tuple(-int(x) for x in g.reshape(-1)))
    elif order!='forward':
        raise ValueError(order)
    keys=pullback_orbit_keys(product,correspondence,enumerate(gs))
    parent_sqrt=_sqrt_spd(parent)
    product_sqrt=_sqrt_spd(product)
    parent_inv=np.linalg.inv(parent_sqrt)
    representatives=[]; representative_keys=[]; memberships=[]
    conditioning=max(1.0, float(np.linalg.cond(parent)), float(np.linalg.cond(product)))
    for i,g in enumerate(gs):
        f=product_sqrt@correspondence@np.linalg.inv(g)@parent_inv
        u=_sqrt_spd(f.T@f)
        j=match_stretch_orbit(u, representatives,generator_index=i,identity_key=keys[i],
                              existing_keys=representative_keys,conditioning=conditioning)
        if j is None:
            representatives.append(u)
            representative_keys.append(keys[i])
            memberships.append([i])
        else:
            memberships[j].append(i)
            assert np.linalg.norm(u-representatives[j],ord='fro') <= 3e-12
    assert len(representatives)==expected
    assert sum(map(len,memberships))==24
    return memberships


def test_published_orthorhombic_metrics_do_not_create_artificial_variants():
    a=_metric(5.836,5.836,5.836)
    m=_metric(4.382116368337481,5.3562808,4.222001371547299)
    c=np.array([[0,1,1],[1,0,0],[0,1,-1]],float)
    _check_orbit(a,m,c,6)


@pytest.mark.parametrize('eps,n',[(0.0,3),(1e-8,6),(1e-9,6),(1e-11,6),(1e-12,6)])
def test_distinct_resolvable_and_exact_degenerate_variants(eps,n):
    _check_orbit(np.eye(3),np.diag((.92-eps,.92+eps,1.17))**2,np.eye(3),n)


def test_published_james_hane_6m_metrics_do_not_create_artificial_variants():
    parent=_metric(5.836,5.836,5.836)
    product=_metric(4.430,5.330,12.79,ang=(90,95.68,90))
    correspondence=np.array([[0,1,1],[1,0,0],[0,1/3,-1/3]],float)
    members=_check_orbit(parent,product,correspondence,12)
    assert all(len(group)==2 for group in members)


def test_ambiguous_tiny_nonzero_separation_is_not_silently_merged():
    with pytest.raises(ValueError,match='UNRESOLVED_VARIANT_IDENTITY'):
        _check_orbit(np.eye(3),np.diag((.92-1e-14,.92+1e-14,1.17))**2,np.eye(3),6)


def test_inconsistent_metadata_fails_closed():
    U=np.eye(3)
    I=np.eye(3)
    D=np.diag((1.,1.,1.+1e-8))
    keys=pullback_orbit_keys(I,I,[(0,I),(1,D)])
    with pytest.raises(ValueError,match='UNRESOLVED_VARIANT_IDENTITY'):
        match_stretch_orbit(U,[U],generator_index=1,identity_key=keys[1],
                            existing_keys=[keys[0]])


@pytest.mark.parametrize('ordering',['forward','reverse','sorted'])
@pytest.mark.parametrize('scale',[0.01,1.0,100.0])
def test_published_2h_orbit_independent_of_order_and_unit_scale(scale,ordering):
    parent=_metric(5.836*scale,5.836*scale,5.836*scale)
    product=_metric(4.382116368337481*scale,5.3562808*scale,4.222001371547299*scale)
    correspondence=np.array([[0,1,1],[1,0,0],[0,1,-1]],float)
    members=_check_orbit(parent,product,correspondence,6,order=ordering)
    assert sorted(map(len,members))==[4]*6


@pytest.mark.parametrize('ordering',['forward','reverse'])
@pytest.mark.parametrize('scale',[0.01,1.0,100.0])
def test_published_6m_orbit_independent_of_order_and_unit_scale(scale,ordering):
    parent=_metric(5.836*scale,5.836*scale,5.836*scale)
    product=_metric(4.430*scale,5.330*scale,12.79*scale,ang=(90,95.68,90))
    correspondence=np.array([[0,1,1],[1,0,0],[0,1/3,-1/3]],float)
    members=_check_orbit(parent,product,correspondence,12,order=ordering)
    assert sorted(map(len,members))==[2]*12
