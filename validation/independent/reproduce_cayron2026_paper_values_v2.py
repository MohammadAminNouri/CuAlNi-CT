#!/usr/bin/env python3
"""Independently reproduce Cayron 2026 Table 2, Table 3 and Sec. 5.3.

Only standard library and numpy. Absolutely no import from twin_app or
cualni_cryst, and no solver output is used as a reference. JSON on stdout.
The published O2 Type-I lattice relation and the equation-derived
counterfactual are separate; do not silently replace either one.
"""
from __future__ import annotations
import json
import math
import numpy as np


def metric(a, b, c, beta_deg):
    co=math.cos(math.radians(beta_deg))
    return np.array(((a*a,0.,a*c*co),(0.,b*b,0.),(a*c*co,0.,c*c)))


def unit(v):
    return v/np.linalg.norm(v)


def compute():
    C=np.array(((0.,0.,1.),(.5,.5,0.),(-.5,.5,0.)))
    Ci=np.linalg.inv(C)
    assert np.allclose(C@Ci,np.eye(3),atol=1e-14)
    a,b,c=2.898/3.01,math.sqrt(2),4.646/3.01
    beta=97.78
    M=metric(a,b,c,beta)
    H=C.T@M@C-np.eye(3)
    vals,V=np.linalg.eigh(H)
    assert vals[0]<0 and abs(vals[1])<1e-12 and vals[2]>0
    S=np.eye(3)-Ci@np.linalg.inv(M)@Ci.T

    # O2 Type-I, parent mirror (001). Crystallographic shear is derived
    # from the intercorrespondence, not supplied from Cayron's table.
    R=np.diag((1.,1.,-1.))
    I=C@R@Ci
    shear=math.sqrt(max(0.,np.trace(I.T@M@I@np.linalg.inv(M))-3.))
    pA=np.array((0.,0.,1.))
    pM=Ci.T@pA
    pM/=math.sqrt(float(pM@np.linalg.inv(M)@pM))
    twin_dir_M=-(I+np.eye(3))@(np.linalg.inv(M)@pM)
    twin_dir_M/=math.sqrt(float(twin_dir_M@M@twin_dir_M))
    twin_dir_A=unit(Ci@twin_dir_M)
    tw_shear=shear*twin_dir_A

    habits=[]
    for sign in (1.,-1.):
        covector=math.sqrt(vals[2])*V[:,2]+sign*math.sqrt(-vals[0])*V[:,0]
        covector=unit(covector)
        miller=covector/covector[0]
        d=S@covector
        paper_scaled_d=S@miller
        # sign of twin direction is arbitrary: minimize both orientations
        rhs=2.*float(covector@pA)*d
        mismatch=min(np.linalg.norm(rhs-tw_shear),np.linalg.norm(rhs+tw_shear))/shear
        angle=math.degrees(math.acos(min(1.,abs(float(
            d@twin_dir_A/np.linalg.norm(d))))))
        habits.append({
            "miller":[float(x) for x in miller],
            "paper_scaled_IPS_shear":[float(x) for x in paper_scaled_d],
            "epsilon":float(mismatch),
            "angle_deg":float(angle),
        })
    plus=next(h for h in habits if h["miller"][2]>0)
    minus=next(h for h in habits if h["miller"][2]<0)
    actual_b=4.108/3.01
    k=2*a*a+c*c-a*a*c*c*math.sin(math.radians(beta))**2

    # Independent O2 Type-I analytic consistency at beta=98 degrees.
    B=math.radians(98.)
    sin=math.sin(B)
    printed_a=math.sqrt(2)/sin
    corrected_a=1/sin
    def o2_identity(av):
        MM=metric(av,math.sqrt(2),math.sqrt(2),98.)
        s=np.eye(3)-Ci@np.linalg.inv(MM)@Ci.T
        d_z=float((s@pA)[2])
        U2=C.T@MM@C
        type_i_cci_sq=float(np.linalg.inv(U2)[2,2])
        return {"a":float(av),"SMC_dz":d_z,"type_I_CCI_squared":type_i_cci_sq}

    # O2 Type-II analytic family: a=1,c=sqrt2/sin beta, b=sqrt2.
    Mii=metric(1.,math.sqrt(2),math.sqrt(2)/sin,98.)
    Sii=np.eye(3)-Ci@np.linalg.inv(Mii)@Ci.T
    dii=Sii@unit(np.array((1.,-1.,0.)))
    Uii=C.T@Mii@C

    # O4 Appendix C analytic reference at beta=98.
    r=math.sqrt(1-12*math.sqrt(2)/math.tan(B))
    a4=.5*math.sqrt(6-4*math.sqrt(2)/math.tan(B)-2*r)
    u=math.sqrt(6-4*math.sqrt(2)*math.cos(B)/math.sin(B)-2*r)
    v=math.sqrt(9+4*math.sqrt(2)*math.cos(B)*math.sin(B)
                +r+math.cos(2*B)*(3-r))
    c4=.5*(-math.cos(B)*u+v)
    return {
        "reference":"Cayron, Acta Materialia 316 (2026) 122399",
        "doi":"10.1016/j.actamat.2026.122399",
        "normalized_source_input":{
            "a":a,"b":b,"c":c,"beta_deg":beta,
            "source_dimensions_angstrom":{"a_B2":3.01,"a_B19p":2.898,"c_B19p":4.646},
        },
        "table_2":{
            "C1_equality_gap":float(abs(actual_b**2-2)),
            "C1_inequality_margin":float(k-2),
        },
        "table_3":{
            "lambda":[float(x) for x in np.sqrt(np.linalg.eigvalsh(C.T@M@C))],
            "O2_type_I_twin_shear":float(shear),
            "m_plus":plus,"m_minus":minus,
        },
        "O2_type_I":{
            "printed_relation":o2_identity(printed_a),
            "equation_derived_relation_not_published":o2_identity(corrected_a),
        },
        "O2_type_II":{
            "a":1.,"b":math.sqrt(2),"c":math.sqrt(2)/sin,
            "SMC_shear_vector":[float(x) for x in dii],
            "CCI_squared":float(Uii[2,2]),
        },
        "O4_appendix_C":{
            "beta_deg":98.,"a":a4,"b":math.sqrt(2),"c":c4,
        },
    }


if __name__=="__main__":
    print(json.dumps(compute(),indent=2,sort_keys=True))
