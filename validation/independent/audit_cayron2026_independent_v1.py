#!/usr/bin/env python3
from __future__ import annotations

"""Independent audit of Cayron (Acta Materialia 316 (2026) 122399).

NO imports from CuAlNi-CT.  This script transcribes the paper's matrices and
relations directly and checks them with independent NumPy/SymPy algebra.

Checks:
  A. correspondence inverse convention;
  B. Table-3 / Eq.-41 numerical reproduction;
  C. O2 Type-I symbolic shear/shear audit;
  D. independent type-I cofactor (CCI) audit;
  E. O2 Type-II positive-control audit;
  F. O4 Appendix-C positive-control audit;
  G. common length-scale invariance for the disputed O2 Type-I diagnosis.
"""

import math
import numpy as np
import sympy as sp

np.set_printoptions(precision=12, suppress=True)


def metric(a: float, b: float, c: float, beta_deg: float, scale: float = 1.0) -> np.ndarray:
    beta = math.radians(beta_deg)
    a, b, c = a*scale, b*scale, c*scale
    return np.array([
        [a*a, 0.0, a*c*math.cos(beta)],
        [0.0, b*b, 0.0],
        [a*c*math.cos(beta), 0.0, c*c],
    ])


def smc(MA: np.ndarray, MM: np.ndarray, C_MA: np.ndarray) -> np.ndarray:
    # Paper Eq. 41 after using C^{A->M} = (C^{M->A})^{-1}
    CAM = np.linalg.inv(C_MA)
    return np.linalg.inv(MA) - CAM @ np.linalg.inv(MM) @ CAM.T


def cmc(MA: np.ndarray, MM: np.ndarray, C_MA: np.ndarray) -> np.ndarray:
    return C_MA.T @ MM @ C_MA - MA


def volume_ratio(MA: np.ndarray, MM: np.ndarray, C_MA: np.ndarray) -> float:
    # J = sqrt(det(C^T M_M C)/det(M_A))
    s1, l1 = np.linalg.slogdet(C_MA.T @ MM @ C_MA)
    s0, l0 = np.linalg.slogdet(MA)
    assert s1 > 0 and s0 > 0
    return float(np.exp(0.5*(l1-l0)))


def plane_unit(p: np.ndarray, M: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = math.sqrt(float(p @ np.linalg.inv(M) @ p))
    return p/n


def direct_unit(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    n = math.sqrt(float(v @ M @ v))
    return v/n


def projective_distance(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u,float); v=np.asarray(v,float)
    u=u/np.linalg.norm(u); v=v/np.linalg.norm(v)
    return float(min(np.linalg.norm(u-v),np.linalg.norm(u+v)))


# Paper correspondence matrices, quoted in Sec. 5.1.
C_AM = np.array([
    [0.0, 1.0, -1.0],
    [0.0, 1.0,  1.0],
    [1.0, 0.0,  0.0],
])
C_MA = np.array([
    [ 0.0, 0.0, 1.0],
    [ 0.5, 0.5, 0.0],
    [-0.5, 0.5, 0.0],
])

print("=== A. CORRESPONDENCE CONVENTION ===")
print("||C_AM*C_MA-I|| =", np.linalg.norm(C_AM @ C_MA - np.eye(3)))
assert np.allclose(C_AM @ C_MA, np.eye(3), atol=1e-15, rtol=0)
assert np.allclose(C_MA @ C_AM, np.eye(3), atol=1e-15, rtol=0)


print("\n=== B. TABLE-3 / EQ.-41 REPRODUCTION ===")
# Use the dimensional source precision printed before the rounded ratios.
a0 = 3.01
ar = 2.898/a0
cr = 4.646/a0
br = math.sqrt(2.0)  # C1 hypothetical state uses b=sqrt(2)
beta0 = 97.78
MA = np.eye(3)
MM = metric(ar, br, cr, beta0)
S = smc(MA, MM, C_MA)
m_plus = np.array([1.0, -1.0, 2.41966])
d_plus = S @ m_plus
print("d_A+ =", d_plus)
print("paper = [0.36938, -0.36938, -0.05378]")
assert np.allclose(d_plus, [0.36938,-0.36938,-0.05378], atol=7e-6, rtol=0)
print("PASS: independent Eq.-41 transcription reproduces the paper's Table-3 vector.")


print("\n=== C. O2 TYPE-I: EXACT SYMBOLIC AUDIT ===")
beta = sp.symbols('beta', positive=True, real=True)
a,c,b = sp.symbols('a c b', positive=True, real=True)
CMAs = sp.Matrix([[0,0,1],[sp.Rational(1,2),sp.Rational(1,2),0],[-sp.Rational(1,2),sp.Rational(1,2),0]])
CAMs = CMAs.inv()
MMs = sp.Matrix([
    [a**2, 0, a*c*sp.cos(beta)],
    [0, b**2, 0],
    [a*c*sp.cos(beta), 0, c**2],
])
SMCs = sp.eye(3) - CAMs*MMs.inv()*CAMs.T
m001 = sp.Matrix([0,0,1])
ds = sp.simplify(sp.trigsimp(SMCs*m001))
print("d_A for m=(001) =")
sp.pprint(ds)
expected_dz = 1 - 1/(a**2*sp.sin(beta)**2)
assert sp.simplify(sp.trigsimp(ds[2]-expected_dz)) == 0

# O2 type-I reflection and its intercorrespondence.
R001 = sp.diag(1,1,-1)
Cint = sp.simplify(CMAs*R001*CAMs)
s2 = sp.simplify(sp.trigsimp(sp.trace(Cint.T*MMs*Cint*MMs.inv())-3))
print("O2 type-I twin shear s^2 =")
sp.pprint(sp.factor(s2))
assert sp.simplify(sp.trigsimp(s2 - 4/sp.tan(beta)**2)) == 0

# With C1 and m=p=(001), the quadratic CMC contains the whole z=0 plane iff b=c=sqrt(2).
CMC_bc = sp.simplify((CMAs.T*MMs*CMAs-sp.eye(3)).subs({b:sp.sqrt(2),c:sp.sqrt(2)}))
x,y,z=sp.symbols('x y z', real=True)
q=sp.factor((sp.Matrix([x,y,z]).T*CMC_bc*sp.Matrix([x,y,z]))[0])
print("CMC quadratic with b=c=sqrt(2):")
sp.pprint(q)
assert sp.simplify(q.subs(z,0)) == 0

# Eq. 43 with m=p=(001): 2 d_A must equal a twin shear vector lying in (001), so d_z=0.
solution_eq43 = sp.solve(sp.Eq(expected_dz,0), a)
print("Eq.-43 positive condition from d_z=0: a =", solution_eq43)
assert 1/sp.sin(beta) in solution_eq43

paper_a = sp.sqrt(2)/sp.sin(beta)   # paper's printed c=sqrt(2), a=c/sin(beta)
candidate_a = 1/sp.sin(beta)        # algebraic closure of Eqs. 41+43
paper_d = sp.simplify(sp.trigsimp(ds.subs({a:paper_a,c:sp.sqrt(2),b:sp.sqrt(2)})))
candidate_d = sp.simplify(sp.trigsimp(ds.subs({a:candidate_a,c:sp.sqrt(2),b:sp.sqrt(2)})))
print("paper O2-I d_A =")
sp.pprint(paper_d)
print("equation-closure O2-I d_A =")
sp.pprint(candidate_d)
assert sp.simplify(paper_d[2]-sp.Rational(1,2)) == 0
assert sp.simplify(candidate_d[2]) == 0


print("\n=== D. INDEPENDENT BALL-JAMES / COFACTOR-I AUDIT ===")
# Paper states the equivalent CCI test for type I: ||U^{-1} e||=1,
# e = unit normal to the type-I twin plane.  Here e=(001), MA=I.
U2 = sp.simplify(CMAs.T*MMs*CMAs)
cci_sq = sp.simplify(sp.trigsimp(U2.inv()[2,2]))
print("||U^{-1} e_001||^2 =")
sp.pprint(cci_sq)
assert sp.simplify(cci_sq - 1/(a**2*sp.sin(beta)**2)) == 0
paper_cci = sp.simplify(sp.trigsimp(cci_sq.subs(a,paper_a)))
candidate_cci = sp.simplify(sp.trigsimp(cci_sq.subs(a,candidate_a)))
print("paper O2-I CCI squared =", paper_cci)
print("equation-closure CCI squared =", candidate_cci)
assert paper_cci == sp.Rational(1,2)
assert candidate_cci == 1

# SC3 for this O2-I family: tr(U^2)-det(U^2)-s^2/4-2 >= 0.
U2_bc = sp.simplify(U2.subs({b:sp.sqrt(2),c:sp.sqrt(2)}))
SC3 = sp.factor(sp.trigsimp(sp.trace(U2_bc)-U2_bc.det()-(4/sp.tan(beta)**2)/4-2))
print("SC3 left side for b=c=sqrt(2) =")
sp.pprint(SC3)
assert sp.simplify(SC3 - (a**2*sp.sin(beta)**2-1)/sp.tan(beta)**2) == 0
print("Interpretation: printed family passes SC1/C1 and SC3, but fails SC2/CCI exactly.")


print("\n=== E. O2 TYPE-II POSITIVE CONTROL ===")
# Paper: a=1, c=sqrt(2)/sin(beta), b=sqrt(2), m=(1,-1,0), d || [00-1],
# type-II rational axis e=[001].  CCII equivalent: ||U e||=1.
MM_II = MMs.subs({a:1,b:sp.sqrt(2),c:sp.sqrt(2)/sp.sin(beta)})
SMC_II = sp.simplify(sp.eye(3)-CAMs*MM_II.inv()*CAMs.T)
m110 = sp.Matrix([1/sp.sqrt(2),-1/sp.sqrt(2),0])
dII = sp.simplify(sp.trigsimp(SMC_II*m110))
U2II = sp.simplify(CMAs.T*MM_II*CMAs)
ccii_sq = sp.simplify(sp.trigsimp(U2II[2,2]))
print("paper O2-II d_A (unit m) =")
sp.pprint(dII)
print("paper O2-II ||U e_001||^2 =", ccii_sq)
assert dII[0] == 0 and dII[1] == 0
assert sp.simplify(dII[2] + 1/sp.tan(beta)) == 0
assert ccii_sq == 1
assert sp.simplify(sp.trigsimp(U2II.det())) == 1
print("PASS: the paper's O2 Type-II analytical family closes independently.")


print("\n=== F. O4 APPENDIX-C POSITIVE CONTROL AT beta=98 deg ===")
beta_deg = 98.0
B = math.radians(beta_deg)
r = math.sqrt(1 - 12*math.sqrt(2)/math.tan(B))
a4 = 0.5*math.sqrt(6 - 4*math.sqrt(2)/math.tan(B) - 2*r)
term1 = math.sqrt(6 - 4*math.sqrt(2)*math.cos(B)/math.sin(B) - 2*r)
term2 = math.sqrt(9 + 4*math.sqrt(2)*math.cos(B)*math.sin(B) + r + math.cos(2*B)*(3-r))
c4 = 0.5*(-math.cos(B)*term1 + term2)
print("Appendix-C a,c =",a4,c4," (paper ~0.8825, 1.6182)")
assert abs(a4-0.8825) < 2e-5 and abs(c4-1.6182) < 4e-5
MM4 = metric(a4,math.sqrt(2),c4,beta_deg)
CMC4 = cmc(np.eye(3),MM4,C_MA)
w,V = np.linalg.eigh(CMC4)
iz=int(np.argmin(np.abs(w))); ip=int(np.argmax(w)); ineg=int(np.argmin(w))
assert abs(w[iz]) < 2e-12 and w[ip] > 0 and w[ineg] < 0
planes=[]
for sign in (+1,-1):
    p = math.sqrt(w[ip])*V[:,ip] + sign*math.sqrt(-w[ineg])*V[:,ineg]
    planes.append(p/np.linalg.norm(p))

# O4 type-I reflection plane (01-1) in cubic parent.
p4 = np.array([0.0,1.0,-1.0]); p4/=np.linalg.norm(p4)
R4 = np.eye(3)-2*np.outer(p4,p4)
Cint4 = C_MA@R4@C_AM
Minv4=np.linalg.inv(MM4)
pM4=C_AM.T@p4   # C_MA^{-T} p_A = C_AM^T p_A
pM4=plane_unit(pM4,MM4)
nM4=Minv4@pM4
s4=math.sqrt(max(0.0,np.trace(Cint4.T@MM4@Cint4@Minv4)-3))
aM4=-(Cint4+np.eye(3))@nM4
aM4=direct_unit(aM4,MM4)
aA4=C_AM@aM4
aA4=aA4/np.linalg.norm(aA4)
twin4=s4*aA4
S4=smc(np.eye(3),MM4,C_MA)
resids=[]
for p in planes:
    d=S4@p
    lhs=2*float(p@p4)*d
    res=min(np.linalg.norm(lhs-twin4),np.linalg.norm(lhs+twin4))/s4
    resids.append(res)
print("O4 Eq.-43 normalized residuals for the two CMC habit planes =",resids)
assert min(resids) < 5e-12
U24=C_MA.T@MM4@C_MA
cci4=float(p4@np.linalg.inv(U24)@p4)
print("O4 independent CCI squared =",cci4)
assert abs(cci4-1.0) < 5e-12
print("PASS: Appendix-C O4 family closes CT Eq.-43 and independent CCI.")


print("\n=== G. LENGTH-SCALE INVARIANCE OF DISPUTED O2-I DIAGNOSIS ===")
beta_deg=97.78
B=math.radians(beta_deg)
ap=math.sqrt(2)/math.sin(B)
ac=1/math.sin(B)
for label,aval in (("paper",ap),("equation-closure",ac)):
    sig=[]
    for scale in (1e-6,1.0,1e6):
        MA=(scale**2)*np.eye(3)
        MM=metric(aval,math.sqrt(2),math.sqrt(2),beta_deg,scale=scale)
        # normalized physical d signature: with unit reciprocal m, SMC*m scales as 1/scale;
        # multiply by scale to compare dimensionlessly.
        m=plane_unit(np.array([0.,0.,1.]),MA)
        d=smc(MA,MM,C_MA)@m
        J=volume_ratio(MA,MM,C_MA)
        sig.append((scale*d[2],J))
    print(label, sig)
    ref=np.array(sig[1])
    for x in sig:
        assert np.allclose(x,ref,atol=5e-10,rtol=5e-10)

print("\n================ FINAL AUDIT VERDICT ================")
print("1) Core paper correspondence/SMC transcription reproduces Table 3: PASS")
print("2) Printed O2 Type-I c=sqrt(2), a=c/sin(beta): FAILS Eq.43 exactly")
print("3) The same printed O2 Type-I family: FAILS independent CCI exactly (1/2 vs 1)")
print("4) O2 Type-II analytical family from the paper: PASS")
print("5) O4 Appendix-C analytical family from the paper: PASS")
print("6) O2 Type-I algebra closes for b=c=sqrt(2), a=1/sin(beta)")
print("7) Diagnosis is invariant to common length-unit scaling: PASS")
print("Conclusion: evidence localizes the discrepancy to the printed O2 Type-I analytical relation, not to the general CT/SMC implementation.")
