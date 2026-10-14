"""Cu-Al-Ni crystallography provenance, metric and phase-setting safeguards.

Literature information is illustrative/validation evidence ONLY: it never sets a
correspondence matrix or provides runtime answers to a twin/habit solver.

Sources:
 - Chen et al., Micron 31 (2000) 17–25, DOI 10.1016/S0968-4328(99)00060-8.
 - Electron microscopy study of 2H and 18R martensites, J. Alloys Compd.
   417 (2006) 138–142, DOI 10.1016/j.jallcom.2005.09.049.
 - Pérez-Cerrato et al., Materials (2024) / PMC11677451, Cu-Al-Ni
   powder metallurgy/EBSD; phase metrics are cited from older diffraction.
 - Zárubová et al., Acta Mater. (2010), DOI (verify in bibliography).
No correspondence/OR is inferred from matching a chemical composition.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Mapping
from math import cos, radians, degrees, acos, sqrt
import numpy as np
import sympy as sp

@dataclass(frozen=True)
class ReferenceCell:
    reference_id: str
    phase: str
    lattice: tuple[float,float,float,float,float,float]
    point_group: str
    cell_setting: str
    source_url: str
    sample_warning: str
    correspondence: None = None
    measured_or: None = None
    interface_normals: None = None

# All six lengths in angstrom and angles in degrees. The two monoclinic
# numbers refer to DIFFERENT cells/measurements and are not interchangeable.
CELLS: dict[str,ReferenceCell] = {
 "CuAlNi_2H_diffraction_reported": ReferenceCell(
  "CuAlNi_2H_diffraction_reported","gamma1_prime_2H",
  (5.3424,4.2244,4.3896,90.,90.,90.),"mmm","orthorhombic conventional Pmmn",
  "https://pmc.ncbi.nlm.nih.gov/articles/PMC11677451/",
  "Metrics reproduced in a review from a cited diffraction study; do not mix with 18R or other alloy batches."),
 "CuAlNi_monoclinic_C2m_EBSD_reported": ReferenceCell(
  "CuAlNi_monoclinic_C2m_EBSD_reported","beta_prime_monoclinic",
  (13.817,5.2856,4.3987,90.,113.6,90.),"2/m","C2/m conventional unique-b (beta-angle)",
  "https://pmc.ncbi.nlm.nih.gov/articles/PMC11677451/",
  "Published EBSD indexing / neutron metric cited in a review; distinguish 18R and reduced-cell labels."),
 "CuAlNi_6M_James_Hane_2000": ReferenceCell(
  "CuAlNi_6M_James_Hane_2000","faulted_monoclinic_reduced_6M",
  (4.430,5.330,12.79,90.,90.,95.68),"2/m",
  "reduced cell, gamma-angle, NOT conventional unique-b",
  "https://aem.umn.edu/~james/research/Publications_Files/PDF/67.pdf",
  "Reduced 6M description, Cu-14 wt% Al-4 wt% Ni. Do not identify with the C2/m reference cell by numerical coincidence."),
}

class CrystalInputError(ValueError): pass


def gram(cell)->np.ndarray:
    a,b,c,alpha,beta,gamma=map(float,cell)
    args=np.asarray((a,b,c,alpha,beta,gamma),float)
    if not np.isfinite(args).all() or min(a,b,c)<=0 or not all(0<x<180 for x in (alpha,beta,gamma)):
        raise CrystalInputError("Finite positive lengths and angles in (0,180) required")
    ca,cb,cg=[cos(radians(x)) for x in (alpha,beta,gamma)]
    G=np.array([[a*a,a*b*cg,a*c*cb],[a*b*cg,b*b,b*c*ca],[a*c*cb,b*c*ca,c*c]],float)
    ev=np.linalg.eigvalsh(G)
    if ev[0]<=0 or ev[0]/ev[-1]<=1e-13:
        raise CrystalInputError("Non-positive or severely ill-conditioned lattice metric")
    return G


def metric_to_cell(G)->tuple[float,...]:
    G=np.asarray(G,float)
    if G.shape!=(3,3) or not np.isfinite(G).all() or np.linalg.norm(G-G.T)>1e-10*max(1.,np.linalg.norm(G)):
        raise CrystalInputError("Metric must be finite, 3x3 symmetric")
    if np.linalg.eigvalsh(G)[0]<=0:raise CrystalInputError("Metric must be positive definite")
    a,b,c=np.sqrt(np.diag(G))
    def angle(i,j,ii,jj):return degrees(acos(float(np.clip(G[i,j]/sqrt(G[ii,ii]*G[jj,jj]),-1,1))))
    return (float(a),float(b),float(c),angle(1,2,1,2),angle(0,2,0,2),angle(0,1,0,1))


# Proper unimodular basis permutation: new a=old a, new b=-old c,
# new c=old b. For an input gamma!=90, new beta=old gamma.
GAMMA_TO_UNIQUE_B=sp.Matrix([[1,0,0],[0,0,1],[0,-1,0]])
assert GAMMA_TO_UNIQUE_B.det()==1


def reduced_gamma_to_unique_b(cell)->dict:
    G=gram(cell);P=np.asarray(GAMMA_TO_UNIQUE_B,float)
    Gb=P.T@G@P
    new=metric_to_cell(Gb)
    if abs(new[3]-90)>1e-7 or abs(new[5]-90)>1e-7:
        raise CrystalInputError("Input is not a gamma-angle monoclinic cell")
    if np.linalg.norm(Gb-P.T@G@P)>1e-12:raise AssertionError("Basis metric mismatch")
    return {"new_cell":new,"basis_new_to_old":list(map(list,GAMMA_TO_UNIQUE_B.tolist())),
            "determinant":1,"volume_ratio":float(np.sqrt(np.linalg.det(Gb)/np.linalg.det(G))),
            "warning":"This is a BASIS CONVERSION, not a proof that two experimentally reported phases/cells are identical. Transform C and all symmetries coherently."}


def _validate_point_group(cell,group,phase):
    a,b,c,al,be,ga=cell
    if group=="mmm":
        if max(abs(al-90),abs(be-90),abs(ga-90))>1e-5:
            raise CrystalInputError("Orthorhombic mmm requires all angles 90 degrees in this conventional basis")
    elif group=="2/m":
        if max(abs(al-90),abs(ga-90))>1e-5:
            raise CrystalInputError("Monoclinic 2/m unique-b setting requires alpha=gamma=90; rebase reduced gamma-cell first")
    elif group in ("m-3m","1"):
        if group=="m-3m" and (max(abs(a-b),abs(a-c),abs(b-c))>1e-7*max(a,b,c) or max(abs(al-90),abs(be-90),abs(ga-90))>1e-6):
            raise CrystalInputError("m-3m is inconsistent with supplied conventional cubic cell")
    else:
        # Other supported crystallographic systems belong to the generic engine.
        pass


def check_cualni_specimen(spec:Mapping)->dict:
    """Fail closed on guessed phase, setting or C; never claim experiment from input presence.

    Required: phase_label, daughter_cell, daughter_point_group, cell_setting,
    parent_cell, parent_point_group. Optional source/correspondence/OR evidence.
    """
    phase=str(spec.get("phase_label","")).strip()
    if phase not in ("gamma1_prime_2H","beta_prime_monoclinic","faulted_monoclinic_reduced_6M"):
        raise CrystalInputError("Specify Cu-Al-Ni phase: gamma1_prime_2H, beta_prime_monoclinic, or reduced_6M; do not guess from Al content")
    if not spec.get("cell_setting"):
        raise CrystalInputError("Explicit martensite cell setting required")
    M=tuple(map(float,spec["daughter_cell"])); A=tuple(map(float,spec["parent_cell"]))
    Gm=gram(M);Ga=gram(A)
    mg=str(spec.get("daughter_point_group",""));pg=str(spec.get("parent_point_group",""))
    if phase=="gamma1_prime_2H" and mg!="mmm":raise CrystalInputError("Reported 2H conventional lattice has mmm point group")
    if phase=="beta_prime_monoclinic" and mg!="2/m":raise CrystalInputError("Monoclinic reference requires explicit 2/m point group")
    if phase=="faulted_monoclinic_reduced_6M" and "gamma" in str(spec["cell_setting"]).lower():
        raise CrystalInputError("Reduced gamma-angle cell must be rebased before use with conventional unique-b 2/m")
    _validate_point_group(M,mg,phase);_validate_point_group(A,pg,"parent")
    if spec.get("experimental_status") in ("validated","verified"):
        raise CrystalInputError("Experimental-validation status cannot be asserted by a user input string")
    correspondence=spec.get("correspondence_M_from_A")
    cdet=None
    if correspondence is not None:
        try:C=sp.Matrix([[sp.Rational(str(x)) for x in row] for row in correspondence])
        except Exception as e:raise CrystalInputError("Correspondence must have exact rational coefficients") from e
        if C.shape!=(3,3) or C.det()==0:raise CrystalInputError("Correspondence must be invertible 3x3")
        cdet=str(C.det())
    has_source=bool(str(spec.get("correspondence_source","")).strip())
    evidence_status=("LATTICE_ONLY" if correspondence is None else
                     "C_UNVERIFIED" if not has_source else
                     "C_DOCUMENTED_NOT_EXPERIMENTALLY_CORROBORATED")
    return {"phase":phase,"evidence_status":evidence_status,"correspondence_determinant":cdet,
            "parent_metric_condition":float(np.linalg.cond(Ga)),"daughter_metric_condition":float(np.linalg.cond(Gm)),
            "accepted_as_physical_prediction":False,
            "reason":"Even a documented C must be checked against phase setting, OR and independent EBSD/TEM interfaces before an experimentally validated claim."}
