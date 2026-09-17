from __future__ import annotations

"""EBSD-facing orientation utilities with conventions made explicit.

Internal convention:
    g maps a vector expressed in an orthonormal Cartesian frame attached to the
    crystal to sample Cartesian coordinates: v_sample = g @ v_crystal_cart.

Vendor files must be converted to this convention at the I/O boundary.  The
module intentionally does not guess whether a vendor stores sample->crystal or
crystal->sample matrices.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .lattice import Lattice, plane_normal_cartesian


def bunge_euler_to_g(phi1_deg:float,Phi_deg:float,phi2_deg:float)->np.ndarray:
    """Intrinsic ZXZ Euler construction, explicitly adopted as this package's Bunge adapter."""
    return Rotation.from_euler('ZXZ',[phi1_deg,Phi_deg,phi2_deg],degrees=True).as_matrix()


def g_to_bunge_euler(g:np.ndarray)->np.ndarray:
    return Rotation.from_matrix(np.asarray(g,float)).as_euler('ZXZ',degrees=True)


def crystal_symmetry_to_cartesian(sym_crystal:list[np.ndarray],lattice:Lattice)->list[np.ndarray]:
    B=lattice.structure_matrix(); Bi=np.linalg.inv(B); out=[]
    for S in sym_crystal:
        R=B@np.asarray(S,float)@Bi
        # numerical orthogonalization only removes roundoff from non-orthogonal cell representation
        U,_,Vt=np.linalg.svd(R); R=U@Vt
        if np.linalg.det(R)<0:
            # EBSD orientation symmetry uses proper rotations only.
            continue
        out.append(R)
    return out


def rotation_axis_angle(R:np.ndarray)->tuple[np.ndarray,float]:
    rv=Rotation.from_matrix(np.asarray(R,float)).as_rotvec(); ang=np.linalg.norm(rv)
    if ang<1e-14: return np.array([1.,0,0]),0.0
    return rv/ang,float(np.degrees(ang))


@dataclass(frozen=True)
class MisorientationResult:
    angle_deg: float
    axis_sample: np.ndarray
    symmetry_index_1: int
    symmetry_index_2: int
    rotation: np.ndarray


def minimum_misorientation(g1:np.ndarray,g2:np.ndarray,product_symmetry_cart:list[np.ndarray])->MisorientationResult:
    """Minimum proper crystallographic misorientation over daughter symmetries."""
    g1=np.asarray(g1,float);g2=np.asarray(g2,float)
    best=None
    for i,S1 in enumerate(product_symmetry_cart):
        A=g1@S1
        for j,S2 in enumerate(product_symmetry_cart):
            B=g2@S2
            D=B@A.T
            axis,ang=rotation_axis_angle(D)
            if best is None or ang<best.angle_deg:
                best=MisorientationResult(ang,axis,i,j,D)
    assert best is not None
    return best


def minimum_misorientation_deg(g1:np.ndarray,g2:np.ndarray,product_symmetry_cart:list[np.ndarray])->float:
    return minimum_misorientation(g1,g2,product_symmetry_cart).angle_deg


@dataclass(frozen=True)
class VariantAssignment:
    variant_index: int|None
    best_deg: float
    second_best_deg: float
    ambiguous: bool
    all_residuals_deg: tuple[float,...]


def assign_variant(measured_g:np.ndarray,predicted_g:list[np.ndarray],product_symmetry_cart:list[np.ndarray],
                   max_angle_deg:float=5.0,ambiguity_gap_deg:float=0.5)->VariantAssignment:
    errors=[minimum_misorientation_deg(measured_g,p,product_symmetry_cart) for p in predicted_g]
    order=np.argsort(errors); bi=int(order[0]); best=float(errors[bi]); second=float(errors[int(order[1])]) if len(order)>1 else 180.0
    if best>max_angle_deg: return VariantAssignment(None,best,second,False,tuple(map(float,errors)))
    return VariantAssignment(bi,best,second,(second-best)<ambiguity_gap_deg,tuple(map(float,errors)))


@dataclass(frozen=True)
class OperatorAssignment:
    operator_index: int|None
    angle_residual_deg: float
    axis_residual_deg: float
    score_deg: float


def _undirected_axis_angle(a:np.ndarray,b:np.ndarray)->float:
    a=np.asarray(a,float);b=np.asarray(b,float);a/=np.linalg.norm(a);b/=np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(a@b),-1,1))))


def assign_operator_by_axis_angle(measured_rotation:np.ndarray,theoretical_rotations:list[np.ndarray],max_score_deg:float=5.0)->OperatorAssignment:
    """Match a measured disorientation to representative theoretical rotations.

    For a complete CT analysis, theoretical_rotations should first be generated
    from the entire orientation-operator double coset, not one arbitrary matrix.
    """
    ma,mang=rotation_axis_angle(measured_rotation); best=None
    for i,R in enumerate(theoretical_rotations):
        ax,ang=rotation_axis_angle(R); ar=abs(mang-ang); xr=_undirected_axis_angle(ma,ax); score=float(np.hypot(ar,xr))
        if best is None or score<best.score_deg: best=OperatorAssignment(i,ar,xr,score)
    assert best is not None
    return best if best.score_deg<=max_score_deg else OperatorAssignment(None,best.angle_residual_deg,best.axis_residual_deg,best.score_deg)


def plane_trace_direction(plane_normal_sample:np.ndarray,surface_normal_sample:np.ndarray)->np.ndarray:
    t=np.cross(surface_normal_sample,plane_normal_sample); n=np.linalg.norm(t)
    if n<1e-14: raise ValueError("Plane trace undefined: plane normal parallel to surface normal")
    return t/n


def predicted_plane_trace(p_crystal:np.ndarray,g:np.ndarray,lattice:Lattice,surface_normal_sample:np.ndarray)->np.ndarray:
    ncr=plane_normal_cartesian(p_crystal,lattice); ns=np.asarray(g,float)@ncr
    return plane_trace_direction(ns,np.asarray(surface_normal_sample,float))


def undirected_angle_deg(v1:np.ndarray,v2:np.ndarray)->float:
    a=np.asarray(v1,float);b=np.asarray(v2,float);a/=np.linalg.norm(a);b/=np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(float(a@b)),-1,1))))


def trace_residual_deg(predicted_trace:np.ndarray,measured_trace:np.ndarray)->float:
    return undirected_angle_deg(predicted_trace,measured_trace)
