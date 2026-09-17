from __future__ import annotations

"""Cayron-style rotational crystallographic quaternions in non-Cartesian bases."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .lattice import metric_norm


def cross_tensor(M:np.ndarray)->np.ndarray:
    r"""Cross tensor X = sqrt(det M) M^-1."""
    M=np.asarray(M,float)
    return float(np.sqrt(np.linalg.det(M)))*np.linalg.inv(M)


@dataclass(frozen=True)
class CrystalQuaternion:
    scalar: float
    vector: np.ndarray

    def as_tuple(self): return self.scalar,self.vector


def quaternion_norm(q:CrystalQuaternion|tuple[float,np.ndarray],M:np.ndarray)->float:
    c,u=q if isinstance(q,tuple) else (q.scalar,q.vector)
    u=np.asarray(u,float).reshape(3)
    return float(np.sqrt(c*c+u@np.asarray(M,float)@u))


def normalize_quaternion(q:CrystalQuaternion|tuple[float,np.ndarray],M:np.ndarray)->CrystalQuaternion:
    c,u=q if isinstance(q,tuple) else (q.scalar,q.vector)
    N=quaternion_norm((c,u),M)
    if N<1e-15: raise ValueError("Zero quaternion")
    return CrystalQuaternion(float(c/N),np.asarray(u,float).reshape(3)/N)


def crystallographic_quaternion_product(
    q1:CrystalQuaternion|tuple[float,np.ndarray],q2:CrystalQuaternion|tuple[float,np.ndarray],M:np.ndarray
)->CrystalQuaternion:
    r"""Cayron product: c3=c1c2-u1^T M u2; u3=c2u1+c1u2+X(u1×u2)."""
    c1,u1=q1 if isinstance(q1,tuple) else (q1.scalar,q1.vector)
    c2,u2=q2 if isinstance(q2,tuple) else (q2.scalar,q2.vector)
    u1=np.asarray(u1,float).reshape(3);u2=np.asarray(u2,float).reshape(3);M=np.asarray(M,float)
    c3=float(c1*c2-u1@M@u2)
    u3=c2*u1+c1*u2+cross_tensor(M)@np.cross(u1,u2)
    return CrystalQuaternion(c3,u3)


def quaternion_conjugate(q:CrystalQuaternion)->CrystalQuaternion:
    return CrystalQuaternion(q.scalar,-np.asarray(q.vector,float))


def quaternion_inverse(q:CrystalQuaternion,M:np.ndarray)->CrystalQuaternion:
    N2=quaternion_norm(q,M)**2
    qc=quaternion_conjugate(q)
    return CrystalQuaternion(qc.scalar/N2,qc.vector/N2)


def quaternion_from_axis_angle(axis_crystal:np.ndarray,angle_deg:float,M:np.ndarray)->CrystalQuaternion:
    axis=np.asarray(axis_crystal,float).reshape(3); axis=axis/metric_norm(axis,M)
    h=np.deg2rad(angle_deg)/2
    return CrystalQuaternion(float(np.cos(h)),np.sin(h)*axis)


def axis_angle_from_quaternion(q:CrystalQuaternion,M:np.ndarray)->tuple[np.ndarray,float]:
    q=normalize_quaternion(q,M); c=float(np.clip(q.scalar,-1,1)); angle=2*np.arccos(c)
    s=np.sin(angle/2)
    if abs(s)<1e-12: return np.array([1.,0,0]),0.0
    axis=q.vector/s; axis=axis/metric_norm(axis,M)
    return axis,float(np.degrees(angle))


def quaternion_to_crystal_rotation(q:CrystalQuaternion,B:np.ndarray,M:np.ndarray)->np.ndarray:
    """Convert to a rotation matrix in crystal coordinates via a Cartesian realization B.

    B columns are physical Cartesian crystallographic basis vectors, B.T B=M.
    This is used mainly to validate the non-Cartesian quaternion algebra.
    """
    axis,ang=axis_angle_from_quaternion(q,M)
    axc=np.asarray(B,float)@axis; axc/=np.linalg.norm(axc)
    Rc=Rotation.from_rotvec(np.deg2rad(ang)*axc).as_matrix()
    return np.linalg.inv(B)@Rc@B
