from __future__ import annotations

"""Explicit staggered hexagonal EDAX ANG research exporter.

Preserves the real HexGrid row parity and phase=0 unindexed mask, unlike the
square-grid exporter. The file is ANG-compatible with the project's reader.
Vendor acceptance and instrument-native metadata parity remain unverified.
"""
from io import StringIO
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from .lattice import Lattice
from .orientation_kernel import require_so3


def export_hex_parent_ang(
    ebsd, pixel_parent_id:object, parent_rotations:object,
    *, parent_lattice:Lattice, phase_name:str="Reconstructed parent hypothesis",
    geometry_tolerance:float=0.015,
)->str:
    if not isinstance(parent_lattice,Lattice):
        raise ValueError("Original indexed parent lattice must be explicitly supplied")
    ids=np.asarray(pixel_parent_id,int)
    rotations=np.asarray(parent_rotations,float)
    n=len(ebsd.phase_id)
    if (ids.shape!=(n,) or rotations.ndim!=3 or rotations.shape[1:]!=(3,3)
            or np.any(ids< -1) or np.any(ids>=len(rotations))):
        raise ValueError("Invalid parent-pixel IDs and rotation arrays")
    for r in rotations:
        require_so3(r,tolerance=2e-8,name="parent rotation")
    if not 0<geometry_tolerance<0.1:
        raise ValueError("Hexagonal-grid geometry tolerance invalid")
    x=np.asarray(ebsd.x,float)
    y=np.asarray(ebsd.y,float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("All hexagonal coordinates must be finite")
    if n<6:
        raise ValueError("Need multiple complete hexagonal rows")
    # Distinct y values are rows. This requires original quantized scanner grid
    # and fails closed rather than accidentally regrouping noisy XY coordinates.
    unique_y,rows=np.unique(y,return_inverse=True)
    if len(unique_y)<3:
        raise ValueError("Need at least three hexagonal scan rows")
    dy=np.diff(unique_y)
    y_step=float(np.median(dy))
    if y_step<=0 or np.max(abs(dy-y_step))>geometry_tolerance*y_step:
        raise ValueError("Hexagonal Y scan positions are not uniformly spaced")
    row_ids=[];row_x=[]
    for k in range(len(unique_y)):
        indices=np.flatnonzero(rows==k)
        ordered=indices[np.argsort(x[indices],kind="stable")]
        row_ids.append(ordered)
        row_x.append(x[ordered])
    odd_count=len(row_ids[0])
    even_count=len(row_ids[1])
    if odd_count<2 or even_count<2 or abs(odd_count-even_count)>1:
        raise ValueError("Expected alternating hexagonal scan rows differing by <=1 pixel")
    for k,row in enumerate(row_ids):
        expected=odd_count if k%2==0 else even_count
        if len(row)!=expected:
            raise ValueError("Hexagonal row-count alternation mismatch")
    dx=np.concatenate([np.diff(xx) for xx in row_x])
    x_step=float(np.median(dx))
    if x_step<=0 or np.max(abs(dx-x_step))>geometry_tolerance*x_step:
        raise ValueError("Nonuniform hexagonal X step")
    # EDAX HexGrid first and second rows must differ by HALF the X step.
    # If row lengths differ by one, the sign depends on which row is longer.
    first_offset=float(row_x[1][0]-row_x[0][0])
    if abs(abs(first_offset)-0.5*x_step)>geometry_tolerance*x_step:
        raise ValueError("Hexagonal row staggering is not half the X step")
    for k in range(2,len(row_x)):
        if abs(row_x[k][0]-row_x[k%2][0])>geometry_tolerance*x_step:
            raise ValueError("Inconsistent stagger direction between rows")
    # Source ordering is reconstructed from XY, not assumed from the input.
    ordered=np.concatenate(row_ids)
    valid=ids>=0
    matrices=np.broadcast_to(np.eye(3),(n,3,3)).copy()
    matrices[valid]=rotations[ids[valid]]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="Gimbal lock detected")
        euler=Rotation.from_matrix(matrices).as_euler("ZXZ")
    out=StringIO()
    out.write("# SOURCE: CONDITIONAL parent reconstruction; NOT vendor or experimental certification\n")
    out.write("# Phase 1\n")
    out.write(f"# MaterialName {phase_name.replace(chr(10),' ')[:80]}\n")
    out.write(f"# LatticeConstants {parent_lattice.a:.8g} {parent_lattice.b:.8g} {parent_lattice.c:.8g} "
              f"{parent_lattice.alpha_deg:.8g} {parent_lattice.beta_deg:.8g} {parent_lattice.gamma_deg:.8g}\n")
    out.write("# GRID: HexGrid\n")
    out.write(f"# XSTEP: {x_step:.10g}\n# YSTEP: {y_step:.10g}\n")
    out.write(f"# NCOLS_ODD: {odd_count}\n# NCOLS_EVEN: {even_count}\n")
    out.write(f"# NROWS: {len(row_ids)}\n")
    for i in ordered:
        angles=euler[i]
        # Standard ten columns: radians ZXZ, X, Y, IQ, CI, phase, SEM, fit.
        out.write(" ".join(f"{u:.11g}" for u in angles)
                  +f" {x[i]:.10g} {y[i]:.10g} 0 0 {int(valid[i])} 0 0\n")
    return out.getvalue()
