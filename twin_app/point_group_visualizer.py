from __future__ import annotations

"""Geometric explanation of a selected *actual* symmetry operation.

A direct axis [uvw] and a reciprocal plane (hkl) are embedded with their
correct metric duals; this is not a stereographic projection nor a full
3D rendering of every group element.
"""

from math import cos, radians, sin, sqrt
from typing import Any

import numpy as np


def cell_basis(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Conventional direct-cell basis in an arbitrary right-handed Cartesian frame."""
    al, be, ga = map(radians, (alpha, beta, gamma))
    sg = sin(ga)
    if abs(sg) <= 1e-12:
        raise ValueError("Cell angle gamma produces a singular Cartesian basis")
    cx = c * cos(be)
    cy = c * (cos(al) - cos(be)*cos(ga)) / sg
    cz2 = c*c-cx*cx-cy*cy
    if cz2 <= 1e-15:
        raise ValueError("Cell angles do not define a positive-volume lattice")
    B = np.array([[a,b*cos(ga),cx],[0.,b*sg,cy],[0.,0.,sqrt(cz2)]], dtype=float)
    if float(np.linalg.det(B)) <= 0:
        raise ValueError("Lattice basis must have positive cell volume")
    return B


def element_geometry(
    basis: np.ndarray,
    kind: str,
    axis_or_plane: tuple[int,int,int] | None,
) -> tuple[str, np.ndarray] | None:
    """Return unit Cartesian axis or unit plane normal with correct dual frame."""
    if axis_or_plane is None:
        return None
    coefficients = np.asarray(axis_or_plane,dtype=float).reshape(3)
    if "rotation" in kind:
        cart = basis @ coefficients
        label = "direct rotation axis [uvw]"
    elif "mirror" in kind:
        cart = np.linalg.solve(basis.T,coefficients)
        label = "reciprocal mirror-plane normal (hkl)"
    else:
        return None
    norm = float(np.linalg.norm(cart))
    if norm < 1e-12:
        raise ValueError("Symmetry element direction must be nonzero")
    return label, cart/norm


def make_operation_scene(
    basis: np.ndarray,
    kind: str,
    axis_or_plane: tuple[int,int,int] | None,
    *,
    theme: str = "dark",
) -> Any:
    import plotly.graph_objects as go
    B = np.asarray(basis,dtype=float)
    if B.shape != (3,3):
        raise ValueError("Expected conventional cell 3x3 basis")
    pts = {(i,j,k): B@np.array([i,j,k],dtype=float) for i in (0,1) for j in (0,1) for k in (0,1)}
    line_coords=[]
    for vertex,p in pts.items():
        for axis in range(3):
            if vertex[axis]==0:
                nei=list(vertex);nei[axis]=1
                q=pts[tuple(nei)]
                line_coords.append((p,q))
    fig=go.Figure()
    for p,q in line_coords:
        fig.add_trace(go.Scatter3d(x=[p[0],q[0]],y=[p[1],q[1]],z=[p[2],q[2]],mode="lines",line=dict(color="#8394a4",width=4),hoverinfo="skip",showlegend=False))
    data=element_geometry(B,kind,axis_or_plane)
    if data is not None:
        role,n=data
        center=B@np.array([.5,.5,.5]);length=float(np.max(np.linalg.norm(B,axis=0)))*.75
        if "rotation" in kind:
            first,last=center-length*n,center+length*n
            fig.add_trace(go.Scatter3d(x=[first[0],last[0]],y=[first[1],last[1]],z=[first[2],last[2]],mode="lines+markers",line=dict(color="#75b7ef",width=10),marker=dict(size=3,color="#75b7ef"),name="Rotation axis [uvw]",showlegend=True))
        else:
            t=np.array([1.,0.,0.]) if abs(n[0])<.8 else np.array([0.,1.,0.])
            u=np.cross(n,t);u/=np.linalg.norm(u);v=np.cross(n,u)
            r=length*.62
            vertices=np.array([center+r*(sx*u+sy*v) for sx,sy in [(-1,-1),(1,-1),(1,1),(-1,1)]])
            fig.add_trace(go.Mesh3d(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],i=[0,0],j=[1,2],k=[2,3],opacity=.48,color="#75b7ef",name="Mirror plane (hkl)",showlegend=True))
    bg="#101922" if theme!="light" else "#ffffff"
    color="#f1f5f8" if theme!="light" else "#19354b"
    fig.update_layout(height=320,margin=dict(l=3,r=3,t=4,b=4),paper_bgcolor=bg,scene=dict(bgcolor=bg,xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False),aspectmode="data",camera=dict(eye=dict(x=1.6,y=1.5,z=1.4))),legend=dict(font=dict(color=color),x=0,y=1),font=dict(color=color),showlegend=data is not None)
    return fig
