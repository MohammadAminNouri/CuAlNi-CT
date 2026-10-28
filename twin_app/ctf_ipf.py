from __future__ import annotations

"""Proper phase-symmetry IPF colors using orix TSL/Laue color keys.

Crystal-to-sample EBSD rotations g are inverted for orix Orientation,
which maps sample directions to crystal directions. IPF is an orientation
visualization, not an independent Kikuchi reindexing or phase certificate.
"""
from pathlib import Path
import tempfile

import numpy as np

from .ctf_independent import declared_ctf_phases, MAX_CTF_BYTES


def symmetry_ipf_colors(file_bytes: bytes, filename: str, *,
                        phase_id: int, sample_direction: str = "Z",
                        three_dimensional_angle_unit: str | None = None):
    if sample_direction not in {"X","Y","Z"}:
        raise ValueError("Sample reference direction must be X, Y or Z")
    if Path(filename).name != filename or Path(filename).suffix.lower() != ".ctf":
        raise ValueError("Invalid CTF filename")
    if len(file_bytes)>MAX_CTF_BYTES:
        raise ValueError("CTF exceeds 35 MB")
    phases={p["phase_id"]:p for p in declared_ctf_phases(file_bytes)}
    if phase_id not in phases:
        raise ValueError("Selected indexed phase has no declared symmetry")
    # External IPFColorKeyTSL is an independently maintained crystallographic
    # Laue fundamental-sector implementation. Never fake colors if unavailable.
    from orix.quaternion import Orientation
    from orix.quaternion.symmetry import get_point_group
    from orix.plot import IPFColorKeyTSL
    from orix.vector import Vector3d
    from cualni_cryst.ebsd_map import AngleUnit
    from cualni_cryst.ebsd_io import load_ctf
    with tempfile.TemporaryDirectory(prefix="ctf-ipf-") as temp:
        path=Path(temp)/filename
        path.write_bytes(file_bytes)
        emap=load_ctf(
            path,three_dimensional_angle_unit=(
                AngleUnit(three_dimensional_angle_unit)
                if three_dimensional_angle_unit is not None else None
            ),
        )
    if not np.any(emap.indexed & (emap.phase_id==phase_id)):
        raise ValueError("No indexed pixels for selected phase")
    pointgroup=get_point_group(phases[phase_id]["space_group_number"])
    axis={"X":(1.,0.,0.),"Y":(0.,1.,0.),"Z":(0.,0.,1.)}[sample_direction]
    key=IPFColorKeyTSL(symmetry=pointgroup,direction=Vector3d(axis))
    active=emap.indexed & (emap.phase_id==int(phase_id))
    g=np.asarray(emap.orientations[active],dtype=float)
    # Native convention is g: crystal Cartesian -> sample Cartesian;
    # orix Orientation is sample -> crystal.
    orientations=Orientation.from_matrix(
        np.swapaxes(g,-1,-2),symmetry=pointgroup,
    )
    colors=np.full((emap.n_points,3),0.94,dtype=float)
    colors[active]=np.clip(key.orientation2color(orientations),0.,1.)
    if not np.all(np.isfinite(colors)):
        raise ValueError("IPF coloring generated nonfinite values")
    return {
        "x":np.asarray(emap.x),
        "y":np.asarray(emap.y),
        "z":np.asarray(emap.z),
        "rgb":colors,
        "active_mask":active,
        "phase_id":int(phase_id),
        "sample_direction":sample_direction,
        "point_group":pointgroup.name,
        "color_convention":"orix.IPFColorKeyTSL Laue, sample direction "+sample_direction,
        "crystal_frame":"CTF-declared conventional phase setting",
        "sample_frame":"CTF sample X/Y/Z (no unverified correction applied)",
        "status":"CONVENTION_VALID_IPF_PHASE_IDENTITY_NOT_INDEPENDENTLY_CERTIFIED",
    }


def plot_symmetry_ipf(result: dict, *, z_value: float | None = None):
    import plotly.graph_objects as go
    x,y,z=(np.asarray(result[k]) for k in ("x","y","z"))
    rgb=np.asarray(result["rgb"])
    if z_value is not None:
        match=np.isclose(z,z_value,atol=1.e-5,rtol=0)
        x,y,rgb=x[match],y[match],rgb[match]
    ux,ix=np.unique(np.round(x,6),return_inverse=True)
    uy,iy=np.unique(np.round(y,6),return_inverse=True)
    keys=iy.astype(int)*len(ux)+ix
    # A complete Cartesian grid need not be equally spaced. Image traces
    # use uniform dx/dy, so preserve nonuniform coordinates as point plots.
    regular_x=(len(ux)<3 or np.allclose(
        np.diff(ux),np.diff(ux)[0],rtol=1.e-5,atol=1.e-6
    ))
    regular_y=(len(uy)<3 or np.allclose(
        np.diff(uy),np.diff(uy)[0],rtol=1.e-5,atol=1.e-6
    ))
    unique_grid=(len(ux)*len(uy)==len(x)
                 and np.unique(keys).size==len(x)
                 and regular_x and regular_y)
    if unique_grid:
        grid=np.full((len(uy),len(ux),3),0.94)
        grid[iy,ix]=rgb
        trace=go.Image(z=np.rint(255*grid).astype(np.uint8),x0=float(ux[0]),
                       dx=float(np.median(np.diff(ux))) if len(ux)>1 else 1.,
                       y0=float(uy[0]),
                       dy=float(np.median(np.diff(uy))) if len(uy)>1 else 1.)
    else:
        codes=["rgb(%d,%d,%d)"%tuple(np.rint(255*v).astype(int)) for v in rgb]
        trace=go.Scattergl(x=x,y=y,mode="markers",marker=dict(
            color=codes,size=4,
        ))
    fig=go.Figure([trace])
    fig.update_layout(
        title="IPF-"+result["sample_direction"]+" · "+result["point_group"],
        height=510,margin=dict(l=20,r=20,t=42,b=28),
        xaxis_title="X (CTF coordinates)",yaxis_title="Y (CTF coordinates)",
    )
    fig.update_yaxes(autorange="reversed",scaleanchor="x",scaleratio=1)
    return fig
