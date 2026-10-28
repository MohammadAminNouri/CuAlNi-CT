from __future__ import annotations

"""3D plane from TWO registered nonparallel specimen sections.

A single 2D EBSD trace is only one linear constraint on a 3D plane normal.
Two nonparallel traces of the SAME physical interface in a common specimen
frame determine the unoriented normal up to sign; a CTF alone does not.
Cayron predicted K1/eta1 and a measured boundary plane remain separate.
"""
import numpy as np
from cualni_cryst.lattice import Lattice
from cualni_cryst.orientation_kernel import require_so3


def reconstruct_registered_interface_plane(
    trace_1, section_normal_1, trace_2, section_normal_2, *,
    g_crystal_to_sample, lattice: Lattice,
    trace_uncertainty_deg: float = 0.0,
) -> dict:
    def unit(v,name):
        a=np.asarray(v,dtype=float)
        if a.shape!=(3,) or not np.isfinite(a).all():
            raise ValueError(f"{name} must be three finite sample Cartesian components")
        norm=float(np.linalg.norm(a))
        if norm<=1.e-12:
            raise ValueError(f"{name} must be nonzero")
        return a/norm
    t1=unit(trace_1,"trace 1")
    t2=unit(trace_2,"trace 2")
    s1=unit(section_normal_1,"section normal 1")
    s2=unit(section_normal_2,"section normal 2")
    if abs(float(t1@s1))>1.e-3 or abs(float(t2@s2))>1.e-3:
        raise ValueError(
            "Each measured trace must lie in its own section plane "
            "(trace dot section normal must vanish)."
        )
    inter=float(np.linalg.norm(np.cross(s1,s2)))
    if inter<np.sin(np.deg2rad(5.)):
        raise ValueError(
            "Two section normals are nearly parallel: a unique 3D "
            "interface plane is not measured by these observations."
        )
    cp=np.cross(t1,t2)
    sinsep=float(np.linalg.norm(cp))
    if sinsep<np.sin(np.deg2rad(5.)):
        raise ValueError(
            "The observed traces are nearly parallel; the 3D normal "
            "is ill-conditioned and cannot be reported uniquely."
        )
    n=cp/sinsep
    g=require_so3(
        np.asarray(g_crystal_to_sample,dtype=float),
        tolerance=2.e-8,name="indexed grain crystal-to-sample orientation",
    )
    B=lattice.structure_matrix()
    # Plane covector h = B.T g.T n in conventional fractional basis.
    h=B.T@g.T@n
    h=h/np.max(np.abs(h))
    if not (0<=trace_uncertainty_deg<45 and np.isfinite(trace_uncertainty_deg)):
        raise ValueError("Trace uncertainty must lie in [0,45) degrees")
    # First-order geometric conditioning, reported as a conservative
    # amplification estimate (NOT a certified confidence interval).
    bound=min(180.,2.*trace_uncertainty_deg/sinsep)
    return {
        "status":"CONDITIONAL_TWO_SECTION_INTERFACE_PLANE",
        "plane_normal_sample":n.tolist(),
        "plane_normal_sample_sign_equivalence":"plus/minus",
        "hkl_reciprocal_covector_crystal_unrounded":h.tolist(),
        "angular_separation_of_trace_lines_deg":
            float(np.rad2deg(np.arcsin(np.clip(sinsep,0.,1.)))),
        "section_normals_sine_separation":inter,
        "trace_uncertainty_input_deg":trace_uncertainty_deg,
        "normal_uncertainty_first_order_bound_deg":bound,
        "classification":"UNASSIGNED_INTERFACE_NOT_CERTIFIED_TWIN_OR_HABIT",
        "limits":(
            "Both trace lines must belong to the same interface in "
            "two registered nonparallel physical sections. The "
            "crystal (hkl) is given as unrounded reciprocal coordinates "
            "using the selected indexed EBSD grain orientation. It "
            "does not identify Type I/II/Compound or twin shear."
        ),
    }
