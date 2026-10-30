from __future__ import annotations

"""Independent reciprocal-space, kinematic diffraction, and 3D cell geometry.

No atomic positions are guessed from an EBSD cell metric. Structure factors
need explicit fractional atomic sites and caller-supplied scattering amplitudes.
The output is single-scattering, not dynamical electron diffraction or a
certified simulation of the legacy ARPGE rendering.
"""
from itertools import product

import numpy as np

from .lattice import Lattice
from .orientation_kernel import require_so3


def _validate_sites(sites: object) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(sites, (list, tuple)) or not 1 <= len(sites) <= 2000:
        raise ValueError("Supply 1..2000 explicit atomic scattering sites")
    positions=[]; weights=[]
    for site in sites:
        if not isinstance(site, dict) or "fractional" not in site or "amplitude" not in site:
            raise ValueError("Each site needs fractional coordinates and scattering amplitude")
        pos=np.asarray(site["fractional"],float)
        amplitude=float(site["amplitude"])
        occupancy=float(site.get("occupancy",1.))
        if (pos.shape!=(3,) or not np.isfinite(pos).all()
            or not np.isfinite(amplitude) or amplitude<0
            or not np.isfinite(occupancy) or not 0<=occupancy<=1):
            raise ValueError("Invalid atomic fractional coordinates/amplitude/occupancy")
        positions.append(pos%1.)
        weights.append(amplitude*occupancy)
    return np.asarray(positions),np.asarray(weights)


def reciprocal_geometry(lattice:Lattice)->dict:
    """Independent direct/reciprocal duality audit using cycle/Angstrom units.

    B columns are Cartesian direct vectors. Bstar=B^-T, NOT 2*pi B^-T;
    the physical scattering vector is 2*pi times the returned reciprocal q.
    """
    if not isinstance(lattice,Lattice):
        raise ValueError("An explicit indexed crystal lattice is required")
    direct=lattice.structure_matrix()
    reciprocal=np.linalg.inv(direct).T
    identity=direct.T@reciprocal
    error=float(np.linalg.norm(identity-np.eye(3),"fro"))
    if error>2e-10:raise AssertionError("Direct and reciprocal bases inconsistent")
    return {"direct_basis":direct,"reciprocal_basis_cycles_per_angstrom":reciprocal,
            "duality_residual":error,"volume_angstrom3":float(np.linalg.det(direct)),
            "reciprocal_convention":"cycles/Angstrom; multiply by 2*pi for radians/Angstrom"}


def kinematic_structure_factors(
    hkl:object, sites:object
)->dict:
    """F_hkl=sum_j occupancy_j*f_j*exp(2*pi*i*hkl dot r_j).

    The amplitude f_j is user-supplied and must match the probe and |q|.
    Intensities |F|^2 are relative kinematic estimates, NOT absolute
    electron/neutron/X-ray intensities without validated scattering factors.
    """
    indices=np.asarray(hkl)
    if indices.ndim!=2 or indices.shape[1]!=3 or not 1<=len(indices)<=50000:
        raise ValueError("HKL must have shape (1..50000, 3)")
    if not np.isfinite(indices).all() or not np.all(indices==np.round(indices)):
        raise ValueError("HKL indices must be finite integers")
    idx=np.asarray(indices,int)
    positions,weights=_validate_sites(sites)
    phases=2j*np.pi*(idx@positions.T)
    amplitude=np.exp(phases)@weights
    intensity=np.abs(amplitude)**2
    return {"hkl":idx,"F_real":np.real(amplitude),"F_imag":np.imag(amplitude),
            "relative_intensity":intensity,"n_atomic_sites":len(positions),
            "model":"kinematic coherent sum, user-supplied scattering amplitudes"}


def electron_wavelength_angstrom(voltage_kv:float)->float:
    """Relativistic electron de Broglie wavelength for an accelerating potential."""
    v=float(voltage_kv)*1000
    if not np.isfinite(v) or not 1e3<=v<=1e6:
        raise ValueError("Voltage must be between 1 and 1000 kV")
    return float(12.2643247/np.sqrt(v*(1+0.978466e-6*v)))


def simulate_kinematic_diffraction(
    lattice:Lattice, crystal_to_sample:object, sites:object, *,
    voltage_kv:float=200, maximum_index:int=5,
    excitation_tolerance_cycles_per_angstrom:float=0.02,
    detector_distance_mm:float=100,
)->dict:
    """Bounded Ewald-sphere intersection and planar detector rays.

    Beam is specimen +z; projection in millimetres is ray intersection with
    detector z=distance. No foil thickness, dynamical multiple scattering,
    extinction length or absolute intensity is calculated.
    """
    if not 1<=maximum_index<=10:
        raise ValueError("Maximum reflection index must be 1..10")
    if not np.isfinite(excitation_tolerance_cycles_per_angstrom) or not 0<excitation_tolerance_cycles_per_angstrom<=0.5:
        raise ValueError("Invalid reciprocal excitation tolerance")
    if not np.isfinite(detector_distance_mm) or not 1<=detector_distance_mm<=10000:
        raise ValueError("Detector distance must be 1..10000 mm")
    g=require_so3(np.asarray(crystal_to_sample,float),tolerance=2e-8,
                 name="crystal to sample detector frame")
    reciprocal=reciprocal_geometry(lattice)["reciprocal_basis_cycles_per_angstrom"]
    hkl=np.asarray([x for x in product(range(-maximum_index,maximum_index+1),repeat=3)
                    if x!=(0,0,0)],dtype=int)
    q=(g@reciprocal@hkl.T).T
    wavelength=electron_wavelength_angstrom(voltage_kv)
    k=1/wavelength
    final=q+np.array([0.,0.,k])
    length=np.linalg.norm(final,axis=1)
    deviation=length-k
    allowed=(np.abs(deviation)<=excitation_tolerance_cycles_per_angstrom)&(final[:,2]>0)
    i=np.flatnonzero(allowed)
    factors=kinematic_structure_factors(hkl[i],sites)
    detector=detector_distance_mm*final[i,:2]/final[i,2,None]
    return {
        "hkl":hkl[i],"q_sample_cycles_per_angstrom":q[i],
        "excitation_error_cycles_per_angstrom":deviation[i],
        "detector_xy_mm":detector,
        "relative_intensity":factors["relative_intensity"],
        "electron_wavelength_angstrom":wavelength,
        "voltage_kv":float(voltage_kv),
        "model":"kinematic explicit-site Ewald intersection; dynamic diffraction NOT simulated",
        "physical_structure_validated":False,
    }


def cell_and_reciprocal_3d(
    lattice:Lattice, crystal_to_sample:object, *,
    reciprocal_scale_angstrom2:float=1,
)->dict:
    """Twelve true 3D unit-cell edges and basis axes, plus reciprocal geometry.

    Returned vector coordinates can be plotted with Plotly using specimen
    orientation; duality identity validates the orthogonal/monoclinic geometry.
    """
    g=require_so3(np.asarray(crystal_to_sample,float),tolerance=2e-8,
                  name="specimen orientation")
    if not np.isfinite(reciprocal_scale_angstrom2) or reciprocal_scale_angstrom2<=0:
        raise ValueError("Reciprocal display scale must be finite and positive")
    geo=reciprocal_geometry(lattice)
    B=g@geo["direct_basis"]
    Bstar=g@geo["reciprocal_basis_cycles_per_angstrom"]
    corners=np.asarray(list(product([0,1],repeat=3)),int)
    xyz=(B@corners.T).T
    edges=np.asarray([(i,j) for i,a in enumerate(corners)
                      for j,b in enumerate(corners)
                      if j>i and np.count_nonzero(a!=b)==1],int)
    return {
        "vertices_sample_angstrom":xyz,"edges_12":edges,
        "direct_axes_sample_angstrom":B.T,
        "reciprocal_axes_sample_cycles_per_angstrom":Bstar.T,
        "reciprocal_axes_scaled_for_plot":(Bstar*reciprocal_scale_angstrom2).T,
        "original_direct_reciprocal_duality":geo["duality_residual"],
        "scientific_status":"exact cell geometry from declared metric, not atomically verified 3D structure",
    }
