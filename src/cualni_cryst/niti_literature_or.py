from __future__ import annotations

"""Published NiTi B2/B19' plane-and-direction OR benchmark.

The natural OR (AQ) and nearby alternative ideal OR A/C come from:
C. Cayron (2020), Crystals 10(7), 562,
https://doi.org/10.3390/cryst10070562

All direct crystallographic directions use the cell structure matrix B and
reciprocal plane normals use B^{-T}; treating Miller indices as Cartesian
vectors gives incorrect orientations for monoclinic B19'.

Q maps martensite crystal-Cartesian vectors to parent crystal-Cartesian
vectors. It is NOT an atomic correspondence or specimen orientation.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .lattice import Lattice
from .orientation_kernel import require_so3

NITI_LITERATURE_URL = "https://doi.org/10.3390/cryst10070562"
NITI_LATTICE_SOURCE = "archived ARPGE 2021 NiTi phase record; cell compatible with Cayron 2020"


def _unit(v: np.ndarray, label: str) -> np.ndarray:
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError(f"{label} must have three finite components")
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        raise ValueError(f"{label} cannot be zero")
    return v / norm


def direction_cartesian(lattice: Lattice, uvw: object) -> np.ndarray:
    return _unit(lattice.structure_matrix() @ np.asarray(uvw, float), "direction")


def plane_normal_cartesian(lattice: Lattice, hkl: object) -> np.ndarray:
    return _unit(np.linalg.solve(lattice.structure_matrix().T,
                               np.asarray(hkl, float)), "reciprocal plane normal")


def or_from_plane_and_direction(
    parent_lattice: Lattice,
    daughter_lattice: Lattice,
    parent_plane_hkl: object,
    daughter_plane_hkl: object,
    parent_direction_uvw: object,
    daughter_direction_uvw: object,
    *, coplanarity_tolerance: float = 1e-7,
) -> np.ndarray:
    """Construct proper Q(parent <- daughter) from two *signed* parallelisms.

    Directions must lie IN their respective planes. A plane is a reciprocal
    covector, not a direct-space [hkl] direction. The resulting three-axis
    frame has no unconstrained twist if the two constraints are independent.
    """
    def triad(lattice: Lattice, hkl: object, uvw: object) -> np.ndarray:
        n = plane_normal_cartesian(lattice, hkl)
        d = direction_cartesian(lattice, uvw)
        if abs(float(n @ d)) > coplanarity_tolerance:
            raise ValueError(
                f"Provided direction is not in its reciprocal (hkl) plane "
                f"(normal dot direction={float(n@d):.6g})"
            )
        d = _unit(d - (d @ n) * n, "in-plane direction")
        return np.column_stack((d, n, np.cross(d, n)))
    a = triad(parent_lattice, parent_plane_hkl, parent_direction_uvw)
    m = triad(daughter_lattice, daughter_plane_hkl, daughter_direction_uvw)
    return require_so3(a @ m.T, tolerance=2e-8, name="OR from plane/direction")


def niti_published_or(
    relationship: str = "AQ",
    *,
    parent: Lattice | None = None,
    daughter: Lattice | None = None,
) -> dict:
    """Literature-defined ideal plane/direction OR on an explicit NiTi cell.

    The names A, AQ and C are NOT synonyms. Different model metrics change
    Q; variant and physical interpretation remain conditional on phase
    indexing and diffraction evidence. No EBSD from the published paper
    is bundled or compared here.
    """
    parent = parent or Lattice.cubic(3.015, label="NiTi B2")
    daughter = daughter or Lattice.monoclinic_unique_b(
        2.889, 4.120, 4.622, 96.8, label="NiTi B19-prime",
    )
    if (abs(parent.a-parent.b)>1e-5 or abs(parent.b-parent.c)>1e-5
            or abs(parent.alpha_deg-90)>1e-5 or abs(parent.beta_deg-90)>1e-5
            or abs(parent.gamma_deg-90)>1e-5):
        raise ValueError("NiTi B2 reference must be cubic")
    if (abs(daughter.alpha_deg-90)>1e-5 or abs(daughter.gamma_deg-90)>1e-5
            or not 90 < daughter.beta_deg < 110):
        raise ValueError("NiTi B19' uses the unique-b monoclinic beta setting")
    if not (2.8<daughter.a<3.1 and 4.0<daughter.b<4.3
            and 4.4<daughter.c<4.9):
        raise ValueError("Cell is not in supported NiTi B19' benchmark metric range")
    pairs = {
        "AQ": ((1,0,1), (-1,1,1)),   # natural dense plane/direction
        "A": ((0,0,1), (1,-1,0)),
        "C": ((1,0,0), (0,0,1)),
    }
    if relationship not in pairs:
        raise ValueError("Published NiTi OR must be one of AQ, A, C")
    daughter_dir, parent_dir = pairs[relationship]
    Q = or_from_plane_and_direction(
        parent, daughter, (1,1,0), (0,1,0),
        parent_dir, daughter_dir,
    )
    # Two genuinely independent geometric constraints.
    p_d = direction_cartesian(parent, parent_dir)
    m_d = direction_cartesian(daughter, daughter_dir)
    p_n = plane_normal_cartesian(parent, (1,1,0))
    m_n = plane_normal_cartesian(daughter, (0,1,0))
    direction_error = float(np.rad2deg(np.arccos(np.clip(p_d @ (Q @ m_d),-1,1))))
    normal_error = float(np.rad2deg(np.arccos(np.clip(p_n @ (Q @ m_n),-1,1))))
    return {
        "material":"NiTi",
        "transformation":"B2 cubic -> B19-prime monoclinic",
        "or_label":relationship,
        "reference":NITI_LITERATURE_URL,
        "parent_b2_plane_hkl":[1,1,0],
        "daughter_b19p_plane_hkl":[0,1,0],
        "parent_b2_direction_uvw":list(parent_dir),
        "daughter_b19p_direction_uvw":list(daughter_dir),
        "Q_parent_from_daughter":Q.tolist(),
        "p2c_daughter_from_parent":Q.T.tolist(),
        "quaternion_xyzw_parent_from_daughter":Rotation.from_matrix(Q).as_quat().tolist(),
        "plane_normal_alignment_error_deg":normal_error,
        "inplane_direction_alignment_error_deg":direction_error,
        "scientific_status":"literature ideal OR geometry; independent measured NiTi EBSD not supplied",
    }
