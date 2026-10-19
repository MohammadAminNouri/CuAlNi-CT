from __future__ import annotations

"""Optional, independent structure-data validation.

Neither Gemmi nor spglib is allowed to infer a parent→product correspondence.
The authoritative exact point-group and rank-one solvers are unchanged.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StructureReference:
    lengths: tuple[float, float, float]
    angles: tuple[float, float, float]
    atomic_positions: tuple[tuple[float, float, float], ...]
    atomic_numbers: tuple[int, ...]
    site_count: int
    filename: str
    # True when the CIF supplies a recognised symmetry model; not experimental proof.
    symmetry_model: str = ""
    # Partial-occupancy sites are not legitimate fully occupied spglib atoms.
    has_partial_occupancy: bool = False


def parse_cif(data: bytes, *, filename: str, max_bytes: int = 2_000_000) -> StructureReference:
    if not data or len(data) > max_bytes:
        raise ValueError("CIF must not be empty or exceed 2 MB")
    try:
        import gemmi
    except ImportError as exc:
        raise RuntimeError("Optional Gemmi CIF reader is not installed") from exc
    try:
        doc = gemmi.cif.read_string(data.decode("utf-8-sig"))
        structures = [gemmi.make_small_structure_from_block(block) for block in doc]
        structures = [struct for struct in structures if struct.sites]
        if len(structures) != 1:
            raise ValueError("Provide a CIF containing exactly one crystal with atom sites")
        structure = structures[0]
        # CIFs commonly store only the asymmetric unit. spglib accepts an
        # explicit full unit cell, not an unexpanded list of CIF sites.
        # Gemmi applies the CIF-declared symmetry including special positions.
        asymmetric_sites = tuple(structure.sites)
        if len(asymmetric_sites) > 20_000:
            raise ValueError("CIF asymmetric unit has too many atomic sites")
        if any(not (0.0 < float(site.occ) <= 1.0) for site in asymmetric_sites):
            raise ValueError("CIF contains a zero, negative or >1 occupancy")
        has_partial_occupancy = any(
            abs(float(site.occ) - 1.0) > 1.0e-8 for site in asymmetric_sites
        )
        sites = tuple(structure.get_all_unit_cell_sites())
        if len(sites) > 20_000:
            raise ValueError("Symmetry-expanded unit cell has too many atomic sites")
        numbers = tuple(int(site.element.atomic_number) for site in sites)
        if not numbers or any(x <= 0 for x in numbers):
            raise ValueError("Each atomic site requires a recognized chemical element")
        cell = structure.cell
        from .input_explainer import validate_metric_parameters
        lengths = (cell.a, cell.b, cell.c)
        angles = (cell.alpha, cell.beta, cell.gamma)
        validate_metric_parameters(lengths, angles)
        return StructureReference(
            lengths=lengths, angles=angles,
            atomic_positions=tuple((site.fract.x, site.fract.y, site.fract.z) for site in sites),
            atomic_numbers=numbers, site_count=len(sites), filename=filename,
            symmetry_model=str(structure.spacegroup_hm),
            has_partial_occupancy=has_partial_occupancy,
        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Invalid or unsupported CIF structure: {type(exc).__name__}") from exc


def independent_spglib_check(structure: StructureReference, *, symprec: float = 1e-5) -> dict[str, Any]:
    """Return evidence about a fully specified structure, never a guessed mapping."""
    if symprec <= 0:
        raise ValueError("symprec must be positive")
    if structure.has_partial_occupancy:
        raise ValueError(
            "Partial occupancies or disorder cannot be represented as fully occupied "
            "atoms in this spglib diagnostic; the CIF cell may still be imported"
        )
    try:
        import spglib
    except ImportError as exc:
        raise RuntimeError("Optional spglib symmetry validator is not installed") from exc
    import numpy as np
    a, b, c = structure.lengths
    alpha, beta, gamma = np.deg2rad(structure.angles)
    cg, cb, ca = np.cos(gamma), np.cos(beta), np.cos(alpha)
    sg = np.sin(gamma)
    basis = np.array([
        [a, 0., 0.],
        [b*cg, b*sg, 0.],
        [c*cb, c*(ca-cb*cg)/sg, 0.],
    ])
    basis[2, 2] = np.sqrt(max(c*c-basis[2, 0]**2-basis[2, 1]**2, 0.))
    result = spglib.get_symmetry_dataset(
        (basis, np.asarray(structure.atomic_positions), np.asarray(structure.atomic_numbers)),
        symprec=symprec,
    )
    if result is None:
        raise ValueError("spglib could not identify symmetry for this structure at the chosen tolerance")
    return {
        "international": str(result.international),
        "number": int(result.number),
        "pointgroup": str(result.pointgroup),
        "hall": str(result.hall),
        "symprec": symprec,
        "verification_scope": (
            "spglib consistency check on the "
            + ("CIF-declared symmetry-expanded unit cell" if structure.symmetry_model
               else "as-listed CIF sites (no declared symmetry; completeness not established)")
            + "; not independent experimental confirmation of atomic symmetry; "
            "no lattice correspondence inferred"
        ),
    }
