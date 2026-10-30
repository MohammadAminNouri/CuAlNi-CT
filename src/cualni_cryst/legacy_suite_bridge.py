from __future__ import annotations

"""Unified modern-Python interface modeled on the Cry/Gen/AR workflow.

Original legacy sources are NOT embedded here. The existing CuAlNi-CT exact
group, metric and EBSD engines are used, with explicit coordinate conventions.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy as sp

from .correspondence import Correspondence
from .ebsd_io import load_ang, load_ctf
from .ebsd_map import EBSDPhase
from .ebsd_reconstruction import VariantOR, reconstruct_parent
from .group_theory import correspondence_groupoid
from .lattice import Lattice
from .point_groups import point_group_operations, metric_preservation_residual
from .parent_variant_audit import daughter_variant_assignment


@dataclass(frozen=True)
class Phase:
    name: str
    point_group: str
    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0

    def lattice(self) -> Lattice:
        return Lattice(
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma,
            label=self.name, length_unit="angstrom"
        )

    def verified_symmetry(self) -> list[sp.Matrix]:
        group = list(point_group_operations(self.point_group))
        residual = metric_preservation_residual(tuple(group), self.lattice().metric())
        if residual > 1e-8:
            raise ValueError(
                f"{self.name}: point group {self.point_group} is not "
                f"compatible with the lattice metric (relative error {residual:.3g})"
            )
        return group


def exact_correspondence(text: str) -> Correspondence:
    """Nine exact integer/rational tokens, no evaluation of Python expressions."""
    lines = [x for x in text.strip().splitlines() if x.strip()]
    if len(lines) != 3:
        raise ValueError("C_(M<-A) must contain exactly three rows")
    values = []
    for line in lines:
        tokens = line.replace(",", " ").split()
        if len(tokens) != 3:
            raise ValueError("Every row of C must have three values")
        for token in tokens:
            if not 0 < len(token) < 32 or any(c not in "0123456789+-/" for c in token):
                raise ValueError("C entries must be exact integers or fractions")
            values.append(sp.Rational(token))
    return Correspondence(sp.Matrix(3, 3, values), label="user specified")


def crystal_geometry(phase: Phase, uvw: tuple[float, float, float],
                     hkl: tuple[float, float, float]) -> dict:
    """Metric-aware direct direction length and reciprocal d-spacing."""
    symmetry = phase.verified_symmetry()
    metric = phase.lattice().metric()
    u, h = np.asarray(uvw, float), np.asarray(hkl, float)
    if u.shape != (3,) or h.shape != (3,) or not np.isfinite(u).all() or not np.isfinite(h).all():
        raise ValueError("Expected finite three-component [uvw] and (hkl)")
    if np.linalg.norm(u) == 0 or np.linalg.norm(h) == 0:
        raise ValueError("Direction and plane must be nonzero")
    return {
        "metric_angstrom_squared": metric.tolist(),
        "direction_length_angstrom": float(np.sqrt(u @ metric @ u)),
        "d_spacing_angstrom": float(1.0 / np.sqrt(h @ np.linalg.solve(metric, h))),
        "point_group_order": len(symmetry),
    }


def variant_groupoid(parent: Phase, product: Phase, c_text: str) -> dict:
    """Gen-type exact cosets, double cosets, and operator adjacency."""
    corr = exact_correspondence(c_text)
    result = correspondence_groupoid(
        parent.verified_symmetry(), product.verified_symmetry(), corr
    )
    return {
        "subgroup_order": len(result.subgroup),
        "variants": result.n_variants,
        "operator_classes": result.n_operators,
        "burnside_count": result.burnside_count,
        "adjacency": result.adjacency,
        "inverse_classes": result.inverse_operators,
        "class_sizes": [len(operator) for operator in result.operators],
        "correspondence": [[str(e) for e in row] for row in corr.C_M_from_A.tolist()],
        "status": "theoretical correspondence classes; NOT an experimental OR",
    }


def inspect_ebsd(path: str | Path) -> tuple[object, dict]:
    """Load EBSD using existing strict readers; no fabricated OR reconstruction."""
    path = Path(path)
    if path.suffix.lower() == ".ctf":
        ebsd = load_ctf(path)
    elif path.suffix.lower() == ".ang":
        ebsd = load_ang(path)
    else:
        raise ValueError("Expected Oxford .ctf or EDAX .ang")
    phases, counts = np.unique(ebsd.phase_id, return_counts=True)
    report = {
        "points": int(len(ebsd.phase_id)),
        "indexed_points": int(np.count_nonzero(ebsd.indexed)),
        "phase_counts": {str(int(p)): int(n) for p, n in zip(phases, counts)},
        "orientation_convention": str(ebsd.metadata.get("orientation_convention", "not specified")),
        "status": "EBSD input only; parent reconstruction requires validated experimental ORs",
    }
    return ebsd, report


def reconstruct_ebsd_parent(
    ebsd: object,
    parent: Phase,
    product: Phase,
    phase_id: int,
    or_parent_from_product: np.ndarray,
    *,
    maximum_points: int = 20,
    tolerance_deg: float = 3.0,
) -> dict:
    """Exploratory robust reconstruction from EXPLICIT Cartesian OR variants.

    The OR matrices map product Cartesian crystal vectors to parent Cartesian
    crystal vectors. They must not be confused with C_(M<-A) in direct lattice
    coordinates. The existing robust parent solver is used without guessed ORs.
    This does not segment complete grains or validate the supplied OR.
    """
    matrices = np.asarray(or_parent_from_product, float)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("OR input must have shape (N,3,3)")
    if not 1 <= len(matrices) <= 6:
        raise ValueError("Provide 1 to 6 verified Cartesian OR variants")
    if not 2 <= maximum_points <= 96:
        raise ValueError("maximum_points must be between 2 and 96")
    if not 0 < tolerance_deg <= 15:
        raise ValueError("Tolerance must be in (0,15] degrees")
    indices = np.flatnonzero(ebsd.indexed & (ebsd.phase_id == int(phase_id)))
    if len(indices) < 2:
        raise ValueError("At least two indexed product-phase points are required")
    if len(indices) > maximum_points:
        indices = indices[np.linspace(0, len(indices)-1, maximum_points, dtype=int)]
    variants = [
        VariantOR(i + 1, matrix, label=f"Experimental OR variant {i+1}")
        for i, matrix in enumerate(matrices)
    ]
    p_sym = EBSDPhase.from_point_group(1, parent.name, parent.lattice(),
                                      parent.point_group).proper_symmetry_cartesian
    m_sym = EBSDPhase.from_point_group(2, product.name, product.lattice(),
                                      product.point_group).proper_symmetry_cartesian
    result = reconstruct_parent(
        ebsd.orientations[indices],
        variants,
        m_sym,
        p_sym,
        inlier_tolerance_deg=tolerance_deg,
        maximum_seed_candidates=64,
        maximum_iterations=20,
    )
    # A single reference OR represents an entire parent-symmetry-generated
    # transformation family. Report assignments for the COMPLETE family,
    # not just the one matrix entered by the user. Extra OR matrices may
    # instead describe independently measured families, so do not collapse
    # them to an arbitrary single reference.
    family = (
        daughter_variant_assignment(
            ebsd.orientations[indices], result.parent_orientation, matrices[0],
            tuple(p_sym), tuple(m_sym),
            acceptance_deg=tolerance_deg,
        )
        if len(matrices) == 1 else None
    )
    return {
        "sampled_points": len(indices),
        "product_points_total": int(np.count_nonzero(
            ebsd.indexed & (ebsd.phase_id == int(phase_id)))),
        "parent_crystal_to_sample_rotation": result.parent_orientation.tolist(),
        "weighted_inlier_fraction": result.weighted_inlier_fraction,
        "mean_inlier_residual_deg": result.mean_inlier_residual_deg,
        "maximum_inlier_residual_deg": result.maximum_inlier_residual_deg,
        "iterations": result.iterations,
        "assigned_variant_labels": [
            x.variant_index for x in result.variant_assignments
        ],
        "full_parent_generated_variant_assignment": family,
        "status": "Exploratory single-parent fit to sampled pixels; "
                  "NOT a spatial parent-grain reconstruction, indexed OR "
                  "identification, or independently validated crystallographic result.",
    }
