from __future__ import annotations

"""Cu-Al-Ni transformation models whose assumptions are explicit.

The reference correspondences in this module are *SOURCE_DERIVED*, not matrices
quoted verbatim from James & Hane or Otsuka et al.  They are reconstructed from
the published unit-cell geometry and then independently validated against the
published James-Hane stretch families.  That distinction is important for any
paper that uses this software.
"""

from dataclasses import dataclass
import sympy as sp

from .correspondence import Correspondence
from .lattice import Lattice
from .provenance import DataStatus, SourcedValue
from .symmetry import cubic_full_m3m, monoclinic_2_over_m_unique_b, orthorhombic_mmm


# u_M = C u_A.  These rational matrices encode the lattice correspondence,
# independently of the actual lattice parameters.
C_DO3_TO_6M = sp.Matrix([
    [0, 1, 1],
    [1, 0, 0],
    [0, sp.Rational(1, 3), -sp.Rational(1, 3)],
])

C_DO3_TO_2H = sp.Matrix([
    [0, 1, 1],
    [1, 0, 0],
    [0, 1, -1],
])


DO3_TO_6M = Correspondence(
    C_DO3_TO_6M,
    label="DO3 -> 6M long-period martensite (reference correspondence)",
    source="james_hane2000 + otsuka_ohba1993",
    derivation=(
        "SOURCE_DERIVED. Reconstructed from the revised 6M cell geometry: "
        "[100]_A maps to [010]_M; the parent face-diagonal half-translations "
        "1/2[0 1 1]_A and 1/2[0 1 -1]_A define the martensite a and one-third "
        "of the long 6M c translation. The metric pullback is unit-tested to "
        "reproduce the 12 cube-edge stretch matrices of James-Hane Eq. (10)."
    ),
)

DO3_TO_2H = Correspondence(
    C_DO3_TO_2H,
    label="DO3 -> 2H orthorhombic martensite (reference correspondence)",
    source="james_hane2000 + classical Cu-Al-Ni 2H crystallography",
    derivation=(
        "SOURCE_DERIVED. Parent [100] maps to martensite [010], while parent "
        "face-diagonal half-translations 1/2[0 1 1] and 1/2[0 1 -1] map to "
        "martensite [100] and [001]. The metric pullback is unit-tested against "
        "the six face-diagonal cubic-to-orthorhombic stretches of James-Hane Eq. (9)."
    ),
)


@dataclass(frozen=True)
class TransformationBranch:
    name: str
    parent_point_group: tuple[sp.Matrix, ...]
    product_point_group: tuple[sp.Matrix, ...]
    correspondence: Correspondence
    notes: str


def do3_to_6m_branch() -> TransformationBranch:
    return TransformationBranch(
        name="DO3_to_6M",
        parent_point_group=tuple(cubic_full_m3m()),
        product_point_group=tuple(monoclinic_2_over_m_unique_b()),
        correspondence=DO3_TO_6M,
        notes=(
            "Long-period martensite represented in the revised monoclinic 6M cell. "
            "Do not confuse this cell choice with dimensional normalization."
        ),
    )


def do3_to_2h_branch() -> TransformationBranch:
    return TransformationBranch(
        name="DO3_to_2H",
        parent_point_group=tuple(cubic_full_m3m()),
        product_point_group=tuple(orthorhombic_mmm()),
        correspondence=DO3_TO_2H,
        notes="Orthorhombic 2H branch; this is not a beta=90 degree special case of 6M.",
    )


# Literature example used only as a reproducibility benchmark, never as a default
# for an unknown specimen. James & Hane Table 4, Cu-14 wt% Al-4 wt% Ni.
JAMES_HANE_CUALNI_6M_EXAMPLE = {
    "composition": SourcedValue(
        "Cu-Al-Ni, 14 wt% Al, 4 wt% Ni",
        DataStatus.SOURCE_MEASURED,
        "james_hane2000",
        notes="Composition as reported in James & Hane Table 4; Cu balance implied.",
    ),
    "a0_A": SourcedValue(5.836, DataStatus.SOURCE_MEASURED, "james_hane2000", units="angstrom"),
    "a_M": SourcedValue(4.430, DataStatus.SOURCE_MEASURED, "james_hane2000", units="angstrom", original_cell="6M"),
    "b_M": SourcedValue(5.330, DataStatus.SOURCE_MEASURED, "james_hane2000", units="angstrom", original_cell="6M"),
    "c_M": SourcedValue(12.79, DataStatus.SOURCE_MEASURED, "james_hane2000", units="angstrom", original_cell="6M"),
    "beta_M_deg": SourcedValue(95.68, DataStatus.SOURCE_MEASURED, "james_hane2000", units="degree", original_cell="6M"),
    "beta_compat_reported_deg": SourcedValue(95.28, DataStatus.SOURCE_DERIVED, "james_hane2000", units="degree", notes="Compatibility angle reported from Eq. (25); rounded lattice values reproduce it only approximately."),
}


def james_hane_6m_example_lattices() -> tuple[Lattice, Lattice]:
    d = JAMES_HANE_CUALNI_6M_EXAMPLE
    A = Lattice.cubic(float(d["a0_A"].value), label="DO3 parent; James-Hane Table 4 example")
    M = Lattice.monoclinic_unique_b(
        float(d["a_M"].value), float(d["b_M"].value), float(d["c_M"].value),
        float(d["beta_M_deg"].value), label="6M Cu-Al-Ni; James-Hane Table 4 example",
    )
    return A, M
