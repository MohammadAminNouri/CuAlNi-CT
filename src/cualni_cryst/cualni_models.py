from __future__ import annotations

"""Cu-Al-Ni transformation models whose assumptions are explicit.

Important distinction for the long-period branch:

- the physical long-period martensite,
- the chosen 6M cell representation,
- one symmetry-equivalent reference correspondence variant,

are separate objects.

The reference correspondence used here is reconstructed from the 6M basis
geometry summarized by James & Hane (2000) following Otsuka et al. It is not
claimed to be a verbatim matrix printed in those papers. Its strongest
independent check is reproduction of the James-Hane cube-edge stretch family.
"""

from dataclasses import dataclass

import sympy as sp

from .correspondence import Correspondence
from .lattice import Lattice
from .provenance import DataStatus, SourcedValue
from .reference_6m import (
    C_REF_M_FROM_A,
    DO3_6M_BASIS_CONVENTION,
    daughter_basis_in_parent_coordinates,
)
from .symmetry import (
    cubic_full_m3m,
    monoclinic_2_over_m_unique_b,
    orthorhombic_mmm,
)

# Backward-compatible public name. Scientifically this is one selected
# symmetry-equivalent reference correspondence variant, not "the one unique C".
C_DO3_TO_6M_REFERENCE = C_REF_M_FROM_A
C_DO3_TO_6M = C_DO3_TO_6M_REFERENCE

# The 2H branch remains outside the present 6M truth-lock pass.
C_DO3_TO_2H = sp.Matrix(
    [
        [0, 1, 1],
        [1, 0, 0],
        [0, 1, -1],
    ]
)


DO3_TO_6M = Correspondence(
    C_DO3_TO_6M_REFERENCE,
    label="DO3 -> 6M long-period martensite (selected reference correspondence variant)",
    source="james_hane2000 + otsuka_ohba1993",
    derivation=(
        "SOURCE_INTERPRETATION. The selected daughter basis in parent cubic "
        "coordinates is a_6M=1/2[0 1 1]_A, b_6M=[1 0 0]_A, "
        "c_6M=3/2[0 1 -1]_A. Inverting that basis matrix under the package "
        "convention u_M=C_M_from_A u_A gives the stored rational matrix. "
        "This is one symmetry-equivalent reference variant. It is independently "
        "cross-checked against the 12 James-Hane Eq.(10) cube-edge stretches."
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
    basis = DO3_6M_BASIS_CONVENTION
    return TransformationBranch(
        name="DO3_to_6M",
        parent_point_group=tuple(cubic_full_m3m()),
        product_point_group=tuple(monoclinic_2_over_m_unique_b()),
        correspondence=DO3_TO_6M,
        notes=(
            "Long-period martensite represented in a selected monoclinic 6M "
            "reference basis. The non-right angle is between a_6M and c_6M; "
            f"internally it is called {basis.internal_angle_symbol}. "
            "This cell/basis choice is not a dimensional normalization."
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


def do3_to_6m_reference_basis_matrix() -> sp.Matrix:
    """Return C_A_from_M for the selected 6M reference variant."""

    return daughter_basis_in_parent_coordinates()


# Literature example used only as a reproducibility benchmark, never as a
# default for an unknown specimen. James & Hane Table 4, Cu-14 wt% Al-4 wt% Ni.
JAMES_HANE_CUALNI_6M_EXAMPLE = {
    "composition": SourcedValue(
        "Cu-Al-Ni, 14 wt% Al, 4 wt% Ni",
        DataStatus.SOURCE_MEASURED,
        "james_hane2000",
        notes="Composition as reported in James & Hane Table 4; Cu balance implied.",
    ),
    "a0_A": SourcedValue(
        5.836,
        DataStatus.SOURCE_MEASURED,
        "james_hane2000",
        units="angstrom",
    ),
    "a_M": SourcedValue(
        4.430,
        DataStatus.SOURCE_MEASURED,
        "james_hane2000",
        units="angstrom",
        original_cell="6M",
    ),
    "b_M": SourcedValue(
        5.330,
        DataStatus.SOURCE_MEASURED,
        "james_hane2000",
        units="angstrom",
        original_cell="6M",
    ),
    "c_M": SourcedValue(
        12.79,
        DataStatus.SOURCE_MEASURED,
        "james_hane2000",
        units="angstrom",
        original_cell="6M",
    ),
    "beta_M_deg": SourcedValue(
        95.68,
        DataStatus.SOURCE_MEASURED,
        "james_hane2000",
        units="degree",
        original_cell="6M",
        notes=(
            "James-Hane source angle is the non-right angle between the "
            "martensite a and c edges; the package stores that angle as beta "
            "in the conventional unique-b setting."
        ),
    ),
    "beta_compat_reported_deg": SourcedValue(
        95.28,
        DataStatus.SOURCE_DERIVED,
        "james_hane2000",
        units="degree",
        notes=(
            "Compatibility angle reported from Eq. (25); rounded lattice values "
            "reproduce it only approximately."
        ),
    ),
}


def james_hane_6m_example_lattices() -> tuple[Lattice, Lattice]:
    d = JAMES_HANE_CUALNI_6M_EXAMPLE
    A = Lattice.cubic(
        float(d["a0_A"].value),
        label="DO3 parent; James-Hane Table 4 example",
    )
    M = Lattice.monoclinic_unique_b(
        float(d["a_M"].value),
        float(d["b_M"].value),
        float(d["c_M"].value),
        float(d["beta_M_deg"].value),
        label="6M Cu-Al-Ni; James-Hane Table 4 example",
    )
    return A, M
