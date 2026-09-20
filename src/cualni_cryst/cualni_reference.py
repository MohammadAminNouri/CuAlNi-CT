from __future__ import annotations

"""Opt-in Cu-Al-Ni literature states.

This module deliberately does **not** define material defaults.  It stores
source-specific crystallographic states so that the same general engine can be
benchmarked against multiple Cu-Al-Ni martensites, compositions, cell settings
and experiments without conflating them.

Nothing in this registry is automatically injected into ProjectState.
"""

from dataclasses import dataclass

from .lattice import Lattice
from .provenance import DataStatus


@dataclass(frozen=True)
class CellMeasurement:
    a: float
    b: float
    c: float
    length_unit: str
    alpha_deg: float | None = 90.0
    beta_deg: float | None = 90.0
    gamma_deg: float | None = 90.0
    space_group: str = ""
    point_group: str = ""
    setting: str = ""
    status: DataStatus = DataStatus.SOURCE_MEASURED
    notes: str = ""

    @property
    def complete_metric(self) -> bool:
        return None not in (self.alpha_deg, self.beta_deg, self.gamma_deg)

    def to_lattice(self, *, label: str = "") -> Lattice:
        if not self.complete_metric:
            raise ValueError(
                "This literature cell does not contain enough angular information "
                "to construct a unique metric.  Do not invent the missing angle."
            )
        return Lattice(
            self.a,
            self.b,
            self.c,
            float(self.alpha_deg),
            float(self.beta_deg),
            float(self.gamma_deg),
            label=label,
            length_unit=self.length_unit,
        )


@dataclass(frozen=True)
class CuAlNiLiteratureState:
    key: str
    source_key: str
    composition_reported: str
    parent_phase: str
    martensite_phase: str
    martensite_representation: str
    parent_cell: CellMeasurement | None
    martensite_cell: CellMeasurement | None
    thermomechanical_state: str = ""
    orientation_anchors: tuple[str, ...] = ()
    twin_anchors: tuple[str, ...] = ()
    notes: str = ""


# James-Hane is retained as one benchmark, not as the global Cu-Al-Ni state.
JAMES_HANE_2000_6M = CuAlNiLiteratureState(
    key="james_hane_2000_6m",
    source_key="james_hane2000",
    composition_reported="Cu-Al-Ni, 14 wt% Al, 4 wt% Ni; balance Cu",
    parent_phase="DO3 ordered cubic parent",
    martensite_phase="long-period monoclinic martensite",
    martensite_representation="6M reduced monoclinic cell used in the compatibility treatment",
    parent_cell=CellMeasurement(
        5.836,
        5.836,
        5.836,
        "angstrom",
        space_group="ordered cubic parent; source discusses DO3",
        point_group="m-3m",
        setting="conventional cubic",
    ),
    martensite_cell=CellMeasurement(
        4.430,
        5.330,
        12.79,
        "angstrom",
        beta_deg=95.68,
        space_group="monoclinic long-period representation",
        point_group="2/m",
        setting="unique-b package convention; source non-right a-c angle mapped to beta",
    ),
    notes=(
        "The source also gives 95.28 deg as the angle required by its exact "
        "single-variant compatibility equation for the rounded lattice values. "
        "That derived angle must not replace the measured 95.68 deg."
    ),
)


LANDA_2007_2H = CuAlNiLiteratureState(
    key="landa_2007_2h",
    source_key="landa2007_2h",
    composition_reported="Cu-13.8Al-4.1Ni wt%",
    parent_phase="bcc/ordered cubic austenite",
    martensite_phase="2H orthorhombic martensite",
    martensite_representation="orthorhombic 2H",
    parent_cell=CellMeasurement(
        5.835,
        5.835,
        5.835,
        "angstrom",
        point_group="m-3m",
        setting="conventional cubic",
        notes="0.5835 nm reported in the lattice-correspondence comparison.",
    ),
    martensite_cell=CellMeasurement(
        4.389,
        5.342,
        4.224,
        "angstrom",
        point_group="mmm",
        setting="orthorhombic 2H setting used in the paper",
        notes="0.4389, 0.5342, 0.4224 nm in the source.",
    ),
    thermomechanical_state=(
        "Same single crystal studied as martensite on heating and austenite on "
        "cooling; room-temperature phase depends on thermal history."
    ),
)


CHEN_2000_EBSD_2H = CuAlNiLiteratureState(
    key="chen_2000_ebsd_2h",
    source_key="chen2000_ebsd_2h",
    composition_reported="Cu-12.55Al-4.84Ni wt%",
    parent_phase="parent beta phase",
    martensite_phase="2H martensite",
    martensite_representation="2H as indexed by EBSD",
    parent_cell=None,
    martensite_cell=None,
    orientation_anchors=(
        "2H basal planes originate from parent {110}",
        "[010]_2H originates from a parent <001> axis",
    ),
    twin_anchors=(
        "A-C and B-D habit variants: {121}_2H mirror relation",
        "A-D and B-C habit variants: {101}_2H mirror relation",
    ),
    notes=(
        "This entry intentionally stores orientation/twin observations without "
        "inventing lattice parameters absent from the cited EBSD result."
    ),
)


OTSUKA_1974_STRESS_18R = CuAlNiLiteratureState(
    key="otsuka_1974_stress_18r",
    source_key="otsuka_nakamura_shimizu1974",
    composition_reported="Cu-14.2Al-4.3Ni wt%",
    parent_phase="beta1 ordered parent",
    martensite_phase="stress-induced beta1' long-period martensite",
    martensite_representation="18R long-period cell reported by TEM/electron diffraction",
    parent_cell=None,
    martensite_cell=CellMeasurement(
        4.382,
        5.356,
        38.00,
        "angstrom",
        beta_deg=None,
        space_group="18R-type long-period structure",
        setting="source long-period cell",
        notes=(
            "The accessible source statement reports a,b,c but not a unique "
            "monoclinic angle.  The metric is therefore deliberately incomplete."
        ),
    ),
    orientation_anchors=(
        "matrix/martensite OR reported consistent with the Kajiwara-Nishiyama relation",
    ),
    notes=(
        "The same alloy study observed both 18R beta1' and less frequent 2H gamma' "
        "under stress; phase selection depends on state and loading."
    ),
)


IBARRA_2006_BETA3 = CuAlNiLiteratureState(
    key="ibarra_2006_beta3",
    source_key="ibarra2006_tem",
    composition_reported="Cu-Al-Ni state used for in-situ superelastic TEM",
    parent_phase="L21 beta3 austenite",
    martensite_phase="beta3' monoclinic martensite",
    martensite_representation="C2/m conventional monoclinic cell",
    parent_cell=CellMeasurement(
        5.8216,
        5.8216,
        5.8216,
        "angstrom",
        space_group="Fm-3m",
        point_group="m-3m",
        setting="conventional cubic L21 cell",
    ),
    martensite_cell=CellMeasurement(
        13.8017,
        5.2856,
        4.3987,
        "angstrom",
        beta_deg=113.60,
        space_group="C2/m",
        point_group="2/m",
        setting="conventional C2/m cell",
    ),
    notes=(
        "This is a different cell representation from the reduced 6M benchmark. "
        "No basis equivalence or correspondence is assumed unless explicitly derived."
    ),
)


IBARRA_2006_GAMMA3 = CuAlNiLiteratureState(
    key="ibarra_2006_gamma3",
    source_key="ibarra2006_tem",
    composition_reported="Cu-Al-Ni state used for in-situ superelastic TEM",
    parent_phase="L21 beta3 austenite",
    martensite_phase="gamma3' orthorhombic martensite",
    martensite_representation="Pmmn conventional orthorhombic cell",
    parent_cell=IBARRA_2006_BETA3.parent_cell,
    martensite_cell=CellMeasurement(
        5.3424,
        4.2244,
        4.3896,
        "angstrom",
        space_group="Pmmn",
        point_group="mmm",
        setting="standard orthorhombic setting used by the cited TEM work",
    ),
    notes=(
        "Axis order differs from some other 2H papers.  A permutation of axis "
        "labels is a representation issue and must be handled by an explicit basis "
        "change, never by silently reordering numbers."
    ),
)


CUALNI_LITERATURE_STATES: dict[str, CuAlNiLiteratureState] = {
    state.key: state
    for state in (
        JAMES_HANE_2000_6M,
        LANDA_2007_2H,
        CHEN_2000_EBSD_2H,
        OTSUKA_1974_STRESS_18R,
        IBARRA_2006_BETA3,
        IBARRA_2006_GAMMA3,
    )
}


def available_cualni_literature_states() -> tuple[str, ...]:
    return tuple(sorted(CUALNI_LITERATURE_STATES))


def get_cualni_literature_state(key: str) -> CuAlNiLiteratureState:
    try:
        return CUALNI_LITERATURE_STATES[key]
    except KeyError as exc:
        available = ", ".join(available_cualni_literature_states())
        raise KeyError(
            f"Unknown Cu-Al-Ni literature state {key!r}; available: {available}"
        ) from exc
