# Cu-Al-Ni reference-data scope

The software must never silently equate "Cu-Al-Ni" with one James-Hane
parameter tuple.

Cu-Al-Ni martensite depends on composition, atomic order, heat treatment,
thermal history, applied stress, loading orientation and temperature.  The
literature also uses several crystallographic cell settings for physically
related phases.

The file `src/cualni_cryst/cualni_reference.py` therefore contains **opt-in
literature states**, not defaults.

Examples currently represented include:

- James-Hane 2000: DO3 -> reduced 6M long-period monoclinic benchmark;
- Landa et al. 2007: Cu-13.8Al-4.1Ni wt%, cubic parent / 2H orthorhombic
  martensite;
- Chen et al. 2000: Cu-12.55Al-4.84Ni wt% EBSD OR and twin anchors for 2H;
- Otsuka-Nakamura-Shimizu 1974: stress-induced 18R beta1' observations;
- Ibarra et al. 2006: L21 beta3 parent with conventional C2/m beta3' and Pmmn
  gamma3' cells used for in-situ TEM indexing.

These entries are deliberately heterogeneous.  Some contain complete metrics,
some contain only OR/twin observations, and some lack an angle needed to build
a unique metric.  Missing data remain missing.

## Rules

A literature state may be used as a benchmark only after the calculation has
explicitly selected it.

A cell measurement does not determine a correspondence.

A correspondence does not determine an experimental OR.

A physical phase name does not determine one unique unit-cell representation.

A different cell setting must be connected by an explicit basis-change
matrix before indices or matrices are compared.

A measured value, a source-derived value and a computed value keep different
provenance labels.

## Consequence for the future PTCLab-like interface

The future UI can expose these reference states as optional presets, but the
underlying general crystallography engine stays parameter-driven.  Users can
supply arbitrary valid parent/product cells, symmetry, correspondence and
experimental metadata without changing the theory modules.
