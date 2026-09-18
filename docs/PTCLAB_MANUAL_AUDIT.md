# PTCLab manual audit and CuAlNi-CT parity plan

Basis: PTCLab User Manual v1.18.8 (94 pages).

The design target is not a literal clone. CuAlNi-CT should preserve PTCLab's
low-friction workflow while retaining stricter direct/reciprocal typing,
metric/Cartesian parity, correspondence/orientation separation, provenance,
explicit tolerances, and cross-theory validation.

## PTCLab capability map

- Crystal structure: CIF/manual input, fractional atom sites, database search,
  symmetry inspection, 3-D structure view, simple-lattice conversion.
- Stereographic tools: directions/planes, symmetry equivalents, pole figures,
  great circles, Wulff net, multiple-crystal overlays and OR variants.
- Transformation crystallography: PTMC, double-shear PTMC, direct/reciprocal
  invariant-line models, O-line, E2E, 3-D NCS, Moire/delta-g, O-lattice,
  matching-distribution, direct lattice matching, GMS/BVC, NCRL, large-misfit
  CSL/DSCL, variants and grain-boundary variant selection.
- Calpad: one-phase angles/lengths/d-spacings/index conversions; two-phase
  calculations with an OR matrix; matrix operations; OR conversions; 2-D atoms.
- Reciprocal space: TEM diffraction, Kikuchi maps, TEM indexing, EBSD
  simulation/indexing, XRD profiles.

## What CuAlNi-CT already does more rigorously

- Direction and Plane are distinct direct/reciprocal object types.
- Metric tensors remain the physical source of truth.
- PTCLab Cartesian, legacy Cartesian and symmetric-metric Cartesian
  representations are parity-audited views of one state.
- Correspondence is not conflated with orientation relationship.
- Physical long-period martensite is separate from its 6M computational cell.
- DO3 -> 6M correspondence/variant/operator topology is truth-locked.
- Cayron CT, Ball-James, PTMC/cofactor paths can be independently cross-checked.
- Provenance, explicit numerical tolerances and dependency fingerprints are
  architectural objects rather than UI conventions.

## Immediate engineering order

1. Explicit dimensional units + precise console terminology.
2. Calpad parity I: phase/cell panel, physical-normal <-> plane-index conversion,
   low-index angle tables.
3. OrientationState + OR engine: Miller parallelisms, axis-angle, Euler,
   matrix validation and symmetry-equivalent ORs.
4. Calpad parity II: true two-phase angles, coordinate transformation,
   direction misfit and variant-angle tables.
5. PySide6 shell.
6. Stereographic/pole-figure engine.
7. Transformation-theory workspace.
8. CIF/atom structure layer.
9. Diffraction/EBSD.
10. Additional PTCLab transformation models as independent plugins.

Product rule:

    choose crystal(s)
    -> type crystallographic input
    -> one obvious action
    -> immediate table/plot

with optional advanced panels for metric form, Cartesian form, PTCLab convention,
derivation, provenance, tolerances, residuals and theory comparison.
