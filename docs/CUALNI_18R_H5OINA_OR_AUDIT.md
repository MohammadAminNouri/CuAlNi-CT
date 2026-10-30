# Cu–Al–Ni 18R: parent orientation-relationship audit from Oxford EBSD

## Exact scope

A native Oxford `.h5oina` map may contain a single indexed martensite phase
and no indexed parent beta-1 phase. Such a file does **not** directly identify
the parent's orientation or a unique experimental OR. A single parent may still
be reconstructed *conditionally* if a properly transformed 18R OR and all
required variant rotations are provided.

### What the AR app can do now

- Direct `.h5oina` import with explicit Oxford radians and crystal→sample
  orientation convention, alongside earlier CTF/ANG import.
- Read full-scan counts and indexed phase-cell metadata.
- Deterministically sample indexed daughter orientations for an exploratory
  single-parent fit; do not confuse the sample with full spatial segmentation.
- Refuse an experimental parent fit if the Cry product phase lattice does
  not agree with the indexed H5OINA phase. This prevents comparing NiTi or
  conventionally indexed 18R cells with a different reduced 18R setting.
- Use only **user-verified** Cartesian parent←daughter OR matrices.
  The lattice correspondence matrix C is **not** itself an OR.
- Verify the reader with synthetic HDF5 files in Python 3.11–3.13 CI.
  No private research maps are committed to this public repository.

### What these checks do *not* establish

- No complete, calibrated indexing-quality or pseudo-symmetry validation.
- No automatic parent-free OR optimization from daughter–daughter
  grain-boundary misorientations.
- No experimental selection of Kajiwara–Nishiyama versus another 18R OR.
- No full-resolution grain segmentation, variant assignment or parent map.
- No numerical parity with all original 2021 AR/Gen/Cry source code.

## Reproducible route to an OR decision

1. Verify indexed 18R CIF, direct and reciprocal metric, phase setting,
   and orientation/coordinate corrections.
2. Derive the integer change-of-basis from any conventional 18R cell in
   literature to the *actual reduced C2/m EBSD basis*. Apply that basis
   transformation to direction, plane and rotation definitions.
3. Segment daughter grains and identify genuine sibling variant pairs;
   exclude unindexed areas, indexing ambiguities, and excessive intra-plate
   lattice rotation before estimating OR.
4. Supply physically justified candidate parent–18R ORs, including
   a properly reindexed Kajiwara–Nishiyama-type hypothesis and plausible
   compatibility-related alternatives where derived independently.
5. Compare predicted vs measured disorientation operators on spatially
   independent withheld boundaries. Optimize OR on training boundaries
   only. Reject candidates with inadequate independent validation.
6. Reconstruct beta-1 orientation/variant distribution **after** the OR
   passes those tests. If possible, cross-check a parent orientation
   experimentally from retained austenite or an in-situ thermal reversion.

The single-phase map alone need not uniquely determine the parent:
see MTEX parent-free OR fitting, and explicitly preserve uncertainty.

Literature:
- Otsuka, Nakamura & Shimizu, 1974, DOI: 10.2320/matertrans1960.15.200
- MTEX, Fitting the Orientation Relationship:
  https://mtex-toolbox.github.io/OrientationRelationshipFit.html
- MTEX, Martensite Parent Grain Reconstruction:
  https://mtex-toolbox.github.io/MaParentGrainReconstruction_py.html
- IUCr, Orientation relationships and the embedding approach (2023):
  https://journals.iucr.org/j/issues/2023/03/00/ap5046/index.html
