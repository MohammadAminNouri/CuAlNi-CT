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


## Updated instrument-frame audit and parent evidence (30 October 2026)

Oxford's native H5OINA specification defines **Scanning Rotation Angle** as the
angle between the specimen-tilt axis and the scanning-tilt axis, in radians.
It is NOT automatically the angle of a laboratory-to-specimen rigid rotation
about z or y. The reader now reports scanning rotation, tilt angle/axis and
specimen orientation Euler explicitly. The latter defines CS1 surface→CS0
primary specimen coordinates. No correction is applied by default.

When a specimen-frame alignment is independently established, the
`load_h5oina_sample(..., sample_frame_correction=proper_3x3)` option explicitly
applies it and stores the exact matrix in provenance. A globally applied proper
rotation must preserve intervariant disorientation angles and cannot, by
itself, identify the physical β parent.

The file's indexed reference names a **reduced 18R CIF**. Its cell lengths and
angles are recoverable, but atomic positions and the conventional→reduced
integer change-of-basis matrix are NOT stored in the EBSD scan.

### What has been implemented

- `parent_or_evidence.audit_direct_basis`: check an **explicit** integer
  right-handed basis matrix `P` using `G_new = P.T @ G_old @ P`.
  Report metric error and volume-index ratio; refuse inconsistent inputs.
  Atomic correspondence must be independently verified.
- `parent_or_evidence.rebase_orientation_relationship`: transform a
  parent←daughter Cartesian OR between a conventional and indexed 18R frame
  using the associated physical basis rotations, rather than relabeling Miller
  indices without changing the rotation.
- `parent_or_evidence.parent_orientation_from_two_directions`: obtain a
  calibrated parent orientation from TWO independent crystal directions and
  their measured specimen directions, rejecting near-collinear pairs and
  inconsistent intervector angles.
- `parent_or_evidence.compare_parent_to_anchor`: measure parent–anchor
  disorientation modulo proper β cubic symmetry.
- `parent_variant_audit.parent_generated_variants`: generate all proper
  parent-derived daughter variants modulo daughter symmetry; under the
  monoclinic C2 / cubic 432 example the family contains 12 or 24 variants
  depending on the OR stabilizer.
- `parent_variant_audit.daughter_variant_assignment`: fast symmetry-aware
  assignments, residuals and runner-up ambiguity margins.
- AR Streamlit: optional conventional 18R CIF + explicit integer P, input OR
  rebasing, and comparison with a supplied experimentally measured parent
  orientation. A metric-compatible P alone does not prove a physical
  correspondence.
- Experimental certification remains **false by default and after any
  self-reported checkboxes**. External data provenance is not authenticated.

### Actual Site 8 numerical revision

The local revised Site 8 parent output applies independent spatial
diagnostics to the earlier **unanchored** fit. A conservative quality mask
requires low indexed MAD, residual ≤3°, connected daughter region ≥50 pixels,
at least 80% of that region fitting within 3°, and mean region residual ≤2°.
It accepts 425,791 of 621,983 high-quality indexed pixels (68.46%).
This is a *more conservative in-sample numerical mask*, not physically
validated reconstruction. The input source H5OINA is not tracked in Git.

An alternative global 180° y-rotation is exported for comparison as an
explicitly **hypothetical, unverified** frame convention. Intervariant
misorientations remain numerically unchanged; no false gain in fit quality
is claimed.

### Evidence required for a physical reconstruction

Provide the original conventional 18R CIF **and** the reduced indexed CIF
or an experimentally documented basis mapping; a physically justified
parent–18R OR; and at least one calibrated parent EBSD orientation from
retained/reverted austenite, or two nonparallel known parent directions
measured in the same specimen frame. Confirm specimen calibration and
pseudo-symmetry before endorsing parent variant labels.

Primary specifications and methods:
- Oxford H5OINA format, https://github.com/oinanoanalysis/h5oina/blob/master/H5OINAFile.md
- MTEX Oxford vendor conventions, https://mtex-toolbox.github.io/EBSDVendors
- MTEX reference-frame warnings, https://mtex-toolbox.github.io/EBSDReferenceFrame
- MTEX parent-grain reconstruction, https://mtex-toolbox.github.io/GrainGraphBasedReconstruction
- MTEX orientation-relationship fitting, https://mtex-toolbox.github.io/OrientationRelationshipFit


### Rational conventional-to-indexed 18R pseudocells

The indexed 18R metric may correspond to a one-third-volume representation
of a conventional long-period cell. Its direct-basis matrix need not have
integer coefficients when expressed in conventional supercell coordinates.
A matrix containing rational values such as `1/3` is now supported.

The audit retains:
- `P` in the convention `B_indexed ≈ B_conventional @ P`
- `det(P)`, the signed indexed-to-conventional volume ratio
- the full metric residual `||Pᵀ G_conventional P - G_indexed||/||G_indexed||`
- whether P is integral or rational
- the polar-projected proper Cartesian frame conversion, explicitly tagged
  as *conditional* when the two metrics are not exactly equal

**Strict mode** (relative metric tolerance 5e-4) is required to permit an
OR-frame conversion. **Literature screen** (3e-2 tolerance) allows users to
inspect candidate cells with different compositions or idealized metrics,
but blocks use of such a match as a physically verified OR.

Even strict matching is metric-only: a fractional P does not establish
atomic translation symmetry of a modulated 18R structure. Verify the actual
indexed EBSD source CIF and conventional structure's atomic sites and
systematic reflection conditions before trusting any parent reconstruction.

### 30 October 2026: full-map phase-basis and source-hash gate

The full H5OINA parent-map exporter now requires an indexed daughter Lattice and compares all six cell parameters against vendor metadata (0.02 Å and 0.2° absolute tolerances). The Streamlit UI supplies the Cry daughter cell. It also preserves the exact input file SHA-256 and both cells in the output HDF5. This is metadata and provenance validation, not evidence of atomic correspondence or physical OR uniqueness.
