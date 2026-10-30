# Parent OR interoperability: MTEX, Python and other crystallography tools

## Same physical hypothesis, same mathematical rotation

The reusable input is one proper Cartesian rotation **Q_parent_from_child**,
mapping indexed daughter (18R) crystal-Cartesian vectors into parent (beta)
crystal-Cartesian vectors. This is *not* an integer direct-lattice
correspondence and is not automatically a named Kajiwara–Nishiyama OR.

- Python indexed orientation: g_M maps child crystal Cartesian to raw Oxford
  specimen CS1. Parent is predicted as
  `g_A = g_M @ S_M @ Q.T @ S_A.T`, with both proper symmetries.
- MTEX p2c misorientation: parent crystal to child crystal. Construct it
  with `orientation.byMatrix(Q.T,csParent,csChild)`; **do not pass Q**.
- Quaternion ordering: SciPy and the portable cross-tool export are xyzw.
- A global specimen-frame correction left-multiplies the indexed g_M and
  predicted g_A. It does not alter Q or daughter/daughter disorientation.
- The indexed **reduced 18R C2/m** Cartesian frame must be the same in all
  tools. The original CIF's atom/basis setting is not proven by the metric
  alone. For a new CIF setting, rebase Q before comparing its numbers.
- MTEX's native Oxford H5OINA reader can apply the header-dependent scan
  correction; our raw Python orientations deliberately did not. Do not
  compare absolute orientations until those specimen frames are aligned.

## What to reproduce

1. Preserve source hashes, phase setting, source orientation conventions,
   sample correction and Q direction.
2. Segment the indexed daughter map at 2, 3 and 5 degrees, record quality
   filtering and the minimum grain size. Do not infer a parent from
   individual raw pixel boundaries.
3. For each daughter-grain boundary, compare proper symmetry-equivalent
   misorientations against `Q.T @ S_parent @ Q` with every proper parent
   operation. Report median/p90 residual and coverage on spatially held-out
   grains; a good median alone does not establish the physical OR.
4. For a graph probability audit, map residual omega to
   `Phi((threshold-omega)/tolerance)` using threshold=2.5°, tolerance=1.5°.
   This reproduces MTEX-style **edge weights**, not the MTEX
   `calcVariantGraph` variant-node structure or MCL cluster solver.
5. In actual MTEX/MATLAB, run `calcVariantGraph`,
   `clusterVariantGraph`, `calcParentFromVote`,
   `mergeSimilar`, independently; compare candidate parent regions and
   accepted variants. Refine the OR with `calcParent2Child` in a **separate**
   job so the frozen, cross-site holdout model remains intact.
6. Reject physically unique OR claims unless the true 18R atomic basis and
   the absolute beta parent orientation/directions are independently
   established, and Kikuchi pattern pseudo-symmetry is addressed.

## Implemented source

- `cualni_cryst.parent_or_interop.mtex_or_contract` exports the two
  rotation directions explicitly.
- `cualni_cryst.parent_or_interop.frozen_or_boundary_audit` computes
  proper-group-equivalent boundary residual and Gaussian edge probability,
  without conflating parent and child groups or claiming actual MTEX parity.
- `cualni_cryst.parent_variant_audit` and
  `cualni_cryst.parent_consensus_complete` handle complete daughter
  variants and bounded quaternion parent voting, respectively.
- `cualni_cryst.ebsd_h5oina` reads native Oxford H5OINA with explicit
  frame metadata and an optional independently calibrated correction.

The actual frozen Site 8 OR quaternion and private Site 6/7/8 grain
orientations are **not hardcoded into this public repository**. They are
contained in the user's local cross-tool reproducibility package.

## Source methods

- https://mtex-toolbox.github.io/OrientationRelationshipFit.html
- https://mtex-toolbox.github.io/GrainGraphBasedReconstruction.html
- https://mtex-toolbox.github.io/MaParentGrainReconstruction.html
- https://mtex-toolbox.github.io/parentGrainReconstructor.calcVariantGraph.html
- https://mtex-toolbox.github.io/EBSDVendors

**Status**: MATLAB/MTEX parity not executed; code-level unit tests and
independent Python grain-table replay are distinct scientific checks.
