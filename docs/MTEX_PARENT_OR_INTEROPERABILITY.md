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


## Interoperability release: three independent numerical stages

The separate AR · Gen · Cry Streamlit AR panel now offers three distinct
numerical stages, each accepting a manually supplied/frozen **Q** and beta
orientation. No hidden fitting is performed in these stages.

1. **Parent ⇄ daughter**: `parent_replay_exchange.py` enumerates the
   inequivalent cubic-generated 18R variants and both proper C2 indexer
   representatives, tests each measured daughter orientation, inverts the
   selected variant, and asserts that forward/inverse SO(3) angular residuals
   agree. Negatively controls wrong input rotations. The session may import
   `frozen_site8_OR_manifest.json` from the user's local kit. No private
   fitted Q is committed to GitHub.
2. **Full-grid conditional parent**:
   `parent_fullmap_h5oina.py` reads every indexed Oxford Euler pixel,
   applies bounded/fixed Q and parent orientation in chunks, filters by
   MAD/residual, and saves HDF5 with parent quaternions, residuals, variant
   IDs, quality mask and connected **variant-patch** support. Patch support
   is not a physically segmented MTEX grain and is not Markov clustering.
   A raw H5OINA source is required and no automatic specimen rotation is
   applied.
3. **Independent boundary table replay**:
   `parent_grain_table_replay.py` reads portable grain-quaternion and
   adjacency CSVs, rejects duplicate/self/missing edges and invalid
   rotations, recomputes the theoretical OR intervariant operators,
   misorientation residuals, and MTEX-style Gaussian edge probabilities.
   This is not MTEX `calcVariantGraph` / MCL. The grain segmentation is
   assumed to have been carried out independently.

### Portable ZIP / MATLAB / AZtec Crystal

`parent_replay_exchange.export_replay_zip` writes a reproducibility
contract, both OR directions, candidate beta and predicted daughter
orientation/quaternion/Bunge-angle CSVs, numerical diagnostics, and a
**generated but not executed** MATLAB/MTEX script. The script uses
`EBSD.load`, `calcGrains`, `parentGrainReconstructor`,
`calcVariantGraph`, `clusterVariantGraph`, `calcParentFromVote`,
`mergeSimilar`, and `calcParentEBSD`; it saves a **separate** alternative
from `calcParent2Child` without overwriting the frozen OR.

The raw H5OINA measurement is never embedded in the exchange ZIP; the
user must place it alongside the MATLAB launcher. Optional original beta
and indexed 18R CIFs are passed through **unchanged** into the ZIP.
AZtec Crystal requires the correct lattice/atomic phase structure and
its own import procedures. A 3×3 OR or an orientation-quaternion CSV is
not an AZtec phase CIF, and the program does not claim a verified
native AZtec importer.

### Independent actual-scan replay, confidential outside GitHub

The user-supplied Site 6, 7 and 8 grain CSVs have been processed using
the *frozen* Site 8 Q (not refitted on Sites 6/7), all 24 proper beta
rotations and both proper monoclinic C2 representatives. Forward median
angular residuals are 0.9568°, 1.1915° and 0.6604°. Grain-area fractions
within 3° are 100%, 99.66% and 99.68%; both identity-Q and a specified
15°-perturbed-Q control give 0% within 5° for the same fixed candidate
parent. Inverse-to-parent residuals agree with the forward residuals
to <1e-10 degrees as required by SO(3) algebra. This is a convention
sanity check and out-of-sample *site* transfer, not a unique-OR proof:
all three scans share the same **idealized 18R indexed phase** and there
is no independently measured retained/reverted beta parent.

The private replay kit contains the full-precision Q, candidate parent,
CSV hashes, independent executable checker, and source scan hash manifest.
It is not version-controlled in the public research repository.

### Scientific acceptance ladder

- **Numerical (supported):** SO(3) operators, inverse correctness,
  proper crystallographic symmetries, source phase/lattice agreement,
  explicit specimen-frame provenance, frozen cross-site operator tests,
  residual and ambiguity masks.
- **External MTEX (prepared but not executed here):** import in a
  separately installed MATLAB/MTEX environment, compare segmentation,
  variant-graph nodes, Markov clusters, parent votes, recovered
  parent-EBSD fit and phase relabeling.
- **Physical (still unverified):** independent indexing/pseudosymmetry
  audit of the reduced 18R CIF and a beta parent measurement or calibrated
  single-crystal axes; distinguish β₁ and alternative crystal-frame
  choices by actual diffraction and phase structure, not only residuals.
