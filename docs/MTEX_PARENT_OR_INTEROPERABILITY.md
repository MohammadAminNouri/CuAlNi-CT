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
   `0.5*erfc(2*(omega-threshold)/tolerance)` with threshold=2.5°,
   tolerance=1.5°. This is the actual `calcVariantGraph.m` transition,
   where `tolerance` is a width parameter (not Gaussian σ). This
   reproduces an *unweighted pair-level probability*, not the MTEX
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


## NiTi literature regression and independent parent anchoring (30 October 2026)

To avoid testing only an arbitrary rotation, `niti_literature_or.py`
derives **three published ideal NiTi B2→B19′ ORs**, A, AQ and C, from
their *crystallographic plane and direction parallelisms*. The source is
Cayron (2020), Crystals **10**, 562, DOI 10.3390/cryst10070562.
The default AQ natural relationship is

- (010) B19′ // (110) B2
- [101] B19′ // [−1 1 1] B2

The NiTi parent cell is cubic B2 a=3.015 Å. The NiTi B19′ archived
monoclinic (unique-b) cell is a=2.889, b=4.120, c=4.622 Å and
beta=96.8°. A **direct direction** [uvw] maps to Cartesian using B*uvw;
a **reciprocal plane normal** (hkl) maps using inv(B.T)*hkl. The code
rejects non-coplanar direction/plane pairs instead of rotating a plane
index as if it were a direct lattice vector.

A *distinct* 2026 correspondence-theory benchmark uses
B2 a0=3.010 Å and B19′ a=2.898, b=4.108, c=4.646 Å,
beta=97.78° with exact direct correspondence

    C_M<-A = [[0,0,1],[1/2,1/2,0],[-1/2,1/2,0]]

This C is **not** the NiTi AQ orientation rotation Q.
The legacy Cry/Gen layer must report 12 correspondence variants and
seven intercorrespondence operator classes for that case. Independent
principal stretches of F=B_M*C/a0 are approximately
0.930026279, 0.965048059 and 1.119484928. These metrics are
not the 2020 ideal AQ study's exact lattice input; keep them separate.

`niti_benchmark_runner.py` independently synthesizes 12 complete
B19′ variants from the **published 2020 AQ OR**, applies a fixed
deterministic small SO(3) measurement perturbation, infers a B2 parent
from the martensite orientations without supplying its hidden
orientation, and compares the inferred beta against TWO orthogonal/
non-collinear independently generated reference axes. It tests
published alternative OR C and a deliberately wrong absolute
parent orientation as negative controls. **All these orientations
and parent axes are synthetic**; the source paper's actual raw EBSD
pixels were not obtained. The test is mathematically genuine but not
an independent NiTi experimental replication.

### Real Cu–Al–Ni single-crystal anchor requirement

The AR panel accepts either a retained/reverted β orientation
(g_beta_crystal→Oxford_sample, proper 3×3) or TWO independently
measured nonparallel β crystallographic directions, formatted:

    {"parent_uvw":[[1,0,0],[0,1,1]],
     "sample_xyz":[[...],[...]]}

The two specimen vectors must be actual calibrated measurements in
the **identical sample frame as the 18R EBSD Euler maps**. The
`parent_orientation_from_two_directions` solver checks pair angular
consistency, rejects parallel directions, calculates proper SO(3), and
compares the anchored orientation against the fitted candidate modulo
the 24 proper cubic symmetry operations. If both full measured beta
orientation and two directions are given, contradictions above 3°
are refused. Self-reported evidence is still marked unverified.

For real Sites 6, 7 and 8 there is **no independently measured β axis
or retained β indexed phase in the uploaded H5OINA data**, and neither
the exact reduced 18R atomic setting nor its physical basis mapping
has been demonstrated. Therefore the numerical cross-site success
does not upgrade to physically established β1 parent OR or absolute
orientation. Oxford header zero specimen Euler and scan rotation 180°
are instrument metadata, not a parent lattice measurement.

### 30 October 2026: source-matched MTEX boundary probability correction

The original Python interoperability audit used `scipy.special.ndtr((threshold-misfit)/tolerance)`. That is **not** the probability in MTEX's MATLAB `calcVariantGraph.m` (develop branch checked October 2026), which uses `1 - 0.5*(1 + erf(2*(misfit-threshold)/tolerance))`. The correct Python equivalent is `0.5 * erfc(2*(misfit-threshold)/tolerance)`.

For threshold=2.5° and tolerance=1.5°, the fixed source-based reference probabilities are approximately **0.9999987858** at misfit 0°, **0.5** at 2.5° and **0.0023388675** at 4°. Regression tests explicitly assert all three and the full curve. Previous Python probability exports must be recalculated. Angular operator residuals and previously reported residual histograms are unchanged by this probability-only fix. This is **not** a claim of MTEX numerical parity: pair segmentation, boundary weights, graph nodes and Markov clustering still require MATLAB/MTEX execution.

Reference: https://github.com/mtex-toolbox/mtex/blob/develop/EBSDAnalysis/%40parentGrainReconstructor/calcVariantGraph.m
