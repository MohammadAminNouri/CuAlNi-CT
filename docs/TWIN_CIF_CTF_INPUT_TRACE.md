# Twin App input routes: CIF structures and indexed EBSD CTF

Status: source-level trace, 28 October 2026. This is not experimental
2H/18R certification or proof that a predicted twin occurs.

## A. CIF -> crystallographic calculations

1. In Step 01 choose **Import CIF structures** and upload exactly one
   parent CIF and one product CIF. Browser uploader is capped at 2 MB per
   file; Gemmi rejects files with more than 2,000,000 bytes.
2. twin_app/streamlit_app.py calls
   twin_app/structure_validation.py:parse_cif. Gemmi reads cell
   lengths, angles and expanded atomic sites (up to 20,000).
3. independent_spglib_check is an advisory atomic-structure consistency
   check. CIF symmetry and a metric-preserving operation alone do not
   independently establish the specimen's actual point group.
4. The user explicitly chooses a geometrically compatible conventional
   crystal system and applies both cells; no cell lengths or angles may
   be silently clamped by the default NiTi family.
5. The user checks point groups, axis setting, and the exact
   *direct-lattice correspondence C*. CIF does NOT provide C or a
   parent<-product orientation relationship R. Changing an imported
   file invalidates the admission/confirmation.
6. CalculationRequest -> project payload -> scientific_engine:
   build_twin_family_report -> validated report -> mode catalogue.
   The solvers use the entered metrics/symmetries/C. They do NOT ingest
   raw atomic positions as an additional energetic prediction model.

## B. CTF -> optional EBSD evidence

1. In **Step 01 > Experimental EBSD / CTF (optional)** upload an
   already-indexed Oxford/AZtec Channel Text File (.ctf). The browser
   uploader limit is 35 MB; backend also rejects files exceeding
   MAX_CTF_BYTES=35*1024*1024.
2. twin_app/ebsd_ctf_ui.py calls
   twin_app/ebsd_integration.py:inspect_ctf_upload with the native
   cualni_cryst.ebsd_io.load_ctf reader. This checks the actual file,
   counts total/indexed pixels and displays phase IDs and counts.
   No OR, theoretical calculation or twin assertion is needed to
   *inspect* the map. Standard 2-D CTF Bunge ZXZ Euler degrees are
   used. A 3-D CTF with ambiguous angle units is rejected by the
   generic parser until its angle-unit convention is established.
**Independent map route (no OR and no completed theoretical calculation):**
`ctf_independent.declared_ctf_phases` parses all AZtec phase-table
entries, including phases with zero indexed points. The phase table's
space-group numbers are **indexing declarations**, not experimental proof
of phase or space-group identity. Once requested, the native loader,
proper phase symmetries and measured orientations are used to segment
orientation-connected regions, compute GOS/KAM, count boundaries and
report phase, region-ID, KAM, MAD, BC and BS maps. The full pixel arrays
are downloadable as NPZ and region statistics as JSON. An independent
vectorized proper-symmetry quotient checks the distribution of
adjacent-pixel misorientations and warns of potentially unreliable
grain interpretation when high-angle jumps are widespread.

These independent outputs do NOT infer an austenite orientation, unique
OR, 2H/18R phase truth, Type I/II twin, shear or habit plane.

3. User maps CTF phase IDs to entered parent/product phases. Parent
   pixels may be absent from a product-only map; product pixels must
   be indexed. Extra indexed phase IDs require separating the maps
   because this interface is a TWO-PHASE analysis.
4. For comparison with theory, the user supplies an independently
   established proper Cartesian rotation
   x_parent = R_(parent<-product) x_product, **not** a raw C matrix.
   The SO(3) contract rejects reflections and nonrotations. The
   checkbox is keyed to CTF content, phase IDs, R, grain threshold
   and crystallographic model, and resets when these change.
5. Before any operator comparison, both the UI AND the backend verify
   that the declared AZtec cells/point groups match the two selected
   theoretical phases within stated same-axis tolerances. A conflicting
   setting blocks theory matching, even if numerical orientation
   operators could be calculated.
   Once the matching crystallographic calculation is available, the
   separate **Compare EBSD with theory** action calls
   run_ctf_observation. A temporary CTF and JSON configuration
   invoke cualni_cryst.ebsd_pipeline.run_pipeline. The configured
   defaults include grain segmentation at 3 degrees (or the explicit
   Step-01 user-selected threshold), diagnostic sweep, boundary-
   operator scoring and candidate parent
   reconstruction. OR refinement and experimental trace validation
   are disabled. No Kikuchi pattern reindexing occurs.
6. The pipeline validates phase IDs and input frames, calculates
   indexed grain orientations, segments grains, evaluates measured
   boundary rotations against symmetry-derived candidates and
   writes summary / boundary / grain-assignment results.
7. summarize_twin_candidates places theoretical candidate-twin pair
   counts SIDE BY SIDE with experimental boundary/operator counts.
   It deliberately NEVER treats EBSD OR-variant IDs as the same
   as correspondence or stretch variant IDs without an independently
   validated cross-map. A boundary match is NOT a Type I/II/Compound
   twin certificate or a measurement of an out-of-plane habit plane.
8. The export records file SHA256, CTF phase-role IDs, raw OR matrix,
   model signature and the complete original pipeline outputs.
   Unlike the previous wrapper, the temporary pipeline's
   map_fields.npz, grain, boundary, source and variant-assignment
   files are retained in an exportable ZIP before cleanup.
   Changes in any scientific input invalidate old displayed results.

## C. Scientific interpretation for Cu–Al–Ni

- β3 cubic austenite and 2H orthorhombic martensite can form
  one parent/product transformation case; β3 and modified 18R
  monoclinic martensite form a DISTINCT case. A mixed 2H+18R map
  requires phase-aware treatment rather than mislabeling two
  structures as symmetry variants of one phase.
- CIF indexing, the orientation relationship, specimen orientation
  and correspondence must be checked in a common basis and
  crystallographic convention.
- If there is no independently verified OR, the app performs
  OR-free phase/region/KAM/quality cartography but does **not** run
  an OR-dependent boundary-operator fit or parent reconstruction. The existing separate blind-OR research backend
  addresses an inverse observable class and is NOT activated as a
  default OR guess in this simple interface.
- To identify physical twin systems in a thesis, independently
  compare the experimentally indexed inter-grain misorientations,
  crystallographic planes/directions, uncertainties and
  sample-local constraints. Neither a CTF operator match nor
  numeric rank-one compatibility alone proves a selected twin.

Source files:
  twin_app/streamlit_app.py
  twin_app/structure_validation.py
  twin_app/ebsd_ctf_ui.py
  twin_app/ctf_independent.py
  twin_app/ebsd_integration.py
  src/cualni_cryst/ebsd_io.py
  src/cualni_cryst/ebsd_pipeline.py
  src/cualni_cryst/ebsd_theory_bridge.py

Tests:
  tests/twin_app/test_ctf_step01_workflow_2026.py
  tests/twin_app/test_ebsd_ctf_integration.py
  tests/twin_app/test_independent_ctf_cartography_2026.py
  tests/twin_app/test_ebsd_contract_2026_10_23.py
  tests/twin_app/test_cif_pair_gate_2026_10_23.py
  tests/twin_app/test_input_cif_ui_simplification_2026.py

Primary scientific context:
  Cayron, Acta Materialia 316 (2026) 122399;
  Ball & James, Arch. Rat. Mech. Anal. 100 (1987) 13-52;
  Chen et al., JMPS 61 (2013) 2566-2587.
