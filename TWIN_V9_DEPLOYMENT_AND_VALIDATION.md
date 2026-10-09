# Twin-family workbench — V9 scientific input and accessible results release

Target branch: `feature/twin-family-tree-app-v1`.

## Scientific scope and deliberate non-changes

**Do not overwrite** `src/cualni_cryst/`, `twin_app/scientific_engine.py`, `twin_app/scientific_certifier.py`, existing benchmark JSONs, or the deployed `twin_app/navigator_frontend/index.html`. The current existing matrix-based Ball–James, CT, PTMC and cofactor solvers remain authoritative.

This release changes only the **input boundary**, the **presentation of calculated M/M twin branches and their nested A/M habit solutions**, adds **independent diagnostic tools**, and tests these behaviours.

### Input options

* **Published example:** editable NiTi B2/B19' metrics and exact correspondence *inputs*, never published twin/habit results.
* **Import CIF:** Gemmi reads exactly one crystal with atomic sites; `spglib` separately reports space-group and point-group evidence at a stated tolerance. Imported cell metrics can populate the fields; a correspondence **must still be entered/verified independently**. Neither tool is used to fabricate it.
* **Manual:** specify lattice metrics, point groups and exact parent→product correspondence.

Every correspondence column is explained as the product-basis coefficients of a parent basis direction. A correspondence matrix is not a rigid orientation relationship. The optional reverse matrix is inverted exactly using SymPy. The six lattice parameters undergo positive-definite-metric checks. Switching length unit **converts numeric lengths** to preserve the same physical cell. CIF lengths supplied in Å are converted to the selected UI unit.

### M/M versus A/M interface branches

Two algebraic ± branches are **not automatically** Type I/Type II. The view labels distinct physical rank-one tensors `Interface A` and `Interface B` (or more where actually returned), sorting them deterministically by geometry and collapsing duplicated rank-one tensors for presentation **without dropping source indices or unique habit results**. If the source classifications conflict, the UI warns. The underlying report and scientific export are untouched. Each interface's A/M habit solutions are then shown **one at a time** with their own λ, physical b, Cartesian m and explicit coordinate frame. No-compatibility and continuous-compatibility outcomes remain explicit.

### Dependencies

`requirements.txt` already installs `-e .[app]`; this release adds `gemmi` and `spglib` to that **app extra**, while `hypothesis` is included in the **dev extra** for property-based scientific tests. No new package is required by the base numerical solver. If standalone CIF parsing or spglib cannot run, the UI reports that diagnostic as unavailable; no symmetry is guessed.

### The genealogy frontend

The last verified, self-contained, keyboard-operable HTML navigator is **preserved unchanged**, including native Streamlit fallback. This is an intentional safety constraint after the missing-asset/manifest and component-registration failures. React Flow + ELK / Streamlit Components V2 are **not** claimed in this release: shipping an unbuilt package would risk redeployment failure. A future separately proven frontend migration can replace this component while keeping the stable Python navigator contract. This release does not fully satisfy the desired final graph-engine migration.

### Validation before merging

1. Upload files to their **exact repository paths** using github.dev; do not flatten folder structures.
2. Verify there is no modification under `src/cualni_cryst/`, `twin_app/scientific_engine.py`, or any `data/benchmarks/*.json`.
3. Commit and wait for all three Python-version GitHub Actions matrix jobs.
4. On Streamlit Cloud, test manual → example → CIF inputs, change Å↔nm and confirm length conversion, select two M/M interfaces, inspect each corresponding A/M habit result, and select twins with no exact habit plane.
5. Verify the navigator still shows every family; if the component is unavailable the native fallback must remain usable.
6. A successful CI run does **not** demonstrate universal martensitic completeness or autism-accessibility certification; both need independent broader cases and usability evaluation.

### Tested locally

Only the new standalone modules and existing standalone tests can execute in this environment; the repository's complete `src/` backend and Streamlit itself are not available in the local extract. Full numerical, real `spglib` and Hypothesis validation must run on GitHub with the new dependencies.

### Correct approach to manifests

The manifest is a deliberately tracked collection of workbench artifacts. `tools/rebuild_twin_manifest.py` hashes **only listed files**, refuses missing or unsafe entries and supports `--check`. It intentionally does not hash the entire repository or include transient caches. This V9 package contains a freshly updated manifest, and does not modify the deployed HTML navigator.
