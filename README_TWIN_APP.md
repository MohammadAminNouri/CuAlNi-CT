# Standalone Twin Crystallography Workbench — Final integrated package

Target branch: `feature/twin-family-tree-app-v1`

This package integrates the scientific core, validation layer, and final standalone UI from the earlier batches into **one repository-root set of new files**.

It does **not** replace or edit the existing `app/` workstation and does **not** replace or edit any file under `src/cualni_cryst/`. The new front end reuses the repository's existing scientific engines through imports only.

## Run

From the repository root:

```bash
pip install -e ".[app]"
streamlit run twin_app/streamlit_app.py
```

The entry point also bootstraps the repository root and `src/` directory, which makes it robust in GitHub Dev / Codespaces-style launches once the project dependencies are installed.

## Runtime scientific chain

```text
parent/product lattice metrics + point groups + correspondence
    -> exact correspondence subgroup H
    -> correspondence variants M_i
    -> explicit M_i -> U_j stretch mapping
    -> double-coset / inverse-operator relation families
    -> classical exact route OR higher-order weak route
    -> pair-specific exact M/M rank-one relation
    -> physical K1, eta1, shear s for that pair
    -> independent discrete/nonlinear-elasticity cross-lock
    -> twinned laminate fraction lambda
    -> exact A/M habit plane m and shape vector b, when a solution exists
```

No material name or published output value is used to generate runtime answers.

## Scientific safeguards

- `M_i` correspondence variants and `U_j` stretch variants are different objects and remain separately identified.
- Metric collapse is reported explicitly; it is never hidden by renumbering variants.
- Unoriented variant couples are grouped using an operator together with its inverse operator class.
- A higher-order representative does not make a family weak if the same complete operator family already contains an exact classical mirror/twofold route.
- `Compound` is resolved on one physical rank-one branch: a Type-I `K1/eta1` geometry+shear lock and an independent Type-II exact-parent-twofold+shear lock must both pass. Conjugate `K2/eta2` are never equated with physical `K1/eta1`.
- Pair-specific `K1`, `eta1`, and `s` are calculated from that pair's rank-one relation.
- Type-I/compound cross-lock uses product-crystal `K1/eta1` geometry plus shear.
- Type-II cross-lock uses the exact parent twofold provenance plus shear; the code intentionally does **not** compare conjugate `K2/eta2` elements directly with physical `K1/eta1`.
- Every habit solution is keyed to the exact `(base U, other U, rank-one branch)` that generated it.
- “No exact A/M habit-plane solution” is a valid calculated outcome, not an error state.
- Higher-order weak-plane enumeration requires an explicit product primitive-node basis. Bravais centering is never guessed from point-group symmetry.
- Published benchmark values exist only under `data/benchmarks/` and are never imported by `twin_app/` production code.

## Correspondence input

The user explicitly chooses the direction of the matrix being entered:

```text
A -> M : u_M = C_(M<-A) u_A
M -> A : u_A = C_(A<-M) u_M
```

The scientific backend always receives the canonical `A -> M` form. Reverse input is inverted **exactly and symbolically** before it reaches the backend.

## Point-group and UI design

All 32 crystallographic point groups are exposed in their built-in conventional settings. The UI shows the selected group's exact operation inventory and keeps a rotation axis `[uvw]` distinct from a mirror-plane reciprocal covector `(hkl)`. Trigonal groups use hexagonal axes and monoclinic groups use the built-in unique-b setting, matching the backend registry. The app never infers Bravais centering from point-group symmetry.

The page is intentionally predictable and low-clutter:

1. parent input and point-group contents;
2. product input and point-group contents;
3. correspondence direction and exact matrix;
4. optional weak-plane primitive-node basis;
5. one explicit **Calculate twin family and habit planes** button;
6. a semantic result hierarchy immediately below the input: **Root → family → variant pair → twin branch → habit branch**;
7. the representative pair is shown first, while symmetry-equivalent repetitions and technical residuals remain available through clearly labelled expanders.

There are no tabs, animations, automatic twin/habit recalculation, hidden literature presets, or surprise navigation changes. Result presentation is vertical; no result dataframe or multi-column result grid is used. Definitive outcomes such as **no exact A/M habit-plane solution** are displayed as explicit result messages rather than low-priority captions. These choices follow cognitive-accessibility principles of predictable structure, descriptive controls, progressive disclosure and clear control/result relationships.

## Final files

```text
twin_app/
  __init__.py
  input_logic.py
  input_ui.py
  scientific_models.py
  scientific_engine.py
  symmetry_inventory.py
  tree_renderer.py
  streamlit_app.py

tests/twin_app/
  test_batch2_contracts.py
  test_benchmark_integrity.py
  test_final_crosslock_contract.py
  test_input_logic.py
  test_no_runtime_benchmark_answers.py
  test_scientific_engine_contract.py
  test_symmetry_inventory.py
  test_published_niti_twin_habit.py
  test_published_cualni_habit.py
  test_published_niti_weak.py
  test_cognitive_accessibility_contract.py

data/benchmarks/
  bhattacharya_niti_twin_habit_v1.json
  bhattacharya_cualni_orthorhombic_habit_v1.json
  niti_higher_order_weak_v1.json

TWIN_APP_SCIENTIFIC_CONTRACT.md
TWIN_APP_GITHUB_DEV_INSTALL.md
README_TWIN_APP.md
```

## Validation policy

The benchmark files are test oracles only. Their input/expected blocks are SHA-256 locked by `test_benchmark_integrity.py`. The non-NiTi CuAlNi regression exercises the full root path from cubic→orthorhombic topology through Type-I/Type-II twinning to published A/M habit-plane `lambda`, `b` and `m` targets.

The full numerical suite is intended to run **inside the repository** so it can import the existing `cualni_cryst` backend. When this ZIP is tested by itself, backend-dependent modules are skipped explicitly; inside the repository they remain strict and must run. Do not loosen a published-validation tolerance or delete a failing scientific test merely to make CI green. Investigate the implementation, frame convention, branch matching, or a genuine source inconsistency instead.

Recommended repository check after installation:

```bash
pytest tests/twin_app -q
```

Then run the existing repository suite as usual before merging the feature branch.
