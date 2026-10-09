# V9.1: scientific interpretation and CI regression repair

This overlay upgrades the **V9 branch** `feature/twin-family-tree-app-v1`.
The solver, scientific data models, numerical tolerances, published benchmarks,
and deployed navigator frontend HTML are deliberately **not replaced**.

## Exactly what is fixed

- `tests/app/test_cayron_v11_geometry.py`: verify the parsed `pyproject.toml`
  extras and that UI dependencies remain separate from the core solver;
  stop asserting a two-dependency app list after spglib/Gemmi were added.
- `twin_app/input_ui.py` and new `twin_app/point_group_guide.py`:
  visible point-group meaning and scientifically grounded decomposition
  by exact operations, with counts of **operations** separated from counts
  of distinct indexed **rotation axes or mirror planes**. The actual
  setting, metric and coordinate-action convention are stated. Matrices and
  3D illustrations remain optional.
- `twin_app/tree_renderer.py` and new `twin_app/interface_interpretation.py`:
  two alternative M/M rank-one interface geometries are selectable in
  large, separately labelled panels. Their unoriented *parent-Cartesian*
  normal-angle and rank-one tensor differences are compared. A/B does not
  mean Type I/Type II and never chooses those labels from sign alone.
  Exact A/M habit solutions remain nested underneath the selected geometry;
  mathematically unresolved classifications stay explicitly unresolved.
- `twin_app/streamlit_app.py`: more consistent readable quantities and
  keyboard focus targets; no animations and no changes to calculations.
- `MANIFEST.sha256`: reproducibly updated after all edits; navigator HTML's
  existing digest is preserved and checked.

## What should users now understand

1. Select **one pair of martensite variants** in the single family navigator.
2. If two distinct M/M interface geometries exist, compare A and B. These
   are two possible solutions to `R_t U_j - U_i = a ⊗ n` for that pair.
3. Inspect one geometry; the Type-I/II/Compound classification is a
   *separate* independently cross-checked claim.
4. Examine `K1` (the **product-crystal twin plane**), `eta1` (product-crystal
   shear direction), and `s` (shear magnitude).
5. Only then examine its calculated A/M habit alternatives satisfying
   `R_h (U_i + lambda a ⊗ n) - I = b ⊗ m`. `m` is in **parent Cartesian**
   coordinates, not automatically a low-index (hkl) plane.

## Unchanged / limitations

- We did not "repair" an unresolved Type-I/II crystallographic cross-lock by
  reassigning its name. Such a case still needs a dedicated scientific audit.
- No exact A/M habit plane is **sometimes a correct result** for entered metrics.
- Parent/product point groups remain user-declared without verified atomic
  positions; spglib is an *independent* structure diagnostic, not a replacement
  for correspondence or M/M equations.
- The existing custom navigator and native fallback remain unchanged. This
  release is deliberately not an untested React Flow/ELK deployment.
- The upstream spglib 2.7 DeprecationWarning seen in CI is a **warning**, not a
  failed test; changing global legacy error handling is intentionally avoided
  until fully tested across the supported 2.5–2.x dependency range.

## Full-repository acceptance gates

- Python 3.11/3.12/3.13 `Scientific Python checks` matrix green.
- `python tools/rebuild_twin_manifest.py --check` green.
- published NiTi/Cu-Al-Ni integrations still green.
- deployed Streamlit: select both interface cards; switch between couples;
  confirm that all habit records and the original technical audit are reachable.
- Keyboard and screen-reader manual check is still required; source-level UI
  tests alone cannot certify cognitive accessibility.
