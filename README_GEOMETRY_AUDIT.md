# Independent twin and habit geometry audit — 22 October 2026

**Scope:** Read-only independent mpmath construction of the nonlinear-elasticity rank-one tensors and invariant-plane tensors for the existing 2H / James–Hane 6M benchmark input states. No production edits, no commits, and no implicit relaxation of thresholds.

## Codespace

Upload the ZIP into the `CuAlNi-CT` project root on your existing feature branch:

```bash
unzip -o independent_geometry_audit_2026_10_22.zip
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u independent_geometry_audit_2026_10_22.py --selftest
python -m pytest -q test_independent_geometry_audit_2026_10_22.py
python -u independent_geometry_audit_2026_10_22.py --case both
```

The last command produces **`audit_findings/geometry_independent_2026_10_22.json`**. Upload this JSON here for a detailed investigation; also upload it if the command exits nonzero. Do not change the source files or lower any audit tolerances to make it green. A 6M calculation can take longer than 2H due to high-precision eigensolves.

## What it truly tests

- Reconstructs two rank-one twin dyads using 85-digit eigendecomposition from the production **stretch tensors** and their ordered pairs. Does **not** use production twin shear vectors/normals to derive oracle dyads.
- Validates each production `R U_j - U_i = a ⊗ n`, `RᵀR=I`, `det R=+1`, and compares the two complete dyads as unordered **one-to-one** sets (branch label sign may differ).
- Reconstructs the determinant polynomial from the **independently reconstructed** twin dyads, isolates real roots in `[0,1]`, and compares the root set to the PTMC report. Degenerate determinants and tangencies close to the numerical boundary are **inconclusive**, never forced to pass.
- Independently constructs **both habit-plane rank-one dyads** from each compatible `F(f)`, checks physical geometry and rotation of each reported habit branch, and checks for extra reported habit solutions.
- Records necessary symmetry-axis geometric evidence for candidate Type I/II labels, **without** falsely treating this as an independently certified Cayron correspondence-theory cross-lock.

`PASS_CONDITIONAL` means all numerically resolved checks match for the encoded benchmark states. `INCONCLUSIVE` marks precision or eigenvalue boundaries. `MISMATCH` is a reproducible difference requiring investigation. `EXECUTION_ERROR` includes a traceback. Exit codes 0, 3, 2 respectively; operational errors also 2.

## Limitations

Uses production U variants as starting states: not an independent verification of variant generation, parent correspondence, experimental EBSD indexing, or the full input provenance. The rank-one construction implements the same continuum mechanics theorem with a different numeric library; it is an independent implementation/crosscheck, **not an independent physical theory**. The symmetry-axis screen is necessary geometric evidence, not a certification of Type I/II/Compound labels. This audit does **not** verify final Streamlit mode grouping, K₁/η₁ from discrete CT, all lattice geometries, measurement uncertainty, or a proof of completeness in near-degenerate cases. These remain separate gates.
