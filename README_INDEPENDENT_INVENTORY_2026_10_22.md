# Independent crystallographic solution-inventory audit — 22 October 2026

**Purpose:** Adversarial, read-only comparison of an existing PTMC report with an 85-decimal-digit oracle. This release does not patch a single scientific production file and does not change any equation, numerical policy, Git branch, or existing benchmark.

## Codespace instructions

Upload `independent_solution_inventory_audit_2026_10_22.zip` into the root of the `CuAlNi-CT` repository on the existing feature branch, then run:

```bash
unzip -o independent_solution_inventory_audit_2026_10_22.zip
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u independent_solution_inventory_audit_2026_10_22.py --selftest
python -m pytest -q test_inventory_audit_selftest_2026_10_22.py
python -u independent_solution_inventory_audit_2026_10_22.py --case 2h
```

The scientific audit writes `audit_findings/independent_inventory_2026_10_22.json`. Upload **that JSON to ChatGPT**, even if the terminal shows MISMATCH or INCONCLUSIVE. Do **not** change production files and do **not** commit the archive or its extracted files.

Optional expanded audit, **after the first result has been examined**:

```bash
python -u independent_solution_inventory_audit_2026_10_22.py --case both
```

This includes James–Hane 6M and overwrites the JSON with the two-case version.

## Independent mathematics

1. Exhaustively enumerate ordered pairs of every distinct stretch tensor in the PTMC public report. With n tensors, inspect n(n-1) ordered pairs.
2. Independently calculate the three eigenvalues of `(U_j U_i^{-1})^T(U_j U_i^{-1})`. Compare well-conditioned middle-eigenvalue compatibility with both expected proper rank-one branches.
3. For every reported PTMC twin branch, reconstruct `F(f) = U_i + f (a ⊗ n)` from the *original binary64 factors* in 85-digit mpmath and form `det(F(f)^T F(f) - I)` independently from values at -1, 0, and +1. Check its degree-two identity at three other points.
4. Isolate **all** real polynomial roots, including non-sign-changing double roots. Restrict to [0,1]. Independently require the middle singular value to be 1 to within a stated numerical threshold, with outer singular values separated from 1.
5. Compare the root sets one-to-one; reject duplicates, missing and extra roots in well-conditioned cases. Check that each stable PTMC root has both ± invariant-plane habit branches in the final PTMC report.

## Important interpretation

- `PASS_CONDITIONAL`: all *resolved, well-conditioned* checks matched. **Not a global proof**.
- `INCONCLUSIVE`: the audit cannot certify at least one numerical boundary, degeneracy, or continuum; that is an honest unresolved result, not a pass or a new physical defect.
- `MISMATCH`: a reproducible discrepancy is recorded; review the JSON before assigning blame to production (the oracle may have modeling limitations).

Exit codes: `0` conditional pass; `4` inconclusive; `2` mismatch; `3` operational error.

**Limitations:** This is a mathematical check on the encoded, binary64 input state, *not* on experimental measurement uncertainty. The script uses the production-reported variant stretches and twin rank-one factors as inputs, so it independently checks algebra and solution inventories but does not independently derive all lattice correspondences, physical twin classifications (Type I/II/Compound), or EBSD observations. Near-zero polynomials, ill-conditioned multiple roots, numerical eigenvalue boundaries, and putative continua are reported as inconclusive. The generator and type-classification modules are deliberately not edited or audited here. A subsequent projection audit will be needed to certify the final Streamlit UI as opposed to the PTMC report.

**Provenance:** report JSON includes the local Git HEAD, branch and SHA-256 digest of the audit script. All science checks run after the previous variant-identity, symmetry-resolution, and tangency repairs; they do not replace published benchmark checks.
