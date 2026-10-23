# CuAlNi-CT: variant identity, targeted repair candidate (2026-10-21)

**Not yet production- or CI-validated. Do not deploy or merge.**

This is a **candidate repair** for the reproduced numerical orbit collapse.
It changes only `src/cualni_cryst/ptmc_adapter.py`,
`src/cualni_cryst/ball_james_adapter.py`, and adds a shared
`src/cualni_cryst/variant_identity.py` helper. Both scientific compatibility
solvers remain independent. It does **not** repair or certify habit-plane
fraction completeness or downstream Type-I/II/Compound labels.

## Instructions (from repository root, on feature/twin-family-tree-app-v1)

1. Unzip the archive in the repository root.
2. Ensure the audit helper `variant_identity_forensics_2026_10_21.py`
   from the previous bundle remains in that root.
3. Run:

```bash
unzip -o variant_identity_patch_2026_10_21.zip
python -u apply_variant_identity_fix.py
python -m pytest -q test_variant_identity_repair_2026_10_21.py
```

Expected synthetic orbit: six variants at nonzero epsilon (including 1e-12),
three variants at epsilon=0, exact generator memberships. If tests are green:

```bash
python -m pytest -q tests/test_ball_james_adapter.py tests/twin_app/test_published_cualni_habit.py
python -m pytest -q
```

**Stop at the first failing stage and send the complete traceback.** The patch
is intentionally uncommitted. Do not push/merge/deploy until full CI and
independent physical branch completeness tests have passed.

### Scientific basis

A crystallographic parent symmetry `g` transports the metric pullback
`H = C.T @ G_M @ C` to `g^{-T} H g^{-1}` in the same parent crystal basis.
For a fixed parent metric, identical pullback tensors imply identical stretches.
The helper compares exact rational values *of the supplied binary64 entries*;
this means exact identity of the model's recorded inputs, not of a real specimen.
It also crosschecks floating-point stretch results against a conservative
conditioning-aware backward-error band, rejecting unresolved differences rather
than silently merging different metric signatures. `algebraic_tolerance` and
`generate_stretch_variants(tolerance=...)` are not used for identity merging.

### Important limitations

- Signed-permutation cubic tests were independently exercised with the helper.
  The PTMC/Ball-James repository functions and their full CI were **not** run here.
- Rational binary64 input signatures may treat rounded crystallographic angles
  as slightly non-symmetric; the code should fail closed in that situation.
  If this causes real-case failure, reconcile the declared symmetry/metric
  constraints instead of increasing a merge tolerance.
- A shared identity helper does not compromise independence of the two
  downstream rank-one/compatibility solvers, but their predictions still need
  separate numerical falsification.
- Backups are created alongside both edited modules with filename suffix
  `.py.pre_variant_identity_2026_10_21.bak` so the local changes can be undone.
  Keep these backups out of commits.

### Revert if needed

```bash
cp src/cualni_cryst/ptmc_adapter.py.pre_variant_identity_2026_10_21.bak src/cualni_cryst/ptmc_adapter.py
cp src/cualni_cryst/ball_james_adapter.py.pre_variant_identity_2026_10_21.bak src/cualni_cryst/ball_james_adapter.py
rm src/cualni_cryst/variant_identity.py
```

This reverts **only** the candidate's three production files; it does not
reset any unrelated user changes or repository history.
